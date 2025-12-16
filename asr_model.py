from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Paths ----------------

VOCAB_PATH = Path("asr_vocab.json")
MODEL_PATH = Path("asr_model.pt")

# ---------------- Device ----------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Shared config ----------------

TARGET_SAMPLE_RATE = 16000

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FMIN = 50.0
FMAX = 7600.0

# ---------------- Results ----------------

@dataclass
class ASRResult:
    transcript: Optional[str]
    confidence: float  # 0..1


# ---------------- Vocab ----------------

@dataclass
class Vocab:
    chars: List[str]
    char_to_idx: Dict[str, int]
    idx_to_char: Dict[int, str]
    blank_idx: int


def load_vocab() -> Vocab:
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Missing {VOCAB_PATH}. Train the model first.")
    data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    chars = data["chars"]
    blank_idx = data["blank_idx"]
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return Vocab(chars, char_to_idx, idx_to_char, blank_idx)


def indices_to_text(indices: List[int], vocab: Vocab) -> str:
    return "".join(vocab.idx_to_char[i] for i in indices if i != vocab.blank_idx)


# ---------------- Audio utils ----------------

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear resample using torch interpolate."""
    if orig_sr == target_sr:
        return audio
    t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    new_len = int(round(len(audio) * target_sr / orig_sr))
    t_new = F.interpolate(t, size=new_len, mode="linear", align_corners=False)
    return t_new.squeeze(0).squeeze(0).cpu().numpy()


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Remove DC + RMS normalize (helps inference too)."""
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(audio * audio)) + 1e-12)
    target_rms = 0.08
    audio = audio * (target_rms / rms)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


# ---------------- Mel filterbank (torch-only) ----------------

def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(torch.tensor(1.0, device=hz.device) + hz / 700.0)

def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def make_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float, device: torch.device):
    n_freqs = n_fft // 2 + 1
    m_min = hz_to_mel(torch.tensor(fmin, device=device))
    m_max = hz_to_mel(torch.tensor(fmax, device=device))
    m_pts = torch.linspace(m_min, m_max, steps=n_mels + 2, device=device)
    hz_pts = mel_to_hz(m_pts)

    bin_pts = torch.floor((n_fft + 1) * hz_pts / sr).long()
    bin_pts = torch.clamp(bin_pts, 0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs, device=device)
    for m in range(n_mels):
        left = int(bin_pts[m].item())
        center = int(bin_pts[m + 1].item())
        right = int(bin_pts[m + 2].item())
        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)

        if left < center:
            fb[m, left:center] = (torch.arange(left, center, device=device) - left) / max(1, (center - left))
        if center < right:
            fb[m, center:right] = (right - torch.arange(center, right, device=device)) / max(1, (right - center))

    fb = fb / (fb.sum(dim=1, keepdim=True) + 1e-12)
    return fb


def audio_to_logmel(audio_t: torch.Tensor, mel_fb: torch.Tensor) -> torch.Tensor:
    """
    audio_t: [T] float32
    returns log-mel: [frames, mels]
    """
    window = torch.hann_window(N_FFT, device=audio_t.device)
    spec = torch.stft(
        audio_t,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (spec.real ** 2 + spec.imag ** 2)
    mel = torch.matmul(mel_fb, power)
    mel = torch.clamp(mel, min=1e-10)
    logmel = torch.log(mel).transpose(0, 1)
    return logmel


# ---------------- Model A: old raw-waveform SimpleASR ----------------

class SimpleASR(nn.Module):
    def __init__(self, vocab_size: int, blank_idx: int):
        super().__init__()
        self.blank_idx = blank_idx
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(64, 96, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(96, 128, 5, 2, 2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(192, vocab_size)

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor):
        x = audio.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out_lengths = audio_lengths // 16

        packed = nn.utils.rnn.pack_padded_sequence(
            x, out_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(y)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1), out_lengths


# ---------------- Model B: newer log-mel + ConvSubsample + BiLSTM ----------------

class ConvSubsample(nn.Module):
    def __init__(self, out_ch: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, Tp, Fp = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, Tp, C * Fp)

        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return x, lengths, Fp


class MelASR(nn.Module):
    def __init__(self, vocab_size: int, blank_idx: int):
        super().__init__()
        self.blank_idx = blank_idx
        self.sub = ConvSubsample(out_ch=128)

        feat_dim = 128 * ((N_MELS + 3) // 4)

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        x, out_lengths, _ = self.sub(feats, feat_lengths)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, out_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(y)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1), out_lengths


# ---------------- Decode + confidence ----------------

def greedy_decode_with_confidence(
    log_probs_TBV: torch.Tensor,
    blank_idx: int
) -> Tuple[List[int], float]:
    probs = log_probs_TBV.exp()
    best_idx = probs.argmax(dim=-1)
    best_p = probs.max(dim=-1).values

    seq = best_idx[:, 0].detach().cpu().tolist()
    seq_p = best_p[:, 0].detach().cpu().tolist()

    decoded: List[int] = []
    conf_vals: List[float] = []
    prev = None

    for idx, p in zip(seq, seq_p):
        if idx == blank_idx:
            prev = None
            continue
        if idx == prev:
            continue
        decoded.append(int(idx))
        conf_vals.append(float(p))
        prev = idx

    if not conf_vals:
        return decoded, 0.0
    return decoded, float(sum(conf_vals) / len(conf_vals))


# ---------------- Wrapper ----------------

class MyASRModel:
    def __init__(self):
        self.vocab = load_vocab()
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing {MODEL_PATH}. Train the model first.")

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise RuntimeError("asr_model.pt was not a state_dict dict.")

        keys = set(ckpt.keys())

        if any(k.startswith("sub.conv.") for k in keys):
            self.model_type = "mel"
            self.model = MelASR(vocab_size=len(self.vocab.chars), blank_idx=self.vocab.blank_idx)
        elif any(k.startswith("conv.") for k in keys):
            self.model_type = "raw"
            self.model = SimpleASR(vocab_size=len(self.vocab.chars), blank_idx=self.vocab.blank_idx)
        else:
            raise RuntimeError(
                "Could not detect model type from checkpoint keys. "
                "Expected keys starting with 'sub.conv.' (mel) or 'conv.' (raw)."
            )

        self.model.load_state_dict(ckpt, strict=True)
        self.model.to(DEVICE)
        self.model.eval()

        self._mel_fb = None
        if self.model_type == "mel":
            self._mel_fb = make_mel_filterbank(
                sr=TARGET_SAMPLE_RATE,
                n_fft=N_FFT,
                n_mels=N_MELS,
                fmin=FMIN,
                fmax=FMAX,
                device=DEVICE,
            )

        print(f"[MyASRModel] Loaded checkpoint as '{self.model_type}' model on {DEVICE}.")

    def transcribe(self, sr: int, audio: np.ndarray) -> ASRResult:
        """
        sr: input sample rate
        audio: float32 [-1,1] mono
        """
        if audio is None or len(audio) == 0:
            return ASRResult(transcript=None, confidence=0.0)

        audio = resample_audio(audio.astype(np.float32), sr, TARGET_SAMPLE_RATE)
        audio = normalize_audio(audio)

        with torch.no_grad():
            if self.model_type == "raw":
                a = torch.from_numpy(audio).float().to(DEVICE)
                a_len = torch.tensor([a.numel()], dtype=torch.long, device=DEVICE)
                log_probs, _ = self.model(a.unsqueeze(0), a_len)
                decoded, conf = greedy_decode_with_confidence(log_probs, self.vocab.blank_idx)
                text = indices_to_text(decoded, self.vocab).strip()
                return ASRResult(transcript=text if text else None, confidence=conf)

            a = torch.from_numpy(audio).float().to(DEVICE)
            feats = audio_to_logmel(a, self._mel_fb)
            feat_len = torch.tensor([feats.shape[0]], dtype=torch.long, device=DEVICE)
            log_probs, _ = self.model(feats.unsqueeze(0), feat_len)
            decoded, conf = greedy_decode_with_confidence(log_probs, self.vocab.blank_idx)
            text = indices_to_text(decoded, self.vocab).strip()
            return ASRResult(transcript=text if text else None, confidence=conf)

def _test():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=str)
    args = p.parse_args()

    m = MyASRModel()
    with wave.open(args.wav, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)

    r = m.transcribe(sr, audio)
    print("Transcript:", r.transcript)
    print("Confidence:", r.confidence)


if __name__ == "__main__":
    _test()
