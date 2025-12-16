#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
from tqdm import tqdm

from asr_model import MyASRModel, ASRResult

CLIP_DIR = Path("speech_clips")
CLIP_DB_PATH = Path("clip_transcriptions.json")
VOCAB_PATH = Path("vocab.json")
SETTINGS_PATH = Path("settings.json")

CONFIDENCE_THRESHOLD = 0.95
VERY_SHORT_TRANSCRIPT_LEN = 2
RANDOM_SAMPLE_COUNT = 100
WORD_REGEX = re.compile(r"[A-Za-z']+")

MIC_SR = 16000
MIC_CHANNELS = 1
MIC_DTYPE = "int16"

FRAME_MS = 30
START_THRESHOLD_RMS = 0.015
STOP_THRESHOLD_RMS = 0.010
PRE_ROLL_MS = 250
MIN_UTTERANCE_SECONDS = 0.40
SILENCE_STOP_SECONDS = 0.60
MAX_UTTERANCE_SECONDS = 12.0


def load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_settings() -> Dict[str, Any]:
    return load_json(SETTINGS_PATH, default={})


def save_settings(settings: Dict[str, Any]) -> None:
    save_json(SETTINGS_PATH, settings)


def sanitize_uid(uid: str) -> str:
    uid = uid.strip()
    uid = re.sub(r"[^A-Za-z0-9_-]+", "_", uid)
    uid = uid.strip("_")
    return uid or "anon"


def get_or_set_uid(settings: Dict[str, Any]) -> str:
    uid = settings.get("uid")
    if uid:
        return sanitize_uid(uid)
    uid = sanitize_uid(input("Enter UID (saved to settings.json): "))
    settings["uid"] = uid
    save_settings(settings)
    return uid


def change_uid_interactive(settings: Dict[str, Any]) -> str:
    cur = settings.get("uid")
    if cur:
        print(f"Current saved UID: {cur}")
    raw = input("New UID (blank cancels): ").strip()
    if not raw:
        return sanitize_uid(cur or "anon")
    uid = sanitize_uid(raw)
    settings["uid"] = uid
    save_settings(settings)
    print(f"Saved UID: {uid}")
    return uid


def list_input_devices() -> List[Tuple[int, Dict[str, Any]]]:
    devs = sd.query_devices()
    out = []
    for i, d in enumerate(devs):
        if int(d.get("max_input_channels", 0)) > 0:
            out.append((i, d))
    return out


def apply_device_audio_params_to_globals(settings: Dict[str, Any]) -> None:
    global MIC_SR, MIC_CHANNELS
    sr = settings.get("mic_sr")
    ch = settings.get("mic_channels")
    if isinstance(sr, (int, float)) and sr > 0:
        MIC_SR = int(round(sr))
    if isinstance(ch, int) and ch > 0:
        MIC_CHANNELS = int(ch)


def select_microphone_interactive(settings: Dict[str, Any]) -> Optional[int]:
    print("\n" + "=" * 60)
    print("Microphone selection")
    print("=" * 60)

    current = settings.get("input_device", None)
    if current is not None:
        try:
            cur_info = sd.query_devices(current)
            print(f"Current saved mic: #{current} — {cur_info.get('name')}")
            print(f"Saved params: sr={settings.get('mic_sr')} channels={settings.get('mic_channels')}")
        except Exception:
            print(f"Current saved mic: #{current} (could not query)")

    inputs = list_input_devices()
    if not inputs:
        print("No input devices found. Using system default.")
        return None

    print("\nAvailable input devices:")
    for idx, d in inputs:
        name = d.get("name", "Unknown")
        ch = int(d.get("max_input_channels", 0))
        sr = d.get("default_samplerate", "?")
        marker = " (saved)" if current == idx else ""
        print(f"  [{idx}] {name} | max_in_channels={ch} | default_sr={sr}{marker}")

    print("\nChoose:")
    print(" - ENTER = keep saved/default")
    print(" - Type device index (e.g. 3) = select that mic")
    print(" - 'd' = clear saved mic (use system default)")
    choice = input("Selection: ").strip().lower()

    if choice == "":
        apply_device_audio_params_to_globals(settings)
        return current

    if choice == "d":
        settings.pop("input_device", None)
        settings.pop("mic_sr", None)
        settings.pop("mic_channels", None)
        save_settings(settings)
        print("Cleared saved mic. Using system default.")
        return None

    if choice.isdigit():
        dev_index = int(choice)
        try:
            info = sd.query_devices(dev_index)
            if int(info.get("max_input_channels", 0)) <= 0:
                print("That device is not an input device. Keeping saved/default.")
                apply_device_audio_params_to_globals(settings)
                return current

            settings["input_device"] = dev_index
            settings["mic_sr"] = float(info.get("default_samplerate", MIC_SR))
            settings["mic_channels"] = int(info.get("max_input_channels", 1))
            save_settings(settings)
            apply_device_audio_params_to_globals(settings)

            print(f"Saved mic: #{dev_index} — {info.get('name')}")
            print(f"Applied params: sr={MIC_SR} channels={MIC_CHANNELS}")
            return dev_index
        except Exception as e:
            print(f"Invalid device index: {e}. Keeping saved/default.")
            apply_device_audio_params_to_globals(settings)
            return current

    print("Unrecognised input. Keeping saved/default.")
    apply_device_audio_params_to_globals(settings)
    return current


def infer_uid_from_filename(filename: str) -> Optional[str]:
    base = Path(filename).name
    if "_" not in base:
        return None
    candidate = base.split("_", 1)[0]
    candidate = sanitize_uid(candidate)
    return candidate or None


def ensure_uid(existing: Dict[str, Any], filename: str, fallback_uid: Optional[str] = None) -> Optional[str]:
    if existing.get("uid"):
        return existing["uid"]
    inferred = infer_uid_from_filename(filename)
    if inferred:
        return inferred
    return fallback_uid


def load_wav_mono(path: Path) -> Tuple[int, np.ndarray]:
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return sr, audio


def save_wav_mono_16bit(path: Path, sr: int, audio_f32: np.ndarray) -> None:
    audio_f32 = np.asarray(audio_f32, dtype=np.float32)
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def play_audio(sr: int, audio: np.ndarray):
    try:
        sd.play(audio, sr)
        sd.wait()
    except Exception as e:
        print(f"Audio playback failed: {e}")


def tokenize(text: str) -> List[str]:
    return WORD_REGEX.findall(text)


def normalize_word(token: str) -> str:
    return token.lower().strip("'").strip()


def update_vocab_from_transcript(transcript: str, vocab: Dict[str, Dict[str, Any]]):
    tokens = tokenize(transcript)
    for tok in tokens:
        key = normalize_word(tok)
        if not key:
            continue
        if key not in vocab:
            vocab[key] = {"count": 0, "transcription": tok}
        vocab[key]["count"] += 1


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float32), dtype=np.float32)))


def find_audio_files(root: Path) -> List[Path]:
    files = [p for p in root.glob("*.wav") if p.is_file()]
    files.sort()
    return files


def is_confirmed(clip_db: Dict[str, Any], clip_name: str) -> bool:
    info = clip_db.get(clip_name)
    return bool(info and info.get("confirmed"))


def get_label(clip_db: Dict[str, Any], clip_name: str) -> Optional[str]:
    info = clip_db.get(clip_name) or {}
    return info.get("label")


def count_confirmed(all_clips: List[Path], clip_db: Dict[str, Any]) -> int:
    return sum(1 for c in all_clips if is_confirmed(clip_db, c.name))


def print_progress_bar(confirmed: int, total: int):
    if total <= 0:
        return
    ratio = confirmed / total
    width = 30
    filled = int(width * ratio)
    bar = "█" * filled + "-" * (width - filled)
    pct = int(ratio * 100)
    print(f"\nProgress: [{bar}] {pct:3d}%  ({confirmed}/{total} confirmed)")


def show_existing_entry(existing: Optional[Dict[str, Any]]):
    if not existing:
        return
    print("   Existing entry:")
    print(f"     uid        : {existing.get('uid')}")
    print(f"     label      : {existing.get('label')}")
    print(f"     transcript : {existing.get('transcript')}")
    print(f"     confidence : {existing.get('confidence')}")
    if existing.get("model_suggestion") is not None:
        print(f"     suggestion : {existing.get('model_suggestion')}")
        print(f"     sug_conf   : {existing.get('model_confidence')}")


def should_auto_accept(transcript: str, conf: float, threshold: float) -> bool:
    t = (transcript or "").strip()
    if not t:
        return False
    if len(t) <= VERY_SHORT_TRANSCRIPT_LEN:
        return False
    return conf >= threshold


def ask_review_confirmed_speech(
    clip_path: Path,
    sr: int,
    audio: np.ndarray,
    existing_transcript: str,
    model_transcript: str,
    model_conf: float,
) -> Tuple[Optional[str], str]:
    print("\n" + "=" * 60)
    print(f"REVIEW (already confirmed speech): {clip_path.name}")
    print(f"Existing : {existing_transcript}")
    print(f"Model    : {model_transcript if model_transcript else '(blank)'}")
    print(f"Conf     : {model_conf:.4f}")
    play_audio(sr, audio)

    while True:
        print("\nOptions:")
        print(" - ENTER = keep EXISTING transcript")
        print(" - 'o' + ENTER = overwrite with MODEL transcript")
        print(" - Type new text = replace transcript manually")
        print(" - 'r' + ENTER = replay audio")
        print(" - 'j' + ENTER = relabel as JUNK")
        print(" - 'u' + ENTER = relabel as UNLABELED")
        print(" - 's' + ENTER = skip (no changes)")
        user = input("Your input: ").strip()
        lower = user.lower()

        if lower == "r":
            play_audio(sr, audio)
            continue
        if lower == "s":
            return None, "skip"
        if lower == "j":
            return None, "junk"
        if lower == "u":
            return None, "unlabeled"
        if lower == "o":
            if not model_transcript.strip():
                print("Model transcript is blank; refusing overwrite.")
                continue
            return model_transcript.strip(), "speech"
        if user == "":
            return existing_transcript, "speech"

        return user, "speech"


def ask_full_transcription(
    clip_path: Path,
    sr: int,
    audio: np.ndarray,
    model_conf: float,
    existing_entry: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    print("\n" + "=" * 60)
    print(f"Clip: {clip_path.name}")
    print(f"Model confidence: {model_conf:.4f}")
    show_existing_entry(existing_entry)
    play_audio(sr, audio)

    while True:
        print("\nType what you hear.")
        print(" - ENTER = skip for now")
        print(" - 'r' + ENTER = replay")
        print(" - 'j' + ENTER = mark as JUNK")
        print(" - 'u' + ENTER = mark as UNLABELED")
        text = input("Transcription (or 'r'/'j'/'u'): ").strip()
        lower = text.lower()

        if lower == "r":
            play_audio(sr, audio)
            continue
        if lower == "j":
            return None, "junk"
        if lower == "u":
            return None, "unlabeled"
        if text == "":
            return None, "skip"

        return text, "speech"


def ask_confirm_or_edit(
    suggested: str,
    confidence: float,
    sr: int,
    audio: np.ndarray,
    clip_name: str,
    existing_entry: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    print("\n" + "=" * 60)
    print(f"Clip: {clip_name}")
    print(f"Suggested : {suggested}")
    print(f"Confidence: {confidence:.4f}")
    show_existing_entry(existing_entry)
    play_audio(sr, audio)

    while True:
        print("\nOptions:")
        print(" - ENTER = accept suggestion as speech")
        print(" - Type correction = new speech transcription")
        print(" - 'r' + ENTER = replay")
        print(" - 'j' + ENTER = mark as JUNK")
        print(" - 'u' + ENTER = mark as UNLABELED")
        print(" - 's' + ENTER = skip for now")
        user = input("Your input: ").strip()
        lower = user.lower()

        if lower == "r":
            play_audio(sr, audio)
            continue
        if lower == "j":
            return None, "junk"
        if lower == "u":
            return None, "unlabeled"
        if lower == "s":
            return None, "skip"
        if user == "":
            return suggested, "speech"

        return user, "speech"


def process_clip(
    clip_path: Path,
    asr_model: MyASRModel,
    clip_db: Dict[str, Any],
    vocab: Dict[str, Dict[str, Any]],
    threshold: float,
) -> bool:
    key = clip_path.name
    existing = clip_db.get(key) or {}
    was_confirmed = bool(existing.get("confirmed"))
    was_label = existing.get("label")
    uid = ensure_uid(existing, key)

    print("\n" + "=" * 60)
    print(f"Processing: {key}")
    if existing:
        show_existing_entry(existing)

    try:
        sr, audio = load_wav_mono(clip_path)
    except Exception as e:
        print(f"ERROR loading {key}: {e}")
        return False

    result: ASRResult = asr_model.transcribe(sr, audio)
    model_transcript = (result.transcript or "").strip()
    conf = float(getattr(result, "confidence", 0.0) or 0.0)

    if was_confirmed and was_label == "speech":
        existing_transcript = (existing.get("transcript") or "").strip()
        new_transcript, new_label = ask_review_confirmed_speech(
            clip_path, sr, audio, existing_transcript, model_transcript, conf
        )

        if new_label == "skip":
            print("No changes saved.")
            return False

        if new_label in ("junk", "unlabeled"):
            clip_db[key] = {"uid": uid, "transcript": None, "confidence": conf, "label": new_label, "confirmed": True}
            print(f"Relabeled as {new_label.upper()}.")
            return False

        clip_db[key] = {"uid": uid, "transcript": new_transcript, "confidence": conf, "label": "speech", "confirmed": True}
        if new_transcript:
            update_vocab_from_transcript(new_transcript, vocab)
        print("Saved speech transcript.")
        return False

    final_transcript: Optional[str] = None
    final_label: str = "speech"

    if not model_transcript:
        final_transcript, final_label = ask_full_transcription(clip_path, sr, audio, conf, existing)
    else:
        is_very_short = len(model_transcript) <= VERY_SHORT_TRANSCRIPT_LEN
        if is_very_short or conf < threshold:
            final_transcript, final_label = ask_confirm_or_edit(model_transcript, conf, sr, audio, key, existing)
        else:
            print("\nAuto-accepted")
            print(f"Transcript : {model_transcript}")
            print(f"Confidence : {conf:.4f}")
            play_audio(sr, audio)
            final_transcript = model_transcript
            final_label = "speech"

    if final_label == "skip":
        print("No changes saved.")
        return False

    if final_label in ("junk", "unlabeled") and final_transcript is None:
        clip_db[key] = {"uid": uid, "transcript": None, "confidence": conf, "label": final_label, "confirmed": True}
        print(f"Saved as {final_label.upper()}.")
        return (not was_confirmed)

    if final_label == "speech" and final_transcript:
        clip_db[key] = {"uid": uid, "transcript": final_transcript, "confidence": conf, "label": "speech", "confirmed": True}
        update_vocab_from_transcript(final_transcript, vocab)
        print("\nSaved speech:")
        print(f"Transcript : {final_transcript}")
        print(f"Confidence : {conf:.4f}")
        return (not was_confirmed)

    clip_db[key] = {"uid": uid, "transcript": None, "confidence": conf, "label": "unlabeled", "confirmed": True}
    print("Saved as UNLABELED.")
    return (not was_confirmed)


def auto_label_clips(
    all_clips: List[Path],
    asr_model: MyASRModel,
    clip_db: Dict[str, Any],
    vocab: Dict[str, Dict[str, Any]],
    *,
    threshold: float,
    only_unconfirmed: bool,
    save_every: int = 25,
) -> None:
    processed = 0
    accepted = 0
    skipped_confirmed_speech = 0

    for clip_path in tqdm(all_clips, desc="Auto-labeling"):
        key = clip_path.name
        existing = clip_db.get(key) or {}
        was_confirmed = bool(existing.get("confirmed"))
        was_label = existing.get("label")
        uid = ensure_uid(existing, key)

        if only_unconfirmed and was_confirmed:
            continue
        if was_confirmed and was_label == "speech":
            skipped_confirmed_speech += 1
            continue

        try:
            sr, audio = load_wav_mono(clip_path)
            result: ASRResult = asr_model.transcribe(sr, audio)
            model_transcript = (result.transcript or "").strip()
            conf = float(getattr(result, "confidence", 0.0) or 0.0)
        except Exception as e:
            clip_db[key] = {
                "uid": uid,
                "transcript": existing.get("transcript"),
                "confidence": existing.get("confidence", 0.0),
                "label": existing.get("label", "unlabeled"),
                "confirmed": bool(existing.get("confirmed", False)),
                "error": str(e),
            }
            processed += 1
            continue

        if should_auto_accept(model_transcript, conf, threshold):
            clip_db[key] = {"uid": uid, "transcript": model_transcript, "confidence": conf, "label": "speech", "confirmed": True}
            update_vocab_from_transcript(model_transcript, vocab)
            accepted += 1
        else:
            clip_db[key] = {
                "uid": uid,
                "transcript": existing.get("transcript"),
                "confidence": existing.get("confidence", 0.0),
                "label": existing.get("label"),
                "confirmed": bool(existing.get("confirmed", False)),
                "model_suggestion": model_transcript,
                "model_confidence": conf,
            }

        processed += 1
        if processed % save_every == 0:
            save_json(CLIP_DB_PATH, clip_db)
            save_json(VOCAB_PATH, vocab)

    save_json(CLIP_DB_PATH, clip_db)
    save_json(VOCAB_PATH, vocab)

    print("\nAuto-label summary")
    print(f"Processed               : {processed}")
    print(f"Auto-accepted (speech)  : {accepted}")
    print(f"Skipped confirmed speech: {skipped_confirmed_speech}")
    print(f"Threshold               : {threshold:.2f}")


def continuous_utterance_record_and_transcribe(
    *,
    uid: str,
    asr_model: MyASRModel,
    clip_db: Dict[str, Any],
    vocab: Dict[str, Dict[str, Any]],
    threshold: float,
    input_device: Optional[int],
):
    print("\nContinuous utterance recording")
    print(f"UID: {uid}")
    print(f"sr={MIC_SR} channels={MIC_CHANNELS}")
    if input_device is None:
        print("Mic: system default")
    else:
        try:
            info = sd.query_devices(input_device)
            print(f"Mic: #{input_device} — {info.get('name')}")
        except Exception:
            print(f"Mic: #{input_device}")

    frame_len = int(MIC_SR * (FRAME_MS / 1000.0))
    pre_roll_len = int(MIC_SR * (PRE_ROLL_MS / 1000.0))
    silence_stop_frames = int((SILENCE_STOP_SECONDS * MIC_SR) / frame_len)
    max_frames = int((MAX_UTTERANCE_SECONDS * MIC_SR) / frame_len)
    min_samples = int(MIN_UTTERANCE_SECONDS * MIC_SR)

    pre_roll = np.zeros((0,), dtype=np.float32)
    recording = False
    utter_frames: List[np.ndarray] = []
    silent_frames = 0
    frames_in_utt = 0
    ring = np.zeros((0,), dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal ring
        if status:
            print(status, file=sys.stderr)
        audio = indata.astype(np.float32) / 32768.0
        mono = audio.mean(axis=1)
        ring = np.concatenate([ring, mono])

    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    def finalize_utterance(samples_f32: np.ndarray):
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{uid}_{ts}_{random.randint(1000,9999)}.wav"
        path = CLIP_DIR / fname
        save_wav_mono_16bit(path, MIC_SR, samples_f32)

        result: ASRResult = asr_model.transcribe(MIC_SR, samples_f32)
        transcript = (result.transcript or "").strip()
        conf = float(getattr(result, "confidence", 0.0) or 0.0)

        print(f"\n{fname}")
        print(f"Guess: {transcript if transcript else '(blank)'}")
        print(f"Conf : {conf:.4f}")

        existing = clip_db.get(fname) or {}
        if existing.get("confirmed") and existing.get("label") == "speech":
            print("Exists as confirmed speech; not overwriting DB entry.")
            return

        if should_auto_accept(transcript, conf, threshold):
            clip_db[fname] = {"uid": uid, "transcript": transcript, "confidence": conf, "label": "speech", "confirmed": True}
            update_vocab_from_transcript(transcript, vocab)
            print("Auto-confirmed")
        else:
            clip_db[fname] = {
                "uid": uid,
                "transcript": None,
                "confidence": conf,
                "label": None,
                "confirmed": False,
                "model_suggestion": transcript,
                "model_confidence": conf,
            }
            print("Stored as guess")

        save_json(CLIP_DB_PATH, clip_db)
        save_json(VOCAB_PATH, vocab)

    try:
        with sd.InputStream(
            samplerate=MIC_SR,
            channels=MIC_CHANNELS,
            dtype=MIC_DTYPE,
            blocksize=frame_len,
            callback=callback,
            device=input_device,
        ):
            print("Ctrl+C to stop.\n")
            while True:
                if ring.shape[0] < frame_len:
                    time.sleep(0.01)
                    continue

                frame = ring[:frame_len]
                ring = ring[frame_len:]

                e = rms(frame)

                pre_roll = np.concatenate([pre_roll, frame])
                if pre_roll.shape[0] > pre_roll_len:
                    pre_roll = pre_roll[-pre_roll_len:]

                if not recording:
                    if e >= START_THRESHOLD_RMS:
                        recording = True
                        utter_frames = [pre_roll.copy()]
                        silent_frames = 0
                        frames_in_utt = 0
                    continue

                utter_frames.append(frame)
                frames_in_utt += 1

                if e <= STOP_THRESHOLD_RMS:
                    silent_frames += 1
                else:
                    silent_frames = 0

                too_long = frames_in_utt >= max_frames
                enough_silence = silent_frames >= silence_stop_frames

                if enough_silence or too_long:
                    samples = np.concatenate(utter_frames, axis=0)
                    recording = False
                    utter_frames = []
                    silent_frames = 0
                    frames_in_utt = 0

                    if samples.shape[0] < min_samples:
                        continue

                    finalize_utterance(samples)

    except KeyboardInterrupt:
        print("\nStopped.")


def mode_random_review(all_clips, asr_model, clip_db, vocab, n: int, threshold: float):
    total = len(all_clips)
    confirmed = count_confirmed(all_clips, clip_db)
    print(f"Total clips: {total}")
    print(f"Confirmed : {confirmed}")
    print_progress_bar(confirmed, total)

    n = min(n, total)
    chosen = random.sample(all_clips, k=n)

    for i, clip in enumerate(chosen, start=1):
        print(f"\n--- Clip {i}/{n} ---")
        newly_confirmed = process_clip(clip, asr_model, clip_db, vocab, threshold=threshold)
        if newly_confirmed:
            confirmed += 1
        save_json(CLIP_DB_PATH, clip_db)
        save_json(VOCAB_PATH, vocab)
        print_progress_bar(confirmed, total)


def mode_review_confirmed_speech(all_clips, asr_model, clip_db, vocab, n: int, threshold: float):
    confirmed_speech = [c for c in all_clips if is_confirmed(clip_db, c.name) and get_label(clip_db, c.name) == "speech"]
    if not confirmed_speech:
        print("No confirmed speech clips to review.")
        return

    n = min(n, len(confirmed_speech))
    chosen = random.sample(confirmed_speech, k=n) if n < len(confirmed_speech) else confirmed_speech

    for i, clip in enumerate(chosen, start=1):
        print(f"\n--- Clip {i}/{len(chosen)} ---")
        _ = process_clip(clip, asr_model, clip_db, vocab, threshold=threshold)
        save_json(CLIP_DB_PATH, clip_db)
        save_json(VOCAB_PATH, vocab)


def mode_stats(all_clips, clip_db):
    total = len(all_clips)
    confirmed = count_confirmed(all_clips, clip_db)
    labels = {"speech": 0, "junk": 0, "unlabeled": 0, "other": 0}
    unconfirmed = 0

    for c in all_clips:
        info = clip_db.get(c.name) or {}
        if not info.get("confirmed"):
            unconfirmed += 1
            continue
        lab = info.get("label")
        if lab in labels:
            labels[lab] += 1
        else:
            labels["other"] += 1

    print("\nStats")
    print(f"Total clips       : {total}")
    print(f"Confirmed clips   : {confirmed}")
    print(f"Unconfirmed clips : {unconfirmed}")
    print(f"speech            : {labels['speech']}")
    print(f"junk              : {labels['junk']}")
    print(f"unlabeled         : {labels['unlabeled']}")
    if labels["other"]:
        print(f"other             : {labels['other']}")
    print_progress_bar(confirmed, total)


def run_menu(all_clips, asr_model, clip_db, vocab, threshold: float, settings: Dict[str, Any]):
    while True:
        print("\n" + "=" * 60)
        print("ASR Recording & Transcription Menu")
        print("=" * 60)
        print("1) Continuous recording + immediate ASR")
        print("2) Auto-label ALL clips")
        print("3) Auto-label UNCONFIRMED only")
        print("4) Random review session")
        print("5) Review CONFIRMED speech only")
        print("6) Stats")
        print("7) Select microphone")
        print("8) Change UID")
        print("0) Exit")

        choice = input("Select option: ").strip()

        if choice == "0":
            return
        if choice == "7":
            select_microphone_interactive(settings)
            continue
        if choice == "8":
            change_uid_interactive(settings)
            continue

        if choice == "1":
            uid = get_or_set_uid(settings)
            apply_device_audio_params_to_globals(settings)
            input_device = settings.get("input_device", None)
            continuous_utterance_record_and_transcribe(
                uid=uid,
                asr_model=asr_model,
                clip_db=clip_db,
                vocab=vocab,
                threshold=threshold,
                input_device=input_device,
            )
            all_clips[:] = find_audio_files(CLIP_DIR)
        elif choice == "2":
            auto_label_clips(all_clips, asr_model, clip_db, vocab, threshold=threshold, only_unconfirmed=False)
        elif choice == "3":
            auto_label_clips(all_clips, asr_model, clip_db, vocab, threshold=threshold, only_unconfirmed=True)
        elif choice == "4":
            n = input(f"How many random clips? (default {RANDOM_SAMPLE_COUNT}): ").strip()
            n = int(n) if n.isdigit() else RANDOM_SAMPLE_COUNT
            mode_random_review(all_clips, asr_model, clip_db, vocab, n=n, threshold=threshold)
        elif choice == "5":
            n = input("How many confirmed speech clips to review? (default 50): ").strip()
            n = int(n) if n.isdigit() else 50
            mode_review_confirmed_speech(all_clips, asr_model, clip_db, vocab, n=n, threshold=threshold)
        elif choice == "6":
            mode_stats(all_clips, clip_db)
        else:
            print("Invalid option.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["menu", "continuous", "auto_all", "auto_unconfirmed", "review", "review_confirmed", "stats", "mic_select", "set_uid"],
        default="menu",
    )
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    ap.add_argument("--review_n", type=int, default=RANDOM_SAMPLE_COUNT)
    ap.add_argument("--uid", type=str, default="")
    args = ap.parse_args()

    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    clip_db = load_json(CLIP_DB_PATH, default={})
    vocab = load_json(VOCAB_PATH, default={})

    apply_device_audio_params_to_globals(settings)

    try:
        asr_model = MyASRModel()
    except Exception as e:
        print(f"ERROR loading ASR model: {e}")
        sys.exit(1)

    all_clips = find_audio_files(CLIP_DIR)

    if args.mode == "menu":
        run_menu(all_clips, asr_model, clip_db, vocab, threshold=args.threshold, settings=settings)
    elif args.mode == "mic_select":
        select_microphone_interactive(settings)
    elif args.mode == "set_uid":
        change_uid_interactive(settings)
    elif args.mode == "continuous":
        if args.uid.strip():
            settings["uid"] = sanitize_uid(args.uid)
            save_settings(settings)
        uid = get_or_set_uid(settings)
        apply_device_audio_params_to_globals(settings)
        input_device = settings.get("input_device", None)
        continuous_utterance_record_and_transcribe(
            uid=uid,
            asr_model=asr_model,
            clip_db=clip_db,
            vocab=vocab,
            threshold=args.threshold,
            input_device=input_device,
        )
    elif args.mode == "auto_all":
        auto_label_clips(all_clips, asr_model, clip_db, vocab, threshold=args.threshold, only_unconfirmed=False)
    elif args.mode == "auto_unconfirmed":
        auto_label_clips(all_clips, asr_model, clip_db, vocab, threshold=args.threshold, only_unconfirmed=True)
    elif args.mode == "review":
        mode_random_review(all_clips, asr_model, clip_db, vocab, n=args.review_n, threshold=args.threshold)
    elif args.mode == "review_confirmed":
        mode_review_confirmed_speech(all_clips, asr_model, clip_db, vocab, n=args.review_n, threshold=args.threshold)
    elif args.mode == "stats":
        mode_stats(all_clips, clip_db)

    save_json(CLIP_DB_PATH, clip_db)
    save_json(VOCAB_PATH, vocab)


if __name__ == "__main__":
    main()
