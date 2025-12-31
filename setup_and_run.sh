#!/usr/bin/env bash
set -euo pipefail

echo
echo "==============================="
echo " ASR Environment Setup (macOS)"
echo "==============================="
echo

# Go to the directory this script is in (project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------------
# Check python3
# -------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found."
  echo "Install Python 3 from https://www.python.org/ or via Homebrew: brew install python"
  exit 1
fi

python3 --version

# -------------------------------
# Create virtual environment
# -------------------------------
if [[ ! -d "venv" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
else
  echo "Virtual environment already exists."
fi

# -------------------------------
# Activate virtual environment
# -------------------------------
# shellcheck disable=SC1091
source "venv/bin/activate"

# -------------------------------
# Upgrade pip
# -------------------------------
echo
echo "Upgrading pip..."
python -m pip install --upgrade pip

# -------------------------------
# Install dependencies
# -------------------------------
echo
echo "Installing dependencies..."
pip install \
  numpy \
  torch \
  sounddevice \
  tqdm

echo
echo "========================================="
echo "Setup complete."
echo "Virtual environment is ACTIVE."
echo "========================================="
echo
echo "Common commands:"
echo "  python speech-recognition-main.py"
echo "  python speech-recognition-main.py --mode menu"
echo

# Keep an interactive shell open with the venv active
exec "$SHELL" -i
