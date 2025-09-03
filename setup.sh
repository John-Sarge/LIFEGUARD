#!/bin/bash

# setup.sh - Optional helper to install dependencies and models (Linux/macOS)

set -e

echo "Upgrading pip..."
pip install --upgrade pip

# --- CHANGED: Install from pyproject.toml instead of a hardcoded list ---
echo "Installing Python dependencies from pyproject.toml in editable mode..."
pip install -e .

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Downloading Vosk model (small English)..."
mkdir -p vosk_models
cd vosk_models
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    VOSK_ZIP="vosk-model-small-en-us-0.15.zip"
    echo "Downloading Vosk model..."
    wget -q --show-progress "https://alphacephei.com/vosk/models/${VOSK_ZIP}"
    echo "Unzipping Vosk model..."
    unzip -q "${VOSK_ZIP}"
    rm "${VOSK_ZIP}"
fi
cd ..

echo "Setup complete."