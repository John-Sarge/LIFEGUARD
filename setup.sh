#!/bin/bash

# setup.sh - Install dependencies for lifeguard.py

set -e

#echo "Creating Python virtual environment..."
#python -m venv venv
#source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install "numpy<2.0" "scipy<1.14" "spacy==3.7.5" pyaudio noisereduce vosk pynput pyttsx3 statemachine pymavlink lat_lon_parser

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Downloading Vosk model (small English)..."
mkdir -p vosk_models
cd vosk_models
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip vosk-model-small-en-us-0.15.zip
    #rm vosk-model-small-en-us-0.15.zip
fi
cd ..

echo "Setup complete."
