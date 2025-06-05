#!/bin/bash
set -e

# Install system dependencies for PyAudio (Linux example)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio
fi

# Create and activate virtual environment if you want
# python3 -m venv venv
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download Vosk model
VOSK_DIR="vosk_models"
VOSK_MODEL="vosk-model-small-en-us-0.15"
mkdir -p $VOSK_DIR
if [ ! -d "$VOSK_DIR/$VOSK_MODEL" ]; then
    wget https://alphacephei.com/vosk/models/$VOSK_MODEL.zip -O $VOSK_DIR/$VOSK_MODEL.zip
    unzip $VOSK_DIR/$VOSK_MODEL.zip -d $VOSK_DIR
    rm $VOSK_DIR/$VOSK_MODEL.zip
fi

echo "Setup complete."