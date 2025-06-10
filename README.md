![Lifeguard](https://github.com/user-attachments/assets/144d6ab6-ae71-4783-b9d1-0b1ada70befc)

# LIFEGUARD
**Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments**

---

## Overview

LIFEGUARD is a Python-based system that interprets spoken natural language commands, allowing a human operator (Commander) to pass intent and instructions directly to autonomous units (drones/robots). It bridges human intent and autonomous action for field and rescue operations, with robust multi-agent support and advanced command understanding.

---

## Key Features

- **Speech-to-Text (STT):** Converts live spoken commands to text using Vosk.
- **Natural Language Understanding (NLU):** Extracts intent and entities (GPS, landmarks, targets) using spaCy with custom entity recognizers and spoken number conversion.
- **Multi-Agent MAVLink Integration:** Supports multiple MAVLink-compatible vehicles (drones/robots) with agent selection by voice command.
- **Waypoint & Grid Search:** Translates intent into actionable commands, including waypoint navigation and automatic grid search pattern generation.
- **Audio Preprocessing:** Noise reduction and band-pass filtering for reliable operation in noisy environments.
- **Interactive Confirmation:** Prompts for confirmation before sending critical commands.
- **Push-to-Talk (PTT):** Uses the spacebar as a push-to-talk trigger; ESC for clean shutdown.
- **Robust State Machine:** Ensures reliable transitions and error handling.
- **Cross-Platform:** Works on Windows and Linux.

---

## How It Works

1. **Push-to-Talk:** Press and hold the spacebar to record a command. Release to stop.
2. **Audio Preprocessing:** Denoising and filtering improve recognition.
3. **Speech Recognition:** Vosk transcribes the processed audio.
4. **Intent Extraction:** spaCy NLU extracts intent, entities, and spoken numbers (e.g., "forty one point three seven" → 41.37).
5. **Agent Selection:** Say commands like "select drone two" to switch active agents.
6. **Command Translation:** Valid intent (e.g., "Search a 100 meter grid at latitude 41.37 longitude -72.09") is mapped to MAVLink commands.
7. **User Confirmation:** System asks for confirmation before sending commands.
8. **Action Execution:** Upon confirmation, commands are sent to the selected autonomous unit.

---

## Example Use Cases

- **Waypoint Search:**  
  > "Search at latitude 41.37 longitude -72.09 for a person in a life ring."
- **Grid Search:**  
  > "Search a 100 meter grid at latitude 41.37 longitude -72.09."
- **Agent Selection:**  
  > "Select drone two."

LIFEGUARD will recognize, transcribe, extract details, prompt for confirmation, and send the appropriate mission to the selected drone.

---

## Major Components

- **`lifeguard.py`:** Main application file integrating all subsystems.
- **Audio Processing:** Uses `noisereduce` and `scipy.signal` for noise reduction and filtering.
- **Speech-to-Text:** Utilizes Vosk models for offline recognition.
- **NLU:** spaCy pipelines with custom entity recognition for GPS, grid size, agent selection, and spoken number conversion.
- **MAVLink Controller:** Employs `pymavlink` for sending navigation and mission commands to multiple agents.
- **Push-to-Talk Handler:** Uses `pynput` for spacebar (PTT) and ESC (shutdown) events.
- **State Machine:** Manages command, confirmation, and execution states for reliability.

---

## Setup and Requirements

### Minimum Requirements

- Python 3.7+
- [Vosk](https://alphacephei.com/vosk/) (STT engine and English model)
- [spaCy](https://spacy.io/) with `en_core_web_sm` model
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (microphone input)
- [noisereduce](https://github.com/timsainb/noisereduce)
- [pymavlink](https://github.com/ArduPilot/pymavlink)
- [lat_lon_parser](https://pypi.org/project/lat-lon-parser/)
- numpy, scipy, pynput, pyttsx3, statemachine, word2number

### Installation Steps

1. **(Optional) Create a Virtual Environment**
   ```bash
   python3 -m venv lifeguard_bot
   # On Linux/macOS:
   source lifeguard_bot/bin/activate
   # On Windows:
   lifeguard_bot\Scripts\activate
   ```

2. **Clone the Repository**
   ```bash
   git clone https://github.com/John-Sarge/LIFEGUARD.git
   cd LIFEGUARD
   ```

3. **Install Dependencies**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. **(Linux Only) Additional Audio Dependencies**
   ```bash
   sudo apt-get install portaudio19-dev python3-pyaudio
   ```

5. **Download Vosk Model**
   - Download the English model (e.g., `vosk-model-small-en-us-0.15`) and place it in `vosk_models/`.

---

## Usage

1. **Connect your MAVLink-compatible vehicle(s) or start a simulator**  
   Adjust the `AGENT_CONNECTION_CONFIGS` in `lifeguard.py` for your hardware or simulator.

2. **Run the Application**
   ```bash
   python lifeguard.py
   ```

3. **Speak your command into the microphone.**
   - Example: "Search a 100 meter grid at latitude 41.37 longitude -72.09."
   - The system will transcribe, understand, and prompt for confirmation.

4. **Confirm the interpreted intent when prompted.**
   - Respond 'yes' or 'no' when asked before sending the command.

5. **Switch agents by voice:**
   - Example: "Select drone two."

---

## Safety and Robustness

- Prompts for confirmation before sending critical commands.
- Robust error handling and state checks.
- Multi-agent support with safe switching.
- Designed for field reliability.

---

## Customization

- Extend entity patterns and NLU rules in `lifeguard.py` for new command types.
- Adjust `AGENT_CONNECTION_CONFIGS` for different vehicles.
- Tune grid search parameters and default altitudes as needed.

---

## Troubleshooting

- **Vosk model not found:**  
  Ensure the model is downloaded and placed in the correct folder (`vosk_models/`).
- **PyAudio errors:**  
  Check microphone permissions and install system dependencies.
- **MAVLink connection issues:**  
  Verify connection strings and that your vehicle/simulator is running.

---

## Acknowledgments

Built with open-source libraries: Vosk, spaCy, pymavlink, and community-contributed tools.

---

## Primary Purpose

LIFEGUARD enables operators to naturally and effectively pass Commander’s Intent to autonomous units, bridging voice commands and machine action for efficient, reliable, and intuitive search and rescue deployments.

---
