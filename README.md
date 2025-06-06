# LIFEGUARD
**Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments**

---

## Overview

LIFEGUARD is a Python-based system designed to interpret and process spoken natural language commands, allowing a human operator (Commander) to pass intent and instructions directly to autonomous units such as drones or robots. Its primary goal is to provide a robust, reliable, and intuitive bridge between human intent and autonomous action, especially for field and rescue operations.

---

## Key Features

- **Speech-to-Text (STT):** Converts live spoken commands to text using Vosk.
- **Natural Language Understanding (NLU):** Extracts intent and entities (like GPS coordinates, landmarks, and target descriptions) from the Commander’s speech using spaCy and custom entity recognizers.
- **MAVLink Integration:** Translates high-level intent into actionable commands for MAVLink-compatible vehicles (e.g., drones), supporting waypoint navigation, mission uploads, mode changes, arming/disarming, and more.
- **Audio Preprocessing:** Enhances command recognition with noise reduction and band-pass filtering for reliable operation in noisy environments.
- **Interactive Confirmation:** Prompts for confirmation before sending critical commands to autonomous units, ensuring accuracy and safety.
- **Push-to-Talk (PTT):** Uses the spacebar as a push-to-talk trigger for capturing commands, with ESC for clean shutdown.
- **Cross-Platform:** Works on both Windows and Linux (see notes below).

---

## How It Works

1. **Push-to-Talk:** The system waits for the operator to press and hold the spacebar to begin recording a command. Releasing the spacebar stops recording.
2. **Audio Preprocessing:** Incoming audio is denoised and filtered to improve recognition accuracy.
3. **Speech Recognition:** The processed audio is transcribed to text using the Vosk STT engine.
4. **Intent Extraction:** The text is analyzed to identify the Commander’s intent and extract actionable entities (such as GPS coordinates, targets, and locations) using spaCy with custom rules and entity recognition.
5. **Command Translation:** If a valid intent is detected (e.g., "Search at latitude 41.37 longitude -72.09 for a person in a life ring"), the system prepares corresponding MAVLink commands to instruct the autonomous unit.
6. **User Confirmation:** Before sending commands to the vehicle, the system asks the operator to confirm the interpreted intent by pressing space and saying "yes" or "no".
7. **Action Execution:** Upon confirmation, the system sends the appropriate commands to the autonomous unit over MAVLink.

---

## Example Use Case

A field operator says:

> "Search at latitude 41.37 longitude -72.09 for a person in a life ring."

LIFEGUARD will:

- Recognize and transcribe the command,
- Extract the GPS coordinates and target description,
- Prompt the operator for confirmation,
- Send the search waypoint to the drone once confirmed.

---

## Major Components

- **`lifeguard.py`:** Main application file integrating all subsystems—audio processing, STT, NLU, and MAVLink communication.
- **Audio Processing:** Uses `noisereduce` and `scipy.signal` for noise reduction and filtering.
- **Speech-to-Text:** Utilizes Vosk models for offline speech recognition.
- **NLU:** Implements spaCy pipelines with custom entity recognition for GPS and search/rescue-specific attributes.
- **MAVLink Controller:** Employs `pymavlink` for sending navigation and mission commands to compatible vehicles.
- **Push-to-Talk Handler:** Uses `pynput` to listen for spacebar (PTT) and ESC (shutdown) events, with robust cross-platform handling.

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
- numpy, scipy, pynput, pyttsx3, statemachine

### Installation Steps

1. **(Optional but Recommended) Create a Virtual Environment**
   ```bash
   python3 -m venv lifeguard_bot
   source lifeguard_bot/bin/activate  # On Windows: lifeguard_bot\Scripts\activate
   ```
   
1. **Clone the Repository**
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
   - You may need to install system dependencies for PyAudio:
     ```bash
     sudo apt-get install portaudio19-dev python3-pyaudio
     ```

---

## Usage

1. **Connect your MAVLink-compatible vehicle or start a simulator** (e.g., ArduPilot SITL).  
   Adjust the MAVLink connection string in `lifeguard.py` as needed for your hardware or simulator.

2. **Run the Application**
   ```bash
   python lifeguard.py
   ```

3. **Speak your command into the microphone.**
   - Example: "Search at latitude 41.37 longitude -72.09 for a person in a life ring."
   - The system will transcribe, understand, and prompt for confirmation.

4. **Confirm the interpreted intent when prompted.**
   - Respond 'yes' or 'no' when the system asks for confirmation before sending the command to the autonomous unit.

---

## Safety and Robustness

- The system prompts for confirmation before sending critical commands.
- Designed for field reliability, with robust error handling and state checks.
- Assumes the vehicle is already in the correct mode and armed unless additional safety automation is implemented.

---

## Customization

- Entity patterns and NLU rules can be extended for additional command types or domain-specific needs (see `lifeguard.py`).
- MAVLink connection strings and parameters can be adjusted for different vehicle types and communication setups.

---

## Acknowledgments

Built with open-source libraries: Vosk, spaCy, pymavlink, and community-contributed tools.

---

## Primary Purpose

LIFEGUARD enables operators to naturally and effectively pass Commander’s Intent to autonomous units, bridging voice commands and machine action for efficient, reliable, and intuitive search and rescue deployments.

---
