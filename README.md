<p align="center">
  <img width="800" height="802" alt="LIFEGUARD_github_logo" src="https://github.com/user-attachments/assets/068ae446-1a9b-40af-93cb-5b18b39b2531" />
</p>

# LIFEGUARD

**Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments**

> LIFEGUARD enables operators to naturally and effectively pass Commander’s Intent to autonomous units, bridging voice commands and machine action for efficient, reliable, and intuitive search and rescue deployments.

-----

## Overview

LIFEGUARD is a Python-based system that interprets spoken natural language commands, allowing a human operator (Commander) to pass intent and instructions directly to autonomous units (drones/robots). It bridges human intent and autonomous action for field and rescue operations, with robust multi-agent support and advanced command understanding. The system is designed for field-deployable, edge-AI applications where internet connectivity is unreliable or non-existent, prioritizing operational robustness and self-sufficiency.

-----

## Key Features

  - **Speech-to-Text (STT):** Converts live spoken commands to text using the robust, offline Vosk engine.
  - **Natural Language Understanding (NLU):** Extracts intent and entities (GPS, landmarks, targets) using spaCy with custom entity recognizers and a sophisticated spoken number conversion module.
  - **Multi-Agent MAVLink Integration:** Supports multiple MAVLink-compatible vehicles (drones/robots) with seamless agent selection by voice command.
  - **Waypoint & Grid Search:** Translates high-level intent into actionable commands, including waypoint navigation and automatic generation of grid search patterns.
  - **Audio Preprocessing:** Employs noise reduction and band-pass filtering for reliable command recognition in noisy field environments.
  - **Interactive Confirmation:** A critical safety feature that prompts the operator for confirmation before dispatching commands to autonomous units.
  - **Push-to-Talk (PTT):** Uses the spacebar as a push-to-talk trigger to prevent accidental activation, with ESC for a clean system shutdown.
  - **Robust State Machine:** Ensures reliable state transitions and error handling throughout the command lifecycle.
  - **Cross-Platform:** Works on both Windows and Linux operating systems.

-----

## System Architecture and Operational Flow

The LIFEGUARD system follows a multi-stage pipeline designed for safety and accuracy, transforming a spoken utterance into a verified command for an autonomous agent.

### Operational Workflow

1.  **Push-to-Talk:** The operator presses and holds the spacebar to begin recording a command. The system only listens while this key is held.
2.  **Audio Preprocessing:** The captured audio is immediately processed to remove background noise and is filtered to isolate the frequency range of human speech, significantly improving recognition accuracy.
3.  **Speech Recognition:** The cleaned audio is transcribed into text in real-time by the offline Vosk STT engine.
4.  **Intent Extraction:** A custom-trained spaCy NLU model parses the transcribed text to identify the core intent (e.g., `GRID_SEARCH`) and extract key entities, including converting spoken numbers to digits (e.g., "forty one point three seven" → 41.37).
5.  **Agent Selection:** If the command includes an agent directive (e.g., "select drone two"), the system switches its active MAVLink connection. The currently selected agent is retained for subsequent commands.
6.  **Command Translation:** A valid intent is mapped to a sequence of MAVLink commands. For example, a `GRID_SEARCH` intent is translated into a complete mission plan with calculated waypoints.
7.  **User Confirmation:** The system verbalizes its interpretation of the command and asks the operator for confirmation (e.g., "Confirm grid search at latitude 41.37, longitude -72.09. Say yes or no."). This is the final safeguard against misinterpretation.
8.  **Action Execution:** Upon receiving a "yes" confirmation, the MAVLink commands are sent to the selected autonomous unit for execution.

### Technology Stack

The system is built on a foundation of powerful, open-source libraries chosen for their performance and suitability for offline, real-time applications.

| Component/Library | Role in LIFEGUARD |
| :---------------------------------------------------- | :----------------------------------------------------- |
| **Vosk** | Offline Speech-to-Text (STT) Engine |
| **spaCy** | Natural Language Understanding (NLU) & Entity Extraction |
| **pymavlink** | MAVLink Protocol Communication with Autonomous Units |
| **noisereduce** | Real-time Audio Noise Reduction |
| **pynput** | Keyboard Event Handling (Push-to-Talk & Shutdown) |
| **python-statemachine** | Manages Application State (Listening, Confirming, etc.) |
| **PyAudio** | Microphone Audio Input Stream |
| **scipy, numpy** | Signal Processing & Numerical Operations |
| **pyttsx3, word2number** | Text-to-Speech (for confirmation) & Number Conversion |

-----

## Command Reference and Use Cases

The NLU model is trained to recognize a specific grammar of commands tailored for search and rescue operations.

### Command Grammar Examples

| Voice Command Example | Extracted Intent | Key Entities Extracted |
| :--------------------------------------------------------------------------- | :------------------ | :------------------------------------------------------------- |
| "Search a **100 meter** grid at latitude **38.99** longitude **-76.48**." | `GRID_SEARCH` | `GRID_SIZE: 100`, `LATITUDE: 41.37`, `LONGITUDE: -72.09` |
| "Search at latitude **38.99** longitude **-76.48** for a person in a life ring." | `WAYPOINT_SEARCH` | `LATITUDE: 41.37`, `LONGITUDE: -72.09`, `TARGET: person in a life ring` |
| "Select **drone two**." | `SELECT_AGENT` | `AGENT_ID: 2` |
| "Set altitude to **fifty meters**." | `SET_ALTITUDE` | `ALTITUDE: 50` |
| "Confirm command." / "Yes." | `CONFIRM_ACTION` | `CONFIRMATION: True` |
| "Cancel command." / "No." | `CANCEL_ACTION` | `CONFIRMATION: False` |

-----

## Setup and Requirements

### Minimum Requirements

  - Python 3.7+
  - [Vosk](https://alphacephei.com/vosk/) version 0.3.45 (STT engine and English model)
  - [spaCy](https://spacy.io/) version 3.0.0 with `en_core_web_sm` model
  - [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (microphone input)
  - [noisereduce](https://github.com/timsainb/noisereduce)
  - [pymavlink](https://github.com/ArduPilot/pymavlink)
  - [lat\_lon\_parser](https://pypi.org/project/lat-lon-parser/)
  - numpy, scipy, pynput, pyttsx3, statemachine, word2number

### Installation Steps

1.  **(Recommended) Create and Activate a Virtual Environment**

    ```bash
    python3 -m venv lifeguard_bot
    ```

    On Linux/macOS:

    ```bash
    source lifeguard_bot/bin/activate
    ```

    On Windows:

    ```bash
    lifeguard_bot\Scripts\activate
    ```

2.  **Clone the Repository**

    ```bash
    git clone https://github.com/John-Sarge/LIFEGUARD.git
    cd LIFEGUARD
    ```

3.  **Install Dependencies**
    The provided script will install all necessary Python packages.

    ```bash
    # Make the script executable on Linux/macOS
    chmod +x setup.sh
    ./setup.sh
    ```

4.  **(Linux Only) Install Audio Dependencies**
    PyAudio on Linux requires the PortAudio development libraries.

    ```bash
    sudo apt-get update && sudo apt-get install portaudio19-dev python3-pyaudio
    ```

5.  **Download Vosk Model**

      - Download a Vosk English model (e.g., `vosk-model-small-en-us-0.15`) from the [Vosk website](https://alphacephei.com/vosk/models).
      - Unzip the model and place the contents into the `vosk_models/` directory. The final path should look like `vosk_models/vosk-model-small-en-us-0.15/`.

-----

## Usage Guide

1.  **Connect MAVLink Vehicle(s)**

      - Connect your MAVLink-compatible vehicle(s) via USB, telemetry radio, or network.
      - Alternatively, start a simulator such as SITL or Gazebo.
      - Update the `AGENT_CONNECTION_CONFIGS` dictionary in `lifeguard.py` with the correct connection strings for your hardware or simulator setup.

2.  **Run the Application**

    ```bash
    python lifeguard.py
    ```

    The system will initialize and indicate that it is "Listening for commands...".

3.  **Issue a Voice Command**

      - Press and hold the **spacebar**.
      - Speak a command clearly into the microphone.
      - Example: "Search a 100 meter grid at latitude 41.37 longitude -72.09."
      - Release the spacebar.

4.  **Confirm the Intent**

      - The system will transcribe your command, display its interpretation, and ask for verbal confirmation.
      - Respond with "yes" to send the command or "no" to cancel.

5.  **Switch Agents (if applicable)**

      - To direct a command to a different vehicle, first issue a selection command.
      - Example: "Select drone two."
      - The system will confirm the agent switch, and all subsequent commands will be sent to that agent until a new one is selected.

-----

## Customization and Extensibility

The system is designed to be modular and extensible.

  - **Add New Commands:** Extend the NLU entity patterns and intent-handling logic in `lifeguard.py` to support new command types (e.g., "return to launch," "follow target").
  - **Configure Agents:** Modify the `AGENT_CONNECTION_CONFIGS` dictionary to add, remove, or update connection strings for different autonomous vehicles.
  - **Tune Parameters:** Adjust grid search parameters, default flight altitudes, and audio processing settings within the main script to match specific operational requirements.

-----

## Troubleshooting

  - **Vosk model not found:** Ensure the model has been downloaded, unzipped, and placed in the correct `vosk_models/` directory. The path in the script must match the folder name.
  - **PyAudio errors on startup:** Check that your microphone is connected and recognized by the operating system. On Linux, ensure the `portaudio19-dev` package is installed.
  - **MAVLink connection issues:** Verify your connection strings in `AGENT_CONNECTION_CONFIGS`. Ensure the vehicle/simulator is running and that the specified port (e.g., `/dev/ttyACM0`, `udp:127.0.0.1:14550`) is correct and not in use by another application.

-----

## Acknowledgments

This project stands on the shoulders of giants and is made possible by the incredible work of the open-source community, especially the teams behind Vosk, spaCy, and pymavlink.
