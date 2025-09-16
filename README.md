<p align="center">
  <img width="400" alt="LIFEGUARD_github_logo" src="https://github.com/user-attachments/assets/068ae446-1a9b-40af-93cb-5b18b39b2531" />
</p>

# LIFEGUARD

**Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments**

> LIFEGUARD enables operators to naturally and effectively pass Command Intent to autonomous units, bridging voice commands and machine action for efficient, reliable, and intuitive search and rescue deployments.

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
  - **Push-to-Talk (PTT):** Via GUI button (press & hold). (Legacy spacebar handler replaced in current GUI launcher.)
  - **Decoupled Subsystems:** Audio Input/PTT, STT, NLU, MAVLink I/O, and TTS Output run in dedicated threads and communicate via thread-safe queues for improved reliability and responsiveness.
  - **Modular Message-Passing Architecture:** Improves reliability, thread safety, and responsiveness for mission-critical operations.
  - **Cross-Platform:** Works on both Windows and Linux operating systems.

-----

## System Architecture and Operational Flow

The LIFEGUARD system uses a decoupled, multi-threaded message-passing architecture. Subsystems (Audio Input/PTT, STT, NLU, MAVLink I/O, TTS Output) run in dedicated threads and communicate via thread-safe queues. This design improves reliability, thread safety, and responsiveness for mission-critical operations.

### Operational Workflow

1.  **Push-to-Talk:** The operator presses and holds the GUI "Push to Talk" button. Audio is captured only while held.
2.  **Audio Preprocessing:** The captured audio is processed for noise reduction and bandpass filtering to isolate speech frequencies.
3.  **Speech Recognition:** The cleaned audio is transcribed into text in real-time by the offline Vosk STT engine.
4.  **Intent Extraction:** spaCy NLU parses the transcribed text to identify intent and extract entities, including spoken number conversion.
5.  **Agent Selection:** Voice commands can select the active MAVLink agent (drone/robot). The selected agent is retained for subsequent commands.
6.  **Command Translation:** Valid intents are mapped to MAVLink commands (e.g., grid search mission upload).
7.  **User Confirmation:** The system verbalizes its interpretation and asks for confirmation before executing safety-critical actions.
8.  **Action Execution:** Upon confirmation, commands are sent to the selected autonomous unit for execution.

### Technology Stack

The system is built on a foundation of powerful, open-source libraries chosen for their performance and suitability for offline, real-time applications.

| Component/Library | Role in LIFEGUARD |
| :---------------------------------------------------- | :----------------------------------------------------- |
| **Vosk** | Offline Speech-to-Text (STT) Engine |
| **spaCy** | Natural Language Understanding (NLU) & Entity Extraction |
| **pymavlink** | MAVLink Protocol Communication with Autonomous Units |
| **noisereduce** | Real-time Audio Noise Reduction |
| **pynput** | (Legacy) keyboard event handling (may be unused with GUI button) |
| **customtkinter / Pillow** | GUI framework and image assets |
| **Threading/Queues** | Decoupled subsystem communication and message routing |
| **PyAudio** | Microphone Audio Input Stream |
| **scipy, numpy** | Signal Processing & Numerical Operations |
| **pyttsx3, word2number** | Text-to-Speech (for confirmation) & Number Conversion |

-----

## Command Reference and Use Cases

The NLU model is trained to recognize a specific grammar of commands tailored for search and rescue operations.

### Command Grammar Examples

| Voice Command Example | Extracted Intent | Key Entities Extracted |
| :--------------------------------------------------------------------------- | :------------------------------ | :------------------------------------------------------------- |
| "Search a 100 meter grid at latitude 38.99 longitude -76.48." | `REQUEST_GRID_SEARCH` | `grid_size_meters=100`, `latitude`, `longitude` |
| "Search at latitude 38.99 longitude -76.48 for a person in a life ring." | `COMBINED_SEARCH_AND_TARGET` or `REQUEST_SEARCH_AT_LOCATION` | `latitude`, `longitude`, `target_description_full` |
| "Fly to latitude 38.99 longitude -76.48." | `REQUEST_FLY_TO` | `latitude`, `longitude` |
| "Select drone two." | `SELECT_AGENT` | `selected_agent_id=agent2` |
| "Set altitude to fifty meters." | `SET_AGENT_ALTITUDE` | `altitude_meters=50` |
| "Yes." / "No." (after prompt) | (Handled by confirmation mode) | (No separate intent label) |

Note: Implemented intents now include `SELECT_AGENT`, `REQUEST_GRID_SEARCH`, `REQUEST_SEARCH_AT_LOCATION`, `REQUEST_FLY_TO`, `COMBINED_SEARCH_AND_TARGET`, `SET_AGENT_ALTITUDE`, and `PROVIDE_TARGET_DESCRIPTION` (fallback when only a target description is given). Yes/No confirmations are handled by a coordinator confirmation mode (no explicit confirmation intent constant).

-----

## Setup and Requirements

### Minimum Requirements

  - Python 3.9+ (matches `pyproject.toml`)
  - [Vosk](https://alphacephei.com/vosk/) version 0.3.45
  - [spaCy](https://spacy.io/) version 3.7.5 with `en_core_web_sm` model
  - [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (microphone input)
  - [noisereduce](https://github.com/timsainb/noisereduce)
  - [pymavlink](https://github.com/ArduPilot/pymavlink)
  - [numpy](https://numpy.org/) <2.0
  - [scipy](https://scipy.org/) <1.14
  - [pynput](https://pypi.org/project/pynput/) (optional / legacy)
  - [pyttsx3](https://pypi.org/project/pyttsx3/)
  - [word2number](https://pypi.org/project/word2number/)
  - [customtkinter](https://github.com/TomSchimansky/CustomTkinter) (GUI)
  - [tkintermapview](https://github.com/TomSchimansky/TkinterMapView) (map display & offline tile caching)
  - [Pillow](https://python-pillow.org/) (logo/icon / image assets)

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
  The provided script will install core Python packages (those listed in `pyproject.toml`).

    ```bash
    # Make the script executable on Linux/macOS
    chmod +x setup.sh
    ./setup.sh
    ```

  If the GUI reports missing modules (e.g. `ModuleNotFoundError: customtkinter` or `tkintermapview`), install them manually:
  ```bash
  pip install customtkinter tkintermapview Pillow
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
  - Edit `config.json` (auto-created on first run) or use the GUI Settings dialog (Agents tab) to adjust connection strings.

2.  **Run the Application**

  GUI launcher (recommended for current PTT workflow and map display — tiles cached to `map_cache.db`):

  ```bash
  python run_gui.py
  ```

  Headless/CLI (legacy entry point):

  ```bash
  python -m lifeguard
  ```

  Ensure your PYTHONPATH includes `src/` or install in editable mode (`pip install -e .`).

3.  **Issue a Voice Command**

  - Press and hold the GUI **Push to Talk** button.
  - Speak a command clearly (e.g., "Search a 100 meter grid at latitude 41.37 longitude -72.09.").
  - Release the button.

4.  **Confirm the Intent**

      - The system will transcribe your command, display its interpretation, and ask for verbal confirmation.
      - Respond with "yes" to send the command or "no" to cancel.

5.  **Switch Agents (if applicable)**

      - To direct a command to a different vehicle, first issue a selection command.
      - Example: "Select drone two."
      - The system will confirm the agent switch, and all subsequent commands will be sent to that agent until a new one is selected.

-----

## Customization and Extensibility

Key extension points:

  - **Commands & NLU:** Update patterns/logic in `src/lifeguard/components/nlu.py`.
  - **Agents:** Adjust via `config.json` (or GUI Settings); environment overrides also supported via `settings.py`.
  - **Audio & STT:** Tune pre-processing in `src/lifeguard/utils/audio_processing.py` and STT in `src/lifeguard/components/stt.py`.

-----

## Troubleshooting

  - **Vosk model not found:** Ensure the model has been downloaded, unzipped, and placed in the correct `vosk_models/` directory. The path in the script must match the folder name.
  - **PyAudio errors on startup:** Check that your microphone is connected and recognized by the operating system. On Linux, ensure the `portaudio19-dev` package is installed.
  - **MAVLink connection issues:** Verify your connection strings in `AGENT_CONNECTION_CONFIGS`. Ensure the vehicle/simulator is running and that the specified port (e.g., `/dev/ttyACM0`, `udp:127.0.0.1:14550`) is correct and not in use by another application.


## Test Simulator: agent1_grid_sim.py

A small simulator script, `agent1_grid_sim.py`, is included to exercise the LIFEGUARD workflow without a full field setup.

  - Note: Had to switch from using mavproxy on agents as the STATUSTEXT messages used to send target messages was being silently consumed by the autopilot, using mavlink-router now (https://github.com/alireza787b/mavlink-anywhere).
  - What it does: Connects to a MAVLink endpoint (e.g., SITL) and, once a mission is running in AUTO, reports a simulated find by sending a STATUSTEXT message: `FOUND:<lat>,<lon>` at a waypoint roughly in the middle of the mission.
  - How it works: It infers mission readiness from STATUSTEXT, can pull the mission from the vehicle to get waypoint coordinates, waits for `MISSION_ITEM_REACHED`, and then emits the `FOUND` message.
  - Why it’s useful: LIFEGUARD listens for `FOUND:` messages to automatically dispatch another available agent to verify the reported location.
  - Quick start: Adjust the connection string at the bottom of the desired simulator file (e.g., `agent1_grid_sim.py`) for your SITL/vehicle (e.g., `tcp:127.0.0.1:5763`) and run it in parallel while following the normal README flow. (add "--handshake-test --duration 30" to verify STATUSTEXT message functionality in hardware)

## Packaging and Deployment

### Build and Install as a Python Package

1. Ensure you have `setuptools` and `wheel` installed:
  ```bash
  pip install setuptools wheel
  ```

2. Build the package:
  ```bash
  python -m build
  ```

3. Install the package locally:
  ```bash
  pip install dist/lifeguard-*.whl
  ```

### Publish to PyPI (optional)

1. Ensure you have `twine` installed:
  ```bash
  pip install twine
  ```

2. Upload the package:
  ```bash
  twine upload dist/*
  ```

Refer to [PyPI packaging guide](https://packaging.python.org/tutorials/packaging-projects/) for more details.

## Acknowledgments



This project stands on the shoulders of giants and is made possible by the incredible work of the open-source community, especially the teams behind Vosk, spaCy, and pymavlink.
