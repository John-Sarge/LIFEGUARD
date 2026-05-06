<p align="center">
  <img width="400" alt="LIFEGUARD_github_logo" src="https://github.com/user-attachments/assets/068ae446-1a9b-40af-93cb-5b18b39b2531" />
</p>

# LIFEGUARD — Yard Patrol Edition

**Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments**

> LIFEGUARD enables operators to pass Command Intent to autonomous units through natural voice commands, bridging human intent and machine action for search and rescue operations.

-----

## Overview

LIFEGUARD is a Python-based mission control system for multi-agent drone SAR operations. Operators issue spoken commands that are transcribed offline, parsed for intent, and translated into MAVLink missions. The system also tracks a ship (YP) via its own MAVLink connection and includes a dedicated Man Overboard workflow. All subsystems run offline, prioritising operational robustness without internet dependency.

-----

[![Youtube Video](https://github.com/user-attachments/assets/fd5a14a0-3b2e-4914-af99-747e4163ee3d)](https://youtu.be/H5sFDNrni64?si=bcq4UI8XKhi0WGQx)

-----

## Key Features

- **Speech-to-Text (STT):** Live spoken commands transcribed offline by the Vosk engine.
- **Natural Language Understanding (NLU):** spaCy extracts intent and entities (GPS coordinates, targets, agent selection) with spoken-number conversion.
- **Multi-Agent MAVLink:** Connects to multiple MAVLink vehicles simultaneously; active agent is selected by voice.
- **Waypoint & Grid Search:** Generates and uploads lawnmower grid search missions from a single voice command.
- **Ship Tracking:** Connects to the ship's MAVLink system (position-only), displays an orange track on the map with configurable history, and shows a heading-oriented ship icon.
- **Man Overboard (MOB):** Dedicated red GUI button. Dispatches the first available agent on a curved-track-following parallel search pattern that follows every turn in the ship's recorded track. On FOUND, both the searching agent and a verification agent return dynamically to the ship's moving GPS position rather than a fixed point.
- **Heading-Oriented Map Icons:** Agent and ship icons rotate to reflect live MAVLink heading from ``GLOBAL_POSITION_INT``.
- **Push-to-Talk (PTT):** Press-and-hold GUI button captures audio only while held.
- **Confirmation Loop:** The system reads back its interpretation and waits for a verbal "yes" before dispatching any mission.
- **Offline Map Cache:** Tile cache stored in ``map_cache.db``; no internet required in the field.
- **Decoupled Threads:** Audio, STT, NLU, MAVLink I/O, and TTS each run in dedicated threads communicating via thread-safe queues.

-----

## System Architecture

All subsystems run as daemon threads and communicate via ``queue.Queue``. A ``UIMessenger`` bridge forwards status strings and typed tuples into the GUI queue, which the main thread drains every 100 ms via ``after()``.

```
Microphone → AudioInputWorker → STTWorker → Coordinator → NLUWorker
                                                  ↓
                                           MavlinkWorker ← MsgMOBActivated (GUI button)
                                                  ↓
                                        MAVLink vehicles + Ship
                                                  ↓
                                          UIMessenger → LifeguardGUI queue
```

### Technology Stack

| Component | Role |
| :--- | :--- |
| **Vosk** | Offline Speech-to-Text |
| **spaCy** | NLU — intent classification and entity extraction |
| **pymavlink** | MAVLink communication with drones and ship |
| **noisereduce / scipy** | Audio noise reduction and bandpass filtering |
| **customtkinter / Pillow** | GUI framework and heading-oriented map icon rendering |
| **tkintermapview** | Interactive map with offline tile cache |
| **PyAudio** | Microphone input stream |
| **pyttsx3** | Text-to-Speech confirmation output |
| **word2number** | Spoken-number-to-digit conversion |

-----

## Configuration

``config.json`` is created automatically on first run. Edit it directly or use the **Settings** dialog in the GUI.

| Section | Key fields |
| :--- | :--- |
| ``agents`` | ``name``, ``connection_string`` (e.g. ``tcp:10.24.5.232:5763``) |
| ``mission`` | ``default_waypoint_altitude``, ``default_swath_width`` |
| ``nlu`` | ``confidence_threshold`` |
| ``ship`` | ``name``, ``connection_string``, ``track_history_minutes``, ``mob_corridor_half_width_m`` |

The ship connection is read-only; no commands are sent to the ship.

-----

## Command Reference

| Voice Command | Intent | Entities |
| :--- | :--- | :--- |
| "Search a 100 meter grid at latitude 38.99 longitude -76.48." | ``REQUEST_GRID_SEARCH`` | ``grid_size_meters``, ``latitude``, ``longitude`` |
| "Search at latitude 38.99 longitude -76.48 for a person in a life ring." | ``COMBINED_SEARCH_AND_TARGET`` | ``latitude``, ``longitude``, ``target_description_full`` |
| "Fly to latitude 38.99 longitude -76.48." | ``REQUEST_FLY_TO`` | ``latitude``, ``longitude`` |
| "Select drone two." | ``SELECT_AGENT`` | ``selected_agent_id`` |
| "Set altitude to fifty meters." | ``SET_AGENT_ALTITUDE`` | ``altitude_meters`` |
| "Yes." / "No." | *(confirmation mode)* | — |

-----

## Man Overboard Workflow

1. Operator presses **MAN OVERBOARD** (red button, right of Push to Talk).
2. ``MsgMOBActivated`` is injected directly into the MAVLink worker inbox.
3. The first idle connected agent is dispatched on a **curved-track-following parallel search** built from the ship's recorded GPS track (filtered and bearing-smoothed to remove GPS noise).
4. The search pattern starts from whichever end of the corridor is closest to the drone's current position.
5. On ``FOUND:<lat>,<lon>`` STATUSTEXT from the searching agent:
   - That agent returns to the ship's **current** GPS position (updated every 5 s until arrival).
   - A second agent is dispatched on a 50 m verification grid at the reported location.
6. On ``FOUND`` from the verification agent, it also returns dynamically to the ship.

-----

## Setup

### Requirements

- Python 3.9+ (see ``pyproject.toml``)
- [Vosk](https://alphacephei.com/vosk/) + model ``vosk-model-small-en-us-0.15``
- [spaCy](https://spacy.io/) with ``en_core_web_sm``
- pymavlink, PyAudio, noisereduce, scipy, numpy, pyttsx3, word2number
- customtkinter, tkintermapview, Pillow

### Installation

1. **Create and activate a virtual environment**

   ```bash
   python3 -m venv lifeguard_build_env
   # Windows:
   lifeguard_build_env\Scripts\activate
   # Linux/macOS:
   source lifeguard_build_env/bin/activate
   ```

2. **Clone and install**

   ```bash
   git clone https://github.com/John-Sarge/LIFEGUARD.git
   cd LIFEGUARD
   chmod +x setup.sh && ./setup.sh   # Linux/macOS
   # or on Windows: pip install -e .
   ```

3. **Linux only — PortAudio**

   ```bash
   sudo apt-get install portaudio19-dev python3-pyaudio
   ```

4. **Download Vosk model**

   Download ``vosk-model-small-en-us-0.15`` from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and place it at ``vosk_models/vosk-model-small-en-us-0.15/``.

-----

## Running

```bash
python run_gui.py
```

The package must be installed in editable mode (``pip install -e .``) or ``src/`` must be on ``PYTHONPATH``.

-----

## Usage

1. Connect MAVLink vehicles and (optionally) the ship; configure connection strings in **Settings → Agents** and **Settings → Ship**.
2. Press and hold **Push to Talk**, speak a command, release.
3. LIFEGUARD reads back its interpretation — say **"yes"** to confirm or **"no"** to cancel.
4. To switch active agent: *"Select drone two."*
5. For a Man Overboard event: press **MAN OVERBOARD** — no voice confirmation required.

-----

## Test Simulator

``agent1_grid_sim.py`` exercises the FOUND workflow without hardware:

- Connects to a SITL/MAVLink endpoint, waits for an AUTO mission, then sends ``FOUND:<lat>,<lon>`` via STATUSTEXT at a mid-mission waypoint.
- LIFEGUARD responds by returning the searching agent to ship and dispatching a verifier.
- Append ``--handshake-test --duration 30`` to verify STATUSTEXT round-trip latency on hardware.
- **Note:** mavlink-router ([mavlink-anywhere](https://github.com/alireza787b/mavlink-anywhere)) is used instead of MAVProxy because MAVProxy silently consumed STATUSTEXT messages on some firmware versions.

Adjust the connection string at the bottom of ``agent1_grid_sim.py`` for your setup (e.g. ``tcp:127.0.0.1:5763``).

-----

## Packaging

```bash
pip install setuptools wheel
python -m build
pip install dist/lifeguard-*.whl
```

-----

## Acknowledgments

This project is made possible by the open-source community, especially the teams behind Vosk, spaCy, and pymavlink.
