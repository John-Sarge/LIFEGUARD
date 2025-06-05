LIFEGUARD
Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments

Overview
LIFEGUARD is a Python-based system designed to interpret and process spoken natural language commands, allowing a human operator (Commander) to pass intent and instructions directly to autonomous units (such as drones or robots) during search and rescue operations. Its primary goal is to provide a robust bridge between human intent and autonomous action, focusing on ease of use, natural interaction, and reliability in dynamic field environments.

Key Features
Speech-to-Text (STT): Converts live spoken commands to text using Vosk.
Natural Language Understanding (NLU): Extracts intent and entities (like GPS coordinates, landmarks, and target descriptions) from the Commander’s speech using spaCy and custom entity recognizers.
MAVLink Integration: Translates high-level intent into actionable commands for MAVLink-compatible vehicles (e.g., drones), supporting waypoint navigation, mission uploads, mode changes, arming/disarming, and mission control.
Audio Preprocessing: Enhances command recognition with noise reduction and band-pass filtering for reliable operation in noisy environments.
Interactive Confirmation: Prompts for confirmation before sending critical commands to autonomous units, ensuring accuracy and safety.
How It Works
Microphone Input: The system continuously listens for spoken commands from the operator.
Audio Preprocessing: Incoming audio is denoised and filtered to improve recognition accuracy.
Speech Recognition: The processed audio is transcribed to text using the Vosk STT engine.
Intent Extraction: The text is analyzed to identify the Commander’s intent and extract actionable entities (such as GPS coordinates, targets, and locations) using spaCy with custom rules and entity recognition.
Command Translation: If a valid intent is detected (e.g., "Search at latitude 34.05 longitude -118.24 for a person in a life ring"), the system prepares corresponding MAVLink commands to instruct the autonomous unit.
User Confirmation: Before sending commands to the vehicle, the system asks the operator to confirm the interpreted intent.
Action Execution: Upon confirmation, the system sends the appropriate commands to the autonomous unit over MAVLink.
Example Use Case
A field operator says:

"Search at latitude 34.05 longitude -118.24 for a person in a life ring."

LIFEGUARD will:

Recognize and transcribe the command,
Extract the GPS coordinates and target description,
Prompt the operator for confirmation,
Send the search waypoint to the drone once confirmed.
Major Components
commander_intent_system.py: Main application file integrating all subsystems—audio processing, STT, NLU, and MAVLink communication.
Audio Processing: Uses noisereduce and scipy.signal for noise reduction and filtering.
Speech-to-Text: Utilizes Vosk models for offline speech recognition.
NLU: Implements spaCy pipelines with custom entity recognition for GPS and search/rescue-specific attributes.
MAVLink Controller: Employs pymavlink for sending navigation and mission commands to compatible vehicles.
Setup and Requirements
Python 3.7+
Vosk (STT engine and English model)
spaCy with en_core_web_sm model
PyAudio (microphone input)
noisereduce
pymavlink
lat_lon_parser
numpy, scipy
See requirements.txt (if available) or install the above via pip.

Usage
Download and extract the required Vosk model (e.g., vosk-model-small-en-us-0.15) into the vosk_models directory.
Connect your MAVLink-compatible vehicle or simulator (e.g., ArduPilot SITL).
Run the main application:
bash
python commander_intent_system.py
Speak your command into the microphone.
Confirm the interpreted intent when prompted.
Safety and Robustness
The system prompts for confirmation before sending critical commands.
Designed for field reliability, with robust error handling and state checks.
Assumes the vehicle is already in the correct mode and armed unless additional safety automation is implemented.
Customization
Entity patterns and NLU rules can be extended for additional command types or domain-specific needs.
MAVLink connection strings and parameters can be adjusted for different vehicle types and communication setups.
License
MIT License (or as specified in the repository)

Acknowledgments
Built with open-source libraries: Vosk, spaCy, pymavlink, and community-contributed tools.
Primary Purpose:
LIFEGUARD enables operators to naturally and effectively pass Commander’s Intent to autonomous units, bridging voice commands and machine action for efficient, reliable, and intuitive search and rescue deployments.

Let me know if you want further details, a badge section, or more technical setup instructions!
