# lifeguard.py
# This script implements a voice-controlled ground station for MAVLink-compatible uncrewed vehicles.
# It integrates Speech-to-Text (STT), Natural Language Understanding (NLU), and MAVLink communication
# to allow users to issue commands like "search a grid at [location]" or "fly to [location]".

import os
import json
import re
import time
import asyncio # Required for some Pymavlink operations if they become async in future versions
import threading
import math # For geographical calculations (e.g., grid generation)

# Audio Processing Libraries
import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr

# Speech-to-Text (STT) Library
import vosk

# Natural Language Understanding (NLU) Libraries
import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import EntityRuler
from lat_lon_parser import parse as parse_lat_lon_external # External library for robust lat/lon parsing

# MAVLink Communication Libraries
from pymavlink import mavutil, mavwp

# Push-To-Talk (PTT), Text-To-Speech (TTS), and State Machine Libraries
from pynput import keyboard
import pyttsx3
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed
from word2number import w2n # Converts spoken numbers (e.g., "forty two") to digits (42)
import re

# --- CONFIGURATION CONSTANTS ---

# STT Configuration
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15" # Name of the Vosk model folder
VOSK_MODEL_PATH = os.path.join("vosk_models", VOSK_MODEL_NAME) # Full path to the Vosk model directory
AUDIO_SAMPLE_RATE = 16000 # Sample rate for audio recording (Hz)
AUDIO_CHANNELS = 1 # Number of audio channels (1 for mono)
AUDIO_FORMAT = pyaudio.paInt16 # Audio format (16-bit integers)
AUDIO_FRAMES_PER_BUFFER = 8192 # Number of frames per buffer for PyAudio stream opening
AUDIO_READ_CHUNK = 4096 # Amount of audio data (frames) to read from the stream at a time

# Audio Pre-processing Configuration
NOISE_REDUCE_TIME_CONSTANT_S = 2.0 # Time constant for the noise reduction algorithm (seconds)
NOISE_REDUCE_PROP_DECREASE = 0.9 # Proportion by which noise is decreased
BANDPASS_LOWCUT_HZ = 300.0 # Lower cutoff frequency for the band-pass filter (Hz)
BANDPASS_HIGHCUT_HZ = 3400.0 # Upper cutoff frequency for the band-pass filter (Hz)
BANDPASS_ORDER = 5 # Order of the Butterworth band-pass filter

# NLU Configuration
SPACY_MODEL_NAME = "en_core_web_sm" # Name of the spaCy language model to load
NLU_CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence score for NLU interpretation to trigger a command

# MAVLink Configuration
MAVLINK_BAUDRATE = 115200 # Baudrate for serial MAVLink connections
MAVLINK_SOURCE_SYSTEM_ID = 255 # MAVLink system ID for the Ground Control Station (GCS)
DEFAULT_WAYPOINT_ALTITUDE = 30.0 # Default altitude for generated waypoints (meters)
DEFAULT_SWATH_WIDTH_M = 20.0 # Default swath width for search grid patterns (meters)

# Agent Configuration (for single or multi-agent setup)
# This dictionary defines the connection parameters for each uncrewed vehicle (agent).
# For mavlink-router, each agent would typically be a different UDP endpoint.
AGENT_CONNECTION_CONFIGS = {
    "agent1": { # Default single agent configuration
        "connection_string": 'tcp:10.24.5.232:5762', # MAVLink connection string (e.g., TCP, UDP, serial)
        "source_system_id": MAVLINK_SOURCE_SYSTEM_ID, # Source system ID for this agent's connection
        "baudrate": MAVLINK_BAUDRATE # Baudrate, relevant only for serial connections
    }
    # Additional agents can be added here, e.g.:
    # "agent2": {
    #    "connection_string": 'udpin:localhost:14560', # Example for a second agent
    #    "source_system_id": MAVLINK_SOURCE_SYSTEM_ID,
    #    "baudrate": MAVLINK_BAUDRATE
    # }
}
INITIAL_ACTIVE_AGENT_ID = "agent1" # The ID of the agent to command by default upon startup

# Earth radius in meters for geographical calculations (used in grid generation)
EARTH_RADIUS_METERS = 6378137.0

# --- AUDIO PRE-PROCESSING FUNCTIONS ---

def apply_noise_reduction(audio_data_int16, sample_rate):
    """
    Applies noise reduction to int16 audio data.
    Converts audio to float for processing, then back to int16.
    """
    audio_data_float = audio_data_int16.astype(np.float32) / 32767.0 # Normalize to float range [-1.0, 1.0]
    reduced_audio_float = nr.reduce_noise(y=audio_data_float,
                                           sr=sample_rate,
                                           time_constant_s=NOISE_REDUCE_TIME_CONSTANT_S,
                                           prop_decrease=NOISE_REDUCE_PROP_DECREASE,
                                           n_fft=512) # FFT window size for noise reduction
    reduced_audio_int16 = (reduced_audio_float * 32767.0).astype(np.int16) # Convert back to int16
    return reduced_audio_int16

def apply_bandpass_filter(audio_data, sample_rate):
    """
    Applies a Butterworth band-pass filter to the audio data.
    This helps to isolate the human voice frequency range.
    """
    nyq = 0.5 * sample_rate # Nyquist frequency
    low = BANDPASS_LOWCUT_HZ / nyq # Normalized low cutoff frequency
    high = BANDPASS_HIGHCUT_HZ / nyq # Normalized high cutoff frequency
    b, a = butter(BANDPASS_ORDER, [low, high], btype='band') # Design the Butterworth filter
    filtered_audio = lfilter(b, a, audio_data) # Apply the filter
    return filtered_audio.astype(audio_data.dtype) # Ensure output type matches input

def spoken_numbers_to_digits(text):
    """
    Converts spoken number phrases (e.g., "forty two", "negative five point one")
    within a text string into their numerical digit equivalents (e.g., "42", "-5.1").
    Handles common STT misrecognitions like 'native' for 'negative'.
    """
    # Fix common STT misrecognitions for "negative"
    text = re.sub(r'\bnative\b', 'negative', text, flags=re.IGNORECASE)
    # Replace 'oh' or 'o' with 'zero' when used as a number (e.g., "oh five" -> "05")
    text = re.sub(r'\boh\b', 'zero', text, flags=re.IGNORECASE)
    text = re.sub(r'\bo\b', 'zero', text, flags=re.IGNORECASE)

    # Regex pattern to find potential spoken number phrases, including signs and decimals
    number_pattern = re.compile(
        r'\b(?P<sign>negative |minus )?(?P<number>(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
        r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|point|and| )+)\b',
        re.IGNORECASE
    )

    def repl(match):
        """Replacement function for the regex match."""
        phrase = match.group('number')
        sign = match.group('sign')
        try:
            num = w2n.word_to_num(phrase.strip()) # Convert words to number
            if sign:
                num = -abs(num) # Apply negative sign if present
            return f" {num} " # Return the number as a string, surrounded by spaces
        except Exception:
            return match.group(0) # If conversion fails, return the original matched text

    return number_pattern.sub(repl, text) # Apply the replacement across the text

# --- SPEECH-TO-TEXT (STT) MODULE ---

class SpeechToText:
    """
    Handles speech-to-text conversion using the Vosk library.
    Applies audio pre-processing before recognition.
    """
    def __init__(self, model_path, sample_rate):
        self.sample_rate = sample_rate
        if not os.path.exists(model_path):
            print(f"ERROR: Vosk model not found at {model_path}. Please download and place it correctly.")
            raise FileNotFoundError(f"Vosk model not found at {model_path}")

        self.model = vosk.Model(model_path) # Load the Vosk speech recognition model
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate) # Initialize the Kaldi recognizer
        self.recognizer.SetWords(True) # Enable word-level timestamps (optional, but can be useful)

    def recognize_buffer(self, raw_audio_buffer_bytes):
        """
        Processes a complete raw audio buffer (bytes) and returns the final transcription.
        Applies noise reduction and band-pass filtering before passing to Vosk.
        """
        if not raw_audio_buffer_bytes:
            self.recognizer.Reset() # Reset recognizer state if no audio
            return ""

        # Convert raw bytes to numpy array for audio processing
        audio_data_np = np.frombuffer(raw_audio_buffer_bytes, dtype=np.int16)
        # Apply pre-processing steps
        denoised_audio_np = apply_noise_reduction(audio_data_np, self.sample_rate)
        filtered_audio_np = apply_bandpass_filter(denoised_audio_np, self.sample_rate)
        processed_audio_bytes = filtered_audio_np.tobytes() # Convert back to bytes for Vosk

        text_result = ""
        # Process the audio waveform. AcceptWaveform processes a chunk, FinalResult gets the full result.
        if self.recognizer.AcceptWaveform(processed_audio_bytes):
            result_json = self.recognizer.Result() # Get result if a phrase is recognized
        else:
            result_json = self.recognizer.FinalResult() # Get final result if no more audio is expected

        self.recognizer.Reset() # Reset the recognizer for the next utterance

        try:
            result_dict = json.loads(result_json)
            text_result = result_dict.get('text', '') # Extract the 'text' field from the JSON result
        except json.JSONDecodeError:
            print(f"STT (Vosk): Could not decode JSON result: {result_json}")
        except Exception as e:
            print(f"STT (Vosk): Error processing result: {e}")
        return text_result

# --- NATURAL LANGUAGE UNDERSTANDING (NLU) MODULE ---

@Language.factory("lat_lon_entity_recognizer", default_config={"gps_label": "LOCATION_GPS_COMPLEX"})
def create_lat_lon_entity_recognizer(nlp: Language, name: str, gps_label: str):
    """
    SpaCy factory function to create a custom pipeline component for recognizing
    latitude and longitude entities.
    """
    return LatLonEntityRecognizer(nlp, gps_label)

class LatLonEntityRecognizer:
    """
    A spaCy custom pipeline component that identifies and extracts
    latitude and longitude coordinates from text using regular expressions.
    It adds a custom extension `parsed_gps_coords` to the spaCy Span object.
    """
    def __init__(self, nlp: Language, gps_label: str):
        self.gps_label = gps_label
        # Regex to capture various formats of latitude and longitude
        self.gps_regex = re.compile(
            r"""
            (?: # Non-capturing group for different formats
                # Match: latitude 41.37, longitude -72.09
                latitude\s*([+-]?\d{1,3}(?:\.\d+)?)\s*,\s*longitude\s*([+-]?\d{1,3}(?:\.\d+)?)
                | # OR: lat/lon or decimal pair (e.g., "41.37, -72.09" or "lat 41.37 lon -72.09")
                (?:[Ll][Aa](?:itude)?\s*[:\s]*)? # Optional "lat" or "latitude" prefix
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*?)\s*
                (?:[,;\s]+\s*)? # Separator (comma, semicolon, or space)
                (?:[Ll][Oo][Nn](?:gitude)?\s*[:\s]*)? # Optional "lon" or "longitude" prefix
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*[EeWw]?)
                | # OR simple decimal pair (e.g., "41.37, -72.09")
                ([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)
            )
            """,
            re.VERBOSE # Allows for multi-line regex with comments
        )
        # Add a custom extension to spaCy Span objects to store parsed GPS coordinates
        if not Span.has_extension("parsed_gps_coords"):
            Span.set_extension("parsed_gps_coords", default=None)

    def __call__(self, doc: Doc) -> Doc:
        """
        Processes a spaCy Doc object to find and label GPS coordinates.
        Iterates through regex matches, creates Span entities, and attaches parsed coordinates.
        Handles potential overlaps with existing entities.
        """
        print(f"[DEBUG] LatLonEntityRecognizer called on: '{doc.text}'")
        new_entities = list(doc.ents) # Start with existing entities
        for match in self.gps_regex.finditer(doc.text):
            print(f"[DEBUG] GPS regex match: {match.group(0)}")
            start_char, end_char = match.span() # Character start and end of the match
            lat_val, lon_val = None, None

            # Extract latitude and longitude based on which regex group matched
            if match.group(1) and match.group(2):
                lat_val = float(match.group(1))
                lon_val = float(match.group(2))
            elif match.group(3) and match.group(4):
                try:
                    lat_val = float(match.group(3))
                    lon_val = float(match.group(4))
                except Exception:
                    continue # Skip if conversion fails
            elif match.group(5) and match.group(6):
                try:
                    lat_val = float(match.group(5))
                    lon_val = float(match.group(6))
                except Exception:
                    continue # Skip if conversion fails
            else:
                continue # Skip if no valid lat/lon pair found in the match

            if lat_val is not None and lon_val is not None:
                # Validate coordinates are within realistic ranges
                if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                    try:
                        # Create a spaCy Span for the recognized entity
                        span = doc.char_span(start_char, end_char, label=self.gps_label, alignment_mode="expand")
                        if span is None:
                            # Try contract mode if expand fails
                            contract_span = doc.char_span(start_char, end_char, alignment_mode="contract")
                            if contract_span is not None:
                                span = Span(doc, contract_span.start, contract_span.end, label=self.gps_label)
                            else:
                                # Fallback: find the closest tokens covering the match
                                token_start = None
                                token_end = None
                                for i, token in enumerate(doc):
                                    if token.idx <= start_char < token.idx + len(token):
                                        token_start = i
                                    if token.idx < end_char <= token.idx + len(token):
                                        token_end = i + 1
                                if token_start is None:
                                    for i, token in enumerate(doc):
                                        if token.idx > start_char:
                                            token_start = i
                                            break
                                    if token_start is None:
                                        token_start = 0
                                if token_end is None:
                                    for i, token in enumerate(doc):
                                        if token.idx + len(token) >= end_char:
                                            token_end = i + 1
                                            break
                                    if token_end is None:
                                        token_end = len(doc)
                                # Final fallback: if still not valid, cover the whole doc
                                if token_start is None or token_end is None or token_start >= token_end:
                                    token_start = 0
                                    token_end = len(doc)
                                span = Span(doc, token_start, token_end, label=self.gps_label)
                        if span is not None:
                            # Attach the parsed coordinates to the custom extension
                            span._.set("parsed_gps_coords", {"latitude": lat_val, "longitude": lon_val})
                            # Remove ALL overlapping entities to avoid spaCy E1010 error (overlapping spans)
                            temp_new_entities = []
                            for ent in new_entities:
                                # Keep entities that do not overlap with the new span
                                if span.end <= ent.start or span.start >= ent.end:
                                    temp_new_entities.append(ent)
                            temp_new_entities.append(span) # Add the new, valid GPS span
                            new_entities = temp_new_entities
                    except Exception as e:
                        print(f"[DEBUG] Exception in LatLonEntityRecognizer: {e}")
        # Update the document's entities, ensuring they are sorted and unique
        doc.ents = tuple(sorted(list(set(new_entities)), key=lambda e: e.start_char))
        return doc

class NaturalLanguageUnderstanding:
    """
    Manages the spaCy NLP pipeline for parsing user commands.
    Identifies intents and extracts relevant entities (e.g., locations, target objects, agent IDs).
    """
    def __init__(self, spacy_model_name):
        try:
            self.nlp = spacy.load(spacy_model_name) # Load the specified spaCy model
        except OSError:
            print(f"spaCy model '{spacy_model_name}' not found. Please download it: python -m spacy download {spacy_model_name}")
            raise

        # Add the custom latitude/longitude entity recognizer to the pipeline
        if "lat_lon_entity_recognizer" not in self.nlp.pipe_names:
            # Add it after the built-in 'ner' component if it exists, otherwise at the end
            self.nlp.add_pipe("lat_lon_entity_recognizer", after="ner" if self.nlp.has_pipe("ner") else None)

        # Add a custom EntityRuler for domain-specific patterns (e.g., search and rescue terms)
        if "sar_entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", name="sar_entity_ruler", before="lat_lon_entity_recognizer")
            sar_patterns = [
                {"label": "TARGET_OBJECT", "pattern": [{"LOWER": "person"}]}, # Pattern for "person" as a target
                {"label": "TARGET_OBJECT", "pattern": [{"LOWER": "boat"}]}, # Pattern for "boat" as a target
                # Flexible patterns for "grid size" (e.g., "100 meter grid", "grid 50m")
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}, {"LOWER": "grid"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": "meter"}, {"LOWER": "grid"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": "meters"}, {"LOWER": "grid"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}, {"LOWER": "meter"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}, {"LOWER": "meters"}]},
                # Patterns for identifying agent IDs (e.g., "drone 1", "agent two")
                {"label": "AGENT_ID_NUM", "pattern": [{"LOWER": {"IN": ["drone", "agent"]}}, {"IS_DIGIT": True}]},
                {"label": "AGENT_ID_TEXT", "pattern": [{"LOWER": {"IN": ["drone", "agent"]}}, {"IS_ALPHA": True}]},
                # Action verbs for selecting agents
                {"label": "ACTION_SELECT", "pattern": [{"LEMMA": {"IN": ["select", "target", "choose", "activate"]}}]},
                # Action verb for search commands
                {"label": "ACTION_VERB", "pattern": [{"LEMMA": "search"}]},
            ]
            ruler.add_patterns(sar_patterns) # Add the defined patterns to the ruler

    def parse_command(self, text):
        """
        Parses a given text command using the spaCy NLP pipeline to determine intent
        and extract relevant entities.
        Returns a dictionary containing the recognized intent, confidence, and extracted entities.
        """
        doc = self.nlp(text) # Process the text with the spaCy pipeline
        intent = "UNKNOWN_INTENT"
        entities_payload = {}
        confidence = 0.5 # Default confidence

        extracted_spacy_ents = []
        for ent in doc.ents:
            entity_data = {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            # Handle custom entity data based on label
            if ent.label_ == "LOCATION_GPS_COMPLEX" and Span.has_extension("parsed_gps_coords") and ent._.get("parsed_gps_coords"):
                entity_data["parsed_gps"] = ent._.parsed_gps_coords
            elif ent.label_ == "GRID_SIZE_METERS":
                # Extract numeric part from grid size entity text
                numeric_part = "".join(filter(str.isdigit, ent.text))
                if numeric_part:
                    entity_data["value"] = int(numeric_part)
            elif ent.label_ == "AGENT_ID_NUM":
                # Format numeric agent ID (e.g., "drone 1" -> "agent1")
                numeric_part = "".join(filter(str.isdigit, ent.text))
                entity_data["value"] = f"agent{numeric_part}"
            elif ent.label_ == "AGENT_ID_TEXT":
                # Extract text part for agent ID (e.g., "drone alpha" -> "alpha")
                name_part = ent.text.lower().replace("drone", "").replace("agent", "").strip()
                entity_data["value"] = name_part
            extracted_spacy_ents.append(entity_data)
        entities_payload["raw_spacy_entities"] = extracted_spacy_ents # Store all extracted entities

        def get_entity_values(label, value_key="text"):
            """Helper to get values for a specific entity label."""
            return [e[value_key] for e in extracted_spacy_ents if e["label"] == label and value_key in e]

        def get_parsed_gps():
            """
            Helper to retrieve the most likely parsed GPS coordinates.
            Prefers the coordinate with latitude furthest from zero (more likely to be the primary one).
            """
            valid_gps = []
            for e in extracted_spacy_ents:
                if e["label"] == "LOCATION_GPS_COMPLEX" and "parsed_gps" in e:
                    lat = e["parsed_gps"].get("latitude")
                    lon = e["parsed_gps"].get("longitude")
                    if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                        valid_gps.append(e["parsed_gps"])
            if valid_gps:
                return max(valid_gps, key=lambda g: abs(g["latitude"])) # Select the one with largest absolute latitude
            return None

        # Extract specific entity types
        action_verbs = get_entity_values("ACTION_VERB")
        action_select_verbs = get_entity_values("ACTION_SELECT")
        target_objects = get_entity_values("TARGET_OBJECT")
        parsed_gps_coords = get_parsed_gps()
        grid_sizes = get_entity_values("GRID_SIZE_METERS", "value")
        # Fallback: regex extraction for grid size if spaCy ruler didn't catch it
        if not grid_sizes:
            grid_size_match = re.search(r'(\d+)\s*(?:meter|meters|m)\s*grid', text, re.IGNORECASE)
            if grid_size_match:
                grid_sizes = [int(grid_size_match.group(1))]
                print(f"[DEBUG] Regex fallback grid_sizes={grid_sizes}")
        agent_ids_num = get_entity_values("AGENT_ID_NUM", "value")
        agent_ids_text = get_entity_values("AGENT_ID_TEXT", "value")
        print(f"[DEBUG] grid_sizes={grid_sizes}")
        print(f"[DEBUG] parsed_gps_coords={parsed_gps_coords}")

        # Populate entities payload
        if parsed_gps_coords:
            entities_payload.update(parsed_gps_coords)

        if grid_sizes:
            entities_payload["grid_size_meters"] = grid_sizes[0] # Take the first found grid size

        selected_agent_id = None
        if agent_ids_num: selected_agent_id = agent_ids_num[0]
        elif agent_ids_text: selected_agent_id = agent_ids_text[0]

        # Determine intent based on extracted entities and keywords
        if selected_agent_id and action_select_verbs:
            intent = "SELECT_AGENT"
            entities_payload["selected_agent_id"] = selected_agent_id
            confidence = 0.9 # High confidence for clear agent selection

        target_details_parts = []
        if target_objects: target_details_parts.extend(target_objects)

        if target_details_parts:
            entities_payload["target_description_full"] = " ".join(target_details_parts)

        # Force grid search intent if command contains "grid" and grid size and GPS are present
        if (
            ("grid" in text.lower()) and
            grid_sizes and
            parsed_gps_coords
        ):
            intent = "REQUEST_GRID_SEARCH"
            entities_payload["grid_size_meters"] = grid_sizes[0] # Ensure grid size is in payload

        # Refine intent if still UNKNOWN_INTENT
        if intent == "UNKNOWN_INTENT":
            if action_verbs:
                entities_payload["action_verb"] = action_verbs[0]
                if parsed_gps_coords:
                    if "grid_size_meters" in entities_payload:
                        intent = "REQUEST_GRID_SEARCH" # Overlap with forced grid search, but good for robustness
                    elif target_details_parts:
                        intent = "COMBINED_SEARCH_AND_TARGET" # Search at location for a target
                    else:
                        intent = "REQUEST_SEARCH_AT_LOCATION" # Simple search at location
                elif target_details_parts:
                    intent = "PROVIDE_TARGET_DESCRIPTION" # Only providing target info, no movement
                else:
                    intent = "GENERIC_COMMAND" # General command, no specific action yet

        # Boost confidence for well-formed commands
        if intent not in ("UNKNOWN_INTENT",):
            if parsed_gps_coords or target_details_parts or intent == "SELECT_AGENT":
                confidence = max(confidence, 0.85) # Increase confidence for commands with key entities
            if intent == "REQUEST_GRID_SEARCH" and "grid_size_meters" in entities_payload and parsed_gps_coords:
                confidence = 0.95 # Very high confidence for complete grid search commands

        return {"text": text, "intent": intent, "confidence": confidence, "entities": entities_payload}

# --- MAVLINK INTEGRATION MODULE ---
class MavlinkController:
    """
    Manages MAVLink communication with a single uncrewed vehicle.
    Provides methods for connecting, sending waypoints, uploading missions,
    setting modes, arming/disarming, and sending status text.
    """
    def __init__(self, connection_string, baudrate=None, source_system_id=255):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.source_system_id = source_system_id
        self.master = None # MAVLink connection object
        self._connect() # Attempt to establish connection upon initialization

    def _connect(self):
        """
        Establishes a MAVLink connection to the specified endpoint.
        Waits for a heartbeat to confirm a valid connection to a vehicle.
        """
        print(f"MAVLink: Attempting to connect to {self.connection_string} with source_system {self.source_system_id}...")
        try:
            # Determine connection type (serial vs. UDP/TCP)
            if "com" in self.connection_string.lower() or "/dev/tty" in self.connection_string.lower():
                self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate, source_system=self.source_system_id)
            else: # Assume UDP/TCP connection
                self.master = mavutil.mavlink_connection(
                    self.connection_string,
                    source_system=self.source_system_id,
                    dialect="ardupilotmega", # Specify dialect for ArduPilot vehicles
                    autoreconnect=True, # Attempt to reconnect automatically if connection drops
                    use_native=False, # Use Python-based MAVLink parsing
                    force_connected=True, # Force connection even if heartbeat not immediately received (still waits for heartbeat later)
                    mavlink2=True # Use MAVLink 2 protocol
                )
            print(f"MAVLink: mavlink_connection object created for {self.connection_string}.")
            print(f"MAVLink: Waiting for heartbeat on {self.connection_string} (timeout 10s)...")
            
            # Wait for the first heartbeat message from the vehicle
            self.master.wait_heartbeat(timeout=10) 

            # CRITICAL CHECK: Validate the received heartbeat to ensure a real vehicle is connected
            if self.master.target_system == 0 and self.master.target_component == 0:
                print(f"MAVLink: Received invalid heartbeat (System ID 0). No actual vehicle detected on {self.connection_string}.")
                self.master = None # Treat as no connection if invalid heartbeat
            elif self.master.target_system!= 0 : # Check if a valid system ID was received
                print(f"MAVLink: Heartbeat received from system {self.master.target_system} component {self.master.target_component} on {self.connection_string}. Connection established.")
            else: # Catch-all for other unexpected scenarios after wait_heartbeat
                print(f"MAVLink: wait_heartbeat returned but no valid target system ID (is {self.master.target_system}). Connection failed for {self.connection_string}.")
                self.master = None
            
            print(f"MAVLink: Using target_system={self.master.target_system}, target_component={self.master.target_component}")
            # Always use target_component=1 for mission protocol (typically the autopilot)
            self.master.target_component = 1

        except Exception as e:
            print(f"MAVLink: EXCEPTION during connect or wait_heartbeat on {self.connection_string}: {e}")
            self.master = None # Set master to None if connection fails
            
    def is_connected(self):
        """Checks if the MAVLink connection is currently active."""
        return self.master is not None

    def send_nav_waypoint(self, lat, lon, alt, hold_time=0, accept_radius=10, pass_radius=0, yaw_angle=float('nan')):
        """
        Sends a single MAV_CMD_NAV_WAYPOINT command as a "guided mode go to" instruction.
        This is suitable for immediate, single-point navigation, not for full missions.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot send waypoint.")
            return False
        
        print(f"MAVLink: Sending MAV_CMD_NAV_WAYPOINT (as MISSION_ITEM_INT) to Lat: {lat}, Lon: {lon}, Alt: {alt}")
        try:
            self.master.mav.mission_item_int_send(
                self.master.target_system, self.master.target_component,
                0, # seq: Sequence number (0 for guided mode commands)
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, # Frame: Global coordinates with relative altitude
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # Command ID for navigation waypoint
                2, # current: 2 for guided mode command, 0 for mission item
                1, # autocontinue: 1 to proceed to next command (not applicable for single guided command)
                float(hold_time), float(accept_radius), float(pass_radius), float(yaw_angle), # Command parameters
                int(lat * 1e7), int(lon * 1e7), float(alt) # Latitude, Longitude (scaled), Altitude
            )
            print("MAVLink: MAV_CMD_NAV_WAYPOINT (as MISSION_ITEM_INT) sent.")
            return True
        except Exception as e:
            print(f"MAVLink: Error sending MAV_CMD_NAV_WAYPOINT (as MISSION_ITEM_INT): {e}")
            return False

    def upload_mission(self, waypoints_data):
        """
        Uploads a sequence of waypoints to the vehicle as a MAVLink mission.
        The process involves clearing existing missions, sending the waypoint count,
        and then sending each waypoint item upon request from the vehicle.
        """
        print("DEBUG: Entered upload_mission() with waypoints:", waypoints_data)
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot upload mission.")
            return False
        print("DEBUG: Connection is alive, proceeding with mission upload.")
        self.master.target_component = 1 # Ensure target component is autopilot (usually 1)
        print(f"MAVLink: Mission upload using target_system={self.master.target_system}, target_component={self.master.target_component}")
        
        wp_loader = mavwp.MAVWPLoader() # Helper for managing waypoints
        seq = 0
        # Always add a "home" waypoint as the first item (sequence 0) for ArduPilot compatibility.
        # This typically represents the vehicle's current location at mission start.
        home_lat, home_lon, home_alt = waypoints_data[0][0], waypoints_data[0][1], 0 # Use first waypoint's lat/lon, alt 0 for home
        home_item = mavutil.mavlink.MAVLink_mission_item_int_message(
            self.master.target_system, 1, seq, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0,
            int(home_lat * 1e7), int(home_lon * 1e7), float(home_alt)
        )
        wp_loader.add(home_item)
        seq += 1 # Increment sequence for subsequent waypoints

        # Add all provided waypoints to the loader
        for wp_item_tuple in waypoints_data:
            if len(wp_item_tuple) == 8: # (lat, lon, alt, command_id, p1, p2, p3, p4)
                lat, lon, alt, command_id, p1, p2, p3, p4 = wp_item_tuple
                frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
                is_current = 0
                autocontinue = 1
            elif len(wp_item_tuple) == 11: # (lat, lon, alt, command_id, p1, p2, p3, p4, frame, is_current, autocontinue)
                lat, lon, alt, command_id, p1, p2, p3, p4, frame, is_current, autocontinue = wp_item_tuple
            else:
                print(f"MAVLink: Waypoint item {seq} has incorrect format. Skipping.")
                continue
            
            mission_item = mavutil.mavlink.MAVLink_mission_item_int_message(
                self.master.target_system, 1, seq, frame, command_id,
                is_current, autocontinue, p1, p2, p3, p4,
                int(lat * 1e7), int(lon * 1e7), float(alt)
            )
            wp_loader.add(mission_item)
            seq += 1

        if wp_loader.count() == 0:
            print("MAVLink: No valid waypoints to upload.")
            return False

        print(f"MAVLink: Uploading {wp_loader.count()} waypoints...")
        try:
            # Step 1: Clear any existing mission on the vehicle
            print("MAVLink: Sending mission_clear_all_send")
            self.master.mav.mission_clear_all_send(self.master.target_system, 1)
            print("MAVLink: Waiting for MISSION_ACK after clear_all")
            ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            print(f"MAVLink: Received ACK: {ack}")
            time.sleep(1) # Short delay

            # Step 2: Inform the vehicle about the total number of waypoints to expect
            print("MAVLink: Sending mission_count_send")
            self.master.mav.mission_count_send(
                self.master.target_system,
                1, # Target component
                wp_loader.count(), # Total number of waypoints
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION # Mission type
            )

            # Step 3: Send waypoints one by one as requested by the vehicle
            for i in range(wp_loader.count()):
                print(f"MAVLink: Waiting for MISSION_REQUEST_INT for waypoint {i}")
                # Vehicle requests waypoints using MISSION_REQUEST or MISSION_REQUEST_INT
                msg = self.master.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=5)
                print(f"MAVLink: Received MISSION_REQUEST_INT: {msg}")
                if not msg:
                    print(f"MAVLink: No MISSION_REQUEST received for waypoint {i}, upload failed.")
                    return False
                print(f"MAVLink: Sending waypoint {msg.seq}")
                self.master.mav.send(wp_loader.wp(msg.seq)) # Send the requested waypoint
                if msg.seq == wp_loader.count() - 1:
                    break # Break after sending the last waypoint

            # Step 4: Wait for final MISSION_ACK to confirm successful upload
            print("MAVLink: Waiting for final MISSION_ACK")
            final_ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
            print(f"MAVLink: Received final ACK: {final_ack}")
            if final_ack and final_ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("MAVLink: Mission upload successful.")
                return True
            else:
                print(f"MAVLink: Mission upload failed or no/bad final ACK. ACK: {final_ack}")
                return False
        except Exception as e:
            print(f"MAVLink: Error during mission upload: {e}")
            return False

    def _calculate_rectangular_grid_waypoints(self, center_lat, center_lon, grid_width_m, grid_height_m, swath_width_m, altitude_m, orientation_deg=0):
        """
        Calculates waypoints for a rectangular lawnmower search pattern.
        The pattern consists of parallel tracks covering the specified grid area.
        Orientation 0 means North-South tracks, 90 means East-West tracks.
        This simplified example assumes North-South tracks (orientation_deg = 0).
        Returns a list of (latitude, longitude, altitude) tuples.
        """
        waypoints = []
        if swath_width_m <= 0 or grid_width_m <= 0 or grid_height_m <= 0:
            print("MAVLink: Invalid grid parameters for calculation.")
            return waypoints

        # Convert center lat/lon to radians for calculations
        center_lat_rad = math.radians(center_lat)
        center_lon_rad = math.radians(center_lon)

        # Calculate half dimensions of the grid
        half_width_m = grid_width_m / 2.0
        half_height_m = grid_height_m / 2.0
        
        # Calculate latitude boundaries (South and North) from the center
        lat_delta_rad = half_height_m / EARTH_RADIUS_METERS
        south_lat_rad = center_lat_rad - lat_delta_rad
        north_lat_rad = center_lat_rad + lat_delta_rad

        # Calculate longitude boundaries (West) from the center.
        # The longitude delta depends on the latitude (Earth's circumference varies with latitude).
        lon_delta_rad = half_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))
        west_lon_rad = center_lon_rad - lon_delta_rad
        # east_lon_rad = center_lon_rad + lon_delta_rad # Not directly used if stepping by swath

        # Determine the number of parallel tracks needed to cover the grid width
        num_tracks = int(math.floor(grid_width_m / swath_width_m))
        if num_tracks == 0 and grid_width_m > 0: num_tracks = 1 # Ensure at least one track if width > 0
        if num_tracks == 0: return waypoints # No tracks if grid width is zero

        # Calculate the step size in longitude radians for each swath
        swath_lon_rad_step = swath_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))

        # Generate waypoints for each track
        for i in range(num_tracks):
            # Calculate the longitude for the current track
            current_track_lon_rad = west_lon_rad + i * swath_lon_rad_step
            # Convert back to degrees for waypoint definition
            current_track_lon_deg = math.degrees(current_track_lon_rad)
            south_lat_deg = math.degrees(south_lat_rad)
            north_lat_deg = math.degrees(north_lat_rad)

            # Alternate direction for each track (lawnmower pattern)
            if i % 2 == 0:  # Northbound leg (starts from South latitude, goes to North latitude)
                waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
                waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
            else:  # Southbound leg (starts from North latitude, goes to South latitude)
                waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
                waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
        
        print(f"MAVLink: Calculated {len(waypoints)} waypoints for grid.")
        return waypoints

    def generate_and_upload_search_grid_mission(self, center_lat, center_lon, grid_size_m, swath_width_m, altitude_m):
        """
        Generates a square search grid mission and uploads it to the active vehicle.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot generate/upload search grid.")
            return False

        print(f"MAVLink: Generating search grid: Center=({center_lat:.6f}, {center_lon:.6f}), Size={grid_size_m}m, Swath={swath_width_m}m, Alt={altitude_m}m")

        # Calculate the actual waypoints for the grid
        grid_waypoints_tuples = self._calculate_rectangular_grid_waypoints(
            center_lat, center_lon, grid_size_m, grid_size_m, swath_width_m, altitude_m
        )

        if not grid_waypoints_tuples:
            print("MAVLink: Failed to generate grid waypoints.")
            return False

        # Format waypoints for mission upload (adding MAVLink command parameters)
        waypoints_data_for_upload = []
        for lat, lon, alt in grid_waypoints_tuples:
            waypoints_data_for_upload.append(
                (lat, lon, alt,
                 mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # Command for a navigation waypoint
                 0.0,  # param1 (hold time seconds)
                 10.0, # param2 (acceptance radius meters)
                 0.0,  # param3 (pass radius meters, 0 to pass through center)
                 float('nan') # param4 (desired yaw angle at waypoint, NaN for unchanged)
                )
            )
            
        return self.upload_mission(waypoints_data_for_upload) # Upload the generated mission

    def send_status_text(self, text_to_send, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
        """
        Sends a STATUSTEXT MAVLink message to the vehicle.
        This can be used to send arbitrary text messages to the vehicle's GCS or logs.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot send STATUSTEXT.")
            return False

        if isinstance(text_to_send, str):
            text_bytes = text_to_send.encode('utf-8') # Encode string to bytes
        else: # Assume already bytes
            text_bytes = text_to_send

        # STATUSTEXT has a 50-byte limit for text payload (49 characters + NUL terminator)
        payload_text_bytes = text_bytes[:49]

        try:
            self.master.mav.statustext_send(severity, payload_text_bytes)
            # Pymavlink handles NUL padding if shorter than 50 bytes.
            print(f"MAVLink: Sent STATUSTEXT: '{payload_text_bytes.decode('utf-8', errors='ignore')}' with severity {severity}")
            return True
        except Exception as e:
            print(f"MAVLink: Error sending STATUSTEXT: {e}")
            return False

    def wait_for_waypoint_reached(self, waypoint_sequence_id, timeout_seconds=120):
        """
        Waits for the vehicle to report that it has reached a specific mission waypoint.
        Monitors for MISSION_ITEM_REACHED messages.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot wait for waypoint.")
            return False
            
        print(f"MAVLink: Waiting for vehicle to reach mission waypoint sequence ID {waypoint_sequence_id}...")
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Receive MAVLink messages, blocking for up to 1 second
            msg = self.master.recv_match(type='MISSION_ITEM_REACHED', blocking=True, timeout=1)
            if msg:
                print(f"MAVLink: Received MISSION_ITEM_REACHED with sequence {msg.seq}")
                if msg.seq == waypoint_sequence_id:
                    print(f"MAVLink: Vehicle has reached target waypoint {waypoint_sequence_id}.")
                    return True
        print(f"MAVLink: Timeout waiting for waypoint {waypoint_sequence_id} to be reached.")
        return False

    def set_mode(self, mode_name):
        """
        Sets the flight mode of the vehicle (e.g., "AUTO", "GUIDED", "LOITER").
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot set mode.")
            return False

        # Get the mapping of mode names to mode IDs from the MAVLink connection
        mode_mapping = self.master.mode_mapping()
        if mode_mapping is None or mode_name.upper() not in mode_mapping:
            print(f"MAVLink: Unknown mode: {mode_name}")
            return False
        mode_id = mode_mapping[mode_name.upper()]

        print(f"MAVLink: Setting mode to {mode_name.upper()} (ID: {mode_id})...")
        try:
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, # Flag indicating custom mode is enabled
                mode_id # The ID of the desired flight mode
            )
            # Wait for a COMMAND_ACK message to confirm the mode change
            ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack_msg:
                print(f"MAVLink: Received COMMAND_ACK for mode change: {ack_msg}")
                if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print(f"MAVLink: Mode change to {mode_name.upper()} accepted.")
                    return True
                else:
                    print(f"MAVLink: Mode change to {mode_name.upper()} failed with result: {ack_msg.result}")
                    return False
            else:
                print("MAVLink: No COMMAND_ACK received for mode change.")
                return False
        except Exception as e:
            print(f"MAVLink: Error setting mode: {e}")
            return False

    def start_mission(self):
        """
        Sends a command to the vehicle to start the uploaded mission.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot start mission.")
            return False
        print("MAVLink: Sending MAV_CMD_MISSION_START...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START, # Command ID to start mission
            0, # confirmation (0 for no confirmation needed)
            0, # param1 (first_item: 0 to start from the beginning)
            0, # param2 (last_item: 0 for all items)
            0, 0, 0, 0, 0 # Unused parameters
        )
        # Wait for COMMAND_ACK to confirm mission start
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_MISSION_START and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Mission start command accepted.")
            return True
        else:
            print(f"MAVLink: Mission start command failed or no/bad ACK. ACK: {ack_msg}")
            return False

    def arm_vehicle(self):
        """
        Sends a command to arm the vehicle's motors.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot arm.")
            return False
        print("MAVLink: Attempting to arm vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, # Command ID for arm/disarm
            0, # confirmation
            1, # param1 (1 to arm, 0 to disarm)
            0, 0, 0, 0, 0, 0 # Unused parameters
        )
        # Wait for COMMAND_ACK to confirm arming
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5) # Longer timeout for arming
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Vehicle armed successfully.")
            return True
        else:
            # Common arming failure reasons: MAV_RESULT_TEMPORARILY_REJECTED (4), MAV_RESULT_FAILED (5)
            # These often correspond to safety checks failing (e.g., no GPS lock, pre-arm checks failed)
            res_text = f"result code {ack_msg.result}" if ack_msg else "no ACK"
            print(f"MAVLink: Arming failed ({res_text}). Ensure pre-arm checks are passing. ACK: {ack_msg}")
            return False

    def disarm_vehicle(self):
        """
        Sends a command to disarm the vehicle's motors.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot disarm.")
            return False
        print("MAVLink: Attempting to disarm vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0 # param1 (0 to disarm)
        )
        # Wait for COMMAND_ACK to confirm disarming
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Vehicle disarmed successfully.")
            return True
        else:
            print(f"MAVLink: Disarming failed or no ACK received. ACK: {ack_msg}")
            return False

    def close_connection(self):
        """Closes the MAVLink connection."""
        if self.master:
            self.master.close()
            print(f"MAVLink: Connection closed for {self.connection_string}.")

# --- STATE MACHINE DEFINITION ---
class LifeguardAppSM(StateMachine):
    """
    Defines the state machine for the Lifeguard application, managing the flow
    of user interaction (listening, processing, confirming, executing).
    """
    # Define states
    idle = State(initial=True) # Initial state: ready for a new command
    listening_command = State() # Recording user's command
    processing_command = State() # Processing recorded command (STT, NLU)
    speaking_confirmation = State() # Speaking confirmation prompt to user
    awaiting_yes_no = State() # Waiting for user to press PTT for yes/no
    listening_yes_no = State() # Recording user's yes/no response
    processing_yes_no = State() # Processing yes/no response (STT)
    executing_action = State() # Performing the MAVLink command

    # Define transitions between states based on events
    event_ptt_start_command = idle.to(listening_command) # PTT pressed in idle state
    event_ptt_stop_command = listening_command.to(processing_command) # PTT released after command
    event_command_recognized = processing_command.to(speaking_confirmation, cond="confirmation_needed_flag") | \
                               processing_command.to(executing_action, unless="confirmation_needed_flag") # Command recognized, transition based on confirmation need
    event_stt_failed_command = processing_command.to(idle) # STT failed for command, return to idle
    event_confirmation_spoken = speaking_confirmation.to(awaiting_yes_no) # Confirmation spoken, await yes/no
    event_ptt_start_yes_no = awaiting_yes_no.to(listening_yes_no) # PTT pressed for yes/no
    event_ptt_stop_yes_no = listening_yes_no.to(processing_yes_no) # PTT released after yes/no
    event_yes_confirmed = processing_yes_no.to(executing_action) # User confirmed with "yes"
    event_no_confirmed = processing_yes_no.to(idle) # User confirmed with "no", return to idle
    event_stt_failed_yes_no = processing_yes_no.to(awaiting_yes_no) # STT failed for yes/no, re-prompt
    event_action_completed = executing_action.to(idle) # MAVLink action completed, return to idle
    # Global shutdown event, can transition from any state to idle
    event_shutdown_requested = idle.to(idle) | listening_command.to(idle) | \
                               processing_command.to(idle) | speaking_confirmation.to(idle) | \
                               awaiting_yes_no.to(idle) | listening_yes_no.to(idle) | \
                               processing_yes_no.to(idle) | executing_action.to(idle)

    def __init__(self, model_ref):
        self.app = model_ref # Reference to the main LifeguardApp instance
        self.captured_command_audio_data = None # Stores audio for the main command
        self.captured_yes_no_audio_data = None # Stores audio for yes/no confirmation
        self._confirmation_needed_flag_internal = False # Internal flag to control confirmation flow
        super().__init__() # Initialize the StateMachine base class

    def confirmation_needed_flag(self, event_data):
        """Condition for state transition: checks if confirmation is needed."""
        return self._confirmation_needed_flag_internal

    # State entry/exit actions (callbacks)
    def on_enter_listening_command(self, event_data):
        """Action when entering the LISTENING_COMMAND state."""
        print("SM: State -> LISTENING_COMMAND")
        self.app.speak("Listening for command...")
        self.app.start_audio_recording()

    def on_exit_listening_command(self, event_data):
        """Action when exiting the LISTENING_COMMAND state."""
        print("SM: Exiting LISTENING_COMMAND")
        self.captured_command_audio_data = self.app.stop_audio_recording()

    def on_enter_processing_command(self, event_data):
        """Action when entering the PROCESSING_COMMAND state."""
        print("SM: State -> PROCESSING_COMMAND")
        # Process the captured audio data using STT and NLU
        command_text, needs_conf, is_stop_command = self.app.process_command_stt(self.captured_command_audio_data)
        if is_stop_command:
            self.app.initiate_shutdown("Stop command recognized.")
            self.send("event_shutdown_requested") # Trigger shutdown event
            return

        if command_text:
            self._confirmation_needed_flag_internal = needs_conf # Set confirmation flag based on NLU result
            self.send("event_command_recognized") # Trigger command recognized event
        else:
            self.app.speak("Sorry, I could not understand the command. Please try again.")
            self.send("event_stt_failed_command") # Trigger STT failed event

    def on_enter_speaking_confirmation(self, event_data):
        """Action when entering the SPEAKING_CONFIRMATION state."""
        print("SM: State -> SPEAKING_CONFIRMATION")
        prompt = f"Did you say: {self.app.command_to_execute_display_text}? Please press space and say yes or no."
        self.app.speak(prompt)
        self.send("event_confirmation_spoken") # Trigger confirmation spoken event

    def on_enter_awaiting_yes_no(self, event_data):
        """Action when entering the AWAITING_YES_NO state."""
        print("SM: State -> AWAITING_YES_NO")

    def on_enter_listening_yes_no(self, event_data):
        """Action when entering the LISTENING_YES_NO state."""
        print("SM: State -> LISTENING_YES_NO")
        self.app.speak("Listening for yes or no...")
        self.app.start_audio_recording()

    def on_exit_listening_yes_no(self, event_data):
        """Action when exiting the LISTENING_YES_NO state."""
        print("SM: Exiting LISTENING_YES_NO")
        self.captured_yes_no_audio_data = self.app.stop_audio_recording()

    def on_enter_processing_yes_no(self, event_data):
        """Action when entering the PROCESSING_YES_NO state."""
        print("SM: State -> PROCESSING_YES_NO")
        response = self.app.process_yes_no_stt(self.captured_yes_no_audio_data) # Process yes/no audio
        if response == "yes":
            self.send("event_yes_confirmed") # Trigger yes confirmed event
        elif response == "no":
            self.app.speak("Okay, command cancelled.")
            self.send("event_no_confirmed") # Trigger no confirmed event
        else:
            self.app.speak("Sorry, I didn't catch that. Please try saying yes or no again.")
            self.send("event_stt_failed_yes_no") # Trigger STT failed for yes/no, re-prompt

    def on_enter_executing_action(self, event_data):
        """Action when entering the EXECUTING_ACTION state."""
        print("DEBUG: Starting execute_lifeguard_command_wrapper thread")
        # Execute the MAVLink command in a separate thread to avoid blocking the main loop
        self.app.action_thread = threading.Thread(target=self.app.execute_lifeguard_command_wrapper)
        self.app.action_thread.daemon = False # Make it non-daemon so we can join it during shutdown
        self.app.action_thread.start()

    def on_enter_idle(self, event_data):
        """Action when entering the IDLE state (resetting for next command)."""
        source_id = event_data.source.id if event_data.source else "INITIAL"
        event_name = event_data.event if event_data.event else "INIT"
        print(f"SM: State -> IDLE (from {source_id} via {event_name})")
        # Clear command data and reset flags
        self.app.command_to_execute_display_text = None
        self.app.original_nlu_result_for_execution = None
        self._confirmation_needed_flag_internal = False
        if self.app.running_flag:
            active_agent_display = f" (Active Agent: {self.app.active_agent_id})" if self.app.active_agent_id else ""
            print(f"\nSystem ready{active_agent_display}. Press and hold SPACE to speak, or ESC to exit.")

# --- MAIN APPLICATION CLASS ---
class LifeguardApp:
    """
    The main application class that orchestrates STT, NLU, MAVLink communication,
    and the state machine for voice-controlled vehicle operations.
    """
    def __init__(self):
        print("Initializing Commander Intent System with PTT...")
        self.running_flag = True # Flag to control the main application loop
        self.pyaudio_instance = None
        self.stt_system = None
        self.nlu_system = None
        
        self.mav_controllers = {} # Dictionary to store MavlinkController instances for each agent
        self.active_agent_id = INITIAL_ACTIVE_AGENT_ID # Currently selected agent for commands
        if not AGENT_CONNECTION_CONFIGS:
            print("FATAL ERROR: No agents defined in AGENT_CONNECTION_CONFIGS. Exiting.")
            self.running_flag = False
            return
        if self.active_agent_id not in AGENT_CONNECTION_CONFIGS:
            # Fallback to the first agent in the config if initial active ID is invalid
            self.active_agent_id = next(iter(AGENT_CONNECTION_CONFIGS))
            print(f"Warning: Initial active agent ID '{INITIAL_ACTIVE_AGENT_ID}' not found. Defaulting to '{self.active_agent_id}'.")

        self.tts_engine = None # Text-to-Speech engine
        self.keyboard_listener = None # Keyboard listener for Push-To-Talk
        self.sm = None # State machine instance

        self.audio_frames = [] # Buffer to store recorded audio chunks
        self.is_space_currently_pressed = False # Flag for PTT state
        self.current_audio_stream = None # PyAudio stream object

        self.command_to_execute_display_text = None # Text of the command for confirmation
        self.original_nlu_result_for_execution = None # Full NLU result for command execution
        self.action_thread = None # Thread for executing MAVLink actions

        self._initialize_components() # Initialize all sub-systems
        if self.running_flag: # Only initialize SM if components are okay
            self.sm = LifeguardAppSM(model_ref=self)

    def _initialize_components(self):
        """Initializes PyAudio, STT, NLU, MAVLink controllers, and TTS engine."""
        self.pyaudio_instance = pyaudio.PyAudio()
        try:
            self.stt_system = SpeechToText(model_path=VOSK_MODEL_PATH, sample_rate=AUDIO_SAMPLE_RATE)
            print("STT (Vosk): Model loaded successfully.")
        except Exception as e:
            print(f"Fatal Error: Could not initialize STT system: {e}")
            self.running_flag = False; return

        try:
            self.nlu_system = NaturalLanguageUnderstanding(spacy_model_name=SPACY_MODEL_NAME)
            print("NLU (spaCy): Initialized successfully.")
        except Exception as e:
            print(f"Fatal Error: Could not initialize NLU system: {e}")
            self.running_flag = False; return

        # MAVLink Controllers Initialization
        if not AGENT_CONNECTION_CONFIGS:
            print("Fatal Error: AGENT_CONNECTION_CONFIGS is empty. No MAVLink agents to connect to.")
            self.running_flag = False; return

        for agent_id, config in AGENT_CONNECTION_CONFIGS.items():
            conn_str = config["connection_string"]
            # Baudrate is primarily for serial connections. For UDP/TCP, mavutil often infers or ignores it.
            baud = config.get("baudrate") if conn_str.startswith(("/dev/", "COM", "com")) else None
            src_sys_id = config.get("source_system_id", MAVLINK_SOURCE_SYSTEM_ID)
            
            print(f"Initializing MAVLink controller for agent: {agent_id} on {conn_str}")
            controller = MavlinkController(connection_string=conn_str, baudrate=baud, source_system_id=src_sys_id)
            if controller.is_connected():
                self.mav_controllers[agent_id] = controller
                print(f"MAVLink controller for agent {agent_id} connected successfully.")
            else:
                print(f"Warning: Failed to connect MAVLink controller for agent {agent_id} on {conn_str}.")
                # For now, we'll allow the app to run with some agents offline, but they won't be selectable.

        if not self.mav_controllers: # If no controllers connected successfully
            print("Fatal Error: Could not connect to any MAVLink agents specified in AGENT_CONNECTION_CONFIGS.")
            self.running_flag = False; return
            
        if self.active_agent_id not in self.mav_controllers:
            # If the initially selected active agent failed to connect, try to pick one that did
            if self.mav_controllers:
                self.active_agent_id = next(iter(self.mav_controllers))
                print(f"Warning: Default active agent '{INITIAL_ACTIVE_AGENT_ID}' failed to connect. Switched to '{self.active_agent_id}'.")
            else: # Should have been caught by the previous check, but as a safeguard
                print("Fatal Error: No MAVLink controllers are connected, cannot set an active agent.")
                self.running_flag = False; return


        try:
            self.tts_engine = pyttsx3.init() # Initialize the Text-to-Speech engine
            print("TTS (pyttsx3): Engine initialized.")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}") # Non-fatal, app can still run without TTS

    def _get_active_mav_controller(self):
        """Retrieves the MavlinkController instance for the currently active agent."""
        if not self.active_agent_id:
            print("Error: No active agent selected.")
            return None
        controller = self.mav_controllers.get(self.active_agent_id)
        if not controller:
            print(f"Error: Active agent '{self.active_agent_id}' not found in connected controllers.")
            return None
        if not controller.is_connected():
            print(f"Error: Active agent '{self.active_agent_id}' is not currently connected.")
            return None
        return controller

    def _setup_keyboard_listener(self):
        """Sets up the keyboard listener for Push-To-Talk (PTT) functionality."""
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_ptt_press,
            on_release=self._on_ptt_release
        )
        self.keyboard_listener.run() # This blocks and processes keyboard events in the main thread

    def _on_ptt_press(self, key):
        """Callback function for keyboard key press events (PTT activation)."""
        if key == keyboard.Key.space:
            if not self.is_space_currently_pressed:
                self.is_space_currently_pressed = True
                try:
                    # Trigger state machine events based on current state
                    if self.sm.current_state == self.sm.idle:
                        self.sm.event_ptt_start_command()
                    elif self.sm.current_state == self.sm.awaiting_yes_no:
                        self.sm.event_ptt_start_yes_no()
                except TransitionNotAllowed:
                    print(f"SM Warning: PTT press not allowed in state {self.sm.current_state.id}")
        elif key == keyboard.Key.esc:
            self.initiate_shutdown("ESC key pressed.")
            return False # Stop the keyboard listener

    def _on_ptt_release(self, key):
        """Callback function for keyboard key release events (PTT deactivation)."""
        if key == keyboard.Key.space:
            if self.is_space_currently_pressed:
                self.is_space_currently_pressed = False
                try:
                    # Trigger state machine events based on current state
                    if self.sm.current_state == self.sm.listening_command:
                        self.sm.event_ptt_stop_command()
                    elif self.sm.current_state == self.sm.listening_yes_no:
                        self.sm.event_ptt_stop_yes_no()
                except TransitionNotAllowed:
                    print(f"SM Warning: PTT release not allowed in state {self.sm.current_state.id}")

    def start_audio_recording(self):
        """Starts recording audio from the microphone."""
        if not self.running_flag:
            return
        print("Audio: Starting recording stream...")
        self.audio_frames = [] # Clear previous audio frames
        try:
            self.current_audio_stream = self.pyaudio_instance.open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                input=True, # Set as input stream
                frames_per_buffer=AUDIO_FRAMES_PER_BUFFER
            )
            self.current_audio_stream.start_stream()
        except Exception as e:
            print(f"PyAudio Error: Could not open stream: {e}")
            self.speak("Error: Could not access microphone.")
            self.current_audio_stream = None
            # If microphone access fails, transition the state machine accordingly
            if self.sm.current_state == self.sm.listening_command:
                self.sm.send("event_stt_failed_command")
            elif self.sm.current_state == self.sm.listening_yes_no:
                self.sm.send("event_stt_failed_yes_no")

    def stop_audio_recording(self):
        """Stops recording audio and returns the captured audio data as bytes."""
        if not self.current_audio_stream:
            return b''
        print("Audio: Stopping recording stream...")
        try:
            if self.current_audio_stream.is_active():
                self.current_audio_stream.stop_stream()
            self.current_audio_stream.close()
        except Exception as e:
            print(f"PyAudio Error: Could not properly close stream: {e}")
        self.current_audio_stream = None
        return b''.join(self.audio_frames) # Concatenate all recorded frames

    def run(self):
        """
        Starts the main application loop.
        This loop continuously reads audio if PTT is active and manages the application's lifecycle.
        """
        if not self.running_flag:
            print("Application cannot start due to initialization errors.")
            self.cleanup()
            return

        def main_loop():
            """Background thread for continuous audio reading when PTT is active."""
            while self.running_flag:
                # Check if PTT is pressed and audio stream is active for recording
                if self.is_space_currently_pressed and \
                   (self.sm.current_state == self.sm.listening_command or \
                    self.sm.current_state == self.sm.listening_yes_no) and \
                   self.current_audio_stream and not self.current_audio_stream.is_stopped():
                    try:
                        # Read audio data from the stream
                        data = self.current_audio_stream.read(AUDIO_READ_CHUNK, exception_on_overflow=False)
                        self.audio_frames.append(data)
                    except IOError as e:
                        print(f"Audio read error: {e}")
                        self.speak("Microphone error during recording.")
                        # If an audio error occurs, simulate PTT release to stop recording
                        if self.is_space_currently_pressed:
                            self.is_space_currently_pressed = False
                            if self.sm.current_state == self.sm.listening_command:
                                self.sm.event_ptt_stop_command()
                            elif self.sm.current_state == self.sm.listening_yes_no:
                                self.sm.event_ptt_stop_yes_no()
                time.sleep(0.01) # Small delay to prevent busy-waiting
            print("Exiting main application loop.")
            self.cleanup() # Perform cleanup when the loop exits

        # Start the main audio reading loop in a background thread
        t = threading.Thread(target=main_loop, daemon=True)
        t.start()

        self.speak("System initialized.")
        active_agent_display = f" (Active Agent: {self.active_agent_id})" if self.active_agent_id else ""
        print(f"\nSystem ready{active_agent_display}. Press and hold SPACE to speak, or ESC to exit.")
        # Ensure the state machine is in the idle state at startup
        if self.sm.current_state != self.sm.idle:
            try:
                self.sm.send("event_shutdown_requested") # Force transition to idle
            except Exception as e:
                print(f"Error forcing SM to idle: {e}")

        # Run the keyboard listener in the main thread (this call is blocking)
        self._setup_keyboard_listener()

        # After the keyboard listener exits (e.g., ESC key pressed), set running_flag to False
        # to signal the main_loop thread to stop.
        self.running_flag = False
        t.join() # Wait for the main loop thread to finish before exiting the application

    def speak(self, text_to_speak):
        """
        Uses the TTS engine to speak the given text.
        Handles running TTS in the main thread to avoid pyttsx3 issues.
        """
        def _do_tts(text):
            """Internal function to perform the TTS synthesis."""
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")

        if self.tts_engine and self.running_flag:
            print(f"TTS: Saying: '{text_to_speak}'")
            if threading.current_thread() is threading.main_thread():
                _do_tts(text_to_speak) # If already in main thread, execute directly
            else:
                # If not in main thread, schedule TTS on main thread using a queue
                import queue
                if not hasattr(self, '_tts_queue'):
                    self._tts_queue = queue.Queue()
                    # Start a dedicated thread to process TTS requests from the queue
                    def tts_main_loop():
                        while self.running_flag:
                            try:
                                text = self._tts_queue.get(timeout=0.5) # Get text from queue with timeout
                                _do_tts(text)
                            except queue.Empty:
                                continue # Continue if queue is empty
                    threading.Thread(target=tts_main_loop, daemon=True).start()
                self._tts_queue.put(text_to_speak) # Add text to the queue
        elif not self.tts_engine:
            print(f"TTS (Disabled): '{text_to_speak}'") # Log if TTS engine is not initialized

    def process_command_stt(self, audio_data_bytes):
        """
        Processes the captured audio for a command: performs STT, converts spoken numbers,
        and then performs NLU to get intent and entities.
        """
        if not audio_data_bytes: return None, False, False
        print("STT: Processing command...")
        transcribed_text = self.stt_system.recognize_buffer(audio_data_bytes) # Perform STT
        print(f"STT Result (Command): '{transcribed_text}'")
        if not transcribed_text: return None, False, False

        # Convert spoken numbers (e.g., "one hundred") to digits (e.g., "100")
        transcribed_text = spoken_numbers_to_digits(transcribed_text)
        # Normalize spaces (remove extra spaces)
        transcribed_text = re.sub(r'\s+', ' ', transcribed_text).strip()
        # Optionally, add a comma for easier GPS extraction if not already present
        transcribed_text = re.sub(
            r'latitude\s*(-?\d+\.\d+)\s+longitude\s*(-?\d+\.\d+)',
            r'latitude \1, longitude \2',
            transcribed_text
        )
        print(f"STT (after num conversion): '{transcribed_text}'")

        # Check for explicit shutdown commands
        if "stop listening" in transcribed_text.lower() or "shutdown system" in transcribed_text.lower():
            return transcribed_text, False, True # Return True for is_stop_command

        # Perform Natural Language Understanding
        self.original_nlu_result_for_execution = self.nlu_system.parse_command(transcribed_text)
        self.command_to_execute_display_text = self.original_nlu_result_for_execution.get("text", transcribed_text)
        
        needs_confirmation = True
        intent = self.original_nlu_result_for_execution.get("intent")
        confidence = self.original_nlu_result_for_execution.get("confidence", 0.0)
        # Commands like "select agent" might not need explicit confirmation if confidence is high
        if intent == "SELECT_AGENT" and confidence > 0.85:
            needs_confirmation = False
        return self.command_to_execute_display_text, needs_confirmation, False

    def process_yes_no_stt(self, audio_data_bytes):
        """
        Processes the captured audio for a simple "yes" or "no" response.
        """
        if not audio_data_bytes: return "unrecognized"
        print("STT: Processing yes/no...")
        text_result = self.stt_system.recognize_buffer(audio_data_bytes).lower() # Perform STT and convert to lowercase
        print(f"STT Result (Yes/No): '{text_result}'")
        # Simple keyword matching for yes/no
        if "yes" in text_result and "no" not in text_result: return "yes"
        if "no" in text_result and "yes" not in text_result: return "no"
        if "yeah" in text_result or "yep" in text_result: return "yes"
        if "nope" in text_result: return "no"
        return "unrecognized"

    def execute_lifeguard_command_wrapper(self):
        """
        Wrapper function for `execute_lifeguard_command` to handle exceptions
        and ensure the state machine transitions back to idle upon completion.
        This is designed to be run in a separate thread.
        """
        print("DEBUG: Entered execute_lifeguard_command_wrapper")
        try:
            self.execute_lifeguard_command() # Call the actual command execution logic
        except Exception as e:
            print(f"Error during command execution: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            self.speak("An error occurred while executing the command.")
        finally:
            # Ensure state machine transitions back to idle after action, regardless of success/failure
            if self.sm.current_state == self.sm.executing_action:
                self.sm.send("event_action_completed")

    def execute_lifeguard_command(self):
        """
        Executes the MAVLink command based on the NLU result.
        This is the core logic for interacting with the uncrewed vehicle.
        """
        print("DEBUG: Entered execute_lifeguard_command()")
        nlu_result = self.original_nlu_result_for_execution
        if not nlu_result:
            print("DEBUG: No NLU result, returning early")
            self.speak("Error: No command data to execute.")
            return

        intent = nlu_result.get("intent")
        entities = nlu_result.get("entities", {})
        confidence = nlu_result.get("confidence", 0.0)
        command_text_for_log = nlu_result.get("text", self.command_to_execute_display_text)

        print(f"DEBUG: intent={intent}, confidence={confidence}, entities={entities}")
        print(f"\nNLU Interpretation for Execution: Intent='{intent}', Confidence={confidence:.2f}, Text='{command_text_for_log}'")
        print(f"Entities: {json.dumps(entities, indent=2)}")

        active_mav_controller = self._get_active_mav_controller()
        if not active_mav_controller:
            print("DEBUG: No active MAV controller, returning early")
            self.speak(f"Cannot execute command: Active agent '{self.active_agent_id}' is not available or not connected.")
            return

        if intent == "UNKNOWN_INTENT" or confidence < NLU_CONFIDENCE_THRESHOLD:
            print("DEBUG: Intent unknown or confidence too low, returning early")
            self.speak("Command not understood clearly or intent is unknown. Action cancelled.")
            return

        # Handle SELECT_AGENT intent: change the active agent
        if intent == "SELECT_AGENT":
            new_agent_id = entities.get("selected_agent_id")
            if new_agent_id and new_agent_id in self.mav_controllers:
                if self.mav_controllers[new_agent_id].is_connected():
                    self.active_agent_id = new_agent_id
                    self.speak(f"Active agent set to {self.active_agent_id}.")
                else:
                    self.speak(f"Cannot select agent {new_agent_id}, it is not connected.")
            else:
                self.speak(f"Agent ID {new_agent_id} not recognized or not configured.")
            return # Action completed for SELECT_AGENT, no further MAVLink commands needed

        # Extract common parameters for MAVLink commands
        parsed_gps_coords = None
        if "latitude" in entities and "longitude" in entities:
            parsed_gps_coords = {"latitude": entities["latitude"], "longitude": entities["longitude"]}
        print(f"DEBUG: parsed_gps_coords={parsed_gps_coords}")

        target_desc = entities.get("target_description_full")

        # Handle REQUEST_GRID_SEARCH intent
        if intent == "REQUEST_GRID_SEARCH" and parsed_gps_coords and "grid_size_meters" in entities:
            print("DEBUG: Entering grid search block")
            lat = parsed_gps_coords["latitude"]
            lon = parsed_gps_coords["longitude"]
            grid_size_m = entities["grid_size_meters"]
            swath_width_m = entities.get("swath_width_meters", DEFAULT_SWATH_WIDTH_M) # Use default if not specified
            alt = DEFAULT_WAYPOINT_ALTITUDE # Use default altitude

            action_summary = f"Search a {grid_size_m} meter grid at Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt}m."
            if target_desc: action_summary += f" Looking for: {target_desc}."
            self.speak(f"Executing for {self.active_agent_id}: {action_summary}")

            # Generate and upload the search grid mission
            mission_uploaded = active_mav_controller.generate_and_upload_search_grid_mission(lat, lon, grid_size_m, swath_width_m, alt)
            if mission_uploaded:
                self.speak("Search grid mission uploaded. Attempting to arm and start.")
                # Sequence of operations: Arm vehicle -> Set mode to AUTO -> Start mission
                if active_mav_controller.arm_vehicle():
                    time.sleep(1) # Short delay after arming
                    if active_mav_controller.set_mode("AUTO"): # AUTO mode is required for mission execution
                        time.sleep(1)
                        if active_mav_controller.start_mission():
                            self.speak("Mission started. Vehicle is en route to search area.")
                            if target_desc:
                                self.speak("Waiting for vehicle to reach search area before sending target details.")
                                # Assuming the first waypoint of the grid mission is sequence 0 (the home waypoint)
                                # Wait for the vehicle to reach the first actual search waypoint (sequence 1)
                                if active_mav_controller.wait_for_waypoint_reached(1, timeout_seconds=180): # Wait up to 3 mins
                                    active_mav_controller.send_status_text(target_desc, severity=mavutil.mavlink.MAV_SEVERITY_INFO)
                                    self.speak("Target details sent to vehicle.")
                                else:
                                    self.speak("Failed to confirm vehicle reached search area in time. Target details not sent by STATUSTEXT.")
                        else: self.speak("Failed to start mission after arming and setting mode.")
                    else: self.speak("Failed to set mode to AUTO after arming.")
                else: self.speak("Failed to arm vehicle for mission.")
            else:
                self.speak("Failed to upload search grid mission.")

        # Handle simple fly to waypoint (can also include target description)
        elif intent in ("COMBINED_SEARCH_AND_TARGET", "REQUEST_SEARCH_AT_LOCATION") and parsed_gps_coords:
            print("DEBUG: Entering waypoint upload block")
            print("DEBUG: About to call upload_mission()")
            lat = parsed_gps_coords["latitude"]
            lon = parsed_gps_coords["longitude"]
            alt = DEFAULT_WAYPOINT_ALTITUDE # Use default altitude
            action_summary = f"Fly to Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt}m."
            if target_desc: action_summary += f" Search for: {target_desc}."
            self.speak(f"Executing for {self.active_agent_id}: {action_summary}")

            # Prepare single waypoint data for mission upload
            single_wp_data = [
                (lat, lon, alt,
                 mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                 0.0,  # param1 (hold time seconds)
                 10.0, # param2 (acceptance radius meters)
                 0.0,  # param3 (pass radius meters, 0 to pass through center)
                 float('nan') # param4 (desired yaw angle at waypoint, NaN for unchanged)
                )
            ]
            print("DEBUG: About to call upload_mission()")
            try:
                mission_uploaded = active_mav_controller.upload_mission(single_wp_data)
            except Exception as e:
                print(f"Exception during mission upload: {e}")
                mission_uploaded = False
            
            if mission_uploaded:
                self.speak("Waypoint mission uploaded. Attempting to set mode and arm.")
                # Set mode to AUTO for missions
                if active_mav_controller.set_mode("AUTO"):
                    time.sleep(1)
                    if active_mav_controller.arm_vehicle():
                        time.sleep(1)
                        if active_mav_controller.start_mission():
                            self.speak("Mission to waypoint started.")
                            if target_desc: # Send STATUSTEXT after reaching the waypoint
                                self.speak("Waiting for vehicle to reach waypoint before sending target details.")
                                # Wait for the vehicle to reach the first actual waypoint (sequence 1, after home)
                                if active_mav_controller.wait_for_waypoint_reached(1, timeout_seconds=180):
                                    active_mav_controller.send_status_text(target_desc, severity=mavutil.mavlink.MAV_SEVERITY_INFO)
                                    self.speak("Target details sent to vehicle.")
                                else:
                                    self.speak("Failed to confirm vehicle reached waypoint. Target details not sent.")
                        else:
                            self.speak("Failed to start mission.")
                    else:
                        self.speak("Failed to arm vehicle.")
                else:
                    self.speak("Failed to set mode.")
            else:
                self.speak("Failed to upload waypoint mission.")


        # Handle providing target description without movement command
        elif intent == "PROVIDE_TARGET_DESCRIPTION" and target_desc:
            print("DEBUG: Entering target description block")
            self.speak(f"Noted target description for {self.active_agent_id}: {target_desc}. Sending to vehicle.")
            if active_mav_controller.send_status_text(target_desc, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
                self.speak("Target information sent to vehicle.")
            else:
                self.speak("Failed to send target information to vehicle.")
            
        else:
            print("DEBUG: No matching intent block, returning early")
            self.speak("Command understood, but no specific MAVLink action defined for this combination or missing critical info (like GPS).")


    def initiate_shutdown(self, reason=""):
        """Initiates the application shutdown process."""
        if self.running_flag:
            print(f"Shutdown initiated: {reason}")
            self.running_flag = False # Set flag to stop main loops
            self.speak("System shutting down.")
            # The main loop in run() will detect running_flag change and call cleanup.

    def cleanup(self):
        """Cleans up all initialized resources (audio, MAVLink connections, threads)."""
        print("Cleaning up resources...")
        # Wait for action thread to finish if it's still running
        if self.action_thread and self.action_thread.is_alive():
            print("Waiting for action thread to finish...")
            self.action_thread.join(timeout=10) # Wait up to 10 seconds for the thread to complete
        
        # Stop keyboard listener if active
        if self.keyboard_listener and hasattr(self.keyboard_listener, 'is_alive') and self.keyboard_listener.is_alive():
            self.keyboard_listener.stop()
            # self.keyboard_listener.join() # Joining can block indefinitely if not stopped cleanly

        # Close current audio stream if open
        if self.current_audio_stream:
            try:
                if self.current_audio_stream.is_active(): self.current_audio_stream.stop_stream()
                self.current_audio_stream.close()
            except Exception as e: print(f"Error closing active audio stream: {e}")
        self.current_audio_stream = None

        # Terminate PyAudio instance
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            print("PyAudio terminated.")

        # Close all MAVLink connections
        for agent_id, controller in self.mav_controllers.items():
            if controller:
                print(f"Closing connection for agent {agent_id}...")
                controller.close_connection()
        self.mav_controllers.clear() # Clear the dictionary of controllers

        print("System shutdown complete.")


if __name__ == "__main__":
    app = LifeguardApp() # Create an instance of the main application
    try:
        if app.running_flag: # Only run if initialization was successful
            # Optional: Upload a test mission ONCE at startup for validation
            # controller = app._get_active_mav_controller()
            # if controller:
            #     print("DEBUG: Uploading test mission at startup")
            #     controller.set_mode("GUIDED")
            #     controller.upload_mission([
            #         (41.37, -72.09, 30.0,
            #          mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            #          0.0, 10.0, 0.0, float('nan')),
            #         (41.38, -72.10, 30.0,
            #          mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            #          0.0, 10.0, 0.0, float('nan'))
            #     ])
            # Now start the main voice-driven loop
            app.run()
        else:
            print("Application did not start due to initialization failures.")
    except KeyboardInterrupt:
        print("\nApplication interrupted by Ctrl+C.")
        app.initiate_shutdown("KeyboardInterrupt")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        app.initiate_shutdown(f"Unexpected error: {e}")
    finally:
        # Ensure cleanup is called even if an error occurs during initialization or main run
        if hasattr(app, 'running_flag') and app.running_flag:
            app.cleanup()
        elif not hasattr(app, 'running_flag'):
            # Special case for very early exit before `running_flag` is set or `cleanup` is fully initialized
            print("Cleanup called due to early exit or incomplete initialization.")
            if hasattr(app, 'pyaudio_instance') and app.pyaudio_instance: app.pyaudio_instance.terminate()
            if hasattr(app, 'mav_controllers'):
                for controller in app.mav_controllers.values():
                    if controller: controller.close_connection()
    print("Application has terminated.")
