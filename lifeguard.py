# lifeguard.py

import os
import json
import re
import time
import asyncio # Required for some Pymavlink operations if they become async in future versions
import threading
import math # For grid calculations

# Audio Processing
import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr

# Speech-to-Text (STT)
import vosk

# Natural Language Understanding (NLU)
import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import EntityRuler
from lat_lon_parser import parse as parse_lat_lon_external

# MAVLink Communication
from pymavlink import mavutil, mavwp

# PTT, TTS, State Machine
from pynput import keyboard
import pyttsx3
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed
from word2number import w2n
import re

# --- CONFIGURATION CONSTANTS ---

# STT Configuration
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15" # Name of the Vosk model folder
VOSK_MODEL_PATH = os.path.join("vosk_models", VOSK_MODEL_NAME)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_FRAMES_PER_BUFFER = 8192 # For PyAudio stream opening
AUDIO_READ_CHUNK = 4096 # Amount of data to read from stream at a time

# Audio Pre-processing Configuration
NOISE_REDUCE_TIME_CONSTANT_S = 2.0 # For noisereduce
NOISE_REDUCE_PROP_DECREASE = 0.9
BANDPASS_LOWCUT_HZ = 300.0
BANDPASS_HIGHCUT_HZ = 3400.0
BANDPASS_ORDER = 5

# NLU Configuration
SPACY_MODEL_NAME = "en_core_web_sm"
# Confidence threshold for NLU interpretation to trigger command execution
NLU_CONFIDENCE_THRESHOLD = 0.7 # Example value

# MAVLink Configuration
# MAVLINK_CONNECTION_STRING = 'udpin:localhost:14550' # FOR SIMULATION/TESTING - Replaced by AGENT_CONNECTION_CONFIGS
MAVLINK_BAUDRATE = 115200 # Only for serial connections
MAVLINK_SOURCE_SYSTEM_ID = 255 # GCS typically uses 255
DEFAULT_WAYPOINT_ALTITUDE = 30.0 # Meters
DEFAULT_SWATH_WIDTH_M = 20.0 # Meters, for search grid

# Agent Configuration (for single or multi-agent setup)
# For mavlink-router: each agent would be a different UDP endpoint provided by the router
# e.g., "drone1": {'connection_string': 'udpin:localhost:14551',...}
#       "drone2": {'connection_string': 'udpin:localhost:14552',...}
AGENT_CONNECTION_CONFIGS = {
    "agent1": { # Default single agent
        "connection_string": 'udpin:localhost:14550', # Replace with your actual connection
        "source_system_id": MAVLINK_SOURCE_SYSTEM_ID,
        "baudrate": MAVLINK_BAUDRATE # Relevant only for serial connections
    }
    # Add more agents here if using mavlink-router or multiple direct connections
    # "agent2": {
    #     "connection_string": 'udpin:localhost:14560', # Example for a second agent
    #     "source_system_id": MAVLINK_SOURCE_SYSTEM_ID,
    #     "baudrate": MAVLINK_BAUDRATE
    # }
}
INITIAL_ACTIVE_AGENT_ID = "agent1" # The agent to command by default

# Earth radius in meters for geo calculations
EARTH_RADIUS_METERS = 6378137.0

# --- AUDIO PRE-PROCESSING FUNCTIONS ---

def apply_noise_reduction(audio_data_int16, sample_rate):
    """Applies noise reduction to int16 audio data."""
    audio_data_float = audio_data_int16.astype(np.float32) / 32767.0
    reduced_audio_float = nr.reduce_noise(y=audio_data_float,
                                          sr=sample_rate,
                                          time_constant_s=NOISE_REDUCE_TIME_CONSTANT_S,
                                          prop_decrease=NOISE_REDUCE_PROP_DECREASE,
                                          n_fft=512)
    reduced_audio_int16 = (reduced_audio_float * 32767.0).astype(np.int16)
    return reduced_audio_int16

def apply_bandpass_filter(audio_data, sample_rate):
    """Applies a band-pass filter to the audio data."""
    nyq = 0.5 * sample_rate
    low = BANDPASS_LOWCUT_HZ / nyq
    high = BANDPASS_HIGHCUT_HZ / nyq
    b, a = butter(BANDPASS_ORDER, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio_data)
    return filtered_audio.astype(audio_data.dtype)

# Place this near your imports, outside the class
def spoken_numbers_to_digits(text):

    # Fix common STT misrecognitions
    text = re.sub(r'\bnative\b', 'negative', text, flags=re.IGNORECASE)
    # Replace 'oh' or 'o' with 'zero' when used as a number
    text = re.sub(r'\boh\b', 'zero', text, flags=re.IGNORECASE)
    text = re.sub(r'\bo\b', 'zero', text, flags=re.IGNORECASE)

    number_pattern = re.compile(
        r'\b(?P<sign>negative |minus )?(?P<number>(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
        r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|point|and| )+)\b',
        re.IGNORECASE
    )

    def repl(match):
        phrase = match.group('number')
        sign = match.group('sign')
        try:
            num = w2n.word_to_num(phrase.strip())
            if sign:
                num = -abs(num)
            return f" {num} "
        except Exception:
            return match.group(0)

    return number_pattern.sub(repl, text)

# --- SPEECH-TO-TEXT (STT) MODULE ---

class SpeechToText:
    def __init__(self, model_path, sample_rate):
        self.sample_rate = sample_rate
        if not os.path.exists(model_path):
            print(f"ERROR: Vosk model not found at {model_path}. Please download and place it correctly.")
            raise FileNotFoundError(f"Vosk model not found at {model_path}")

        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True) # Optional: for word-level timestamps

    def recognize_buffer(self, raw_audio_buffer_bytes):
        """Processes a complete audio buffer and returns final transcription."""
        if not raw_audio_buffer_bytes:
            self.recognizer.Reset()
            return ""

        audio_data_np = np.frombuffer(raw_audio_buffer_bytes, dtype=np.int16)
        denoised_audio_np = apply_noise_reduction(audio_data_np, self.sample_rate)
        filtered_audio_np = apply_bandpass_filter(denoised_audio_np, self.sample_rate)
        processed_audio_bytes = filtered_audio_np.tobytes()

        text_result = ""
        if self.recognizer.AcceptWaveform(processed_audio_bytes):
            result_json = self.recognizer.Result()
        else:
            result_json = self.recognizer.FinalResult()

        self.recognizer.Reset()

        try:
            result_dict = json.loads(result_json)
            text_result = result_dict.get('text', '')
        except json.JSONDecodeError:
            print(f"STT (Vosk): Could not decode JSON result: {result_json}")
        except Exception as e:
            print(f"STT (Vosk): Error processing result: {e}")
        return text_result

# --- NATURAL LANGUAGE UNDERSTANDING (NLU) MODULE ---
@Language.factory("lat_lon_entity_recognizer", default_config={"gps_label": "LOCATION_GPS_COMPLEX"})
def create_lat_lon_entity_recognizer(nlp: Language, name: str, gps_label: str):
    return LatLonEntityRecognizer(nlp, gps_label)

class LatLonEntityRecognizer:
    def __init__(self, nlp: Language, gps_label: str):
        self.gps_label = gps_label
        self.gps_regex = re.compile(
            r"""
            (?: # Non-capturing group for different formats
                # Match: latitude 41.37, longitude -72.09
                latitude\s*([+-]?\d{1,3}(?:\.\d+)?)\s*,\s*longitude\s*([+-]?\d{1,3}(?:\.\d+)?)
                | # OR: lat/lon or decimal pair
                (?:[Ll][Aa](?:itude)?\s*[:\s]*)? # Optional "lat" or "latitude"
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*?)\s*
                (?:[,;\s]+\s*)? # Separator
                (?:[Ll][Oo][Nn](?:gitude)?\s*[:\s]*)? # Optional "lon" or "longitude"
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*[EeWw]?)
                | # OR simple decimal pair
                ([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)
            )
            """,
            re.VERBOSE
        )
        if not Span.has_extension("parsed_gps_coords"):
            Span.set_extension("parsed_gps_coords", default=None)

    def __call__(self, doc: Doc) -> Doc:
        print(f"[DEBUG] LatLonEntityRecognizer called on: '{doc.text}'")
        new_entities = list(doc.ents)
        for match in self.gps_regex.finditer(doc.text):
            print(f"[DEBUG] GPS regex match: {match.group(0)}")
            start_char, end_char = match.span()
            lat_val, lon_val = None, None

            # Only create a GPS entity if BOTH latitude and longitude are present in the same match
            if match.group(1) and match.group(2):
                lat_val = float(match.group(1))
                lon_val = float(match.group(2))
            elif match.group(3) and match.group(4):
                try:
                    lat_val = float(match.group(3))
                    lon_val = float(match.group(4))
                except Exception:
                    continue
            elif match.group(5) and match.group(6):
                try:
                    lat_val = float(match.group(5))
                    lon_val = float(match.group(6))
                except Exception:
                    continue
            else:
                continue

            if lat_val is not None and lon_val is not None:
                if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                    try:
                        span = doc.char_span(start_char, end_char, label=self.gps_label, alignment_mode="expand")
                        if span is None:
                            # Try contract mode
                            contract_span = doc.char_span(start_char, end_char, alignment_mode="contract")
                            if contract_span is not None:
                                span = Span(doc, contract_span.start, contract_span.end, label=self.gps_label)
                            else:
                                # Fallback: use the closest tokens covering the match
                                token_start = None
                                token_end = None
                                for i, token in enumerate(doc):
                                    if token.idx <= start_char < token.idx + len(token):
                                        token_start = i
                                    if token.idx < end_char <= token.idx + len(token):
                                        token_end = i + 1
                                # If still not found, use the nearest tokens
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
                            span._.set("parsed_gps_coords", {"latitude": lat_val, "longitude": lon_val})
                            # Remove ALL overlapping entities (not just GPS) to avoid spaCy E1010 error
                            temp_new_entities = []
                            for ent in new_entities:
                                # Remove any entity that overlaps in tokens with the new span
                                if span.end <= ent.start or span.start >= ent.end:
                                    temp_new_entities.append(ent)
                            temp_new_entities.append(span)
                            new_entities = temp_new_entities
                    except Exception as e:
                        print(f"[DEBUG] Exception in LatLonEntityRecognizer: {e}")
        doc.ents = tuple(sorted(list(set(new_entities)), key=lambda e: e.start_char))
        return doc

class NaturalLanguageUnderstanding:
    def __init__(self, spacy_model_name):
        try:
            self.nlp = spacy.load(spacy_model_name)
        except OSError:
            print(f"spaCy model '{spacy_model_name}' not found. Please download it: python -m spacy download {spacy_model_name}")
            raise

        if "lat_lon_entity_recognizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("lat_lon_entity_recognizer", after="ner" if self.nlp.has_pipe("ner") else None)

        if "sar_entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", name="sar_entity_ruler", before="lat_lon_entity_recognizer")
            sar_patterns = [
                {"label": "TARGET_OBJECT", "pattern": [{"LOWER": "person"}]},
                {"label": "TARGET_OBJECT", "pattern": [{"LOWER": "boat"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}]},
                {"label": "AGENT_ID_NUM", "pattern": [{"LOWER": {"IN": ["drone", "agent"]}}, {"IS_DIGIT": True}]},
                {"label": "AGENT_ID_TEXT", "pattern": [{"LOWER": {"IN": ["drone", "agent"]}}, {"IS_ALPHA": True}]},
                {"label": "ACTION_SELECT", "pattern": [{"LEMMA": {"IN": ["select", "target", "choose", "activate"]}}]},
                {"label": "ACTION_VERB", "pattern": [{"LEMMA": "search"}]},  # <-- Add this line
            ]
            ruler.add_patterns(sar_patterns)

    def parse_command(self, text):
        doc = self.nlp(text)
        intent = "UNKNOWN_INTENT"
        entities_payload = {}
        confidence = 0.5

        extracted_spacy_ents = []
        for ent in doc.ents:
            entity_data = {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            if ent.label_ == "LOCATION_GPS_COMPLEX" and Span.has_extension("parsed_gps_coords") and ent._.get("parsed_gps_coords"):
                entity_data["parsed_gps"] = ent._.parsed_gps_coords
            elif ent.label_ == "GRID_SIZE_METERS":
                numeric_part = "".join(filter(str.isdigit, ent.text))
                if numeric_part:
                    entity_data["value"] = int(numeric_part)
            elif ent.label_ == "AGENT_ID_NUM":
                numeric_part = "".join(filter(str.isdigit, ent.text))
                entity_data["value"] = f"agent{numeric_part}"
            elif ent.label_ == "AGENT_ID_TEXT":
                name_part = ent.text.lower().replace("drone", "").replace("agent", "").strip()
                entity_data["value"] = name_part
            extracted_spacy_ents.append(entity_data)
        entities_payload["raw_spacy_entities"] = extracted_spacy_ents

        def get_entity_values(label, value_key="text"):
            return [e[value_key] for e in extracted_spacy_ents if e["label"] == label and value_key in e]

        def get_parsed_gps():
            for e in extracted_spacy_ents:
                if e["label"] == "LOCATION_GPS_COMPLEX" and "parsed_gps" in e:
                    return e["parsed_gps"]
            return None

        action_verbs = get_entity_values("ACTION_VERB")
        action_select_verbs = get_entity_values("ACTION_SELECT")
        target_objects = get_entity_values("TARGET_OBJECT")
        parsed_gps_coords = get_parsed_gps()
        grid_sizes = get_entity_values("GRID_SIZE_METERS", "value")
        agent_ids_num = get_entity_values("AGENT_ID_NUM", "value")
        agent_ids_text = get_entity_values("AGENT_ID_TEXT", "value")


        if parsed_gps_coords:
            entities_payload.update(parsed_gps_coords)

        if grid_sizes:
            entities_payload["grid_size_meters"] = grid_sizes[0] # Take first found

        selected_agent_id = None
        if agent_ids_num: selected_agent_id = agent_ids_num[0]
        elif agent_ids_text: selected_agent_id = agent_ids_text[0]

        if selected_agent_id and action_select_verbs:
            intent = "SELECT_AGENT"
            entities_payload["selected_agent_id"] = selected_agent_id
            confidence = 0.9

        target_details_parts = []
        if target_objects: target_details_parts.extend(target_objects)

        if target_details_parts:
            entities_payload["target_description_full"] = " ".join(target_details_parts)


        if intent == "UNKNOWN_INTENT":
            if action_verbs:
                entities_payload["action_verb"] = action_verbs[0]
                if parsed_gps_coords:
                    if "grid_size_meters" in entities_payload:
                        intent = "REQUEST_GRID_SEARCH"
                    elif target_details_parts:
                        intent = "COMBINED_SEARCH_AND_TARGET"
                    else:
                        intent = "REQUEST_SEARCH_AT_LOCATION"
                elif target_details_parts:
                    intent = "PROVIDE_TARGET_DESCRIPTION"
                else:
                    intent = "GENERIC_COMMAND"

        # Boost confidence for well-formed commands
        if intent not in ("UNKNOWN_INTENT",):
            if parsed_gps_coords or target_details_parts or intent == "SELECT_AGENT":
                confidence = max(confidence, 0.85)
            if intent == "REQUEST_GRID_SEARCH" and "grid_size_meters" in entities_payload and parsed_gps_coords:
                confidence = 0.95


        return {"text": text, "intent": intent, "confidence": confidence, "entities": entities_payload}


# --- MAVLINK INTEGRATION MODULE ---
class MavlinkController:
    def __init__(self, connection_string, baudrate=None, source_system_id=255):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.source_system_id = source_system_id
        self.master = None
        self._connect()

    def _connect(self):
        print(f"MAVLink: Attempting to connect to {self.connection_string}...")
        try:
            # For serial, baudrate is used. For UDP/TCP, it's often ignored by mavutil.
            if "com" in self.connection_string.lower() or "/dev/tty" in self.connection_string.lower():
                 self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate, source_system=self.source_system_id)
            else: # UDP/TCP
                 self.master = mavutil.mavlink_connection(self.connection_string, source_system=self.source_system_id)

            self.master.wait_heartbeat(timeout=10) # Increased timeout for initial connection
            print(f"MAVLink: Heartbeat received from system {self.master.target_system} on {self.connection_string}. Connection established.")
            print(f"MAVLink: Target system: {self.master.target_system}, component: {self.master.target_component}")
        except Exception as e:
            print(f"MAVLink: Failed to connect or receive heartbeat on {self.connection_string}: {e}")
            self.master = None

    def is_connected(self):
        # Optionally, also check if the last heartbeat was recent
        return self.master is not None

    def send_nav_waypoint(self, lat, lon, alt, hold_time=0, accept_radius=10, pass_radius=0, yaw_angle=float('nan')):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot send waypoint.")
            return False
        # Simplified: send as MISSION_ITEM_INT for guided mode style "goto"
        # For full mission, use upload_mission
        print(f"MAVLink: Sending MAV_CMD_NAV_WAYPOINT (as MISSION_ITEM_INT) to Lat: {lat}, Lon: {lon}, Alt: {alt}")
        try:
            self.master.mav.mission_item_int_send(
                self.master.target_system, self.master.target_component,
                0, # seq
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                2, # current (2 for guided mode type command, 0 for mission item part of sequence)
                1, # autocontinue
                float(hold_time), float(accept_radius), float(pass_radius), float(yaw_angle),
                int(lat * 1e7), int(lon * 1e7), float(alt)
            )
            # Note: MISSION_ITEM_INT sent this way might not get a MISSION_ACK.
            # It's more like a "guided mode go to". For missions, use upload_mission.
            # For simplicity in execute_lifeguard_command, this might be used for single point nav.
            print("MAVLink: MAV_CMD_NAV_WAYPOINT (as MISSION_ITEM_INT) sent.")
            return True
        except Exception as e:
            print(f"MAVLink: Error sending MAV_CMD_NAV_WAYPOINT (as MISSION_ITEM_INT): {e}")
            return False

    def upload_mission(self, waypoints_data):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot upload mission.")
            return False

        wp_loader = mavwp.MAVWPLoader()
        for seq, wp_item_tuple in enumerate(waypoints_data):
            # Expected tuple: (lat, lon, alt, command_id, p1, p2, p3, p4, frame, current, autocontinue)
            # Or simplified: (lat, lon, alt, command_id, p1, p2, p3, p4) and use defaults for others
            if len(wp_item_tuple) == 8: # lat, lon, alt, cmd, p1, p2, p3, p4
                lat, lon, alt, command_id, p1, p2, p3, p4 = wp_item_tuple
                frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
                is_current = 0 # For mission items, usually 0. First item can be 1 if it's the start.
                autocontinue = 1
            elif len(wp_item_tuple) == 11: # Full spec
                lat, lon, alt, command_id, p1, p2, p3, p4, frame, is_current, autocontinue = wp_item_tuple
            else:
                print(f"MAVLink: Waypoint item {seq} has incorrect format. Skipping.")
                continue

            mission_item = mavutil.mavlink.MAVLink_mission_item_int_message(
                self.master.target_system, self.master.target_component, seq, frame, command_id,
                is_current, autocontinue, p1, p2, p3, p4,
                int(lat * 1e7), int(lon * 1e7), float(alt)
            )
            wp_loader.add(mission_item)

        if wp_loader.count() == 0:
            print("MAVLink: No valid waypoints to upload.")
            return False

        print(f"MAVLink: Uploading {wp_loader.count()} waypoints...")
        try:
            self.master.mav.mission_clear_all_send(self.master.target_system, self.master.target_component)
            ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            if not ack or ack.type!= mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print(f"MAVLink: Failed to clear mission or no/bad ACK. ACK: {ack}")
                # return False # Some autopilots might not send ACK for clear_all if no mission existed

            self.master.mav.mission_count_send(self.master.target_system, self.master.target_component, wp_loader.count())

            for i in range(wp_loader.count()):
                msg = self.master.recv_match(type='MISSION_REQUEST_INT', blocking=True, timeout=5)
                if not msg:
                    print(f"MAVLink: No MISSION_REQUEST received for waypoint {i}, upload failed.")
                    return False
                print(f"MAVLink: Sending waypoint {msg.seq}")
                self.master.mav.send(wp_loader.wp(msg.seq))
                if msg.seq == wp_loader.count() - 1:
                    break # Sent all waypoints

            final_ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=10) # Longer timeout for final ack
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
        Calculates waypoints for a rectangular lawnmower pattern.
        Orientation 0 means N-S tracks, 90 means E-W tracks.
        For simplicity, this example implements N-S tracks (orientation_deg = 0).
        Returns a list of (lat, lon, alt) tuples.
        """
        waypoints = []
        if swath_width_m <= 0 or grid_width_m <= 0 or grid_height_m <= 0:
            print("MAVLink: Invalid grid parameters for calculation.")
            return waypoints

        # Convert center lat/lon to radians
        center_lat_rad = math.radians(center_lat)
        center_lon_rad = math.radians(center_lon)

        # For N-S tracks (orientation 0), grid_width is along longitude, grid_height is along latitude
        # For E-W tracks (orientation 90), grid_width is along latitude, grid_height is along longitude
        # This simplified version assumes orientation_deg = 0 (N-S tracks)

        # Half dimensions
        half_width_m = grid_width_m / 2.0
        half_height_m = grid_height_m / 2.0

        # Calculate deltas in lat/lon for grid corners from center using simple equirectangular projection
        # More accurate methods would use Vincenty or Haversine for destination point given start, bearing, distance
        # dLat = dy / R_earth
        # dLon = dx / (R_earth * cos(center_lat_rad))
        
        # Start point of the grid (e.g., South-West corner for N-S tracks starting from West)
        # Calculate SW corner relative to center
        # Bearing to West is 270 deg, to South is 180 deg.
        # For simplicity, let's define the grid boundaries first.
        
        # Latitude boundaries
        lat_delta_rad = half_height_m / EARTH_RADIUS_METERS
        south_lat_rad = center_lat_rad - lat_delta_rad
        north_lat_rad = center_lat_rad + lat_delta_rad

        # Longitude boundaries
        lon_delta_rad = half_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))
        west_lon_rad = center_lon_rad - lon_delta_rad
        # east_lon_rad = center_lon_rad + lon_delta_rad # Not directly used if stepping by swath

        num_tracks = int(math.floor(grid_width_m / swath_width_m))
        if num_tracks == 0 and grid_width_m > 0: num_tracks = 1
        if num_tracks == 0: return waypoints

        # Calculate swath width in radians of longitude
        swath_lon_rad_step = swath_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))

        for i in range(num_tracks):
            current_track_lon_rad = west_lon_rad + i * swath_lon_rad_step
            # To make it perfectly centered, the first track could be at west_lon_rad + swath_lon_rad_step / 2
            # For now, starting at the western edge of the first swath.

            current_track_lon_deg = math.degrees(current_track_lon_rad)
            south_lat_deg = math.degrees(south_lat_rad)
            north_lat_deg = math.degrees(north_lat_rad)

            if i % 2 == 0:  # Northbound leg (starting from South)
                waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
                waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
            else:  # Southbound leg (starting from North)
                waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
                waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
        
        print(f"MAVLink: Calculated {len(waypoints)} waypoints for grid.")
        return waypoints

    def generate_and_upload_search_grid_mission(self, center_lat, center_lon, grid_size_m, swath_width_m, altitude_m):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot generate/upload search grid.")
            return False

        print(f"MAVLink: Generating search grid: Center=({center_lat:.6f}, {center_lon:.6f}), Size={grid_size_m}m, Swath={swath_width_m}m, Alt={altitude_m}m")

        grid_waypoints_tuples = self._calculate_rectangular_grid_waypoints(
            center_lat, center_lon, grid_size_m, grid_size_m, swath_width_m, altitude_m
        )

        if not grid_waypoints_tuples:
            print("MAVLink: Failed to generate grid waypoints.")
            return False

        waypoints_data_for_upload = []
        for lat, lon, alt in grid_waypoints_tuples:
            waypoints_data_for_upload.append(
                (lat, lon, alt,
                 mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                 0.0,  # param1 (hold time seconds)
                 10.0, # param2 (acceptance radius meters)
                 0.0,  # param3 (pass radius meters, 0 to pass through center)
                 float('nan') # param4 (desired yaw angle at waypoint, NaN for unchanged)
                )
            )
        
        return self.upload_mission(waypoints_data_for_upload)

    def send_status_text(self, text_to_send, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot send STATUSTEXT.")
            return False

        if isinstance(text_to_send, str):
            text_bytes = text_to_send.encode('utf-8')
        else: # Assume already bytes
            text_bytes = text_to_send

        # STATUSTEXT has a 50-byte limit for text (49 chars + NUL)
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
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot wait for waypoint.")
            return False
            
        print(f"MAVLink: Waiting for vehicle to reach mission waypoint sequence ID {waypoint_sequence_id}...")
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Use blocking with a short timeout to allow other processing if this were in a thread
            msg = self.master.recv_match(type='MISSION_ITEM_REACHED', blocking=True, timeout=1)
            if msg:
                print(f"MAVLink: Received MISSION_ITEM_REACHED with sequence {msg.seq}")
                if msg.seq == waypoint_sequence_id:
                    print(f"MAVLink: Vehicle has reached target waypoint {waypoint_sequence_id}.")
                    return True
            # Add a small delay to prevent tight loop if not using blocking or if timeout is very short
            # time.sleep(0.05) # Not strictly needed with blocking=True and timeout=1
        print(f"MAVLink: Timeout waiting for waypoint {waypoint_sequence_id} to be reached.")
        return False

    def set_mode(self, mode_name):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot set mode.")
            return False
        
        mode_mapping = self.master.mode_mapping()
        if mode_mapping is None:
            print(f"MAVLink: Mode mapping not available for {self.connection_string}. Cannot set mode.")
            # Fallback for some custom firmwares or if heartbeat didn't fully populate mode_mapping
            # This is a basic attempt, might not work for all autopilots.
            # ArduPilot custom modes: https://ardupilot.org/copter/docs/parameters.html#fltmode1
            # PX4 custom modes: https://docs.px4.io/main/en/flight_modes/
            # This requires knowing the specific mode IDs.
            # For ArduCopter: STABILIZE=0, ACRO=1, ALT_HOLD=2, AUTO=3, GUIDED=4, LOITER=5, RTL=6, LAND=9
            # For ArduPlane: MANUAL=0, CIRCLE=1, STABILIZE=2, FBWA=5, FBWB=6, AUTO=10, RTL=11, LOITER=12, GUIDED=15
            # This is highly autopilot-specific if mode_mapping() fails.
            # For now, we'll rely on mode_mapping() being present.
            print("MAVLink: Attempting to set mode by name directly (may not be reliable).")
            try:
                mode_id = self.master.mode_mapping()[mode_name.upper()]
            except KeyError:
                 print(f"MAVLink: Unknown mode: {mode_name}. Available modes from mapping: {list(mode_mapping.keys()) if mode_mapping else 'None'}")
                 return False
            except TypeError: # mode_mapping is None
                 print(f"MAVLink: Mode mapping is None. Cannot set mode {mode_name}.")
                 return False

        else: # mode_mapping is available
            if mode_name.upper() not in mode_mapping:
                print(f"MAVLink: Unknown mode: {mode_name}")
                print(f"MAVLink: Available modes: {list(mode_mapping.keys())}")
                return False
            mode_id = mode_mapping[mode_name.upper()]

        print(f"MAVLink: Setting mode to {mode_name.upper()} (ID: {mode_id})...")
        # master.set_mode(mode_id) # pymavlink set_mode is a higher-level helper
        try:
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, # base_mode
                mode_id # custom_mode
            )
            ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print(f"MAVLink: Mode change to {mode_name.upper()} successful.")
                    return True
                else:
                    print(f"MAVLink: Mode change to {mode_name.upper()} failed with result: {ack_msg.result}")
                    return False
            else:
                print(f"MAVLink: No/unexpected COMMAND_ACK received for mode change. ACK: {ack_msg}")
                return False
        except Exception as e:
            print(f"MAVLink: Error setting mode: {e}")
            return False


    def start_mission(self):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot start mission.")
            return False
        print("MAVLink: Sending MAV_CMD_MISSION_START...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0, # confirmation
            0, # param1 (first_item)
            0, # param2 (last_item, 0 for all)
            0, 0, 0, 0, 0 # params 3-7
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_MISSION_START and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Mission start command accepted.")
            return True
        else:
            print(f"MAVLink: Mission start command failed or no/bad ACK. ACK: {ack_msg}")
            return False

    def arm_vehicle(self):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot arm.")
            return False
        print("MAVLink: Attempting to arm vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, # confirmation
            1, # param1 (1 to arm, 0 to disarm)
            0, 0, 0, 0, 0, 0 # params 2-7
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5) # Longer timeout for arming
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Vehicle armed successfully.")
            return True
        else:
            # Common arming failure reasons: MAV_RESULT_TEMPORARILY_REJECTED (4), MAV_RESULT_FAILED (5)
            # These often correspond to safety checks failing (e.g., not GPS lock, pre-arm checks failed)
            res_text = f"result code {ack_msg.result}" if ack_msg else "no ACK"
            print(f"MAVLink: Arming failed ({res_text}). Ensure pre-arm checks are passing. ACK: {ack_msg}")
            return False

    def disarm_vehicle(self):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot disarm.")
            return False
        print("MAVLink: Attempting to disarm vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Vehicle disarmed successfully.")
            return True
        else:
            print(f"MAVLink: Disarming failed or no ACK received. ACK: {ack_msg}")
            return False

    def close_connection(self):
        if self.master:
            self.master.close()
            print(f"MAVLink: Connection closed for {self.connection_string}.")


# --- STATE MACHINE DEFINITION ---
class LifeguardAppSM(StateMachine):
    idle = State(initial=True)
    listening_command = State()
    processing_command = State()
    speaking_confirmation = State()
    awaiting_yes_no = State()
    listening_yes_no = State()
    processing_yes_no = State()
    executing_action = State()

    event_ptt_start_command = idle.to(listening_command)
    event_ptt_stop_command = listening_command.to(processing_command)
    event_command_recognized = processing_command.to(speaking_confirmation, cond="confirmation_needed_flag") | \
                               processing_command.to(executing_action, unless="confirmation_needed_flag")
    event_stt_failed_command = processing_command.to(idle)
    event_confirmation_spoken = speaking_confirmation.to(awaiting_yes_no)
    event_ptt_start_yes_no = awaiting_yes_no.to(listening_yes_no)
    event_ptt_stop_yes_no = listening_yes_no.to(processing_yes_no)
    event_yes_confirmed = processing_yes_no.to(executing_action)
    event_no_confirmed = processing_yes_no.to(idle)
    event_stt_failed_yes_no = processing_yes_no.to(awaiting_yes_no) # Re-prompt
    event_action_completed = executing_action.to(idle)
    event_shutdown_requested = idle.to(idle) | listening_command.to(idle) | \
                               processing_command.to(idle) | speaking_confirmation.to(idle) | \
                               awaiting_yes_no.to(idle) | listening_yes_no.to(idle) | \
                               processing_yes_no.to(idle) | executing_action.to(idle)


    def __init__(self, model_ref):
        self.app = model_ref
        self.captured_command_audio_data = None
        self.captured_yes_no_audio_data = None
        self._confirmation_needed_flag_internal = False
        super().__init__()

    def confirmation_needed_flag(self, event_data):
        return self._confirmation_needed_flag_internal

    def on_enter_listening_command(self, event_data):
        print("SM: State -> LISTENING_COMMAND")
        self.app.speak("Listening for command...")
        self.app.start_audio_recording()

    def on_exit_listening_command(self, event_data):
        print("SM: Exiting LISTENING_COMMAND")
        self.captured_command_audio_data = self.app.stop_audio_recording()

    def on_enter_processing_command(self, event_data):
        print("SM: State -> PROCESSING_COMMAND")
        command_text, needs_conf, is_stop_command = self.app.process_command_stt(self.captured_command_audio_data)
        if is_stop_command:
            self.app.initiate_shutdown("Stop command recognized.")
            self.send("event_shutdown_requested")
            return

        if command_text:
            self._confirmation_needed_flag_internal = needs_conf
            self.send("event_command_recognized")
        else:
            self.app.speak("Sorry, I could not understand the command. Please try again.")
            self.send("event_stt_failed_command")

    def on_enter_speaking_confirmation(self, event_data):
        print("SM: State -> SPEAKING_CONFIRMATION")
        prompt = f"Did you say: {self.app.command_to_execute_display_text}? Please press space and say yes or no."
        self.app.speak(prompt)
        self.send("event_confirmation_spoken")

    def on_enter_awaiting_yes_no(self, event_data):
        print("SM: State -> AWAITING_YES_NO")

    def on_enter_listening_yes_no(self, event_data):
        print("SM: State -> LISTENING_YES_NO")
        self.app.speak("Listening for yes or no...")
        self.app.start_audio_recording()

    def on_exit_listening_yes_no(self, event_data):
        print("SM: Exiting LISTENING_YES_NO")
        self.captured_yes_no_audio_data = self.app.stop_audio_recording()

    def on_enter_processing_yes_no(self, event_data):
        print("SM: State -> PROCESSING_YES_NO")
        response = self.app.process_yes_no_stt(self.captured_yes_no_audio_data)
        if response == "yes":
            self.send("event_yes_confirmed")
        elif response == "no":
            self.app.speak("Okay, command cancelled.")
            self.send("event_no_confirmed")
        else:
            self.app.speak("Sorry, I didn't catch that. Please try saying yes or no again.")
            self.send("event_stt_failed_yes_no")

    def on_enter_executing_action(self, event_data):
        print("SM: State -> EXECUTING_ACTION")
        # Run execution in a separate thread to keep SM responsive if action is long
        action_thread = threading.Thread(target=self.app.execute_lifeguard_command_wrapper)
        action_thread.daemon = True # Ensure thread doesn't block app exit
        action_thread.start()
        # The execute_lifeguard_command_wrapper will call self.send("event_action_completed")

    def on_enter_idle(self, event_data):
        source_id = event_data.source.id if event_data.source else "INITIAL"
        event_name = event_data.event if event_data.event else "INIT"
        print(f"SM: State -> IDLE (from {source_id} via {event_name})")
        self.app.command_to_execute_display_text = None
        self.app.original_nlu_result_for_execution = None
        self._confirmation_needed_flag_internal = False
        if self.app.running_flag:
            active_agent_display = f" (Active Agent: {self.app.active_agent_id})" if self.app.active_agent_id else ""
            print(f"\nSystem ready{active_agent_display}. Press and hold SPACE to speak, or ESC to exit.")


# --- MAIN APPLICATION CLASS ---
class LifeguardApp:
    def __init__(self):
        print("Initializing Commander Intent System with PTT...")
        self.running_flag = True
        self.pyaudio_instance = None
        self.stt_system = None
        self.nlu_system = None
        
        self.mav_controllers = {} # Dict to store {'agent_id': MavlinkController_instance}
        self.active_agent_id = INITIAL_ACTIVE_AGENT_ID
        if not AGENT_CONNECTION_CONFIGS:
            print("FATAL ERROR: No agents defined in AGENT_CONNECTION_CONFIGS. Exiting.")
            self.running_flag = False
            return
        if self.active_agent_id not in AGENT_CONNECTION_CONFIGS:
            # Fallback to the first agent in the config if initial active ID is invalid
            self.active_agent_id = next(iter(AGENT_CONNECTION_CONFIGS))
            print(f"Warning: Initial active agent ID '{INITIAL_ACTIVE_AGENT_ID}' not found. Defaulting to '{self.active_agent_id}'.")


        self.tts_engine = None
        self.keyboard_listener = None
        self.sm = None

        self.audio_frames = []
        self.is_space_currently_pressed = False
        self.current_audio_stream = None

        self.command_to_execute_display_text = None
        self.original_nlu_result_for_execution = None

        self._initialize_components()
        if self.running_flag: # Only init SM if components are okay
            self.sm = LifeguardAppSM(model_ref=self)

    def _initialize_components(self):
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
            # Baudrate is primarily for serial. For UDP/TCP, mavutil often infers or ignores it.
            baud = config.get("baudrate") if conn_str.startswith(("/dev/", "COM", "com")) else None
            src_sys_id = config.get("source_system_id", MAVLINK_SOURCE_SYSTEM_ID)
            
            print(f"Initializing MAVLink controller for agent: {agent_id} on {conn_str}")
            controller = MavlinkController(connection_string=conn_str, baudrate=baud, source_system_id=src_sys_id)
            if controller.is_connected():
                self.mav_controllers[agent_id] = controller
                print(f"MAVLink controller for agent {agent_id} connected successfully.")
            else:
                print(f"Warning: Failed to connect MAVLink controller for agent {agent_id} on {conn_str}.")
                # Decide if this is fatal or if the app can run with some agents offline
                # For now, we'll allow it but it won't be selectable.

        if not self.mav_controllers: # No controllers connected successfully
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
            self.tts_engine = pyttsx3.init()
            print("TTS (pyttsx3): Engine initialized.")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}") # Non-fatal

    def _get_active_mav_controller(self):
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
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_ptt_press,
            on_release=self._on_ptt_release
        )
        self.keyboard_listener.run()  # This blocks and processes events in the main thread

    def _on_ptt_press(self, key):
        if key == keyboard.Key.space:
            if not self.is_space_currently_pressed:
                self.is_space_currently_pressed = True
                try:
                    if self.sm.current_state == self.sm.idle:
                        self.sm.event_ptt_start_command()
                    elif self.sm.current_state == self.sm.awaiting_yes_no:
                        self.sm.event_ptt_start_yes_no()
                except TransitionNotAllowed:
                    print(f"SM Warning: PTT press not allowed in state {self.sm.current_state.id}")
        elif key == keyboard.Key.esc:
            self.initiate_shutdown("ESC key pressed.")
            return False # Stop listener

    def _on_ptt_release(self, key):
        if key == keyboard.Key.space:
            if self.is_space_currently_pressed:
                self.is_space_currently_pressed = False
                try:
                    if self.sm.current_state == self.sm.listening_command:
                        self.sm.event_ptt_stop_command()
                    elif self.sm.current_state == self.sm.listening_yes_no:
                        self.sm.event_ptt_stop_yes_no()
                except TransitionNotAllowed:
                    print(f"SM Warning: PTT release not allowed in state {self.sm.current_state.id}")

    def start_audio_recording(self):
        if not self.running_flag:
            return
        print("Audio: Starting recording stream...")
        self.audio_frames = []
        try:
            self.current_audio_stream = self.pyaudio_instance.open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_FRAMES_PER_BUFFER
            )
            self.current_audio_stream.start_stream()
        except Exception as e:
            print(f"PyAudio Error: Could not open stream: {e}")
            self.speak("Error: Could not access microphone.")
            self.current_audio_stream = None
            if self.sm.current_state == self.sm.listening_command:
                self.sm.send("event_stt_failed_command")
            elif self.sm.current_state == self.sm.listening_yes_no:
                self.sm.send("event_stt_failed_yes_no")

    def stop_audio_recording(self):
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
        return b''.join(self.audio_frames)

    def run(self):
        if not self.running_flag:
            print("Application cannot start due to initialization errors.")
            self.cleanup()
            return

        def main_loop():
            while self.running_flag:
                if self.is_space_currently_pressed and \
                   (self.sm.current_state == self.sm.listening_command or \
                    self.sm.current_state == self.sm.listening_yes_no) and \
                   self.current_audio_stream and not self.current_audio_stream.is_stopped():
                    try:
                        data = self.current_audio_stream.read(AUDIO_READ_CHUNK, exception_on_overflow=False)
                        self.audio_frames.append(data)
                    except IOError as e:
                        print(f"Audio read error: {e}")
                        self.speak("Microphone error during recording.")
                        if self.is_space_currently_pressed:
                            self.is_space_currently_pressed = False
                            if self.sm.current_state == self.sm.listening_command:
                                self.sm.event_ptt_stop_command()
                            elif self.sm.current_state == self.sm.listening_yes_no:
                                self.sm.event_ptt_stop_yes_no()
                time.sleep(0.01)
            print("Exiting main application loop.")
            self.cleanup()

        # Start main loop in a background thread
        t = threading.Thread(target=main_loop, daemon=True)
        t.start()

        self.speak("System initialized.")
        active_agent_display = f" (Active Agent: {self.active_agent_id})" if self.active_agent_id else ""
        print(f"\nSystem ready{active_agent_display}. Press and hold SPACE to speak, or ESC to exit.")
        if self.sm.current_state != self.sm.idle:
            try:
                self.sm.send("event_shutdown_requested")
            except Exception as e:
                print(f"Error forcing SM to idle: {e}")

        # Run the keyboard listener in the main thread (blocking)
        self._setup_keyboard_listener()

        # After the keyboard listener exits (ESC pressed), set running_flag to False to stop main_loop
        self.running_flag = False
        # Wait for the main loop thread to finish before exiting
        t.join()

    def speak(self, text_to_speak):
        if self.tts_engine and self.running_flag:
            print(f"TTS: Saying: '{text_to_speak}'")
            try:
                self.tts_engine.say(text_to_speak)
                self.tts_engine.runAndWait()
            except Exception as e: print(f"TTS Error: {e}")
        elif not self.tts_engine:
            print(f"TTS (Disabled): '{text_to_speak}'")

    # In your process_command_stt method, before NLU:
    def process_command_stt(self, audio_data_bytes):
        if not audio_data_bytes: return None, False, False
        print("STT: Processing command...")
        transcribed_text = self.stt_system.recognize_buffer(audio_data_bytes)
        print(f"STT Result (Command): '{transcribed_text}'")
        if not transcribed_text: return None, False, False

        # Convert spoken numbers to digits for better NLU
        transcribed_text = spoken_numbers_to_digits(transcribed_text)
        # Normalize spaces
        transcribed_text = re.sub(r'\s+', ' ', transcribed_text).strip()
        # Optionally, add a comma for easier GPS extraction
        transcribed_text = re.sub(
            r'latitude\s*(-?\d+\.\d+)\s+longitude\s*(-?\d+\.\d+)',
            r'latitude \1, longitude \2',
            transcribed_text
        )
        print(f"STT (after num conversion): '{transcribed_text}'")

        if "stop listening" in transcribed_text.lower() or "shutdown system" in transcribed_text.lower():
            return transcribed_text, False, True

        self.original_nlu_result_for_execution = self.nlu_system.parse_command(transcribed_text)
        self.command_to_execute_display_text = self.original_nlu_result_for_execution.get("text", transcribed_text)
        needs_confirmation = True
        intent = self.original_nlu_result_for_execution.get("intent")
        confidence = self.original_nlu_result_for_execution.get("confidence", 0.0)
        if intent == "SELECT_AGENT" and confidence > 0.85:
            needs_confirmation = False
        return self.command_to_execute_display_text, needs_confirmation, False

    def process_yes_no_stt(self, audio_data_bytes):
        if not audio_data_bytes: return "unrecognized"
        print("STT: Processing yes/no...")
        text_result = self.stt_system.recognize_buffer(audio_data_bytes).lower()
        print(f"STT Result (Yes/No): '{text_result}'")
        if "yes" in text_result and "no" not in text_result: return "yes"
        if "no" in text_result and "yes" not in text_result: return "no"
        if "yeah" in text_result or "yep" in text_result: return "yes"
        if "nope" in text_result: return "no"
        return "unrecognized"

    def execute_lifeguard_command_wrapper(self):
        """Wrapper to call actual execution and then signal SM completion."""
        try:
            self.execute_lifeguard_command()
        except Exception as e:
            print(f"Error during command execution: {e}")
            self.speak("An error occurred while executing the command.")
        finally:
            # Ensure SM transitions back to idle
            if self.sm.current_state == self.sm.executing_action:
                 self.sm.send("event_action_completed")


    def execute_lifeguard_command(self):
        nlu_result = self.original_nlu_result_for_execution
        if not nlu_result:
            self.speak("Error: No command data to execute.")
            return

        intent = nlu_result.get("intent")
        entities = nlu_result.get("entities", {})
        confidence = nlu_result.get("confidence", 0.0)
        command_text_for_log = nlu_result.get("text", self.command_to_execute_display_text)

        print(f"\nNLU Interpretation for Execution: Intent='{intent}', Confidence={confidence:.2f}, Text='{command_text_for_log}'")
        print(f"Entities: {json.dumps(entities, indent=2)}")

        active_mav_controller = self._get_active_mav_controller()
        if not active_mav_controller:
            self.speak(f"Cannot execute command: Active agent '{self.active_agent_id}' is not available or not connected.")
            return

        if intent == "UNKNOWN_INTENT" or confidence < NLU_CONFIDENCE_THRESHOLD:
            self.speak("Command not understood clearly or intent is unknown. Action cancelled.")
            return

        # Handle SELECT_AGENT intent
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
            return # Action completed for SELECT_AGENT

        parsed_gps_coords = None
        if "latitude" in entities and "longitude" in entities:
            parsed_gps_coords = {"latitude": entities["latitude"], "longitude": entities["longitude"]}
        
        target_desc = entities.get("target_description_full")

        # Handle Grid Search
        if intent == "REQUEST_GRID_SEARCH" and parsed_gps_coords and "grid_size_meters" in entities:
            lat = parsed_gps_coords["latitude"]
            lon = parsed_gps_coords["longitude"]
            grid_size_m = entities["grid_size_meters"]
            swath_width_m = entities.get("swath_width_meters", DEFAULT_SWATH_WIDTH_M) # Allow NLU override later
            alt = DEFAULT_WAYPOINT_ALTITUDE

            action_summary = f"Search a {grid_size_m} meter grid at Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt}m."
            if target_desc: action_summary += f" Looking for: {target_desc}."
            self.speak(f"Executing for {self.active_agent_id}: {action_summary}")

            mission_uploaded = active_mav_controller.generate_and_upload_search_grid_mission(lat, lon, grid_size_m, swath_width_m, alt)
            if mission_uploaded:
                self.speak("Search grid mission uploaded. Attempting to arm and start.")
                # Sequence: Arm -> Set Mode AUTO -> Start Mission
                if active_mav_controller.arm_vehicle():
                    time.sleep(1) # Short delay after arming
                    if active_mav_controller.set_mode("AUTO"): # Or GUIDED if appropriate for mission start
                        time.sleep(1)
                        if active_mav_controller.start_mission():
                            self.speak("Mission started. Vehicle is en route to search area.")
                            # Send STATUSTEXT after vehicle reaches first WP of the grid (WP 0)
                            if target_desc:
                                self.speak("Waiting for vehicle to reach search area before sending target details.")
                                # Assuming the first waypoint of the grid mission is sequence 0
                                if active_mav_controller.wait_for_waypoint_reached(0, timeout_seconds=180): # Wait up to 3 mins
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
            lat = parsed_gps_coords["latitude"]
            lon = parsed_gps_coords["longitude"]
            alt = DEFAULT_WAYPOINT_ALTITUDE
            action_summary = f"Fly to Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt}m."
            if target_desc: action_summary += f" Search for: {target_desc}."
            self.speak(f"Executing for {self.active_agent_id}: {action_summary}")

            single_wp_data = [
                (lat, lon, alt,
                 mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                 0.0,  # param1 (hold time seconds)
                 10.0, # param2 (acceptance radius meters)
                 0.0,  # param3 (pass radius meters, 0 to pass through center)
                 float('nan') # param4 (desired yaw angle at waypoint, NaN for unchanged)
                )
            ]
            if active_mav_controller.upload_mission(single_wp_data):
                self.speak("Waypoint mission uploaded. Attempting to arm and start.")
                if active_mav_controller.arm_vehicle():
                    time.sleep(1)
                    if active_mav_controller.set_mode("AUTO"): # Or GUIDED
                        time.sleep(1)
                        if active_mav_controller.start_mission():
                            self.speak("Mission to waypoint started.")
                            if target_desc: # Send STATUSTEXT after reaching the waypoint
                                self.speak("Waiting for vehicle to reach waypoint before sending target details.")
                                if active_mav_controller.wait_for_waypoint_reached(0, timeout_seconds=180):
                                    active_mav_controller.send_status_text(target_desc, severity=mavutil.mavlink.MAV_SEVERITY_INFO)
                                    self.speak("Target details sent to vehicle.")
                                else:
                                    self.speak("Failed to confirm vehicle reached waypoint. Target details not sent.")
                        else: self.speak("Failed to start mission.")
                    else: self.speak("Failed to set mode.")
                else: self.speak("Failed to arm vehicle.")
            else:
                self.speak("Failed to upload waypoint mission.")


        # Handle providing target description without movement
        elif intent == "PROVIDE_TARGET_DESCRIPTION" and target_desc:
            self.speak(f"Noted target description for {self.active_agent_id}: {target_desc}. Sending to vehicle.")
            if active_mav_controller.send_status_text(target_desc, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
                self.speak("Target information sent to vehicle.")
            else:
                self.speak("Failed to send target information to vehicle.")
        
        else:
            self.speak("Command understood, but no specific MAVLink action defined for this combination or missing critical info (like GPS).")


    def initiate_shutdown(self, reason=""):
        if self.running_flag:
            print(f"Shutdown initiated: {reason}")
            self.running_flag = False
            self.speak("System shutting down.")
            # The main loop in run() will detect running_flag and call cleanup.

    def run(self):
        if not self.running_flag:
            print("Application cannot start due to initialization errors.")
            self.cleanup()
            return

        def main_loop():
            while self.running_flag:
                if self.is_space_currently_pressed and \
                   (self.sm.current_state == self.sm.listening_command or \
                    self.sm.current_state == self.sm.listening_yes_no) and \
                   self.current_audio_stream and not self.current_audio_stream.is_stopped():
                    try:
                        data = self.current_audio_stream.read(AUDIO_READ_CHUNK, exception_on_overflow=False)
                        self.audio_frames.append(data)
                    except IOError as e:
                        print(f"Audio read error: {e}")
                        self.speak("Microphone error during recording.")
                        if self.is_space_currently_pressed:
                            self.is_space_currently_pressed = False
                            if self.sm.current_state == self.sm.listening_command:
                                self.sm.event_ptt_stop_command()
                            elif self.sm.current_state == self.sm.listening_yes_no:
                                self.sm.event_ptt_stop_yes_no()
                time.sleep(0.01)
            print("Exiting main application loop.")
            self.cleanup()

        # Start main loop in a background thread
        t = threading.Thread(target=main_loop, daemon=True)
        t.start()

        self.speak("System initialized.")
        active_agent_display = f" (Active Agent: {self.active_agent_id})" if self.active_agent_id else ""
        print(f"\nSystem ready{active_agent_display}. Press and hold SPACE to speak, or ESC to exit.")
        if self.sm.current_state != self.sm.idle:
            try:
                self.sm.send("event_shutdown_requested")
            except Exception as e:
                print(f"Error forcing SM to idle: {e}")

        # Run the keyboard listener in the main thread (blocking)
        self._setup_keyboard_listener()

        # After the keyboard listener exits (ESC pressed), set running_flag to False to stop main_loop
        self.running_flag = False
        # Wait for the main loop thread to finish before exiting
        t.join()

    def cleanup(self):
        print("Cleaning up resources...")
        if self.keyboard_listener and hasattr(self.keyboard_listener, 'is_alive') and self.keyboard_listener.is_alive():
            self.keyboard_listener.stop()
            # self.keyboard_listener.join() # Can block

        if self.current_audio_stream:
            try:
                if self.current_audio_stream.is_active(): self.current_audio_stream.stop_stream()
                self.current_audio_stream.close()
            except Exception as e: print(f"Error closing active audio stream: {e}")
        self.current_audio_stream = None

        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            print("PyAudio terminated.")

        for agent_id, controller in self.mav_controllers.items():
            if controller:
                print(f"Closing connection for agent {agent_id}...")
                controller.close_connection()
        self.mav_controllers.clear()

        print("System shutdown complete.")


if __name__ == "__main__":
    # vosk.SetLogLevel(-1) # Suppress Vosk logging for cleaner output

    app = LifeguardApp()
    try:
        if app.running_flag: # Only run if initialization was successful
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
        # Ensure cleanup is called if run() exits or if init failed but app object exists
        # The run() method itself calls cleanup on exit. This is a fallback.
        if hasattr(app, 'running_flag') and app.running_flag: # If shutdown wasn't called from within run's try/except
             app.cleanup() # This might be redundant if run() completed its cleanup
        elif not hasattr(app, 'running_flag'): # App object might not be fully initialized
            print("Cleanup called due to early exit or incomplete initialization.")
            # Attempt minimal cleanup if app object is partially formed
            if hasattr(app, 'pyaudio_instance') and app.pyaudio_instance: app.pyaudio_instance.terminate()
            if hasattr(app, 'mav_controllers'):
                for controller in app.mav_controllers.values():
                    if controller: controller.close_connection()

    print("Application has terminated.")
