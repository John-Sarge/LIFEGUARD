# lifeguard.py

import os
import json
import re
import time
import asyncio # Required for some Pymavlink operations if they become async in future versions

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

# --- CONFIGURATION CONSTANTS ---

# STT Configuration
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15" # Name of the Vosk model folder
VOSK_MODEL_PATH = os.path.join("vosk_models", VOSK_MODEL_NAME)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_FRAMES_PER_BUFFER = 8192
AUDIO_READ_CHUNK = 4096 # Amount of data to read from stream at a time

# Audio Pre-processing Configuration
NOISE_REDUCE_TIME_CONSTANT_S = 2.0 # For noisereduce 
NOISE_REDUCE_PROP_DECREASE = 0.9
BANDPASS_LOWCUT_HZ = 300.0
BANDPASS_HIGHCUT_HZ = 3400.0
BANDPASS_ORDER = 5

# NLU Configuration
SPACY_MODEL_NAME = "en_core_web_sm"
# Confidence threshold for NLU interpretation to trigger confirmation
NLU_CONFIDENCE_THRESHOLD = 0.7 # Example value

# MAVLink Configuration
# IMPORTANT: Change this to your vehicle's connection string
# Examples: 'udpin:localhost:14550' (SITL), 'udp:192.168.X.X:14550' (real drone), '/dev/ttyUSB0' (serial)
MAVLINK_CONNECTION_STRING = 'udpin:localhost:14550' # FOR SIMULATION/TESTING
MAVLINK_BAUDRATE = 115200 # Only for serial connections
MAVLINK_SOURCE_SYSTEM_ID = 255 # GCS typically uses 255
DEFAULT_WAYPOINT_ALTITUDE = 30.0 # Meters

# --- AUDIO PRE-PROCESSING FUNCTIONS ---

def apply_noise_reduction(audio_data_int16, sample_rate):
    """Applies noise reduction to int16 audio data."""
    # Convert int16 to float32 for noisereduce
    audio_data_float = audio_data_int16.astype(np.float32) / 32767.0
    
    # Perform noise reduction (non-stationary is default)
    
    reduced_audio_float = nr.reduce_noise(y=audio_data_float, 
                                          sr=sample_rate,
                                          time_constant_s=NOISE_REDUCE_TIME_CONSTANT_S, 
                                          prop_decrease=NOISE_REDUCE_PROP_DECREASE, 
                                          n_fft=512) # Recommended for speech 
    
    # Convert back to int16
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

# --- SPEECH-TO-TEXT (STT) MODULE ---

class SpeechToText:
    def __init__(self, model_path, sample_rate):
        if not os.path.exists(model_path):
            print(f"ERROR: Vosk model not found at {model_path}. Please download and place it correctly.")
            raise FileNotFoundError(f"Vosk model not found at {model_path}")
        
        self.model = vosk.Model(model_path) 
        self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate) 
        self.recognizer.SetWords(True) # Optional: for word-level timestamps
        print("STT (Vosk): Model loaded successfully.")

    def transcribe_chunk(self, audio_chunk_bytes):
        """Processes an audio chunk and returns partial or final transcription."""
        if self.recognizer.AcceptWaveform(audio_chunk_bytes):
            result_json = self.recognizer.Result()
            result_dict = json.loads(result_json)
            return result_dict.get('text', ''), True # Final result
        else:
            partial_result_json = self.recognizer.PartialResult()
            partial_result_dict = json.loads(partial_result_json)
            return partial_result_dict.get('partial', ''), False # Partial result

# --- NATURAL LANGUAGE UNDERSTANDING (NLU) MODULE ---

# Custom spaCy component for GPS parsing using lat-lon-parser
@Language.factory("lat_lon_entity_recognizer", default_config={"gps_label": "LOCATION_GPS_COMPLEX"})
def create_lat_lon_entity_recognizer(nlp: Language, name: str, gps_label: str):
    return LatLonEntityRecognizer(nlp, gps_label) 

class LatLonEntityRecognizer:
    def __init__(self, nlp: Language, gps_label: str):
        self.gps_label = gps_label
        # A broad regex to find potential GPS-like strings.
        # This needs to be comprehensive enough for lat-lon-parser.
        # Example: "latitude 34.05, longitude -118.24", "34.05N 118.24W", "40° 26' 46\" N, 79° 58' 56\" W"
        # This regex is illustrative and may need significant refinement for production.
        self.gps_regex = re.compile(
            r"""
            (?:                           # Non-capturing group for different formats
                (?:[Ll][Aa](?:itude)?\s*[:\s]*)?  # Optional "lat" or "latitude"
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*?)\s* # Latitude part
                (?:[,;\s]+\s*)?                  # Separator
                (?:[Ll][Oo][Nn](?:gitude)?\s*[:\s]*)? # Optional "lon" or "longitude"
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*[EeWw]?)    # Longitude part
| # OR simple decimal pair
                (?:([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+))
            )
            """, 
            re.VERBOSE
        )
        if not Span.has_extension("parsed_gps_coords"):
            Span.set_extension("parsed_gps_coords", default=None) 

    def __call__(self, doc: Doc) -> Doc:
        new_entities = list(doc.ents)
        processed_texts = set() # To avoid processing the same match multiple times if regex is too broad

        for match in self.gps_regex.finditer(doc.text):
            start_char, end_char = match.span()
            gps_string_candidate = doc.text[start_char:end_char]

            if gps_string_candidate in processed_texts:
                continue
            processed_texts.add(gps_string_candidate)

            try:
                # lat-lon-parser often works best on individual lat or lon strings, or clearly separated pairs.
                # We need to be careful how we feed it.
                # This is a simplified approach; a robust solution might split the candidate string
                # into potential lat/lon parts before parsing.
                
                # Attempt to parse the whole candidate string.
                # If it's a pair like "34.05 N, 118.24 W", lat-lon-parser might not handle it directly.
                # We'll try to split by common separators if it looks like a pair.
                
                lat_val, lon_val = None, None
                
                # Try to parse as a pair if a comma or semicolon is present
                parts = re.split(r'[,;]', gps_string_candidate)
                if len(parts) == 2:
                    try:
                        lat_str = parts[0].strip()
                        lon_str = parts[1].strip()
                        # Check if cardinal directions are present, if not, lat-lon-parser might misinterpret
                        if not re.search(r'', lat_str) and not re.search(r'[+-]', lat_str):
                             # Heuristic: if no sign or N/S, assume positive for typical northern hemisphere
                             pass # lat-lon-parser will handle it
                        if not re.search(r'[EeWw]', lon_str) and not re.search(r'[+-]', lon_str):
                             pass # lat-lon-parser will handle it

                        lat_val = parse_lat_lon_external(lat_str) 
                        lon_val = parse_lat_lon_external(lon_str) 
                    except Exception:
                        lat_val, lon_val = None, None # Parsing as pair failed

                if lat_val is not None and lon_val is not None:
                    span = doc.char_span(start_char, end_char, label=self.gps_label, alignment_mode="expand") 
                    if span is not None:
                        span._.set("parsed_gps_coords", {"latitude": lat_val, "longitude": lon_val})
                        
                        is_overlapping = False
                        temp_new_entities = []
                        for ent in new_entities: # Check for overlaps and replace if this is more specific
                            if not (span.end_char <= ent.start_char or span.start_char >= ent.end_char):
                                # Overlap detected. If existing is less specific, remove it.
                                # This is a simple strategy; more complex logic might be needed.
                                if ent.label_!= self.gps_label: # Keep more specific GPS label
                                    is_overlapping = True # Mark to not add the less specific one
                                else: # if same label, keep longer one or new one
                                    if span.end_char - span.start_char > ent.end_char - ent.start_char:
                                        continue # Old entity will be removed by not adding it to temp_new_entities
                                    else:
                                        is_overlapping = True # Keep old, more specific one
                            temp_new_entities.append(ent)
                        new_entities = temp_new_entities

                        if not is_overlapping:
                             new_entities.append(span)
                # else:
                    # print(f"lat-lon-parser could not parse '{gps_string_candidate}' as a pair.")
                    # Optionally, try parsing parts if it's a single coordinate (not typical for waypoints)
                    pass

            except Exception as e:
                # print(f"Error in LatLonEntityRecognizer for '{gps_string_candidate}': {e}")
                pass
        
        doc.ents = tuple(sorted(list(set(new_entities)), key=lambda e: e.start_char)) # Remove duplicates and sort
        return doc

class NaturalLanguageUnderstanding:
    def __init__(self, spacy_model_name):
        try:
            self.nlp = spacy.load(spacy_model_name) 
        except OSError:
            print(f"spaCy model '{spacy_model_name}' not found. Please download it: python -m spacy download {spacy_model_name}")
            raise
        
        # Add custom GPS parser to the pipeline
        if "lat_lon_entity_recognizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("lat_lon_entity_recognizer", after="ner" if self.nlp.has_pipe("ner") else None) 
            print("NLU (spaCy): Custom lat_lon_entity_recognizer added to pipeline.")

        # Add EntityRuler for SAR-specific terms (landmarks, target descriptions)
        if "sar_entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", name="sar_entity_ruler", before="lat_lon_entity_recognizer" if "lat_lon_entity_recognizer" in self.nlp.pipe_names else ("ner" if self.nlp.has_pipe("ner") else None)) # [22, 23, 24, 25]
            
            sar_patterns = [
                # Target Attributes - Color
                {"label": "TARGET_ATTRIBUTE_COLOR", "pattern": [{"LOWER": "red"}]},
                # Target Attributes - Status
                {"label": "TARGET_ATTRIBUTE_STATUS", "pattern": [{"LOWER": "moving"}]},
                # Target Attributes - General Description
                {"label": "TARGET_ATTRIBUTE_DESCRIPTION", "pattern": [{"LOWER": "life"}, {"LOWER": "ring"}]},
                # Simple Landmarks
                {"label": "LOCATION_LANDMARK_ABSOLUTE", "pattern": [{"LOWER": "lighthouse"}]},
                # Example multi-word landmark
                {"label": "LOCATION_LANDMARK_ABSOLUTE", "pattern": [{"LOWER": "north"}, {"LOWER": "dock"}]},
                # Action Verbs
                {"label": "ACTION_VERB", "pattern": [{"LEMMA": {"IN": ["search", "look", "find", "investigate", "proceed", "go", "scan", "locate"]}}]},
                # Spatial Relations
                {"label": "SPATIAL_RELATION", "pattern": [{"LOWER": "near"}]}
            ]
            ruler.add_patterns(sar_patterns)
            print("NLU (spaCy): SAR EntityRuler added to pipeline.")
        
        print("NLU (spaCy): Initialized successfully.")

    def parse_command(self, text):
        """Parses text to extract intent and entities."""
        doc = self.nlp(text)
        
        intent = "UNKNOWN_INTENT"
        entities = {}
        
        extracted_spacy_ents = []
        for ent in doc.ents:
            entity_data = {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            if ent.label_ == "LOCATION_GPS_COMPLEX" and Span.has_extension("parsed_gps_coords") and ent._.get("parsed_gps_coords"):
                entity_data["parsed_gps"] = ent._.parsed_gps_coords
            extracted_spacy_ents.append(entity_data)
        
        entities["raw_spacy_entities"] = extracted_spacy_ents

        # --- Rule-based Intent Recognition and Structured Entity Population ---
                
        # Helper to get entities of a specific label
        def get_entity_values(label):
            return [e["text"] for e in extracted_spacy_ents if e["label"] == label]
        
        def get_parsed_gps():
            for e in extracted_spacy_ents:
                if e["label"] == "LOCATION_GPS_COMPLEX" and "parsed_gps" in e:
                    return e["parsed_gps"]
            return None

        action_verbs = get_entity_values("ACTION_VERB")
        target_objects = get_entity_values("TARGET_OBJECT")
        target_colors = get_entity_values("TARGET_ATTRIBUTE_COLOR")
        target_statuses = get_entity_values("TARGET_ATTRIBUTE_STATUS")
        target_descriptions = get_entity_values("TARGET_ATTRIBUTE_DESCRIPTION")
        
        landmarks_absolute = get_entity_values("LOCATION_LANDMARK_ABSOLUTE")
        # For relative landmarks, more complex parsing of spatial relations + objects is needed.
        # This is a simplified placeholder.
        spatial_relations = get_entity_values("SPATIAL_RELATION")

        parsed_gps_coords = get_parsed_gps()

        # Populate structured entities
        if parsed_gps_coords:
            entities = parsed_gps_coords
        
        if landmarks_absolute:
            entities = ", ".join(landmarks_absolute) # Simple concatenation for now
            if spatial_relations: # Very basic relative landmark handling
                 entities = f"{spatial_relations} {entities}"


        target_details_parts = []
        if target_objects: target_details_parts.extend(target_objects)
        if target_colors: target_details_parts.extend(target_colors)
        if target_statuses: target_details_parts.extend(target_statuses)
        if target_descriptions: target_details_parts.extend(target_descriptions)
        
        if target_details_parts:
            entities = " ".join(target_details_parts)

        # Intent Logic (simplified)
        if action_verbs:
            entities = action_verbs # Take the first one for simplicity
            
            if (parsed_gps_coords or landmarks_absolute):
                if target_details_parts:
                    intent = "COMBINED_SEARCH_AND_TARGET"
                else:
                    intent = "REQUEST_SEARCH_AT_LOCATION"
            elif target_details_parts:
                intent = "PROVIDE_TARGET_DESCRIPTION"
            else:
                intent = "GENERIC_COMMAND" # e.g. "stop listening"
        
        # Basic confidence (placeholder - rule-based doesn't have inherent confidence like ML)
        # We can assign higher confidence if critical entities are found.
        confidence = 0.5 
        if intent!= "UNKNOWN_INTENT" and (parsed_gps_coords or landmarks_absolute or target_details_parts):
            confidence = 0.85 # Higher if key elements are present

        return {"text": text, "intent": intent, "confidence": confidence, "entities": entities}


# --- MAVLINK INTEGRATION MODULE ---

class MavlinkController:
    def __init__(self, connection_string, baudrate=None, source_system_id=255):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.source_system_id = source_system_id
        self.master = None
        self._connect()

    def _connect(self):
        """Establishes MAVLink connection."""
        print(f"MAVLink: Attempting to connect to {self.connection_string}...")
        try:
            if self.baudrate: # Serial connection
                self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate, source_system=self.source_system_id)
            else: # UDP/TCP connection
                self.master = mavutil.mavlink_connection(self.connection_string, source_system=self.source_system_id) # [16]
            
            self.master.wait_heartbeat(timeout=10) 
            print("MAVLink: Heartbeat received. Connection established.")
            print(f"MAVLink: Target system: {self.master.target_system}, component: {self.master.target_component}")
        except Exception as e:
            print(f"MAVLink: Failed to connect or receive heartbeat: {e}")
            self.master = None

    def is_connected(self):
        return self.master is not None

    def send_nav_waypoint(self, lat, lon, alt, hold_time=0, accept_radius=10, pass_radius=0, yaw_angle=float('nan')):
        """Sends a single MAV_CMD_NAV_WAYPOINT using MISSION_ITEM_INT."""
        
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot send waypoint.")
            return False

        target_sys = self.master.target_system
        target_comp = self.master.target_component
        
        seq_num = 0  # For a single waypoint, sequence can be 0. For missions, it's incremental.
        frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT 
        command_id = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT 
        is_current = 1  # Make this the current waypoint to navigate to (for immediate action in some modes)
        auto_cont = 1   # Autocontinue (relevant for missions)
        
        lat_int = int(lat * 1e7)
        lon_int = int(lon * 1e7)

        print(f"MAVLink: Sending MAV_CMD_NAV_WAYPOINT to Lat: {lat}, Lon: {lon}, Alt: {alt}")
        try:
            self.master.mav.mission_item_int_send( 
                target_sys,        # target_system
                target_comp,       # target_component
                seq_num,           # sequence
                frame,             # frame
                command_id,        # command
                is_current,        # current
                auto_cont,         # autocontinue
                float(hold_time),       # param1 (Hold time) 
                float(accept_radius),   # param2 (Acceptance radius)
                float(pass_radius),     # param3 (Pass radius)
                float(yaw_angle),       # param4 (Yaw angle)
                lat_int,           # x (latitude * 10^7)
                lon_int,           # y (longitude * 10^7)
                float(alt)         # z (altitude in meters)
            )
            print("MAVLink: MAV_CMD_NAV_WAYPOINT sent.")
            # Note: For immediate action, vehicle must be in a mode like GUIDED and configured
            # to accept mission items for navigation. Uploading a full mission is more robust.
            return True
        except Exception as e:
            print(f"MAVLink: Error sending MAV_CMD_NAV_WAYPOINT: {e}")
            return False

    def upload_mission(self, waypoints_data):
        """
        Uploads a list of waypoints as a mission to the vehicle.
        waypoints_data: list of tuples, e.g., [(lat1, lon1, alt1, cmd_id, p1, p2, p3, p4),...]
                        cmd_id defaults to NAV_WAYPOINT if not specified or is None.
                        p1-p4 are command-specific parameters.
        Based on Section 5.3 of the report.
        """
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot upload mission.")
            return False

        wp_loader = mavwp.MAVWPLoader()
        
        for seq, wp_item in enumerate(waypoints_data):
            lat, lon, alt = wp_item, wp_item[1], wp_item[2]
            command_id = wp_item[3] if len(wp_item) > 3 and wp_item[3] is not None else mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
            
            # Default params for NAV_WAYPOINT, can be overridden
            p1 = wp_item[4] if len(wp_item) > 4 and wp_item[4] is not None else 0.0 # Hold time
            p2 = wp_item[5] if len(wp_item) > 5 and wp_item[5] is not None else 10.0 # Acceptance radius
            p3 = wp_item[6] if len(wp_item) > 6 and wp_item[6] is not None else 0.0 # Pass radius
            p4 = wp_item[7] if len(wp_item) > 7 and wp_item[7] is not None else float('nan') # Yaw

            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            is_current = 1 if seq == 0 else 0 # First item can be current for takeoff, otherwise 0
            autocontinue = 1
            
            # Special handling for TAKEOFF command parameters
            if command_id == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                p1 = wp_item[4] if len(wp_item) > 4 and wp_item[4] is not None else 0.0 # Pitch (Plane) / Empty (Copter)
                # p2, p3, p4 are often 0 or NaN for takeoff
                p2 = wp_item[5] if len(wp_item) > 5 and wp_item[5] is not None else 0.0
                p3 = wp_item[6] if len(wp_item) > 6 and wp_item[6] is not None else 0.0
                p4 = wp_item[7] if len(wp_item) > 7 and wp_item[7] is not None else float('nan') # Yaw

            mission_item = mavutil.mavlink.MAVLink_mission_item_int_message(
                self.master.target_system, self.master.target_component, seq, frame, command_id,
                is_current, autocontinue,
                p1, p2, p3, p4,
                int(lat * 1e7), int(lon * 1e7), float(alt)
            )
            wp_loader.add(mission_item)
        
        print(f"MAVLink: Uploading {wp_loader.count()} waypoints...")
        
        try:
            self.master.mav.mission_clear_all_send(self.master.target_system, self.master.target_component) 
            ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            if not ack or ack.type!= mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print(f"MAVLink: Failed to clear mission or no ACK. ACK: {ack}")
                # return False # Continue to try upload anyway for some systems

            self.master.mav.mission_count_send(self.master.target_system, self.master.target_component, wp_loader.count()) 

            for i in range(wp_loader.count()):
                msg = self.master.recv_match(type='MISSION_REQUEST_INT', blocking=True, timeout=5) 
                if not msg:
                    print("MAVLink: No MISSION_REQUEST received, upload failed.")
                    return False
                print(f"MAVLink: Sending waypoint {msg.seq}")
                self.master.mav.send(wp_loader.wp(msg.seq))
                
                if msg.seq == wp_loader.count() - 1:
                    break
            
            final_ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            if final_ack and final_ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("MAVLink: Mission upload successful.")
                return True
            else:
                print(f"MAVLink: Mission upload failed or no final ACK. ACK: {final_ack}")
                return False
        except Exception as e:
            print(f"MAVLink: Error during mission upload: {e}")
            return False

    def set_mode(self, mode_name):
        """Sets the vehicle mode."""
        
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot set mode.")
            return False
        
        if mode_name not in self.master.mode_mapping(): 
            print(f"MAVLink: Unknown mode: {mode_name}")
            print(f"MAVLink: Available modes: {list(self.master.mode_mapping().keys())}")
            return False

        mode_id = self.master.mode_mapping()[mode_name]
        
        print(f"MAVLink: Setting mode to {mode_name} (ID: {mode_id})...")
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        
        # Wait for COMMAND_ACK
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"MAVLink: Mode change to {mode_name} successful.")
                return True
            else:
                print(f"MAVLink: Mode change to {mode_name} failed with result: {ack_msg.result}")
                return False
        else:
            print(f"MAVLink: No COMMAND_ACK received for mode change to {mode_name}. ACK: {ack_msg}")
            return False

    def start_mission(self):
        """Sends MAV_CMD_MISSION_START to begin the uploaded mission."""
        # Based on Section 5.4 of the report
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot start mission.")
            return False

        print("MAVLink: Sending MAV_CMD_MISSION_START...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0, # confirmation
            0, # param1: first_item (0 to start from beginning)
            0, # param2: last_item (0 for all items from first_item)
            0, 0, 0, 0, 0 # params 3-7 unused
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_MISSION_START and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Mission start command accepted.")
            return True
        else:
            print(f"MAVLink: Mission start command failed or no ACK. ACK: {ack_msg}")
            return False
            
    def arm_vehicle(self):
        """Arms the vehicle."""
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot arm.")
            return False
        print("MAVLink: Attempting to arm vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, # confirmation
            1, # param1: 1 to arm, 0 to disarm
            0, 0, 0, 0, 0, 0 # params 2-7 unused
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Vehicle armed successfully.")
            return True
        else:
            print(f"MAVLink: Arming failed or no ACK received. ACK: {ack_msg}")
            return False

    def disarm_vehicle(self):
        """Disarms the vehicle."""
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot disarm.")
            return False
        print("MAVLink: Attempting to disarm vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, # confirmation
            0, # param1: 1 to arm, 0 to disarm
            0, 0, 0, 0, 0, 0 # params 2-7 unused
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
            print("MAVLink: Connection closed.")


# --- MAIN APPLICATION LOGIC ---

def process_commander_intent(nlu_result, mav_controller):
    """Processes NLU result and sends MAVLink commands if confirmed."""
    intent = nlu_result.get("intent")
    entities = nlu_result.get("entities", {})
    confidence = nlu_result.get("confidence", 0.0)

    print(f"\nNLU Interpretation: Intent='{intent}', Confidence={confidence:.2f}")
    print(f"Entities: {json.dumps(entities, indent=2)}")

    if intent == "UNKNOWN_INTENT" or confidence < NLU_CONFIDENCE_THRESHOLD:
        print("Commander Intent: Could not confidently understand the command or intent is unknown. Please rephrase or be more specific.")
        return

    # Extract key information for MAVLink
    gps_coords = entities.get("LOCATION_GPS_PARSED")
    # landmark_loc = entities.get("LOCATION_LANDMARK") # For future use, convert landmark to GPS
    target_desc = entities.get("TARGET_DESCRIPTION_FULL")
    
    # For now, we primarily act on GPS coordinates
    if intent in ["COMBINED_SEARCH_AND_TARGET", "REQUEST_SEARCH_AT_LOCATION"] and gps_coords:
        lat = gps_coords.get("latitude")
        lon = gps_coords.get("longitude")
        alt = DEFAULT_WAYPOINT_ALTITUDE # Use default altitude

        if lat is not None and lon is not None:
            confirmation_message = f"Understood: {nlu_result['text']}\n"
            confirmation_message += f"Action: Fly to Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt}m.\n"
            if target_desc:
                confirmation_message += f"Search for: {target_desc}.\n"
            confirmation_message += "Is this correct? (yes/no): "
            
            # Confirmation Loop 
            user_confirmation = input(confirmation_message).strip().lower()

            if user_confirmation == "yes":
                print("Commander Intent: Command confirmed. Sending waypoint to vehicle...")
                if mav_controller.is_connected():
                    # Example: Create a simple mission with one waypoint
                    # For a real mission, you might add takeoff, RTL, etc.
                    mission_waypoints = [(lat, lon, alt, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 10.0, 0.0, float('nan'))]
                    # For a more complete mission:
                    # 1. Takeoff (if applicable, e.g., from current location or first waypoint location)
                    #    (vehicle_current_lat, vehicle_current_lon, takeoff_altitude, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, pitch, 0,0,yaw)
                    # 2. Waypoint(s)
                    #    (lat, lon, alt, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, hold_time, accept_radius, pass_radius, yaw)
                    # 3. Return to Launch (RTL) (optional, at the end)
                    #    (0,0,0, mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0,0,0,0) # Lat/Lon/Alt often ignored for RTL

                    # Ensure vehicle is in GUIDED mode to accept waypoints or AUTO for missions
                    # This is a simplified flow. A robust system would check current mode and state.
                    # if mav_controller.set_mode("GUIDED"): # Or "AUTO" for missions
                    #     time.sleep(1) # Allow mode change to propagate
                    #     if mav_controller.arm_vehicle():
                    #         time.sleep(1)
                    #         # For a single waypoint in GUIDED mode:
                    #         # mav_controller.send_nav_waypoint(lat, lon, alt)
                    #         # For a mission:
                    #         if mav_controller.upload_mission(mission_waypoints):
                    #             mav_controller.start_mission()
                    #     else:
                    #         print("Commander Intent: Failed to arm vehicle.")
                    # else:
                    #     print("Commander Intent: Failed to set vehicle mode.")
                    
                    # Simplified: just send the single waypoint for now
                    # Assumes vehicle is already in a mode that accepts waypoints (e.g., GUIDED) and armed.
                    # THIS IS HIGHLY SIMPLIFIED. REAL OPERATIONS NEED MORE ROBUST STATE MANAGEMENT.
                    print("MAVLink: (Simplified) Sending single waypoint. Ensure vehicle is in GUIDED mode and ARMED.")
                    mav_controller.send_nav_waypoint(lat, lon, alt)

                else:
                    print("Commander Intent: MAVLink not connected. Cannot send command.")
            else:
                print("Commander Intent: Command cancelled by user.")
        else:
            print("Commander Intent: GPS coordinates not fully parsed. Cannot proceed.")
    elif intent == "PROVIDE_TARGET_DESCRIPTION" and target_desc:
        print(f"Commander Intent: Target description noted: {target_desc}. This information can be used by onboard perception systems (not implemented in this script).")
    else:
        print("Commander Intent: Command understood, but no actionable GPS location or full target description found for MAVLink.")


if __name__ == "__main__":
    # --- Initialize Components ---
    print("Initializing Commander Intent System...")
    
    # Initialize STT
    try:
        stt_system = SpeechToText(model_path=VOSK_MODEL_PATH, sample_rate=AUDIO_SAMPLE_RATE)
    except Exception as e:
        print(f"Fatal Error: Could not initialize STT system: {e}")
        exit()

    # Initialize NLU
    try:
        nlu_system = NaturalLanguageUnderstanding(spacy_model_name=SPACY_MODEL_NAME)
    except Exception as e:
        print(f"Fatal Error: Could not initialize NLU system: {e}")
        exit()

    # Initialize MAVLink Controller
    # Note: Connection is attempted here. If it fails, mav_controller.is_connected() will be False.
    mav_controller = MavlinkController(connection_string=MAVLINK_CONNECTION_STRING, baudrate=MAVLINK_BAUDRATE if MAVLINK_CONNECTION_STRING.startswith("/dev/") else None)

    # Initialize PyAudio for microphone input
    p_audio = pyaudio.PyAudio()
    try:
        audio_stream = p_audio.open(format=AUDIO_FORMAT,
                                    channels=AUDIO_CHANNELS,
                                    rate=AUDIO_SAMPLE_RATE,
                                    input=True,
                                    frames_per_buffer=AUDIO_FRAMES_PER_BUFFER)
        audio_stream.start_stream()
        print("\nMicrophone stream opened. System is ready.")
        print("Speak your command (e.g., 'Search at latitude 34.05 longitude -118.24 for a person in a life ring').")
        print("Say 'stop listening' to end the program.")
    except Exception as e:
        print(f"Fatal Error: Could not open microphone stream: {e}")
        p_audio.terminate()
        exit()

    # --- Main Listening Loop ---
    try:
        while True:
            print("\nListening for command...")
            raw_audio_data_bytes = audio_stream.read(AUDIO_READ_CHUNK, exception_on_overflow=False)
            
            # Convert byte data to numpy array for pre-processing
            audio_data_np = np.frombuffer(raw_audio_data_bytes, dtype=np.int16)

            # 1. Audio Pre-processing
            # print("Applying noise reduction...")
            denoised_audio_np = apply_noise_reduction(audio_data_np, AUDIO_SAMPLE_RATE)
            # print("Applying band-pass filter...")
            filtered_audio_np = apply_bandpass_filter(denoised_audio_np, AUDIO_SAMPLE_RATE)
            
            # Convert processed numpy array back to bytes for Vosk
            processed_audio_bytes = filtered_audio_np.tobytes()

            # 2. Speech-to-Text
            transcribed_text, is_final = stt_system.transcribe_chunk(processed_audio_bytes)
            
            if is_final and transcribed_text:
                print(f"Commander Said (STT): {transcribed_text}")
                
                if "stop listening" in transcribed_text.lower():
                    print("Stop command received. Exiting.")
                    break

                # 3. Natural Language Understanding
                nlu_result = nlu_system.parse_command(transcribed_text)
                
                # 4. Process Intent and Send MAVLink (includes confirmation)
                process_commander_intent(nlu_result, mav_controller)
                
                # Reset recognizer for next command after a final result
                stt_system.recognizer.Reset()


            elif transcribed_text: # Partial result
                print(f"Partial: {transcribed_text}", end='\r', flush=True)

    except KeyboardInterrupt:
        print("\nSystem interrupted by user. Shutting down.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        if 'audio_stream' in locals() and audio_stream.is_active():
            audio_stream.stop_stream()
            audio_stream.close()
        if 'p_audio' in locals():
            p_audio.terminate()
        if 'mav_controller' in locals():
            mav_controller.close_connection()
        print("System shutdown complete.")
