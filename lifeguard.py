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

# PTT, TTS, State Machine
from pynput import keyboard
import pyttsx3
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed


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
MAVLINK_CONNECTION_STRING = 'udpin:localhost:14550' # FOR SIMULATION/TESTING
MAVLINK_BAUDRATE = 115200 # Only for serial connections
MAVLINK_SOURCE_SYSTEM_ID = 255 # GCS typically uses 255
DEFAULT_WAYPOINT_ALTITUDE = 30.0 # Meters

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

# --- SPEECH-TO-TEXT (STT) MODULE ---

class SpeechToText:
    def __init__(self, model_path, sample_rate):
        self.sample_rate = sample_rate
        if not os.path.exists(model_path):
            print(f"ERROR: Vosk model not found at {model_path}. Please download and place it correctly.")
            # Let main app handle exit or fallback
            raise FileNotFoundError(f"Vosk model not found at {model_path}")

        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True) # Optional: for word-level timestamps
        # Removed print statement, main app will confirm initialization

    def recognize_buffer(self, raw_audio_buffer_bytes):
        """Processes a complete audio buffer and returns final transcription."""
        if not raw_audio_buffer_bytes:
            self.recognizer.Reset() # Reset even if buffer is empty
            return ""

        audio_data_np = np.frombuffer(raw_audio_buffer_bytes, dtype=np.int16)

        # Apply pre-processing
        denoised_audio_np = apply_noise_reduction(audio_data_np, self.sample_rate)
        filtered_audio_np = apply_bandpass_filter(denoised_audio_np, self.sample_rate)
        processed_audio_bytes = filtered_audio_np.tobytes()

        text_result = ""
        if self.recognizer.AcceptWaveform(processed_audio_bytes):
            result_json = self.recognizer.Result()
        else:
            result_json = self.recognizer.FinalResult() # Get any remaining part

        self.recognizer.Reset() # CRITICAL: Reset after every full utterance

        try:
            result_dict = json.loads(result_json)
            text_result = result_dict.get('text', '')
        except json.JSONDecodeError:
            print(f"STT (Vosk): Could not decode JSON result: {result_json}")
        except Exception as e:
            print(f"STT (Vosk): Error processing result: {e}")
        return text_result

# --- NATURAL LANGUAGE UNDERSTANDING (NLU) MODULE ---
# (LatLonEntityRecognizer class remains unchanged from original)
@Language.factory("lat_lon_entity_recognizer", default_config={"gps_label": "LOCATION_GPS_COMPLEX"})
def create_lat_lon_entity_recognizer(nlp: Language, name: str, gps_label: str):
    return LatLonEntityRecognizer(nlp, gps_label)

class LatLonEntityRecognizer:
    def __init__(self, nlp: Language, gps_label: str):
        self.gps_label = gps_label
        self.gps_regex = re.compile(
            r"""
            (?: # Non-capturing group for different formats
                (?:[Ll][Aa](?:itude)?\s*[:\s]*)? # Optional "lat" or "latitude"
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*?)\s* # Latitude part
                (?:[,;\s]+\s*)? # Separator
                (?:[Ll][Oo][Nn](?:gitude)?\s*[:\s]*)? # Optional "lon" or "longitude"
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*[EeWw]?) # Longitude part
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
        processed_texts = set()

        for match in self.gps_regex.finditer(doc.text):
            start_char, end_char = match.span()
            gps_string_candidate = doc.text[start_char:end_char]

            if gps_string_candidate in processed_texts:
                continue
            processed_texts.add(gps_string_candidate)

            try:
                lat_val, lon_val = None, None
                parts = re.split(r'[,;]', gps_string_candidate)
                if len(parts) == 2:
                    try:
                        lat_str = parts.strip()
                        lon_str = parts[1].strip()
                        if not re.search(r'', lat_str) and not re.search(r'[+-]', lat_str):
                            pass
                        if not re.search(r'[EeWw]', lon_str) and not re.search(r'[+-]', lon_str):
                            pass
                        lat_val = parse_lat_lon_external(lat_str)
                        lon_val = parse_lat_lon_external(lon_str)
                    except Exception:
                        lat_val, lon_val = None, None

                if lat_val is not None and lon_val is not None:
                    span = doc.char_span(start_char, end_char, label=self.gps_label, alignment_mode="expand")
                    if span is not None:
                        span._.set("parsed_gps_coords", {"latitude": lat_val, "longitude": lon_val})
                        is_overlapping = False
                        temp_new_entities = []
                        for ent in new_entities:
                            if not (span.end_char <= ent.start_char or span.start_char >= ent.end_char):
                                if ent.label_!= self.gps_label:
                                    is_overlapping = True
                                else:
                                    if span.end_char - span.start_char > ent.end_char - ent.start_char:
                                        continue
                                    else:
                                        is_overlapping = True
                            temp_new_entities.append(ent)
                        new_entities = temp_new_entities
                        if not is_overlapping:
                            new_entities.append(span)
            except Exception as e:
                # print(f"Error in LatLonEntityRecognizer for '{gps_string_candidate}': {e}")
                pass
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
            # print("NLU (spaCy): Custom lat_lon_entity_recognizer added to pipeline.")

        if "sar_entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", name="sar_entity_ruler", before="lat_lon_entity_recognizer" if "lat_lon_entity_recognizer" in self.nlp.pipe_names else ("ner" if self.nlp.has_pipe("ner") else None))
            sar_patterns = [
                # Example patterns (add your own as needed)
                # {"label": "TARGET_OBJECT", "pattern": [{"LOWER": "person"}]},
                # {"label": "TARGET_ATTRIBUTE_COLOR", "pattern": [{"LOWER": "red"}]},
                {"label": "ACTION_VERB", "pattern": [{"LEMMA": {"IN": ["search", "look", "find", "investigate", "proceed", "go", "scan", "locate"]}}]}
            ]
            ruler.add_patterns(sar_patterns)
            # print("NLU (spaCy): SAR EntityRuler added to pipeline.")
        # print("NLU (spaCy): Initialized successfully.")

    def parse_command(self, text):
        doc = self.nlp(text)
        intent = "UNKNOWN_INTENT"
        entities_payload = {} # Changed name to avoid conflict with spaCy's entities

        extracted_spacy_ents = []
        for ent in doc.ents:
            entity_data = {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            if ent.label_ == "LOCATION_GPS_COMPLEX" and Span.has_extension("parsed_gps_coords") and ent._.get("parsed_gps_coords"):
                entity_data["parsed_gps"] = ent._.parsed_gps_coords
            extracted_spacy_ents.append(entity_data)
        entities_payload["raw_spacy_entities"] = extracted_spacy_ents

        def get_entity_values(label):
            return [e["text"] for e in extracted_spacy_ents if e["label"] == label]

        def get_parsed_gps():
            for e in extracted_spacy_ents:
                if e["label"] == "LOCATION_GPS_COMPLEX" and "parsed_gps" in e:
                    return e["parsed_gps"]
            return None

        action_verbs = get_entity_values("ACTION_VERB")
        target_objects = get_entity_values("TARGET_OBJECT") # Note: TARGET_OBJECT not in patterns
        target_colors = get_entity_values("TARGET_ATTRIBUTE_COLOR")
        target_statuses = get_entity_values("TARGET_ATTRIBUTE_STATUS")
        target_descriptions = get_entity_values("TARGET_ATTRIBUTE_DESCRIPTION")
        landmarks_absolute = get_entity_values("LOCATION_LANDMARK_ABSOLUTE")
        spatial_relations = get_entity_values("SPATIAL_RELATION")
        parsed_gps_coords = get_parsed_gps()

        if parsed_gps_coords:
            entities_payload.update(parsed_gps_coords) # Add lat/lon directly

        if landmarks_absolute:
            entities_payload["landmark_absolute"] = ", ".join(landmarks_absolute)
            if spatial_relations:
                 entities_payload["landmark_absolute"] = f"{' '.join(spatial_relations)} {entities_payload['landmark_absolute']}"

        target_details_parts = []
        if target_objects: target_details_parts.extend(target_objects)
        if target_colors: target_details_parts.extend(target_colors)
        if target_statuses: target_details_parts.extend(target_statuses)
        if target_descriptions: target_details_parts.extend(target_descriptions)

        if target_details_parts:
            entities_payload["target_description_full"] = " ".join(target_details_parts)

        if action_verbs:
            entities_payload["action_verb"] = action_verbs # Take first for simplicity
            if (parsed_gps_coords or landmarks_absolute):
                if target_details_parts:
                    intent = "COMBINED_SEARCH_AND_TARGET"
                else:
                    intent = "REQUEST_SEARCH_AT_LOCATION"
            elif target_details_parts:
                intent = "PROVIDE_TARGET_DESCRIPTION"
            else:
                intent = "GENERIC_COMMAND"

        confidence = 0.5
        if intent!= "UNKNOWN_INTENT" and (parsed_gps_coords or landmarks_absolute or target_details_parts):
            confidence = 0.85

        return {"text": text, "intent": intent, "confidence": confidence, "entities": entities_payload}


# --- MAVLINK INTEGRATION MODULE ---
# (MavlinkController class remains unchanged from original)
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
            if self.baudrate:
                self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate, source_system=self.source_system_id)
            else:
                self.master = mavutil.mavlink_connection(self.connection_string, source_system=self.source_system_id)
            self.master.wait_heartbeat(timeout=10)
            print("MAVLink: Heartbeat received. Connection established.")
            print(f"MAVLink: Target system: {self.master.target_system}, component: {self.master.target_component}")
        except Exception as e:
            print(f"MAVLink: Failed to connect or receive heartbeat: {e}")
            self.master = None

    def is_connected(self):
        return self.master is not None

    def send_nav_waypoint(self, lat, lon, alt, hold_time=0, accept_radius=10, pass_radius=0, yaw_angle=float('nan')):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot send waypoint.")
            return False
        target_sys = self.master.target_system
        target_comp = self.master.target_component
        seq_num = 0
        frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
        command_id = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
        is_current = 1
        auto_cont = 1
        lat_int = int(lat * 1e7)
        lon_int = int(lon * 1e7)
        print(f"MAVLink: Sending MAV_CMD_NAV_WAYPOINT to Lat: {lat}, Lon: {lon}, Alt: {alt}")
        try:
            self.master.mav.mission_item_int_send(
                target_sys, target_comp, seq_num, frame, command_id, is_current, auto_cont,
                float(hold_time), float(accept_radius), float(pass_radius), float(yaw_angle),
                lat_int, lon_int, float(alt)
            )
            print("MAVLink: MAV_CMD_NAV_WAYPOINT sent.")
            return True
        except Exception as e:
            print(f"MAVLink: Error sending MAV_CMD_NAV_WAYPOINT: {e}")
            return False

    def upload_mission(self, waypoints_data):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot upload mission.")
            return False
        wp_loader = mavwp.MAVWPLoader()
        for seq, wp_item in enumerate(waypoints_data):
            lat, lon, alt = wp_item, wp_item[1], wp_item[2]
            command_id = wp_item[3] if len(wp_item) > 3 and wp_item[3] is not None else mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
            p1 = wp_item[4] if len(wp_item) > 4 and wp_item[4] is not None else 0.0
            p2 = wp_item[5] if len(wp_item) > 5 and wp_item[5] is not None else 10.0
            p3 = wp_item[6] if len(wp_item) > 6 and wp_item[6] is not None else 0.0
            p4 = wp_item[7] if len(wp_item) > 7 and wp_item[7] is not None else float('nan')
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            is_current = 1 if seq == 0 else 0
            autocontinue = 1
            if command_id == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                p1 = wp_item[4] if len(wp_item) > 4 and wp_item[4] is not None else 0.0
                p2 = wp_item[5] if len(wp_item) > 5 and wp_item[5] is not None else 0.0
                p3 = wp_item[6] if len(wp_item) > 6 and wp_item[6] is not None else 0.0
                p4 = wp_item[7] if len(wp_item) > 7 and wp_item[7] is not None else float('nan')
            mission_item = mavutil.mavlink.MAVLink_mission_item_int_message(
                self.master.target_system, self.master.target_component, seq, frame, command_id,
                is_current, autocontinue, p1, p2, p3, p4,
                int(lat * 1e7), int(lon * 1e7), float(alt)
            )
            wp_loader.add(mission_item)
        print(f"MAVLink: Uploading {wp_loader.count()} waypoints...")
        try:
            self.master.mav.mission_clear_all_send(self.master.target_system, self.master.target_component)
            ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            # if not ack or ack.type!= mavutil.mavlink.MAV_MISSION_ACCEPTED:
                # print(f"MAVLink: Failed to clear mission or no ACK. ACK: {ack}")
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
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot start mission.")
            return False
        print("MAVLink: Sending MAV_CMD_MISSION_START...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_MISSION_START and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Mission start command accepted.")
            return True
        else:
            print(f"MAVLink: Mission start command failed or no ACK. ACK: {ack_msg}")
            return False

    def arm_vehicle(self):
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot arm.")
            return False
        print("MAVLink: Attempting to arm vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Vehicle armed successfully.")
            return True
        else:
            print(f"MAVLink: Arming failed or no ACK received. ACK: {ack_msg}")
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
            print("MAVLink: Connection closed.")

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
        self.model = model_ref # Reference to the LifeguardApp instance
        self.captured_command_audio_data = None
        self.captured_yes_no_audio_data = None
        self._confirmation_needed_flag_internal = False # For SM conditions
        super().__init__()

    # Condition for event_command_recognized transition
    def confirmation_needed_flag(self, event_data): # Name must match cond string
        return self._confirmation_needed_flag_internal

    # --- Actions ---
    def on_enter_listening_command(self, event_data):
        print("SM: State -> LISTENING_COMMAND")
        self.model.speak("Listening for command...") # Brief feedback
        self.model.start_audio_recording()

    def on_exit_listening_command(self, event_data):
        print("SM: Exiting LISTENING_COMMAND")
        self.captured_command_audio_data = self.model.stop_audio_recording()

    def on_enter_processing_command(self, event_data):
        print("SM: State -> PROCESSING_COMMAND")
        command_text, needs_conf, is_stop_command = self.model.process_command_stt(self.captured_command_audio_data)
        if is_stop_command:
            self.model.initiate_shutdown("Stop command recognized.")
            self.send("event_shutdown_requested") # Ensure SM goes to idle before shutdown
            return

        if command_text:
            self._confirmation_needed_flag_internal = needs_conf
            self.send("event_command_recognized")
        else:
            self.model.speak("Sorry, I could not understand the command. Please try again.")
            self.send("event_stt_failed_command")

    def on_enter_speaking_confirmation(self, event_data):
        print("SM: State -> SPEAKING_CONFIRMATION")
        prompt = f"Did you say: {self.model.command_to_execute_display_text}? Please press space and say yes or no."
        self.model.speak(prompt)
        self.send("event_confirmation_spoken")

    def on_enter_awaiting_yes_no(self, event_data):
        print("SM: State -> AWAITING_YES_NO")
        # Optionally start a timeout timer here

    def on_enter_listening_yes_no(self, event_data):
        print("SM: State -> LISTENING_YES_NO")
        self.model.speak("Listening for yes or no...")
        self.model.start_audio_recording()

    def on_exit_listening_yes_no(self, event_data):
        print("SM: Exiting LISTENING_YES_NO")
        self.captured_yes_no_audio_data = self.model.stop_audio_recording()

    def on_enter_processing_yes_no(self, event_data):
        print("SM: State -> PROCESSING_YES_NO")
        response = self.model.process_yes_no_stt(self.captured_yes_no_audio_data)
        if response == "yes":
            self.send("event_yes_confirmed")
        elif response == "no":
            self.model.speak("Okay, command cancelled.")
            self.send("event_no_confirmed")
        else:
            self.model.speak("Sorry, I didn't catch that. Please try saying yes or no again.")
            self.send("event_stt_failed_yes_no")

    def on_enter_executing_action(self, event_data):
        print("SM: State -> EXECUTING_ACTION")
        self.model.execute_lifeguard_command()
        # execute_lifeguard_command should call self.send("event_action_completed")
        # or speak and then the SM transitions. For now, let's assume it's synchronous.
        # self.model.speak("Action completed.") # Moved to execute_lifeguard_command
        self.send("event_action_completed")


    def on_enter_idle(self, event_data):
        source_id = event_data.source.id if event_data.source else "INITIAL"
        event_name = event_data.event if event_data.event else "INIT"
        print(f"SM: State -> IDLE (from {source_id} via {event_name})")
        self.model.command_to_execute_display_text = None
        self.model.original_nlu_result_for_execution = None
        self._confirmation_needed_flag_internal = False
        if self.model.running_flag: # Don't prompt if shutting down
             print("\nSystem ready. Press and hold SPACE to speak a command, or ESC to exit.")


# --- MAIN APPLICATION CLASS ---
class LifeguardApp:
    def __init__(self):
        print("Initializing Commander Intent System with PTT...")
        self.running_flag = True
        self.pyaudio_instance = None
        self.stt_system = None
        self.nlu_system = None
        self.mav_controller = None
        self.tts_engine = None
        self.keyboard_listener = None
        self.sm = None

        self.audio_frames = []
        self.is_space_currently_pressed = False # Tracks physical key state
        self.current_audio_stream = None

        # For storing command details between states
        self.command_to_execute_display_text = None
        self.original_nlu_result_for_execution = None

        self._initialize_components()
        self.sm = LifeguardAppSM(model_ref=self)

    def _initialize_components(self):
        # PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()

        # STT
        try:
            self.stt_system = SpeechToText(model_path=VOSK_MODEL_PATH, sample_rate=AUDIO_SAMPLE_RATE)
            print("STT (Vosk): Model loaded successfully.")
        except Exception as e:
            print(f"Fatal Error: Could not initialize STT system: {e}")
            self.running_flag = False; return

        # NLU
        try:
            self.nlu_system = NaturalLanguageUnderstanding(spacy_model_name=SPACY_MODEL_NAME)
            print("NLU (spaCy): Initialized successfully.")
        except Exception as e:
            print(f"Fatal Error: Could not initialize NLU system: {e}")
            self.running_flag = False; return

        # MAVLink
        baud = MAVLINK_BAUDRATE if MAVLINK_CONNECTION_STRING.startswith(("/dev/", "COM")) else None
        self.mav_controller = MavlinkController(connection_string=MAVLINK_CONNECTION_STRING, baudrate=baud)

        # TTS
        try:
            self.tts_engine = pyttsx3.init()
            # Optionally adjust TTS properties here
            # self.tts_engine.setProperty('rate', 150)
            print("TTS (pyttsx3): Engine initialized.")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            # Non-fatal, but speak() will not work

    def _setup_keyboard_listener(self):
        self.keyboard_listener = keyboard.Listener(on_press=self._on_ptt_press, on_release=self._on_ptt_release)
        self.keyboard_listener.start()

    def _on_ptt_press(self, key):
        if key == keyboard.Key.space:
            if not self.is_space_currently_pressed: # First press
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
            if self.is_space_currently_pressed: # Was pressed
                self.is_space_currently_pressed = False
                try:
                    if self.sm.current_state == self.sm.listening_command:
                        self.sm.event_ptt_stop_command()
                    elif self.sm.current_state == self.sm.listening_yes_no:
                        self.sm.event_ptt_stop_yes_no()
                except TransitionNotAllowed:
                     print(f"SM Warning: PTT release not allowed in state {self.sm.current_state.id}")

    def start_audio_recording(self):
        if not self.running_flag: return
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
            # Transition SM back to idle or error state
            if self.sm.current_state == self.sm.listening_command:
                self.sm.send("event_stt_failed_command") # Or a dedicated audio_error event
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

    def speak(self, text_to_speak):
        if self.tts_engine and self.running_flag:
            print(f"TTS: Saying: '{text_to_speak}'")
            try:
                self.tts_engine.say(text_to_speak)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        elif not self.tts_engine:
            print(f"TTS (Disabled): '{text_to_speak}'")


    def process_command_stt(self, audio_data_bytes):
        if not audio_data_bytes:
            return None, False, False # text, needs_confirmation, is_stop_command

        print("STT: Processing command...")
        transcribed_text = self.stt_system.recognize_buffer(audio_data_bytes)
        print(f"STT Result (Command): '{transcribed_text}'")

        if not transcribed_text:
            return None, False, False

        if "stop listening" in transcribed_text.lower() or \
           "shutdown system" in transcribed_text.lower():
            return transcribed_text, False, True # Stop command detected

        self.original_nlu_result_for_execution = self.nlu_system.parse_command(transcribed_text)
        self.command_to_execute_display_text = self.original_nlu_result_for_execution.get("text", transcribed_text)

        # For now, always confirm commands that are not "stop listening"
        needs_confirmation = True
        return self.command_to_execute_display_text, needs_confirmation, False

    def process_yes_no_stt(self, audio_data_bytes):
        if not audio_data_bytes:
            return "unrecognized"

        print("STT: Processing yes/no...")
        text_result = self.stt_system.recognize_buffer(audio_data_bytes).lower()
        print(f"STT Result (Yes/No): '{text_result}'")

        if "yes" in text_result and "no" not in text_result:
            return "yes"
        elif "no" in text_result and "yes" not in text_result:
            return "no"
        else:
            # Check for common misrecognitions or variations if needed
            if "yeah" in text_result or "yep" in text_result: return "yes"
            if "nope" in text_result: return "no"
            return "unrecognized"

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


        if intent == "UNKNOWN_INTENT" or confidence < NLU_CONFIDENCE_THRESHOLD:
            self.speak("Command not understood clearly or intent is unknown. Action cancelled.")
            return

        parsed_gps_coords = None
        if "latitude" in entities and "longitude" in entities:
            parsed_gps_coords = {"latitude": entities["latitude"], "longitude": entities["longitude"]}

        if intent in ("COMBINED_SEARCH_AND_TARGET", "REQUEST_SEARCH_AT_LOCATION") and parsed_gps_coords:
            lat = parsed_gps_coords.get("latitude")
            lon = parsed_gps_coords.get("longitude")
            alt = DEFAULT_WAYPOINT_ALTITUDE # Use default altitude

            if lat is not None and lon is not None:
                action_summary = f"Fly to Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt}m."
                target_desc = entities.get("target_description_full")
                if target_desc:
                    action_summary += f" Search for: {target_desc}."

                self.speak(f"Executing: {action_summary}")

                if self.mav_controller.is_connected():
                    print("MAVLink: (Simplified) Sending single waypoint. Ensure vehicle is in GUIDED mode and ARMED.")
                    success = self.mav_controller.send_nav_waypoint(lat, lon, alt)
                    if success:
                        self.speak("Waypoint sent to vehicle.")
                    else:
                        self.speak("Failed to send waypoint to vehicle.")
                else:
                    self.speak("MAVLink not connected. Cannot send command to vehicle.")
            else:
                self.speak("GPS coordinates incomplete in command. Action cancelled.")
        elif intent == "PROVIDE_TARGET_DESCRIPTION":
            target_desc = entities.get("target_description_full", "the described target")
            self.speak(f"Target description noted: {target_desc}. This information can be used by onboard perception systems.")
        else:
            self.speak("Command understood, but no actionable MAVLink operation for this intent or missing GPS. Action cancelled.")


    def initiate_shutdown(self, reason=""):
        if self.running_flag:
            print(f"Shutdown initiated: {reason}")
            self.running_flag = False
            self.speak("System shutting down.")


    def run(self):
        if not self.running_flag: # Initialization failed
            print("Application cannot start due to initialization errors.")
            self.cleanup()
            return

        self._setup_keyboard_listener()
        self.speak("System initialized.")
        # The initial print from on_enter_idle will provide PTT instructions.
        # Ensure SM starts in idle to trigger that.
        if self.sm.current_state!= self.sm.idle: # Should be idle by default
            try:
                self.sm.send("event_shutdown_requested") # Force to idle if not
            except Exception as e:
                print(f"Error forcing SM to idle: {e}")


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
                    # Attempt to gracefully stop current PTT action
                    if self.is_space_currently_pressed: # If key still seems pressed
                        self.is_space_currently_pressed = False # Force release logic
                        if self.sm.current_state == self.sm.listening_command:
                            self.sm.event_ptt_stop_command()
                        elif self.sm.current_state == self.sm.listening_yes_no:
                            self.sm.event_ptt_stop_yes_no()
            time.sleep(0.01) # Keep main thread responsive, prevent busy-waiting

        print("Exiting main application loop.")
        self.cleanup()


    def cleanup(self):
        print("Cleaning up resources...")
        if self.keyboard_listener and self.keyboard_listener.is_alive():
            self.keyboard_listener.stop()
            # self.keyboard_listener.join() # Can block if not stopped cleanly

        if self.current_audio_stream:
            try:
                if self.current_audio_stream.is_active():
                    self.current_audio_stream.stop_stream()
                self.current_audio_stream.close()
            except Exception as e: print(f"Error closing active audio stream: {e}")
        self.current_audio_stream = None

        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            print("PyAudio terminated.")

        if self.mav_controller:
            self.mav_controller.close_connection()

        # TTS engine doesn't have an explicit close, relies on Python's GC.
        print("System shutdown complete.")


if __name__ == "__main__":
    # Suppress Vosk logging unless needed for debugging
    # vosk.SetLogLevel(-1) # Or a higher level for less verbosity

    app = LifeguardApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by Ctrl+C.")
        app.initiate_shutdown("KeyboardInterrupt")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        app.initiate_shutdown(f"Unexpected error: {e}")
    finally:
        # Ensure cleanup is called if run() exits normally or via unhandled exception
        # (though initiate_shutdown should lead to cleanup via run loop ending)
        if app.running_flag: # If shutdown wasn't called
             app.cleanup()