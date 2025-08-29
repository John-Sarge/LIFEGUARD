# LIFEGUARD: Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments

# LIFEGUARD enables operators to naturally and effectively pass Command Intent to autonomous units,
# bridging voice commands and machine action for efficient, reliable, and intuitive search and rescue deployments.

# This file implements a decoupled, message-passing architecture for a search-and-rescue drone system.
# Subsystems (Audio Input/PTT, STT, NLU, MAVLink I/O, TTS Output) run in dedicated threads and communicate via thread-safe queues.
# This design improves reliability, thread safety, and responsiveness for mission-critical operations.

import os
import json
import re
import time
import math
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

# Audio Processing Libraries: Used for capturing, filtering, and denoising microphone input.
import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr

# Speech-to-Text (STT) Library: Vosk for offline speech recognition.
import vosk

# Natural Language Understanding (NLU) Libraries: spaCy for intent/entity extraction from transcribed speech.
import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.pipeline import EntityRuler

# External Libraries: Used for number parsing and keyboard input handling.
from word2number import w2n
from pynput import keyboard

# MAVLink: Communication protocol for drone control and mission upload.
from pymavlink import mavutil, mavwp

# =========================
# ===== CONFIGURATION =====
# =========================
# All system-wide constants and configuration values are defined here.

# STT Configuration: Vosk model and audio input settings.
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_PATH = os.path.join("vosk_models", VOSK_MODEL_NAME)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_FRAMES_PER_BUFFER = 8192
AUDIO_READ_CHUNK = 4096

# Audio Pre-processing: Parameters for noise reduction and bandpass filtering.
NOISE_REDUCE_TIME_CONSTANT_S = 2.0
NOISE_REDUCE_PROP_DECREASE = 0.9
BANDPASS_LOWCUT_HZ = 300.0
BANDPASS_HIGHCUT_HZ = 3400.0
BANDPASS_ORDER = 5

# NLU Configuration: spaCy model and intent confidence threshold.
SPACY_MODEL_NAME = "en_core_web_sm"
NLU_CONFIDENCE_THRESHOLD = 0.7

# MAVLink Configuration: Serial/network connection settings and default mission parameters.
MAVLINK_BAUDRATE = 115200
MAVLINK_SOURCE_SYSTEM_ID = 255
DEFAULT_WAYPOINT_ALTITUDE = 30.0
DEFAULT_SWATH_WIDTH_M = 20.0

# Agents: Dictionary of available drone agents and their connection details.
AGENT_CONNECTION_CONFIGS = {
    "agent1": {
        "connection_string": 'tcp:10.24.5.232:5762',
        "source_system_id": MAVLINK_SOURCE_SYSTEM_ID,
        "baudrate": MAVLINK_BAUDRATE
    },
    # Placeholder for a second verification agent. Update connection_string to your agent2 endpoint.
    # Example: SITL instance 2 often uses the next TCP port (5763)
    "agent2": {
        "connection_string": 'tcp:10.24.5.232:5772',  # TODO: confirm/update
        "source_system_id": MAVLINK_SOURCE_SYSTEM_ID,
        "baudrate": MAVLINK_BAUDRATE
    }
}
INITIAL_ACTIVE_AGENT_ID = "agent1"

# Earth radius (m): Used for waypoint calculations.
EARTH_RADIUS_METERS = 6378137.0

# =========================
# ====== UTILITIES ========
# =========================
# Utility functions for audio processing and text normalization.

def apply_noise_reduction(audio_data_int16, sample_rate):
    # Apply noise reduction to audio data using noisereduce library.
    if audio_data_int16 is None:
        return audio_data_int16
    audio_data_float = audio_data_int16.astype(np.float32) / 32767.0
    # Replace NaN/Inf and guard against empty or near-silent buffers to avoid errors in processing.
    audio_data_float = np.nan_to_num(audio_data_float, nan=0.0, posinf=0.0, neginf=0.0)
    if audio_data_float.size == 0:
        return audio_data_int16
    # Simple energy check: skip noise reduction for extremely low-energy frames (likely silence).
    energy = float(np.mean(np.abs(audio_data_float)))
    if energy < 1e-4:
        # nothing to denoise; return original
        return audio_data_int16
    # Call noisereduce while suppressing runtime warnings from internal divide-by-zero.
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            reduced_audio_float = nr.reduce_noise(
                y=audio_data_float,
                sr=sample_rate,
                time_constant_s=NOISE_REDUCE_TIME_CONSTANT_S,
                prop_decrease=NOISE_REDUCE_PROP_DECREASE,
                n_fft=512,
            )
    except Exception:
        # Fall back to the input if noise reduction fails for any reason.
        reduced_audio_float = audio_data_float
    reduced_audio_int16 = (reduced_audio_float * 32767.0).astype(np.int16)
    return reduced_audio_int16

def apply_bandpass_filter(audio_data, sample_rate):
    # Apply a Butterworth bandpass filter to audio data to isolate speech frequencies.
    nyq = 0.5 * sample_rate
    low = BANDPASS_LOWCUT_HZ / nyq
    high = BANDPASS_HIGHCUT_HZ / nyq
    b, a = butter(BANDPASS_ORDER, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio_data)
    return filtered_audio.astype(audio_data.dtype)

def spoken_numbers_to_digits(text):
    # Convert spoken numbers in text (e.g., "one hundred") to digit form (e.g., "100").
    text = re.sub(r'\bnative\b', 'negative', text, flags=re.IGNORECASE)
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

def extract_altitude_from_text(text):
    """
    Extracts an altitude value from a string, handling both digits and number words.
    Returns the altitude as an integer, or None if not found.
    Handles cases like 'altitude 50 to 100 feet' by returning the largest number.
    Note: Unit indicators such as 'feet', 'meters', etc. are ignored and not parsed or converted.
    """
    # Normalize text to lower case
    text = text.lower()
    # Try to find a number (digit or word) after 'altitude' in the text
    NUMBER_WORDS = (
        "zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|point|and"
    )
    number_word_pattern = rf'(?:{NUMBER_WORDS}| |-)+'
    regex = rf'altitude\s*(to|is|at)?\s*((?:\d+(?:\s*-\s*\d+)?|{number_word_pattern}))'
    match = re.search(regex, text)
    numbers = []
    if match:
        candidate = match.group(2).strip()
        # Find all digit numbers
        digit_matches = re.findall(r'\d+', candidate)
        for d in digit_matches:
            try:
                numbers.append(int(d))
            except ValueError:
                pass
        # Try to extract number words (split on 'to', 'and', etc.)
        parts = re.split(r'\bto\b|\band\b|,', candidate)
        for part in parts:
            part = part.strip()
            if part:
                try:
                    num = w2n.word_to_num(part)
                    numbers.append(num)
                except ValueError:
                    pass
        if numbers:
            return max(numbers)
    # Fallback: look for any number in text
    num_matches = re.findall(r'\d+', text)
    if num_matches:
        return max(int(n) for n in num_matches)
    return None

# =========================
# ===== STT COMPONENT =====
# =========================
# Speech-to-Text subsystem: Handles conversion of audio buffers to text using Vosk.

class SpeechToText:
    # Handles both batch and streaming speech recognition using Vosk.
    def __init__(self, model_path, sample_rate):
        self.sample_rate = sample_rate
        if not os.path.exists(model_path):
            print(f"ERROR: Vosk model not found at {model_path}.")
            raise FileNotFoundError(f"Vosk model not found at {model_path}")
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        # Streaming recognizer instance (created per utterance)
        self._streaming_recognizer = None

    def recognize_buffer(self, raw_audio_buffer_bytes) -> str:
        # Recognize speech from a complete audio buffer (non-streaming).
        if not raw_audio_buffer_bytes:
            self.recognizer.Reset()
            return ""
        audio_data_np = np.frombuffer(raw_audio_buffer_bytes, dtype=np.int16)
        denoised_audio_np = apply_noise_reduction(audio_data_np, self.sample_rate)
        filtered_audio_np = apply_bandpass_filter(denoised_audio_np, self.sample_rate)
        processed_audio_bytes = filtered_audio_np.tobytes()
        if self.recognizer.AcceptWaveform(processed_audio_bytes):
            result_json = self.recognizer.Result()
        else:
            result_json = self.recognizer.FinalResult()
        self.recognizer.Reset()
        try:
            result_dict = json.loads(result_json)
            return result_dict.get('text', '')
        except Exception:
            return ""

    # Streaming API: create new recognizer state for a new utterance
    def start_utterance(self):
        # Start a new streaming utterance (for chunked audio input).
        self._streaming_recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self._streaming_recognizer.SetWords(True)

    def accept_waveform(self, audio_chunk_bytes) -> Optional[str]:
        """Feed a chunk to the streaming recognizer. Returns partial text if available, otherwise None."""
        if not self._streaming_recognizer:
            self.start_utterance()
        try:
            # Preprocess chunk (convert to int16 numpy, denoise and bandpass)
            try:
                audio_np = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
                denoised = apply_noise_reduction(audio_np, self.sample_rate)
                filtered = apply_bandpass_filter(denoised, self.sample_rate)
                processed_bytes = filtered.tobytes()
            except Exception:
                # Fallback: use raw bytes if preprocessing fails
                processed_bytes = audio_chunk_bytes

            if self._streaming_recognizer.AcceptWaveform(processed_bytes):
                res = self._streaming_recognizer.Result()
            else:
                res = self._streaming_recognizer.PartialResult()
            try:
                d = json.loads(res)
                # For partial results, 'partial' may be present; for final results 'text' is present
                return d.get('text', d.get('partial', ''))
            except Exception:
                return None
        except Exception:
            return None

    def end_utterance_and_get_result(self) -> str:
        # Finalize streaming recognition and return the final result.
        if not self._streaming_recognizer:
            return ""
        try:
            res = self._streaming_recognizer.FinalResult()
            self._streaming_recognizer = None
            try:
                d = json.loads(res)
                return d.get('text', '')
            except Exception:
                return ""
        except Exception:
            self._streaming_recognizer = None
            return ""

    def close(self):
        # Explicitly release Vosk recognizer and model references to avoid interpreter-shutdown destructor warnings.
        try:
            self._streaming_recognizer = None
        except Exception:
            pass
        try:
            self.recognizer = None
        except Exception:
            pass
        try:
            self.model = None
        except Exception:
            pass
        # encourage immediate garbage collection
        try:
            import gc
            gc.collect()
        except Exception:
            pass

# =========================
# ===== NLU COMPONENT =====
# =========================
# Natural Language Understanding subsystem: Extracts intent and entities from transcribed text.

# spaCy custom entity recognizer for latitude/longitude patterns in text.
@Language.factory("lat_lon_entity_recognizer", default_config={"gps_label": "LOCATION_GPS_COMPLEX"})
def create_lat_lon_entity_recognizer(nlp: Language, name: str, gps_label: str):
    return LatLonEntityRecognizer(nlp, gps_label)

class LatLonEntityRecognizer:
    # Recognizes and parses latitude/longitude entities from text using regex.
    def __init__(self, nlp: Language, gps_label: str):
        self.gps_label = gps_label
        self.gps_regex = re.compile(
            r"""
            (?:
                latitude\s*([+-]?\d{1,3}(?:\.\d+)?)\s*,\s*longitude\s*([+-]?\d{1,3}(?:\.\d+)?)
                |
                (?:[Ll][Aa](?:itude)?\s*[:\s]*)?
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*?)\s*
                (?:[,;\s]+\s*)?
                (?:[Ll][Oo][Nn](?:gitude)?\s*[:\s]*)?
                ([+-]?\d{1,3}(?:\.\d+)?\s*(?:°|d|degrees)?\s*(?:\d{1,2}(?:\.\d+)?\s*['m|minutes])?\s*(?:\d{1,2}(?:\.\d+)?\s*["s|seconds])?\s*[EeWw]?)
                |
                ([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)
            )
            """, re.VERBOSE
        )
        if not Span.has_extension("parsed_gps_coords"):
            Span.set_extension("parsed_gps_coords", default=None)

    def __call__(self, doc: Doc) -> Doc:
        new_entities = list(doc.ents)
        for match in self.gps_regex.finditer(doc.text):
            start_char, end_char = match.span()
            lat_val, lon_val = None, None
            try:
                if match.group(1) and match.group(2):
                    lat_val = float(match.group(1)); lon_val = float(match.group(2))
                elif match.group(3) and match.group(4):
                    lat_val = float(match.group(3)); lon_val = float(match.group(4))
                elif match.group(5) and match.group(6):
                    lat_val = float(match.group(5)); lon_val = float(match.group(6))
                else:
                    continue
                if not (-90 <= lat_val <= 90 and -180 <= lon_val <= 180):
                    continue
                span = doc.char_span(start_char, end_char, label=self.gps_label, alignment_mode="expand")
                if span is None:
                    contract_span = doc.char_span(start_char, end_char, alignment_mode="contract")
                    if contract_span is not None:
                        span = Span(doc, contract_span.start, contract_span.end, label=self.gps_label)
                    else:
                        token_start, token_end = None, None
                        for i, token in enumerate(doc):
                            if token.idx <= start_char < token.idx + len(token): token_start = i
                            if token.idx < end_char <= token.idx + len(token): token_end = i + 1
                        if token_start is None: token_start = 0
                        if token_end is None: token_end = len(doc)
                        if token_start >= token_end: token_start, token_end = 0, len(doc)
                        span = Span(doc, token_start, token_end, label=self.gps_label)
                if span is None: continue
                span._.set("parsed_gps_coords", {"latitude": lat_val, "longitude": lon_val})
                temp = []
                for ent in new_entities:
                    if span.end <= ent.start or span.start >= ent.end:
                        temp.append(ent)
                temp.append(span)
                new_entities = temp
            except Exception:
                continue
        doc.ents = tuple(sorted(list(set(new_entities)), key=lambda e: e.start_char))
        return doc

class NaturalLanguageUnderstanding:
    # Loads spaCy, adds custom entity rules, and parses commands into intent/entities.
    def __init__(self, spacy_model_name):
        try:
            self.nlp = spacy.load(spacy_model_name)
        except OSError:
            print(f"spaCy model '{spacy_model_name}' not found. Install it: python -m spacy download {spacy_model_name}")
            raise
        if "lat_lon_entity_recognizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("lat_lon_entity_recognizer", after="ner" if self.nlp.has_pipe("ner") else None)
        if "sar_entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", name="sar_entity_ruler", before="lat_lon_entity_recognizer")
            sar_patterns = [
                    # Altitude set patterns
                {"label": "ALTITUDE_SET", "pattern": [{"LEMMA": {"IN": ["set", "change", "update"]}}, {"LOWER": "altitude"}, {"LIKE_NUM": True}]},
                {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"LEMMA": {"IN": ["set", "change", "update"]}}, {"LIKE_NUM": True}]},
                {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"LIKE_NUM": True}]},
                {"label": "TARGET_OBJECT", "pattern": [{"LOWER": "person"}]},
                {"label": "TARGET_OBJECT", "pattern": [{"LOWER": "boat"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}, {"LOWER": "grid"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["meter", "meters", "m"]}}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": "meter"}, {"LOWER": "grid"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LIKE_NUM": True}, {"LOWER": "meters"}, {"LOWER": "grid"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}, {"LOWER": "meter"}]},
                {"label": "GRID_SIZE_METERS", "pattern": [{"LOWER": "grid"}, {"LIKE_NUM": True}, {"LOWER": "meters"}]},
                {"label": "AGENT_ID_NUM", "pattern": [{"LOWER": {"IN": ["drone", "agent"]}}, {"IS_DIGIT": True}]},
                {"label": "AGENT_ID_TEXT", "pattern": [{"LOWER": {"IN": ["drone", "agent"]}}, {"IS_ALPHA": True}]},
                {"label": "ACTION_SELECT", "pattern": [{"LEMMA": {"IN": ["select", "target", "choose", "activate"]}}]},
                {"label": "ACTION_VERB", "pattern": [{"LEMMA": "search"}]},
            ]
            # Add more robust patterns for altitude commands
            sar_patterns.extend([
                {"label": "ALTITUDE_SET", "pattern": [{"LEMMA": {"IN": ["set", "change", "update", "adjust", "make", "put"]}}, {"LOWER": "altitude"}, {"IS_ALPHA": True}]},
                {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"LEMMA": {"IN": ["set", "change", "update", "adjust", "make", "put"]}}, {"IS_ALPHA": True}]},
                {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"IS_ALPHA": True}]},
                {"label": "ALTITUDE_SET", "pattern": [{"LEMMA": {"IN": ["set", "change", "update", "adjust", "make", "put"]}}, {"LOWER": "altitude"}, {"LOWER": "to"}, {"IS_ALPHA": True}]},
                {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"LOWER": "to"}, {"IS_ALPHA": True}]},
            ])
            ruler.add_patterns(sar_patterns)

    def parse_command(self, text):
    # Parse a command string and extract intent, confidence, and entities.
    # Heuristics:
    # - Use spaCy (EntityRuler + custom GPS) to tag entities.
    # - Backfill grid size/altitude via regex/word-number parsing when missing.
    # - Favor explicit altitude phrases for SET_AGENT_ALTITUDE to avoid false positives.
        doc = self.nlp(text)
        intent = "UNKNOWN_INTENT"
        entities_payload = {}
        confidence = 0.5

        def spoken_agent_to_id(agent_text):
            agent_text = agent_text.lower().replace("drone", "").replace("agent", "").strip()
            # Fuzzy correction for common misrecognitions using a dictionary
            corrections = {
                "to": "two", "too": "two", "tu": "two", "tow": "two",
                "for": "four", "fore": "four", "fohr": "four", "fawr": "four"
            }
            if agent_text in corrections:
                agent_text = corrections[agent_text]
            # Try direct digit
            if agent_text.isdigit():
                return f"agent{agent_text}"
            # Try word2number
            try:
                num = w2n.word_to_num(agent_text)
                return f"agent{num}"
            except ValueError:
                pass
            return None
        extracted_spacy_ents = []
        for ent in doc.ents:
            entity_data = {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            if ent.label_ == "ALTITUDE_SET":
                try:
                    altitude_str = ent.text
                    entity_data["altitude_value"] = float(altitude_str) if '.' in altitude_str else int(altitude_str)
                except Exception:
                    pass
            if ent.label_ == "LOCATION_GPS_COMPLEX" and Span.has_extension("parsed_gps_coords"):
                parsed = getattr(ent._, "parsed_gps_coords", None)
                if parsed:
                    entity_data["parsed_gps"] = parsed
            elif ent.label_ == "GRID_SIZE_METERS":
                numeric_part = "".join(filter(str.isdigit, ent.text))
                if numeric_part:
                    entity_data["value"] = int(numeric_part)
            elif ent.label_ == "AGENT_ID_NUM":
                agent_id = spoken_agent_to_id(ent.text)
                if agent_id:
                    entity_data["value"] = agent_id
            elif ent.label_ == "AGENT_ID_TEXT":
                agent_id = spoken_agent_to_id(ent.text)
                if agent_id:
                    entity_data["value"] = agent_id
            extracted_spacy_ents.append(entity_data)
        entities_payload["raw_spacy_entities"] = extracted_spacy_ents

        def get_entity_values(label, value_key="text"):
            return [e[value_key] for e in extracted_spacy_ents if e["label"] == label and value_key in e]

        def get_parsed_gps():
            valid_gps = []
            for e in extracted_spacy_ents:
                if e["label"] == "LOCATION_GPS_COMPLEX" and "parsed_gps" in e:
                    lat = e["parsed_gps"].get("latitude")
                    lon = e["parsed_gps"].get("longitude")
                    if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                        valid_gps.append(e["parsed_gps"])
            if valid_gps:
                return max(valid_gps, key=lambda g: abs(g["latitude"]))
            return None

        action_verbs = get_entity_values("ACTION_VERB")
        action_select_verbs = get_entity_values("ACTION_SELECT")
        target_objects = get_entity_values("TARGET_OBJECT")
        parsed_gps_coords = get_parsed_gps()
        grid_sizes = [v for v in get_entity_values("GRID_SIZE_METERS", "value") if isinstance(v, int)]
        if not grid_sizes:
            grid_size_match = re.search(r'(\d+)\s*(?:meter|meters|m)\s*grid', text, re.IGNORECASE)
            if grid_size_match:
                grid_sizes = [int(grid_size_match.group(1))]
        agent_ids_num = get_entity_values("AGENT_ID_NUM", "value")
        agent_ids_text = get_entity_values("AGENT_ID_TEXT", "value")

        # Fallbacks: if the entity ruler missed simple forms, recover from raw text.
        # 1) Treat plain 'select' as ACTION_SELECT if present.
        if not action_select_verbs and re.search(r"\bselect\b", text, re.IGNORECASE):
            action_select_verbs = ["select"]
        # 2) Extract agent id from patterns like 'agent 1' or 'agent one' if not found in entities.
        if not agent_ids_num and not agent_ids_text:
            m = re.search(r"\b(?:drone|agent)\s+([A-Za-z]+|\d+)\b", text, re.IGNORECASE)
            if m:
                sid = spoken_agent_to_id(m.group(1))
                if sid:
                    agent_ids_num = [sid]

        altitude_values = [e["altitude_value"] for e in extracted_spacy_ents if e["label"] == "ALTITUDE_SET" and "altitude_value" in e]
        # If no altitude found in entities, try to extract from text (digits or number words)
        if not altitude_values:
            alt_from_text = extract_altitude_from_text(text)
            if alt_from_text is not None:
                altitude_values = [alt_from_text]
            else:
                # Try number words in altitude-related phrases only
                def extract_altitude_phrase(text):
                    # Look for phrases like "altitude <number>", "set altitude to <number>", etc.
                    patterns = [
                        # Match phrases like "altitude to <number>", "set altitude to <number>", etc.
                        r"(?:altitude|set altitude|change altitude|update altitude)\s*(?:to|at|is|of)?\s*([A-Za-z\s\-]+|\d+)\s*(?:meters|meter|m)?\b"
                    ]
                    for pat in patterns:
                        m = re.search(pat, text, re.IGNORECASE)
                        if m:
                            return m.group(1)
                    return None
                try:
                    phrase = extract_altitude_phrase(text)
                    if phrase:
                        num = w2n.word_to_num(phrase)
                        if num > 0:
                            altitude_values = [num]
                except Exception:
                    pass

        if parsed_gps_coords:
            entities_payload.update(parsed_gps_coords)
        if grid_sizes:
            entities_payload["grid_size_meters"] = grid_sizes[0]

        if altitude_values:
            entities_payload["altitude_meters"] = altitude_values[0]
        selected_agent_id = None
        if agent_ids_num: selected_agent_id = agent_ids_num[0]
        elif agent_ids_text: selected_agent_id = agent_ids_text[0]

        # Helper function to detect explicit altitude command in text
        def is_altitude_command(text, altitude_values):
            altitude_phrases = ["altitude", "set altitude", "change altitude", "update altitude"]
            return bool(altitude_values) and any(phrase in text.lower() for phrase in altitude_phrases)

        # Only trigger SELECT_AGENT if agent id is present and no altitude command is detected.
        if selected_agent_id and action_select_verbs and not is_altitude_command(text, altitude_values):
            intent = "SELECT_AGENT"
            entities_payload["selected_agent_id"] = selected_agent_id
            confidence = 0.9

        # Only trigger altitude intent if an explicit altitude command is present in the text.
        if is_altitude_command(text, altitude_values):
            intent = "SET_AGENT_ALTITUDE"
            entities_payload["altitude_meters"] = altitude_values[0]
            confidence = max(confidence, 0.95)

        target_details_parts = []
        if target_objects: target_details_parts.extend(target_objects)
        if target_details_parts:
            entities_payload["target_description_full"] = " ".join(target_details_parts)

        if ("grid" in text.lower()) and grid_sizes and parsed_gps_coords:
            intent = "REQUEST_GRID_SEARCH"
            entities_payload["grid_size_meters"] = grid_sizes[0]

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

        if intent not in ("UNKNOWN_INTENT",):
            if parsed_gps_coords or target_details_parts or intent == "SELECT_AGENT":
                confidence = max(confidence, 0.85)
            if intent == "REQUEST_GRID_SEARCH" and "grid_size_meters" in entities_payload and parsed_gps_coords:
                confidence = 0.95

        return {"text": text, "intent": intent, "confidence": confidence, "entities": entities_payload}

# =========================
# === MAVLINK COMPONENT ===
# =========================
# MAVLink subsystem: Handles drone connection, mission upload, and vehicle control.

class MavlinkController:
    # Controls a single drone via MAVLink, including mission upload and mode changes.
    MAX_POSITION_RETRIES = 5  # Reduced number of attempts to receive current position
    POSITION_RETRY_TIMEOUT = 1.0  # Increased timeout per attempt (seconds)
    def __init__(self, connection_string, baudrate=None, source_system_id=255):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.source_system_id = source_system_id
        self.master = None
        self._connect()

    def _connect(self):
        # Establish connection to drone and wait for heartbeat.
        print(f"MAVLink: Connecting to {self.connection_string} src_sys={self.source_system_id}...")
        try:
            if "com" in self.connection_string.lower() or "/dev/tty" in self.connection_string.lower():
                self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate, source_system=self.source_system_id)
            else:
                self.master = mavutil.mavlink_connection(
                    self.connection_string,
                    source_system=self.source_system_id,
                    dialect="ardupilotmega",
                    autoreconnect=True,
                    use_native=False,
                    force_connected=True,
                    mavlink2=True
                )
            print(f"MAVLink: Waiting for heartbeat (timeout 10s) on {self.connection_string}...")
            self.master.wait_heartbeat(timeout=10)
            if self.master.target_system == 0 and self.master.target_component == 0:
                print("MAVLink: Invalid heartbeat (system=0).")
                self.master = None
            elif self.master.target_system != 0:
                print(f"MAVLink: Heartbeat from system {self.master.target_system} component {self.master.target_component}.")
            else:
                print(f"MAVLink: wait_heartbeat returned without valid target system ({self.master.target_system}).")
                self.master = None
            if self.master:
                print(f"MAVLink: Using target_system={self.master.target_system}, target_component={self.master.target_component}")
                self.master.target_component = 1
        except Exception as e:
            print(f"MAVLink: EXCEPTION connect/wait_heartbeat on {self.connection_string}: {e}")
            self.master = None

    def is_connected(self):
        return self.master is not None

    def upload_mission(self, waypoints_data):
    # Upload a list of waypoints as a mission to the drone.
    # Accepts a sequence of tuples shaped like (lat, lon, alt, cmd, p1, p2, p3, p4 [, frame, is_current, autocontinue]).
    # A home/dummy item is inserted at seq=0 to satisfy mission protocols that expect a home entry.
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot upload mission.")
            return False
        self.master.target_component = 1
        wp_loader = mavwp.MAVWPLoader()
        seq = 0
        home_lat, home_lon, home_alt = waypoints_data[0][0], waypoints_data[0][1], 0
        home_item = mavutil.mavlink.MAVLink_mission_item_int_message(
            self.master.target_system, 1, seq, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0,
            int(home_lat * 1e7), int(home_lon * 1e7), float(home_alt)
        )
        wp_loader.add(home_item)
        seq += 1
        for item in waypoints_data:
            if len(item) == 8:
                lat, lon, alt, command_id, p1, p2, p3, p4 = item
                frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
                is_current = 0
                autocontinue = 1
            elif len(item) == 11:
                lat, lon, alt, command_id, p1, p2, p3, p4, frame, is_current, autocontinue = item
            else:
                continue
            wp_loader.add(
                mavutil.mavlink.MAVLink_mission_item_int_message(
                    self.master.target_system, 1, seq, frame, command_id, is_current, autocontinue,
                    p1, p2, p3, p4, int(lat * 1e7), int(lon * 1e7), float(alt)
                )
            )
            seq += 1
        if wp_loader.count() == 0:
            print("MAVLink: No valid waypoints to upload.")
            return False
        try:
            self.master.mav.mission_clear_all_send(self.master.target_system, 1)
            self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            time.sleep(1)
            self.master.mav.mission_count_send(
                self.master.target_system, 1, wp_loader.count(), mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            )
            for i in range(wp_loader.count()):
                msg = self.master.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=5)
                if not msg:
                    print(f"MAVLink: No MISSION_REQUEST for wp {i}")
                    return False
                self.master.mav.send(wp_loader.wp(msg.seq))
                if msg.seq == wp_loader.count() - 1:
                    break
            final_ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
            if final_ack and final_ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("MAVLink: Mission upload successful.")
                return True
            print(f"MAVLink: Mission upload failed or bad ACK: {final_ack}")
            return False
        except Exception as e:
            print(f"MAVLink: Mission upload error: {e}")
            return False

    def set_mode(self, mode_name):
        # Set the drone's flight mode (e.g., AUTO, GUIDED).
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot set mode.")
            return False
        mode_mapping = self.master.mode_mapping()
        if mode_mapping is None or mode_name.upper() not in mode_mapping:
            print(f"MAVLink: Unknown mode: {mode_name}")
            return False
        mode_id = mode_mapping[mode_name.upper()]
        try:
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack_msg and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"MAVLink: Mode {mode_name} accepted.")
                return True
            print(f"MAVLink: Mode change failed or no ACK: {ack_msg}")
            return False
        except Exception as e:
            print(f"MAVLink: Error setting mode: {e}")
            return False

    def arm_vehicle(self):
        # Arm the drone for flight.
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot arm.")
            return False
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Armed.")
            return True
        print(f"MAVLink: Arming failed: {ack_msg}")
        return False

    def start_mission(self):
        # Start the uploaded mission on the drone.
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot start mission.")
            return False
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0
        )
        ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_MISSION_START and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("MAVLink: Mission start accepted.")
            return True
        print(f"MAVLink: Mission start failed: {ack_msg}")
        return False

    def wait_for_waypoint_reached(self, waypoint_sequence_id, timeout_seconds=120):
        # Wait until the drone reaches a specific waypoint.
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot wait for waypoint.")
            return False
        start = time.time()
        while time.time() - start < timeout_seconds:
            msg = self.master.recv_match(type='MISSION_ITEM_REACHED', blocking=True, timeout=1)
            if msg and msg.seq == waypoint_sequence_id:
                return True
        return False

    def send_status_text(self, text_to_send, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
        # Send a status text message to the drone (for operator feedback).
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot send STATUSTEXT.")
            return False
        text_bytes = text_to_send.encode('utf-8') if isinstance(text_to_send, str) else text_to_send
        payload_text_bytes = text_bytes[:49]
        try:
            self.master.mav.statustext_send(severity, payload_text_bytes)
            return True
        except Exception as e:
            print(f"MAVLink: STATUSTEXT error: {e}")
            return False

    def close_connection(self):
        # Close the MAVLink connection to the drone.
        if self.master:
            self.master.close()
            print(f"MAVLink: Connection closed for {self.connection_string}.")

    def _calculate_rectangular_grid_waypoints(self, center_lat, center_lon, grid_width_m, grid_height_m, swath_width_m, altitude_m):
    # Generate a rectangular grid of waypoints for a lawnmower search pattern.
    # Small-area approximation: convert meter offsets to lat/lon deltas using a spherical Earth.
    # Tracks run N-S and alternate direction each pass to reduce turn time.
        waypoints = []
        if swath_width_m <= 0 or grid_width_m <= 0 or grid_height_m <= 0:
            return waypoints
        center_lat_rad = math.radians(center_lat)
        center_lon_rad = math.radians(center_lon)
        half_width_m = grid_width_m / 2.0
        half_height_m = grid_height_m / 2.0
        lat_delta_rad = half_height_m / EARTH_RADIUS_METERS
        south_lat_rad = center_lat_rad - lat_delta_rad
        north_lat_rad = center_lat_rad + lat_delta_rad
        lon_delta_rad = half_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))
        west_lon_rad = center_lon_rad - lon_delta_rad
        num_tracks = int(math.floor(grid_width_m / swath_width_m))
        if num_tracks == 0 and grid_width_m > 0: num_tracks = 1
        if num_tracks == 0: return waypoints
        swath_lon_rad_step = swath_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))
        for i in range(num_tracks):
            current_track_lon_rad = west_lon_rad + i * swath_lon_rad_step
            current_track_lon_deg = math.degrees(current_track_lon_rad)
            south_lat_deg = math.degrees(south_lat_rad)
            north_lat_deg = math.degrees(north_lat_rad)
            if i % 2 == 0:
                waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
                waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
            else:
                waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
                waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
        return waypoints

    def generate_and_upload_search_grid_mission(self, center_lat, center_lon, grid_size_m, swath_width_m, altitude_m):
        # Generate a grid search mission and upload it to the drone.
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot generate/upload search grid.")
            return False
        grid_waypoints_tuples = self._calculate_rectangular_grid_waypoints(
            center_lat, center_lon, grid_size_m, grid_size_m, swath_width_m, altitude_m
        )
        if not grid_waypoints_tuples:
            print("MAVLink: Failed to generate grid waypoints.")
            return False
        waypoints_data_for_upload = []
        for lat, lon, alt in grid_waypoints_tuples:
            waypoints_data_for_upload.append(
                (lat, lon, alt, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 10.0, 0.0, float('nan'))
            )
        return self.upload_mission(waypoints_data_for_upload)

    # Timeout (in seconds) to wait for COMMAND_ACK after sending DO_CHANGE_ALTITUDE
    ACK_TIMEOUT_SECONDS = 3

    def set_altitude(self, altitude_m):
        # Change the current altitude of the drone.
        if not self.is_connected():
            print("MAVLink: Not connected. Cannot set altitude.")
            return False

        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_CHANGE_ALTITUDE,
                0,
                float(altitude_m), 0, 0, 0, 0, 0, 0
            )
            # Wait for the correct ACK for DO_CHANGE_ALTITUDE, ignoring unrelated ACKs until timeout
            start_time = time.time()
            ack = None
            while time.time() - start_time < self.ACK_TIMEOUT_SECONDS:
                remaining_time = self.ACK_TIMEOUT_SECONDS - (time.time() - start_time)
                # Wait up to 0.5 second or the remaining time, whichever is smaller
                timeout = min(0.5, remaining_time)
                ack_candidate = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=timeout)
                if ack_candidate and hasattr(ack_candidate, 'command'):
                    if ack_candidate.command == mavutil.mavlink.MAV_CMD_DO_CHANGE_ALTITUDE:
                        ack = ack_candidate
                        break
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"MAVLink: Altitude change to {altitude_m} accepted via DO_CHANGE_ALTITUDE.")
                return True
            print(f"MAVLink: DO_CHANGE_ALTITUDE not accepted or no ACK: {ack}. Falling back to position target...")
        except Exception as e:
            print(f"MAVLink: Error issuing DO_CHANGE_ALTITUDE: {e}. Falling back to position target...")

        print("MAVLink: Setting mode to GUIDED...")
        if not self.set_mode("GUIDED"):
            print("MAVLink: Failed to set GUIDED mode. Cannot change altitude.")
            return False

        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                200000,  # Message interval in microseconds (200ms) for GLOBAL_POSITION_INT messages
                0, 0, 0, 0, 0
            )
        except Exception as e:
            print(f"MAVLink: Error setting message interval: {e}")

        print("MAVLink: Waiting for current position...")
        current_pos = None
        # Try up to MAX_POSITION_RETRIES times to receive the current position before falling back to alternative retrieval.
        for _ in range(self.MAX_POSITION_RETRIES):
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=self.POSITION_RETRY_TIMEOUT)
            if msg:
                current_pos = msg
                break
        if not current_pos:
            try:
                current_pos = self.master.messages.get('GLOBAL_POSITION_INT')
            except (AttributeError, TypeError):
                current_pos = None
        if not current_pos:
            print("MAVLink: Could not get current position. Altitude change aborted.")
            return False

        print(f"MAVLink: Commanding altitude change to {altitude_m}m...")
        try:
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
            self.master.mav.set_position_target_global_int_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                type_mask,
                int(current_pos.lat),
                int(current_pos.lon),
                float(altitude_m),
                0, 0, 0,
                0, 0, 0,
                0, 0
            )
            print(f"MAVLink: Altitude command sent. Drone should be moving to {altitude_m}m.")
            return True
        except Exception as e:
            print(f"MAVLink: Error sending set_position_target_global_int: {e}")
            return False
            
# =========================
# ====== MESSAGES =========
# =========================
# Message dataclasses for inter-thread communication.

class CaptureMode(str, Enum):
    COMMAND = "command"
    YESNO = "yesno"

@dataclass
class MsgBase:
    pass

@dataclass
class MsgShutdown(MsgBase):
    reason: str = ""

@dataclass
class MsgSystemShutdownRequested(MsgBase):
    reason: str = ""

@dataclass
class MsgSetCaptureMode(MsgBase):
    mode: CaptureMode

@dataclass
class MsgAudioData(MsgBase):
    data: bytes
    mode: CaptureMode
    final: bool = False

@dataclass
class MsgSTTResult(MsgBase):
    text: str
    mode: CaptureMode

@dataclass
class MsgNLURequest(MsgBase):
    text: str

@dataclass
class MsgNLUResult(MsgBase):
    text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]

@dataclass
class MsgSpeak(MsgBase):
    text: str

@dataclass
class MsgSelectAgent(MsgBase):
    agent_id: str

@dataclass
class MsgCommandGridSearch(MsgBase):
    lat: float
    lon: float
    grid_size_m: int
    swath_m: float = DEFAULT_SWATH_WIDTH_M
    altitude_m: float = DEFAULT_WAYPOINT_ALTITUDE
    target_desc: Optional[str] = None

@dataclass
class MsgCommandFlyTo(MsgBase):
    lat: float
    lon: float
    altitude_m: float = DEFAULT_WAYPOINT_ALTITUDE
    target_desc: Optional[str] = None

# New message for altitude change
@dataclass
class MsgCommandSetAltitude(MsgBase):
    altitude_m: float

@dataclass
class MsgMavStatus(MsgBase):
    text: str

# =========================
# ===== WORKER BASE =======
# =========================

class WorkerThread:
    # Base class for all worker threads. Handles thread lifecycle and message processing.
    def __init__(self, name: str, inbox: queue.Queue):
        self.name = name
        self.inbox = inbox
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_wrapper, name=name, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.inbox.put(MsgShutdown(reason=f"Stop requested for {self.name}"))

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)

    def stopped(self):
        return self._stop_event.is_set()

    def _run_wrapper(self):
        try:
            self.run()
        except Exception as e:
            print(f"[{self.name}] Unhandled exception: {e}")
            import traceback; traceback.print_exc()

    def run(self):
        # Subclasses must implement this method to process messages.
        raise NotImplementedError

# =========================
# ==== TTS WORKER =========
# =========================

class TTSWorker(WorkerThread):
    # Text-to-Speech worker: speaks messages using pyttsx3 (falls back to console logs if unavailable).
    def __init__(self, inbox: queue.Queue):
        super().__init__("TTSWorker", inbox)
        self._engine = None

    def run(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            print("[TTS] Engine initialized.")
        except Exception as e:
            print(f"[TTS] Initialization failed: {e}")
            self._engine = None

        while not self.stopped():
            try:
                msg = self.inbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgSpeak):
                if self._engine:
                    try:
                        self._engine.say(msg.text)
                        self._engine.runAndWait()
                    except Exception as e:
                        print(f"[TTS] Error speaking: {e}")
                else:
                    print(f"[TTS DISABLED] {msg.text}")

# =========================
# === AUDIO/PTT WORKER ====
# =========================

class AudioInputWorker(WorkerThread):
    # Audio/PTT worker: captures microphone input and streams audio chunks to STT.
    def __init__(self, inbox: queue.Queue, stt_out: queue.Queue, coord_out: Optional[queue.Queue] = None):
        super().__init__("AudioInputWorker", inbox)
        self.stt_out = stt_out
        # Coordinator output queue for system-level messages (e.g., ESC -> shutdown)
        self.coord_out = coord_out
        self._pyaudio = None
        self._stream = None
        self._is_pressed = False
        self._frames: List[bytes] = []
        self._mode: CaptureMode = CaptureMode.COMMAND
        self._keyboard_listener = None

    def _on_press(self, key):
        # Handle keyboard press events (SPACE to record, ESC to request shutdown).
        if key == keyboard.Key.space:
            if not self._is_pressed:
                self._is_pressed = True
                self._start_recording()
        elif key == keyboard.Key.esc:
            # Request system shutdown via coordinator (preferred) or fallback to STT queue
            if self.coord_out:
                try:
                    self.coord_out.put(MsgSystemShutdownRequested(reason="ESC pressed"))
                except Exception:
                    pass
            else:
                try:
                    self.stt_out.put(MsgSystemShutdownRequested(reason="ESC pressed"))
                except Exception:
                    pass

    def _on_release(self, key):
        # Handle keyboard release events (SPACE to stop recording).
        if key == keyboard.Key.space:
            if self._is_pressed:
                self._is_pressed = False
                self._stop_recording_and_emit()

    def _start_recording(self):
        # Start audio recording from microphone.
        self._frames = []
        try:
            self._stream = self._pyaudio.open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_FRAMES_PER_BUFFER
            )
            self._stream.start_stream()
            print("[Audio] Recording started.")
        except Exception as e:
            print(f"[Audio] Error opening stream: {e}")
            self._stream = None

    def _stop_recording_and_emit(self):
    # Stop recording and emit a zero-length final marker to signal end-of-utterance to STT.
        if self._stream:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                print(f"[Audio] Error closing stream: {e}")
            self._stream = None
        data = b''.join(self._frames)
        self._frames = []
        if data:
            # Chunks have already been streamed; only send a final marker so STT finalizes.
            try:
                self.stt_out.put(MsgAudioData(data=b'', mode=self._mode, final=True))
                print(f"[Audio] Emitted final AudioData marker ({self._mode}).")
            except Exception:
                # Fallback: if put fails, log and continue
                print("[Audio] Failed to emit final marker.")
        else:
            print("[Audio] No audio captured.")

    def run(self):
    # Main loop: handle audio capture and control messages.
        self._pyaudio = pyaudio.PyAudio()
        print("Audio/PTT: Press and hold SPACE to speak, ESC to exit.")
        self._keyboard_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._keyboard_listener.start()
        try:
            while not self.stopped():
                # If recording, pull audio
                if self._stream and not self._stream.is_stopped():
                    try:
                        data = self._stream.read(AUDIO_READ_CHUNK, exception_on_overflow=False)
                        self._frames.append(data)
            # Emit streaming chunk to STT worker (non-final)
                        try:
                            self.stt_out.put(MsgAudioData(data=data, mode=self._mode, final=False))
                        except Exception:
                            pass
                    except IOError as e:
                        print(f"[Audio] Read error: {e}")
                        self._stop_recording_and_emit()
                # Handle control messages
                try:
                    msg = self.inbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                if isinstance(msg, MsgShutdown):
                    break
                if isinstance(msg, MsgSetCaptureMode):
                    self._mode = msg.mode
                    print(f"[Audio] Capture mode set to: {self._mode}")
        finally:
            try:
                if self._keyboard_listener:
                    self._keyboard_listener.stop()
            except Exception:
                pass
            if self._stream:
                try:
                    if self._stream.is_active():
                        self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    pass
            if self._pyaudio:
                self._pyaudio.terminate()
            print("[Audio] Worker stopped.")

# =========================
# ===== STT WORKER ========
# =========================

class STTWorker(WorkerThread):
    # Speech-to-Text worker: receives audio chunks and emits transcribed text.
    def __init__(self, inbox: queue.Queue, coord_out: queue.Queue):
        super().__init__("STTWorker", inbox)
        self.coord_out = coord_out
        self.stt = None

    def run(self):
        # Main loop: process audio data and emit STT results.
        try:
            self.stt = SpeechToText(model_path=VOSK_MODEL_PATH, sample_rate=AUDIO_SAMPLE_RATE)
            print("[STT] Vosk model loaded.")
        except Exception as e:
            print(f"[STT] Initialization failed: {e}")
            self.stt = None
    # Streaming mode: process chunked audio messages
        current_mode = None
        streaming_active = False
        while not self.stopped():
            try:
                msg = self.inbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgAudioData):
                if not self.stt:
                    # Send empty result so coordinator can react (e.g., speak error)
                    self.coord_out.put(MsgSTTResult(text="", mode=msg.mode))
                    continue
                # If starting a new utterance
                if not streaming_active:
                    streaming_active = True
                    current_mode = msg.mode
                    try:
                        self.stt.start_utterance()
                    except Exception:
                        pass
                # Feed chunk
                try:
                    partial = self.stt.accept_waveform(msg.data)
                    # Partial results are available but we only emit final results for simplicity.
                except Exception:
                    partial = None
                # If final chunk, finalize and emit
                if msg.final:
                    try:
                        final_text = self.stt.end_utterance_and_get_result()
                    except Exception:
                        final_text = ""
                    streaming_active = False
                    current_mode = None
                    self.coord_out.put(MsgSTTResult(text=final_text or "", mode=msg.mode))
                    print(f"[STT] Result ({msg.mode}): '{final_text}'")
        # cleanup
        try:
            if self.stt:
                self.stt.close()
        except Exception:
            pass

# =========================
# ===== NLU WORKER ========
# =========================

class NLUWorker(WorkerThread):
    # NLU worker: receives text and emits parsed intent/entities.
    def __init__(self, inbox: queue.Queue, coord_out: queue.Queue):
        super().__init__("NLUWorker", inbox)
        self.coord_out = coord_out
        self.nlu = None

    def run(self):
        # Main loop: process NLU requests and emit results.
        try:
            self.nlu = NaturalLanguageUnderstanding(spacy_model_name=SPACY_MODEL_NAME)
            print("[NLU] Initialized.")
        except Exception as e:
            print(f"[NLU] Initialization failed: {e}")
            self.nlu = None
        while not self.stopped():
            try:
                msg = self.inbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgNLURequest):
                if not self.nlu:
                    self.coord_out.put(MsgNLUResult(text=msg.text, intent="UNKNOWN_INTENT", confidence=0.0, entities={}))
                    continue
                parsed = self.nlu.parse_command(msg.text)
                self.coord_out.put(MsgNLUResult(
                    text=parsed.get("text", msg.text),
                    intent=parsed.get("intent", "UNKNOWN_INTENT"),
                    confidence=parsed.get("confidence", 0.0),
                    entities=parsed.get("entities", {})
                ))
                print(f"[NLU] Intent: {parsed.get('intent')} conf={parsed.get('confidence')}")

# =========================
# ==== MAVLINK WORKER =====
# =========================

class MavlinkWorker(WorkerThread):
    # MAVLink worker: manages drone controllers and executes mission commands.
    def __init__(self, inbox: queue.Queue, tts_out: queue.Queue):
        super().__init__("MavlinkWorker", inbox)
        self.tts_out = tts_out
        self.controllers: Dict[str, MavlinkController] = {}
        self.active_agent_id: Optional[str] = None

    def _speak(self, text: str):
        # Send a message to the TTS worker for operator feedback.
        self.tts_out.put(MsgSpeak(text=text))

    def _get_active(self) -> Optional[MavlinkController]:
        # Get the currently active drone controller.
        if not self.active_agent_id:
            return None
        ctrl = self.controllers.get(self.active_agent_id)
        if ctrl and ctrl.is_connected():
            return ctrl
        return None

    def run(self):
    # Main loop: process MAVLink commands and manage drone state.
    # Initialize controllers and select an active agent.
        if not AGENT_CONNECTION_CONFIGS:
            self._speak("Fatal error: No MAVLink agents configured.")
            return
        # Choose initial agent
        candidate_active = INITIAL_ACTIVE_AGENT_ID if INITIAL_ACTIVE_AGENT_ID in AGENT_CONNECTION_CONFIGS else next(iter(AGENT_CONNECTION_CONFIGS))
        for agent_id, cfg in AGENT_CONNECTION_CONFIGS.items():
            conn_str = cfg["connection_string"]
            baud = cfg.get("baudrate") if conn_str.lower().startswith(("com", "/dev/")) else None
            src_id = cfg.get("source_system_id", MAVLINK_SOURCE_SYSTEM_ID)
            print(f"[MAV] Init controller for {agent_id} -> {conn_str}")
            ctrl = MavlinkController(connection_string=conn_str, baudrate=baud, source_system_id=src_id)
            if ctrl.is_connected():
                self.controllers[agent_id] = ctrl
                print(f"[MAV] Connected: {agent_id}")
        if not self.controllers:
            self._speak("Fatal error: No MAVLink agents connected.")
            return
        # Choose initial active agent: prefer configured default, otherwise first connected.
        self.active_agent_id = candidate_active if candidate_active in self.controllers else next(iter(self.controllers))
        self._speak(f"{self.active_agent_id}, is selected")

        def _decode_statustext(msg) -> str:
            try:
                raw = msg.text
                # Normalize to text: pymavlink may return bytes or str depending on dialect/version.
                return raw.decode() if hasattr(raw, 'decode') else str(raw)
            except Exception:
                return ""

        def _parse_found(text: str):
            # Parse FOUND messages in format: FOUND:<lat>,<lon>
            if not text:
                return None
            s = text.strip()
            if not s.upper().startswith("FOUND:"):
                return None
            try:
                payload = s.split(":", 1)[1].strip()
                parts = [p.strip() for p in payload.split(",")]
                if len(parts) != 2:
                    return None
                lat = float(parts[0]); lon = float(parts[1])
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    return None
                return lat, lon
            except Exception:
                return None

        def _poll_statustext_all():
            # Non-blocking poll across all controllers for STATUSTEXT messages and react to FOUND reports.
            for aid, c in self.controllers.items():
                try:
                    # Drain a couple of STATUSTEXT messages per loop to avoid starvation.
                    for _ in range(3):
                        msg = c.master.recv_match(type='STATUSTEXT', blocking=False)
                        if not msg:
                            break
                        text = _decode_statustext(msg)
                        if not text:
                            continue
                        # Log for visibility
                        print(f"[MAV:{aid}] STATUSTEXT: {text}")
                        found = _parse_found(text)
                        if found:
                            _handle_found(aid, found[0], found[1])
                except Exception:
                    # Ignore transient errors while polling
                    pass

        def _handle_found(source_agent_id: str, lat: float, lon: float):
            # When a source agent reports a found target, dispatch another available agent (if any) to verify.
            self._speak(f"Agent {source_agent_id} reported a target at {lat:.6f}, {lon:.6f}.")
            # Set the reporting agent to LOITER mode and complete its mission if possible
            source_ctrl = self.controllers.get(source_agent_id)
            if source_ctrl and source_ctrl.is_connected():
                # Switch to GUIDED mode and send a position hold at current lat/lon/alt
                if source_ctrl.set_mode("GUIDED"):
                    self._speak(f"{source_agent_id} set to GUIDED mode after finding target.")
                    try:
                        msg = source_ctrl.master.recv_match(type=["GLOBAL_POSITION_INT", "VFR_HUD"], blocking=True, timeout=2)
                        if msg and msg.get_type() == "GLOBAL_POSITION_INT":
                            lat_deg = float(getattr(msg, "lat", 0)) / 1e7
                            lon_deg = float(getattr(msg, "lon", 0)) / 1e7
                            alt_m = float(getattr(msg, "alt", 0)) / 1000.0
                        elif msg and msg.get_type() == "VFR_HUD":
                            lat_deg = lat
                            lon_deg = lon
                            alt_m = float(getattr(msg, "alt", 0))
                        else:
                            lat_deg = lat
                            lon_deg = lon
                            alt_m = DEFAULT_WAYPOINT_ALTITUDE
                        # Ensure altitude is at least 10m above ground
                        safe_alt = max(alt_m, 10.0)
                        source_ctrl.master.mav.command_long_send(
                            source_ctrl.master.target_system,
                            source_ctrl.master.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                            0,  # confirmation
                            0,  # hold time
                            0,  # acceptance radius
                            0,  # pass through
                            lat_deg,  # latitude
                            lon_deg,  # longitude
                            safe_alt, # altitude
                            0         # yaw
                        )
                        print(f"Sent GUIDED position hold at {lat_deg},{lon_deg},{safe_alt}m.")
                    except Exception as e:
                        print(f"Failed to send GUIDED position hold: {e}")
                else:
                    self._speak(f"Failed to set {source_agent_id} to GUIDED mode.")
                # Optionally, send a mission complete command if supported
                try:
                    source_ctrl.master.mav.mission_ack_send(
                        source_ctrl.master.target_system,
                        source_ctrl.master.target_component,
                        mavutil.mavlink.MAV_MISSION_ACCEPTED,
                        mavutil.mavlink.MAV_MISSION_TYPE_MISSION
                    )
                except Exception:
                    pass

            # Choose a verifying agent: first connected controller that's not the reporter
            # and not currently busy with a Lifeguard-dispatched mission.
            verify_agent_id = None
            for agent_id, controller in self.controllers.items():
                if agent_id == source_agent_id:
                    continue
                if not controller or not controller.is_connected():
                    continue
                # Skip agents on missions we dispatched (tracked via 'lifeguard_mission_active').
                if getattr(controller, 'lifeguard_mission_active', False):
                    # If agent is in AUTO or GUIDED and lifeguard_mission_active is True, skip
                    try:
                        mode = None
                        if hasattr(controller.master, 'flightmode'):
                            mode = controller.master.flightmode
                        elif hasattr(controller.master, 'mode'):
                            mode = controller.master.mode
                        if mode and mode.upper() in ["AUTO", "GUIDED"]:
                            continue
                    except Exception:
                        continue
                verify_agent_id = agent_id
                break

            if not verify_agent_id:
                self._speak("No other agents are available to verify.")
                return

            verifier = self.controllers.get(verify_agent_id)
            self._speak(f"Dispatching {verify_agent_id} to verify.")
            def deactivate_lifeguard_mission(verifier):
                setattr(verifier, 'lifeguard_mission_active', False)

            setattr(verifier, 'lifeguard_mission_active', True)
            try:
                # Upload a 50m grid search at the FOUND coordinates, at a slightly lower altitude
                grid_size_m = 50
                verify_altitude = DEFAULT_WAYPOINT_ALTITUDE - 5  # Lower by 5 meters to reduce collision risk
                ok = verifier.generate_and_upload_search_grid_mission(
                    center_lat=lat,
                    center_lon=lon,
                    grid_size_m=grid_size_m,
                    swath_width_m=DEFAULT_SWATH_WIDTH_M,
                    altitude_m=verify_altitude
                )
                if not ok:
                    self._speak(f"Failed to upload verification grid to {verify_agent_id}.")
                    return
                if verifier.set_mode("AUTO"):
                    time.sleep(1)
                    if verifier.arm_vehicle():
                        time.sleep(1)
                        if verifier.start_mission():
                            self._speak(f"{verify_agent_id} dispatched to verify target location with a 50 meter grid search.")
                        else:
                            self._speak(f"{verify_agent_id} failed to start mission.")
                            deactivate_lifeguard_mission(verifier)
                    else:
                        self._speak(f"{verify_agent_id} failed to arm.")
                        deactivate_lifeguard_mission(verifier)
                else:
                    self._speak(f"{verify_agent_id} failed to set AUTO mode.")
                    deactivate_lifeguard_mission(verifier)
            except Exception as e:
                print(f"[MAV] Verification dispatch error: {e}")
            finally:
                pass


        while not self.stopped():
            try:
                msg = self.inbox.get(timeout=0.2)
            except queue.Empty:
                # Even if there are no incoming commands, poll for STATUSTEXT updates.
                _poll_statustext_all()
                continue
            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgSelectAgent):
                new_id = msg.agent_id
                if new_id in self.controllers and self.controllers[new_id].is_connected():
                    self.active_agent_id = new_id
                    self._speak(f"Active agent set to {new_id}.")
                else:
                    self._speak(f"Cannot select agent {new_id}; not connected or unknown.")
                continue
            ctrl = self._get_active()
            if not ctrl:
                self._speak("Active agent not available.")
                continue
            if isinstance(msg, MsgCommandGridSearch):
                text = f"Generating {msg.grid_size_m}m grid at {msg.lat:.6f}, {msg.lon:.6f}."
                if msg.target_desc: text += f" Target: {msg.target_desc}."
                self._speak(text)
                ok = ctrl.generate_and_upload_search_grid_mission(
                    center_lat=msg.lat, center_lon=msg.lon,
                    grid_size_m=msg.grid_size_m, swath_width_m=msg.swath_m, altitude_m=msg.altitude_m
                )
                if not ok:
                    self._speak("Failed to upload search grid mission.")
                    continue
                self._speak("Grid uploaded. Arming and starting mission.")
                if ctrl.arm_vehicle():
                    time.sleep(1)
                    if ctrl.set_mode("AUTO"):
                        time.sleep(1)
                        if ctrl.start_mission():
                            self._speak("Mission started.")
                            if msg.target_desc:
                                self._speak("Waiting to reach area before sending target details.")
                                if ctrl.wait_for_waypoint_reached(1, timeout_seconds=180):
                                    ctrl.send_status_text(f"TARGET:{msg.target_desc}")
                                    self._speak("Target details sent.")
                                else:
                                    self._speak("Timeout waiting for area. Target details not sent.")
                        else:
                            self._speak("Failed to start mission.")
                    else:
                        self._speak("Failed to set AUTO mode.")
                else:
                    self._speak("Failed to arm vehicle.")
            elif isinstance(msg, MsgCommandFlyTo):
                self._speak(f"Uploading waypoint to {msg.lat:.6f}, {msg.lon:.6f}.")
                wp = [
                    (msg.lat, msg.lon, msg.altitude_m,
                     mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 10.0, 0.0, float('nan'))
                ]
                ok = ctrl.upload_mission(wp)
                if not ok:
                    self._speak("Failed to upload waypoint.")
                    continue
                if ctrl.set_mode("AUTO"):
                    time.sleep(1)
                    if ctrl.arm_vehicle():
                        time.sleep(1)
                        if ctrl.start_mission():
                            self._speak("Mission to waypoint started.")
                            if msg.target_desc:
                                self._speak("Waiting to reach waypoint before sending target details.")
                                if ctrl.wait_for_waypoint_reached(1, timeout_seconds=180):
                                    ctrl.send_status_text(f"TARGET:{msg.target_desc}")
                                    self._speak("Target details sent.")
                                else:
                                    self._speak("Failed to confirm waypoint reached.")
                        else:
                            self._speak("Failed to start mission.")
                    else:
                        self._speak("Failed to arm vehicle.")
                else:        
                    self._speak("Failed to set AUTO mode.")
            elif isinstance(msg, MsgCommandSetAltitude):
                self._speak(f"Setting altitude to {msg.altitude_m} meters.")
                if ctrl.set_altitude(msg.altitude_m):
                    self._speak(f"Altitude set to {msg.altitude_m} meters.")
                else:
                    self._speak("Failed to set altitude.")
        
        # Final quick poll for STATUSTEXT updates before shutdown.
        _poll_statustext_all()
        # Cleanup: try to put vehicles into GUIDED mode for a safer operator handoff,
        # then close the MAVLink connections.
        for aid, ctrl in self.controllers.items():
            try:
                if ctrl and ctrl.is_connected():
                    try:
                        print(f"[MAV] Attempting to set {aid} to GUIDED for safe shutdown.")
                        if ctrl.set_mode("GUIDED"):
                            # announce via TTS that mode change was issued
                            self._speak(f"Setting {aid} to GUIDED.")
                            time.sleep(1)
                        else:
                            print(f"[MAV] Could not set {aid} to GUIDED.")
                    except Exception as e:
                        print(f"[MAV] Error setting GUIDED on {aid}: {e}")
            except Exception:
                pass
            try:
                ctrl.close_connection()
            except Exception:
                pass
        print("[MAV] Worker stopped.")

    def attempt_set_guided_all(self):
        """Synchronous attempt to set all connected controllers to GUIDED mode.
        This is intended to be called from the main thread during shutdown so
        we don't depend on the worker thread being scheduled to perform the mode change."""
        for aid, ctrl in self.controllers.items():
            try:
                if ctrl and ctrl.is_connected():
                    try:
                        print(f"[MAV] (sync) Attempting to set {aid} to GUIDED for safe shutdown.")
                        success = ctrl.set_mode("GUIDED")
                        if success:
                            print(f"[MAV] (sync) {aid} set to GUIDED.")
                        else:
                            print(f"[MAV] (sync) Failed to set {aid} to GUIDED.")
                    except Exception as e:
                        print(f"[MAV] (sync) Error setting GUIDED on {aid}: {e}")
            except Exception:
                pass

# =========================
# ==== COORDINATOR =========
# =========================

class Coordinator(WorkerThread):
    # Coordinator worker: routes messages and maintains simple conversation state (e.g., confirmations).
    def __init__(self,
                 inbox: queue.Queue,
                 audio_ctrl_out: queue.Queue,
                 nlu_outbox: queue.Queue,
                 mav_outbox: queue.Queue,
                 tts_outbox: queue.Queue):
        super().__init__("Coordinator", inbox)
        self.audio_ctrl_out = audio_ctrl_out
        self.nlu_outbox = nlu_outbox
        self.mav_outbox = mav_outbox
        self.tts_outbox = tts_outbox
        self.awaiting_yes_no = False
        self.pending_nlu: Optional[MsgNLUResult] = None

    def _speak(self, text: str):
    # Enqueue a message for the TTS worker.
        self.tts_outbox.put(MsgSpeak(text=text))

    def _set_capture_mode(self, mode: CaptureMode):
    # Switch the audio capture mode (COMMAND or YESNO) in the audio worker.
        self.audio_ctrl_out.put(MsgSetCaptureMode(mode=mode))

    def _requires_confirmation(self, nlu: MsgNLUResult) -> bool:
    # Determine if a command requires operator confirmation before execution.
    # Skip confirmation for SELECT_AGENT if confidence is high.
        if nlu.intent == "SELECT_AGENT" and nlu.confidence > 0.85:
            return False
        return True

    def _handle_intent(self, nlu: MsgNLUResult):
    # Handle parsed intent and route commands to appropriate subsystems.
        intent = nlu.intent
        entities = nlu.entities or {}
        conf = nlu.confidence or 0.0
        text = nlu.text or ""

        if intent == "UNKNOWN_INTENT" or conf < NLU_CONFIDENCE_THRESHOLD:
            self._speak("Command not understood clearly or intent unknown. Cancelled.")
            return

        if intent == "SELECT_AGENT":
            new_agent = entities.get("selected_agent_id")
            if not new_agent:
                self._speak("No agent specified.")
                return
            self.mav_outbox.put(MsgSelectAgent(agent_id=new_agent))
            return

        if intent == "SET_AGENT_ALTITUDE":
            altitude_m = entities.get("altitude_meters")
            if altitude_m is None:
                self._speak("No altitude specified.")
                return
            self.mav_outbox.put(MsgCommandSetAltitude(altitude_m=altitude_m))
            return

        parsed_gps = None
        if "latitude" in entities and "longitude" in entities:
            parsed_gps = {"latitude": entities["latitude"], "longitude": entities["longitude"]}

        target_desc = entities.get("target_description_full")
        if intent == "REQUEST_GRID_SEARCH" and parsed_gps and "grid_size_meters" in entities:
            self.mav_outbox.put(MsgCommandGridSearch(
                lat=parsed_gps["latitude"],
                lon=parsed_gps["longitude"],
                grid_size_m=entities["grid_size_meters"],
                swath_m=entities.get("swath_width_meters", DEFAULT_SWATH_WIDTH_M),
                altitude_m=DEFAULT_WAYPOINT_ALTITUDE,
                target_desc=target_desc
            ))
            return
        if intent in ("COMBINED_SEARCH_AND_TARGET", "REQUEST_SEARCH_AT_LOCATION") and parsed_gps:
            self.mav_outbox.put(MsgCommandFlyTo(
                lat=parsed_gps["latitude"],
                lon=parsed_gps["longitude"],
                altitude_m=DEFAULT_WAYPOINT_ALTITUDE,
                target_desc=target_desc
            ))
            return
        if intent == "PROVIDE_TARGET_DESCRIPTION" and target_desc:
            # Not wired to send STATUSTEXT directly yet; acknowledge verbally for now.
            self._speak(f"Target: {target_desc}. Sending to vehicle.")
            return

        self._speak("Command understood, but no specific action defined or missing GPS.")

    def _normalize_transcribed(self, text: str) -> str:
    # Normalize transcribed text (convert spoken numbers, clean whitespace, tidy GPS format).
        if not text:
            return ""
        text = spoken_numbers_to_digits(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'latitude\s*(-?\d+\.\d+)\s+longitude\s*(-?\d+\.\d+)',
                      r'latitude \1, longitude \2', text, flags=re.IGNORECASE)
        return text

    def run(self):
    # Main loop: process system messages and coordinate subsystem actions.
    # Set initial capture mode and greet the operator.
        self._set_capture_mode(CaptureMode.COMMAND)
        self._speak("LIFEGUARD is initialized! Press and hold space to speak. Press escape to exit.")
        while not self.stopped():
            try:
                msg = self.inbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgSystemShutdownRequested):
                self._speak("Shutdown requested.")
                break
            if isinstance(msg, MsgSTTResult):
                if msg.mode == CaptureMode.COMMAND:
                    text = self._normalize_transcribed(msg.text)
                    print(f"[Coordinator] Command STT: '{text}'")
                    if not text:
                        self._speak("Sorry, I could not understand. Please try again.")
                        continue
                    if "stop listening" in text.lower() or "shutdown system" in text.lower():
                        self._speak("System shutting down.")
                        break
                    # Forward normalized command text to NLU worker
                    self.nlu_outbox.put(MsgNLURequest(text=text))
                elif msg.mode == CaptureMode.YESNO:
                    answer = (msg.text or "").lower()
                    print(f"[Coordinator] Yes/No STT: '{answer}'")
                    decision = "unrecognized"
                    if "yes" in answer and "no" not in answer: decision = "yes"
                    elif "no" in answer and "yes" not in answer: decision = "no"
                    elif "yeah" in answer or "yep" in answer: decision = "yes"
                    elif "nope" in answer: decision = "no"

                    if not self.awaiting_yes_no or self.pending_nlu is None:
                        self._speak("No pending command to confirm.")
                        self._set_capture_mode(CaptureMode.COMMAND)
                        continue
                    if decision == "yes":
                        self._handle_intent(self.pending_nlu)
                        self.awaiting_yes_no = False
                        self.pending_nlu = None
                        self._set_capture_mode(CaptureMode.COMMAND)
                    elif decision == "no":
                        self._speak("Okay, command cancelled.")
                        self.awaiting_yes_no = False
                        self.pending_nlu = None
                        self._set_capture_mode(CaptureMode.COMMAND)
                    else:
                        self._speak("Sorry, I didn't catch that. Please say yes or no.")
                        # Remain in YESNO mode and ask again next utterance
            if isinstance(msg, MsgNLUResult):
                print(f"[Coordinator] NLU: {msg.intent} conf={msg.confidence}")
                if self._requires_confirmation(msg):
                    self.pending_nlu = msg
                    self.awaiting_yes_no = True
                    self._speak(f"Did you say: {msg.text}? Please press space and say yes or no.")
                    self._set_capture_mode(CaptureMode.YESNO)
                else:
                    self._handle_intent(msg)
                    self._set_capture_mode(CaptureMode.COMMAND)
        # Broadcast shutdown to others by putting MsgShutdown in their inboxes via main controller stop.
        print("[Coordinator] Stopping.")

# =========================
# ========= MAIN ==========
# =========================

class LifeguardSystem:
    # Main application: initializes queues/workers and coordinates startup/shutdown.
    def __init__(self):
        # Queues
        self.q_audio_ctrl = queue.Queue()
        self.q_stt_in = queue.Queue()      # Audio -> STT inbox
        self.q_coord_in = queue.Queue()    # STT/NLU -> Coordinator inbox
        self.q_nlu_in = queue.Queue()      # Coordinator -> NLU inbox
        self.q_mav_in = queue.Queue()      # Coordinator -> MAV inbox
        self.q_tts_in = queue.Queue()      # Coordinator/MAV -> TTS inbox

        # Workers
        self.tts = TTSWorker(inbox=self.q_tts_in)
        # pass coordinator inbox into audio worker so it can request system-level shutdowns
        self.audio = AudioInputWorker(inbox=self.q_audio_ctrl, stt_out=self.q_stt_in, coord_out=self.q_coord_in)
        self.stt = STTWorker(inbox=self.q_stt_in, coord_out=self.q_coord_in)
        self.nlu = NLUWorker(inbox=self.q_nlu_in, coord_out=self.q_coord_in)
        self.mav = MavlinkWorker(inbox=self.q_mav_in, tts_out=self.q_tts_in)
        self.coord = Coordinator(
            inbox=self.q_coord_in,
            audio_ctrl_out=self.q_audio_ctrl,
            nlu_outbox=self.q_nlu_in,
            mav_outbox=self.q_mav_in,
            tts_outbox=self.q_tts_in
        )
        self._workers: List[WorkerThread] = [self.tts, self.mav, self.nlu, self.stt, self.audio, self.coord]

    def start(self):
    # Start all worker threads, then wait for the coordinator to finish (join).
        for w in self._workers:
            w.start()
    # Join coordinator thread until it signals shutdown.
        self.coord.join()
        self.stop()

    def stop(self):
    # Stop all worker threads and attempt safe drone shutdown.
    # First try to set vehicles to GUIDED synchronously before stopping threads.
        try:
            if hasattr(self, 'mav') and isinstance(self.mav, MavlinkWorker):
                try:
                    self.mav.attempt_set_guided_all()
                except Exception as e:
                    print(f"Error attempting sync GUIDED set: {e}")
        except Exception:
            pass

        for w in self._workers:
            if w is self.coord:
                continue
            w.stop()
        # Also stop audio worker if still running (in case coordinator exited without broadcasting)
        self.audio.stop()
        # Give workers time to exit
        for w in self._workers:
            if w is self.coord:
                continue
            w.join(timeout=5)

if __name__ == "__main__":
    # Entry point: start the Lifeguard system and handle shutdown gracefully.
    app = LifeguardSystem()
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: shutting down...")
        app.stop()

    print("Application terminated.")
