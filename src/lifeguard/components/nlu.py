"""spaCy-powered NLU: extract intents and entities from transcribed text for SAR commands."""
import re
import spacy
import sys
import os
import logging
from spacy.tokens import Doc, Span
from spacy.language import Language
from lifeguard.utils.text_normalization import spoken_numbers_to_digits, extract_altitude_from_text
from lifeguard.system.exceptions import NLUError
from word2number import w2n

@Language.factory("lat_lon_entity_recognizer", default_config={"gps_label": "LOCATION_GPS_COMPLEX"})
def create_lat_lon_entity_recognizer(nlp: Language, name: str, gps_label: str):
	return LatLonEntityRecognizer(nlp, gps_label)

class LatLonEntityRecognizer:
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
    """Thin wrapper around a spaCy pipeline configured for SAR intents/entities."""
    def __init__(self, spacy_model_name="en_core_web_sm"):
        self.logger = logging.getLogger(__name__)
        try:
            model_path = None

            if getattr(sys, "frozen", False):
                # Running from PyInstaller single-file bundle; model is bundled under _MEIPASS
                base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
                root = os.path.join(base, "en_core_web_sm")

                # Prefer the inner directory that contains config.cfg
                cfg_candidates = []
                meta_candidates = []
                if os.path.isdir(root):
                    for cur_root, dirs, files in os.walk(root):
                        if "config.cfg" in files:
                            cfg_candidates.append(cur_root)
                        if "meta.json" in files:
                            meta_candidates.append(cur_root)

                if cfg_candidates:
                    # Pick the shortest path with config.cfg (typically the inner model dir)
                    model_path = sorted(cfg_candidates, key=len)[0]
                elif meta_candidates:
                    # Fallback: if only meta.json is present at root, search child dirs for config.cfg
                    try:
                        subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
                        for d in subdirs:
                            if os.path.exists(os.path.join(d, "config.cfg")):
                                model_path = d
                                break
                    except Exception:
                        pass

                if not model_path:
                    raise OSError(
                        f"Bundled spaCy model directory with config.cfg not found under '{root}'."
                    )
            else:
                # Non-frozen: prefer package-based load; falls back to name-based load
                try:
                    import en_core_web_sm  # type: ignore
                    self.nlp = en_core_web_sm.load()  # most robust in dev
                    model_path = None  # informational only
                except Exception:
                    # Fallback to name-based load (requires model installed)
                    self.nlp = spacy.load(spacy_model_name)
                    model_path = spacy_model_name

            # For frozen builds, load using the resolved model_path
            if getattr(sys, "frozen", False):
                self.logger.info(
                    f"Loading spaCy model from bundled path: {model_path}"
                )
                self.nlp = spacy.load(model_path)

        except OSError as e:
            self.logger.error(
                f"spaCy model '{spacy_model_name}' not found. "
                f"Detail: {e}. If running non-bundled, install with: "
                f"python -m spacy download {spacy_model_name}"
            )
            raise NLUError(f"spaCy model '{spacy_model_name}' not found.")
        try:
            if "lat_lon_entity_recognizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("lat_lon_entity_recognizer", after="ner" if self.nlp.has_pipe("ner") else None)
            if "sar_entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", name="sar_entity_ruler", before="lat_lon_entity_recognizer")
                sar_patterns = [
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
                    {"label": "ACTION_FLY_TO", "pattern": [{"LEMMA": {"IN": ["fly", "go", "navigate", "proceed", "move", "head"]}}, {"LOWER": {"IN": ["to", "towards", "toward"]}}]},
                    {"label": "ACTION_FLY_TO", "pattern": [{"LEMMA": {"IN": ["go", "navigate", "proceed", "move", "head"]}}, {"LOWER": {"IN": ["to"]}}]},
                    {"label": "ACTION_FLY_TO", "pattern": [{"LOWER": {"IN": ["goto", "go-to"]}}]},
                ]
                sar_patterns.extend([
                    {"label": "ALTITUDE_SET", "pattern": [{"LEMMA": {"IN": ["set", "change", "update", "adjust", "make", "put"]}}, {"LOWER": "altitude"}, {"IS_ALPHA": True}]},
                    {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"LEMMA": {"IN": ["set", "change", "update", "adjust", "make", "put"]}}, {"IS_ALPHA": True}]},
                    {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"IS_ALPHA": True}]},
                    {"label": "ALTITUDE_SET", "pattern": [{"LEMMA": {"IN": ["set", "change", "update", "adjust", "make", "put"]}}, {"LOWER": "altitude"}, {"LOWER": "to"}, {"IS_ALPHA": True}]},
                    {"label": "ALTITUDE_SET", "pattern": [{"LOWER": "altitude"}, {"LOWER": "to"}, {"IS_ALPHA": True}]},
                ])
                ruler.add_patterns(sar_patterns)
        except Exception as e:
            self.logger.error(f"Error configuring spaCy pipeline: {e}")
            raise NLUError(f"Error configuring spaCy pipeline: {e}")

    def parse_command(self, text):
        """Return a dict: {text, intent, confidence, entities} derived from input text."""
        try:
            doc = self.nlp(text)
        except Exception as e:
            self.logger.error(f"Error running spaCy NLP pipeline: {e}")
            raise NLUError(f"Error running spaCy NLP pipeline: {e}")
        intent = "UNKNOWN_INTENT"
        entities_payload = {}
        confidence = 0.5

        number_words = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
            "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
            "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
            "hundred": "100", "thousand": "1000"
        }
        def spoken_agent_to_id(agent_text):
            agent_text = agent_text.lower().replace("drone", "").replace("agent", "").strip()
            if agent_text in ["to", "too", "tu", "tow"]: agent_text = "two"
            if agent_text in ["for", "four", "fore", "fohr", "fawr"]: agent_text = "four"
            if agent_text.isdigit(): return f"agent{agent_text}"
            if agent_text in number_words: return f"agent{number_words[agent_text]}"
            try:
                num = w2n.word_to_num(agent_text)
                return f"agent{num}"
            except Exception: pass
            return None

        extracted_spacy_ents = []
        for ent in doc.ents:
            entity_data = {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            if ent.label_ == "ALTITUDE_SET":
                numeric_part = "".join(filter(str.isdigit, ent.text))
                if numeric_part:
                    entity_data["altitude_value"] = int(numeric_part)
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
                if agent_id: entity_data["value"] = agent_id
            elif ent.label_ == "AGENT_ID_TEXT":
                agent_id = spoken_agent_to_id(ent.text)
                if agent_id: entity_data["value"] = agent_id
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
        action_fly_to_verbs = get_entity_values("ACTION_FLY_TO")
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

        if not action_select_verbs and re.search(r"\bselect\b", text, re.IGNORECASE):
            action_select_verbs = ["select"]
        if not agent_ids_num and not agent_ids_text:
            m = re.search(r"\b(?:drone|agent)\s+([A-Za-z]+|\d+)\b", text, re.IGNORECASE)
            if m:
                sid = spoken_agent_to_id(m.group(1))
                if sid:
                    agent_ids_num = [sid]
        
        altitude_values = [e["altitude_value"] for e in extracted_spacy_ents if e["label"] == "ALTITUDE_SET" and "altitude_value" in e]
        if not altitude_values:
            alt_from_text = extract_altitude_from_text(text)
            if alt_from_text is not None:
                altitude_values = [alt_from_text]

        if parsed_gps_coords:
            entities_payload.update(parsed_gps_coords)
        if grid_sizes:
            entities_payload["grid_size_meters"] = grid_sizes[0]
        if altitude_values:
            entities_payload["altitude_meters"] = altitude_values[0]

        selected_agent_id = None
        if agent_ids_num: selected_agent_id = agent_ids_num[0]
        elif agent_ids_text: selected_agent_id = agent_ids_text[0]
        
        target_details_parts = []
        if target_objects: target_details_parts.extend(target_objects)
        target_match = re.search(r"\bfor\s+a\s+(.+)", text, re.IGNORECASE)
        if target_match:
            if "latitude" not in target_match.group(1) and "longitude" not in target_match.group(1):
                target_details_parts.append(target_match.group(1).strip())
        
        if target_details_parts:
            unique_parts = sorted(list(set(" ".join(target_details_parts).split())), key=" ".join(target_details_parts).split().index)
            entities_payload["target_description_full"] = " ".join(unique_parts)

        if selected_agent_id and action_select_verbs:
            intent = "SELECT_AGENT"
            entities_payload["selected_agent_id"] = selected_agent_id
            confidence = 0.9
        elif "altitude" in text.lower() and altitude_values:
            intent = "SET_AGENT_ALTITUDE"
            confidence = 0.95
        elif parsed_gps_coords:
            if "grid" in text.lower() and grid_sizes:
                intent = "REQUEST_GRID_SEARCH"
                confidence = 0.95
            elif action_fly_to_verbs or re.search(r"\b(fly|go|navigate)\s+(to)\b", text, re.IGNORECASE):
                intent = "REQUEST_FLY_TO"
                confidence = 0.9
            elif "search" in text.lower():
                if target_details_parts:
                    intent = "COMBINED_SEARCH_AND_TARGET"
                else:
                    intent = "REQUEST_SEARCH_AT_LOCATION"
                confidence = 0.9
        
        if intent == "UNKNOWN_INTENT" and target_details_parts:
            intent = "PROVIDE_TARGET_DESCRIPTION"
            confidence = 0.8

        return {"text": text, "intent": intent, "confidence": confidence, "entities": entities_payload}