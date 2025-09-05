"""Worker threads for the LIFEGUARD system."""
from __future__ import annotations

import queue
import threading
import logging
import re
import time
from typing import Optional, Dict, Any

import pyaudio
from lifeguard.components.stt import SpeechToText
from lifeguard.components.nlu import NaturalLanguageUnderstanding
from lifeguard.components.mavlink_io import MavlinkController
from lifeguard.utils.text_normalization import spoken_numbers_to_digits
from lifeguard.system.state import *

# Bridge class to forward backend status to the GUI queue
class UIMessenger:
    def __init__(self, gui_queue: Optional[queue.Queue] = None):
        self.gui_queue = gui_queue
    
    def post(self, message: str):
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(message)
            except queue.Full:
                pass

# Base WorkerThread with a simple run-loop wrapper and cooperative stop
class WorkerThread:
    """Base class for all worker threads."""
    def __init__(self, name: str, inbox: queue.Queue):
        self.name = name
        self.inbox = inbox
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_wrapper, name=name, daemon=True)
        self.logger = logging.getLogger(name)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            self.inbox.put_nowait(MsgShutdown(reason=f"Stop requested for {self.name}"))
        except Exception:
            pass

    def join(self, timeout: Optional[float] = None):
        self._thread.join(timeout=timeout)

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def _run_wrapper(self):
        try:
            self.logger.info("started")
            self.run()
        except Exception as e:
            self.logger.exception(f"Unhandled exception: {e}")
        finally:
            self.logger.info("stopped")

    def run(self):
        raise NotImplementedError

# Text-to-Speech worker using pyttsx3 in non-blocking loop
class TTSWorker(WorkerThread):
    """Text-to-Speech output using a non-blocking event loop."""
    def __init__(self, inbox: queue.Queue):
        super().__init__("TTSWorker", inbox)
        self._engine = None

    def run(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self.logger.info("TTS engine initialized")
        except Exception as e:
            self.logger.warning(f"TTS initialization failed: {e}")
            self._engine = None
            return

    # Start the engine's event loop in a non-blocking way
        self._engine.startLoop(False)

        while not self.stopped():
            try:
                msg: MsgBase = self.inbox.get_nowait()
                if isinstance(msg, MsgShutdown):
                    break
                if isinstance(msg, MsgSpeak):
                    if msg.text:
                        self._engine.say(msg.text)
            except queue.Empty:
                pass

            try:
                # Continuously process the running event loop
                self._engine.iterate()
            except Exception as e:
                self.logger.error(f"TTS engine iteration failed: {e}")
                # Attempt to re-initialize on failure
                try:
                    self._engine = pyttsx3.init()
                    self._engine.startLoop(False)  # Re-start the loop
                except Exception as init_e:
                    self.logger.error(f"TTS engine re-initialization failed: {init_e}")
                    self.stop()
            
            time.sleep(0.1)

    # Cleanly end the loop when the worker stops
        self._engine.endLoop()

# AudioInputWorker captures audio when the GUI toggles Push-to-Talk
class AudioInputWorker(WorkerThread):
    def start_recording(self):
        if not self._is_recording:
            self._is_recording = True
            self.logger.info("PTT: Recording started by GUI")
            if self._stream:
                self.logger.info(f"PTT: Stream exists, active={self._stream.is_active()}")
                if not self._stream.is_active():
                    try:
                        self._stream.start_stream()
                        self.logger.info("PTT: Stream started successfully.")
                    except Exception as e:
                        self.logger.error(f"PTT: Failed to start stream: {e}")
            else:
                self.logger.error("PTT: No stream object available when starting recording.")

    def stop_recording(self):
        if self._is_recording:
            self._is_recording = False
            self.logger.info("PTT: Recording stopped by GUI")
            if self._stream:
                self.logger.info(f"PTT: Stream exists, active={self._stream.is_active()}")
                if self._stream.is_active():
                    try:
                        self._stream.stop_stream()
                        self.logger.info("PTT: Stream stopped successfully.")
                    except Exception as e:
                        self.logger.error(f"PTT: Failed to stop stream: {e}")
            else:
                self.logger.error("PTT: No stream object available when stopping recording.")
        self.stt_out.put(MsgAudioData(data=b"", mode=self._mode, final=True))
    """Captures audio when triggered by the GUI."""
    def __init__(self, inbox: queue.Queue, stt_out: queue.Queue):
        super().__init__("AudioInputWorker", inbox)
        self.stt_out = stt_out
        self._pyaudio = None
        self._stream = None
        self._is_recording = False
        self._mode: CaptureMode = CaptureMode.COMMAND

    def run(self):
        try:
            self.logger.info("PTT: Initializing PyAudio...")
            self._pyaudio = pyaudio.PyAudio()
            self.logger.info("PTT: PyAudio initialized.")
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096,
                start=False
            )
            self.logger.info("PTT: PyAudio stream created.")
        except Exception as e:
            self.logger.critical(f"PTT: Failed to initialize PyAudio stream: {e}")
            return

        try:
            while not self.stopped():
                try:
                    msg: MsgBase = self.inbox.get(timeout=0.01)
                    if isinstance(msg, MsgShutdown):
                        break
                    if isinstance(msg, MsgSetCaptureMode):
                        self._mode = msg.mode
                        self.logger.info(f"Capture mode set to {self._mode.value}")
                except queue.Empty:
                    pass

                if self._is_recording and self._stream and self._stream.is_active():
                    try:
                        chunk = self._stream.read(2048, exception_on_overflow=False)
                        self.logger.debug(f"PTT: Read audio chunk of size {len(chunk)} bytes.")
                        self.stt_out.put(MsgAudioData(data=chunk, mode=self._mode, final=False))
                    except Exception as e:
                        self.logger.error(f"PTT: Audio read error: {e}")
                else:
                    time.sleep(0.02)
        finally:
            if self._stream:
                self._stream.close()
            if self._pyaudio:
                self._pyaudio.terminate()

# STT worker: streams audio frames to Vosk and posts final transcriptions
class STTWorker(WorkerThread):
    """Streams audio frames to Vosk STT and emits final transcriptions."""
    def __init__(self, inbox: queue.Queue, coord_out: queue.Queue, settings: dict):
        super().__init__("STTWorker", inbox)
        self.coord_out = coord_out
        self.settings = settings
        self.stt = None

    def run(self):
        try:
            model_path = self.settings.get("vosk_model_path", "vosk_models/vosk-model-small-en-us-0.15")
            sample_rate = self.settings.get("audio", {}).get("sample_rate", 16000)
            self.stt = SpeechToText(model_path, sample_rate)
        except Exception as e:
            self.logger.error(f"Failed to init STT: {e}")
            return

        streaming_active = False
        current_mode = CaptureMode.COMMAND

        while not self.stopped():
            try:
                msg: MsgBase = self.inbox.get(timeout=0.2)
            except queue.Empty:
                continue

            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgAudioData):
                current_mode = msg.mode
                if msg.final:
                    if streaming_active:
                        text = self.stt.end_utterance_and_get_result() or ""
                        self.coord_out.put(MsgSTTResult(text=text, mode=current_mode))
                        streaming_active = False
                    else:
                        self.coord_out.put(MsgSTTResult(text="", mode=current_mode))
                else:
                    if not streaming_active:
                        self.stt.start_utterance()
                        streaming_active = True
                    self.stt.accept_waveform(msg.data)
        
        if self.stt: self.stt.close()

class NLUWorker(WorkerThread):
    """Runs spaCy NLU to derive intents/entities from transcribed text."""
    def __init__(self, inbox: queue.Queue, coord_out: queue.Queue, settings: dict):
        super().__init__("NLUWorker", inbox)
        self.coord_out = coord_out
        self.settings = settings
        self.nlu = None

    def run(self):
        try:
            spacy_model = self.settings.get("nlu", {}).get("spacy_model_name", "en_core_web_sm")
            self.nlu = NaturalLanguageUnderstanding(spacy_model)
        except Exception as e:
            self.logger.error(f"Failed to init NLU: {e}")
            return

        while not self.stopped():
            try:
                msg: MsgBase = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgNLURequest):
                try:
                    res = self.nlu.parse_command(msg.text or "")
                    self.coord_out.put(
                        MsgNLUResult(
                            text=res.get("text", ""),
                            intent=res.get("intent", "UNKNOWN_INTENT"),
                            confidence=float(res.get("confidence", 0.0)),
                            entities=res.get("entities", {}),
                        )
                    )
                except Exception as e:
                    self.logger.error(f"NLU error: {e}")

class MavlinkWorker(WorkerThread):
    """Manages MAVLink connections, executes commands, and polls STATUSTEXT."""
    def __init__(self, inbox: queue.Queue, tts_out: queue.Queue, settings: dict, ui_messenger: UIMessenger):
        super().__init__("MavlinkWorker", inbox)
        self.tts_out = tts_out
        self.settings = settings
        self.ui_messenger = ui_messenger
        self.controllers: Dict[str, MavlinkController] = {}
        self.active_agent_id: Optional[str] = None
        self.active_missions: Dict[str, Optional[Dict[str, Any]]] = {}

    def _speak(self, text: str):
        self.ui_messenger.post(f"LIFEGUARD: {text}")
    # Speak via TTS and also mirror to GUI
        self.tts_out.put(MsgSpeak(text=text))

    def _get_active(self) -> Optional[MavlinkController]:
        if not self.active_agent_id: return None
        ctrl = self.controllers.get(self.active_agent_id)
        if ctrl and ctrl.is_connected(): return ctrl
        return None

    def set_guided_on_exit(self):
        self.logger.info("Attempting to set all agents to GUIDED for safe shutdown...")
        for agent_id, ctrl in self.controllers.items():
            try:
                if ctrl and ctrl.is_connected():
                    self.logger.info(f"Setting {agent_id} to GUIDED mode.")
                    ctrl.set_mode("GUIDED")
            except Exception as e:
                self.logger.error(f"Could not set {agent_id} to GUIDED on exit: {e}")

    def _handle_found_target(self, source_agent_id: str, lat: float, lon: float):
        self._speak(f"Agent {source_agent_id} reported a target at {lat:.6f}, {lon:.6f}.")

        source_ctrl = self.controllers.get(source_agent_id)
        if source_ctrl and source_ctrl.is_connected():
            self.logger.info(f"Commanding {source_agent_id} to loiter at its current position.")
            current_pos = source_ctrl.get_current_position()
            if current_pos:
                if source_ctrl.set_mode("GUIDED"):
                    self.ui_messenger.post(f"[{source_agent_id}] Mode set to GUIDED.")
                source_ctrl.fly_to(current_pos['lat'], current_pos['lon'], current_pos['alt'])
                self.ui_messenger.post(f"[{source_agent_id}] Loiter command sent.")
                self._speak(f"{source_agent_id} is holding position.")
            else:
                self._speak(f"Could not get position for {source_agent_id} to loiter.")

        responder_id = None
        for agent_id, controller in self.controllers.items():
            if agent_id != source_agent_id and controller.is_connected() and not self.active_missions.get(agent_id):
                responder_id = agent_id
                break
        
        if not responder_id:
            self._speak("No other agents are available to verify.")
            return

        responder = self.controllers.get(responder_id)
        if not responder: return

        mission_details = self.active_missions.get(source_agent_id, {})
        original_target_desc = mission_details.get("target_desc") if mission_details else None

        self._speak(f"Dispatching {responder_id} to verify.")
        
        default_alt = self.settings.get("mission", {}).get("default_waypoint_altitude", 30.0)
        swath = self.settings.get("mission", {}).get("default_swath_width", 20.0)
        verify_altitude = max(15.0, default_alt - 5)
        
        verification_msg = MsgCommandGridSearch(
            lat=lat, lon=lon, grid_size_m=50, swath_m=swath,
            altitude_m=verify_altitude, target_desc=original_target_desc
        )
        
        self.active_missions[responder_id] = {"target_desc": original_target_desc}
        mission_thread = threading.Thread(
            target=self._execute_mission_sequence,
            args=(responder, verification_msg, responder_id)
        )
        mission_thread.daemon = True
        mission_thread.start()

    def _execute_mission_sequence(self, ctrl: MavlinkController, msg: MsgCommandGridSearch | MsgCommandFlyTo, agent_id: str):
        try:
            if isinstance(msg, MsgCommandGridSearch):
                self._speak(f"Uploading {msg.grid_size_m}m grid mission to {agent_id}.")
                ok = ctrl.generate_and_upload_search_grid_mission(msg.lat, msg.lon, msg.grid_size_m, msg.swath_m, msg.altitude_m)
            else:
                self._speak(f"Uploading waypoint to {agent_id}.")
                wp = [(msg.lat, msg.lon, msg.altitude_m, 16, 0.0, 10.0, 0.0, float('nan'))]
                ok = ctrl.upload_mission(wp)

            if ok: self.ui_messenger.post(f"[{agent_id}] Mission upload successful.")
            else: raise Exception("Mission upload failed.")
            
            if ctrl.set_mode("AUTO"): self.ui_messenger.post(f"[{agent_id}] Mode set to AUTO.")
            else: raise Exception("Set mode AUTO failed.")

            if ctrl.arm_vehicle(): self.ui_messenger.post(f"[{agent_id}] Vehicle armed.")
            else: raise Exception("Arming failed.")
            
            if ctrl.start_mission(): self.ui_messenger.post(f"[{agent_id}] Mission start command sent.")
            else: raise Exception("Mission start failed.")

            self._speak(f"Mission started for {agent_id}.")
            self.ui_messenger.post(f"[{agent_id}] Mission is executing.")

            if msg.target_desc:
                self._speak(f"Waiting to reach first waypoint to send target details to {agent_id}.")
                if ctrl.wait_for_waypoint_reached(1, timeout_seconds=180):
                    ctrl.send_status_text(f"TARGET:{msg.target_desc}")
                    self._speak("Target details sent.")
                else:
                    self._speak("Timeout waiting for waypoint. Target details not sent.")
            
            self.logger.info(f"Mission thread for {agent_id} completed its sequence.")
        except Exception as e:
            self.logger.error(f"Mission sequence for {agent_id} failed: {e}")
            self._speak(f"Mission for {agent_id} failed.")
            self.active_missions[agent_id] = None

    def run(self):
        for agent_config in self.settings.get("agents", []):
            try:
                agent_id = agent_config["name"]
                conn_str = agent_config["connection_string"]
                baud = self.settings.get("mavlink", {}).get("baudrate")
                src_id = self.settings.get("mavlink", {}).get("source_system_id")

                ctrl = MavlinkController(conn_str, baud, src_id)
                ctrl.connect()
                if ctrl.is_connected():
                    self.controllers[agent_id] = ctrl
                    self.ui_messenger.post(f"[{agent_id}] Connection successful.")
                    self.active_missions[agent_id] = None
            except Exception as e:
                self.ui_messenger.post(f"[{agent_config.get('name')}] Connection failed.")
                self.logger.error(f"Connection to {agent_config.get('name')} failed: {e}")

        if not self.controllers: self._speak("No agents available")
        
        self.active_agent_id = next(iter(self.controllers)) if self.controllers else None
        if self.active_agent_id: self._speak(f"{self.active_agent_id} is selected")

        last_poll = 0.0
        poll_interval = 1.0

        while not self.stopped():
            now = time.time()
            if now - last_poll >= poll_interval:
                for aid, ctrl in list(self.controllers.items()):
                    try:
                        if not ctrl.is_connected(): continue
                        msg = ctrl.master.recv_match(type='STATUSTEXT', blocking=False)
                        if msg and hasattr(msg, 'text'):
                            text_val = msg.text
                            if isinstance(text_val, (bytes, bytearray)):
                                text_val = text_val.decode('utf-8', errors='ignore').rstrip('\x00')
                            if isinstance(text_val, str) and text_val:
                                self.logger.info(f"Status from {aid}: {text_val}")
                                if text_val.upper().startswith("FOUND:"):
                                    try:
                                        coords = text_val.split(":", 1)[1]
                                        lat_s, lon_s = coords.split(",", 1)
                                        self._handle_found_target(aid, float(lat_s), float(lon_s))
                                    except (ValueError, IndexError) as e:
                                        self.logger.warning(f"Could not parse FOUND message '{text_val}': {e}")
                                else:
                                    # Send all other STATUSTEXT to the GUI
                                    self.ui_messenger.post(f"[{aid}] {text_val}")
                    except Exception as e:
                        self.logger.warning(f"Error polling STATUSTEXT from {aid}: {e}")
                last_poll = now

            try:
                msg: MsgBase = self.inbox.get(timeout=0.2)
            except queue.Empty: continue

            if isinstance(msg, MsgShutdown): break
            
            active_controller = self._get_active()
            if not active_controller and not isinstance(msg, MsgSelectAgent):
                self._speak("No active agent.")
                continue
            
            if isinstance(msg, (MsgCommandGridSearch, MsgCommandFlyTo)):
                if self.active_agent_id:
                    self.active_missions[self.active_agent_id] = {"target_desc": msg.target_desc}
                    mission_thread = threading.Thread(
                        target=self._execute_mission_sequence,
                        args=(active_controller, msg, self.active_agent_id)
                    )
                    mission_thread.daemon = True
                    mission_thread.start()
            elif isinstance(msg, MsgSelectAgent):
                if msg.agent_id in self.controllers:
                    self.active_agent_id = msg.agent_id
                    self._speak(f"{msg.agent_id} is selected")
                else:
                    self._speak(f"Agent {msg.agent_id} not found")
            elif isinstance(msg, MsgCommandSetAltitude):
                ok = active_controller.set_altitude(msg.altitude_m)
                self.ui_messenger.post(f"[{self.active_agent_id}] Altitude command {'accepted' if ok else 'failed'}.")

        for aid, ctrl in self.controllers.items():
            try:
                ctrl.close_connection()
            except Exception: pass

class Coordinator(WorkerThread):
    """Routes messages, manages capture mode and confirmation, and dispatches actions."""
    def __init__(self, inbox: queue.Queue, audio_ctrl_out: queue.Queue, nlu_outbox: queue.Queue, 
                 mav_outbox: queue.Queue, tts_outbox: queue.Queue, settings: dict, ui_messenger: UIMessenger):
        super().__init__("Coordinator", inbox)
        self.audio_ctrl_out = audio_ctrl_out
        self.nlu_outbox = nlu_outbox
        self.mav_outbox = mav_outbox
        self.tts_outbox = tts_outbox
        self.settings = settings
        self.ui_messenger = ui_messenger
        self.awaiting_yes_no = False
        self.pending_nlu: Optional[MsgNLUResult] = None
    
    def _speak(self, text: str):
        self.logger.info(f"SAY: {text}")  # Log the message; UI handler mirrors to GUI
        self.tts_outbox.put(MsgSpeak(text=text))
    
    # Helper to update the audio capture mode in the audio worker
    def _set_capture_mode(self, mode: CaptureMode):
        self.audio_ctrl_out.put(MsgSetCaptureMode(mode=mode))

    def _normalize_transcribed(self, text: str) -> str:
        if not text: return ""
        text = spoken_numbers_to_digits(text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(
            r"latitude\s*(-?\d+\.\d+)\s+longitude\s*(-?\d+\.\d+)",
            r"latitude \1, longitude \2", text, flags=re.IGNORECASE,
        )
        return text

    def _requires_confirmation(self, nlu: MsgNLUResult) -> bool:
        if nlu.intent == "SELECT_AGENT" and nlu.confidence > 0.85:
            return False
        if nlu.intent not in ("UNKNOWN_INTENT", "GENERIC_COMMAND", "PROVIDE_TARGET_DESCRIPTION"):
             return True
        return False

    def _handle_intent(self, nlu: MsgNLUResult):
        intent = nlu.intent
        entities = nlu.entities or {}
        conf = float(nlu.confidence or 0.0)
        conf_threshold = self.settings.get("nlu", {}).get("confidence_threshold", 0.7)

        if intent == "UNKNOWN_INTENT" or conf < conf_threshold:
            self._speak("Command not understood clearly. Please try again.")
            return

        if intent == "SELECT_AGENT":
            agent = entities.get("selected_agent_id")
            if agent: self.mav_outbox.put(MsgSelectAgent(agent_id=str(agent)))
            return

        if intent == "SET_AGENT_ALTITUDE":
            alt = entities.get("altitude_meters")
            if isinstance(alt, (int, float)):
                self.mav_outbox.put(MsgCommandSetAltitude(altitude_m=float(alt)))
            else:
                 self._speak("Altitude not specified.")
            return

        lat = entities.get("latitude")
        lon = entities.get("longitude")
        alt = entities.get("altitude_meters", self.settings.get("mission", {}).get("default_waypoint_altitude", 30.0))
        target_desc = entities.get("target_description_full")

        if lat is not None and lon is not None:
            if intent == "REQUEST_GRID_SEARCH":
                grid = entities.get("grid_size_meters")
                if grid:
                    self.mav_outbox.put(
                        MsgCommandGridSearch(
                            lat=float(lat), lon=float(lon), grid_size_m=int(grid),
                            swath_m=self.settings.get("mission", {}).get("default_swath_width", 20.0), 
                            altitude_m=float(alt), target_desc=target_desc
                        )
                    )
                else: self._speak("Grid size not specified.")
            elif intent in ("REQUEST_FLY_TO", "REQUEST_SEARCH_AT_LOCATION", "COMBINED_SEARCH_AND_TARGET"):
                 self.mav_outbox.put(
                    MsgCommandFlyTo(lat=float(lat), lon=float(lon), altitude_m=float(alt), target_desc=target_desc)
                )
        else:
            self._speak("Command understood, but GPS coordinates are missing.")

    def run(self):
        self._set_capture_mode(CaptureMode.COMMAND)
        self._speak("LIFEGUARD initialized. Press and Hold Push to Talk button to speak.")
        
        while not self.stopped():
            try:
                msg: MsgBase = self.inbox.get(timeout=0.5)
            except queue.Empty: continue

            if isinstance(msg, MsgShutdown): break
            if isinstance(msg, MsgSystemShutdownRequested):
                self._speak("Shutting down.")
                break

            if isinstance(msg, MsgSTTResult):
                text = self._normalize_transcribed(msg.text or "")
                self.ui_messenger.post(f"Operator: \"{text}\"")
                self.logger.info(f"STT Result ({msg.mode.value}): '{text}'")

                if not text:
                    if msg.mode == CaptureMode.COMMAND:
                         self._speak("Sorry, I didn't get that. Please try again.")
                    continue

                if msg.mode == CaptureMode.COMMAND:
                    self.nlu_outbox.put(MsgNLURequest(text=text))
                elif msg.mode == CaptureMode.YESNO:
                    self.awaiting_yes_no = False
                    self._set_capture_mode(CaptureMode.COMMAND)
                    if re.search(r"\b(yes|yeah|affirmative|confirm)\b", text, re.IGNORECASE):
                        if self.pending_nlu:
                            self._handle_intent(self.pending_nlu)
                    else:
                        self._speak("Cancelled.")
                    self.pending_nlu = None

            elif isinstance(msg, MsgNLUResult):
                self.logger.info(f"NLU Result: {msg.intent} (Confidence: {msg.confidence:.2f})")
                if self._requires_confirmation(msg) and msg.confidence >= self.settings.get("nlu", {}).get("confidence_threshold", 0.7):
                    self.pending_nlu = msg
                    self.awaiting_yes_no = True
                    self._set_capture_mode(CaptureMode.YESNO)
                    self._speak(f"Did you say: {msg.text}? Please say yes or no.")
                else:
                    self._handle_intent(msg)

            elif isinstance(msg, MsgMavStatus):
                self.logger.info(f"Forwarding MAV status: '{msg.text}'")
                self.ui_messenger.post(msg.text)  # Post directly to GUI log
                if "Reached" in msg.text or "Mission" in msg.text:
                    self._speak(msg.text)