"""
Worker threads for the LIFEGUARD system.

Threads:
- TTSWorker: speaks operator feedback.
- AudioInputWorker: push-to-talk capture; streams audio chunks to STT.
- STTWorker: streaming Vosk transcription; emits final text.
- NLUWorker: spaCy-based intent/entity extraction.
- MavlinkWorker: manages agent connections and executes commands.
- Coordinator: orchestrates modes, confirmations, and routing.
"""
from __future__ import annotations

import queue
import threading
import logging
import re
from typing import Optional, Dict, Any

import pyaudio
from pynput import keyboard

from lifeguard.config import settings
from lifeguard.components.stt import SpeechToText
from lifeguard.components.nlu import NaturalLanguageUnderstanding
from lifeguard.components.mavlink_io import MavlinkController
from lifeguard.utils.text_normalization import spoken_numbers_to_digits
from lifeguard.system.state import (
    CaptureMode,
    MsgBase,
    MsgShutdown,
    MsgSystemShutdownRequested,
    MsgSetCaptureMode,
    MsgAudioData,
    MsgSTTResult,
    MsgNLURequest,
    MsgNLUResult,
    MsgSpeak,
    MsgSelectAgent,
    MsgCommandGridSearch,
    MsgCommandFlyTo,
    MsgCommandSetAltitude,
    MsgMavStatus,
)


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


class TTSWorker(WorkerThread):
    """Text-to-Speech output using pyttsx3 with console fallback."""
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

        while not self.stopped():
            try:
                msg: MsgBase = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue

            if isinstance(msg, MsgShutdown):
                break
            if isinstance(msg, MsgSpeak):
                text = msg.text or ""
                if self._engine:
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
                    except Exception as e:
                        self.logger.warning(f"TTS speak error: {e}")
                else:
                    self.logger.info(f"SAY: {text}")


class AudioInputWorker(WorkerThread):
    """Push-to-talk audio capture (SPACE to speak, ESC to request shutdown)."""
    def __init__(self, inbox: queue.Queue, stt_out: queue.Queue, coord_out: Optional[queue.Queue] = None):
        super().__init__("AudioInputWorker", inbox)
        self.stt_out = stt_out
        self.coord_out = coord_out
        self._pyaudio = None
        self._stream = None
        self._is_pressed = False
        self._mode: CaptureMode = CaptureMode.COMMAND
        self._keyboard_listener = None
        # --- FIX: Removed self._frames buffer, as it was causing duplication ---

    def _on_press(self, key):
        if key == keyboard.Key.space:
            if not self._is_pressed:
                self._is_pressed = True
                self._start_recording()
        elif key == keyboard.Key.esc:
            if self.coord_out:
                self.coord_out.put(MsgSystemShutdownRequested(reason="ESC pressed"))

    def _on_release(self, key):
        if key == keyboard.Key.space:
            if self._is_pressed:
                self._is_pressed = False
                self._stop_recording_and_emit()

    def _start_recording(self):
        # --- FIX: No longer need to reset the self._frames buffer ---
        try:
            if self._stream and not self._stream.is_active():
                self._stream.start_stream()
                self.logger.info("Recording started")
        except Exception as e:
            self.logger.error(f"Failed to start stream: {e}")

    def _stop_recording_and_emit(self):
        try:
            if self._stream and self._stream.is_active():
                self._stream.stop_stream()
        except Exception:
            pass
        
        # --- FIX: ONLY send the final marker. Do NOT resend buffered audio. ---
        self.stt_out.put(MsgAudioData(data=b"", mode=self._mode, final=True))
        self.logger.info("Recording stopped and final marker emitted")

    def run(self):
        try:
            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=settings.AUDIO_CHANNELS,
                rate=settings.AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=settings.AUDIO_FRAMES_PER_BUFFER,
                start=False, # Start stream manually
            )
            self.logger.info("Press and hold SPACE to speak, ESC to exit.")
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._keyboard_listener.start()
        except Exception as e:
            self.logger.critical(f"Failed to initialize PyAudio stream: {e}")
            return # Exit thread if audio device fails

        try:
            while not self.stopped():
                try:
                    msg: MsgBase = self.inbox.get(timeout=0.05)
                    if isinstance(msg, MsgShutdown):
                        break
                    if isinstance(msg, MsgSetCaptureMode):
                        self._mode = msg.mode
                        self.logger.info(f"Capture mode set to {self._mode}")
                except queue.Empty:
                    pass

                if self._is_pressed and self._stream and self._stream.is_active():
                    try:
                        chunk = self._stream.read(
                            settings.AUDIO_READ_CHUNK, exception_on_overflow=False
                        )
                        # --- FIX: No longer append to self._frames, just stream directly ---
                        self.stt_out.put(
                            MsgAudioData(data=chunk, mode=self._mode, final=False)
                        )
                    except Exception as e:
                        self.logger.warning(f"Audio read error: {e}")
        finally:
            if self._keyboard_listener: self._keyboard_listener.stop()
            if self._stream: self._stream.close()
            if self._pyaudio: self._pyaudio.terminate()


class STTWorker(WorkerThread):
    """Streams audio frames to Vosk STT and emits final transcriptions."""
    def __init__(self, inbox: queue.Queue, coord_out: queue.Queue):
        super().__init__("STTWorker", inbox)
        self.coord_out = coord_out
        self.stt = None

    def run(self):
        try:
            self.stt = SpeechToText(settings.VOSK_MODEL_PATH, settings.AUDIO_SAMPLE_RATE)
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
                        # No audio captured; still emit empty result to clear state
                        self.coord_out.put(MsgSTTResult(text="", mode=current_mode))
                else:
                    # Start streaming if not started
                    if not streaming_active:
                        self.stt.start_utterance()
                        streaming_active = True
                    partial = self.stt.accept_waveform(msg.data)
                    # We could forward partials if desired; skip for now

        try:
            if self.stt:
                self.stt.close()
        except Exception:
            pass


class NLUWorker(WorkerThread):
    """Runs spaCy NLU to derive intents/entities from transcribed text."""
    def __init__(self, inbox: queue.Queue, coord_out: queue.Queue):
        super().__init__("NLUWorker", inbox)
        self.coord_out = coord_out
        self.nlu = None

    def run(self):
        try:
            self.nlu = NaturalLanguageUnderstanding(settings.SPACY_MODEL_NAME)
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
    def __init__(self, inbox: queue.Queue, tts_out: queue.Queue):
        super().__init__("MavlinkWorker", inbox)
        self.tts_out = tts_out
        self.controllers: Dict[str, MavlinkController] = {}
        self.active_agent_id: Optional[str] = None
        self.active_missions: Dict[str, Optional[Dict[str, Any]]] = {}

    def _speak(self, text: str):
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
                source_ctrl.set_mode("GUIDED")
                source_ctrl.fly_to(current_pos['lat'], current_pos['lon'], current_pos['alt'])
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
        
        verify_altitude = max(15.0, settings.DEFAULT_WAYPOINT_ALTITUDE - 5)
        verification_msg = MsgCommandGridSearch(
            lat=lat, lon=lon, grid_size_m=50,
            swath_m=settings.DEFAULT_SWATH_WIDTH_M,
            altitude_m=verify_altitude,
            target_desc=original_target_desc # Pass the original target desc
        )
        
        # Reuse the non-blocking mission execution logic
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
                ok = ctrl.generate_and_upload_search_grid_mission(
                    msg.lat, msg.lon, msg.grid_size_m, msg.swath_m, msg.altitude_m
                )
            else:
                self._speak(f"Uploading waypoint to {agent_id}.")
                wp = [(msg.lat, msg.lon, msg.altitude_m, 16, 0.0, 10.0, 0.0, float('nan'))]
                ok = ctrl.upload_mission(wp)

            if not ok: raise Exception("Mission upload failed.")
            if not ctrl.set_mode("AUTO"): raise Exception("Set mode AUTO failed.")
            if not ctrl.arm_vehicle(): raise Exception("Arming failed.")
            if not ctrl.start_mission(): raise Exception("Mission start failed.")
            self._speak(f"Mission started for {agent_id}.")

            if msg.target_desc:
                self._speak(f"Waiting to reach first waypoint to send target details to {agent_id}.")
                if ctrl.wait_for_waypoint_reached(1, timeout_seconds=180):
                    ctrl.send_status_text(f"TARGET:{msg.target_desc}")
                    self._speak("Target details sent.")
                else:
                    self._speak("Timeout waiting for waypoint. Target details not sent.")
            self.logger.info(f"Mission for {agent_id} is executing.")
        except Exception as e:
            self.logger.error(f"Mission sequence for {agent_id} failed: {e}")
            self._speak(f"Mission for {agent_id} failed.")
            self.active_missions[agent_id] = None # Free up the agent on failure

    def run(self):
        for agent_id, cfg in settings.AGENT_CONNECTION_CONFIGS.items():
            try:
                ctrl = MavlinkController(
                    cfg["connection_string"], cfg.get("baudrate"), cfg.get("source_system_id", 255)
                )
                ctrl.connect()
                if ctrl.is_connected():
                    self.controllers[agent_id] = ctrl
                    self.logger.info(f"Connected to {agent_id}")
                    self.active_missions[agent_id] = None
            except Exception as e:
                self.logger.error(f"Connection to {agent_id} failed: {e}")

        if not self.controllers: self._speak("No agents available")
        candidate = settings.INITIAL_ACTIVE_AGENT_ID
        self.active_agent_id = candidate if candidate in self.controllers else (next(iter(self.controllers)) if self.controllers else None)
        if self.active_agent_id: self._speak(f"{self.active_agent_id}, is selected")

        import time
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
                                    self.tts_out.put(MsgMavStatus(text=f"{aid} reports: {text_val}"))
                    except Exception as e:
                        self.logger.warning(f"Error polling STATUSTEXT from {aid}: {e}")
                last_poll = now

            try:
                msg: MsgBase = self.inbox.get(timeout=0.2)
            except queue.Empty: continue

            if isinstance(msg, MsgShutdown): break
            
            active_controller = self._get_active()
            if not active_controller:
                if not isinstance(msg, MsgSelectAgent):
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
                    self._speak(f"{msg.agent_id}, is selected")
                else:
                    self._speak(f"Agent {msg.agent_id} not found")
            elif isinstance(msg, MsgCommandSetAltitude):
                ok = active_controller.set_altitude(msg.altitude_m)
                self._speak("Altitude updated" if ok else "Altitude update failed")

        for aid, ctrl in self.controllers.items():
            try:
                ctrl.close_connection()
            except Exception: pass


class Coordinator(WorkerThread):
    """Routes messages, manages capture mode and confirmation, and dispatches actions."""
    def __init__(
        self,
        inbox: queue.Queue,
        audio_ctrl_out: queue.Queue,
        nlu_outbox: queue.Queue,
        mav_outbox: queue.Queue,
        tts_outbox: queue.Queue,
    ):
        super().__init__("Coordinator", inbox)
        self.audio_ctrl_out = audio_ctrl_out
        self.nlu_outbox = nlu_outbox
        self.mav_outbox = mav_outbox
        self.tts_outbox = tts_outbox
        self.awaiting_yes_no = False
        self.pending_nlu: Optional[MsgNLUResult] = None

    def _speak(self, text: str):
        self.tts_outbox.put(MsgSpeak(text=text))

    def _set_capture_mode(self, mode: CaptureMode):
        self.audio_ctrl_out.put(MsgSetCaptureMode(mode=mode))

    def _normalize_transcribed(self, text: str) -> str:
        if not text:
            return ""
        text = spoken_numbers_to_digits(text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(
            r"latitude\s*(-?\d+\.\d+)\s+longitude\s*(-?\d+\.\d+)",
            r"latitude \1, longitude \2",
            text,
            flags=re.IGNORECASE,
        )
        return text

    def _requires_confirmation(self, nlu: MsgNLUResult) -> bool:
        if nlu.intent == "SELECT_AGENT" and nlu.confidence > 0.85:
            return False
        if nlu.intent not in ("UNKNOWN_INTENT", "GENERIC_COMMAND", "PROVIDE_TARGET_DESCRIPTION"):
             return True
        return False

    def _handle_intent(self, nlu: MsgNLUResult):
        """Dispatches a confirmed command to the appropriate worker."""
        intent = nlu.intent
        entities = nlu.entities or {}
        conf = float(nlu.confidence or 0.0)

        if intent == "UNKNOWN_INTENT" or conf < settings.NLU_CONFIDENCE_THRESHOLD:
            self._speak("Command not understood clearly. Please try again.")
            return

        if intent == "SELECT_AGENT":
            agent = entities.get("selected_agent_id")
            if agent:
                self.mav_outbox.put(MsgSelectAgent(agent_id=str(agent)))
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
        alt = entities.get("altitude_meters", settings.DEFAULT_WAYPOINT_ALTITUDE)
        
        target_desc = entities.get("target_description_full")

        if lat is not None and lon is not None:
            if intent == "REQUEST_GRID_SEARCH":
                grid = entities.get("grid_size_meters")
                if grid:
                    self.mav_outbox.put(
                        MsgCommandGridSearch(
                            lat=float(lat), lon=float(lon), grid_size_m=int(grid),
                            swath_m=settings.DEFAULT_SWATH_WIDTH_M, altitude_m=float(alt),
                            target_desc=target_desc  # --- FIX: Pass it here ---
                        )
                    )
                else:
                    self._speak("Grid size not specified.")
            elif intent in ("REQUEST_FLY_TO", "REQUEST_SEARCH_AT_LOCATION", "COMBINED_SEARCH_AND_TARGET"):
                 self.mav_outbox.put(
                    MsgCommandFlyTo(lat=float(lat), lon=float(lon), altitude_m=float(alt), 
                                  target_desc=target_desc) # --- FIX: Pass it here ---
                )
        else:
            self._speak("Command understood, but GPS coordinates are missing.")

    def run(self):
        self._set_capture_mode(CaptureMode.COMMAND)
        self._speak("LIFEGUARD initialized. Press and hold space to speak. Press escape to exit.")
        while not self.stopped():
            try:
                msg: MsgBase = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue

            if isinstance(msg, MsgShutdown): break
            if isinstance(msg, MsgSystemShutdownRequested):
                self._speak("Shutting down.")
                break

            if isinstance(msg, MsgSTTResult):
                text = self._normalize_transcribed(msg.text or "")
                self.logger.info(f"STT Result ({msg.mode}): '{text}'")

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
                if self._requires_confirmation(msg) and msg.confidence >= settings.NLU_CONFIDENCE_THRESHOLD:
                    self.pending_nlu = msg
                    self.awaiting_yes_no = True
                    self._set_capture_mode(CaptureMode.YESNO)
                    self._speak(f"Did you say: {msg.text}? Please say yes or no.")
                else:
                    self._handle_intent(msg)

            elif isinstance(msg, MsgMavStatus):
                self.logger.info(f"Forwarding MAV status to TTS: '{msg.text}'")
                self._speak(msg.text)
