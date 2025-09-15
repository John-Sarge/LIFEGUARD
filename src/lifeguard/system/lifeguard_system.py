"""High-level system wiring: spins up and shuts down all worker threads."""
import queue
import logging
from .workers import (
    AudioInputWorker, STTWorker, NLUWorker, MavlinkWorker, TTSWorker, Coordinator, UIMessenger
)

class LifeguardSystem:
    """Manages the lifecycle of all backend worker threads."""
    def __init__(self, settings: dict, ui_messenger: UIMessenger):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.ui_messenger = ui_messenger

    # Queues for inter-thread communication between backend subsystems
        self.audio_ctrl_queue = queue.Queue()
        self.stt_in_queue = queue.Queue()
        self.coord_in_queue = queue.Queue()
        self.nlu_in_queue = queue.Queue()
        self.mav_in_queue = queue.Queue()
        self.tts_in_queue = queue.Queue()

    # Initialize subsystem threads (settings and ui_messenger passed to relevant workers)
        self.tts = TTSWorker(inbox=self.tts_in_queue)
        self.audio = AudioInputWorker(inbox=self.audio_ctrl_queue, stt_out=self.stt_in_queue)
        self.stt = STTWorker(inbox=self.stt_in_queue, coord_out=self.coord_in_queue, settings=self.settings)
        self.nlu = NLUWorker(inbox=self.nlu_in_queue, coord_out=self.coord_in_queue, settings=self.settings)
        self.mav = MavlinkWorker(inbox=self.mav_in_queue, tts_out=self.tts_in_queue, settings=self.settings, ui_messenger=self.ui_messenger)
        self.coord = Coordinator(
            inbox=self.coord_in_queue,
            audio_ctrl_out=self.audio_ctrl_queue,
            nlu_outbox=self.nlu_in_queue,
            mav_outbox=self.mav_in_queue,
            tts_outbox=self.tts_in_queue,
            settings=self.settings,
            ui_messenger=self.ui_messenger
        )
        self.workers = [self.tts, self.mav, self.nlu, self.stt, self.audio, self.coord]

    def start(self):
        self.logger.info("Starting all Lifeguard backend threads...")
        for worker in self.workers:
            worker.start()

    def stop(self):
        self.logger.info("Stopping all Lifeguard backend threads...")
        if self.mav:
            self.mav.set_guided_on_exit()
        
        for worker in self.workers:
            worker.stop()
        
        for worker in self.workers:
            worker.join(timeout=2)
        self.logger.info("All backend threads stopped.")