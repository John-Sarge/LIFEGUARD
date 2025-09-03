"""
Main controller: initializes subsystems (Audio, STT, NLU, MAVLink, TTS, Coordinator),
starts threads, and coordinates graceful shutdown.
"""
import threading
import queue
import logging
import logging.config
from lifeguard.config import settings
from lifeguard.system.state import CaptureMode
from lifeguard.system.workers import (
    AudioInputWorker,
    STTWorker,
    NLUWorker,
    MavlinkWorker,
    TTSWorker,
    Coordinator,
)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            'datefmt': '%Y-%m-%dT%H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

def main():
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Starting LIFEGUARD system...")

    # Queues for inter-thread communication
    audio_ctrl_queue = queue.Queue()
    stt_in_queue = queue.Queue()      # Audio -> STT inbox
    coord_in_queue = queue.Queue()    # STT/NLU -> Coordinator inbox
    nlu_in_queue = queue.Queue()      # Coordinator -> NLU inbox
    mav_in_queue = queue.Queue()      # Coordinator -> MAV inbox
    tts_in_queue = queue.Queue()      # Coordinator/MAV -> TTS inbox

    # Initialize subsystem threads using workers
    tts_thread = TTSWorker(inbox=tts_in_queue)
    audio_thread = AudioInputWorker(inbox=audio_ctrl_queue, stt_out=stt_in_queue, coord_out=coord_in_queue)
    stt_thread = STTWorker(inbox=stt_in_queue, coord_out=coord_in_queue)
    nlu_thread = NLUWorker(inbox=nlu_in_queue, coord_out=coord_in_queue)
    mav_thread = MavlinkWorker(inbox=mav_in_queue, tts_out=tts_in_queue)
    coord_thread = Coordinator(
        inbox=coord_in_queue,
        audio_ctrl_out=audio_ctrl_queue,
        nlu_outbox=nlu_in_queue,
        mav_outbox=mav_in_queue,
        tts_outbox=tts_in_queue,
    )

    # Start all threads
    for t in (tts_thread, mav_thread, nlu_thread, stt_thread, audio_thread, coord_thread):
        t.start()

    logger.info("Subsystems initialized. Ready for integration.")

    # Main loop: wait for Coordinator to finish or until interrupted
    try:
        while True:
            try:
                coord_thread.join(timeout=0.5)
                if not coord_thread._thread.is_alive():
                    break
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down.")
                break
    finally:
        logger.info("Main loop exiting. Initiating shutdown.")
        if mav_thread:
            mav_thread.set_guided_on_exit()

        for t in (coord_thread, audio_thread, stt_thread, nlu_thread, mav_thread, tts_thread):
            t.stop()
        for t in (coord_thread, audio_thread, stt_thread, nlu_thread, mav_thread, tts_thread):
            t.join(timeout=3)
        logger.info("All subsystem threads stopped.")
        logger.info("LIFEGUARD system shutdown complete.")
