"""Configuration for LIFEGUARD; values are read from environment with defaults."""
from decouple import config, Csv
import os

# STT Configuration
VOSK_MODEL_NAME = config('VOSK_MODEL_NAME', default='vosk-model-small-en-us-0.15')
VOSK_MODEL_PATH = config('VOSK_MODEL_PATH', default=os.path.join('vosk_models', VOSK_MODEL_NAME))
AUDIO_SAMPLE_RATE = config('AUDIO_SAMPLE_RATE', default=16000, cast=int)
AUDIO_CHANNELS = config('AUDIO_CHANNELS', default=1, cast=int)
AUDIO_FORMAT = None  # Set in audio input module if needed
AUDIO_FRAMES_PER_BUFFER = config('AUDIO_FRAMES_PER_BUFFER', default=8192, cast=int)
AUDIO_READ_CHUNK = config('AUDIO_READ_CHUNK', default=4096, cast=int)

# Audio Pre-processing
NOISE_REDUCE_TIME_CONSTANT_S = config('NOISE_REDUCE_TIME_CONSTANT_S', default=2.0, cast=float)
NOISE_REDUCE_PROP_DECREASE = config('NOISE_REDUCE_PROP_DECREASE', default=0.9, cast=float)
BANDPASS_LOWCUT_HZ = config('BANDPASS_LOWCUT_HZ', default=300.0, cast=float)
BANDPASS_HIGHCUT_HZ = config('BANDPASS_HIGHCUT_HZ', default=3400.0, cast=float)
BANDPASS_ORDER = config('BANDPASS_ORDER', default=5, cast=int)

# NLU Configuration
SPACY_MODEL_NAME = config('SPACY_MODEL_NAME', default='en_core_web_sm')
NLU_CONFIDENCE_THRESHOLD = config('NLU_CONFIDENCE_THRESHOLD', default=0.7, cast=float)

# MAVLink Configuration
MAVLINK_BAUDRATE = config('MAVLINK_BAUDRATE', default=115200, cast=int)
MAVLINK_SOURCE_SYSTEM_ID = config('MAVLINK_SOURCE_SYSTEM_ID', default=255, cast=int)
DEFAULT_WAYPOINT_ALTITUDE = config('DEFAULT_WAYPOINT_ALTITUDE', default=30.0, cast=float)
DEFAULT_SWATH_WIDTH_M = config('DEFAULT_SWATH_WIDTH_M', default=20.0, cast=float)

# Dynamic Agent Configuration
AGENT_CONNECTION_CONFIGS = {}
AGENT_NAMES = config('AGENTS', default='agent1,agent2', cast=Csv())

for agent_name in AGENT_NAMES:
    agent_name = agent_name.strip()
    connection_string_key = f"{agent_name.upper()}_CONNECTION_STRING"
    # Provide sensible defaults for agent1 and agent2 for backward compatibility
    default_conn_str = None
    if agent_name == 'agent1':
        default_conn_str = 'tcp:10.24.5.232:5762'
    elif agent_name == 'agent2':
        default_conn_str = 'tcp:10.24.5.232:5772'

    AGENT_CONNECTION_CONFIGS[agent_name] = {
        "connection_string": config(connection_string_key, default=default_conn_str),
        "source_system_id": MAVLINK_SOURCE_SYSTEM_ID,
        "baudrate": MAVLINK_BAUDRATE
    }

INITIAL_ACTIVE_AGENT_ID = config('INITIAL_ACTIVE_AGENT_ID', default='agent1')

# Earth radius (m)
EARTH_RADIUS_METERS = config('EARTH_RADIUS_METERS', default=6378137.0, cast=float)