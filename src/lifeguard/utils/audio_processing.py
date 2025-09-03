"""Audio preprocessing utilities: noise reduction and bandpass filter."""
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import logging
from lifeguard.config import settings # --- ADDED ---

def apply_noise_reduction(audio_data_int16, sample_rate):
    logger = logging.getLogger(__name__)
    if audio_data_int16 is None:
        logger.warning("apply_noise_reduction called with None input.")
        return audio_data_int16
    try:
        audio_data_float = audio_data_int16.astype(np.float32) / 32767.0
        audio_data_float = np.nan_to_num(audio_data_float, nan=0.0, posinf=0.0, neginf=0.0)
        if audio_data_float.size == 0:
            logger.warning("apply_noise_reduction called with empty buffer.")
            return audio_data_int16
        energy = float(np.mean(np.abs(audio_data_float)))
        if energy < 1e-4:
            return audio_data_int16
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            reduced_audio_float = nr.reduce_noise(
                y=audio_data_float,
                sr=sample_rate,
                time_constant_s=settings.NOISE_REDUCE_TIME_CONSTANT_S,
                prop_decrease=settings.NOISE_REDUCE_PROP_DECREASE,
                n_fft=512,
            )
        reduced_audio_int16 = (reduced_audio_float * 32767.0).astype(np.int16)
        return reduced_audio_int16
    except (ValueError, FloatingPointError) as e:
        logger.warning(f"Noise reduction failed: {e}. Returning original audio.")
        return audio_data_int16
    except Exception as e:
        logger.error(f"Unexpected error in apply_noise_reduction: {e}")
        return audio_data_int16

def apply_bandpass_filter(audio_data, sample_rate):
    logger = logging.getLogger(__name__)
    try:
        nyq = 0.5 * sample_rate
        low = settings.BANDPASS_LOWCUT_HZ / nyq
        high = settings.BANDPASS_HIGHCUT_HZ / nyq
        b, a = butter(settings.BANDPASS_ORDER, [low, high], btype='band')
        filtered_audio = lfilter(b, a, audio_data)
        return filtered_audio.astype(audio_data.dtype)
    except Exception as e:
        logger.error(f"Bandpass filter error: {e}. Returning original audio.")
        return audio_data