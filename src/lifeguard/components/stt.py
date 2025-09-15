import os
import json
import numpy as np
import vosk
import logging
from lifeguard.utils.audio_processing import apply_noise_reduction, apply_bandpass_filter
from lifeguard.system.exceptions import STTError

class SpeechToText:
	"""Batch and streaming speech recognition using Vosk for LIFEGUARD."""
	def __init__(self, model_path, sample_rate):
		self.logger = logging.getLogger(__name__)
		self.sample_rate = sample_rate
		import sys
		import os
		actual_model_path = model_path
		if getattr(sys, 'frozen', False):
			base = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
			actual_model_path = os.path.join(base, 'vosk_models', 'vosk-model-small-en-us-0.15')
		if not os.path.exists(actual_model_path):
			self.logger.error(f"Vosk model not found at {actual_model_path}.")
			raise STTError(f"Vosk model not found at {actual_model_path}")
		try:
			self.model = vosk.Model(actual_model_path)
		except Exception as e:
			self.logger.error(f"Failed to load Vosk model: {e}")
			raise STTError(f"Failed to load Vosk model: {e}")
		try:
			self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
			self.recognizer.SetWords(True)
		except Exception as e:
			self.logger.error(f"Failed to initialize recognizer: {e}")
			raise STTError(f"Failed to initialize recognizer: {e}")
		self._streaming_recognizer = None

	def recognize_buffer(self, raw_audio_buffer_bytes) -> str:
		if not raw_audio_buffer_bytes:
			self.recognizer.Reset()
			return ""
		try:
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
			except Exception as e:
				self.logger.warning(f"Failed to parse STT result JSON: {e}")
				return ""
		except Exception as e:
			self.logger.error(f"STT recognition failed: {e}")
			return ""

	def start_utterance(self):
		try:
			self._streaming_recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
			self._streaming_recognizer.SetWords(True)
		except Exception as e:
			self.logger.error(f"Failed to start streaming recognizer: {e}")
			self._streaming_recognizer = None

	def accept_waveform(self, audio_chunk_bytes) -> str | None:
		if not self._streaming_recognizer:
			self.start_utterance()
		try:
			try:
				audio_np = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
				denoised = apply_noise_reduction(audio_np, self.sample_rate)
				filtered = apply_bandpass_filter(denoised, self.sample_rate)
				processed_bytes = filtered.tobytes()
			except Exception as e:
				self.logger.warning(f"Audio preprocessing failed: {e}")
				processed_bytes = audio_chunk_bytes

			if self._streaming_recognizer.AcceptWaveform(processed_bytes):
				res = self._streaming_recognizer.Result()
			else:
				res = self._streaming_recognizer.PartialResult()
			try:
				d = json.loads(res)
				return d.get('text', d.get('partial', ''))
			except Exception as e:
				self.logger.warning(f"Failed to parse streaming STT result JSON: {e}")
				return None
		except Exception as e:
			self.logger.error(f"Streaming STT recognition failed: {e}")
			return None

	def end_utterance_and_get_result(self) -> str:
		if not self._streaming_recognizer:
			return ""
		try:
			res = self._streaming_recognizer.FinalResult()
			self._streaming_recognizer = None
			try:
				d = json.loads(res)
				return d.get('text', '')
			except Exception as e:
				self.logger.warning(f"Failed to parse final streaming STT result JSON: {e}")
				return ""
		except Exception as e:
			self.logger.error(f"Final streaming STT recognition failed: {e}")
			self._streaming_recognizer = None
			return ""

	def close(self):
		try:
			self._streaming_recognizer = None
		except Exception as e:
			self.logger.warning(f"Error closing streaming recognizer: {e}")
		try:
			self.recognizer = None
		except Exception as e:
			self.logger.warning(f"Error closing recognizer: {e}")
		try:
			self.model = None
		except Exception as e:
			self.logger.warning(f"Error releasing model: {e}")
		try:
			import gc
			gc.collect()
		except Exception as e:
			self.logger.warning(f"Error during garbage collection: {e}")
