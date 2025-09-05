# pyi_rth_audio.py
import os
import sys

# For frozen apps, add bundled PyAudio DLL directory to the OS search path
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    pyaudio_path = os.path.join(sys._MEIPASS, 'pyaudio')
    if os.path.exists(pyaudio_path):
        os.add_dll_directory(pyaudio_path)