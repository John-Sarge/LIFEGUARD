# lifeguard.spec (Final Version)

import os
import sys
import spacy
import en_core_web_sm
from PyInstaller.utils.hooks import get_package_paths, collect_data_files

spec_root = os.getcwd()
block_cipher = None
spacy_model_path = en_core_web_sm.__path__[0]
vosk_pkg_path = get_package_paths('vosk')[0]
pyaudio_pkg_path = get_package_paths('pyaudio')[0]

a = Analysis(
    ['run_gui.py'],
    pathex=['src'],
    binaries=[
        (os.path.join(vosk_pkg_path, 'vosk'), 'vosk')
    ],
    datas=[
        # App data
        (os.path.join(spec_root, 'config.json'), '.'),
        (os.path.join(spec_root, 'lifeguard_logo.png'), '.'),
        (os.path.join(spec_root, 'lifeguard.ico'), '.'),
        (os.path.join(spec_root, 'map_cache.db'), '.')
        
        # AI Models
        (os.path.join(spec_root, 'vosk_models/vosk-model-small-en-us-0.15'), 'vosk_models/vosk-model-small-en-us-0.15'),
        (spacy_model_path, 'en_core_web_sm'),
        
        # Bundle the entire pyaudio and pymavlink packages
        (pyaudio_pkg_path, 'pyaudio'),
    ] + collect_data_files('pymavlink'),

    hiddenimports=['pynput.keyboard._win32', 'pynput.mouse._win32', 'comtypes', 'customtkinter'],
    
    # Use the runtime hook to set DLL path for PyAudio on Windows
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_audio.py'],
    
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='LIFEGUARD',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    windowed=True,
    icon='lifeguard.ico',
)