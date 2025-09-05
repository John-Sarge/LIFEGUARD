import customtkinter as ctk
import threading
import logging
import sys
from src.lifeguard.gui import LifeguardGUI
from src.lifeguard.config.config_manager import ConfigManager
from src.lifeguard.system.lifeguard_system import LifeguardSystem
from src.lifeguard.system.workers import UIMessenger

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S')

if __name__ == "__main__":
    setup_logging()
    
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    config_manager = ConfigManager()
    settings = config_manager.load_settings()
    app = LifeguardGUI(config_manager=config_manager)

    # Bridge for backend threads to post messages into the GUI log
    ui_messenger = UIMessenger(gui_queue=app.gui_update_queue)

    lifeguard_backend = LifeguardSystem(settings=settings, ui_messenger=ui_messenger)
    
    def on_ptt_press():
        if lifeguard_backend.audio:
            lifeguard_backend.audio.start_recording()

    def on_ptt_release():
        if lifeguard_backend.audio:
            lifeguard_backend.audio.stop_recording()
        
    def on_shutdown():
        logging.info("Shutdown requested from GUI...")
        lifeguard_backend.stop()
        app.destroy()
        sys.exit()

    app.ptt_button.bind("<ButtonPress-1>", lambda e: on_ptt_press())
    app.ptt_button.bind("<ButtonRelease-1>", lambda e: on_ptt_release())
    app.protocol("WM_DELETE_WINDOW", on_shutdown)

    backend_thread = threading.Thread(target=lifeguard_backend.start, daemon=True)
    backend_thread.start()

    app.mainloop()