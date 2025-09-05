"""Lightweight JSON-backed configuration manager for app settings."""
import json
import os

class ConfigManager:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.settings = self.load_settings()

    def load_settings(self):
        """Loads settings from the JSON file, creating it if it doesn't exist."""
        if not os.path.exists(self.config_path):
            print(f"Config file not found. Creating default '{self.config_path}'")
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading config file: {e}. Reverting to default settings.")
            return self._create_default_config()

    def save_settings(self, settings_dict):
        """Saves the given settings dictionary to the JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(settings_dict, f, indent=4)
            self.settings = settings_dict
            print("Settings saved successfully.")
        except IOError as e:
            print(f"Error saving config file: {e}")

    def get(self, key, default=None):
        """Gets a setting value by key."""
        return self.settings.get(key, default)

    def _create_default_config(self):
        """Creates and saves a default configuration file."""
        default_settings = {
            "agents": [
                {"name": "agent1", "connection_string": "tcp:10.24.5.232:5762"},
                {"name": "agent2", "connection_string": "tcp:10.24.5.232:5772"}
            ],
            "mavlink": {
                "baudrate": 57600,
                "source_system_id": 255
            },
            "mission": {
                "default_waypoint_altitude": 30.0,
                "default_swath_width": 20.0
            },
            "nlu": {
                "spacy_model_name": "en_core_web_sm",
                "confidence_threshold": 0.7
            },
            "audio": {
                "sample_rate": 16000,
                "read_chunk": 4096
            }
        }
        self.save_settings(default_settings)
        return default_settings