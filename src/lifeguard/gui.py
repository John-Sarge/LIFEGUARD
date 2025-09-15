# LIFEGUARD GUI: Mission control interface for agents and map display
import customtkinter as ctk
from tkintermapview import TkinterMapView
from PIL import Image
import queue
import os
import sys

def resource_path(relative_path):
    """Get absolute path to bundled/static resources (dev and PyInstaller)."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        # Prefer resources inside the optional _internal folder first
        internal_path = os.path.join(base_path, '_internal', relative_path)
        if os.path.exists(internal_path):
            return internal_path
    except Exception:
        # In a normal development environment, use the project's root
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Settings window (Toplevel dialog) for agents/mission/NLU options
class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master, config_manager):
        super().__init__(master)
        self.config_manager = config_manager
        self.transient(master)
        self.title("Settings")
        self.geometry("600x450")
        self.grab_set()

        self.agent_entries = []

        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(expand=True, fill="both", padx=10, pady=10)
        self.tab_view.add("Agents")
        self.tab_view.add("Mission")
        self.tab_view.add("Advanced")

        self.setup_agents_tab()
        self.setup_mission_tab()
        self.setup_advanced_tab()

        self.save_button = ctk.CTkButton(self, text="Save and Restart", command=self.save_and_exit)
        self.save_button.pack(pady=10)

    def setup_agents_tab(self):
        self.agent_scroll_frame = ctk.CTkScrollableFrame(self.tab_view.tab("Agents"), label_text="Configured Agents")
        self.agent_scroll_frame.pack(expand=True, fill="both", padx=5, pady=5)
        self.agent_scroll_frame.grid_columnconfigure(1, weight=1)

        for agent in self.config_manager.get("agents", []):
            self.add_agent_entry(agent)
        
        add_agent_button = ctk.CTkButton(self.tab_view.tab("Agents"), text="Add New Agent", command=lambda: self.add_agent_entry())
        add_agent_button.pack(pady=10)

    def add_agent_entry(self, agent=None):
        name = agent['name'] if agent else f"agent{len(self.agent_entries) + 1}"
        conn_str = agent['connection_string'] if agent else ""

        row = len(self.agent_entries) * 3
        
        name_label = ctk.CTkLabel(self.agent_scroll_frame, text="Agent Name:")
        name_label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        name_entry = ctk.CTkEntry(self.agent_scroll_frame)
        name_entry.insert(0, name)
        name_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        conn_label = ctk.CTkLabel(self.agent_scroll_frame, text="Connection:")
        conn_label.grid(row=row + 1, column=0, padx=5, pady=5, sticky="w")
        conn_entry = ctk.CTkEntry(self.agent_scroll_frame, width=300)
        conn_entry.insert(0, conn_str)
        conn_entry.grid(row=row + 1, column=1, padx=5, pady=5, sticky="ew")
        
        remove_button = ctk.CTkButton(self.agent_scroll_frame, text="Remove", command=lambda r=len(self.agent_entries): self.remove_agent_entry(r), fg_color="firebrick")
        remove_button.grid(row=row, column=2, rowspan=2, padx=5, pady=5)
        
        separator = ctk.CTkFrame(self.agent_scroll_frame, height=2, fg_color="gray20")
        separator.grid(row=row + 2, column=0, columnspan=3, pady=10, sticky="ew")

        entry_widgets = { "frame": [name_label, name_entry, conn_label, conn_entry, remove_button, separator], "name": name_entry, "conn": conn_entry }
        self.agent_entries.append(entry_widgets)

    def remove_agent_entry(self, row_index):
        entry_to_remove = self.agent_entries.pop(row_index)
        for widget in entry_to_remove["frame"]:
            widget.destroy()
        
        for i, entry in enumerate(self.agent_entries):
            new_row = i * 3
            entry["frame"][0].grid(row=new_row, column=0)
            entry["frame"][1].grid(row=new_row, column=1)
            entry["frame"][2].grid(row=new_row + 1, column=0)
            entry["frame"][3].grid(row=new_row + 1, column=1)
            entry["frame"][4].grid(row=new_row, column=2, rowspan=2)
            entry["frame"][5].grid(row=new_row + 2, column=0, columnspan=3)
            entry["frame"][4].configure(command=lambda r=i: self.remove_agent_entry(r))

    def setup_mission_tab(self):
        ctk.CTkLabel(self.tab_view.tab("Mission"), text="Default Waypoint Altitude (m):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.altitude_entry = ctk.CTkEntry(self.tab_view.tab("Mission"))
        self.altitude_entry.insert(0, self.config_manager.get("mission", {}).get("default_waypoint_altitude", 30.0))
        self.altitude_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ctk.CTkLabel(self.tab_view.tab("Mission"), text="Default Swath Width (m):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.swath_entry = ctk.CTkEntry(self.tab_view.tab("Mission"))
        self.swath_entry.insert(0, self.config_manager.get("mission", {}).get("default_swath_width", 20.0))
        self.swath_entry.grid(row=1, column=1, padx=10, pady=10)

    def setup_advanced_tab(self):
        ctk.CTkLabel(self.tab_view.tab("Advanced"), text="NLU Confidence Threshold (0.1-1.0):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.confidence_entry = ctk.CTkEntry(self.tab_view.tab("Advanced"))
        self.confidence_entry.insert(0, self.config_manager.get("nlu", {}).get("confidence_threshold", 0.7))
        self.confidence_entry.grid(row=0, column=1, padx=10, pady=10)
    
    def save_and_exit(self):
        new_agents = [{"name": entry["name"].get(), "connection_string": entry["conn"].get()} for entry in self.agent_entries]
        
        current_settings = self.config_manager.load_settings()
        current_settings.update({
            "agents": new_agents,
            "mission": { "default_waypoint_altitude": float(self.altitude_entry.get()), "default_swath_width": float(self.swath_entry.get()) },
            "nlu": { "confidence_threshold": float(self.confidence_entry.get()) },
        })
        
        self.config_manager.save_settings(current_settings)
        self.save_button.configure(text="Saved! Please restart Lifeguard.", fg_color="green")

class LifeguardGUI(ctk.CTk):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        
        self.title("LIFEGUARD Mission Control")
        self.iconbitmap(resource_path("lifeguard.ico"))
        self.geometry("1280x720")

        self.agent_to_follow = None
        self.follow_agent_checkbox_var = ctk.BooleanVar(value=True)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        logo_image = ctk.CTkImage(Image.open(resource_path("lifeguard_logo.png")), size=(128, 128))
        logo_label = ctk.CTkLabel(header_frame, image=logo_image, text="")
        logo_label.grid(row=0, column=0, rowspan=2, padx=(0, 20))
        
        title_label = ctk.CTkLabel(header_frame, text="LIFEGUARD", font=ctk.CTkFont(size=48, weight="bold"))
        title_label.grid(row=0, column=1, sticky="w")
        
        subtitle_label = ctk.CTkLabel(header_frame, text="Lightweight Intent-Focused Engine for Guidance in Unmanned Autonomous Rescue Deployments", font=ctk.CTkFont(size=20))
        subtitle_label.grid(row=1, column=1, sticky="w")
        
        self.settings_button = ctk.CTkButton(header_frame, text="Settings", command=self.open_settings_window)
        self.settings_button.grid(row=0, column=2, rowspan=2, padx=10)

        self.log_textbox = ctk.CTkTextbox(self, state="disabled", font=("Consolas", 16))
        self.log_textbox.grid(row=1, column=0, padx=(10, 5), pady=(0,10), sticky="nsew")

        self.ptt_button = ctk.CTkButton(self, text="Push to Talk", height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.ptt_button.grid(row=2, column=0, padx=(10, 5), pady=10, sticky="ew")
        
        map_frame = ctk.CTkFrame(self)
        map_frame.grid(row=1, column=1, rowspan=2, padx=(5, 10), pady=(0, 10), sticky="nsew")
        map_frame.grid_rowconfigure(0, weight=1)
        map_frame.grid_columnconfigure(0, weight=1)

        # Initialize map with database_path so background threads use the offline cache
        self.map_widget = TkinterMapView(
            map_frame,
            corner_radius=8,
            database_path=resource_path("map_cache.db"),
        )
        self.map_widget.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        
        self.follow_checkbox = ctk.CTkCheckBox(map_frame, text="Follow Tasked Agent", variable=self.follow_agent_checkbox_var)
        self.follow_checkbox.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.map_widget.set_position(38.98, -76.48)
        self.map_widget.set_zoom(12)

        self.agent_markers = {}
        self.agent_paths = {}
        
        footer_frame = ctk.CTkFrame(self, height=20, fg_color="transparent")
        footer_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")
        created_by_label = ctk.CTkLabel(
            footer_frame, 
            text="Created by John Seargeant - 2025", 
            font=ctk.CTkFont(size=12), 
            text_color="gray60"
        )
        created_by_label.pack(side="right", padx=10)

        self.gui_update_queue = queue.Queue()
        self.after(100, self.process_queue)

    def update_agent_on_map(self, agent_id, lat, lon):
        if agent_id in self.agent_markers:
            self.agent_markers[agent_id].set_position(lat, lon)
        else:
            marker = self.map_widget.set_marker(lat, lon, text=agent_id)
            self.agent_markers[agent_id] = marker
        
        if agent_id == self.agent_to_follow and self.follow_agent_checkbox_var.get():
            self.map_widget.set_position(lat, lon)

    def update_agent_path(self, agent_id, position_list):
        if agent_id in self.agent_paths:
            self.agent_paths[agent_id].delete()

        if len(position_list) >= 1:
            path = self.map_widget.set_path(position_list, color="cyan", width=2)
            self.agent_paths[agent_id] = path
            
            if len(position_list) > 1:
                min_lat = min(p[0] for p in position_list)
                max_lat = max(p[0] for p in position_list)
                min_lon = min(p[1] for p in position_list)
                max_lon = max(p[1] for p in position_list)
                
                self.map_widget.fit_bounding_box((max_lat, min_lon), (min_lat, max_lon))

    def process_queue(self):
        try:
            while not self.gui_update_queue.empty():
                message = self.gui_update_queue.get_nowait()
                if isinstance(message, tuple):
                    msg_type = message[0]
                    if msg_type == "map_update":
                        _, agent_id, lat, lon = message
                        self.update_agent_on_map(agent_id, lat, lon)
                    elif msg_type == "follow_agent_update":
                        _, agent_id = message
                        self.agent_to_follow = agent_id
                        self.add_log_message(f"Map is now following {agent_id}.")
                    elif msg_type == "path_update":
                        _, agent_id, position_list = message
                        self.update_agent_path(agent_id, position_list)
                else:
                    self.add_log_message(str(message))
        finally:
            self.after(100, self.process_queue)
            
    def add_log_message(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def open_settings_window(self):
        if not hasattr(self, "settings_window") or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.config_manager)
        self.settings_window.focus()