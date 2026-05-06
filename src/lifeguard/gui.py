# LIFEGUARD GUI: settings dialog and main mission control window.
import customtkinter as ctk
from tkintermapview import TkinterMapView
from PIL import Image, ImageDraw, ImageTk
import queue
import os
import sys

def resource_path(relative_path):
    """Get absolute path to bundled/static resources (dev and PyInstaller)."""
    try:
        base_path = sys._MEIPASS
        internal_path = os.path.join(base_path, '_internal', relative_path)
        if os.path.exists(internal_path):
            return internal_path
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

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
        self.tab_view.add("Ship")

        self.setup_agents_tab()
        self.setup_mission_tab()
        self.setup_advanced_tab()
        self.setup_ship_tab()

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

    def setup_ship_tab(self):
        tab = self.tab_view.tab("Ship")
        tab.grid_columnconfigure(1, weight=1)
        ship = self.config_manager.get("ship", {})

        ctk.CTkLabel(tab, text="Ship Name:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.ship_name_entry = ctk.CTkEntry(tab)
        self.ship_name_entry.insert(0, ship.get("name", "ship"))
        self.ship_name_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(tab, text="MAVLink Connection:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.ship_conn_entry = ctk.CTkEntry(tab, width=300)
        self.ship_conn_entry.insert(0, ship.get("connection_string", ""))
        self.ship_conn_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(tab, text="Track History (minutes):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.ship_track_hist_entry = ctk.CTkEntry(tab)
        self.ship_track_hist_entry.insert(0, ship.get("track_history_minutes", 30))
        self.ship_track_hist_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(tab, text="MOB Corridor Half-Width (m):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.ship_mob_corridor_entry = ctk.CTkEntry(tab)
        self.ship_mob_corridor_entry.insert(0, ship.get("mob_corridor_half_width_m", 50.0))
        self.ship_mob_corridor_entry.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
    
    def save_and_exit(self):
        new_agents = [{"name": entry["name"].get(), "connection_string": entry["conn"].get()} for entry in self.agent_entries]
        
        current_settings = self.config_manager.load_settings()
        current_settings.update({
            "agents": new_agents,
            "mission": { "default_waypoint_altitude": float(self.altitude_entry.get()), "default_swath_width": float(self.swath_entry.get()) },
            "nlu": { "confidence_threshold": float(self.confidence_entry.get()) },
            "ship": {
                "name": self.ship_name_entry.get(),
                "connection_string": self.ship_conn_entry.get(),
                "track_history_minutes": int(float(self.ship_track_hist_entry.get())),
                "mob_corridor_half_width_m": float(self.ship_mob_corridor_entry.get()),
            },
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
        
        subtitle_label = ctk.CTkLabel(header_frame, text="Lightweight Intent-Focused Engine for Guidance in\nUnmanned Autonomous Rescue Deployments: Yard Patrol", font=ctk.CTkFont(size=20))
        subtitle_label.grid(row=1, column=1, sticky="w")
        
        self.settings_button = ctk.CTkButton(header_frame, text="Settings", command=self.open_settings_window)
        self.settings_button.grid(row=0, column=2, rowspan=2, padx=10)

        self.log_textbox = ctk.CTkTextbox(self, state="disabled", font=("Consolas", 16))
        self.log_textbox.grid(row=1, column=0, padx=(10, 5), pady=(0,10), sticky="nsew")

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, padx=(10, 5), pady=10, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        self.ptt_button = ctk.CTkButton(button_frame, text="Push to Talk", height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.ptt_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.mob_callback = None
        self.mob_button = ctk.CTkButton(
            button_frame, text="MAN OVERBOARD", height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="firebrick", hover_color="#8B0000",
            command=self._on_mob_press,
        )
        self.mob_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        map_frame = ctk.CTkFrame(self)
        map_frame.grid(row=1, column=1, rowspan=2, padx=(5, 10), pady=(0, 10), sticky="nsew")
        map_frame.grid_rowconfigure(0, weight=1)
        map_frame.grid_columnconfigure(0, weight=1)

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
        self.ship_marker = None
        self.ship_track_path = None
        self._drone_base_img = self._make_drone_icon(38)
        self._ship_base_img = self._make_ship_icon(42)
        self._drone_icon = self._rotate_icon(self._drone_base_img, 0.0)
        self._ship_icon = self._rotate_icon(self._ship_base_img, 0.0)
        self._ship_icon_ref: ImageTk.PhotoImage = self._ship_icon
        self._agent_headings: dict = {}   # agent_id → last known heading (deg)
        self._agent_icon_refs: dict = {}  # agent_id → live PhotoImage (prevents GC)
        self._ship_heading: float = 0.0
        
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

    @staticmethod
    def _make_drone_icon(size: int = 38) -> Image.Image:
        """Draw a top-down quadcopter as a PIL Image pointing north (bow up)."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        arm = int(size * 0.30)
        r_body = max(3, size // 10)
        r_rotor = max(4, size // 8)
        color = "#00CFFF"
        lw = max(2, size // 18)
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            ex, ey = cx + dx * arm, cy + dy * arm
            draw.line([(cx, cy), (ex, ey)], fill=color, width=lw)
            draw.ellipse([ex - r_rotor, ey - r_rotor, ex + r_rotor, ey + r_rotor],
                         outline=color, width=lw)
        draw.ellipse([cx - r_body, cy - r_body, cx + r_body, cy + r_body], fill=color)
        # White forward-indicator triangle pointing north (up)
        tri_half = max(2, size // 14)
        tip_y = cy - r_body - max(3, size // 8)
        draw.polygon([(cx - tri_half, cy - r_body), (cx + tri_half, cy - r_body),
                      (cx, tip_y)], fill="white")
        return img

    @staticmethod
    def _make_ship_icon(size: int = 42) -> Image.Image:
        """Draw a top-down vessel as a PIL Image pointing north (bow up)."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx = size // 2
        hull = [
            (cx, 1),
            (int(size * 0.78), int(size * 0.28)),
            (int(size * 0.82), int(size * 0.60)),
            (int(size * 0.72), int(size * 0.92)),
            (int(size * 0.28), int(size * 0.92)),
            (int(size * 0.18), int(size * 0.60)),
            (int(size * 0.22), int(size * 0.28)),
        ]
        draw.polygon(hull, fill="orange")
        bw, bh = int(size * 0.32), int(size * 0.22)
        bx, by = cx - bw // 2, int(size * 0.50)
        draw.rectangle([bx, by, bx + bw, by + bh], fill="#8B4500")
        return img

    @staticmethod
    def _rotate_icon(base_img: Image.Image, heading_deg: float) -> ImageTk.PhotoImage:
        """Rotate base_img clockwise by heading_deg and return a PhotoImage."""
        rotated = base_img.rotate(-heading_deg, resample=Image.BICUBIC, expand=False)
        return ImageTk.PhotoImage(rotated)

    def update_agent_on_map(self, agent_id, lat, lon, heading_deg=None):
        # Regenerate rotated icon only when heading changes by ≥5°
        if heading_deg is not None:
            last = self._agent_headings.get(agent_id)
            if last is None:
                should_update = True
            else:
                diff = abs(heading_deg - last) % 360.0
                diff = min(diff, 360.0 - diff)
                should_update = diff >= 5.0
            if should_update:
                new_icon = self._rotate_icon(self._drone_base_img, heading_deg)
                self._agent_icon_refs[agent_id] = new_icon
                self._agent_headings[agent_id] = heading_deg
                if agent_id in self.agent_markers:
                    self.agent_markers[agent_id].change_icon(new_icon)
        if agent_id in self.agent_markers:
            self.agent_markers[agent_id].set_position(lat, lon)
        else:
            icon = self._agent_icon_refs.get(agent_id, self._drone_icon)
            marker = self.map_widget.set_marker(
                lat, lon, text=agent_id,
                icon=icon, icon_anchor="center",
            )
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
                        _, agent_id, lat, lon, heading_deg = message
                        self.update_agent_on_map(agent_id, lat, lon, heading_deg)
                    elif msg_type == "follow_agent_update":
                        _, agent_id = message
                        self.agent_to_follow = agent_id
                        self.add_log_message(f"Map is now following {agent_id}.")
                    elif msg_type == "path_update":
                        _, agent_id, position_list = message
                        self.update_agent_path(agent_id, position_list)
                    elif msg_type == "ship_map_update":
                        _, lat, lon, heading_deg = message
                        self._update_ship_on_map(lat, lon, heading_deg)
                    elif msg_type == "ship_track_update":
                        _, position_list = message
                        self._update_ship_track_on_map(position_list)
                else:
                    self.add_log_message(str(message))
        finally:
            self.after(100, self.process_queue)

    def _update_ship_on_map(self, lat, lon, heading_deg=None):
        # Regenerate rotated icon only when heading changes by ≥5°
        if heading_deg is not None:
            if self.ship_marker is None:
                should_update = True
            else:
                diff = abs(heading_deg - self._ship_heading) % 360.0
                diff = min(diff, 360.0 - diff)
                should_update = diff >= 5.0
            if should_update:
                new_icon = self._rotate_icon(self._ship_base_img, heading_deg)
                self._ship_icon_ref = new_icon
                self._ship_heading = heading_deg
                if self.ship_marker:
                    self.ship_marker.change_icon(new_icon)
        if self.ship_marker:
            self.ship_marker.set_position(lat, lon)
        else:
            ship_name = self.config_manager.get("ship", {}).get("name", "SHIP") or "SHIP"
            self.ship_marker = self.map_widget.set_marker(
                lat, lon, text=ship_name,
                icon=self._ship_icon_ref, icon_anchor="center",
            )

    def _update_ship_track_on_map(self, position_list):
        if self.ship_track_path:
            self.ship_track_path.delete()
            self.ship_track_path = None
        if len(position_list) >= 2:
            self.ship_track_path = self.map_widget.set_path(position_list, color="orange", width=3)
            
    def add_log_message(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _on_mob_press(self):
        if self.mob_callback:
            self.mob_callback()

    def open_settings_window(self):
        if not hasattr(self, "settings_window") or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.config_manager)
        self.settings_window.focus()