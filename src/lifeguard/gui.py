# LIFEGUARD GUI: settings dialog and main mission control window.
import customtkinter as ctk
from tkintermapview import TkinterMapView
from PIL import Image, ImageDraw, ImageTk
import queue
import os
import sys
import io
import sqlite3
import requests
import threading

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

def writable_data_path(relative_path):
    """Get path to a persistent, writable data file.

    For PyInstaller bundles, resolves to the directory containing the exe so
    that writes survive across restarts.  For dev, resolves to the working
    directory (same location create_offline_db.py writes to).
    """
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


TILE_SERVER_STREET = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
TILE_SERVER_SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


class CachingMapView(TkinterMapView):
    """TkinterMapView subclass that writes freshly-downloaded tiles back to the
    cache database so they persist across sessions.

    tkintermapview's built-in request_image reads from the DB but never writes
    to it.  This override adds the write-back step after a successful network
    fetch so that tiles accumulate in the DB over time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-thread DB connections keyed by database_path so that when
        # database_path changes (street ↔ satellite switch) each thread
        # automatically opens a fresh connection to the correct DB.
        self._local = threading.local()
        self._ensure_db_schema()

    def _get_cursor(self):
        """Return a per-thread cursor that always points to the current database_path."""
        if self.database_path is None:
            return None
        local = self._local
        # Reopen if path changed or connection not yet created for this thread
        if getattr(local, 'db_path', None) != self.database_path:
            try:
                if getattr(local, 'db_conn', None) is not None:
                    local.db_conn.close()
            except Exception:
                pass
            local.db_conn = sqlite3.connect(self.database_path, timeout=5, check_same_thread=False)
            local.db_cursor = local.db_conn.cursor()
            local.db_path = self.database_path
        return local.db_cursor

    def _ensure_db_schema(self):
        """Create the tiles/server tables if the DB file is brand-new."""
        if self.database_path is None:
            return
        try:
            with sqlite3.connect(self.database_path, timeout=5) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS server ("
                    "  url VARCHAR(300) PRIMARY KEY NOT NULL,"
                    "  max_zoom INTEGER NOT NULL"
                    ")"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS tiles ("
                    "  zoom INTEGER NOT NULL,"
                    "  x INTEGER NOT NULL,"
                    "  y INTEGER NOT NULL,"
                    "  server VARCHAR(300) NOT NULL,"
                    "  tile_image BLOB NOT NULL,"
                    "  CONSTRAINT pk_tiles PRIMARY KEY (zoom, x, y, server)"
                    ")"
                )
                conn.execute(
                    "INSERT OR IGNORE INTO server (url, max_zoom) VALUES (?, ?)",
                    (self.tile_server, 19),
                )
        except Exception:
            pass

    def request_image(self, zoom: int, x: int, y: int, db_cursor=None):
        key = f"{zoom}{x}{y}"

        # 1. In-memory cache
        if key in self.tile_image_cache:
            return self.tile_image_cache[key]

        # 2. Database read — always use our own cursor (tracks current database_path),
        #    ignoring the stale cursor the parent thread opened at widget startup.
        cursor = self._get_cursor()
        if cursor is not None:
            try:
                cursor.execute(
                    "SELECT t.tile_image FROM tiles t "
                    "WHERE t.zoom=? AND t.x=? AND t.y=? AND t.server=?;",
                    (zoom, x, y, self.tile_server),
                )
                row = cursor.fetchone()
                if row is not None:
                    image_tk = ImageTk.PhotoImage(Image.open(io.BytesIO(row[0])))
                    self.tile_image_cache[key] = image_tk
                    return image_tk
                if self.use_database_only:
                    return self.empty_tile_image
            except sqlite3.OperationalError:
                if self.use_database_only:
                    return self.empty_tile_image
            except Exception:
                return self.empty_tile_image

        # 3. Network fetch + write-back to DB
        try:
            url = (self.tile_server
                   .replace("{x}", str(x))
                   .replace("{y}", str(y))
                   .replace("{z}", str(zoom)))
            response = requests.get(url, headers={"User-Agent": "Lifeguard Map Viewer/1.0 (github.com/lifeguard-yp)"}, timeout=10)
            if response.status_code != 200:
                return self.empty_tile_image
            tile_bytes = response.content
            image = Image.open(io.BytesIO(tile_bytes))

            if self.overlay_tile_server is not None:
                # Overlay tiles are composited; don't bake overlay into the DB
                overlay_url = (self.overlay_tile_server
                               .replace("{x}", str(x))
                               .replace("{y}", str(y))
                               .replace("{z}", str(zoom)))
                image_overlay = Image.open(
                    requests.get(overlay_url, headers={"User-Agent": "TkinterMapView"}, timeout=10).raw
                )
                image = image.convert("RGBA")
                image_overlay = image_overlay.convert("RGBA")
                if image_overlay.size != (self.tile_size, self.tile_size):
                    image_overlay = image_overlay.resize((self.tile_size, self.tile_size), Image.LANCZOS)
                image.paste(image_overlay, (0, 0), image_overlay)
            elif self.database_path is not None:
                # Persist the clean base tile so future sessions skip the download
                try:
                    with sqlite3.connect(self.database_path, timeout=5) as conn:
                        conn.execute(
                            "INSERT OR IGNORE INTO tiles (zoom, x, y, server, tile_image) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (zoom, x, y, self.tile_server, tile_bytes),
                        )
                except Exception:
                    pass

            if not self.running:
                return self.empty_tile_image

            image_tk = ImageTk.PhotoImage(image)
            self.tile_image_cache[key] = image_tk
            return image_tk

        except Image.UnidentifiedImageError:
            self.tile_image_cache[key] = self.empty_tile_image
            return self.empty_tile_image
        except requests.exceptions.ConnectionError:
            return self.empty_tile_image
        except Exception:
            return self.empty_tile_image


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
        frame_type_val = agent.get('frame_type', 'UAV') if agent else 'UAV'

        row = len(self.agent_entries) * 4  # 4 rows per agent

        name_label = ctk.CTkLabel(self.agent_scroll_frame, text="Agent Name:")
        name_label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        name_entry = ctk.CTkEntry(self.agent_scroll_frame)
        name_entry.insert(0, name)
        name_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        ft_label = ctk.CTkLabel(self.agent_scroll_frame, text="Vehicle Type:")
        ft_label.grid(row=row + 1, column=0, padx=5, pady=5, sticky="w")
        ft_var = ctk.StringVar(value=frame_type_val)
        ft_menu = ctk.CTkOptionMenu(self.agent_scroll_frame, variable=ft_var,
                                    values=["UAV", "USV", "UGV", "UUV", "Ship"])
        ft_menu.grid(row=row + 1, column=1, padx=5, pady=5, sticky="w")

        conn_label = ctk.CTkLabel(self.agent_scroll_frame, text="Connection:")
        conn_label.grid(row=row + 2, column=0, padx=5, pady=5, sticky="w")
        conn_entry = ctk.CTkEntry(self.agent_scroll_frame, width=300)
        conn_entry.insert(0, conn_str)
        conn_entry.grid(row=row + 2, column=1, padx=5, pady=5, sticky="ew")

        remove_button = ctk.CTkButton(self.agent_scroll_frame, text="Remove",
                                      command=lambda r=len(self.agent_entries): self.remove_agent_entry(r),
                                      fg_color="firebrick")
        remove_button.grid(row=row, column=2, rowspan=3, padx=5, pady=5)

        separator = ctk.CTkFrame(self.agent_scroll_frame, height=2, fg_color="gray20")
        separator.grid(row=row + 3, column=0, columnspan=3, pady=10, sticky="ew")

        entry_widgets = {
            "frame": [name_label, name_entry, ft_label, ft_menu, conn_label, conn_entry, remove_button, separator],
            "name": name_entry,
            "conn": conn_entry,
            "frame_type": ft_var,
        }
        self.agent_entries.append(entry_widgets)

    def remove_agent_entry(self, row_index):
        entry_to_remove = self.agent_entries.pop(row_index)
        for widget in entry_to_remove["frame"]:
            widget.destroy()

        for i, entry in enumerate(self.agent_entries):
            new_row = i * 4
            entry["frame"][0].grid(row=new_row, column=0)           # name_label
            entry["frame"][1].grid(row=new_row, column=1)           # name_entry
            entry["frame"][2].grid(row=new_row + 1, column=0)       # ft_label
            entry["frame"][3].grid(row=new_row + 1, column=1)       # ft_menu
            entry["frame"][4].grid(row=new_row + 2, column=0)       # conn_label
            entry["frame"][5].grid(row=new_row + 2, column=1)       # conn_entry
            entry["frame"][6].grid(row=new_row, column=2, rowspan=3) # remove_button
            entry["frame"][7].grid(row=new_row + 3, column=0, columnspan=3)  # separator
            entry["frame"][6].configure(command=lambda r=i: self.remove_agent_entry(r))

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

        ctk.CTkLabel(tab, text="MOB Takeoff Altitude (m):").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.ship_mob_takeoff_alt_entry = ctk.CTkEntry(tab)
        self.ship_mob_takeoff_alt_entry.insert(0, ship.get("mob_takeoff_altitude_m", 100.0))
        self.ship_mob_takeoff_alt_entry.grid(row=4, column=1, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(tab, text="MOB Climb Speed (m/s):").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.ship_mob_climb_speed_entry = ctk.CTkEntry(tab)
        self.ship_mob_climb_speed_entry.insert(0, ship.get("mob_climb_speed_ms", 8.0))
        self.ship_mob_climb_speed_entry.grid(row=5, column=1, padx=10, pady=10, sticky="ew")
    
    def save_and_exit(self):
        new_agents = [{"name": entry["name"].get(), "connection_string": entry["conn"].get(), "frame_type": entry["frame_type"].get()} for entry in self.agent_entries]
        
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
                "mob_takeoff_altitude_m": float(self.ship_mob_takeoff_alt_entry.get()),
                "mob_climb_speed_ms": float(self.ship_mob_climb_speed_entry.get()),
            },
        })
        
        self.config_manager.save_settings(current_settings)
        self.save_button.configure(text="Saved! Please restart LIFEGUARD.", fg_color="green")

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

        self.map_widget = CachingMapView(
            map_frame,
            corner_radius=8,
            database_path=writable_data_path("map_cache.db"),
        )
        self.map_widget.grid(row=0, column=0, columnspan=2, sticky="nsew")

        self._satellite_mode = False

        self.follow_checkbox = ctk.CTkCheckBox(map_frame, text="Follow Tasked Agent", variable=self.follow_agent_checkbox_var)
        self.follow_checkbox.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.satellite_button = ctk.CTkButton(
            map_frame, text="Satellite", width=100,
            command=self._toggle_satellite,
        )
        self.satellite_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

        self.map_widget.set_position(38.98, -76.48)
        self.map_widget.set_zoom(12)

        self.agent_markers = {}
        self.agent_paths = {}
        self.ship_marker = None
        self.ship_track_path = None
        self._drone_base_img = self._make_drone_icon(38)
        self._ship_base_img = self._make_ship_icon(42)
        self._home_ship_base_img = self._make_home_ship_icon(48)
        self._drone_icon = self._rotate_icon(self._drone_base_img, 0.0)
        self._ship_icon = self._rotate_icon(self._home_ship_base_img, 0.0)
        self._ship_icon_ref: ImageTk.PhotoImage = self._ship_icon
        self._agent_headings: dict = {}   # agent_id → last known heading (deg)
        self._agent_icon_refs: dict = {}  # agent_id → live PhotoImage (prevents GC)
        self._agent_frame_types: dict = {}  # agent_id → frame_type string
        self._ship_heading: float = 0.0
        # Base image lookup keyed by frame_type string
        self._base_icons: dict = {
            "UAV":  self._drone_base_img,
            "USV":  self._make_usv_icon(42),
            "UGV":  self._make_ugv_icon(38),
            "UUV":  self._make_uuv_icon(38),
            "Ship": self._ship_base_img,
        }
        
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
        """Draw a top-down vessel as a PIL Image pointing north (bow up). Haze grey — Ship agents."""
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
        draw.polygon(hull, fill="#8896A3")  # USN haze grey
        bw, bh = int(size * 0.32), int(size * 0.22)
        bx, by = cx - bw // 2, int(size * 0.50)
        draw.rectangle([bx, by, bx + bw, by + bh], fill="#6B7B8A")  # darker grey superstructure
        return img

    @staticmethod
    def _make_home_ship_icon(size: int = 48) -> Image.Image:
        """Draw the home/tracking vessel icon — haze grey hull with gold superstructure."""
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
        draw.polygon(hull, fill="#8896A3")  # USN haze grey hull
        bw, bh = int(size * 0.32), int(size * 0.22)
        bx, by = cx - bw // 2, int(size * 0.50)
        draw.rectangle([bx, by, bx + bw, by + bh], fill="#D4AF37")  # gold superstructure
        return img

    @staticmethod
    def _make_usv_icon(size: int = 42) -> Image.Image:
        """Draw a top-down USV (RHIB/speedboat) as a PIL Image pointing north (bow up).
        Narrow, elongated hull with a sharp bow — distinct from the wider ship silhouette."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx = size // 2
        # Narrow pointed bow, near-parallel sides, flat squared stern
        hull = [
            (cx, 1),                                  # bow tip
            (int(size * 0.64), int(size * 0.22)),     # bow-starboard shoulder
            (int(size * 0.66), int(size * 0.82)),     # stern-starboard corner
            (int(size * 0.34), int(size * 0.82)),     # stern-port corner
            (int(size * 0.36), int(size * 0.22)),     # bow-port shoulder
        ]
        draw.polygon(hull, fill="#2471A3")  # navy blue hull
        # Small central console/cabin block
        cw, ch = int(size * 0.20), int(size * 0.24)
        draw.rectangle(
            [cx - cw // 2, int(size * 0.44), cx + cw // 2, int(size * 0.44) + ch],
            fill="#1A5276",
        )
        # White forward-indicator line from bow tip
        draw.line([(cx, 1), (cx, int(size * 0.22))], fill="white", width=max(1, size // 20))
        return img

    @staticmethod
    def _make_ugv_icon(size: int = 38) -> Image.Image:
        """Draw a top-down ground vehicle as a PIL Image pointing north (forward up)."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        # Body rectangle
        bw, bh = int(size * 0.50), int(size * 0.72)
        bx, by = cx - bw // 2, cy - bh // 2
        draw.rectangle([bx, by, bx + bw, by + bh], fill="#4D7C3E")  # army green
        # 4 wheels at body corners
        wr = max(3, size // 9)
        for wx, wy in [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]:
            draw.ellipse([wx - wr, wy - wr, wx + wr, wy + wr], fill="#2C4A1E")
        # White forward-indicator arrow pointing north
        tri_half = max(2, size // 12)
        tip_y = by - max(3, size // 8)
        draw.polygon([(cx - tri_half, by), (cx + tri_half, by), (cx, tip_y)], fill="white")
        return img

    @staticmethod
    def _make_uuv_icon(size: int = 38) -> Image.Image:
        """Draw a top-down torpedo/UUV as a PIL Image pointing north (nose up)."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        # Elongated ellipse body
        ew, eh = int(size * 0.38), int(size * 0.84)
        draw.ellipse([cx - ew // 2, cy - eh // 2, cx + ew // 2, cy + eh // 2], fill="#1565C0")
        # Rear stabiliser fins
        fin_w = int(size * 0.22)
        fin_h = int(size * 0.16)
        tail_y = cy + eh // 2 - 1
        draw.polygon([(cx, tail_y), (cx - fin_w, tail_y + fin_h), (cx - fin_w // 4, tail_y)], fill="#0D47A1")
        draw.polygon([(cx, tail_y), (cx + fin_w, tail_y + fin_h), (cx + fin_w // 4, tail_y)], fill="#0D47A1")
        # Bright nose tip
        nose_r = max(2, size // 14)
        nose_y = cy - eh // 2
        draw.ellipse([cx - nose_r, nose_y - nose_r, cx + nose_r, nose_y + nose_r], fill="#BBDEFB")
        return img

    @staticmethod
    def _rotate_icon(base_img: Image.Image, heading_deg: float) -> ImageTk.PhotoImage:
        """Rotate base_img clockwise by heading_deg and return a PhotoImage."""
        rotated = base_img.rotate(-heading_deg, resample=Image.BICUBIC, expand=False)
        return ImageTk.PhotoImage(rotated)

    def update_agent_on_map(self, agent_id, lat, lon, heading_deg=None, frame_type=None):
        if frame_type is not None:
            self._agent_frame_types[agent_id] = frame_type
        current_frame = self._agent_frame_types.get(agent_id, "UAV")
        base_img = self._base_icons.get(current_frame, self._drone_base_img)

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
                new_icon = self._rotate_icon(base_img, heading_deg)
                self._agent_icon_refs[agent_id] = new_icon
                self._agent_headings[agent_id] = heading_deg
                if agent_id in self.agent_markers:
                    self.agent_markers[agent_id].change_icon(new_icon)
        if agent_id in self.agent_markers:
            self.agent_markers[agent_id].set_position(lat, lon)
        else:
            if agent_id not in self._agent_icon_refs:
                self._agent_icon_refs[agent_id] = self._rotate_icon(base_img, 0.0)
            icon = self._agent_icon_refs[agent_id]
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

    def process_queue(self):
        try:
            while not self.gui_update_queue.empty():
                message = self.gui_update_queue.get_nowait()
                if isinstance(message, tuple):
                    msg_type = message[0]
                    if msg_type == "map_update":
                        _, agent_id, lat, lon, heading_deg, frame_type = message
                        self.update_agent_on_map(agent_id, lat, lon, heading_deg, frame_type)
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
                new_icon = self._rotate_icon(self._home_ship_base_img, heading_deg)
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

    def _toggle_satellite(self):
        self._satellite_mode = not self._satellite_mode
        if self._satellite_mode:
            self.map_widget.database_path = writable_data_path("map_cache_satellite.db")
            self.map_widget._ensure_db_schema()
            self.map_widget.set_tile_server(TILE_SERVER_SATELLITE, max_zoom=19)
            self.satellite_button.configure(text="Street Map")
        else:
            self.map_widget.database_path = writable_data_path("map_cache.db")
            self.map_widget._ensure_db_schema()
            self.map_widget.set_tile_server(TILE_SERVER_STREET, max_zoom=19)
            self.satellite_button.configure(text="Satellite")

    def open_settings_window(self):
        if not hasattr(self, "settings_window") or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.config_manager)
        self.settings_window.focus()