# create_offline_db.py
import sqlite3
import requests
from PIL import Image
import io
import math
import time

# --- CONFIGURATION ---
# Define the GPS bounding box for your area of operations
# Bounding box for Annapolis, MD area
bounding_box = {
    "top_lat": 41.366,
    "bottom_lat": 41.34,
    "left_lon": -74.07,
    "right_lon": -74.028
}

# Define the zoom levels you want to download (e.g., 10=regional, 17=street level)
zoom_levels = range(12, 20) # Downloads zoom levels 12, 13, 14, 15, 16, 17

# The name of the output database file
DB_FILE = "map_cache.db"

# Must match tkintermapview's default tile_server template exactly
TILE_SERVER = "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
# --- END CONFIGURATION ---


# Helper function to convert GPS to tile numbers
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def download_tile(zoom, x, y):
    """Downloads a single map tile from OpenStreetMap."""
    url = f"https://a.tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    headers = {'User-Agent': 'Lifeguard Map Downloader/1.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            # Validate that it's a valid image
            Image.open(io.BytesIO(response.content))
            return response.content
    except Exception as e:
        print(f"  - Error downloading tile {zoom}/{x}/{y}: {e}")
    return None

def create_database():
    """Creates and populates the SQLite database with map tiles."""
    print(f"Creating database: {DB_FILE}")
    db = sqlite3.connect(DB_FILE)
    cursor = db.cursor()

    # Ensure schema is compatible; migrate if an old schema exists
    def ensure_schema():
        # server table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS server (
                url VARCHAR(300) PRIMARY KEY NOT NULL,
                max_zoom INTEGER NOT NULL
            );
            """
        )

        # Does tiles table exist?
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tiles';")
        has_tiles = cursor.fetchone() is not None

        if not has_tiles:
            # Create fresh tiles table with the expected schema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tiles (
                    zoom INTEGER NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    server VARCHAR(300) NOT NULL,
                    tile_image BLOB NOT NULL,
                    CONSTRAINT pk_tiles PRIMARY KEY (zoom, x, y, server)
                );
                """
            )
        else:
            # Inspect existing columns
            cursor.execute("PRAGMA table_info(tiles)")
            cols = [row[1] for row in cursor.fetchall()]
            needs_migration = not ("server" in cols and "tile_image" in cols)

            if needs_migration:
                print("[create_offline_db] Detected old tiles schema; migrating to new format...")
                # Rename old table, create new, migrate rows
                cursor.execute("ALTER TABLE tiles RENAME TO tiles_old;")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tiles (
                        zoom INTEGER NOT NULL,
                        x INTEGER NOT NULL,
                        y INTEGER NOT NULL,
                        server VARCHAR(300) NOT NULL,
                        tile_image BLOB NOT NULL,
                        CONSTRAINT pk_tiles PRIMARY KEY (zoom, x, y, server)
                    );
                    """
                )
                # Migrate existing data if it uses tile_data column
                cursor.execute("PRAGMA table_info(tiles_old)")
                old_cols = [row[1] for row in cursor.fetchall()]
                if "tile_data" in old_cols:
                    cursor.execute(
                        "INSERT INTO tiles (zoom, x, y, server, tile_image) "
                        "SELECT zoom, x, y, ?, tile_data FROM tiles_old;",
                        (TILE_SERVER,)
                    )
                # Drop old table
                cursor.execute("DROP TABLE tiles_old;")

        # sections table (not required for read but created for consistency)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sections (
                position_a VARCHAR(100) NOT NULL,
                position_b VARCHAR(100) NOT NULL,
                zoom_a INTEGER NOT NULL,
                zoom_b INTEGER NOT NULL,
                server VARCHAR(300) NOT NULL,
                CONSTRAINT pk_tiles PRIMARY KEY (position_a, position_b, zoom_a, zoom_b, server)
            );
            """
        )

        # Ensure server row exists
        cursor.execute("INSERT OR IGNORE INTO server (url, max_zoom) VALUES (?, ?)", (TILE_SERVER, max(zoom_levels)))
        db.commit()

    ensure_schema()

    total_tiles = 0
    
    for zoom in zoom_levels:
        top_left = deg2num(bounding_box["top_lat"], bounding_box["left_lon"], zoom)
        bottom_right = deg2num(bounding_box["bottom_lat"], bounding_box["right_lon"], zoom)

        x_range = range(top_left[0], bottom_right[0] + 1)
        y_range = range(top_left[1], bottom_right[1] + 1)
        
        tile_count_for_zoom = len(x_range) * len(y_range)
        total_tiles += tile_count_for_zoom
        print(f"\nDownloading {tile_count_for_zoom} tiles for zoom level {zoom}...")
        
        count = 0
        for x in x_range:
            for y in y_range:
                count += 1
                # Check if tile already exists for this server
                cursor.execute("SELECT 1 FROM tiles WHERE zoom=? AND x=? AND y=? AND server=?", (zoom, x, y, TILE_SERVER))
                if cursor.fetchone():
                    print(f"  ({count}/{tile_count_for_zoom}) Skipping existing tile {zoom}/{x}/{y}")
                    continue

                print(f"  ({count}/{tile_count_for_zoom}) Downloading tile {zoom}/{x}/{y}")
                tile_data = download_tile(zoom, x, y)
                if tile_data:
                    cursor.execute(
                        "INSERT INTO tiles (zoom, x, y, server, tile_image) VALUES (?, ?, ?, ?, ?)",
                        (zoom, x, y, TILE_SERVER, tile_data),
                    )
                    db.commit()
                time.sleep(0.1) # Be nice to the tile server

    print(f"\nFinished. Downloaded a total of {total_tiles} tiles.")
    db.close()

if __name__ == "__main__":
    create_database()