# Agent grid search simulator: connects to a MAVLink vehicle (e.g., SITL) and
# reports a simulated FOUND status partway through an uploaded mission by
# emitting a STATUSTEXT like "FOUND:<lat>,<lon>" when a chosen waypoint is reached.

# created to test LIFEGUARD: https://github.com/John-Sarge/LIFEGUARD

import time
import re
import random
import math
from typing import Dict, Optional, Tuple, List
from pymavlink import mavutil

def _mode_is_auto(hb) -> bool:
    """Return True if the heartbeat indicates AUTO mode, with common fallbacks."""
    try:
        mode_str = mavutil.mode_string_v10(hb)
    except Exception:
        mode_str = None
    if mode_str and mode_str.upper() == 'AUTO':
        return True
    # Fallbacks for common ArduPilot flags/modes
    if hasattr(hb, 'custom_mode') and hb.custom_mode == 3:  # Copter AUTO
        return True
    if hasattr(hb, 'base_mode') and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED):
        return True
    return False


def _parse_mission_count_from_text(text: str) -> Optional[int]:
    # Parse mission count from STATUSTEXT lines like: "Mission: 10 WP" or "Mission: 2 WP".
    m = re.search(r"Mission:\s*(\d+)\s*WP", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _pull_mission_items(master: mavutil.mavfile, sysid: int, compid: int, timeout: float = 5.0) -> Dict[int, Tuple[float, float]]:
    """Pull mission items from the vehicle and map seq -> (lat, lon) in degrees.
    May include home at seq 0 if present.
    """

    items: Dict[int, Tuple[float, float]] = {}
    retries = 3
    for attempt in range(retries):
        try:
            master.mav.mission_request_list_send(sysid, compid, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
            count_msg = master.recv_match(type=['MISSION_COUNT'], blocking=True, timeout=timeout)
            if not count_msg:
                continue
            count = int(getattr(count_msg, 'count', 0))
            print(f"MISSION_COUNT received: {count}")
            for i in range(count):
                try:
                    master.mav.mission_request_int_send(sysid, compid, i, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
                    item = master.recv_match(type=['MISSION_ITEM_INT', 'MISSION_ITEM'], blocking=True, timeout=timeout)
                except Exception:
                    item = None
                if not item:
                    continue
                if item.get_type() == 'MISSION_ITEM_INT':
                    lat = float(item.x) / 1e7
                    lon = float(item.y) / 1e7
                    seq = int(item.seq)
                else:
                    lat = float(item.x)
                    lon = float(item.y)
                    seq = int(item.seq)
                items[seq] = (lat, lon)
            print(f"Pulled {len(items)} mission items (attempt {attempt+1}/{retries})")
            if len(items) == count:
                break  # Got all items
        except Exception:
            pass
        time.sleep(1)  # Wait before retrying
    return items


def simulate_found_midway(connection_string):
    """Connect to the vehicle, wait for a mission in AUTO, then report FOUND midway.

    Logic:
    - Connect and wait for heartbeat.
    - Infer mission presence via STATUSTEXT (Mission: N WP) and/or pull items.
    - Wait until mode is AUTO and mission is present/ready.
    - Choose a target waypoint index roughly in the middle half of the mission.
    - On MISSION_ITEM_REACHED for that index, emit a FOUND:<lat>,<lon> STATUSTEXT.
    """
    print(f"Connecting to agent at {connection_string}...")
    master = mavutil.mavlink_connection(
        connection_string,
        autoreconnect=True,
        use_native=False,
        force_connected=True,
        mavlink2=True,
    )
    hb = master.wait_heartbeat(timeout=10)
    if not hb:
        print("No heartbeat. Exiting.")
        return
    sysid = master.target_system or getattr(hb, 'sysid', 1)
    compid = master.target_component or getattr(hb, 'compid', 1)
    print("Connected. Waiting for mission upload and AUTO mode...")

    mission_count: Optional[int] = None
    mission_started = False
    in_auto = False
    target_idx: Optional[int] = None
    mission_items: Dict[int, Tuple[float, float]] = {}

    last_count_seen_at = 0.0
    last_count_value: Optional[int] = None
    mission_count_max: Optional[int] = None
    last_pull_attempt = 0.0

    def mission_is_ready() -> bool:
        # Either we have pulled items or the count has stabilized for a short period
        if mission_items and len(mission_items) > 0:
            return True
        if mission_count_max and (time.time() - last_count_seen_at) > 1.0:
            return True
        return False

    # Wait for mission upload and AUTO mode; don't rely solely on MISSION_COUNT (typically GCS->vehicle only)
    while not mission_started:
        msg = master.recv_match(type=['HEARTBEAT', 'STATUSTEXT', 'MISSION_CURRENT'], blocking=True, timeout=2)
        if not msg:
            continue
        mtype = msg.get_type()
        if mtype == 'HEARTBEAT':
            in_auto = _mode_is_auto(msg)
            if in_auto and (mission_count and mission_count > 0):
                print("AUTO mode detected and mission present. Mission started.")
                mission_started = True
        elif mtype == 'STATUSTEXT':
            text = msg.text.decode() if hasattr(msg.text, 'decode') else msg.text
            print(f"STATUSTEXT: {text}")
            # Infer mission count from STATUSTEXT (vehicle typically announces during upload)
            cnt = _parse_mission_count_from_text(text or '')
            if cnt is not None:
                mission_count = cnt
                # Update max count and debounce timer
                if (last_count_value is None) or (cnt != last_count_value):
                    mission_count_max = max(mission_count_max or 0, cnt)
                    last_count_value = cnt
                    last_count_seen_at = time.time()
            # Treat common autopilot cues as mission start triggers
            if text and ("Mission start accepted" in text or "Mode AUTO" in text):
                if in_auto and mission_is_ready():
                    print("STATUSTEXT indicates mission start and AUTO; Mission started.")
                    mission_started = True
        elif mtype == 'MISSION_CURRENT':
            # Treat as started only if already in AUTO and we have a mission
            if in_auto and mission_is_ready():
                print("MISSION_CURRENT received in AUTO with mission ready. Mission started.")
                mission_started = True

        # If a mission count was seen recently but items not yet known, pull them proactively
        if (mission_count is not None) and mission_count >= 0 and not mission_items and (time.time() - last_count_seen_at) > 0.5:
            # Avoid spamming requests
            if (time.time() - last_pull_attempt) > 0.8:
                last_pull_attempt = time.time()
                mission_items = _pull_mission_items(master, sysid, compid, timeout=3)
                if mission_items:
                    print(f"Pulled {len(mission_items)} mission items from vehicle.")
                    # If in AUTO and we now know mission exists, we can start
                    if in_auto:
                        print("Mission items present and AUTO. Mission started.")
                        mission_started = True

    # Wait until all mission items are pulled and match expected count
    expected_count = None
    if mission_count_max is not None:
        expected_count = mission_count_max
    elif mission_count is not None:
        expected_count = mission_count
    else:
        expected_count = None

    # Try to pull until we have all waypoints
    max_wait = 15  # seconds
    start_wait = time.time()
    while True:
        if mission_items and expected_count and len(mission_items) >= expected_count:
            break
        if time.time() - start_wait > max_wait:
            print(f"Timeout waiting for full mission upload. Proceeding with {len(mission_items)} items.")
            break
        time.sleep(0.5)
        mission_items = _pull_mission_items(master, sysid, compid, timeout=3)

    # Now pick target index
    if mission_items:
        all_seqs = sorted([s for s in mission_items.keys() if s >= 1])
        num_waypoints = len(all_seqs)
        if num_waypoints > 4:
            start_idx = int(num_waypoints * 0.25)
            end_idx = int(num_waypoints * 0.75) - 1
            # Clamp indices to valid bounds
            start_idx = max(0, min(start_idx, num_waypoints - 1))
            end_idx = max(0, min(end_idx, num_waypoints - 1))
            candidates = all_seqs[start_idx:end_idx+1] if end_idx >= start_idx else all_seqs
            print(f"Middle-half candidates: {candidates}")
            if candidates:
                target_idx = random.choice(candidates)
            else:
                target_idx = all_seqs[num_waypoints // 2]
        elif num_waypoints > 0:
            target_idx = all_seqs[0]
        else:
            target_idx = 1
    elif expected_count:
        num_waypoints = max(1, expected_count - 1)
        if num_waypoints > 4:
            start_idx = math.ceil(num_waypoints * 0.25)
            end_idx = math.floor(num_waypoints * 0.75)
            # Clamp indices to valid bounds (indices start from 1)
            start_idx = max(1, min(int(start_idx), num_waypoints))
            end_idx = max(1, min(int(end_idx), num_waypoints))
            # Ensure candidates is non-empty and indices are valid
            if end_idx >= start_idx:
                candidates = list(range(start_idx, end_idx + 1))
            else:
                candidates = list(range(1, num_waypoints + 1))
            print(f"Middle-half candidates: {candidates}")
            if candidates:
                target_idx = random.choice(candidates)
            else:
                target_idx = (num_waypoints + 1) // 2
        else:
            target_idx = 1
    else:
        target_idx = 1
    print(f"Target will be found at waypoint {target_idx} (middle half heuristic)")

    # Wait for MISSION_ITEM_REACHED and report FOUND at target_idx
    found_sent = False
    while not found_sent:
        msg = master.recv_match(type=['MISSION_ITEM_REACHED', 'STATUSTEXT'], blocking=True, timeout=2)
        if not msg:
            continue
        if msg.get_type() == 'STATUSTEXT':
            text = msg.text.decode() if hasattr(msg.text, 'decode') else msg.text
            print(f"STATUSTEXT: {text}")
            continue
        reached = int(getattr(msg, 'seq', -1))
        print(f"Waypoint reached: {reached}")
        if reached == target_idx:
            lat, lon = (0.0, 0.0)
            if mission_items and reached in mission_items:
                lat, lon = mission_items[reached]
            found_msg = f"FOUND:{lat:.6f},{lon:.6f}"
            print(f"Reporting found target: {found_msg}")
            master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, found_msg.encode('utf-8'))
            found_sent = True

    print("Done.")

if __name__ == "__main__":
    connection_string = "tcp:10.24.5.232:5763"  # Change per SITL instance
    simulate_found_midway(connection_string)
