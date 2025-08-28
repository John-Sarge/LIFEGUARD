# Agent grid search simulator: connects to a MAVLink vehicle (e.g., SITL) and
# reports a simulated FOUND status partway through an uploaded mission by
# emitting a STATUSTEXT like "FOUND:<lat>,<lon>" when a chosen waypoint is reached.

# created to test LIFEGUARD: https://github.com/John-Sarge/LIFEGUARD

import time
import re
import random
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
    # Request the list
    try:
        master.mav.mission_request_list_send(sysid, compid, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
        count_msg = master.recv_match(type=['MISSION_COUNT'], blocking=True, timeout=timeout)
        if not count_msg:
            return items
        count = int(getattr(count_msg, 'count', 0))
        for i in range(count):
            # Request each item; prefer *_INT variant (degreesE7) when available
            try:
                master.mav.mission_request_int_send(sysid, compid, i, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
                item = master.recv_match(type=['MISSION_ITEM_INT', 'MISSION_ITEM'], blocking=True, timeout=timeout)
            except Exception:
                item = None
            if not item:
                continue
            # Extract lat/lon depending on type
            if item.get_type() == 'MISSION_ITEM_INT':
                lat = float(item.x) / 1e7
                lon = float(item.y) / 1e7
                seq = int(item.seq)
            else:
                lat = float(item.x)  # already degrees
                lon = float(item.y)
                seq = int(item.seq)
            items[seq] = (lat, lon)
    except Exception:
        pass
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

    # Decide target index if not already set
    # Determine highest valid waypoint seq (exclude home seq 0)
    if mission_items:
        all_seqs = sorted(mission_items.keys())
        max_seq = max([s for s in all_seqs if s >= 1], default=1)
        mission_count = max_seq  # use highest reachable seq as count proxy
    elif mission_count_max is not None:
        # If mission_count_max includes home, cap at (max-1) for reachable seqs; otherwise assume it's a seq count
        mission_count = max(1, mission_count_max - 1)
        max_seq = mission_count
    else:
        # Fallback to a reasonable default
        mission_count = 10
        max_seq = 10

    # Choose a target index in [1..max_seq] using a middle-half heuristic
    if max_seq > 4:
        start_idx = max(1, max_seq // 4)
        end_idx = max(start_idx, (max_seq * 3) // 4)
        target_idx = random.randint(start_idx, end_idx)
    else:
        target_idx = 1
    # Clamp to valid range
    target_idx = max(1, min(target_idx, max_seq))
    print(f"Target will be found at waypoint {target_idx} (middle half heuristic)")

    # If mission items are still unknown, try one more pull
    if not mission_items:
        mission_items = _pull_mission_items(master, sysid, compid, timeout=3)

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
        # Handle MISSION_ITEM_REACHED
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
