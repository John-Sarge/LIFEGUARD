# Agent grid search simulator: connects to a MAVLink vehicle (e.g., SITL) and
# reports a simulated FOUND status partway through an uploaded mission.

import time
import math
import random
from typing import Dict, Optional, Tuple
from pymavlink import mavutil

def _mode_is_auto(master: mavutil.mavfile) -> bool:
    """Return True if the heartbeat indicates AUTO mode."""
    return master.flightmode == 'AUTO'

def _pull_mission_items(master: mavutil.mavfile) -> Optional[Dict[int, Tuple[float, float]]]:
    """Pull all mission items from the vehicle."""
    try:
        master.mav.mission_request_list_send(master.target_system, master.target_component)
        count_msg = master.recv_match(type=['MISSION_COUNT'], blocking=True, timeout=3.0)
        if not count_msg or count_msg.count == 0:
            return None

        items: Dict[int, Tuple[float, float]] = {}
        for i in range(count_msg.count):
            master.mav.mission_request_int_send(master.target_system, master.target_component, i)
            item = master.recv_match(type=['MISSION_ITEM_INT'], blocking=True, timeout=1.0)
            if item:
                items[item.seq] = (item.x / 1e7, item.y / 1e7)
        
        if len(items) == count_msg.count:
            print(f"Successfully pulled {len(items)} mission items.")
            return items
    except Exception as e:
        print(f"Failed to pull mission items: {e}")
    return None

def _select_target_waypoint(num_waypoints: int) -> int:
    """Selects a waypoint from the middle half of the mission, with fallbacks."""
    if num_waypoints <= 1:
        return 1
    if num_waypoints <= 4:
        target_idx = math.ceil(num_waypoints / 2)
        print(f"Short mission ({num_waypoints} WPs), picking middle: waypoint {target_idx}")
        return target_idx
    
    start_idx = max(1, math.ceil(num_waypoints * 0.25))
    end_idx = math.floor(num_waypoints * 0.75)
    candidates = list(range(start_idx, end_idx + 1))
    if not candidates:
        return math.ceil(num_waypoints / 2)

    target_idx = random.choice(candidates)
    print(f"Middle-half candidates: {candidates}. Picking waypoint {target_idx}")
    return target_idx

def simulate_found_midway(connection_string: str):
    print(f"Connecting to agent at {connection_string}...")
    master = mavutil.mavlink_connection(connection_string, autoreconnect=True, mavlink2=True)
    master.wait_heartbeat(timeout=10)
    
    if master.target_system == 0:
        print("No heartbeat. Exiting.")
        return

    print(f"Connected to system {master.target_system}. Waiting for mission and AUTO mode...")

    mission_items: Optional[Dict[int, Tuple[float, float]]] = None
    
    while True:
        master.wait_heartbeat()
        if _mode_is_auto(master):
            if not mission_items:
                mission_items = _pull_mission_items(master)
            if mission_items:
                print("In AUTO mode with a valid mission. Proceeding to select target.")
                break
        time.sleep(1)

    mission_waypoints = {seq: coords for seq, coords in mission_items.items() if seq > 0}
    num_waypoints = len(mission_waypoints)
    target_idx = _select_target_waypoint(num_waypoints)

    found_sent = False
    while not found_sent:
        msg = master.recv_match(type=['MISSION_ITEM_REACHED', 'STATUSTEXT'], blocking=True)
        if not msg:
            continue

        if msg.get_type() == 'STATUSTEXT':
            # --- FIX: Check if the text is bytes before trying to decode it ---
            text_val = msg.text
            if isinstance(text_val, bytes):
                text_val = text_val.decode('utf-8', errors='ignore').rstrip('\x00')
            
            if text_val.upper().startswith("TARGET:"):
                print(f"SUCCESS: Received target details from Lifeguard: '{text_val}'")
        
        elif msg.get_type() == 'MISSION_ITEM_REACHED':
            print(f"Waypoint reached: {msg.seq}")
            if msg.seq == target_idx:
                lat, lon = mission_items.get(target_idx, (0.0, 0.0))
                found_msg = f"FOUND:{lat:.6f},{lon:.6f}"
                print(f"Reporting found target: {found_msg}")
                master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, found_msg.encode('utf-8'))
                found_sent = True

    print("Done.")

if __name__ == "__main__":
    # --- Remember to set the correct IP for each agent instance ---
    # Example for agent 1: "tcp:127.0.0.1:5763"
    # Example for agent 2: "tcp:127.0.0.1:5773"
    connection_string = "tcp:10.24.5.232:5763"
    simulate_found_midway(connection_string)