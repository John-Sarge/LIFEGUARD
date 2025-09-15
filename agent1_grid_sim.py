# Agent grid search simulator: connects to a MAVLink vehicle (e.g., SITL) and
# reports a simulated FOUND status partway through an uploaded mission.
import time
import math
import random
import argparse
from typing import Dict, Optional, Tuple
from pymavlink import mavutil

def _mode_is_auto(master: mavutil.mavfile) -> bool:
	"""Return True if the heartbeat indicates AUTO mode."""
	return master.flightmode == 'AUTO'

def _pull_mission_items(master: mavutil.mavfile) -> Optional[Dict[int, Tuple[float, float]]]:
	"""Pull all mission items from the vehicle."""
	try:
		master.mav.mission_request_list_send(master.target_system, master.target_component)
		count_msg = master.recv_match(type=['MISSION_COUNT'], blocking=True, timeout=5.0)
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
    # Use unique source_system ID (253) distinct from Lifeguard (252) and vehicle (usually 1)
    master = mavutil.mavlink_connection(
        connection_string,
        source_system=253,
        autoreconnect=True,
        mavlink2=True
    )
    master.wait_heartbeat(timeout=10)
    
    if master.target_system == 0:
        print("No heartbeat. Exiting.")
        return

    print(f"Connected to system {master.target_system}. Waiting for mission and AUTO mode...")

    mission_items: Optional[Dict[int, Tuple[float, float]]] = None
    
    # It will keep trying to get the mission until the vehicle is in AUTO
    while True:
        master.wait_heartbeat() # Wait for any message to keep the connection alive

        # Always try to pull the mission, even before AUTO mode.
        # This way, we get it as soon as it's available.
        if not mission_items:
            mission_items = _pull_mission_items(master)

        # Only break the loop and proceed once BOTH conditions are met.
        if _mode_is_auto(master) and mission_items:
            print("In AUTO mode with a valid mission. Proceeding to select target.")
            break
            
        time.sleep(1) # Wait a second before trying again

    mission_waypoints = {seq: coords for seq, coords in mission_items.items() if seq > 0}
    num_waypoints = len(mission_waypoints)
    target_idx = _select_target_waypoint(num_waypoints)

    found_sent = False
    while not found_sent:
        msg = master.recv_match(type=['MISSION_ITEM_REACHED', 'STATUSTEXT'], blocking=True)
        if not msg:
            continue

        if msg.get_type() == 'STATUSTEXT':
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
    parser = argparse.ArgumentParser(description="Agent1 grid sim / handshake test")
    parser.add_argument("connection", nargs='?', default="tcp:10.24.5.232:5762", help="MAVLink connection string")
    parser.add_argument("--handshake-test", action="store_true", help="Run 30s STATUSTEXT handshake bench test and exit")
    parser.add_argument("--duration", type=int, default=30, help="Handshake test duration seconds (default 30)")
    args = parser.parse_args()

    if args.handshake_test:
        def run_handshake(conn: str, duration: int):
            print(f"Connecting (handshake test) to {conn}")
            master = mavutil.mavlink_connection(
                conn,
                source_system=253,
                autoreconnect=True,
                mavlink2=True
            )
            master.wait_heartbeat(timeout=10)
            if master.target_system == 0:
                print("No heartbeat; aborting.")
                return
            print("Heartbeat OK. Starting handshake loop.")
            start = time.time()
            seq = 0
            pending = {}
            acks = 0
            latencies = []
            while time.time() - start < duration:
                # Send a request every 1s
                now = time.time()
                if seq == 0 or now - pending.get(seq, (0,))[0] >= 1.0:
                    seq += 1
                    ts_ms = int(now * 1000)
                    txt = f"HANDSHAKE_REQ:{seq}:{ts_ms}"
                    master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, txt.encode('utf-8')[:49])
                    pending[seq] = (now, ts_ms)
                # Drain messages
                for _ in range(10):
                    msg = master.recv_match(type=['STATUSTEXT'], blocking=False)
                    if not msg:
                        break
                    txt = msg.text
                    if isinstance(txt, bytes):
                        txt = txt.decode('utf-8', errors='ignore').rstrip('\x00')
                    if txt.startswith("HANDSHAKE_ACK:"):
                        try:
                            ack_seq = int(txt.split(":", 1)[1])
                        except ValueError:
                            continue
                        if ack_seq in pending:
                            send_time, ts_ms = pending.pop(ack_seq)
                            rtt = (time.time() - send_time) * 1000
                            latencies.append(rtt)
                            acks += 1
                            print(f"ACK seq {ack_seq} RTT {rtt:.1f} ms")
                time.sleep(0.1)
            if latencies:
                avg = sum(latencies)/len(latencies)
                print(f"Handshake complete: sent {seq}, acks {acks}, avg RTT {avg:.1f} ms, min {min(latencies):.1f}, max {max(latencies):.1f}")
            else:
                print("No ACKs received. Check routing (mavlink-router) setup.")
        run_handshake(args.connection, args.duration)
    else:
        simulate_found_midway(args.connection)