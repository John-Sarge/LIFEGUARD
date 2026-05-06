"""MAVLink controller: connect, manage modes, upload missions, and navigation for drones."""
import math
import time
import logging
from pymavlink import mavutil, mavwp
from lifeguard.system.exceptions import MavlinkError, MavlinkConnectionError

EARTH_RADIUS_METERS = 6378137.0


class MavlinkController:
	"""Controls a single drone via MAVLink: connection, mode, missions, altitude, fly-to."""
	def __init__(self, connection_string: str, baudrate: int | None = None, source_system_id: int = 255):
		self.logger = logging.getLogger(__name__)
		self.connection_string = connection_string
		self.baudrate = baudrate
		self.source_system_id = source_system_id
		self.master = None

	def connect(self):
		"""Establish connection to drone and wait for heartbeat."""
		try:
			if "com" in self.connection_string.lower() or "/dev/tty" in self.connection_string.lower():
				self.master = mavutil.mavlink_connection(
					self.connection_string, baud=self.baudrate, source_system=self.source_system_id
				)
			else:
				self.master = mavutil.mavlink_connection(
					self.connection_string,
					source_system=self.source_system_id,
					dialect="ardupilotmega",
					autoreconnect=True,
					use_native=False,
					force_connected=True,
					mavlink2=True,
				)
			self.logger.info(f"MAVLink: Waiting for heartbeat (timeout 10s) on {self.connection_string}...")
			self.master.wait_heartbeat(timeout=10)
			if self.master.target_system == 0 and self.master.target_component == 0:
				self.logger.error("MAVLink: Invalid heartbeat (system=0).")
				self.master = None
				raise MavlinkConnectionError(f"Invalid heartbeat (system=0) for {self.connection_string}")
			elif self.master.target_system != 0:
				self.logger.info(
					f"MAVLink: Heartbeat from system {self.master.target_system} component {self.master.target_component}."
				)
			else:
				self.logger.error(
					f"MAVLink: wait_heartbeat returned without valid target system ({self.master.target_system})."
				)
				self.master = None
				raise MavlinkConnectionError(f"No valid target system for {self.connection_string}")
			if self.master:
				self.logger.info(
					f"MAVLink: Using target_system={self.master.target_system}, target_component={self.master.target_component}"
				)
				self.master.target_component = 1
				self.start_position_stream()
		except Exception as e:
			self.logger.error(
				f"MAVLink: EXCEPTION connect/wait_heartbeat on {self.connection_string}: {e}", exc_info=True
			)
			self.master = None
			raise MavlinkError(f"Exception during MAVLink connection: {e}") from e

	def start_position_stream(self, rate_hz: int = 2):
		"""Request GLOBAL_POSITION_INT streaming at rate_hz from the vehicle."""
		if not self.is_connected():
			return

		interval_us = int(1_000_000 / rate_hz)
		self.logger.info(f"Requesting GLOBAL_POSITION_INT stream at {rate_hz} Hz from system {self.master.target_system}.")
		try:
			self.master.mav.command_long_send(
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
				0,
				mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
				interval_us,
				0, 0, 0, 0, 0
			)
			# Legacy fallback for older firmware stacks
			self.master.mav.request_data_stream_send(
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_DATA_STREAM_POSITION,
				max(1, int(rate_hz)),
				1
			)
			self.master.mav.request_data_stream_send(
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
				max(1, int(rate_hz)),
				1
			)
		except Exception as e:
			self.logger.error(f"Failed to request position stream(s): {e}")

	def is_connected(self) -> bool:
		return self.master is not None
        
	def get_current_position(self) -> dict | None:
		"""Return current position as {lat, lon, alt} from the vehicle's GLOBAL_POSITION_INT."""
		if not self.is_connected():
			return None
		try:
			self.master.mav.command_long_send(
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
				0,
				mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
				200000,
				0, 0, 0, 0, 0,
			)
		except Exception:
			pass
		start = time.time()
		msg = None
		while time.time() - start < 2.0:
			try:
				msg = self.master.messages.get('GLOBAL_POSITION_INT')
			except Exception:
				msg = None
			if msg:
				break
			time.sleep(0.1)
		if not msg:
			return None
		return {"lat": msg.lat / 1e7, "lon": msg.lon / 1e7, "alt": msg.relative_alt / 1000.0}

	def upload_mission(self, waypoints_data):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot upload mission.")
			return False
		self.master.target_component = 1
		wp_loader = mavwp.MAVWPLoader()
		seq = 0
		home_lat, home_lon, home_alt = waypoints_data[0][0], waypoints_data[0][1], 0
		home_item = mavutil.mavlink.MAVLink_mission_item_int_message(
			self.master.target_system,
			1,
			seq,
			mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
			mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
			0,
			1,
			0,
			0,
			0,
			0,
			int(home_lat * 1e7),
			int(home_lon * 1e7),
			float(home_alt),
		)
		wp_loader.add(home_item)
		seq += 1
		for item in waypoints_data:
			if len(item) == 8:
				lat, lon, alt, command_id, p1, p2, p3, p4 = item
				frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
				is_current = 0
				autocontinue = 1
			elif len(item) == 11:
				lat, lon, alt, command_id, p1, p2, p3, p4, frame, is_current, autocontinue = item
			else:
				continue
			wp_loader.add(
				mavutil.mavlink.MAVLink_mission_item_int_message(
					self.master.target_system,
					1,
					seq,
					frame,
					command_id,
					is_current,
					autocontinue,
					p1,
					p2,
					p3,
					p4,
					int(lat * 1e7),
					int(lon * 1e7),
					float(alt),
				)
			)
			seq += 1
		if wp_loader.count() == 0:
			self.logger.error("MAVLink: No valid waypoints to upload.")
			return False
		try:
			self.master.mav.mission_clear_all_send(self.master.target_system, 1)
			self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
			time.sleep(1)
			self.master.mav.mission_count_send(
				self.master.target_system, 1, wp_loader.count(), mavutil.mavlink.MAV_MISSION_TYPE_MISSION
			)
			for i in range(wp_loader.count()):
				msg = self.master.recv_match(
					type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=5
				)
				if not msg:
					self.logger.error(f"MAVLink: No MISSION_REQUEST for wp {i}")
					return False
				self.master.mav.send(wp_loader.wp(msg.seq))
				if msg.seq == wp_loader.count() - 1:
					break
			final_ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
			if final_ack and final_ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
				self.logger.info("MAVLink: Mission upload successful.")
				return True
			self.logger.error(f"MAVLink: Mission upload failed or bad ACK: {final_ack}")
			return False
		except Exception as e:
			self.logger.error(f"MAVLink: Mission upload error: {e}")
			return False

	def set_mode(self, mode_name):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot set mode.")
			return False
		mode_mapping = self.master.mode_mapping()
		if mode_mapping is None or mode_name.upper() not in mode_mapping:
			self.logger.error(f"MAVLink: Unknown mode: {mode_name}")
			return False
		mode_id = mode_mapping[mode_name.upper()]
		
		try:
			self.master.mav.set_mode_send(
				self.master.target_system,
				mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
				mode_id,
			)
			ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
			
			if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
				if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
					self.logger.info(f"MAVLink: Mode {mode_name} change confirmed by ACK.")
					return True
				else:
					self.logger.error(f"MAVLink: Mode {mode_name} change rejected by vehicle (Result: {ack_msg.result}).")
					return False
			
			# If we get here, we timed out waiting for the ACK.
			self.logger.warning(f"MAVLink: No ACK received for {mode_name} mode change. Assuming success for real hardware.")
			return True # Assume it worked

		except Exception as e:
			self.logger.error(f"MAVLink: Error setting mode: {e}")
			return False

	def arm_vehicle(self):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot arm.")
			return False
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
			mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
			0,
			1,
			0,
			0,
			0,
			0,
			0,
			0,
		)
		ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
		if (
			ack_msg
			and ack_msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
			and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
		):
			self.logger.info("MAVLink: Armed.")
			return True
		self.logger.error(f"MAVLink: Arming failed: {ack_msg}")
		return False

	def start_mission(self):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot start mission.")
			return False
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
			mavutil.mavlink.MAV_CMD_MISSION_START,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
		)
		ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
		if (
			ack_msg
			and ack_msg.command == mavutil.mavlink.MAV_CMD_MISSION_START
			and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
		):
			self.logger.info("MAVLink: Mission start accepted.")
			return True
		self.logger.error(f"MAVLink: Mission start failed: {ack_msg}")
		return False

	def wait_for_waypoint_reached(self, waypoint_sequence_id, timeout_seconds=120):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot wait for waypoint.")
			return False
		start = time.time()
		while time.time() - start < timeout_seconds and self.is_connected():
			reached_msg = self.master.messages.get('MISSION_ITEM_REACHED') if hasattr(self.master, 'messages') else None
			if reached_msg and getattr(reached_msg, 'seq', None) == waypoint_sequence_id:
				return True
			time.sleep(0.5)
		return False

	def send_status_text(self, text_to_send, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot send STATUSTEXT.")
			return False
		text_bytes = text_to_send.encode('utf-8') if isinstance(text_to_send, str) else text_to_send
		payload_text_bytes = text_bytes[:49]
		try:
			self.master.mav.statustext_send(severity, payload_text_bytes)
			return True
		except Exception as e:
			self.logger.error(f"MAVLink: STATUSTEXT error: {e}")
			return False

	def close_connection(self):
		if self.master:
			self.master.close()
			self.logger.info(f"MAVLink: Connection closed for {self.connection_string}.")

	def _calculate_rectangular_grid_waypoints(
		self, center_lat, center_lon, grid_width_m, grid_height_m, swath_width_m, altitude_m
	):
		waypoints = []
		if swath_width_m <= 0 or grid_width_m <= 0 or grid_height_m <= 0:
			return waypoints
		center_lat_rad = math.radians(center_lat)
		center_lon_rad = math.radians(center_lon)
		half_width_m = grid_width_m / 2.0
		half_height_m = grid_height_m / 2.0
		lat_delta_rad = half_height_m / EARTH_RADIUS_METERS
		south_lat_rad = center_lat_rad - lat_delta_rad
		north_lat_rad = center_lat_rad + lat_delta_rad
		lon_delta_rad = half_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))
		west_lon_rad = center_lon_rad - lon_delta_rad
		num_tracks = int(math.floor(grid_width_m / swath_width_m))
		if num_tracks == 0 and grid_width_m > 0:
			num_tracks = 1
		if num_tracks == 0:
			return waypoints
		swath_lon_rad_step = swath_width_m / (EARTH_RADIUS_METERS * math.cos(center_lat_rad))
		for i in range(num_tracks):
			current_track_lon_rad = west_lon_rad + i * swath_lon_rad_step
			current_track_lon_deg = math.degrees(current_track_lon_rad)
			south_lat_deg = math.degrees(south_lat_rad)
			north_lat_deg = math.degrees(north_lat_rad)
			if i % 2 == 0:
				waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
				waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
			else:
				waypoints.append((north_lat_deg, current_track_lon_deg, altitude_m))
				waypoints.append((south_lat_deg, current_track_lon_deg, altitude_m))
		return waypoints

	def generate_and_upload_search_grid_mission(self, center_lat, center_lon, grid_size_m, swath_width_m, altitude_m):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot generate/upload search grid.")
			return None
			
		grid_waypoints_tuples = self._calculate_rectangular_grid_waypoints(
			center_lat, center_lon, grid_size_m, grid_size_m, swath_width_m, altitude_m
		)
		if not grid_waypoints_tuples:
			self.logger.error("MAVLink: Failed to generate grid waypoints.")
			return None

		waypoints_data_for_upload = []
		path_positions = []
		for lat, lon, alt in grid_waypoints_tuples:
			waypoints_data_for_upload.append(
				(lat, lon, alt, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 10.0, 0.0, float('nan'))
			)
			path_positions.append((lat, lon))

		if self.upload_mission(waypoints_data_for_upload):
			return path_positions
		else:
			return None

	@staticmethod
	def _offset_position(lat: float, lon: float, bearing_deg: float, distance_m: float):
		"""Return (lat, lon) reached by travelling distance_m along bearing_deg from (lat, lon)."""
		lat_r = math.radians(lat)
		lon_r = math.radians(lon)
		bearing_r = math.radians(bearing_deg)
		d = distance_m / EARTH_RADIUS_METERS
		lat2 = math.asin(
			math.sin(lat_r) * math.cos(d)
			+ math.cos(lat_r) * math.sin(d) * math.cos(bearing_r)
		)
		lon2 = lon_r + math.atan2(
			math.sin(bearing_r) * math.sin(d) * math.cos(lat_r),
			math.cos(d) - math.sin(lat_r) * math.sin(lat2),
		)
		return math.degrees(lat2), math.degrees(lon2)

	@staticmethod
	def _bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
		"""Forward azimuth in degrees from (lat1,lon1) to (lat2,lon2)."""
		lat1_r = math.radians(lat1)
		lat2_r = math.radians(lat2)
		dlon_r = math.radians(lon2 - lon1)
		y = math.sin(dlon_r) * math.cos(lat2_r)
		x = (math.cos(lat1_r) * math.sin(lat2_r)
			 - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
		return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

	@staticmethod
	def _mean_bearing(b1: float, b2: float) -> float:
		"""Circular mean of two bearings (degrees), handles 0/360 wrap."""
		x = math.cos(math.radians(b1)) + math.cos(math.radians(b2))
		y = math.sin(math.radians(b1)) + math.sin(math.radians(b2))
		return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

	def _calculate_mob_curved_track_waypoints(
		self,
		track_points: list,
		corridor_half_width_m: float,
		swath_m: float,
		altitude_m: float,
	):
		"""
		Generate curved-track-following MOB search lanes.

		Each lane is a copy of the ship's recorded track offset perpendicular to
		the local heading at every point, so it faithfully follows every turn.
		Lanes expand outward: 0 (on-track), +swath stbd, -swath port, +2*swath, …
		Boustrophedon order: even lanes oldest→newest, odd lanes newest→oldest.

		track_points: list of (lat, lon) tuples, oldest first.
		"""
		if len(track_points) < 2 or swath_m <= 0 or corridor_half_width_m <= 0:
			return []

		# --- Step 1: discard consecutive points that are too close together ----
		# GPS noise can place successive 10-s samples within a few metres of each
		# other; _bearing_between on nearly-identical coords returns a near-random
		# heading that distorts the offset lanes.
		MIN_SEP_M = 12.0
		filtered = [track_points[0]]
		for pt in track_points[1:]:
			la1, lo1 = filtered[-1]
			la2, lo2 = pt
			dlat = math.radians(la2 - la1)
			dlon = math.radians(lo2 - lo1)
			a = (math.sin(dlat / 2) ** 2
				 + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
				 * math.sin(dlon / 2) ** 2)
			dist = EARTH_RADIUS_METERS * 2.0 * math.atan2(
				math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))
			)
			if dist >= MIN_SEP_M:
				filtered.append(pt)
		# Ensure we kept enough points after filtering
		if len(filtered) < 2:
			filtered = list(track_points)
		track_points = filtered
		n = len(track_points)

		# --- Step 2: raw per-point bearings (in+out average at interior points) -
		raw_bearings = []
		for i in range(n):
			if i == 0:
				b = self._bearing_between(*track_points[0], *track_points[1])
			elif i == n - 1:
				b = self._bearing_between(*track_points[-2], *track_points[-1])
			else:
				b_in = self._bearing_between(*track_points[i - 1], *track_points[i])
				b_out = self._bearing_between(*track_points[i], *track_points[i + 1])
				b = self._mean_bearing(b_in, b_out)
			raw_bearings.append(b)

		# --- Step 3: smooth bearings with a 5-point circular-mean window --------
		# This removes isolated spikes that remain after filtering.
		HALF = 2
		local_bearings = []
		for i in range(n):
			lo = max(0, i - HALF)
			hi = min(n, i + HALF + 1)
			xs = sum(math.cos(math.radians(b)) for b in raw_bearings[lo:hi])
			ys = sum(math.sin(math.radians(b)) for b in raw_bearings[lo:hi])
			local_bearings.append((math.degrees(math.atan2(ys, xs)) + 360.0) % 360.0)

		# Lane offsets: 0, +swath (stbd), -swath (port), +2*swath, -2*swath …
		num_each_side = max(1, int(math.ceil(corridor_half_width_m / swath_m)))
		lane_offsets_m = [0.0]
		for i in range(1, num_each_side + 1):
			lane_offsets_m.append(i * swath_m)   # starboard
			lane_offsets_m.append(-i * swath_m)  # port

		waypoints = []
		for lane_idx, offset_m in enumerate(lane_offsets_m):
			lane = []
			for i, (lat, lon) in enumerate(track_points):
				if offset_m == 0.0:
					lane.append((lat, lon, altitude_m))
				else:
					# Perpendicular bearing: +90 = stbd, -90 = port
					perp = (local_bearings[i] + (90.0 if offset_m > 0 else 270.0)) % 360.0
					p_lat, p_lon = self._offset_position(lat, lon, perp, abs(offset_m))
					lane.append((p_lat, p_lon, altitude_m))
			# Boustrophedon: reverse every other lane
			if lane_idx % 2 != 0:
				lane = list(reversed(lane))
			waypoints.extend(lane)

		return waypoints

	def generate_and_upload_mob_search_mission(
		self,
		track_points: list,
		corridor_half_width_m: float,
		swath_m: float,
		altitude_m: float,
	):
		"""
		Generate and upload a curved-track-following MOB search mission.

		track_points: list of (lat, lon) tuples from the ship track, oldest first.
		Returns a list of (lat, lon) path positions on success, or None on failure.
		"""
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot upload MOB search mission.")
			return None

		waypoints_tuples = self._calculate_mob_curved_track_waypoints(
			track_points, corridor_half_width_m, swath_m, altitude_m,
		)
		if not waypoints_tuples:
			self.logger.error("MAVLink: Failed to generate MOB curved track waypoints.")
			return None

		# Orient the pattern so the drone starts from the closest end.
		# The boustrophedon is a continuous snake, so reversing the flat list is
		# equivalent to flying the same coverage from the opposite entry point.
		drone_pos = self.get_current_position()
		if drone_pos and len(waypoints_tuples) >= 2:
			def _dist_sq(pt):
				dlat = drone_pos["lat"] - pt[0]
				dlon = drone_pos["lon"] - pt[1]
				return dlat * dlat + dlon * dlon
			if _dist_sq(waypoints_tuples[-1]) < _dist_sq(waypoints_tuples[0]):
				waypoints_tuples = list(reversed(waypoints_tuples))
				self.logger.info("MOB search: reversed waypoint order to start at closer end.")

		waypoints_data = []
		path_positions = []
		for lat, lon, alt in waypoints_tuples:
			waypoints_data.append(
				(lat, lon, alt, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 10.0, 0.0, float('nan'))
			)
			path_positions.append((lat, lon))

		if self.upload_mission(waypoints_data):
			return path_positions
		return None

	def set_altitude(self, altitude_m):
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot set altitude.")
			return False
		try:
			self.master.mav.command_long_send(
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_CMD_DO_CHANGE_ALTITUDE,
				0,
				float(altitude_m),
				0,
				0,
				0,
				0,
				0,
				0,
			)
			ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
			if (
				ack
				and ack.command == mavutil.mavlink.MAV_CMD_DO_CHANGE_ALTITUDE
				and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
			):
				self.logger.info(
					f"MAVLink: Altitude change to {altitude_m} accepted via DO_CHANGE_ALTITUDE."
				)
				return True
			self.logger.warning(
				f"MAVLink: DO_CHANGE_ALTITUDE not accepted or no ACK: {ack}. Falling back to position target..."
			)
		except Exception as e:
			self.logger.error(
				f"MAVLink: Error issuing DO_CHANGE_ALTITUDE: {e}. Falling back to position target..."
			)

		self.logger.info("MAVLink: Setting mode to GUIDED...")
		if not self.set_mode("GUIDED"):
			self.logger.error("MAVLink: Failed to set GUIDED mode. Cannot change altitude.")
			return False

		try:
			self.master.mav.command_long_send(
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
				0,
				mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
				200000,
				0,
				0,
				0,
				0,
				0,
			)
		except Exception:
			pass

		self.logger.info("MAVLink: Waiting for current position (non-blocking polling)...")
		current_pos = None
		start_poll = time.time()
		while time.time() - start_poll < 4.0 and self.is_connected():
			try:
				current_pos = self.master.messages.get('GLOBAL_POSITION_INT')
			except Exception:
				current_pos = None
			if current_pos:
				break
			time.sleep(0.15)
		if not current_pos:
			self.logger.error("MAVLink: Could not get current position. Altitude change aborted.")
			return False

		self.logger.info(f"MAVLink: Commanding altitude change to {altitude_m}m...")
		try:
			type_mask = (
				mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
			)
			self.master.mav.set_position_target_global_int_send(
				0,
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
				type_mask,
				int(current_pos.lat),
				int(current_pos.lon),
				float(altitude_m),
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			)
			self.logger.info(
				f"MAVLink: Altitude command sent. Drone should be moving to {altitude_m}m."
			)
			return True
		except Exception as e:
			self.logger.error(f"MAVLink: Error sending set_position_target_global_int: {e}")
			return False

	def fly_to(self, lat: float, lon: float, altitude_m: float):
		"""Fly to a specific lat/lon/alt using GUIDED mode and position target."""
		if not self.is_connected():
			self.logger.error("MAVLink: Not connected. Cannot fly to location.")
			return False
		# Ensure GUIDED mode before sending position target
		if not self.set_mode("GUIDED"):
			self.logger.error("MAVLink: Failed to set GUIDED mode. Cannot fly to location.")
			return False
		try:
			type_mask = (
				mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
				| mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
			)
			self.master.mav.set_position_target_global_int_send(
				0,
				self.master.target_system,
				self.master.target_component,
				mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
				type_mask,
				int(lat * 1e7),
				int(lon * 1e7),
				float(altitude_m),
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			)
			self.logger.info(f"MAVLink: Fly-to command sent to ({lat}, {lon}) at {altitude_m}m.")
			return True
		except Exception as e:
			self.logger.error(f"MAVLink: Error sending fly-to command: {e}")
			return False
