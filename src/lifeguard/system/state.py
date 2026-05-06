"""Message dataclasses for inter-thread communication in the LIFEGUARD system."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

class CaptureMode(str, Enum):
	COMMAND = "command"
	YESNO = "yesno"

@dataclass
class MsgBase:
	pass

@dataclass
class MsgShutdown(MsgBase):
	reason: str = ""

@dataclass
class MsgSystemShutdownRequested(MsgBase):
	reason: str = ""

@dataclass
class MsgSetCaptureMode(MsgBase):
	mode: CaptureMode

@dataclass
class MsgAudioData(MsgBase):
	data: bytes
	mode: CaptureMode
	final: bool = False

@dataclass
class MsgSTTResult(MsgBase):
	text: str
	mode: CaptureMode

@dataclass
class MsgNLURequest(MsgBase):
	text: str

@dataclass
class MsgNLUResult(MsgBase):
	text: str
	intent: str
	confidence: float
	entities: Dict[str, Any]

@dataclass
class MsgSpeak(MsgBase):
	text: str

@dataclass
class MsgSelectAgent(MsgBase):
	agent_id: str

@dataclass
class MsgCommandGridSearch(MsgBase):
	lat: float
	lon: float
	grid_size_m: int
	swath_m: float
	altitude_m: float
	target_desc: Optional[str] = None

@dataclass
class MsgCommandFlyTo(MsgBase):
	lat: float
	lon: float
	altitude_m: float
	target_desc: Optional[str] = None

@dataclass
class MsgCommandSetAltitude(MsgBase):
	altitude_m: float

@dataclass
class MsgMavStatus(MsgBase):
	text: str

# ---------------------------------------------------------------------------
# Man Overboard (MOB) messages
# ---------------------------------------------------------------------------

@dataclass
class MsgMOBActivated(MsgBase):
	"""Triggered by the MOB button in the GUI."""
	pass

@dataclass
class MsgCommandMOBSearch(MsgBase):
	"""Command to execute a Man Overboard curved-track-following search mission."""
	track_points: list  # list of (lat, lon) tuples, oldest first
	corridor_half_width_m: float
	swath_m: float
	altitude_m: float
	target_desc: Optional[str] = None
