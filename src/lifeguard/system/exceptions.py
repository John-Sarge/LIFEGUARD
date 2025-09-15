"""Custom exception hierarchy for LIFEGUARD system."""
class LifeguardError(Exception):
    """Base exception for the LIFEGUARD application."""
    pass

class NLUError(LifeguardError):
    """Exception for errors during NLU processing."""
    pass

class MavlinkError(LifeguardError):
    """Exception for MAVLink communication failures."""
    pass

class MavlinkConnectionError(MavlinkError):
    """Exception for failure to establish a MAVLink connection."""
    pass

class STTError(LifeguardError):
    """Exception for Speech-to-Text errors."""
    pass
