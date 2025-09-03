# LIFEGUARD Architecture

High-level design:

- Multi-threaded, queue-based message passing between subsystems.
- Coordinator thread maintains capture modes and confirmation flow.
- MAVLink I/O encapsulated in `MavlinkController` for command execution.
- STT via Vosk; NLU via spaCy with custom entity recognizer for GPS.

For operational flow and command grammar, see the README.
