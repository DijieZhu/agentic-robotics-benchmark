"""`Frame`: one CAN frame on one of the pair's two buses. PUBLIC.

`Frame(bus, can_id, data)`: bus 0 = left, 1 = right; an 11-bit id; data as bytes. A well-formed frame has exactly 8 data
bytes, a known bus and an 11-bit id. A frame that is NOT well formed can still be constructed, because the runtime
contract receives such frames from a controller and must drop and count them (`malformed`, `unknown_id`); it is
the bus layer's job to classify them, not this type's.

On a wire to another process: 12 bytes per frame = bus (1), length (1, must be 8), id (2, big endian), data (8); a
cycle's frames are sent as one message with a 16-bit count in front, big endian like the id.
"""
from dataclasses import dataclass

BUS_LEFT = 0
BUS_RIGHT = 1
BUSES = (BUS_LEFT, BUS_RIGHT)
DATA_LEN = 8
ID_MAX = 0x7FF
WIRE_FRAME_BYTES = 12
WIRE_COUNT_BYTES = 2


class WireError(ValueError):
    """A wire message that cannot be parsed (wrong total length, a count that does not match)."""


@dataclass(frozen=True)
class Frame:
    bus: int
    can_id: int
    data: bytes

    def __post_init__(self):
        if not isinstance(self.data, (bytes, bytearray)):
            object.__setattr__(self, "data", bytes(self.data))
        elif isinstance(self.data, bytearray):
            object.__setattr__(self, "data", bytes(self.data))
        if not isinstance(self.bus, int) or not isinstance(self.can_id, int):
            raise TypeError("bus and can_id are integers")

    def is_wellformed(self):
        """True when the bus is known, the id fits 11 bits and there are exactly 8 data bytes."""
        return self.bus in BUSES and 0 <= self.can_id <= ID_MAX and len(self.data) == DATA_LEN

    def hex(self):
        return f"{self.bus}:{self.can_id:03X}#{self.data.hex().upper()}"


def pack_frame(f):
    """12 wire bytes of one frame. The length byte carries len(data); the data field carries the first 8 bytes,
    zero-padded (a frame of another length is thus carried and recognised as malformed by its length byte)."""
    if not (0 <= f.bus <= 0xFF and 0 <= f.can_id <= 0xFFFF):
        raise WireError("bus or id does not fit its wire field")
    n = len(f.data)
    if n > 0xFF:
        raise WireError("data longer than 255 bytes")
    body = f.data[:DATA_LEN] + bytes(max(0, DATA_LEN - n))
    return bytes([f.bus, n]) + f.can_id.to_bytes(2, "big") + body


def unpack_frame(b):
    """Inverse of pack_frame. A length above 8 cannot be carried in full: the frame keeps the 8 bytes the wire has,
    padded with zeros to its stated length, so that it stays malformed with the same length."""
    if len(b) != WIRE_FRAME_BYTES:
        raise WireError("a wire frame is 12 bytes")
    bus, n = b[0], b[1]
    can_id = int.from_bytes(b[2:4], "big")
    data = b[4:12]
    if n <= DATA_LEN:
        data = data[:n]
    else:
        data = data + bytes(n - DATA_LEN)
    return Frame(bus, can_id, data)


def pack_cycle(frames):
    """One message: a 16-bit count, then 12 bytes per frame."""
    frames = list(frames)
    if len(frames) > 0xFFFF:
        raise WireError("at most 65535 frames per message")
    return len(frames).to_bytes(WIRE_COUNT_BYTES, "big") + b"".join(pack_frame(f) for f in frames)


def unpack_cycle(msg):
    if len(msg) < WIRE_COUNT_BYTES:
        raise WireError("a message starts with a 16-bit count")
    count = int.from_bytes(msg[:WIRE_COUNT_BYTES], "big")
    if len(msg) != WIRE_COUNT_BYTES + WIRE_FRAME_BYTES * count:
        raise WireError("the count does not match the message length")
    return [unpack_frame(msg[WIRE_COUNT_BYTES + WIRE_FRAME_BYTES * i: WIRE_COUNT_BYTES + WIRE_FRAME_BYTES * (i + 1)])
            for i in range(count)]
