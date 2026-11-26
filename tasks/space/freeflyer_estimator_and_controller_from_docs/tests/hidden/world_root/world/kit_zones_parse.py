"""Parse a flight-software zone file (ROS1-serialised SetZones request) into JSON.

Layout (ros::serialization, little endian), from communications/ff_msgs/srv/SetZones.srv + msg/Zone.msg:
  time timestamp            -> uint32 sec, uint32 nsec
  Zone[] zones              -> uint32 count, then per zone:
     string name            -> uint32 len + bytes
     int32 index
     uint8 type             (0 KEEPOUT, 1 KEEPIN, 2 CLUTTER)
     Vector3 min, Vector3 max -> 6 float64
usage: python -I -B zones_parse.py <zones.bin> <out.json>
"""
import json
import struct
import sys

TYPES = {0: "keepout", 1: "keepin", 2: "clutter"}


def parse_zones(buf: bytes) -> dict:
    off = 0
    sec, nsec = struct.unpack_from("<II", buf, off)
    off += 8
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    zones = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", buf, off)
        off += 4
        name = buf[off:off + ln].decode("ascii")
        off += ln
        idx, typ = struct.unpack_from("<iB", buf, off)
        off += 5
        v = struct.unpack_from("<6d", buf, off)
        off += 48
        lo = [min(v[i], v[i + 3]) for i in range(3)]
        hi = [max(v[i], v[i + 3]) for i in range(3)]
        zones.append({"name": name, "index": idx, "type": TYPES.get(typ, str(typ)), "min": lo, "max": hi,
                      "size": [hi[i] - lo[i] for i in range(3)]})
    if off != len(buf):
        raise ValueError("trailing bytes: parsed %d of %d" % (off, len(buf)))
    return {"timestamp_sec": sec, "timestamp_nsec": nsec, "zones": zones}


def main(argv):
    with open(argv[1], "rb") as fh:
        res = parse_zones(fh.read())
    with open(argv[2], "w") as fh:
        json.dump(res, fh, indent=1)
    print(len(res["zones"]), "zones")


if __name__ == "__main__":
    main(sys.argv)
