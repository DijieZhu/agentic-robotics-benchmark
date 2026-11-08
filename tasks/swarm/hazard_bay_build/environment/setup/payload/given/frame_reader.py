"""Read the bay mission's sensor frames (control.md, The sensors): the manifest and the slot files of the folder that
`obs["mission"]["sensors"]` names. NumPy only; copy it into your submission to use it there.

    from frame_reader import Frames
    fr = Frames(obs["mission"]["sensors"])         # once: reads the manifest, maps nothing yet
    for s in fr.sensors:                           # the streamed sensors, in the manifest's order
        img, head = fr.read(s["id"], obs["frames"])    # the frame set the observation names (a copy), or (None, None)

A slot file is a 64-byte header (little-endian: magic b"BAYFRM1\\0", u64 seq, u64 n, u64 capture step t, f64 capture
time, u32 width, u32 height, u32 channels, u32 dtype (1 uint8, 2 float32), u64 payload bytes) and the payload,
row-major. The writer makes seq odd while it writes and even when the frame is whole; frame n sits in slot n % 2
until frame n + 2 replaces it. Inside a reply nothing writes; a read between requests is checked against seq.
"""

import json
import mmap
import os
import struct

import numpy as np

HEADER = struct.Struct("<8sQQQdIIIIQ")
MAGIC = b"BAYFRM1\x00"
DTYPES = {1: np.uint8, 2: np.float32}


class Frames:
    def __init__(self, sensors_block):
        self.dir = sensors_block["dir"]
        with open(os.path.join(self.dir, sensors_block.get("manifest", "manifest.json"))) as f:
            self.manifest = json.load(f)
        self.sensors = self.manifest["sensors"]
        self.by_id = {s["id"]: s for s in self.sensors}
        self.slots = int(self.manifest.get("slots", 2))
        self._maps = {}

    def _map(self, sid, slot):
        key = (sid, slot)
        if key not in self._maps:
            with open(os.path.join(self.dir, f"{self.by_id[sid]['file']}.{slot}"), "rb") as f:
                self._maps[key] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[key]

    def read(self, sensor_id, frames, copy=True, tries=50):
        """The frame of sensor `sensor_id` from the frame set `frames` names ({n, t, time, slot}): (array, header) with
        the array H x W (x 3 for RGB), or (None, None) when that slot no longer holds frame n."""
        if not frames:
            return None, None
        mm = self._map(sensor_id, int(frames["slot"]))
        for _ in range(tries):
            magic, seq, n, t, tm, w, h, c, code, nbytes = HEADER.unpack_from(mm, 0)
            if magic != MAGIC:
                raise ValueError(f"{sensor_id}: not a frame slot")
            if seq % 2:
                continue  # (being written: only between requests)
            dt = DTYPES[code]
            arr = np.frombuffer(mm, dtype=dt, count=nbytes // np.dtype(dt).itemsize, offset=HEADER.size)
            arr = arr.reshape((h, w, c) if c > 1 else (h, w))
            if copy:
                arr = arr.copy()
            if struct.unpack_from("<Q", mm, 8)[0] != seq:
                continue  # (rewritten while copied)
            if n != int(frames["n"]):
                return None, None
            return arr, dict(n=n, t=t, time=tm, width=w, height=h, channels=c, seq=seq)
        return None, None
