"""Reader of lab logs (JSON lines: one header line, then one line per tick).

    log = read_log("lab_logs/circle_r1p5_v2p0.jsonl")
    log.header                     # schema, name, description, floor {kind, grip}, tick_s, tracker_rate_hz, ticks
    log.cmd, log.scan, log.imu     # (n, 2), (n, 1081), (n, 5, 6) numpy arrays (lists without numpy)
    log.wheel_speed, log.steer_echo
    log.tracker                    # (m, 4): [t_ms, x, y, yaw] of the car frame in the lab frame, 100 Hz
    log.rows                       # the raw rows; row["obs"] is exactly what a step reply carries

A row's `cmd` is the [steer, speed] sent with that step; its `obs` is the reply to it.
"""

import json

LAB_LOG_SCHEMA = "racecar-lab-log/1"
HEADER_KEYS = ("schema", "name", "description", "floor", "tick_s", "tracker_rate_hz", "ticks")
ROW_KEYS = ("tick", "cmd", "obs", "tracker")
OBS_KEYS = ("scan", "imu", "wheel_speed", "steer_echo")

try:
    import numpy as _np
except ImportError:                          # pragma: no cover
    _np = None


class LabLog(object):
    def __init__(self, header, rows, numpy=None):
        self.header = header
        self.rows = rows
        use_np = (_np is not None) if numpy is None else (bool(numpy) and _np is not None)
        conv = (lambda v, dt: _np.asarray(v, dtype=dt)) if use_np else (lambda v, dt: v)
        self.tick = conv([r["tick"] for r in rows], "int64")
        self.cmd = conv([r["cmd"] for r in rows], "float64")
        self.scan = conv([r["obs"]["scan"] for r in rows], "int32")
        self.imu = conv([r["obs"]["imu"] for r in rows], "float64")
        self.wheel_speed = conv([r["obs"]["wheel_speed"] for r in rows], "float64")
        self.steer_echo = conv([r["obs"]["steer_echo"] for r in rows], "float64")
        flat = [s for r in rows for s in r["tracker"]]
        self.tracker = conv(flat, "float64") if flat or not use_np else _np.zeros((0, 4))

    def __len__(self):
        return len(self.rows)

    def obs(self, k):
        """The observation of row k as a step reply would carry it (without `tick` and `events`)."""
        return self.rows[k]["obs"]


def check_log(header, rows):
    """Errors of a parsed lab log: keys, schema name, tick numbering, row count, sizes."""
    err = []
    if not isinstance(header, dict) or set(header) != set(HEADER_KEYS):
        return ["the header has not the keys %s" % (HEADER_KEYS,)]
    if header["schema"] != LAB_LOG_SCHEMA:
        err.append("header schema %r is not %r" % (header["schema"], LAB_LOG_SCHEMA))
    if header["ticks"] != len(rows):
        err.append("header says %r ticks, the file has %d rows" % (header["ticks"], len(rows)))
    for k, r in enumerate(rows):
        if not isinstance(r, dict) or set(r) != set(ROW_KEYS) or set(r["obs"]) != set(OBS_KEYS):
            err.append("row %d has not the keys %s / obs %s" % (k + 1, ROW_KEYS, OBS_KEYS))
            break
        if r["tick"] != k + 1:
            err.append("row %d carries tick %r" % (k + 1, r["tick"]))
            break
        if len(r["obs"]["scan"]) != 1081 or len(r["obs"]["imu"]) != 5 or len(r["cmd"]) != 2:
            err.append("row %d: scan, imu or cmd has the wrong size" % (k + 1))
            break
    return err


def read_log(path, numpy=None, check=True):
    """Parse a lab log file. With `check` a malformed file raises ValueError."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        raise ValueError("%s is empty" % path)
    header = json.loads(lines[0])
    rows = [json.loads(ln) for ln in lines[1:]]
    if check:
        err = check_log(header, rows)
        if err:
            raise ValueError("%s: %s" % (path, "; ".join(err)))
    return LabLog(header, rows, numpy=numpy)
