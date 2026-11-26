"""The grader's copies of the givens it reads: per flight the frame stamps (frames.csv) and the on-board estimate
(onboard_estimate.csv); the depth camera's ray table (depth_rays.npy). Paths come from the configuration's layout.

    gv = givens.load(stage_dir, cfg)      # Givens: gv.rays (H, W, 3); gv.flights[name].stamp(frame), .estimate

A given that cannot be read, or that disagrees with the configuration (the ray table's sha256, its shape, its count of
never-returning pixels), is a GivenError: a task error, never a verdict on a twin.

CSV form: lines starting with '#' are comments; the first other line is the header; columns are read by name
(frames.csv: frame, t_s; onboard_estimate.csv: t_s, x_m, y_m, z_m, qx, qy, qz, qw, other columns ignored).
"""
import hashlib
import io
import math
import os

import numpy as np

ESTIMATE_COLUMNS = ("t_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")


class GivenError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    """(header, rows) of a comma-separated file; '#' lines are comments."""
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GivenError("%s cannot be read: %s" % (os.path.basename(path), exc)) from None
    header, rows = None, []
    for line in io.StringIO(text):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            header = cells
        elif len(cells) != len(header):
            raise GivenError("%s: a row has %d cells, the header %d" % (os.path.basename(path), len(cells), len(header)))
        else:
            rows.append(cells)
    if header is None:
        raise GivenError("%s has no header" % os.path.basename(path))
    return header, rows


class FlightGivens:
    def __init__(self, name, stamps, times, positions, quats_xyzw):
        self.name = name
        self.stamps = stamps                # {frame name: recorded image stamp t_s}
        self.times = times
        self.positions = positions
        self.quats_xyzw = quats_xyzw

    def stamp(self, frame):
        try:
            return self.stamps[frame]
        except KeyError:
            raise GivenError("%s: frames.csv has no frame %r" % (self.name, frame)) from None


def _floats(rows, idx, what):
    try:
        a = np.array([[float(r[i]) for i in idx] for r in rows], dtype=float)
    except ValueError as exc:
        raise GivenError("%s: a value is not a number (%s)" % (what, exc)) from None
    if a.size and not np.all(np.isfinite(a)):
        raise GivenError("%s holds a value that is not finite" % what)
    return a


def load_flight(root, name, layout):
    fpath = os.path.join(root, layout["frames_csv"].format(flight=name))
    epath = os.path.join(root, layout["estimate_csv"].format(flight=name))
    head, rows = read_csv(fpath)
    if "frame" not in head or "t_s" not in head:
        raise GivenError("%s frames.csv: columns frame and t_s are required" % name)
    fi, ti = head.index("frame"), head.index("t_s")
    stamps = {}
    for r in rows:
        try:
            t = float(r[ti])
        except ValueError:
            raise GivenError("%s frames.csv: t_s %r is not a number" % (name, r[ti])) from None
        if not math.isfinite(t) or r[fi] in stamps:
            raise GivenError("%s frames.csv: a stamp is not finite or a frame is listed twice" % name)
        stamps[r[fi]] = t
    head, rows = read_csv(epath)
    miss = [c for c in ESTIMATE_COLUMNS if c not in head]
    if miss:
        raise GivenError("%s onboard_estimate.csv: columns %s are missing" % (name, miss))
    a = _floats(rows, [head.index(c) for c in ESTIMATE_COLUMNS], "%s onboard_estimate.csv" % name)
    if len(a) < 2 or not np.all(np.diff(a[:, 0]) > 0):
        raise GivenError("%s onboard_estimate.csv: times must be strictly increasing (at least two rows)" % name)
    return FlightGivens(name, stamps, a[:, 0].copy(), a[:, 1:4].copy(), a[:, 4:8].copy())


def load_rays(path, cfg):
    """The ray table: (H, W, 3) unit rays in the depth frame, NaN at never-returning pixels; checked against the
    configuration (sha256, shape, count of NaN pixels, unit length)."""
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise GivenError("the ray table cannot be read: %s" % exc) from None
    if digest != cfg["depth_camera.ray_table_sha256"]:
        raise GivenError("the ray table's sha256 %s is not the configuration's" % digest[:12])
    rays = np.load(path, allow_pickle=False)
    W, H = cfg["depth_camera.image_size"]
    if rays.shape != (H, W, 3) or rays.dtype != np.float64:
        raise GivenError("the ray table has shape %s and type %s, expected (%d, %d, 3) float64" % (rays.shape, rays.dtype, H, W))
    if H * W != cfg["depth_camera.rays"]:
        raise GivenError("depth_camera.rays is not the image size's product")
    bad = ~np.all(np.isfinite(rays), axis=2)
    if int(bad.sum()) != cfg["depth_camera.never_returning_corner_pixels"]:
        raise GivenError("the ray table has %d non-finite rays, the configuration states %d"
                         % (int(bad.sum()), cfg["depth_camera.never_returning_corner_pixels"]))
    if np.any(np.isfinite(rays) != ~bad[..., None]):
        raise GivenError("a ray of the table is partly finite")
    n = np.linalg.norm(rays[~bad], axis=1)
    if n.size and float(np.max(np.abs(n - 1.0))) > 1e-9:
        raise GivenError("the ray table's rays are not unit vectors")
    return rays


class Givens:
    def __init__(self, rays, flights, digest):
        self.rays = rays
        self.flights = flights
        self.ray_table_sha256 = digest


def load(stage_dir, cfg):
    root = os.path.join(stage_dir, cfg.layout["givens_dir"])
    rpath = os.path.join(root, cfg.layout["ray_table"])
    rays = load_rays(rpath, cfg)
    flights = {n: load_flight(root, n, cfg.layout) for n in cfg["flights.names_neutral"]}
    return Givens(rays, flights, cfg["depth_camera.ray_table_sha256"])
