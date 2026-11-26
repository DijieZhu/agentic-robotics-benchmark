"""The truth of one flight, read from <truth_dir>/<flight>/ and checked against the configuration before anything is
graded. Any defect is a TruthError: a task error, never a verdict on a twin.

Files (written by the truth freeze; this module defines their form):
  truth_meta.json   {"format": "flight-twin-truth/1", "flight": name, "key_frames": [K frame names],
                     "constants": {configuration key: value, ...}, "files": {file name: sha256, ...},
                     "reference_sha256": sha256 of the reference twin's folder (tree_digest)}
                    "constants" holds every configuration value that shaped these files; each must equal the
                    configuration's value (REQUIRED_CONSTANTS at least; a key the configuration does not hold is refused,
                    with one exception: when key_frames.named is false the configuration holds no key_frames.list, and
                    a recorded key_frames.list is the hidden index's own record: its entry for this flight must equal
                    "key_frames").
                    "key_frames" is the truth's hidden index of the scored moments: the grader takes them from here.
                    With key_frames.named true (the first version) they must be the configuration's key_frames.list.
  depth_key.npy     float32 or float64 (K, H, W): the recorded range along each pixel's ray at each key frame (m);
                    finite at every scored pixel, ignored elsewhere.
  scored_sets.npy   uint8 (K, H, W) bit field: HAZARD (1) the twin's range at this pixel feeds the hazard check (the
                    real frame returns here and the pixel is not masked as moving or as the other free-flyer);
                    CONTENT (2) a scored content pixel; STRUCTURE (4) a scored structure pixel. CONTENT and STRUCTURE
                    exclude each other and each implies HAZARD; no bit is set where the ray table has no ray.
  moves.json        {"format": "flight-twin-moves/1", "moves": [{"key_frame": name, "direction": index into the move
                    family (elevation-major, as twinprobe.hazard.directions), "length": m, "class": "clear" |
                    "blocked_by_content" | "blocked_by_structure", "stop": m or null}, ...]}
                    clear: length is one of the family's lengths, stop null; blocked: length is the family's longest,
                    stop the nominal stop distance (a whole number of path samples).
"""
import hashlib
import json
import os

import numpy as np

TRUTH_FORMAT = "flight-twin-truth/1"
MOVES_FORMAT = "flight-twin-moves/1"
HAZARD, CONTENT, STRUCTURE = 1, 2, 4
CLASSES = ("clear", "blocked_by_content", "blocked_by_structure")
FILES = ("depth_key.npy", "scored_sets.npy", "moves.json")
# the configuration values every truth must record (it may record more keys of the configuration)
REQUIRED_CONSTANTS = (
    "key_frames.per_flight", "key_frames.time_offset_wide_minus_depth_stamp",
    "wide_camera.body_mount_translation", "wide_camera.body_mount_quaternion_xyzw",
    "two_camera_transform.translation", "two_camera_transform.quaternion_xyzw",
    "depth_camera.image_size", "depth_camera.ray_table_sha256",
    "depth.scored_range_min", "depth.scored_range_max", "depth.band_full_abs", "depth.band_full_rel",
    "depth.band_zero_factor", "depth.min_class_pixels_per_frame",
    "hazard.voxel", "hazard.min_range", "hazard.max_range", "hazard.inflation_radius", "hazard.inflation_offsets",
    "hazard.path_sample", "hazard.move_start", "hazard.directions_azimuth", "hazard.directions_elevation",
    "hazard.lengths", "hazard.max_length", "gate.ray_cutoff", "gate.mesh_cast",
)


class TruthError(Exception):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def tree_digest(root):
    """sha256 over the sorted (relative path, sha256 of content) of every regular file under root (symbolic links and
    __pycache__ are refused)."""
    items = []
    for top, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for d in dirs:
            if os.path.islink(os.path.join(top, d)) or d == "__pycache__":
                raise TruthError("the reference folder holds a link or a cache folder")
        for f in sorted(files):
            p = os.path.join(top, f)
            if os.path.islink(p):
                raise TruthError("the reference folder holds a symbolic link")
            with open(p, "rb") as fh:
                items.append("%s %s" % (os.path.relpath(p, root).replace(os.sep, "/"), sha256_bytes(fh.read())))
    return sha256_bytes("\n".join(items).encode("utf-8"))


class FlightTruth:
    def __init__(self, name, key_frames, depth, sets, moves):
        self.name = name
        self.key_frames = list(key_frames)
        self.depth = depth                  # (K, H, W) float64
        self.sets = sets                    # (K, H, W) uint8
        self.moves = moves                  # list of dicts
        self.content = (sets & CONTENT) != 0
        self.structure = (sets & STRUCTURE) != 0
        self.hazard = (sets & HAZARD) != 0

    def counts(self):
        return {"key_frames": len(self.key_frames), "content_pixels": int(self.content.sum()),
                "structure_pixels": int(self.structure.sum()),
                "moves": {c: sum(1 for m in self.moves if m["class"] == c) for c in CLASSES}}


def _read_json(path, what):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return data, json.loads(data.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise TruthError("%s cannot be read: %s" % (what, exc)) from None


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# the constant a truth may record that the configuration does not hold when the scored moments are not named
HIDDEN_INDEX_CONSTANT = "key_frames.list"


def check_constants(constants, cfg, what, hidden_index=None):
    """Every recorded constant equals the configuration's value. hidden_index (name, key frames): with
    key_frames.named false, a recorded key_frames.list is checked against the truth's own index instead (its entry for
    this flight must be the index; the configuration holds no list)."""
    if not isinstance(constants, dict):
        raise TruthError("%s: constants is not an object" % what)
    miss = [k for k in REQUIRED_CONSTANTS if k not in constants]
    if miss:
        raise TruthError("%s does not record the constants %s" % (what, miss))
    for k in sorted(constants):
        if k == HIDDEN_INDEX_CONSTANT and hidden_index is not None:
            rec = constants[k]
            name, kf = hidden_index
            if not isinstance(rec, dict) or rec.get(name) != kf:
                raise TruthError("%s records a key_frames.list whose entry for %s is not its own key frames" % (what, name))
            continue
        if k not in cfg:
            raise TruthError("%s records %r, which the configuration does not hold" % (what, k))
        if not _same(constants[k], cfg[k]):
            raise TruthError("%s was built with %s = %r, the configuration holds %r" % (what, k, constants[k], cfg[k]))


def load_flight(truth_dir, reference_dir, name, cfg, rays, n_dirs):
    fdir = os.path.join(truth_dir, name)
    what = "the truth of %s" % name
    _, meta = _read_json(os.path.join(fdir, "truth_meta.json"), what + " (truth_meta.json)")
    if not isinstance(meta, dict) or meta.get("format") != TRUTH_FORMAT or meta.get("flight") != name:
        raise TruthError("%s: truth_meta.json is not %s for this flight" % (what, TRUTH_FORMAT))
    kf = meta.get("key_frames")
    if cfg["key_frames.named"]:                     # the first version: the configuration names the scored moments
        check_constants(meta.get("constants"), cfg, what)
        K = cfg["key_frames.per_flight"]
        listed = cfg["key_frames.list"]
        if name not in listed:
            raise TruthError("the configuration's key_frames.list holds no key frames for %s" % name)
        if kf != listed[name] or not isinstance(kf, list) or len(kf) != K or len(set(kf)) != K:
            raise TruthError("%s: the key frames are not the configuration's %d key frames" % (what, K))
    else:                                           # rule L: the truth's own hidden index names them
        K = cfg["key_frames.per_flight"]
        if not isinstance(kf, list) or len(kf) != K or len(set(kf)) != K or not all(isinstance(x, str) for x in kf):
            raise TruthError("%s: truth_meta.json does not hold %d distinct key frames" % (what, K))
        check_constants(meta.get("constants"), cfg, what, hidden_index=(name, kf))
    files = meta.get("files")
    if not isinstance(files, dict) or sorted(files) != sorted(FILES):
        raise TruthError("%s: truth_meta.json must list the sha256 of %s" % (what, ", ".join(FILES)))
    blobs = {}
    for f in FILES:
        try:
            with open(os.path.join(fdir, f), "rb") as fh:
                blobs[f] = fh.read()
        except OSError as exc:
            raise TruthError("%s: %s cannot be read (%s)" % (what, f, exc)) from None
        if sha256_bytes(blobs[f]) != files[f]:
            raise TruthError("%s: %s is not the file truth_meta.json records" % (what, f))
    ref = os.path.join(reference_dir, name)
    if not os.path.isdir(ref) or tree_digest(ref) != meta.get("reference_sha256"):
        raise TruthError("%s: the reference twin is not the one truth_meta.json records" % what)
    import io
    depth = np.load(io.BytesIO(blobs["depth_key.npy"]), allow_pickle=False)
    sets = np.load(io.BytesIO(blobs["scored_sets.npy"]), allow_pickle=False)
    H, W = rays.shape[:2]
    if depth.shape != (K, H, W) or depth.dtype not in (np.float32, np.float64):
        raise TruthError("%s: depth_key.npy has shape %s and type %s" % (what, depth.shape, depth.dtype))
    if sets.shape != (K, H, W) or sets.dtype != np.uint8:
        raise TruthError("%s: scored_sets.npy has shape %s and type %s" % (what, sets.shape, sets.dtype))
    depth = depth.astype(np.float64)
    if np.any(sets & ~np.uint8(HAZARD | CONTENT | STRUCTURE)):
        raise TruthError("%s: scored_sets.npy holds unknown bits" % what)
    c, s, h = (sets & CONTENT) != 0, (sets & STRUCTURE) != 0, (sets & HAZARD) != 0
    if np.any(c & s) or np.any((c | s) & ~h):
        raise TruthError("%s: a pixel is both content and structure, or scored without the hazard bit" % what)
    noray = ~np.all(np.isfinite(rays), axis=2)
    if np.any(sets[:, noray]):
        raise TruthError("%s: a pixel without a ray carries a bit" % what)
    r = depth[c | s]
    lo, hi = cfg["depth.scored_range_min"], cfg["depth.scored_range_max"]
    if not np.all(np.isfinite(r)) or np.any(r < lo) or np.any(r > hi):
        raise TruthError("%s: a scored range is not finite or lies outside [%g, %g] m" % (what, lo, hi))
    n_min = cfg["depth.min_class_pixels_per_frame"]
    for cls, m in (("content", c), ("structure", s)):
        if not np.any(m.reshape(K, -1).sum(axis=1) >= n_min):
            raise TruthError("%s: no key frame has %d scored %s pixels" % (what, n_min, cls))
    _, mv = _read_json(os.path.join(fdir, "moves.json"), what + " (moves.json)")
    if not isinstance(mv, dict) or mv.get("format") != MOVES_FORMAT or not isinstance(mv.get("moves"), list):
        raise TruthError("%s: moves.json is not %s" % (what, MOVES_FORMAT))
    step = cfg["hazard.path_sample"]
    Lmax = cfg["hazard.max_length"]
    lengths = cfg["hazard.lengths"]
    moves = []
    for k, m in enumerate(mv["moves"]):
        ok = (isinstance(m, dict) and set(m) == {"key_frame", "direction", "length", "class", "stop"}
              and m["key_frame"] in kf and isinstance(m["direction"], int) and not isinstance(m["direction"], bool)
              and 0 <= m["direction"] < n_dirs and m["class"] in CLASSES)
        if ok and m["class"] == "clear":
            ok = m["length"] in lengths and m["stop"] is None
        elif ok:
            st = m["stop"]
            ok = (m["length"] == Lmax and isinstance(st, (int, float)) and not isinstance(st, bool)
                  and 0.0 <= st <= Lmax and abs(st / step - round(st / step)) < 1e-6)
        if not ok:
            raise TruthError("%s: move %d of moves.json is malformed" % (what, k))
        moves.append(dict(m))
    for cls in CLASSES:
        if not any(m["class"] == cls for m in moves):
            raise TruthError("%s: moves.json holds no move of class %s" % (what, cls))
    return FlightTruth(name, kf, depth, sets, moves)
