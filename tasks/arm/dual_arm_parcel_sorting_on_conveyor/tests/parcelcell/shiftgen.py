"""The shift generator: one shift from one seed and the ``shifts`` block of
the configuration; JSON shift packages.

A shift has one belt speed, drawn from a belt-speed class; a parcel mix class
(the share of soft mailers: one class, one parcel in five); a number of
parcels drawn within the belt class's range (``parcels_by_class``: a faster
belt brings more parcels in the same time; ``parcels`` when a class has no
entry; bench shifts: ``bench_parcels`` on the slowest class). The gap between two consecutive
parcels is drawn as BELT DISTANCE, by one rule for every pair whatever their
kinds or label positions (so the arrival pattern tells nothing about a
parcel's class):

    gap = 0.5 (diag_prev + diag_next) + entry_clearance + margin
          + random part (exponential, mean gap_extra_m, capped at
            gap_extra_cap_m; bench shifts: uniform in bench_gap_extra_m)
          + v x arrival_gap_min.t_s + arrival_gap_min.d_m,

and the time between the arrivals is gap / belt speed v (the minimum is the
station's rated job time: a parcel never reaches the work zone before the
job on the one ahead of it can end). A shift's stated arrival times are then
its real ones; a replay that shows an entry delay is a generator fault
(the record reports ``max_entry_delay_s``).

Arrival times lie on the 20 ms tick grid (a parcel lands at the start of the
tick at its arrival time). Duration: fixed, ``duration_s`` for every full shift and ``bench_duration_s`` for every
bench shift, whatever the belt speed and the arrivals. A draw whose natural end (last arrival + (scan line - entry) /
belt speed + ``tail_s``) is later than that is drawn again (count, speed, parcels and arrivals, from the same stream;
the shift's meta records ``redraws`` and ``natural_end_s``), so the last parcel of every shift has passed the exit
before the shift ends and nothing on the wire bounds the last arrival or the parcel count.

Parcels: cartons (rigid, at least 0.11 m high: no flat cartons) and soft
mailers with the configured size, mass and friction ranges. The label is on one of the faces that can hold the waybill
(both sides at least 8 cm; mailers: top or bottom only), drawn with
probability proportional to the face's area (``stream.draw_label_face``).
Arrivals come justified by the line's upstream infeed: each parcel lies
against the near-side guide, aligned with the belt: its near face lies
``near_face_gap_m`` (0-1 cm) inside the belt's near edge, its yaw is a
multiple of 90 degrees within +-``yaw_noise_deg``. Sizes on a millimetre
grid, speeds on a 1 mm/s grid.

Seeds: ``labelled_seed(label, k)`` = the first 8 bytes of
sha256("<task id>/<label>/<version>/<k>") >> 1 (63 bits); labels "hidden" and
"dev" give disjoint candidate lists by construction. ``mix`` (bytes, e.g. a
recorded master draw from OS entropy) is hashed into the seed when given.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .station import StationConfig
from .stream import Arrival, Parcel, Shift, draw_label_face

TICK_S = 0.02


def _on_tick(t: float) -> float:
    """The next control tick at or after t (arrivals land on tick starts, so
    a stated arrival time is the real one)."""
    return round(math.ceil(t / TICK_S - 1e-9) * TICK_S, 2)

__all__ = ["labelled_seed", "draw_shift", "shift_features", "justified_pose", "to_package",
           "from_package", "package_sha256", "load_package", "save_package",
           "PACKAGE_FORMAT", "GENERATOR_VERSION", "LINE_GENERATOR_VERSION", "draw_line_shift",
           "line_features", "line_list_length", "line_seed", "line_setting", "LINE_LABEL_POSITIONS"]

PACKAGE_FORMAT = "parcelcell.shift/1"
GENERATOR_VERSION = "parcel-shiftgen/5"
LINE_GENERATOR_VERSION = "parcel-linegen/1"
LINE_LABEL_POSITIONS = ("underneath", "downstream", "upstream", "near_side", "far_side")


def labelled_seed(label: str, k: int, task_id: str = "dual_arm_parcel_sorting_on_conveyor",
                  version: str = "v1", mix: bytes | None = None) -> int:
    text = ("%s/%s/%s/%d" % (task_id, label, version, int(k))).encode()
    h = hashlib.sha256(text + (b"/" + mix if mix else b"")).digest()
    return int.from_bytes(h[:8], "big") >> 1


def _uniform(rng, lohi):
    return float(rng.uniform(float(lohi[0]), float(lohi[1])))


def _yaw_half_extent_x(sx, sy, yaw):
    """Half extent along world x of a footprint sx x sy turned by yaw."""
    return 0.5 * (abs(sx * math.cos(yaw)) + abs(sy * math.sin(yaw)))


def justified_pose(rng, sc: dict, size, belt_half: float) -> tuple:
    """(yaw, lateral) of a justified arrival: yaw = k x 90 degrees (k uniform
    in 0..3) + a uniform error within +-yaw_noise_deg; the near face (the
    footprint's extent towards -x) ``near_face_gap_m`` inside the belt's
    near edge. ``lateral`` is the centre's offset from the belt centre line."""
    k = int(rng.integers(4))
    noise = math.radians(float(sc["yaw_noise_deg"]))
    yaw = round(k * math.pi / 2 + float(rng.uniform(-noise, noise)), 4)
    gap = _uniform(rng, sc["near_face_gap_m"])
    lateral = round(-belt_half + gap + _yaw_half_extent_x(size[0], size[1], yaw), 4)
    return yaw, lateral


def _parcel(rng, sc: dict, kind: str, belt_half: float) -> Parcel:
    rng_size = sc["carton_size_m"] if kind == "rigid" else sc["mailer_size_m"]
    size = tuple(round(_uniform(rng, r), 3) for r in rng_size)
    mass = round(_uniform(rng, sc["mass_kg"]), 3)
    friction = round(_uniform(rng, sc["friction"]), 3)
    faces = (0, 1, 2, 3, 4, 5) if kind == "rigid" else tuple(sc["mailer_label_faces"])
    face = draw_label_face(rng, size, kind, faces)
    m = float(sc["label_uv_max"])
    uv = (round(float(rng.uniform(-m, m)), 3), round(float(rng.uniform(-m, m)), 3))
    yaw, lateral = justified_pose(rng, sc, size, belt_half)
    co = float(sc["com_offset_max_m"])
    if co > 0:
        com = (round(float(rng.uniform(-1, 1)) * min(co, 0.8 * size[0] / 2), 4),
               round(float(rng.uniform(-1, 1)) * min(co, 0.8 * size[1] / 2), 4),
               round(float(rng.uniform(-0.3, 0.3)) * size[2] / 2, 4))
    else:
        com = (0.0, 0.0, 0.0)
    return Parcel(kind=kind, size=size, mass=mass, friction=friction,
                  com_offset=com, label_face=face, label_uv=uv, yaw=yaw,
                  lateral=lateral)


def draw_shift(seed: int, cfg: dict, profile: str = "full",
               belt_class: str | None = None, mix_class: str | None = None,
               label: str | None = None, k: int | None = None,
               n_parcels: int | None = None) -> tuple[Shift, dict]:
    """(Shift, meta) for one seed. ``cfg``: the whole configuration (its
    ``shifts`` and ``world`` blocks are read). profile: "full" | "bench".
    ``n_parcels`` (bench profile only): the shift's number of parcels instead
    of the drawn one (the draw still happens, so every other draw is the
    same); None leaves every shift exactly as without it."""
    sc, w = cfg["shifts"], cfg["world"]
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF,
                                 0x5A1F7])
    classes = sorted(sc["belt_speed_classes"].items(), key=lambda kv: kv[1][0])
    mixes = sorted(sc["mix_classes"].items(), key=lambda kv: kv[1])
    if profile == "bench":
        bclass = belt_class or classes[0][0]
        mclass = mix_class or mixes[int(rng.integers(len(mixes)))][0]
        limit = float(sc["bench_duration_s"])
        nlo, nhi = int(sc["bench_parcels"][0]), int(sc["bench_parcels"][1])
    elif profile == "full":
        bclass = belt_class or classes[int(rng.integers(len(classes)))][0]
        mclass = mix_class or mixes[int(rng.integers(len(mixes)))][0]
        limit = float(sc["duration_s"])
        n_range = (sc.get("parcels_by_class") or {}).get(bclass) or sc["parcels"]
        nlo, nhi = int(n_range[0]), int(n_range[1])
    else:
        raise ValueError("profile must be full or bench")
    soft_share = float(sc["mix_classes"][mclass])
    belt_half = 0.5 * float(w.get("station", {}).get("belt_width", StationConfig().belt_width))
    clearance = float(w["entry_clearance_m"]) + float(sc["gap_margin_m"])
    gmin = sc["arrival_gap_min"]
    redraws = 0
    while True:
        # one draw of the shift (count, speed within the class, parcels, arrivals); a draw whose natural end is
        # past the fixed duration is drawn again from the same stream
        n = int(rng.integers(nlo, nhi + 1))
        if n_parcels is not None and profile == "bench":
            n = int(n_parcels)
        v = round(_uniform(rng, sc["belt_speed_classes"][bclass]), 3)
        parcels = []
        for _ in range(n):
            kind = "soft" if rng.random() < soft_share else "rigid"
            parcels.append(_parcel(rng, sc, kind, belt_half))
        times = []
        t = _on_tick(_uniform(rng, sc["first_arrival_s"]))
        gaps_m = []
        for i, p in enumerate(parcels):
            if i > 0:
                if profile == "bench":
                    extra = _uniform(rng, sc["bench_gap_extra_m"])
                elif sc["gap_extra"] == "exponential":
                    extra = min(float(rng.exponential(float(sc["gap_extra_m"]))), float(sc["gap_extra_cap_m"]))
                elif sc["gap_extra"] == "uniform":
                    extra = float(rng.uniform(0.0, float(sc["gap_extra_m"])))
                else:
                    extra = 0.0
                gap = (0.5 * (parcels[i - 1].diag + p.diag) + clearance + extra
                       + v * float(gmin["t_s"]) + float(gmin["d_m"]))
                gaps_m.append(round(gap, 6))
                t = _on_tick(t + gap / v)
            times.append(t)
        transit = (float(w["scan_y"]) - float(w["entry_y"])) / v
        natural_end = times[-1] + transit + float(sc["tail_s"])
        if natural_end <= limit + 1e-9:
            break
        redraws += 1
        if redraws > 100000:
            raise RuntimeError("no draw ends within %.0f s" % limit)
    duration = limit
    shift = Shift(belt_speed=((0.0, v),),
                  arrivals=tuple(Arrival(t=ti, parcel=p) for ti, p in zip(times, parcels)),
                  duration=duration, seed=int(seed))
    meta = {"generator": GENERATOR_VERSION, "seed": int(seed), "profile": profile,
            "belt_class": bclass, "mix_class": mclass, "label": label, "k": k,
            "gaps_m": gaps_m, "natural_end_s": round(natural_end, 3), "redraws": redraws}
    meta["features"] = shift_features(shift, cfg)
    return shift, meta


def shift_features(shift: Shift, cfg: dict) -> dict:
    """Counts the selection's quotas and the hollowing check use."""
    arr = shift.arrivals
    from .episode import CLASSES, label_position, parcel_class
    from .config import LABEL_POSITIONS
    faces = [a.parcel.label_face for a in arr]
    places = [label_position(a.parcel) for a in arr]
    duty = (cfg.get("score") or {}).get("duty")
    cls = [parcel_class(a.parcel, duty) for a in arr]
    return {
        "n_parcels": len(arr),
        "belt_speed": shift.belt_speed[0][1],
        "n_soft": sum(a.parcel.kind == "soft" for a in arr),
        "n_label_not_up": sum(f != 0 for f in faces),
        "label_faces": {str(f): faces.count(f) for f in range(6)},
        "label_positions": {pname: places.count(pname) for pname in LABEL_POSITIONS},
        "classes": {c: cls.count(c) for c in CLASSES},
        "duration_s": shift.duration,
        "mass_kg_max": max((a.parcel.mass for a in arr), default=0.0),
    }


def to_package(shift: Shift, meta: dict | None = None) -> dict:
    d = shift.to_dict()
    d["meta"] = dict(meta or {})
    return d


def from_package(d: dict) -> tuple[Shift, dict]:
    if d.get("format") != PACKAGE_FORMAT:
        raise ValueError("not a shift package (%r)" % d.get("format"))
    return Shift.from_dict(d), dict(d.get("meta") or {})


def package_bytes(d: dict) -> bytes:
    return (json.dumps(d, indent=1, sort_keys=True) + "\n").encode()


def package_sha256(d: dict) -> str:
    return hashlib.sha256(package_bytes(d)).hexdigest()


def save_package(path: str, shift: Shift, meta: dict | None = None) -> str:
    d = to_package(shift, meta)
    with open(path, "wb") as fh:
        fh.write(package_bytes(d))
    return package_sha256(d)


def load_package(path: str) -> tuple[Shift, dict, str]:
    with open(path, "rb") as fh:
        raw = fh.read()
    d = json.loads(raw)
    shift, meta = from_package(d)
    return shift, meta, hashlib.sha256(raw).hexdigest()


# ======================================================================
# the line setting: one parcel at a time, fed on settle (see world.py)
# ======================================================================
def line_list_length(v: float, lc: dict, duration: float | None = None) -> int:
    """The number of parcels of a full line shift's list at belt speed v: as many as fit into the shift's duration
    at the station's rated time per parcel, the ride from the set-down line to the scan line
    (``rated_ride_m`` / v) plus ``rated_handling_s``: floor(duration / (rated_ride_m / v + rated_handling_s))."""
    T = float(lc["duration_s"] if duration is None else duration)
    per = float(lc["rated_ride_m"]) / float(v) + float(lc["rated_handling_s"])
    return int(math.floor(T / per + 1e-9))


def line_setting(cfg: dict, indexing: bool) -> dict:
    lc = cfg["line"]
    return {"name": "line", "feed": "on_settle", "indexing": bool(indexing),
            "zone_y": [float(x) for x in lc["indexing_zone_y"]]}


def line_seed(label: str, k: int, cfg: dict, mix: bytes | None = None) -> int:
    """A line shift's seed: ``labelled_seed(label, k)`` with the line generator's own version string."""
    sc = cfg["shifts"]
    return labelled_seed(label, k, task_id=sc["task_id"], version=str(cfg["line"]["shifts"]["version"]), mix=mix)


def _line_parcel(rng, ls: dict, kind: str | None = None, label_face: int | None = None,
                 size_m=None) -> tuple:
    """One parcel of the line setting: kind rigid or soft with the configured shares, sizes per kind, mass,
    friction, label on any face whose two sides are both at least LABEL_MIN_FACE (uniform over those faces;
    the +-z faces when none is), label offset within +-label_uv_max of the free margin, yaw on the full circle,
    on the belt's centre line, the centre of mass offset as configured. ``kind`` / ``label_face`` / ``size_m``
    (development shifts): given instead of drawn (the draws still happen, so the stream stays the same)."""
    from .stream import valid_label_faces
    kinds = sorted(ls["kind_shares"].items())
    u = float(rng.random())
    acc, k_draw = 0.0, kinds[-1][0]
    for name, share in kinds:
        acc += float(share)
        if u < acc:
            k_draw = name
            break
    kind = kind or k_draw
    rs = ls["rigid_size_m"] if kind == "rigid" else ls["soft_size_m"]
    size = tuple(round(_uniform(rng, r), 3) for r in rs)
    if size_m is not None:
        size = tuple(round(float(v), 3) for v in size_m)
    mass = round(_uniform(rng, ls["mass_kg"]), 3)
    faces = valid_label_faces(size, (0, 1, 2, 3, 4, 5))
    f_draw = int(faces[int(rng.integers(len(faces)))])
    face = f_draw if label_face is None else int(label_face)
    friction = round(_uniform(rng, ls["friction"]), 3)
    yaw = round(_uniform(rng, ls["yaw_rad"]), 4)
    y = round(_uniform(rng, ls["set_down_y"]), 4)
    co = float(ls["com_offset_max_m"])
    com = (round(float(rng.uniform(-1, 1)) * min(co, 0.8 * size[0] / 2), 4),
           round(float(rng.uniform(-1, 1)) * min(co, 0.8 * size[1] / 2), 4),
           round(float(rng.uniform(-0.3, 0.3)) * size[2] / 2, 4))
    m = float(ls["label_uv_max"])
    uv = (round(float(rng.uniform(-m, m)), 3), round(float(rng.uniform(-m, m)), 3))
    return Parcel(kind=kind, size=size, mass=mass, friction=friction, com_offset=com, label_face=face,
                  label_uv=uv, yaw=yaw, lateral=0.0), y


def draw_line_shift(seed: int, cfg: dict, profile: str = "full", belt_class: str | None = None,
                    indexing: bool | None = None, label: str | None = None, k: int | None = None,
                    contents=None) -> tuple[Shift, dict]:
    """(Shift, meta) of the line setting for one seed. ``cfg``: the whole configuration (its ``line`` block).
    Draw order: belt class (uniform over the classes unless given), belt speed within the class (1 mm/s grid),
    indexing (probability ``indexing_share`` unless given), then the list: a full shift holds
    ``line_list_length(v)`` parcels, a bench shift ``len(contents)``; each parcel by ``_line_parcel``.
    ``contents`` (bench shifts): a list of dicts with optional kind / label_face / size_m per parcel."""
    ls = cfg["line"]["shifts"]
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 0x11E5])
    classes = sorted(ls["belt_speed_classes"].items(), key=lambda kv: kv[1][0])
    bdraw = classes[int(rng.integers(len(classes)))][0]
    bclass = belt_class or bdraw
    v = round(_uniform(rng, ls["belt_speed_classes"][bclass]), 3)
    idx_draw = bool(rng.random() < float(ls["indexing_share"]))
    idx = idx_draw if indexing is None else bool(indexing)
    if profile == "full":
        duration = float(ls["duration_s"])
        n = line_list_length(v, ls, duration)
        spec = [None] * n
    elif profile == "bench":
        duration = float(ls["bench_duration_s"])
        spec = list(contents or [{}])
        n = len(spec)
    else:
        raise ValueError("profile must be full or bench")
    arrivals = []
    for i in range(n):
        c = spec[i] or {}
        p, y = _line_parcel(rng, ls, kind=c.get("kind"), label_face=c.get("label_face"), size_m=c.get("size_m"))
        arrivals.append(Arrival(t=0.0, parcel=p, y=y))
    shift = Shift(belt_speed=((0.0, v),), arrivals=tuple(arrivals), duration=duration, seed=int(seed),
                  setting=line_setting(cfg, idx))
    meta = {"generator": LINE_GENERATOR_VERSION, "seed": int(seed), "profile": profile, "setting": "line",
            "belt_class": bclass, "indexing": idx, "label": label, "k": k}
    meta["features"] = line_features(shift, cfg)
    return shift, meta


def line_features(shift: Shift, cfg: dict) -> dict:
    """Counts of a line shift's list (classes by the line setting's duty: every parcel)."""
    from .episode import CLASSES, label_position, parcel_class
    from .config import LABEL_POSITIONS
    arr = shift.arrivals
    duty = cfg["line"]["score"]["duty"]
    places = [label_position(a.parcel) for a in arr]
    cls = [parcel_class(a.parcel, duty) for a in arr]
    return {
        "n_parcels": len(arr), "belt_speed": shift.belt_speed[0][1],
        "indexing": bool((shift.setting or {}).get("indexing")),
        "n_soft": sum(a.parcel.kind == "soft" for a in arr),
        "n_label_not_up": sum(p != "top" for p in places),
        "label_faces": {str(f): sum(a.parcel.label_face == f for a in arr) for f in range(6)},
        "label_positions": {pname: places.count(pname) for pname in LABEL_POSITIONS},
        "classes": {c: cls.count(c) for c in CLASSES},
        "n_flat": sum(a.parcel.kind == "rigid" and a.parcel.size[2] < 0.11 for a in arr),
        "duration_s": shift.duration,
        "mass_kg_max": max((a.parcel.mass for a in arr), default=0.0),
    }
