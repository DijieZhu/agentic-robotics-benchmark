"""Parcel stream: one parcel, one shift, and a sampler of shifts.

A ``Shift`` is a complete, serialisable description of what the belt brings:
the belt speed over time and a list of arrivals (time + parcel). The
simulator replays a shift exactly; ``Shift.from_json(shift.to_json())``
reproduces the same shift (floats are written with round-trip precision).

Default ranges are the generic simulator's (a multi-parcel scene and a
parcel randomisation); see ``StreamConfig``. The task's shifts come from
``shiftgen`` with the task's own ranges.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from ._geom import FACE_DIMS

__all__ = ["Parcel", "Arrival", "Shift", "StreamConfig", "sample_parcel",
           "sample_shift", "valid_label_faces", "LABEL_MIN_FACE",
           "label_face_weights", "draw_label_face"]

LABEL_MIN_FACE = 0.08     # a label is never put on a face with a side < 8 cm
KINDS = ("rigid", "soft")


@dataclass(frozen=True)
class Parcel:
    """One parcel. Sizes are full sizes (m) along the parcel's own x, y, z;
    the parcel lands with its z axis up, rotated by ``yaw`` about world z.

    label_face: 0:+z 1:-z 2:+x 3:-x 4:+y 5:-y (parcel frame).
    label_uv: placement on that face, fractions in [-1, 1] of the free margin
        along the face's two tangent axes (0, 0 = centred).
    lateral: offset of the landing point from the belt centre line (m, +x
        is away from the robot).
    com_offset: centre-of-mass offset in the parcel frame (m)."""
    kind: str = "rigid"
    size: tuple = (0.24, 0.18, 0.12)
    mass: float = 0.8
    friction: float = 0.55
    com_offset: tuple = (0.0, 0.0, 0.0)
    label_face: int = 1
    label_uv: tuple = (0.0, 0.0)
    yaw: float = 0.0
    lateral: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not 0 <= int(self.label_face) <= 5:
            raise ValueError("label_face must be 0..5")
        object.__setattr__(self, "size", tuple(float(v) for v in self.size))
        object.__setattr__(self, "com_offset",
                           tuple(float(v) for v in self.com_offset))
        object.__setattr__(self, "label_uv",
                           tuple(float(v) for v in self.label_uv))
        object.__setattr__(self, "label_face", int(self.label_face))
        for k in ("mass", "friction", "yaw", "lateral"):
            object.__setattr__(self, k, float(getattr(self, k)))
        if len(self.size) != 3 or min(self.size) <= 0:
            raise ValueError("size must be three positive numbers")

    @property
    def diag(self) -> float:
        """Footprint diagonal (upper bound of the along-belt extent)."""
        return math.hypot(self.size[0], self.size[1])


@dataclass(frozen=True)
class Arrival:
    t: float          # time the parcel is put on the belt entry (s, shift clock)
    parcel: Parcel
    y: float | None = None
    """Set-down position along the belt (m, world y); None: the station's entry line."""

    def __post_init__(self):
        object.__setattr__(self, "t", float(self.t))
        if self.y is not None:
            object.__setattr__(self, "y", float(self.y))


@dataclass(frozen=True)
class Shift:
    """belt_speed: piecewise-constant profile, a tuple of (t_start, speed)
    pairs sorted by time; the first t_start must be 0. Speed in m/s along +y
    (downstream). duration: shift length (s)."""
    belt_speed: tuple = ((0.0, 0.2),)
    arrivals: tuple = ()
    duration: float = 60.0
    seed: int | None = None      # informative only (what the sampler was given)
    setting: dict | None = None
    """None: arrivals at their stated times (``Arrival.t``). A dict: the shift's feed and belt rules (the line
    setting, ``shiftgen.LINE``): {"name": "line", "feed": "on_settle", "indexing": bool, "zone_y": [lo, hi]}."""

    def __post_init__(self):
        segs = tuple((float(t), float(v)) for t, v in self.belt_speed)
        if not segs or segs[0][0] != 0.0:
            raise ValueError("belt_speed must start at t = 0")
        if any(b[0] <= a[0] for a, b in zip(segs, segs[1:])):
            raise ValueError("belt_speed times must increase")
        object.__setattr__(self, "belt_speed", segs)
        arr = tuple(a if isinstance(a, Arrival) else Arrival(**a)
                    for a in self.arrivals)
        if any(b.t < a.t for a, b in zip(arr, arr[1:])):
            raise ValueError("arrivals must be sorted by time")
        object.__setattr__(self, "arrivals", arr)
        object.__setattr__(self, "duration", float(self.duration))
        if self.setting is not None:
            object.__setattr__(self, "setting", json.loads(json.dumps(dict(self.setting))))

    def speed_at(self, t: float) -> float:
        v = self.belt_speed[0][1]
        for t0, s in self.belt_speed:
            if t0 <= t:
                v = s
            else:
                break
        return v

    # ---- serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "format": "parcelcell.shift/1",
            "duration": self.duration,
            "seed": self.seed,
            "belt_speed": [list(s) for s in self.belt_speed],
            "arrivals": [_arrival_to_dict(a) for a in self.arrivals],
        }
        if self.setting is not None:          # absent from a shift with stated arrival times (its bytes unchanged)
            d["setting"] = json.loads(json.dumps(self.setting))
        return d

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Shift":
        if d.get("format") != "parcelcell.shift/1":
            raise ValueError(f"unknown shift format {d.get('format')!r}")
        arr = tuple(Arrival(t=a["t"], parcel=Parcel(**a["parcel"]), y=a.get("y"))
                    for a in d["arrivals"])
        return cls(belt_speed=tuple(tuple(s) for s in d["belt_speed"]),
                   arrivals=arr, duration=d["duration"], seed=d.get("seed"),
                   setting=d.get("setting"))

    @classmethod
    def from_json(cls, s: str) -> "Shift":
        return cls.from_dict(json.loads(s))


def _arrival_to_dict(a: Arrival) -> dict:
    d = {"t": a.t, "parcel": _parcel_to_dict(a.parcel)}
    if a.y is not None:
        d["y"] = a.y
    return d


def _parcel_to_dict(p: Parcel) -> dict:
    d = asdict(p)
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


# ======================================================================
@dataclass(frozen=True)
class StreamConfig:
    """Ranges of the shift sampler. A (lo, hi) pair of numbers means uniform
    on [lo, hi]; a tuple of other values means a uniform choice.

    Generic defaults: sizes, masses, frictions, kinds and label faces of a
    parcel randomisation; yaw full circle, lateral +-0.12 m and a 0.8 s
    minimum spawn interval; belt speed 0.05-0.35 m/s. A contact domain
    randomisation (an option) widens friction to (0.3, 1.5) and sets
    com_offset_max to 0.04 m; see ``StreamConfig.with_contact_dr``."""
    # belt
    belt_speed: tuple = (0.05, 0.35)
    speed_changes: tuple = (0, 0)        # number of speed changes per shift
    # arrivals: gap = min_gap + random part
    min_gap: float = 0.8                 # s
    gap_extra: str = "exponential"       # "exponential" | "uniform" | "none"
    gap_extra_scale: float = 1.5         # mean (exponential) or max (uniform), s
    first_arrival: tuple = (0.5, 1.5)    # s
    # parcels
    kinds: tuple = ("rigid", "soft")
    size_x: tuple = (0.16, 0.30)
    size_y: tuple = (0.14, 0.26)
    size_z: tuple = (0.07, 0.16)
    soft_size_z: tuple = (0.05, 0.10)
    mass: tuple = (0.2, 2.0)
    friction: tuple = (0.4, 0.75)
    com_offset_max: float = 0.0          # m, 0 = centred
    label_faces: tuple = (0, 1, 2, 3, 4, 5)
    label_uv_max: float = 0.55
    yaw: tuple = (-math.pi, math.pi)
    lateral: tuple = (-0.12, 0.12)

    def with_contact_dr(self) -> "StreamConfig":
        from dataclasses import replace
        return replace(self, friction=(0.3, 1.5), com_offset_max=0.04)


def _draw(rng: np.random.Generator, v):
    if isinstance(v, tuple) and len(v) == 2 \
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in v):
        return float(rng.uniform(v[0], v[1]))
    if isinstance(v, (tuple, list)):
        return v[int(rng.integers(len(v)))]
    return v


def valid_label_faces(size, faces=(0, 1, 2, 3, 4, 5)) -> list[int]:
    """Faces whose two sides are both >= 8 cm (a real shipping label does not
    fit on a thinner face). If none qualifies: the +-z faces."""
    ok = [int(f) for f in faces
          if min(size[FACE_DIMS[int(f)][0]], size[FACE_DIMS[int(f)][1]])
          >= LABEL_MIN_FACE]
    if not ok:
        ok = [f for f in (0, 1) if f in faces] or [0, 1]
    return ok


def label_face_weights(size, kind: str = "rigid",
                       faces=(0, 1, 2, 3, 4, 5)) -> dict[int, float]:
    """Probability of each label face: among the faces that
    can hold the waybill (short side >= 8 cm; soft mailers: top or bottom
    only), proportional to the face area."""
    if kind == "soft":
        cand = [f for f in (0, 1) if f in tuple(int(v) for v in faces)] \
            or [0, 1]
    else:
        cand = valid_label_faces(size, faces)
    area = {f: float(size[FACE_DIMS[f][0]] * size[FACE_DIMS[f][1]])
            for f in cand}
    tot = sum(area.values())
    return {f: a / tot for f, a in area.items()}


def draw_label_face(rng: np.random.Generator, size, kind: str = "rigid",
                    faces=(0, 1, 2, 3, 4, 5)) -> int:
    """One label face drawn with ``label_face_weights``. (The earlier
    ``_draw(rng, tuple(valid faces))`` read a two-face tuple such as (0, 1)
    as the range [0, 1], so those parcels always got face 0.)"""
    w = label_face_weights(size, kind, faces)
    keys = sorted(w)
    p = np.array([w[k] for k in keys])
    return int(keys[int(rng.choice(len(keys), p=p / p.sum()))])


def sample_parcel(rng: np.random.Generator, cfg: StreamConfig) -> Parcel:
    kind = _draw(rng, cfg.kinds)
    size = (_draw(rng, cfg.size_x), _draw(rng, cfg.size_y),
            _draw(rng, cfg.soft_size_z if kind == "soft" else cfg.size_z))
    size = tuple(float(v) for v in np.round(size, 3))     # mm grid
    mass = _draw(rng, cfg.mass)
    friction = _draw(rng, cfg.friction)
    face = draw_label_face(rng, size, kind, cfg.label_faces)
    uv = tuple(float(v) for v in
               rng.uniform(-cfg.label_uv_max, cfg.label_uv_max, 2))
    yaw = _draw(rng, cfg.yaw)
    lateral = _draw(rng, cfg.lateral)
    co = cfg.com_offset_max
    if co > 0:
        hx, hy = size[0] / 2, size[1] / 2
        com = (float(rng.uniform(-1, 1) * min(co, 0.8 * hx)),
               float(rng.uniform(-1, 1) * min(co, 0.8 * hy)),
               float(rng.uniform(-0.3, 0.3) * size[2] / 2))
    else:
        com = (0.0, 0.0, 0.0)
    return Parcel(kind=kind, size=size, mass=mass, friction=friction,
                  com_offset=com, label_face=face, label_uv=uv, yaw=yaw,
                  lateral=lateral)


def sample_shift(rng: np.random.Generator, cfg: StreamConfig | None = None, *,
                 n_parcels: int | None = None, duration: float | None = None,
                 seed: int | None = None) -> Shift:
    """Draw a shift. Give ``n_parcels`` (the shift then lasts until the last
    arrival plus the ride from the entry past the exit at the slowest drawn
    belt speed, plus 1 s) or ``duration`` (arrivals until then)."""
    cfg = cfg or StreamConfig()
    if n_parcels is None and duration is None:
        raise ValueError("give n_parcels or duration")
    times, parcels = [], []
    t = _draw(rng, cfg.first_arrival)
    while True:
        if n_parcels is not None and len(times) >= n_parcels:
            break
        if duration is not None and t > duration:
            break
        times.append(float(t))
        parcels.append(sample_parcel(rng, cfg))
        if cfg.gap_extra == "exponential":
            extra = float(rng.exponential(cfg.gap_extra_scale))
        elif cfg.gap_extra == "uniform":
            extra = float(rng.uniform(0.0, cfg.gap_extra_scale))
        elif cfg.gap_extra == "none":
            extra = 0.0
        else:
            raise ValueError(f"unknown gap_extra {cfg.gap_extra!r}")
        t = t + cfg.min_gap + extra
    v0 = _draw(rng, cfg.belt_speed)
    last = times[-1] if times else 0.0
    ride = 2.6                      # m, entry to past the exit line
    horizon = duration if duration is not None else last + ride / max(v0, 0.05)
    segs = [(0.0, v0)]
    if isinstance(cfg.speed_changes, tuple):
        lo, hi = (int(v) for v in cfg.speed_changes)
        n_ch = int(rng.integers(lo, hi + 1))
    else:
        n_ch = int(cfg.speed_changes)
    if n_ch > 0:
        ts = np.sort(rng.uniform(0.0, horizon, n_ch))
        for tc in ts:
            segs.append((float(tc), _draw(rng, cfg.belt_speed)))
    if duration is None:
        # long enough for the last parcel to ride from the entry past the exit
        vmin = min(v for _, v in segs)
        duration = last + ride / max(vmin, 0.05) + 1.0
    arrivals = tuple(Arrival(t=ti, parcel=p) for ti, p in zip(times, parcels))
    return Shift(belt_speed=tuple(segs), arrivals=arrivals,
                 duration=float(duration), seed=seed)
