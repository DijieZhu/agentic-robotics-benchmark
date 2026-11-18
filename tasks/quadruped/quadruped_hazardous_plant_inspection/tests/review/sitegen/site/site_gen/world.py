"""The world being built: primitives, traversal elements, equipment, instrument slots, lights, textures to make.

Traversal element classes: free, needs_high_step, needs_crouch, tight, slow, barrier. Every element
records its kind, its parameter (height, diameter, clearance, width, slope, friction), its class and its geoms.
Sizes are drawn inside the class band with a margin to the band's edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .prims import PrimStore
from .util import poly_rect, uniform

# element kind -> capability band family
FAMILY = {"sill": "step", "hose": "step", "cable_protector": "step", "bund": "step", "debris": "step",
          "trench_cover": "step",          # version 2: a plate strip 0.20 m deep, the bund kerb's section
          "floor_pipe": "floor_pipe", "low_overhead": "overhead", "narrow_gap": "width", "ramp": "slope_deg",
          "puddle": "friction", "door": "door"}
CLASSES = ("free", "needs_high_step", "needs_crouch", "tight", "slow", "barrier")


def band_interval(cfg, family: str, cls: str) -> tuple:
    """The interval a size of `cls` is drawn from (margin applied)."""
    c = cfg["capability"]
    m = float(c.get("band_margin", 0.15))
    if family in ("step", "floor_pipe"):
        b = c[family]
        if cls == "free":
            lo, hi = 0.012, float(b["free_max"])
            return (lo, hi - m * (hi - lo))
        if cls == "needs_high_step":
            lo, hi = float(b["needs_high_step"][0]), float(b["needs_high_step"][1])
            return (lo + m * (hi - lo), hi - m * (hi - lo))
        lo = float(b["barrier_min"])
        return (lo * (1 + m), lo * 1.8)
    if family == "overhead":
        b = c["overhead"]
        if cls == "free":
            lo = float(b["free_min"])
            return (lo + 0.1, lo + 0.6)
        if cls == "needs_crouch":
            lo, hi = float(b["needs_crouch"][0]), float(b["needs_crouch"][1])
            return (lo + m * (hi - lo), hi - m * (hi - lo))
        hi = float(b["barrier_max"])
        return (0.10, hi * (1 - m))
    if family == "width":
        b = c["width"]
        if cls == "free":
            lo = float(b["free_min"])
            return (lo + 0.08, lo + 0.5)
        if cls == "tight":
            lo, hi = float(b["tight"][0]), float(b["tight"][1])
            return (lo + m * (hi - lo), hi - m * (hi - lo))
        hi = float(b["barrier_max"])
        return (0.15, hi * (1 - m))
    if family == "slope_deg":
        b = c["slope_deg"]
        if cls == "free":
            hi = float(b["free_max"])
            return (3.0, hi * (1 - m))
        if cls == "slow":
            lo, hi = float(b["slow"][0]), float(b["slow"][1])
            return (lo + m * (hi - lo), hi - m * (hi - lo))
        lo = float(b["barrier_min"])
        return (lo * (1 + m), 45.0)
    if family == "friction":
        hi = float(c["friction"]["slow_max"])
        return (0.12, hi * (1 - m))
    raise KeyError(family)


def _ladder_cfg(cfg):
    if str(cfg.get("layout", "draft")).startswith("ladder"):
        return cfg.get("ladder", {})
    return None


def draw_interval(cfg, kind: str, cls: str) -> tuple:
    """The interval a size of `kind` in class `cls` is drawn from. Ladder mode: a needs_high_step element is drawn
    where walk is refused: capability.json's "draw" block when the gait's measurement has written one,
    else ladder.high_step_draw (sills, hoses, kerbs 0.048-0.055 m; round floor pipes 0.045-0.06 m); then
    ladder.high_step_band (a part of that range). Puddles: ladder.puddle_friction."""
    lo, hi = band_interval(cfg, FAMILY[kind], cls)
    L = _ladder_cfg(cfg)
    if L is not None and cls == "needs_high_step":
        draw = ((cfg["capability"].get("draw") or {}).get(kind) or {}).get("needs_high_step")
        if draw is None:
            draw = (L.get("high_step_draw") or {}).get(kind)
        if draw is not None:
            lo, hi = float(draw[0]), float(draw[1])
        f0, f1 = (float(v) for v in L.get("high_step_band", [0.0, 1.0]))
        lo, hi = lo + f0 * (hi - lo), lo + f1 * (hi - lo)
    if L is not None and cls == "needs_crouch":
        draw = ((cfg["capability"].get("draw") or {}).get(kind) or {}).get("needs_crouch")
        if draw is None:
            draw = (L.get("crouch_draw") or {}).get(kind)
        if draw is not None:
            lo, hi = float(draw[0]), float(draw[1])
    if L is not None and kind == "puddle" and L.get("puddle_friction"):
        lo, hi = (float(v) for v in L["puddle_friction"])
    return lo, hi


def sample_size(cfg, g, kind: str, cls: str) -> float:
    lo, hi = draw_interval(cfg, kind, cls)
    return float(round(uniform(g, lo, hi), 4))


def restrict_class(cfg, kind: str, cls: str) -> str:
    """Ladder mode: a needs_high_step element may only be of a kind in ladder.high_step_kinds (the classical controller
    is weak on round floor pipes); any other kind is drawn free instead (a bund kerb becomes a barrier: the ramp is
    then the only way in)."""
    L = _ladder_cfg(cfg)
    if L is None or cls != "needs_high_step":
        return cls
    if kind in [str(k) for k in L.get("high_step_kinds", [])]:
        return cls
    return "barrier" if kind == "bund" else "free"


@dataclass
class Element:
    id: str
    kind: str
    cls: str
    param: dict
    room: str
    footprint: list
    geoms: list = field(default_factory=list)
    zone: bool = False         # the footprint itself carries the class for the map (debris, gap zones)
    note: str = ""

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "class": self.cls, "param": self.param, "room": self.room,
                "footprint": self.footprint, "geoms": self.geoms, "zone": self.zone, "note": self.note}


class World:
    def __init__(self, cfg, lay, rng):
        self.cfg = cfg
        self.lay = lay
        self.rng = rng
        self.P = PrimStore(float(cfg["building"]["overhead_decor_min_z"]))
        self.elements: dict[str, Element] = {}
        self._next_eid = 1                     # element ids are never reused (records can be dropped on a rebuild)
        self.equipment: list[dict] = []
        self.slots: list[dict] = []
        self.lights: list[dict] = []
        self.signs: dict[str, dict] = {}       # material name -> sign texture request
        self.markers: list[dict] = []          # grid marker candidates
        self.decals: list[dict] = []
        self.thermal_groups: list[dict] = []
        self.shields: list[dict] = []          # radiation shield boxes (beyond the walls)
        self.sources_candidates: list[dict] = []
        self.floor_markers: list[dict] = []    # painted survey points (dose points)
        self.repeaters: list[dict] = []
        self.base_station: dict = {}
        self.notes: list[str] = []
        self.keepouts: dict[str, list] = {}    # room id -> world rects kept free (door approaches)
        self.views: list[dict] = []            # recommended eye-level views registered by the rules
        self.ladder = str(cfg.get("layout", "draft")).startswith("ladder")
        self.spots: list[dict] = []            # ladder: free wall spans where an instrument station can be built
        self.ctx: dict = {}                    # room id -> its furnishing context (occupancy, local frame)

    # ------------------------------------------------------------------ elements
    def element(self, kind, cls, param, room, footprint, zone=False, note="") -> Element:
        eid = "E%03d" % self._next_eid
        self._next_eid += 1
        e = Element(id=eid, kind=kind, cls=cls, param=dict(param), room=room,
                    footprint=[[float(x), float(y)] for x, y in footprint], zone=zone, note=note)
        self.elements[eid] = e
        return e

    def tag(self, e: Element, prims):
        for p in prims:
            if p is None:
                continue
            p.elem = e.id
            p.cls = e.cls
            if p.group == 0:
                p.group = 3
            e.geoms.append(p.name)

    def equip(self, kind, label, room, rect_world, height, on_drawing=True, eid=None, body=None, shape="rect",
              center=None, radius=None) -> dict:
        eid = eid or "EQ%03d" % (len(self.equipment) + 1)
        d = {"id": eid, "kind": kind, "label": label, "room": room, "shape": shape,
             "footprint": poly_rect(rect_world) if shape == "rect" else None, "center": center, "radius": radius,
             "height": float(height), "on_drawing": bool(on_drawing), "body": body or ("eq/%s" % eid)}
        self.equipment.append(d)
        return d

    def slot(self, typ, pos, yaw, room, mount, service, owner, pitch=0.0, **extra) -> dict:
        s = {"id": "SL%03d" % (len(self.slots) + 1), "type": typ, "pos": [float(v) for v in pos],
             "yaw": float(yaw), "pitch": float(pitch), "room": room, "mount": mount, "service": service,
             "owner": owner}
        s.update(extra)
        self.slots.append(s)
        return s

    def elem_class(self, g, kind, weights) -> str:
        """Draw an element class (one draw from g, as before) and apply the ladder's kind restriction."""
        from .util import weighted_key
        return restrict_class(self.cfg, kind, weighted_key(g, weights))

    def spot(self, c, wall, s0, s1, zone, orients=("out",), depth=0.5, room=None):
        """Register a ladder mount spot: the stretch [s0, s1] of local wall `wall` of the room of context c."""
        if s1 - s0 < 0.05:
            return None
        sp = {"id": "SP%03d" % (len(self.spots) + 1), "room": room or c.rid, "wall": wall, "s": [float(s0), float(s1)],
              "zone": zone, "orients": list(orients), "depth": float(depth)}
        self.spots.append(sp)
        return sp

    def light(self, pos, level, room, kind="spot"):
        self.lights.append({"pos": [float(v) for v in pos], "level": level, "room": room, "kind": kind})

    def sign(self, name, text, style="room", size=(512, 128)) -> str:
        mat = "sign/" + name
        self.signs[mat] = {"text": text, "style": style, "size": list(size)}
        return mat


def ajar_angle_for_gap(width: float, thick: float, wall_t: float, gap: float) -> float:
    """Leaf angle (deg, 0 = closed) so the narrowest gap between the leaf and the far jamb is `gap`."""
    def g_of(theta_deg):
        th = math.radians(theta_deg)
        # leaf segment from the hinge (0, 0) to the tip; the far jamb corner at (width, wall_t / 2)
        tip = np.array([width * math.cos(th), width * math.sin(th)])
        c = np.array([width, wall_t / 2])
        d = tip / max(np.linalg.norm(tip), 1e-9)
        t = float(np.clip(np.dot(c, d), 0, width))
        dist = float(np.linalg.norm(c - t * d)) - thick / 2
        return dist
    lo, hi = 1.0, 89.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g_of(mid) < gap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
