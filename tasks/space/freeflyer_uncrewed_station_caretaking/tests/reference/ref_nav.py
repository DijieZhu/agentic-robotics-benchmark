"""ref_nav.py -- the reference program's map: everything here is what the operator has (the as-surveyed station map
and the manuals), nothing is a hidden quantity.

* the station catalogue of the mission scene (``station.data.features('mission')``, the cell's copy of the map the
  agent's ~/given/station_map/ is derived from): module axes, the return inlets (face centre, normal, long axis,
  label RI-<M>-<n>), the dock and its berths;
* the stowage units at their REGISTERED sites (runtime site_spec ``world.stowage_unit.registered`` -> the unit site of
  the station catalogue ``station_v03.json``; the pouch-pad plate side ``world.stowage_unit.pad_plate.selected``):
  face centres and axes, the plate frames (SB-<M>-F1 screen +b, -F2 screen -b, -F3 pad P1, -F4 pad P2) -- a moved
  unit is found through its plates (fly_rel), never read;
* the firmware's zone rule on the as-surveyed zones (``firmware.zonecheck.ZoneRules``: stowed 0.30 m / arm-deployed
  0.50 m inflation and the band rule, the robot manual's rule), used to PREDICT which goals the firmware will accept
  from the robot's own pose estimate; every move is still a validated fly_to / fly_rel of the firmware;
* route geometry: survey stops, corridor legs through the hatches, the approach start of a berth.

The pieces: ``look_quat`` (an attitude from a viewing direction), ``StationMap.survey_pose`` (the survey stop of
an inlet), ``StationMap.transit_legs`` (the corridor legs through the hatches, by the hatch points below),
``StationMap.approach_start`` (the start of a dock(berth) call, for both berths), ``StationMap.centreline_point``
(the point of the module axis at an axial position), ``StationMap.unit_part``, ``face`` and ``plate_frame`` (the
stowage units at their registered sites) and ``ZonePredictor`` (the firmware's zone rule).
The centreline point and the trapezoid check are those of the site generator's reach check
(tests/review/sitegen/gen/pregrasp.py).  The map is the station v0.3 catalogue: the centreline height is the
module axis'.  Every move of the route is still a validated fly_to / fly_rel of the task firmware, under its zone
rule.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from spacekit.rotations import mat_to_quat, quat_to_mat

MODULE_ORDER = ("l2", "h2", "l1")
HATCH_L2_H2 = (10.936, -2.75, 4.851)       # L2 side of the L2 / H2 hatch
H2_HUB = (10.936, -0.011, 4.836)
H2_L1 = (6.0, 0.0, 4.835)                  # L1 side of the H2 / L1 hatch
HOME0 = (10.936, -9.0, 4.851)              # L2 centreline in front of the dock
APPROACH_START_X_M = 1.0                   # the approach start: 1.0 m out of the berth along its axis
APPROACH_START_DZ_M = 0.104                # ... and this far toward the module axis (z 4.40 for berth 1)


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def look_quat(direction, down=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Attitude with body +x along ``direction`` and body +z as close as possible to ``down``
    (flight frame: +z deck)."""
    x = unit(direction)
    z = np.asarray(down, float) - x * float(np.dot(down, x))
    if np.linalg.norm(z) < 1e-6:
        z = np.array([1.0, 0.0, 0.0]) - x * x[0]
    z = unit(z)
    y = np.cross(z, x)
    return np.asarray(mat_to_quat(np.column_stack([x, y, z])), float)


def quat_of(R) -> np.ndarray:
    return np.asarray(mat_to_quat(np.asarray(R, float)), float)


def rot_of(q) -> np.ndarray:
    return np.asarray(quat_to_mat(np.asarray(q, float)), float)


def angle_deg(qa, qb) -> float:
    Ra, Rb = rot_of(qa), rot_of(qb)
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ------------------------------------------------------------------------------------------------ the map
class StationMap:
    """The as-surveyed map of the mission scene (module doc).  ``runtime_values``: the cell's runtime site_spec
    values (the registered unit sites and the pad-plate side are map facts)."""

    def __init__(self, runtime_values: dict):
        from station import data as SD
        self.feats = SD.features("mission")
        self.zones_raw = SD.zones("mission")
        self.V = runtime_values
        self.inlets = {i["id"]: i for i in self.feats["v02"]["inlets"]}
        self.labels = {i["id"]: i["label"] for i in self.feats["v02"]["inlets"]}
        self.dock = self.feats["dock"]
        self._units = {}

    # -- modules
    def module(self, m: str) -> dict:
        return self.feats["modules"][m]

    def axes(self, m: str):
        M = self.module(m)
        return (np.asarray(M["axis_origin"], float), unit(M["axis_dir"]), np.asarray(M["lat_u"], float),
                np.asarray(M["lat_w"], float))

    def axial_s(self, m: str, p) -> float:
        o, ax, _, _ = self.axes(m)
        return float((np.asarray(p, float) - o) @ ax)

    def centreline_point(self, m: str, p, margin: float = 0.8) -> np.ndarray:
        """The module axis point at p's axial position, clipped to the module's end walls less ``margin``."""
        o, ax, _, _ = self.axes(m)
        M = self.module(m)
        lo, hi = sorted((M["end_walls_s"]["start"] + margin, M["end_walls_s"]["end"] - margin))
        s = float(np.clip(self.axial_s(m, p), lo, hi))
        return o + ax * s

    def module_of_point(self, p) -> str:
        best, bd = None, 1e9
        for m in MODULE_ORDER:
            M = self.module(m)
            o, ax, _, _ = self.axes(m)
            s = float((np.asarray(p, float) - o) @ ax)
            lo, hi = sorted((M["end_walls_s"]["start"], M["end_walls_s"]["end"]))
            ds = 0.0 if lo <= s <= hi else min(abs(s - lo), abs(s - hi))
            r = float(np.linalg.norm((np.asarray(p, float) - o) - ax * s))
            d = ds + 0.1 * r
            if d < bd:
                best, bd = m, d
        return best

    # -- inlets
    def inlet(self, iid: str) -> dict:
        return self.inlets[iid]

    def inlet_frame(self, iid: str):
        I = self.inlets[iid]
        return np.asarray(I["face_center"], float), unit(I["normal"]), unit(I["t1"])

    def inlets_of(self, m: str) -> list:
        return [i for i, I in self.inlets.items() if I["module"] == m]

    def survey_pose(self, iid: str):
        """The survey stop of an inlet: the module centreline at the inlet's axial position, body +x (the forward
        imaging camera) toward the inlet's face centre; (pos, quat, distance)."""
        c, _, _ = self.inlet_frame(iid)
        m = self.inlets[iid]["module"]
        p = self.centreline_point(m, c, margin=0.5)
        return p, look_quat(c - p), float(np.linalg.norm(c - p))

    def nearest_inlet(self, p, max_m: float = 0.25) -> str | None:
        """The inlet whose face a point lies on (within max_m of the face centre, in front of the face)."""
        best, bd = None, max_m
        for iid in self.inlets:
            c, n, a = self.inlet_frame(iid)
            d = np.asarray(p, float) - c
            if float(d @ n) < -0.02:
                continue
            dist = float(np.linalg.norm(d))
            if dist < bd:
                best, bd = iid, dist
        return best

    # -- transits
    def transit_legs(self, a: str, b: str) -> list:
        if a == b:
            return []
        qx_neg, qx_pos = look_quat((-1.0, 0.0, 0.0)), look_quat((1.0, 0.0, 0.0))
        qy_pos, qy_neg = look_quat((0.0, 1.0, 0.0)), look_quat((0.0, -1.0, 0.0))
        L = []
        if (a, b) in (("l2", "h2"), ("l2", "l1")):
            L += [(HATCH_L2_H2, qy_pos), (H2_HUB, qx_neg)]
            if b == "l1":
                L += [(H2_L1, qx_neg)]
        elif (a, b) == ("h2", "l1"):
            L += [(H2_L1, qx_neg)]
        elif (a, b) in (("l1", "l2"), ("l1", "h2")):
            L += [(H2_L1, qx_pos), (H2_HUB, qy_neg)]
            if b == "l2":
                L += [(HATCH_L2_H2, qy_neg)]
        elif (a, b) == ("h2", "l2"):
            L += [(H2_HUB, qy_neg), (HATCH_L2_H2, qy_neg)]
        return [(np.asarray(p, float), np.asarray(q, float)) for p, q in L]

    # -- dock
    def berth_pose(self, berth: int):
        B = self.dock[f"berth{int(berth)}_body_pose"]
        return np.asarray(B["pos"], float), np.asarray(B["quat_wxyz"], float)

    def approach_start(self, berth: int):
        """(pos, quat) of the start of a dock(berth) call: on the berth axis APPROACH_START_X_M out of the berth's
        body pose, raised toward the module axis, at the berth attitude."""
        p, q = self.berth_pose(berth)
        R = rot_of(self.dock["quat_wxyz"])
        x = R[:, 0]
        o, ax, _, w = self.axes("l2")
        up = o - p
        up = up - (up @ x) * x
        up = unit(up - (up @ ax) * ax) if np.linalg.norm(up - (up @ ax) * ax) > 1e-6 else np.zeros(3)
        return p + x * APPROACH_START_X_M + up * APPROACH_START_DZ_M, q

    # -- stowage units at their registered sites
    def unit_part(self, m: str):
        """The registered stowage unit of module m (station.stowage.UnitPart at the registered unit site)."""
        if m in self._units:
            return self._units[m]
        from station import data as SD
        from station import stowage as ST
        reg = self.V["world.stowage_unit.registered"][m.upper()]
        side = self.V["world.stowage_unit.pad_plate"]["selected"]
        U = ST.unit_from_site(f"u_{m}", SD.unit_site(reg["unit_site"]), pad_fid_side=side)
        self._units[m] = U
        return U

    def face(self, m: str, which: str):
        """A face of the registered unit: 'S' screen, 'P1' pad at -a, 'P2' pad at +a."""
        return self.unit_part(m).face({"S": "hs", "P1": "pp0", "P2": "pp1"}[which])

    def plate_frame(self, plate_id: str):
        """(centre, R columns x = plate long axis, y = z cross x, z = normal into the cabin) of a plate of the
        registered unit (the firmware's plate frame of fly_rel: firmware.stack.plate_record)."""
        _, mod, fid = plate_id.split("-")
        U = self.unit_part(mod.lower())
        which = {"F1": "hs+b", "F2": "hs-b", "F3": "pp0", "F4": "pp1"}[fid]
        pts, n = U.fiducial_points(which)
        t1, _ = U.fiducial_axes(which)
        n = unit(n)
        x = unit(np.asarray(t1, float) - (np.asarray(t1, float) @ n) * n)
        return np.asarray(pts, float).mean(axis=0), np.column_stack([x, np.cross(n, x), n])

    def to_plate(self, plate_id: str, pos_w, quat_w):
        """A world pose (registered unit) in the plate's frame: what fly_rel takes."""
        c, R = self.plate_frame(plate_id)
        p = R.T @ (np.asarray(pos_w, float) - c)
        q = quat_of(R.T @ rot_of(quat_w))
        return [float(v) for v in p], [float(v) for v in q]


# ------------------------------------------------------------------------------------------------ zone prediction
class ZonePredictor:
    """The firmware's zone rule on the as-surveyed zones, evaluated by the program from its own estimate (module doc)."""

    def __init__(self, zones):
        from firmware.spec import default_spec
        from firmware.zonecheck import ZoneRules
        from spacekit.control import load_flight_modes
        self.rules = ZoneRules(zones, default_spec(), {})
        self.mode = load_flight_modes()["standard"]

    def segment(self, p0, q0, p1, q1, arm: bool) -> str:
        """'ok' or the refusal of a straight fly_to from (p0, q0) to (p1, q1) (band rule on)."""
        from spacekit.control import plan_trapezoid
        seg = plan_trapezoid([(np.asarray(p0, float), np.asarray(q0, float)), (np.asarray(p1, float), np.asarray(q1, float))],
                             self.mode, False)
        if not seg:
            return "ok"
        r, _k = self.rules.map.check_segment(seg, band_exit=True, arm=bool(arm))
        return r

    def point(self, p, arm: bool) -> str:
        return self.rules.point(np.asarray(p, float), arm=bool(arm))

    def envelope_ok(self, p) -> bool:
        return self.rules.check_envelope(np.asarray(p, float)) == "ok"


@functools.lru_cache(maxsize=4)
def _zone_predictor_cached(key: str):
    from station import data as SD
    return ZonePredictor(SD.zones(key))


def zone_predictor() -> ZonePredictor:
    return _zone_predictor_cached("mission")
