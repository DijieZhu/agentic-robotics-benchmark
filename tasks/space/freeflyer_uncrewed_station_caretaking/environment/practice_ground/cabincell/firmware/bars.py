"""Bars and the firmware's geometric bar detector of the aft depth camera (robot_api.md, capture; no pixels).

``Bar``, ``_Track``, ``BarDetector``, in the task firmware's form:

* a measurement is taken relative to the robot (body frame: the bar centre, its axis and its face normal, white noise
  firmware.capture.detector_noise per measurement) and mapped into the station frame with the firmware's own pose
  estimate, so the gate, the lock and the approach all live in the estimate's frame (the program's hint is given in the
  body frame and mapped the same way);
* the face normal is measured too (the same random rotation as the axis), so the approach direction carries no truth;
* the crew handrails of the map are candidates as well (``StaticBar``; closing depth of a map handrail), not only the
  item handles; candidates out of range are skipped before the ray tests (101 handrails).

A bar is a candidate when at least ``min_visible`` of its sample points on the face toward the camera are inside the
camera's field of view, between the detector range, seen from the front (incidence <= ``max_incidence_deg``), not
end-on (ray-to-axis angle >= ``min_axis_view_deg``) and not occluded (ray cast; the robot's rigid structure and
display-only geometry are transparent, its arm and gripper are not).  Every candidate gives one measurement per cycle
(firmware.capture.detector_hz).  numpy + mujoco only.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from spacekit.rotations import quat_from_rotvec, quat_to_mat

DETECTOR = dict(range_m=(0.125, 1.2), max_incidence_deg=75.0, min_axis_view_deg=30.0, n_samples=5, min_visible=3)
RAIL_FACE = 0.035          # handrail section depth along the face normal: sample points on the front face [m]


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def line_angle_deg(a, b) -> float:
    """Angle between two lines (axis sign ignored), in [0, 90] deg."""
    c = abs(float(np.dot(unit(a), unit(b))))
    return math.degrees(math.acos(min(1.0, c)))


def perp_unit(axis, rng: np.random.Generator):
    a = unit(axis)
    while True:
        u = rng.normal(size=3)
        u -= (u @ a) * a
        n = np.linalg.norm(u)
        if n > 1e-6:
            return u / n


def rotate(v, rotvec):
    return quat_to_mat(quat_from_rotvec(np.asarray(rotvec, float))) @ np.asarray(v, float)


# ----------------------------------------------------------------------------------------------- bars
class Bar:
    """A handrail-section handle of an item: centre = a site; axis / face normal fixed in the site frame.
    kind: 'strap' | 'loop' | 'handle' (item bars, closing depth of an item bar) or 'rail' (map handrail)."""

    depth_rule = "item"

    def __init__(self, model, name: str, kind: str, site: str, geom_prefix: str, length: float, axis_local=(1, 0, 0),
                 normal_local=(0, 0, 1)):
        self.name, self.kind, self.length = name, kind, float(length)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise KeyError(site)
        self.body = int(model.site_bodyid[self.site])
        self.axis_local = unit(axis_local)
        self.normal_local = unit(normal_local)
        self.geoms = {g for g in range(model.ngeom)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(geom_prefix)}

    def pose(self, data):
        """(centre, axis, face normal) in the station frame (truth)."""
        R = data.site_xmat[self.site].reshape(3, 3)
        return data.site_xpos[self.site].copy(), R @ self.axis_local, R @ self.normal_local

    def sample_points(self, data, n: int):
        c, a, nrm = self.pose(data)
        s = np.linspace(-0.4, 0.4, n) * self.length
        return c[None, :] + nrm[None, :] * (RAIL_FACE / 2) + s[:, None] * a[None, :]


class StaticBar(Bar):
    """A crew handrail of the map (fixed pose; feature record ``pos``, ``quat_wxyz`` (x = bar axis, z = away from the
    wall), ``length``); its geoms are the bar and its feet (names starting with the handrail id)."""

    depth_rule = "rail"

    def __init__(self, model, rec: dict):
        self.name, self.kind, self.length = str(rec["id"]), "rail", float(rec["length"])
        R = quat_to_mat(np.asarray(rec["quat_wxyz"], float))
        self._c, self._a, self._n = np.asarray(rec["pos"], float), R[:, 0].copy(), R[:, 2].copy()
        self.site, self.body = -1, 0
        nm = self.name
        self.geoms = {g for g in range(model.ngeom)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") in (nm,) or
                      (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(nm + "_")}

    def pose(self, data):
        return self._c.copy(), self._a.copy(), self._n.copy()


class _Track:
    """Measurements of one bar since the capture call (estimate frame)."""

    def __init__(self):
        self.t, self.c, self.a, self.n = [], [], [], []
        self.t_last = None

    def add(self, t, c, a, n):
        a = unit(a)
        if self.a and float(np.dot(a, self.a[-1])) < 0:
            a = -a
        self.t.append(float(t))
        self.c.append(np.asarray(c, float))
        self.a.append(a)
        self.n.append(unit(n))
        self.t_last = float(t)

    def estimate(self):
        """(centre, axis, face normal): means of every measurement (a fixed bar)."""
        A = unit(np.sum(self.a[-max(1, min(len(self.a), 10)):], axis=0))
        N = unit(np.sum(self.n[-max(1, min(len(self.n), 10)):], axis=0))
        return np.mean(self.c, axis=0), A, N

    def mean_last(self, k):
        return np.mean(self.c[-k:], axis=0), unit(np.sum(self.a[-k:], axis=0))


class BarDetector:
    """Geometric bar detector of the aft depth camera (module doc).  ``sigma_pos_m`` per axis (body frame) and
    ``sigma_axis_deg`` (rotation about a random axis) per measurement."""

    def __init__(self, model, bars, robot_body: int, rng: np.random.Generator, sigma_pos_m: float, sigma_axis_deg: float,
                 cam: str = "cam_aft_depth", **kw):
        p = dict(DETECTOR, **kw)
        self.p = p
        self.model = model
        self.bars = list(bars)
        self.rng = rng
        self.sigma_pos, self.sigma_axis = float(sigma_pos_m), math.radians(float(sigma_axis_deg))
        self.robot_body = int(robot_body)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cid < 0:
            raise KeyError(cam)
        d = mujoco.MjData(model)
        mujoco.mj_kinematics(model, d)
        mujoco.mj_camlight(model, d)
        Rb = d.xmat[robot_body].reshape(3, 3)
        self.cam_off = Rb.T @ (d.cam_xpos[cid] - d.xpos[robot_body])
        self.R_body_cam = Rb.T @ d.cam_xmat[cid].reshape(3, 3)
        fovy = math.radians(float(model.cam_fovy[cid]))
        W, H = (int(x) for x in model.cam_resolution[cid])
        self.tan_v = math.tan(fovy / 2)
        self.tan_h = self.tan_v * W / H
        weld = int(model.body_weldid[robot_body])
        self.robot_rigid = {b for b in range(model.nbody) if int(model.body_weldid[b]) == weld}
        self.tracks = {}
        self.last = []
        self.n_updates = 0
        self._gid = np.array([-1], dtype=np.int32)

    def reset(self):
        self.tracks = {}

    def camera(self, data):
        b = self.robot_body
        Rb = data.xmat[b].reshape(3, 3)
        return data.xpos[b] + Rb @ self.cam_off, Rb @ self.R_body_cam

    def _ray_clear(self, data, origin, target, bar: Bar) -> bool:
        v = target - origin
        dist = float(np.linalg.norm(v))
        u = v / dist
        o = origin.copy()
        travelled = 0.0
        for _ in range(8):
            h = mujoco.mj_ray(self.model, data, o, u, None, 1, -1, self._gid)
            if h < 0 or travelled + h >= dist - 0.004:
                return True
            g = int(self._gid[0])
            if g in bar.geoms:
                return True
            if int(self.model.geom_bodyid[g]) in self.robot_rigid or not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                o = o + u * (h + 1e-4)
                travelled += h + 1e-4
                continue
            return False
        return False

    def visible(self, data, bar: Bar, cam=None):
        """(visible, number of visible sample points, nearest distance) of one bar at the current state."""
        p = self.p
        cpos, Rc = cam if cam is not None else self.camera(data)
        c, a, nrm = bar.pose(data)
        if np.linalg.norm(c - cpos) > p["range_m"][1] + bar.length:
            return False, 0, None
        pts = bar.sample_points(data, p["n_samples"])
        ok = 0
        dmin = None
        cos_inc = math.cos(math.radians(p["max_incidence_deg"]))
        sin_ax = math.sin(math.radians(p["min_axis_view_deg"]))
        for q in pts:
            v = q - cpos
            dist = float(np.linalg.norm(v))
            if not (p["range_m"][0] <= dist <= p["range_m"][1]):
                continue
            vc = Rc.T @ v
            if vc[2] >= -1e-6:
                continue
            if abs(vc[0] / -vc[2]) > self.tan_h or abs(vc[1] / -vc[2]) > self.tan_v:
                continue
            u = v / dist
            if float(-u @ nrm) < cos_inc:
                continue
            if math.sqrt(max(0.0, 1.0 - float(u @ unit(a)) ** 2)) < sin_ax:
                continue
            if not self._ray_clear(data, cpos, q, bar):
                continue
            ok += 1
            dmin = dist if dmin is None else min(dmin, dist)
        return ok >= p["min_visible"], ok, dmin

    def update(self, data, t: float, est_pos, est_quat):
        """One detector cycle: measure every visible bar (relative to the robot), map it with the estimate
        (est_pos, est_quat), update its track; returns the list of measurements."""
        self.n_updates += 1
        b = self.robot_body
        Rb = data.xmat[b].reshape(3, 3)
        pb = data.xpos[b]
        Re = quat_to_mat(np.asarray(est_quat, float))
        pe = np.asarray(est_pos, float)
        cam = self.camera(data)
        out = []
        for i, bar in enumerate(self.bars):
            vis, n, dist = self.visible(data, bar, cam)
            if not vis:
                continue
            c, a, nrm = bar.pose(data)
            rel = Rb.T @ (c - pb) + self.rng.normal(0.0, self.sigma_pos, 3)
            k = perp_unit(a, self.rng) * self.rng.normal(0.0, self.sigma_axis)
            a_b = Rb.T @ rotate(a, k)
            n_b = Rb.T @ rotate(nrm, k)
            c_m, a_m, n_m = pe + Re @ rel, Re @ a_b, Re @ n_b
            tr = self.tracks.setdefault(i, _Track())
            tr.add(t, c_m, a_m, n_m)
            out.append(dict(bar=i, c=c_m, a=a_m, dist=dist, n=n))
        self.last = out
        return out

    def track(self, i: int):
        tr = self.tracks.get(i)
        return None if tr is None else tr.estimate()


def item_bars(model, info: dict) -> list:
    """Bars of the items of a built site (station_site/2 info): the blocker's carry strap and handle loop, every small
    item's handle (sites and geoms named by the station builder, prefix 'fx_')."""
    out = []
    B = info.get("blocker")
    if B:
        site = info.get("site", {}).get("blocker") or {}
        L = float(2 * B["half"][0] - 0.016)
        if site.get("strap", True):
            out.append(Bar(model, "blocker:strap", "strap", B["sites"]["strap"], "fx_blk_hc", L))
        if site.get("loop", "edge_mid"):
            out.append(Bar(model, "blocker:loop", "loop", B["sites"]["loop"], "fx_blk_hk", L))
    for s in info.get("small_items", []):
        out.append(Bar(model, f"{s['id']}:handle", "handle", s["site"], f"fx_{s['id']}_handle", 0.07))
    return out


def map_bars(model, feats: dict) -> list:
    """The crew handrails of the station catalogue as static bars."""
    return [StaticBar(model, h) for h in feats.get("handrails", [])]
