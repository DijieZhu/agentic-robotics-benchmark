"""Siting of a stowage unit on a rack face: the dense ray check, the corridor and inlet-distance rules, and the air
pre-screen in front of the holding screen.

``unit_site_check``: the footprint is the unit with its pouch pads and every pad-plate candidate
(``stowage.UnitPart.outline``); the rules:
  mountable   every ray hit inside the footprint within +-15 mm of the footprint's own (median) plane, i.e. the unit's
              back sits flat (2 cm grid: a 19 mm handrail is not missed); the plane is the local one -- on a stepped rack face (the practice module) the unit is seated on
              the panel it covers (``seat``) instead of being judged against the candidate's plane
  clear       nothing more than 20 mm proud of that plane within 0.20 m of the holding screen's frame
  corridor    the unit's volume outside the +-0.45 m tube about the module axis
  inlets      the screen centre >= 0.6 m from every return inlet
The criterion of a position is the robot-free release test; the air in front of the screen
(normal component toward the face <= 1 cm/s at 0.15-0.30 m) is only a pre-screen; the tangential air along the rack
face is recorded.
Unit site records: id, module, kind ('registered' | 'moved' | 'practice'), base_site (a rack-face candidate of the
catalogue), shift_a_m / shift_b_m (along the unit's a / b), wall_point, normal, up, a, b.
"""
from __future__ import annotations

import numpy as np

from . import stowage as SW
from .util import obb_corners, unit_axes

UP = (0.0, 0.0, 1.0)            # the modules' overhead axis in the station frame: the units' +b side
RULES = dict(step_m=0.02, probe_m=0.35, mount_tol_m=0.015, ring_m=0.20, ring_proud_m=0.020, corridor_half_m=0.45,
             inlet_min_m=0.6, prescreen_vn_toward_max_ms=0.01)
FLOW_DISTS = (0.05, 0.15, 0.30)


def site_record(feats: dict, base_site: str, shift_a: float = 0.0, shift_b: float = 0.0, kind: str = "registered",
                site_id: str | None = None, up=None) -> dict:
    """A unit site record from a rack-face candidate of the catalogue (v02 receptacle_sites) and a shift.  up: the
    unit's +b direction (default: the overhead axis on a lateral rack face; the module axis on a deck / overhead face)."""
    r = next(x for x in feats["v02"]["receptacle_sites"] if x["id"] == base_site)
    if up is None:
        up = UP
        if abs(float(np.dot(np.asarray(r["normal"], float), UP))) > 0.9:
            up = tuple(float(v) for v in feats["modules"][r["module"]]["axis_dir"])
    n, a, b = unit_axes(r["normal"], up)
    wp = np.asarray(r["wall_point"], float) + a * float(shift_a) + b * float(shift_b)
    sid = site_id or f"{base_site}_{shift_a:+.2f}_{shift_b:+.2f}"
    return dict(id=sid, module=r["module"], kind=kind, base_site=base_site, face=r["face"], shift_a_m=float(shift_a),
                shift_b_m=float(shift_b), wall_point=wp.tolist(), normal=n.tolist(), up=list(up), a=a.tolist(), b=b.tolist())


def seat(rec: dict, plane_offset: float) -> dict:
    """The record with its wall point moved along the normal onto the footprint's own plane (dense_check's
    ``plane_offset_m``)."""
    r = dict(rec)
    r["wall_point"] = (np.asarray(rec["wall_point"], float) + np.asarray(rec["normal"], float) * float(plane_offset)).tolist()
    r["seated_by_m"] = float(rec.get("seated_by_m", 0.0)) + float(plane_offset)
    return r


def module_of(feats: dict, p):
    from .scene import module_of_point
    return module_of_point(feats, p)


def dense_check(model, data_, feats: dict, rec: dict, rules: dict | None = None, outline_sides=SW.SITING_PAD_SIDES) -> dict:
    """Dense site check of a unit at ``rec`` on a scene built WITHOUT units, robot or items (station walls, handrails,
    inlet parts): rays from ``probe_m`` in front of the rack face toward it on a ``step_m`` grid over the footprint
    (pad plates of ``outline_sides``) grown by ``ring_m``."""
    import mujoco
    P = dict(RULES, **(rules or {}))
    U = SW.unit_from_site("u", rec, outline_sides=outline_sides)
    lo, hi = U.outline()
    gg = np.zeros(6, np.uint8)
    gg[1] = gg[2] = gg[3] = 1
    gid = np.zeros(1, np.int32)
    foot, ring = [], []
    fo = SW.SCREEN_FRAME[0] / 2
    step, border = P["step_m"], P["ring_m"]
    n_rays = 0
    for sa in np.arange(lo[0] - border, hi[0] + border + 1e-9, step):
        for sb in np.arange(lo[1] - border, hi[1] + border + 1e-9, step):
            in_foot = lo[0] <= sa <= hi[0] and lo[1] <= sb <= hi[1]
            ds = max(abs(sa), abs(sb))                      # Chebyshev distance from the screen centre
            in_ring = fo < ds <= fo + border
            if not (in_foot or in_ring):
                continue
            p0 = U.o + U.a * sa + U.b * sb + U.n * P["probe_m"]
            dist = mujoco.mj_ray(model, data_, p0, -U.n, gg, 1, -1, gid)
            n_rays += 1
            dv = float(P["probe_m"] - dist) if dist >= 0 else float("nan")
            if in_foot:
                foot.append(dv)
            if in_ring:
                ring.append(dv)
    foot, ring = np.array(foot), np.array(ring)
    c, R, h = U.obb(protrusion=SW.FRAME_DEPTH)
    mid = module_of(feats, c)
    corridor = True
    if mid is not None:
        M = feats["modules"][mid]
        o = np.asarray(M["axis_origin"], float)
        u, w = np.asarray(M["lat_u"], float), np.asarray(M["lat_w"], float)
        rel = obb_corners(c, R, h) - o
        corridor = bool(((np.abs(rel @ u) < P["corridor_half_m"]) & (np.abs(rel @ w) < P["corridor_half_m"])).any())
    fin = bool(np.isfinite(foot).all()) and len(foot) > 0
    ref = float(np.median(foot)) if fin else 0.0                       # the footprint's own plane (proud of the rack face)
    mount = bool(fin and np.nanmax(np.abs(foot - ref)) <= P["mount_tol_m"])
    clear = bool(np.nanmax(ring - ref) <= P["ring_proud_m"]) if np.isfinite(ring).any() else False
    inl = feats["v02"]["inlets"]
    d_inlet = min(float(np.linalg.norm(U.face_center - np.asarray(i["face_center"], float))) for i in inl)
    near_inlet = min(inl, key=lambda i: float(np.linalg.norm(U.face_center - np.asarray(i["face_center"], float))))["id"]
    inlet_ok = d_inlet >= P["inlet_min_m"]
    anyf = np.isfinite(foot).any()
    return dict(site=rec["id"], module=mid, plane_offset_m=ref, foot_max_proud_m=float(np.nanmax(foot) - ref) if anyf else None,
                foot_min_m=float(np.nanmin(foot) - ref) if anyf else None,
                ring_max_proud_m=float(np.nanmax(ring) - ref) if np.isfinite(ring).any() else None, n_rays=n_rays,
                corridor_violation=corridor, mountable=mount, clear=clear, inlet_distance_m=d_inlet, nearest_inlet=near_inlet,
                inlet_ok=inlet_ok, ok=bool(mount and clear and not corridor and inlet_ok and mid == rec["module"]))


def grid_points(face, dist: float, n: int = 5) -> np.ndarray:
    s1 = (np.arange(n) + 0.5) / n * 2 * face.half[0] - face.half[0]
    s2 = (np.arange(n) + 0.5) / n * 2 * face.half[1] - face.half[1]
    return np.array([face.c + face.a * x + face.b * y + face.n * dist for x in s1 for y in s2])


def flow_stats(field, face, dists=FLOW_DISTS) -> dict:
    """Air in front of a face on a 5 x 5 grid over its opening at each distance: mean normal component (negative =
    toward the face), its minimum, mean tangential speed, max speed, mean tangential direction in (a, b)."""
    out = {}
    for dist in dists:
        P = grid_points(face, dist)
        u = np.asarray(field(P))
        un = u @ face.n
        ut = u - un[:, None] * face.n[None, :]
        out[f"d{dist:.2f}"] = dict(vn_mean=float(un.mean()), vn_min=float(un.min()), vt_mean=float(np.linalg.norm(ut, axis=1).mean()),
                                   speed_max=float(np.linalg.norm(u, axis=1).max()),
                                   vt_dir_ab=[float(ut.mean(0) @ face.a), float(ut.mean(0) @ face.b)])
    return out


def prescreen(flow: dict, rules: dict | None = None) -> dict:
    """Pre-screen of the air: the mean normal component toward the face <= 1 cm/s at 0.15 and 0.30 m."""
    P = dict(RULES, **(rules or {}))
    vn = {k: v["vn_mean"] for k, v in flow.items() if k in ("d0.15", "d0.30")}
    ok = all(-v <= P["prescreen_vn_toward_max_ms"] for v in vn.values())
    return dict(ok=bool(ok), vn_mean=vn, limit_toward_ms=P["prescreen_vn_toward_max_ms"])
