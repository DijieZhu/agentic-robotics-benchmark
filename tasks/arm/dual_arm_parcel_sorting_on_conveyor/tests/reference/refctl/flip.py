"""Primitive P1: two-palm squeeze flip.

Both grippers, open, press their fingertip pads on the parcel's upstream and
downstream faces (tool axes along the belt, pointing at each other) with a
set squeeze force; the parcel is lifted, rotated about the squeeze line (the
line through both pad centres, along the belt) and set down again.

Rotation sign: theta about world +y. theta < 0 tips the parcel's top toward
the robot (the direction both arms can turn far: about -115..-150 deg with
horizontal pads); theta > 0 needs the pads started vertical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kin import rot_axis_angle
from .paths import (SG, SL, SIDES, PlanFail, lerp_pose, palm_R, solve_path,
                    time_scale)
from .motion import DT, min_jerk

PAD_HALF_SPAN = 0.051      # pad centre +- (in the opening direction), open
PAD_HALF_H = 0.011         # pad half size across the opening direction
FINGER_HALF = 0.019        # finger links half thickness (TCP y)


@dataclass
class FlipPlan:
    segments: list          # [(name, q (m,14), grip (2,), force_on, extra)]
    squeeze_line: np.ndarray
    lift_z: float
    theta: float
    psi0: float
    info: dict


def corner_radius(Lx, H, xs_rel, zs_rel):
    """Max distance (x-z plane) from the squeeze line to the parcel's
    corners; (xs_rel, zs_rel) = squeeze line relative to the parcel centre."""
    r = 0.0
    for sx in (-1, 1):
        for sz in (-1, 1):
            r = max(r, math.hypot(sx * Lx / 2 - xs_rel, sz * H / 2 - zs_rel))
    return r


def lowest_after(Lx, H, xs_rel, zs_rel, theta):
    """Lowest z of the parcel relative to the squeeze line after rotating by
    theta about +y."""
    c, s = math.cos(theta), math.sin(theta)
    low = 1e9
    for sx in (-1, 1):
        for sz in (-1, 1):
            x = sx * Lx / 2 - xs_rel
            z = sz * H / 2 - zs_rel
            # R_y(theta): x' = c x + s z, z' = -s x + c z
            low = min(low, -s * x + c * z)
    return low


def plan_flip(kins, q14, parcel_pos, dims, belt_top, theta, *, psi0=0.0,
              yaw=0.0, grip_open=0.0, overpress=0.006, z_frac=0.5,
              x_shift=0.0, clear=0.012, vmax=2.5, pre_gap=0.05,
              rot_time_per_rad=0.55, tilt=0.0):
    """Plan the squeeze flip of a parcel resting on the belt (static).

    dims = (Lx, Ly, H) along world x, world y (squeeze axis), z.
    Returns FlipPlan; raises PlanFail."""
    Lx, Ly, H = dims
    xc, yc, zc = parcel_pos
    zs = belt_top + max(0.022, min(H - 0.022, z_frac * H))
    xs = xc + x_shift
    xs_rel, zs_rel = xs - xc, zs - (belt_top + H / 2)
    r = corner_radius(Lx, H, xs_rel, zs_rel)
    lift_z = belt_top + r + clear                   # squeeze line during rotation
    lift_z = max(lift_z, zs + 0.03)
    low = lowest_after(Lx, H, xs_rel, zs_rel, theta)
    down_z = belt_top - low - 0.003                 # press 3 mm on set-down
    R0 = {s: palm_R(s, SG[s] * psi0 if s == "Left" else -psi0, yaw)
          for s in SIDES}
    # palm frames: left psi0, right mirrored (-psi0) -> same pad layout
    R0 = {"Left": palm_R("Left", psi0, yaw, tilt),
          "Right": palm_R("Right", -psi0, yaw, tilt)}
    ay = np.array([-math.sin(yaw), math.cos(yaw), 0.0])     # squeeze axis
    def P(side, gap, z):
        return np.array([xs, yc, z]) + ay * SG[side] * (Ly / 2 + gap)
    segs = []
    q = np.asarray(q14, float).copy()

    def path(fns, n=30, vm=vmax, tmin=0.2, name=""):
        nonlocal q
        Q = np.repeat(q[None], n + 1, axis=0)
        for s, f in fns.items():
            poses = [f(u) for u in np.linspace(0, 1, n + 1)[1:]]
            try:
                Q[1:, SL[s]] = solve_path(kins[s], poses, q[SL[s]])
            except PlanFail as e:
                raise PlanFail(f"[{name or len(segs)}] {e}")
        out = time_scale(Q, vmax=vm, tmin=tmin)
        q = out[-1].copy()
        return out

    g = (grip_open, grip_open)
    # 0. pre-grasp: joint move from the current posture
    pre = {s: (P(s, pre_gap, zs + 0.06), R0[s]) for s in SIDES}
    q_pre = q.copy()
    from .paths import Q_PALM_NOM
    for s in SIDES:
        qs_, ok, ep, er = kins[s].ik(pre[s][0], R=pre[s][1],
                                     q0=Q_PALM_NOM[s], q_rest=Q_PALM_NOM[s],
                                     iters=300, margin=0.01)
        if not ok:
            qs_, ok, ep, er = kins[s].ik(pre[s][0], R=pre[s][1], q0=q[SL[s]],
                                         iters=300, margin=0.01)
        if not ok:
            raise PlanFail(f"{s}: pre-grasp IK fail ({ep:.3f}, {er:.3f})")
        q_pre[SL[s]] = qs_
    from .motion import joint_move
    qm, _, _ = joint_move(q, q_pre, vmax=vmax, tmin=0.4)
    segs.append(("to_pre", qm, g, False, None))
    q = q_pre.copy()
    # 1. descend beside the faces
    segs.append(("descend", path({s: (lambda s: lambda u: lerp_pose(
        pre[s][0], R0[s], P(s, pre_gap, zs), R0[s], u))(s) for s in SIDES}),
        g, False, None))
    # 2. close in to overpress
    segs.append(("close", path({s: (lambda s: lambda u: lerp_pose(
        P(s, pre_gap, zs), R0[s], P(s, -overpress, zs), R0[s], u))(s)
        for s in SIDES}, vm=1.0), g, "ramp_up", None))
    # 3. lift
    segs.append(("lift", path({s: (lambda s: lambda u: lerp_pose(
        P(s, -overpress, zs), R0[s], P(s, -overpress, lift_z), R0[s], u))(s)
        for s in SIDES}, vm=1.5), g, True, None))
    # 4. rotate about the squeeze line (world +y through the pad centres)
    def rot_fn(s):
        def f(u):
            Rw = rot_axis_angle((0, 1, 0), theta * u)
            return P(s, -overpress, lift_z), Rw @ R0[s]
        return f
    nrot = max(20, int(abs(theta) / math.radians(3)))
    segs.append(("rotate", path({s: rot_fn(s) for s in SIDES}, n=nrot,
                                vm=vmax, tmin=rot_time_per_rad * abs(theta)),
                 g, True, None))
    Rf = {s: rot_axis_angle((0, 1, 0), theta) @ R0[s] for s in SIDES}
    # 5. lower onto the belt
    segs.append(("lower", path({s: (lambda s: lambda u: lerp_pose(
        P(s, -overpress, lift_z), Rf[s], P(s, -overpress, down_z), Rf[s], u))(s)
        for s in SIDES}, vm=1.2), g, True, None))
    # 6. release: back off, then up
    segs.append(("release", path({s: (lambda s: lambda u: lerp_pose(
        P(s, -overpress, down_z), Rf[s], P(s, 0.04, down_z), Rf[s], u))(s)
        for s in SIDES}, vm=1.5), g, "ramp_down", None))
    segs.append(("up", path({s: (lambda s: lambda u: lerp_pose(
        P(s, 0.04, down_z), Rf[s], P(s, 0.04, down_z + 0.10), Rf[s], u))(s)
        for s in SIDES}, vm=2.0), g, False, None))
    return FlipPlan(segs, np.array([xs, yc, zs]), lift_z, theta, psi0,
                    dict(r=r, down_z=down_z, pre=pre))
