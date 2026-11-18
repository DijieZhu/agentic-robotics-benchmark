"""The 2F-85 grasps of the race car's loose units (grader side: the privileged reference program's).

A grasp is the tcp pose G in the unit's body frame (the kit's body frames; tcp axes: z = approach, y = closing axis
with the right pad on +y, x = y x z) and the opening the pads stop at. The precedent's generator
(`assembly_quad/grasps.py`) puts the tcp inside the part's bounding box and needs a ray hit in the tcp's layer: it finds
no fingertip pinch of a thin band (the compute module's 34.56 mm heat-sink faces, the deck plate's 5.19 mm edge), and
its 6 mm minimum opening excludes the deck. This module has the families of the grasp study for this car instead
(moved here unchanged in their parameters, mm):

* `wheel_grasps`    top-down across the tyre diameter, the tool axis on the wheel axis (the hub axis frame: origin
                    the wheel centre, z the axle outwards, x the car's up projected on the wheel plane); parameters:
                    the closing axis' angle phi about the axis, the fingertips' depth below the wheel's outer face.
                    `wheel_grasps_unit(kit, corner)` maps them into the frame of the wheel designed for that corner.
* `battery_grasps`  top-down across the pack's width over its equator, the approach leaning by beta about the pack's
                    long axis; parameters: the station along the pack, beta, the fingertips below the contact line,
                    which pad is inboard.
* `deck_grasps`     the side pinch: horizontal from the deck's right long edge, the pads on the plate's two faces;
                    parameters: the station along the edge (car x), the fingertips' depth from the edge, the approach
                    yaw in the plate's plane, which pad is on top.
* `topdown_grasps`  the compute module and the LiDAR: top-down, the closing axis horizontal at theta; parameters: theta,
                    the fingertips below the unit's top, a station across the pads; the tip centred on the box of the
                    unit's surface above the tips.

`FROZEN` is the grasp table frozen on the real kit by the kit recheck (`FROZEN_SOURCE`: its schema and sha256): per slot
the kit study's row, its role ("primary": dynamically validated on the kit and the same parameters valid on the true
solids; "alternate": static only; "kit only": valid on the collision proxy only, never used by the reference
program), the family parameters, the opening on the kit's pieces and G (mm, as frozen: 5 decimals; `as_m`
re-orthonormalises the rotation). `frozen_grasps(slot)` gives the primary ones in metres, best first. `ADDED` holds the
grasps added for the reference program in front of a slot's frozen ones, each with its reason (the stick
pack: the frozen primary cannot reach its 24 mm release on the kit); `reference_grasps(slot)` = ADDED + the frozen
primaries, the reference program's candidates.

`analyse` is the grasp study's pad-closing analysis: the two pads (22 x 37.5 mm faces, from the 2F-85's opening table,
parallel to the approach) close along the closing axis on a unit's surface samples (`unit_surface`: the kit's
collision geoms of the unit sampled on their faces, union only), each stopping at the extreme sample inside its
footprint, iterated on the opening; it reports the opening, both contact patches, their overlap seen along the closing
axis and the centre of mass' offset from the contact line. `family_candidates(kit, unit, ...)` runs a family through it
(the reference program's fallback when no frozen grasp of a unit is feasible in an episode).
"""
import math
import os
import struct

import numpy as np

from ..assembly_quad.gripper_2f85 import MAX_WIDTH_M, PAD_ACROSS_M, PAD_ALONG_M, pad_centre_z, pad_tip_z

__all__ = ["FROZEN", "FROZEN_SOURCE", "ADDED", "SLOT_UNIT", "frozen_entries", "frozen_grasps", "reference_grasps", "as_m", "tool_frame", "G_from_tip",
           "wheel_grasps", "axis_frame", "wheel_grasps_unit", "battery_grasps", "deck_grasps", "topdown_grasps",
           "regenerate", "unit_surface", "pad_faces_tcp", "analyse", "family_candidates"]

# the slots of the four top-down units and the wheels -> the unit family (the kit's unit types)
SLOT_UNIT = {"battery": "BAT", "deck": "DCK", "compute": "CMP", "lidar": "LDR",
             "whl_FL": "WHL", "whl_FR": "WHL", "whl_RL": "WHL", "whl_RR": "WHL"}
CORNER_WHEEL = {"FL": "WHL1", "FR": "WHL2", "RL": "WHL3", "RR": "WHL4"}      # the wheel designed for each corner

PATCH_TOL = 0.5                 # mm: samples within this of a pad's contact plane form its contact patch
ALIGN_MIN_DEG = 20.0            # the median sample normal of a patch within this of the closing axis
MIN_PATCH_ALONG = 3.0           # mm of contact along the pad (approach) on each side
MIN_PATCH_ACROSS = 5.0          # mm across the pad

WHEEL_OUTER_FACE = 13.54        # mm: the wheel's outer face from its centre (unit map frames.md 6)
BAT_CONTACT_Z = 11.94           # the cell pack's cross-section centre above its bottom (23.88 / 2)
DECK_EDGE_Y = -64.785           # the deck's right long edge (DCK frame)
DECK_MID_Z = -2.594             # the plate's mid-plane (5.187 thick, top at 0)
DCK_X0 = 132.826                # DCK frame origin x in the car frame


# ------------------------------------------------------------------ small transforms (mm)
def rotx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rotz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def TR(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def as_m(G_mm):
    """a grasp in mm (3 x 4 or 4 x 4, rounded) -> 4 x 4 in m with the nearest rotation (polar decomposition)"""
    G = np.eye(4)
    A = np.asarray(G_mm, float)
    G[:3, :4] = A[:3, :4]
    U, _s, Vt = np.linalg.svd(G[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    G[:3, :3] = R
    G[:3, 3] *= 1e-3
    return G


# ------------------------------------------------------------------ families (the grasp study's, mm)
def tool_frame(a, c):
    """rotation with z = approach a, y = closing axis c (right pad on +y), x = y x z"""
    a = np.asarray(a, float) / np.linalg.norm(a)
    c = np.asarray(c, float) - np.dot(c, a) * a
    c /= np.linalg.norm(c)
    return np.column_stack([np.cross(c, a), c, a])


def G_from_tip(a, c, tip, width_mm):
    """the tcp pose for an approach a, closing axis c and the fingertips' centre point `tip` (unit frame, mm)"""
    R = tool_frame(a, c)
    p = np.asarray(tip, float) - R[:, 2] * pad_tip_z(width_mm * 1e-3) * 1e3
    return TR(R, p)


def wheel_grasps(phis=range(0, 360, 15), depths=(6.0, 8.0, 10.0, 13.0, 17.0, 21.0), width=65.8):
    """in the hub axis frame: [(params, G_axis)]"""
    out = []
    for phi in phis:
        c = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi)), 0.0])
        a = np.array([0.0, 0.0, -1.0])
        for d in depths:
            tip = np.array([0.0, 0.0, WHEEL_OUTER_FACE - d])
            out.append(({"phi_deg": float(phi), "tip_depth_mm": float(d)}, G_from_tip(a, c, tip, width)))
    return out


def axis_frame(kit, corner):
    """(A, the corner's designed wheel unit, Tw): the hub axis frame of a corner at home (car frame mm): origin = the
    wheel centre, z = the axle outward, x = car up projected on the wheel plane; Tw the designed wheel's body pose"""
    unit = CORNER_WHEEL[corner]
    Tw = np.array(kit["bodies"][unit]["T_design"], float)
    Tw[:3, 3] *= 1e3
    z = Tw[:3, 2]
    up = np.array([0.0, 0.0, 1.0])
    x = up - (up @ z) * z
    x /= np.linalg.norm(x)
    A = TR(np.column_stack([x, np.cross(z, x), z]), Tw[:3, 3])
    return A, unit, Tw


def wheel_grasps_unit(kit, corner, **kw):
    """the wheel family of a corner in the frame of the wheel designed for it (mm): [(params, G_unit)]"""
    A, _u, Tw = axis_frame(kit, corner)
    Gu = inv(Tw) @ A
    return [(p, Gu @ G) for p, G in wheel_grasps(**kw)]


def battery_grasps(stations=np.arange(-55.0, 55.1, 2.5), betas=(-20.0, -10.0, 0.0, 10.0, 20.0, 30.0),
                   tips=(2.0, 4.0, 7.0), width=46.4):
    out = []
    for s in stations:
        for beta in betas:
            R = rotx(beta)
            a = R @ np.array([0.0, 0.0, -1.0])
            for flip in (1.0, -1.0):
                c = R @ np.array([0.0, flip, 0.0])
                for t in tips:
                    tip = np.array([s, 0.0, BAT_CONTACT_Z]) + a * t
                    out.append(({"station_x_mm": float(s), "beta_deg": float(beta), "tip_below_contact_mm": float(t),
                                 "right_pad": "inboard (+y)" if flip > 0 else "outboard (-y)"},
                                G_from_tip(a, c, tip, width)))
    return out


def deck_grasps(stations_car=np.arange(75.0, 150.1, 7.5), depths=(6.0, 7.5, 9.0), yaws=(-20.0, -10.0, 0.0, 10.0, 20.0),
                width=5.19):
    out = []
    for xs in stations_car:
        for g in yaws:
            a = rotz(g) @ np.array([0.0, 1.0, 0.0])
            for flip in (1.0, -1.0):
                c = np.array([0.0, 0.0, flip])
                for d in depths:
                    tip = np.array([xs - DCK_X0, DECK_EDGE_Y + d, DECK_MID_Z])
                    out.append(({"station_x_car_mm": float(xs), "tip_depth_mm": float(d), "yaw_deg": float(g),
                                 "right_pad": "top" if flip > 0 else "bottom"}, G_from_tip(a, c, tip, width)))
    return out


def topdown_grasps(data, thetas, tip_depths, offsets=(-8.0, 0.0, 8.0), width_guess=40.0, centre=None):
    """top-down pinches in the horizontal plane of the unit frame: closing angle theta, fingertips `t` below the top
    (`data`: an object with P, the unit's surface samples, and hi, their upper corner; `unit_surface`)"""
    out = []
    ztop = float(data.hi[2])
    for th in thetas:
        c = np.array([math.cos(math.radians(th)), math.sin(math.radians(th)), 0.0])
        a = np.array([0.0, 0.0, -1.0])
        xt = np.cross(c, a)
        for t in tip_depths:
            zt = ztop - t
            sel = data.P[:, 2] >= zt
            ctr = 0.5 * (data.P[sel].min(0) + data.P[sel].max(0)) if centre is None else np.asarray(centre, float)
            for off in offsets:
                for flip in (1.0, -1.0):
                    tip = np.array([ctr[0], ctr[1], zt]) + xt * off
                    out.append(({"theta_deg": float(th), "tip_below_top_mm": float(t), "offset_mm": float(off),
                                 "flip": int(flip)}, G_from_tip(a, flip * c, tip, width_guess)))
    return out


def regenerate(slot, params, kit=None, data=None, width_mm=None):
    """G (mm, 4 x 4) of one grasp from its family parameters (the wheels need the kit, the top-down families the unit's
    samples; `width_mm`: the opening the family was generated at, its default)"""
    fam = SLOT_UNIT[slot]
    if fam == "WHL":
        g = wheel_grasps_unit(kit, slot.split("_")[1], phis=[params["phi_deg"]], depths=[params["tip_depth_mm"]],
                              **({} if width_mm is None else {"width": width_mm}))
    elif fam == "BAT":
        g = battery_grasps(stations=[params["station_x_mm"]], betas=[params["beta_deg"]], tips=[params["tip_below_contact_mm"]],
                           **({} if width_mm is None else {"width": width_mm}))
        g = [x for x in g if x[0]["right_pad"] == params["right_pad"]]
    elif fam == "DCK":
        g = deck_grasps(stations_car=[params["station_x_car_mm"]], depths=[params["tip_depth_mm"]], yaws=[params["yaw_deg"]],
                        **({} if width_mm is None else {"width": width_mm}))
        g = [x for x in g if x[0]["right_pad"] == params["right_pad"]]
    else:
        g = topdown_grasps(data, thetas=[params["theta_deg"]], tip_depths=[params["tip_below_top_mm"]],
                           offsets=[params["offset_mm"]], width_guess=width_mm or 40.0)
        g = [x for x in g if x[0]["flip"] == params["flip"]]
    if len(g) != 1:
        raise ValueError("%s: %d grasps for %s" % (slot, len(g), params))
    return g[0][1]


# ------------------------------------------------------------------ the frozen table
def _entry(row, role, params, width_mm, G_mm, **extra):
    e = {"row": int(row), "role": role, "params": dict(params), "width_mm": float(width_mm), "G_mm": G_mm}
    e.update(extra)
    return e


FROZEN_SOURCE = {"file": "the grasp table frozen by the kit recheck of 2026-11-14 (its rows: FROZEN)", "schema": 'racecar-grasps-frozen/1', "sha256": '7b458a2641bf3eb411f89675db9074f32659576eccd5c1f95245040bd7b7de66'}
FROZEN = {
    'whl_FL': [
        _entry(68, 'primary', {'phi_deg': 255.0, 'tip_depth_mm': 6.0}, 65.19,
               [[0.95263, -0.30412, 0.0, 0.0], [-0.30412, -0.95263, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-0.456529, -4.263384, 2.030705, -0.914395, -2.72316, 0.25679]),
        _entry(72, 'alternate', {'phi_deg': 270.0, 'tip_depth_mm': 6.0}, 65.35,
               [[0.99889, -0.0472, 0.0, 0.0], [-0.0472, -0.99889, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-0.456529, -4.263384, 2.030705, -0.914395, -2.72316, -0.005009]),
        _entry(76, 'alternate', {'phi_deg': 285.0, 'tip_depth_mm': 6.0}, 65.39,
               [[0.97706, 0.21294, 0.0, 0.0], [0.21294, -0.97706, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-0.456529, -4.263384, 2.030705, -0.914395, -2.72316, -0.266808]),
        _entry(80, 'alternate', {'phi_deg': 300.0, 'tip_depth_mm': 6.0}, 65.19,
               [[0.88866, 0.45857, 0.0, 0.0], [0.45857, -0.88866, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-0.456529, -4.263384, 2.030705, -0.914395, -2.72316, -0.528608]),
        _entry(84, 'alternate', {'phi_deg': 315.0, 'tip_depth_mm': 6.0}, 65.33,
               [[0.73969, 0.67295, 0.0, 0.0], [0.67295, -0.73969, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-0.456529, -4.263384, 2.030705, -0.914395, -2.72316, -0.790407]),
        _entry(64, 'alternate', {'phi_deg': 240.0, 'tip_depth_mm': 6.0}, 65.4,
               [[0.84146, -0.54032, 0.0, 0.0], [-0.54032, -0.84146, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-0.456529, -4.263384, 2.030705, -0.914395, -2.72316, 0.51859]),
    ],
    'whl_FR': [
        _entry(20, 'primary', {'phi_deg': 75.0, 'tip_depth_mm': 6.0}, 65.14,
               [[-0.63156, 0.77533, 0.0, 0.0], [0.77533, 0.63156, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.685074, -1.170907, 2.014236, -0.837823, -2.723119, 0.26683]),
        _entry(24, 'alternate', {'phi_deg': 90.0, 'tip_depth_mm': 6.0}, 65.38,
               [[-0.81071, 0.58545, 0.0, 0.0], [0.58545, 0.81071, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.685074, -1.170907, 2.014236, -0.837823, -2.723119, 0.005031]),
        _entry(28, 'alternate', {'phi_deg': 105.0, 'tip_depth_mm': 6.0}, 65.38,
               [[-0.93461, 0.35567, 0.0, 0.0], [0.35567, 0.93461, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.685074, -1.170907, 2.014236, -0.837823, -2.723119, -0.256768]),
        _entry(32, 'alternate', {'phi_deg': 120.0, 'tip_depth_mm': 6.0}, 65.14,
               [[-0.99482, 0.10166, 0.0, 0.0], [0.10166, 0.99482, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.685074, -1.170907, 2.014236, -0.837823, -2.723119, -0.518568]),
        _entry(16, 'alternate', {'phi_deg': 60.0, 'tip_depth_mm': 6.0}, 65.39,
               [[-0.40937, 0.91237, 0.0, 0.0], [0.91237, 0.40937, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.685074, -1.170907, 2.014236, -0.837823, -2.723119, 0.52863]),
        _entry(40, 'alternate', {'phi_deg': 150.0, 'tip_depth_mm': 6.0}, 65.39,
               [[-0.91237, -0.40937, 0.0, 0.0], [-0.40937, 0.91237, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.685074, -1.170907, 2.014236, -0.837823, -2.723119, -1.042167]),
    ],
    'whl_RL': [
        _entry(0, 'primary', {'phi_deg': 0.0, 'tip_depth_mm': 6.0}, 65.4,
               [[-0.40107, 0.91605, 0.0, 0.0], [0.91605, 0.40107, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-3.864557, -0.887724, 1.523075, -0.624896, -0.722978, 1.562956]),
        _entry(56, 'alternate', {'phi_deg': 210.0, 'tip_depth_mm': 6.0}, 65.35,
               [[0.80536, -0.59278, 0.0, 0.0], [-0.59278, -0.80536, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-3.864557, -0.887724, 1.523075, -0.624896, -0.722978, -2.102236]),
        _entry(88, 'alternate', {'phi_deg': 330.0, 'tip_depth_mm': 6.0}, 65.18,
               [[0.11068, 0.99386, 0.0, 0.0], [0.99386, -0.11068, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-3.864557, -0.887724, 1.523075, -0.624896, -0.722978, 2.086555]),
        _entry(92, 'alternate', {'phi_deg': 345.0, 'tip_depth_mm': 6.0}, 65.34,
               [[-0.15032, 0.98864, 0.0, 0.0], [0.98864, 0.15032, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-3.864557, -0.887724, 1.523075, -0.624896, -0.722978, 1.824755]),
        _entry(4, 'alternate', {'phi_deg': 15.0, 'tip_depth_mm': 6.0}, 65.18,
               [[-0.6245, 0.78103, 0.0, 0.0], [0.78103, 0.6245, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-3.864557, -0.887724, 1.523075, -0.624896, -0.722978, 1.301156]),
        _entry(8, 'alternate', {'phi_deg': 30.0, 'tip_depth_mm': 6.0}, 65.35,
               [[-0.80536, 0.59278, 0.0, 0.0], [0.59278, 0.80536, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-3.864557, -0.887724, 1.523075, -0.624896, -0.722978, 1.039357]),
    ],
    'whl_RR': [
        _entry(24, 'primary', {'phi_deg': 90.0, 'tip_depth_mm': 6.0}, 65.38,
               [[-0.99926, 0.03836, 0.0, 0.0], [0.03836, 0.99926, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.824679, -0.885836, 1.519651, -0.6116, -2.824591, 0.021108]),
        _entry(28, 'alternate', {'phi_deg': 105.0, 'tip_depth_mm': 6.0}, 65.38,
               [[-0.97514, -0.22158, 0.0, 0.0], [-0.22158, 0.97514, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.824679, -0.885836, 1.519651, -0.6116, -2.824591, -0.240691]),
        _entry(32, 'alternate', {'phi_deg': 120.0, 'tip_depth_mm': 6.0}, 65.14,
               [[-0.88457, -0.46641, 0.0, 0.0], [-0.46641, 0.88457, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.824679, -0.885836, 1.519651, -0.6116, -2.824591, -0.502491]),
        _entry(12, 'alternate', {'phi_deg': 45.0, 'tip_depth_mm': 6.0}, 65.37,
               [[-0.67946, 0.73371, 0.0, 0.0], [0.73371, 0.67946, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.824679, -0.885836, 1.519651, -0.6116, -2.824591, 0.806506]),
        _entry(16, 'alternate', {'phi_deg': 60.0, 'tip_depth_mm': 6.0}, 65.39,
               [[-0.84621, 0.53285, 0.0, 0.0], [0.53285, 0.84621, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.824679, -0.885836, 1.519651, -0.6116, -2.824591, 0.544707]),
        _entry(20, 'alternate', {'phi_deg': 75.0, 'tip_depth_mm': 6.0}, 65.15,
               [[-0.95529, 0.29568, 0.0, 0.0], [0.29568, 0.95529, 0.0, 0.0], [0.0, 0.0, -1.0, 18.42738]],
               q_pre=[-2.824679, -0.885836, 1.519651, -0.6116, -2.824591, 0.282907]),
    ],
    'battery': [
        _entry(72, 'kit only', {'station_x_mm': -40.0, 'beta_deg': 5.0, 'tip_below_contact_mm': 2.0, 'right_pad': 'inboard (+y)'}, 46.62,
               [[-1.0, 0.0, 0.0, -40.0], [0.0, 0.99619, 0.08716, -1.12709], [0.0, 0.08716, -0.99619, 24.82266]],
               release_height_mm=24.0, q_pre=[-3.314454, -1.540712, 2.003466, -2.048598, -1.656759, -0.173508]),
        _entry(178, 'primary', {'station_x_mm': -15.0, 'beta_deg': 10.0, 'tip_below_contact_mm': 2.0, 'right_pad': 'outboard (-y)'}, 46.48,
               [[1.0, 0.0, 0.0, -15.0], [0.0, -0.98481, 0.17365, -2.2456], [0.0, -0.17365, -0.98481, 24.67541]],
               release_height_mm=24.0),
        _entry(134, 'kit only', {'station_x_mm': -25.0, 'beta_deg': 5.0, 'tip_below_contact_mm': 2.0, 'right_pad': 'outboard (-y)'}, 46.6,
               [[1.0, 0.0, 0.0, -25.0], [0.0, -0.99619, 0.08716, -1.12709], [0.0, -0.08716, -0.99619, 24.82266]],
               release_height_mm=24.0, q_pre=[-3.318522, -1.565713, 2.029077, -2.049558, -1.656697, 2.964002]),
    ],
    'deck': [
        _entry(130, 'primary', {'station_x_car_mm': 127.5, 'tip_depth_mm': 7.5, 'yaw_deg': -10.0, 'right_pad': 'bottom'}, 5.19,
               [[0.98481, 0.0, 0.17365, -8.38549], [-0.17365, 0.0, 0.98481, -74.63621], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.902493, -1.130123, 1.726644, -0.596521, -2.72796, -1.570796]),
        _entry(129, 'primary', {'station_x_car_mm': 127.5, 'tip_depth_mm': 6.0, 'yaw_deg': -10.0, 'right_pad': 'bottom'}, 5.19,
               [[0.98481, 0.0, 0.17365, -8.38549], [-0.17365, 0.0, 0.98481, -76.13621], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.900508, -1.129163, 1.725154, -0.59599, -2.725976, -1.570796]),
        _entry(131, 'primary', {'station_x_car_mm': 127.5, 'tip_depth_mm': 9.0, 'yaw_deg': -10.0, 'right_pad': 'bottom'}, 5.19,
               [[0.98481, 0.0, 0.17365, -8.38549], [-0.17365, 0.0, 0.98481, -73.13621], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.90448, -1.131078, 1.728128, -0.59705, -2.729947, -1.570796]),
        _entry(149, 'primary', {'station_x_car_mm': 135.0, 'tip_depth_mm': 9.0, 'yaw_deg': -10.0, 'right_pad': 'bottom'}, 5.19,
               [[0.98481, 0.0, 0.17365, -0.88549], [-0.17365, 0.0, 0.98481, -73.13621], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.902054, -1.140451, 1.742638, -0.602187, -2.727521, -1.570796]),
        _entry(165, 'primary', {'station_x_car_mm': 142.5, 'tip_depth_mm': 6.0, 'yaw_deg': -10.0, 'right_pad': 'bottom'}, 5.19,
               [[0.98481, 0.0, 0.17365, 6.61451], [-0.17365, 0.0, 0.98481, -76.13621], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.895537, -1.147851, 1.754032, -0.606181, -2.721004, -1.570796]),
        _entry(167, 'primary', {'station_x_car_mm': 142.5, 'tip_depth_mm': 9.0, 'yaw_deg': -10.0, 'right_pad': 'bottom'}, 5.19,
               [[0.98481, 0.0, 0.17365, 6.61451], [-0.17365, 0.0, 0.98481, -73.13621], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.89958, -1.149798, 1.757022, -0.607223, -2.725047, -1.570796]),
        _entry(118, 'primary (raw release / retreat not checked)', {'station_x_car_mm': 120.0, 'tip_depth_mm': 7.5, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, -12.826], [0.0, 0.0, 1.0, -74.90388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.881737, -1.180323, 1.803383, -0.62306, -2.881737, -1.570796]),
        _entry(136, 'primary (raw release / retreat not checked)', {'station_x_car_mm': 127.5, 'tip_depth_mm': 7.5, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, -5.326], [0.0, 0.0, 1.0, -74.90388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, q_pre=[-2.878945, -1.189526, 1.817173, -0.627647, -2.878945, -1.570796]),
    ],
    'compute': [
        _entry(100, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 6.0, 'offset_mm': 8.0, 'flip': 1.0}, 34.56,
               [[-1.0, 0.0, 0.0, -14.49], [0.0, 1.0, 0.0, -7.648], [0.0, 0.0, -1.0, 46.96762]],
               q_pre=[-3.405292, -1.679706, 1.988972, -1.880062, -1.570796, -0.263699]),
        _entry(94, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 4.5, 'offset_mm': 8.0, 'flip': 1.0}, 34.56,
               [[-1.0, 0.0, 0.0, -11.36], [0.0, 1.0, 0.0, -7.648], [0.0, 0.0, -1.0, 48.46762]],
               q_pre=[-3.406665, -1.686067, 1.991776, -1.876505, -1.570796, -0.265072]),
        _entry(223, 'kit only', {'theta_deg': 90.0, 'tip_below_top_mm': 14.0, 'offset_mm': 0.0, 'flip': -1.0}, 70.79,
               [[1.0, 0.0, 0.0, -7.275], [0.0, -1.0, 0.0, 10.465], [0.0, 0.0, -1.0, 33.85107]],
               q_pre=[-3.437985, -1.685257, 2.018031, -1.903571, -1.570796, 2.845201]),
        _entry(92, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 4.5, 'offset_mm': 0.0, 'flip': 1.0}, 34.56,
               [[-1.0, 0.0, 0.0, -3.36], [0.0, 1.0, 0.0, -7.648], [0.0, 0.0, -1.0, 48.46762]],
               q_pre=[-3.410241, -1.700326, 2.004075, -1.874546, -1.570796, -0.268649]),
        _entry(98, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 6.0, 'offset_mm': 0.0, 'flip': 1.0}, 34.56,
               [[-1.0, 0.0, 0.0, -6.49], [0.0, 1.0, 0.0, -7.648], [0.0, 0.0, -1.0, 46.96762]],
               q_pre=[-3.40883, -1.693924, 2.001336, -1.878208, -1.570796, -0.267238]),
        _entry(96, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 6.0, 'offset_mm': -8.0, 'flip': 1.0}, 34.56,
               [[-1.0, 0.0, 0.0, 1.51], [0.0, 1.0, 0.0, -7.648], [0.0, 0.0, -1.0, 46.96762]],
               q_pre=[-3.412467, -1.708247, 2.013564, -1.876113, -1.570796, -0.270874]),
        _entry(101, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 6.0, 'offset_mm': 8.0, 'flip': -1.0}, 34.56,
               [[1.0, 0.0, 0.0, -14.49], [0.0, -1.0, 0.0, -7.648], [0.0, 0.0, -1.0, 46.96762]],
               q_pre=[-3.405292, -1.679706, 1.988972, -1.880062, -1.570796, 2.877894]),
        _entry(93, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 4.5, 'offset_mm': 0.0, 'flip': -1.0}, 34.56,
               [[1.0, 0.0, 0.0, -3.36], [0.0, -1.0, 0.0, -7.648], [0.0, 0.0, -1.0, 48.46762]],
               q_pre=[-3.410241, -1.700326, 2.004075, -1.874546, -1.570796, 2.872944]),
    ],
    'lidar': [
        _entry(53, 'kit only', {'theta_deg': 90.0, 'tip_below_top_mm': 13.0, 'offset_mm': 8.0, 'flip': -1.0}, 33.64,
               [[1.0, 0.0, 0.0, -9.94], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 59.60076]],
               q_pre=[-3.466109, -1.826588, 2.083056, -1.827264, -1.570796, 2.817077]),
        _entry(59, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 16.0, 'offset_mm': 8.0, 'flip': -1.0}, 33.64,
               [[1.0, 0.0, 0.0, -9.94], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 56.60076]],
               q_pre=[-3.466109, -1.825146, 2.087454, -1.833104, -1.570796, 2.817077]),
        _entry(70, 'primary', {'theta_deg': 90.0, 'tip_below_top_mm': 44.0, 'offset_mm': 8.0, 'flip': 1.0}, 40.8,
               [[-1.0, 0.0, 0.0, -9.3245], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 28.60076]],
               q_pre=[-3.466489, -1.811134, 2.128463, -1.888125, -1.570796, -0.324896]),
        _entry(0, 'primary', {'theta_deg': 0.0, 'tip_below_top_mm': 13.0, 'offset_mm': -8.0, 'flip': 1.0}, 37.52,
               [[0.0, 1.0, 0.0, -1.94], [1.0, 0.0, 0.0, -8.0], [0.0, 0.0, -1.0, 59.60076]],
               q_pre=[-3.456203, -1.841823, 2.093891, -1.822864, -1.570796, 1.256186]),
    ],
}


# Grasps added for the reference program before the frozen ones, with the reason. The frozen battery primary (row
# 178, station -15) cannot release at 24 mm on the kit: its inboard pad comes within 1 mm of the chassis's speed
# controller pieces (component "esc", car x 108.8-118.6, y -5.6..-3.1, z 56-60) at 28 mm and enters them from 27 mm down
# (1.5 mm at 24 mm), the hand's links within 3 mm until 32 mm (the planner's per-body margins fixed, reference.make_planner); the
# kit recheck's 20 drops with it were in fact let go at 26.4-29.2 mm, the pad on those pieces. Rows 358 / 356 (station
# +30, the same tilt, tip and pads; the kit study's and the grasp study's row ids) are clean down to 24 mm on the kit (fingers
# >= 1 mm, links >= 3 mm, the release opening and the retreat), and on the true solids the grasp study held them to 24 mm with the
# release and the retreat checked (fingers 1.01 / 1.81 / 2.42 mm). Row 358 dropped on the kit with the kit recheck's battery-drop
# trial unchanged: 19 / 20 LOCKED.
ADDED = {
    "battery": [
        _entry(358, 'primary (P2-10)', {'station_x_mm': 30.0, 'beta_deg': 10.0, 'tip_below_contact_mm': 2.0, 'right_pad': 'outboard (-y)'}, 46.48,
               [[1.0, 0.0, 0.0, 30.0], [0.0, -0.98481, 0.17365, -2.2456], [0.0, -0.17365, -0.98481, 24.67541]],
               release_height_mm=24.0),
        _entry(356, 'primary (P2-10)', {'station_x_mm': 30.0, 'beta_deg': 10.0, 'tip_below_contact_mm': 2.0, 'right_pad': 'inboard (+y)'}, 46.48,
               [[-1.0, 0.0, 0.0, 30.0], [0.0, 0.98481, 0.17365, -2.2456], [0.0, 0.17365, -0.98481, 24.67541]],
               release_height_mm=24.0),
    ],
}


# The reference program's second tier: grasps it tries only when none of a slot's `reference_grasps` is feasible in an
# episode (so every episode a first-tier grasp can do runs exactly as before). The deck: with the shelf turned the
# negative way (-0.7..-4.9 deg) and the car drawn towards the arm (stand x <= 0.611 m) or turned, the frozen yaw -10 deg
# pinches put the wrist into the car's nose at the pick and only the two frozen yaw 0 deg ones are left (4 of 38
# reference layouts, 3-7 left in 6 more); these are the other ten yaw 0 deg pinches of the kit study (stations car x
# 120-142.5, 6-9 mm in, the right pad below), re-run unchanged through the kit recheck's deck-pinch trial (a 20 s
# carry at 0.25 m/s, planner seeds 0 / 1 / 2): sag <= 0.901 deg and LOCKED in all 30 runs; on the true solids
# the grasp study walked them home with nothing within 6 mm (release and retreat not checked there, as for the frozen 118 / 136).
# Ordered by the worst sag of the three runs.
FALLBACK = {
    "deck": [
        _entry(137, 'fallback (P2-10)', {'station_x_car_mm': 127.5, 'tip_depth_mm': 9.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, -5.326], [0.0, 0.0, 1.0, -73.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.781),
        _entry(173, 'fallback (P2-10)', {'station_x_car_mm': 142.5, 'tip_depth_mm': 9.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, 9.674], [0.0, 0.0, 1.0, -73.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.797),
        _entry(119, 'fallback (P2-10)', {'station_x_car_mm': 120.0, 'tip_depth_mm': 9.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, -12.826], [0.0, 0.0, 1.0, -73.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.801),
        _entry(155, 'fallback (P2-10)', {'station_x_car_mm': 135.0, 'tip_depth_mm': 9.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, 2.174], [0.0, 0.0, 1.0, -73.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.809),
        _entry(154, 'fallback (P2-10)', {'station_x_car_mm': 135.0, 'tip_depth_mm': 7.5, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, 2.174], [0.0, 0.0, 1.0, -74.90388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.811),
        _entry(172, 'fallback (P2-10)', {'station_x_car_mm': 142.5, 'tip_depth_mm': 7.5, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, 9.674], [0.0, 0.0, 1.0, -74.90388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.818),
        _entry(117, 'fallback (P2-10)', {'station_x_car_mm': 120.0, 'tip_depth_mm': 6.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, -12.826], [0.0, 0.0, 1.0, -76.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.858),
        _entry(135, 'fallback (P2-10)', {'station_x_car_mm': 127.5, 'tip_depth_mm': 6.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, -5.326], [0.0, 0.0, 1.0, -76.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.865),
        _entry(171, 'fallback (P2-10)', {'station_x_car_mm': 142.5, 'tip_depth_mm': 6.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, 9.674], [0.0, 0.0, 1.0, -76.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.885),
        _entry(153, 'fallback (P2-10)', {'station_x_car_mm': 135.0, 'tip_depth_mm': 6.0, 'yaw_deg': 0.0, 'right_pad': 'bottom'}, 5.19,
               [[1.0, 0.0, 0.0, 2.174], [0.0, 0.0, 1.0, -76.40388], [0.0, -1.0, 0.0, -2.594]],
               carry_speed_m_s=0.25, sag_deg_max_3_seeds=0.901),
    ],
}


def fallback_grasps(slot):
    """[(G (m), width (m), desc, entry)]: the slot's second-tier grasps (`FALLBACK`)"""
    out = []
    for e in FALLBACK.get(slot, []):
        desc = "%s row %d %s (P2-10 fallback)" % (slot, e["row"], ", ".join("%s %s" % (k, v) for k, v in e["params"].items()))
        out.append((as_m(e["G_mm"]), e["width_mm"] * 1e-3, desc, e))
    return out


def reference_grasps(slot):
    """[(G (m), width (m), desc, entry)]: the reference program's candidates of a slot, best first: the
    added grasps (`ADDED`), then the frozen primaries (`frozen_grasps`)"""
    out = []
    for e in ADDED.get(slot, []):
        desc = "%s row %d %s (P2-10)" % (slot, e["row"], ", ".join("%s %s" % (k, v) for k, v in e["params"].items()))
        out.append((as_m(e["G_mm"]), e["width_mm"] * 1e-3, desc, e))
    seen = set(e["row"] for e in ADDED.get(slot, []))
    out += [g for g in frozen_grasps(slot) if g[3]["row"] not in seen]
    return out


def frozen_entries(slot, roles=("primary",)):
    """the frozen entries of a slot with one of `roles` ("primary" matches "primary (raw release / retreat not
    checked)" too), in the table's order (best first)"""
    out = []
    for e in FROZEN[slot]:
        if any(e["role"] == r or e["role"].startswith(r + " ") for r in roles):
            out.append(e)
    return out


def frozen_grasps(slot, roles=("primary",)):
    """[(G (m, 4 x 4), width (m), desc, entry)] of a slot's frozen grasps (`frozen_entries`)"""
    out = []
    for e in frozen_entries(slot, roles):
        desc = "%s row %d %s" % (slot, e["row"], ", ".join("%s %s" % (k, v) for k, v in e["params"].items()))
        out.append((as_m(e["G_mm"]), e["width_mm"] * 1e-3, desc, e))
    return out


# ------------------------------------------------------------------ the kit's collision geoms as surface samples
def _stl_vertices(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n == len(data):
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        return np.unique(rec["v"].reshape(-1, 3).astype(np.float64), axis=0)
    pts = [[float(x) for x in ln.split()[1:4]] for ln in data.decode("ascii", "replace").splitlines() if ln.strip().startswith("vertex")]
    return np.unique(np.asarray(pts, np.float64).reshape(-1, 3), axis=0)


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def piece_hull(g, kit_dir, cyl_sag_mm=0.01):
    """(V mm, triangles (k, 3), planes (k, 4): n.x <= d inside) of one kit collision geom in its body's frame: a mesh
    piece's convex hull (MuJoCo collides with it), a box, a cylinder as its circumscribed prism of sag <= cyl_sag_mm"""
    from scipy.spatial import ConvexHull
    t = g["type"]
    if t == "mesh":
        V = _stl_vertices(os.path.join(kit_dir, g["file"])) * 1e3
    elif t == "box":
        h = np.asarray(g["size"], float) * 1e3
        V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h
    elif t == "cylinder":
        r, hl = float(g["size"][0]) * 1e3, float(g["size"][1]) * 1e3
        n = max(12, int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - cyl_sag_mm / max(r, 1e-6))))))
        rc = r / math.cos(math.pi / n)
        a = 2 * math.pi * (np.arange(n) + 0.5) / n
        ring = np.stack([rc * np.cos(a), rc * np.sin(a)], 1)
        V = np.concatenate([np.column_stack([ring, np.full(n, -hl)]), np.column_stack([ring, np.full(n, hl)])])
    else:
        raise KeyError(t)
    V = V @ _quat_R(g.get("quat", [1, 0, 0, 0])).T + np.asarray(g.get("pos", [0, 0, 0]), float) * 1e3
    hull = ConvexHull(V)
    planes = np.hstack([hull.equations[:, :3], -hull.equations[:, 3:4]])
    return V, hull.simplices.astype(np.int64), planes


_BARY = {}


def _bary(n):
    """the barycentric grid (u, v) of a triangle split n times per edge (vertices and edges included)"""
    if n not in _BARY:
        u, v = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        m = (u + v) <= n
        _BARY[n] = (u[m] / n, v[m] / n)
    return _BARY[n]


def _tri_samples(V, F, h):
    """points on triangles: a barycentric grid of spacing <= h on each (vertices and edges included), and the
    triangle's index (the triangles grouped by their split count)"""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    L = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], 1), 1)
    ns = np.maximum(1, np.ceil(L / h).astype(np.int64))
    pts, idx = [], []
    for n in np.unique(ns):
        k = np.where(ns == n)[0]
        u, v = _bary(int(n))
        P = a[k][:, None, :] + u[None, :, None] * (b[k] - a[k])[:, None, :] + v[None, :, None] * (c[k] - a[k])[:, None, :]
        pts.append(P.reshape(-1, 3))
        idx.append(np.repeat(k, len(u)))
    return np.concatenate(pts), np.concatenate(idx)


class UnitSurface(object):
    """a loose unit's surface samples (mm, its body frame) with the outward normals of their faces: P, N, hi, lo, com"""

    def __init__(self, P, N, com):
        self.P, self.N = P, N
        self.com = np.asarray(com, float)
        self.lo, self.hi = P.min(0), P.max(0)


_SURFACES = {}


def unit_surface(kit, unit, h=0.5, thin=0.5, roles=("col",)):
    """`UnitSurface` of a loose unit from the kit's collision geoms of its bodies (placed by their T_design chain into
    the unit's root frame): every piece's hull faces sampled at <= h mm, the samples strictly inside another piece
    dropped (the union's surface; only pieces whose boxes overlap are tested), one sample per `thin` mm voxel. Cached
    per (kit directory, unit, h, thin)."""
    key = (kit.get("dir"), unit, float(h), float(thin), tuple(roles))
    if key in _SURFACES:
        return _SURFACES[key]
    B = kit["bodies"]
    kd = kit.get("dir") or "."
    Tui = np.linalg.inv(np.asarray(B[unit]["T_design"], float))
    hulls = []
    for bn, bb in B.items():
        if bb.get("unit") != unit:
            continue
        Tb = Tui @ np.asarray(bb["T_design"], float)
        Tb[:3, 3] *= 1e3
        for g in bb.get("collision", []):
            if g.get("role", "col") not in roles:
                continue
            V, F, planes = piece_hull(g, kd)
            V = V @ Tb[:3, :3].T + Tb[:3, 3]
            nrm = planes[:, :3] @ Tb[:3, :3].T
            planes = np.hstack([nrm, (planes[:, 3] + nrm @ Tb[:3, 3])[:, None]])
            hulls.append((V, F, planes, V.min(0), V.max(0)))
    Ps, Ns = [], []
    for i, (V, F, planes, lo, hi) in enumerate(hulls):
        P, ti = _tri_samples(V, F, h)
        e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
        nf = np.cross(e1, e2)
        ln = np.linalg.norm(nf, axis=1)
        ln[ln == 0] = 1.0
        nf = nf / ln[:, None]
        cen = V.mean(0)
        flip = np.einsum("ij,ij->i", nf, V[F[:, 0]] - cen) < 0
        nf[flip] *= -1.0
        N = nf[ti]
        keep = np.ones(len(P), bool)
        for j, (_V2, _F2, pl2, lo2, hi2) in enumerate(hulls):
            if j == i or np.any(lo2 > hi + 1e-3) or np.any(hi2 < lo - 1e-3):
                continue
            cand = np.where(keep & np.all((P > lo2 - 1e-3) & (P < hi2 + 1e-3), axis=1))[0]
            if not len(cand):
                continue
            inside = np.all(P[cand] @ pl2[:, :3].T - pl2[:, 3] < -1e-3, axis=1)
            keep[cand[inside]] = False
        Ps.append(P[keep])
        Ns.append(N[keep])
    P, N = np.concatenate(Ps), np.concatenate(Ns)
    if thin:
        k = np.floor(P / thin).astype(np.int64)
        _, ui = np.unique(k, axis=0, return_index=True)
        ui = np.sort(ui)
        P, N = P[ui], N[ui]
    com = np.asarray(B[unit].get("com_m", [0, 0, 0]), float) * 1e3
    _SURFACES[key] = UnitSurface(P, N, com)
    return _SURFACES[key]


# ------------------------------------------------------------------ the pad-closing analysis
def pad_faces_tcp(width_m):
    """the two pads' inner faces in the tcp frame (mm) at an opening, from the 2F-85's opening table (faces parallel to
    the approach: the 4-bar's 0-1.8 deg convergence left out): {side: (centre, u_across, v_along, half_u, half_v)}"""
    w = float(width_m) * 1e3
    zc = pad_centre_z(float(width_m)) * 1e3
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    hu, hv = PAD_ACROSS_M * 1e3 / 2, PAD_ALONG_M * 1e3 / 2
    return {"right": (np.array([0.0, w / 2, zc]), u, v, hu, hv), "left": (np.array([0.0, -w / 2, zc]), u, v, hu, hv)}


def analyse(data, G, width_guess_mm, iters=4, faces=pad_faces_tcp):
    """close the pads of the hand at G (mm, unit frame) on the unit's samples (`UnitSurface`); returns {"width_mm",
    "sides": {right, left: n, across_mm, along_mm, normal_dev_deg, contact_y_mm, tip_to_patch_mm}, "overlap_mm",
    "com_offset_mm", "ok"} or None (no sample under a pad, or the opening out of the 2F-85's range)"""
    Gi = inv(G)
    L = data.P @ Gi[:3, :3].T + Gi[:3, 3]                    # samples in the tcp frame
    Nl = data.N @ Gi[:3, :3].T
    w = float(width_guess_mm)
    res = None
    for _ in range(iters):
        pf = faces(max(w, 0.9) * 1e-3)
        sides = {}
        for side, sgn in (("right", 1.0), ("left", -1.0)):
            ctr, u, v, hu, hv = pf[side]
            du = (L - ctr) @ u
            dv = (L - ctr) @ v
            fp = (np.abs(du) <= hu) & (np.abs(dv) <= hv)
            if not fp.any():
                return None
            y = L[fp, 1]
            ext = float(y.max()) if sgn > 0 else float(y.min())
            near = np.abs(y - ext) <= PATCH_TOL
            idx = np.where(fp)[0][near]
            al = np.abs(Nl[idx][:, 1])
            sides[side] = {"ext": ext, "idx": idx, "du": du[idx], "dv": dv[idx], "align": float(np.median(al)),
                           "v_rel_tip": dv[idx] - hv}
        w_new = sides["right"]["ext"] - sides["left"]["ext"]
        res = (w_new, sides)
        if abs(w_new - w) < 0.05:
            w = w_new
            break
        w = w_new
    w, sides = res
    if w <= 0.5 or w > MAX_WIDTH_M * 1e3 - 1.0:
        return None
    out = {"width_mm": w, "sides": {}}
    ok = True
    for side, s in sides.items():
        du, dv = s["du"], s["dv"]
        ext_u = float(du.max() - du.min()) if len(du) else 0.0
        ext_v = float(dv.max() - dv.min()) if len(dv) else 0.0
        ang = math.degrees(math.acos(min(1.0, s["align"])))
        out["sides"][side] = {"n": int(len(du)), "across_mm": ext_u, "along_mm": ext_v, "normal_dev_deg": ang,
                              "contact_y_mm": s["ext"], "tip_to_patch_mm": float(-s["v_rel_tip"].max()) if len(dv) else None}
        ok &= ext_u >= MIN_PATCH_ACROSS and ext_v >= MIN_PATCH_ALONG and ang <= ALIGN_MIN_DEG
    pr = L[sides["right"]["idx"]][:, [0, 2]]
    pl = L[sides["left"]["idx"]][:, [0, 2]]
    lo = np.maximum(pr.min(0), pl.min(0))
    hi = np.minimum(pr.max(0), pl.max(0))
    ov = np.maximum(hi - lo, 0.0)
    out["overlap_mm"] = [float(ov[0]), float(ov[1])]
    ok &= ov[0] >= 3.0 and ov[1] >= 2.0
    com_l = Gi[:3, :3] @ data.com + Gi[:3, 3]
    cc = 0.5 * (pr.mean(0) + pl.mean(0))
    out["com_offset_mm"] = float(np.hypot(com_l[0] - cc[0], com_l[2] - cc[1]))
    out["ok"] = bool(ok)
    return out


FAMILY_DEFAULTS = {
    # the grasp study's dense grids and the families' width guesses
    "BAT": {"kw": {"stations": np.arange(-55.0, 55.1, 5.0), "betas": (-10.0, -5.0, 0.0, 5.0, 10.0), "tips": (2.0, 4.0)}, "wg": 46.4},
    "DCK": {"kw": {"stations_car": np.arange(75.0, 150.1, 7.5), "depths": (6.0, 7.5, 9.0), "yaws": (-10.0, 0.0, 10.0)}, "wg": 5.19},
    "CMP": {"kw": {"thetas": range(0, 180, 15), "tip_depths": (3.0, 4.5, 6.0), "offsets": (-8.0, 0.0, 8.0), "width_guess": 34.6},
            "wg": 34.6},
    "LDR": {"kw": {"thetas": range(0, 180, 45), "tip_depths": (13.0, 16.0, 40.0, 44.0), "offsets": (-8.0, 0.0, 8.0),
                   "width_guess": 40.8}, "wg": 40.8},
}


def family_candidates(kit, unit, data=None, require_ok=True, limit=None):
    """[(G (m), width (m), desc, analysis)] of a top-down unit's family whose pads close on the kit's collision pieces
    with good patches (`analyse` ok), in the family's order; the wheels' families per corner are `wheel_grasps_unit`"""
    fam = kit["bodies"][unit].get("type", unit)
    data = data if data is not None else unit_surface(kit, unit)
    d = FAMILY_DEFAULTS[fam]
    if fam == "BAT":
        raw = battery_grasps(**d["kw"])
    elif fam == "DCK":
        raw = deck_grasps(**d["kw"])
    else:
        raw = topdown_grasps(data, **d["kw"])
    out = []
    for p, G in raw:
        a = analyse(data, G, d["wg"])
        if a is None or (require_ok and not a["ok"]):
            continue
        desc = "%s family %s" % (unit, ", ".join("%s %s" % (k, v) for k, v in p.items()))
        out.append((as_m(G), a["width_mm"] * 1e-3, desc, dict(a, params=p)))
        if limit and len(out) >= limit:
            break
    return out
