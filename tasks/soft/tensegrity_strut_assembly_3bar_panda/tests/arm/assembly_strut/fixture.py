"""The cradle of the strut assembly cell (workcell equipment): a steel plate on the bench with up to
three clamp seats `seat_spacing_m` apart (100 mm; the planner's fallback is 130 mm); each seat clamps only the lower ring of a strut core's electronics bay base (about
7 mm: from the bay's lowest point up to 8 mm below the rod axis along the lid's seat direction, below the lid's lower
edge), with the rod axis horizontal along the cell's y, 75 mm above the bench, the lid's seat direction straight up.
The clamp touches neither the rod nor a motor module (the seats span the bay's length less 6 mm at each end).

In the cell the strut core (M0) is a fixed body (no joint) at its seat: "only the base on the fixture" of the
precedent; M0 and the cradle are both static, so MuJoCo makes no contacts between them.

    cradle_boxes(cfg, struts)            [(name, centre, half sizes)] in the cradle frame
    core_pose_in_cradle(cfg, strut)      the strut core's 4x4 pose (m) in the cradle frame
    board_pose_in_cradle(cfg)            the cradle board's pose (lying on the plate's +x edge, facing up)
    seat_x(cfg, strut)                   a seat's x in the cradle frame (m)

Cradle frame: origin on the bench top under the plate's centre, axes = the cell's (the cradle is moved per episode,
never turned).
"""
import math

import numpy as np

DEFAULT_FIXTURE = {
    # the plate (x across the struts, y along them) and its nominal place: the middle seat's rod axis at x = +0.01 m
    # (0.46 m from the Panda's base axis at (-0.45, 0))
    "plate": [0.36, 0.18, 0.020],
    "nominal_xy": [0.01, 0.0],
    "jitter_m": 0.010,
    # the seats' x in the cradle frame: s2 in the middle, s1 / s3 one spacing towards / away from the robot
    "seat_spacing_m": 0.10,
    "rod_z": 0.075,
    # the strut core's seat direction (the lid's extraction, azimuth in the core's xy plane) points up: kit graph's
    # seat_azimuth_deg; the core's +z (the p end) along the cell's +y
    "seat_azimuth_deg": 120.0,
    # the bay base in the core frame, measured on its collision pieces (u along the seat direction, w = z x u, mm):
    # u -14.93 .. +9.71, w -13.49 .. +19.88, z -71.5 .. +71.5
    "bay_u_min_mm": -14.93,
    "bay_w_mm": [-13.49, 19.88],
    "bay_half_len_mm": 71.5,
    # the clamp: a saddle 0.5 mm under the bay's lowest point, two jaws 0.5 mm off its sides up to u = -8 mm (the lid's
    # lower edge is at u = -6.79 mm), 6 mm thick, 6 mm short of the bay's ends
    "clamp_top_u_mm": -8.0,
    "clamp_gap_mm": 0.5,
    "jaw_t_mm": 6.0,
    "seat_end_margin_mm": 6.0,
    # the cradle board: 5 x 3 squares of 10 mm, 4 mm margin, flat on the plate's +x edge; its columns along the cell's y
    # (yaw 90 deg: 38 mm across x, 133-171 mm from the middle seat; with its columns along x, 58 mm, it reached under the
    # third seat's outer jaw, whose edge is 126.4 mm from the middle seat)
    "board": {"cols": 5, "rows": 3, "square_mm": 10.0, "margin_mm": 4.0, "thickness_mm": 2.0, "x_mm": 152.0, "y_mm": 0.0, "yaw_deg": 90.0},
}


def merged(cfg=None):
    out = {k: (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)) for k, v in DEFAULT_FIXTURE.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def seat_frame(azimuth_deg):
    """R (3x3) taking core-frame vectors to cradle-frame vectors: core z -> +y, the seat direction u -> +z"""
    a = math.radians(float(azimuth_deg))
    u = np.array([math.cos(a), math.sin(a), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    A = np.stack([z, u, np.cross(z, u)], 1)
    B = np.stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], np.cross([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])], 1)
    return B @ np.linalg.inv(A)


def seat_x(cfg, strut):
    """the seat's x in the cradle frame (m): an explicit `seat_x` {strut: x} when given, else -1 / 0 / +1 x the
    spacing for s1 / s2 / s3"""
    c = cfg if (isinstance(cfg, dict) and "rod_z" in cfg) else merged(cfg)
    if c.get("seat_x"):
        return float(c["seat_x"][strut])
    return {"s1": -1.0, "s2": 0.0, "s3": 1.0}[strut] * float(c["seat_spacing_m"])


def core_pose_in_cradle(cfg, strut):
    c = merged(cfg)
    T = np.eye(4)
    T[:3, :3] = seat_frame(c["seat_azimuth_deg"])
    T[:3, 3] = [seat_x(c, strut), 0.0, float(c["rod_z"])]
    return T


def cradle_boxes(cfg=None, struts=("s2",)):
    """the plate and, per strut, the saddle and the two jaws: [(name, centre m, half sizes m)] in the cradle frame
    (the core's w axis = the cell's +x, its u = +z)"""
    c = merged(cfg)
    px, py, pt = (float(v) for v in c["plate"])
    out = [("plate", [0.0, 0.0, pt / 2], [px / 2, py / 2, pt / 2])]
    mm = 1e-3
    zr = float(c["rod_z"])
    top = zr + float(c["clamp_top_u_mm"]) * mm
    bottom_bay = zr + float(c["bay_u_min_mm"]) * mm
    g = float(c["clamp_gap_mm"]) * mm
    jt = float(c["jaw_t_mm"]) * mm
    hl = (float(c["bay_half_len_mm"]) - float(c["seat_end_margin_mm"])) * mm
    w0, w1 = (float(v) * mm for v in c["bay_w_mm"])
    for s in struts:
        x = seat_x(c, s)
        sad_top = bottom_bay - g
        out.append(("%s_saddle" % s, [x + 0.5 * (w0 + w1), 0.0, 0.5 * (pt + sad_top)], [0.5 * (w1 - w0) + g + jt, hl, 0.5 * (sad_top - pt)]))
        for side, wx in (("a", w0 - g - jt / 2), ("b", w1 + g + jt / 2)):
            out.append(("%s_jaw_%s" % (s, side), [x + wx, 0.0, 0.5 * (sad_top + top)], [jt / 2, hl, 0.5 * (top - sad_top)]))
    return out


def board_pose_in_cradle(cfg=None):
    c = merged(cfg)
    b = c["board"]
    T = np.eye(4)
    a = math.radians(float(b.get("yaw_deg", 0.0)))
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [float(b["x_mm"]) * 1e-3, float(b["y_mm"]) * 1e-3, float(c["plate"][2])]
    return T


def cradle_xy(seed, cfg=None, nominal=False):
    """the cradle's place for the episode: nominal +- jitter_m per axis, never turned"""
    c = merged(cfg)
    x, y = (float(v) for v in c["nominal_xy"])
    if nominal or float(c["jitter_m"]) <= 0.0:
        return [x, y]
    r = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 0xF1C7])
    j = float(c["jitter_m"])
    return [x + float(r.uniform(-j, j)), y + float(r.uniform(-j, j))]
