"""The parts trays of the strut assembly cell (workcell equipment): aluminium plates with nests that
give the round modules a definite resting pose, and the resting pose of a module in its nest computed from the
module's collision pieces (the convex hulls MuJoCo collides), not from a bounding box.

Tray sets per scope (`trays_of(cfg, scope)`):
  single   one tray `tray_l` (the minimum version): five nests in a row 70 mm apart;
  default  two trays (the three-strut cell): `tray_l` (left of the robot, +y: three motor modules, three bay lids,
           three end caps) and `tray_r` (right, -y, turned 180 deg about the vertical so that its -y also faces the
           cradle: three motor modules, three end caps). The Panda's hand spans +-0.104 m along its closing axis, 37 mm
           above the tcp, and every module is pinched across its bore, so the hand's long side lies across a row of
           side-by-side nests; a motor module is picked with the hand tilted 6-16 deg about the module's axis, which
           lowers one end of the hand over its neighbours. So the motor modules lie on one line (tray y = -0.025 m,
           the leads' end towards the cradle) 0.15 m apart, the end caps on a second line 0.08 m further out along the
           modules' axes (tray y = +0.055 m), between the motor modules in x (a hand over a motor module never
           reaches the caps' line and the other way round), and the three lids side by side 60 mm apart at the left
           tray's back end, 0.075 m from the first end cap and 0.15 m from the first motor module (a lid is picked
           with the hand's underside 53 mm above the plate: a motor module, 50 mm high, must stay outside the
           hand's span).

Nests (tray frame: origin on the plate's top face at its centre; x across, y along the modules' axes, z up; a module's
bore axis points along +y, its inner mouth, the battery leads' end, towards -y = towards the cradle; a nest row is
(name, kind, x[, y]) in m, y = 0 when omitted):
  motor  two 45 deg V seats under the motor module's ends (module z 79-89 and 119-129: the fingers grip between them)
         and an end stop 0.3 mm off the outer end face (module z 131.4), below the axis; the battery leads overhang the
         inner V seat; the module rests on the V seats at the nest's roll (the nest does not hold the design roll)
  cap    a V seat under the end cap's dome tilted back by 25 deg (the dome is a sphere of r 17.5 about module z 145 and
         the centre of mass lies 5.7 mm beyond it: the tilted seat pushes the cap back) and a rear stop against the step
         face at module z 138.65 (the annulus r 4.8-13.3 below the axis): the stop's reaction 0.66 x the weight acting
         about 8.6 mm below the axis holds the axis horizontal; a 45 deg V guard 0.5 mm under the rear cylinder (r 13.3)
         keeps the cap from rocking sideways and does not touch it at rest
  lid    a shallow pocket (4 mm walls, 1.5 mm clear of the lid's footprint); the lid on its rim, outer face up, in the
         stable rest of its collision hull nearest to "seat direction up"

    trays_of(cfg, scope)                      {tray: {"plate": [x, y, t] m, "nests": [(name, kind, x, y)]}}
    tray_geoms(cfg, contents, plate)          [(name, kind, centre, half sizes, quat wxyz)] in the tray frame
    nest_rest(kit, cfg, nest, build, roll_rad, scope=...)   the module's pose (4x4, m) in its tray's frame, at rest
"""
import math

import numpy as np

S45 = math.sqrt(0.5)

DEFAULT_TRAY = {
    "plate": [0.34, 0.18, 0.010],
    # the single-strut tray: two motor modules, a lid and two end caps; x of each nest in the tray frame (m): 70 mm
    # apart (the grasp trials: the fingers opened for a motor module reach 36 mm off its axis; 60 mm let them press
    # the neighbouring end cap)
    "nests_single": [["motor_a", "motor", -0.140], ["cap_a", "cap", -0.070], ["lid", "lid", 0.0], ["cap_b", "cap", 0.070],
                     ["motor_b", "motor", 0.140]],
    # the three-strut cell's two trays (module docstring): the motor line at y -0.025 (its battery leads reach -0.0885
    # over the plate), the cap line at +0.055, the lids centred at y 0 (their pockets +-77 mm); every motor module 0.15
    # m from the next, every end cap 0.075 m from its motor modules in x and 0.08 m off their line in y
    "trays_default": {
        "tray_l": {"plate": [0.66, 0.18, 0.010],
                   "nests": [["l_lid1", "lid", -0.285, 0.0], ["l_lid2", "lid", -0.225, 0.0], ["l_lid3", "lid", -0.165, 0.0],
                             ["l_cap1", "cap", -0.090, 0.055], ["l_motor1", "motor", -0.015, -0.025], ["l_cap2", "cap", 0.060, 0.055],
                             ["l_motor2", "motor", 0.135, -0.025], ["l_cap3", "cap", 0.210, 0.055], ["l_motor3", "motor", 0.285, -0.025]]},
        "tray_r": {"plate": [0.46, 0.18, 0.010],
                   "nests": [["r_motor1", "motor", -0.1875, -0.025], ["r_cap1", "cap", -0.1125, 0.055], ["r_motor2", "motor", -0.0375, -0.025],
                             ["r_cap2", "cap", 0.0375, 0.055], ["r_motor3", "motor", 0.1125, -0.025], ["r_cap3", "cap", 0.1875, 0.055]]},
    },
    "plate_t_mm": 4.0,
    "motor": {"axis_h_mm": 30.0, "centre_z_mm": 103.9, "seats_z_mm": [[79.0, 89.0], [119.0, 129.0]], "stop_z_mm": 131.4,
              "stop_gap_mm": 0.3, "stop_top_below_axis_mm": 8.0, "r_mm": 20.0},
    "cap": {"axis_h_mm": 26.0, "centre_z_mm": 147.0, "rear_z_mm": [139.5, 144.5], "rear_r_mm": 13.29, "rear_guard_mm": 0.5,
            "dome_tilt_deg": 25.0, "dome_centre_z_mm": 145.0, "dome_r_mm": 17.5, "dome_span_mm": 6.0, "step_z_mm": 138.65,
            "stop_top_below_axis_mm": 6.0},
    "lid": {"wall_h_mm": 4.0, "wall_t_mm": 4.0, "clear_mm": 1.5},
}


def merged(cfg=None):
    out = {k: (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)) for k, v in DEFAULT_TRAY.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _quat_from_R(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def rot_axis(axis, ang):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(a / np.linalg.norm(a) * float(ang)).as_matrix()


def _plate_box(name, q, n, along, t, a_down, b_up, half_along):
    """a box whose top face lies on the plane through q with normal n: `along` = an in-plane direction (the slab's
    axis), the plate spans a_down below q and b_up above it along the in-plane slope direction; returns the geom row"""
    n = np.asarray(n, float) / np.linalg.norm(n)
    e1 = np.asarray(along, float) - n * float(np.dot(along, n))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if e2[2] < 0:                 # up-slope
        e2 = -e2
        e1 = np.cross(e2, n)
    R = np.stack([e1, e2, n], 1)
    c = np.asarray(q, float) + e2 * 0.5 * (b_up - a_down) - n * t / 2
    return (name, "box", c.tolist(), [half_along, 0.5 * (a_down + b_up), t / 2], _quat_from_R(R).tolist())


def _rows(nests):
    """[(name, kind, x, y)] of a nest list whose rows are (name, kind, x) or (name, kind, x, y)"""
    return [(str(r[0]), str(r[1]), float(r[2]), float(r[3]) if len(r) > 3 else 0.0) for r in nests]


def trays_of(cfg=None, scope="single"):
    """{tray: {"plate": [x, y, t], "nests": [(name, kind, x, y)]}} of the scope (module docstring), trays sorted by name"""
    c = merged(cfg)
    if scope == "single":
        return {"tray_l": {"plate": [float(v) for v in c["plate"]], "nests": _rows(c["nests_single"])}}
    out = {}
    for tn in sorted(c["trays_" + scope]):
        t = c["trays_" + scope][tn]
        out[tn] = {"plate": [float(v) for v in t["plate"]], "nests": _rows(t["nests"])}
    return out


def nest_frame_x(cfg, nest, scope=None):
    """(kind, x) of a nest in its tray's frame (any scope's trays when scope is None)"""
    k, x, _y, _t = nest_place(cfg, nest, scope)
    return k, x


def nest_place(cfg, nest, scope=None):
    """(kind, x, y, tray) of a nest"""
    for sc in ((scope,) if scope else ("single", "default")):
        for tn, t in trays_of(cfg, sc).items():
            for n, kind, x, y in t["nests"]:
                if n == nest:
                    return kind, x, y, tn
    raise KeyError(nest)


def tray_geoms(cfg=None, contents=None, plate=None):
    """the plate and every nest's plates (tray frame, m); `contents` = [(nest, kind, x[, y])] (default: the single
    tray), `plate` = [x, y, t] (default: the single tray's)"""
    c = merged(cfg)
    mm = 1e-3
    px, py, pt = (float(v) for v in (plate or c["plate"]))
    out = [("plate", "box", [0.0, 0.0, -pt / 2], [px / 2, py / 2, pt / 2], [1.0, 0.0, 0.0, 0.0])]
    t = float(c["plate_t_mm"]) * mm
    for nest, kind, x, y_n in _rows(contents or c["nests_single"]):
        x = float(x)
        n0 = len(out)
        if kind == "motor":
            mc = c["motor"]
            h, r, zc = float(mc["axis_h_mm"]) * mm, float(mc["r_mm"]) * mm, float(mc["centre_z_mm"]) * mm
            for k, (z0, z1) in enumerate(mc["seats_z_mm"]):
                y0, y1 = float(z0) * mm - zc, float(z1) * mm - zc
                for side, sx in (("l", -1.0), ("r", 1.0)):
                    nrm = np.array([-sx * S45, 0.0, S45])          # the plate on side sx faces the axis
                    q = np.array([x + sx * r * S45, 0.5 * (y0 + y1), h - r * S45])
                    out.append(_plate_box("%s_v%d%s" % (nest, k, side), q, nrm, [0, 1, 0], t, 0.022, 0.004, 0.5 * (y1 - y0)))
            ys = float(mc["stop_z_mm"]) * mm - zc + float(mc["stop_gap_mm"]) * mm
            top = h - float(mc["stop_top_below_axis_mm"]) * mm
            out.append(("%s_stop" % nest, "box", [x, ys + t / 2, top / 2], [0.012, t / 2, top / 2], [1.0, 0.0, 0.0, 0.0]))
        elif kind == "cap":
            cc = c["cap"]
            h, zc = float(cc["axis_h_mm"]) * mm, float(cc["centre_z_mm"]) * mm
            r0 = (float(cc["rear_r_mm"]) + float(cc["rear_guard_mm"])) * mm
            y0, y1 = float(cc["rear_z_mm"][0]) * mm - zc, float(cc["rear_z_mm"][1]) * mm - zc
            for side, sx in (("l", -1.0), ("r", 1.0)):
                nrm = np.array([-sx * S45, 0.0, S45])
                q = np.array([x + sx * r0 * S45, 0.5 * (y0 + y1), h - r0 * S45])
                out.append(_plate_box("%s_rear%s" % (nest, side), q, nrm, [0, 1, 0], t, 0.024, 0.003, 0.5 * (y1 - y0)))
            b = math.radians(float(cc["dome_tilt_deg"]))
            R = float(cc["dome_r_mm"]) * mm
            yc = float(cc["dome_centre_z_mm"]) * mm - zc
            for side, sx in (("l", -1.0), ("r", 1.0)):
                nrm = math.cos(b) * np.array([-sx * S45, 0.0, S45]) + math.sin(b) * np.array([0.0, -1.0, 0.0])
                q = np.array([x, yc, h]) - R * nrm
                # the plate's slab axis: the projection of +y into the tilted plane
                out.append(_plate_box("%s_dome%s" % (nest, side), q, nrm, [0, 1, 0], t, 0.024, 0.003, 0.5 * float(cc["dome_span_mm"]) * mm))
            ys = float(cc["step_z_mm"]) * mm - zc
            top = h - float(cc["stop_top_below_axis_mm"]) * mm
            out.append(("%s_stop" % nest, "box", [x, ys - t / 2, top / 2], [0.010, t / 2, top / 2], [1.0, 0.0, 0.0, 0.0]))
        elif kind == "lid":
            lc = c["lid"]
            wh, wt, cl = float(lc["wall_h_mm"]) * mm, float(lc["wall_t_mm"]) * mm, float(lc["clear_mm"]) * mm
            # the lid's footprint in the tray (outer face up): across x = the core's w (-14.08 .. 19.47 mm), along y =
            # the core's z (-71.5 .. 71.5 mm); flipped lids turn about the nest centre: the pocket is symmetric about it
            hx, hy = 0.5 * (19.47 + 14.08) * mm + cl, 71.5 * mm + cl
            for k, (cx_, cy_, sx_, sy_) in enumerate([(hx + wt / 2, 0.0, wt / 2, hy + wt), (-(hx + wt / 2), 0.0, wt / 2, hy + wt),
                                                      (0.0, hy + wt / 2, hx, wt / 2), (0.0, -(hy + wt / 2), hx, wt / 2)]):
                out.append(("%s_wall%d" % (nest, k), "box", [x + cx_, cy_, wh / 2], [sx_, sy_, wh / 2], [1.0, 0.0, 0.0, 0.0]))
        if y_n:
            # the nest's own line in the tray (the three-strut trays: the motor line and the cap line)
            out[n0:] = [(nm, kd, [c_[0], c_[1] + y_n, c_[2]], hs, q) for nm, kd, c_, hs, q in out[n0:]]
    return out


# ------------------------------------------------------------------ resting poses from the collision hulls
def _hull_edges(V):
    from scipy.spatial import ConvexHull
    try:
        H = ConvexHull(V)
    except Exception:  # noqa: BLE001 - a flat piece: all its vertices
        return V, np.zeros((0, 2), int)
    e = set()
    for s in H.simplices:
        for i in range(3):
            a, b = int(s[i]), int(s[(i + 1) % 3])
            e.add((min(a, b), max(a, b)))
    return V, np.array(sorted(e), int)


class HullSet(object):
    """the build's collision hulls (module frame, m) with their edges, for slab-clipped support functions"""

    def __init__(self, kit, build):
        self.hulls = []
        for V, _pid, _k in kit.collision_hulls(build):
            Vm = np.asarray(V, float) * 1e-3
            self.hulls.append(_hull_edges(Vm))

    def points(self, R, p, axis, lo, hi):
        """the vertices of every hull placed at (R, p), clipped to the slab lo <= x . axis <= hi (the hull vertices
        inside plus the edges' crossings of the two planes): the support of a clipped polytope is one of them"""
        axis = np.asarray(axis, float)
        out = []
        for V, E in self.hulls:
            P = V @ np.asarray(R, float).T + np.asarray(p, float)
            s = P @ axis
            inside = (s >= lo) & (s <= hi)
            if inside.any():
                out.append(P[inside])
            if len(E):
                sa, sb = s[E[:, 0]], s[E[:, 1]]
                for lv in (lo, hi):
                    cross = (sa - lv) * (sb - lv) < 0
                    if cross.any():
                        t = ((lv - sa[cross]) / (sb[cross] - sa[cross]))[:, None]
                        out.append(P[E[cross, 0]] + t * (P[E[cross, 1]] - P[E[cross, 0]]))
        return np.concatenate(out) if out else np.zeros((0, 3))

    def all_points(self, R, p):
        return np.concatenate([V @ np.asarray(R, float).T + np.asarray(p, float) for V, _E in self.hulls])


def _plane_of(row):
    """(n, d, slab axis, lo, hi) of a plate geom row: its top face n . x = d over the plate's extent along its local x"""
    _name, _k, c, hs, q = row
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    R = R.reshape(3, 3)
    n = R[:, 2]
    top = np.asarray(c, float) + n * hs[2]
    ax = R[:, 0]
    s0 = float(top @ ax)
    return n, float(n @ top), ax, s0 - hs[0], s0 + hs[0]


def module_orientation(kind, roll_rad, flip=False):
    """the module's orientation in the tray frame: its bore axis (module +z) along tray +y, turned by `roll_rad` about
    it (motor, cap); the lid: the core frame's seat direction up, its z along the tray's y (turned 180 deg about the
    vertical when `flip`)"""
    from .fixture import seat_frame
    if kind == "lid":
        R = seat_frame(120.0)
        return (rot_axis([0, 0, 1], math.pi) @ R) if flip else R
    # module z -> tray +y, module x -> tray +x at roll 0
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return rot_axis([0, 1, 0], roll_rad) @ R0


def nest_rest(kit, cfg, nest, build, roll_rad=0.0, flip=False, hullsets=None, iters=40, scope=None):
    """the module's pose (4x4, m) in its tray's frame at rest in `nest`, computed from its collision hulls: every
    support plate of the nest touching (Newton on the four plate gaps over the axis point's x / z and the axis's pitch /
    yaw; the clipped hull supports), the cap against its rear stop, the motor module's outer face `stop_gap_mm` off its
    stop; returns (T, info with the residual gaps in um)"""
    c = merged(cfg)
    kind, x, y_n, _tray = nest_place(cfg, nest, scope)
    hs = (hullsets or {}).get(build) or HullSet(kit, build)
    rows = [r for r in tray_geoms(cfg, [(nest, kind, x, y_n)]) if r[0].startswith(nest + "_")]
    R0 = module_orientation(kind, roll_rad, flip)
    mm = 1e-3
    if kind == "lid":
        return _lid_rest(kit, c, x, R0, hs, y=y_n)
    zc = float(c[kind]["centre_z_mm"]) * mm
    p0 = np.array([x, y_n - zc, float(c[kind]["axis_h_mm"]) * mm])     # module z = zc at the nest's y; the axis at axis_h
    planes = [(r[0], _plane_of(r)) for r in rows if ("_v" in r[0] or "_rear" in r[0] or "_dome" in r[0])]
    # turn about the module's point on its axis at its rear support (cap: between the rear seat and the dome's centre)
    pivot_m = np.array([0.0, 0.0, zc if kind == "motor" else 0.5 * (float(c["cap"]["rear_z_mm"][0]) + float(c["cap"]["rear_z_mm"][1])) * mm])

    def pose(q):
        R = rot_axis([1, 0, 0], q[2]) @ rot_axis([0, 0, 1], q[3]) @ R0
        piv_w0 = R0 @ pivot_m + p0
        p = piv_w0 + np.array([q[0], q[4], q[1]]) - R @ pivot_m
        return R, p

    def gaps(q):
        R, p = pose(q)
        out = []
        for _nm, (n, d, ax, lo, hi) in planes:
            P = hs.points(R, p, ax, lo, hi)
            out.append(float((P @ n).min()) - d if len(P) else 0.0)
        return np.array(out)

    q = np.zeros(5)
    if kind == "cap":
        # the axis perpendicular to the stop (the step face flat on it): R0; the step face's lowest y on the stop's
        # footprint on its face; the dome V pair sets the sphere's x / z (a 2 x 2 solve on the clipped supports)
        st = [r for r in rows if r[0].endswith("_stop")][0]
        face = float(st[2][1] + st[3][1])
        dome = [(nm, pl) for nm, pl in planes if "_dome" in nm]
        for _it in range(iters):
            R, p = pose(q)
            P = hs.all_points(R, p)
            near = P[(np.abs(P[:, 0] - x) < st[3][0]) & (P[:, 2] < st[2][2] + st[3][2])]
            q[4] += face - float(near[:, 1].min())
            R, p = pose(q)
            A, b = [], []
            for _nm, (n, d, ax, lo, hi) in dome:
                Pc = hs.points(R, p, ax, lo, hi)
                A.append([n[0], n[2]])
                b.append(d - float((Pc @ n).min()))
            dxz = np.linalg.solve(np.array(A), np.array(b))
            q[0] += dxz[0]
            q[1] += dxz[1]
            if abs(dxz).max() < 1e-8 and abs(face - float(near[:, 1].min())) < 1e-8:
                break
    for _it in range(iters if kind == "motor" else 0):
        g = gaps(q)
        if np.abs(g).max() < 1e-7:
            break
        J = np.zeros((len(g), 4))
        for j, h in enumerate((1e-5, 1e-5, 1e-4, 1e-4)):
            dq = np.zeros(5)
            dq[j] = h
            J[:, j] = (gaps(q + dq) - g) / h
        step = np.linalg.lstsq(J, -g, rcond=None)[0]
        # damped: at most 0.5 mm and 0.01 rad per iteration (the gaps are piecewise linear in the pose)
        f = min(1.0, 5e-4 / max(1e-12, float(np.abs(step[:2]).max())), 0.01 / max(1e-12, float(np.abs(step[2:]).max())))
        q[:4] += f * step
    R, p = pose(q)
    if kind == "motor":
        mc = c["motor"]
        # the outer end face (the hull points' largest y) stop_gap_mm short of the stop
        st = [r for r in rows if r[0].endswith("_stop")][0]
        face = float(st[2][1] - st[3][1])
        P = hs.all_points(R, p)
        near = P[(np.abs(P[:, 0] - x) < st[3][0]) & (P[:, 2] < st[2][2] + st[3][2])]
        if len(near):
            p = p + np.array([0.0, face - float(mc["stop_gap_mm"]) * mm - float(near[:, 1].max()), 0.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    gfin = {}
    for nm, (n, d, ax, lo, hi) in planes:
        P = hs.points(R, p, ax, lo, hi)
        gfin[nm] = round((float((P @ n).min()) - d) * 1e6, 3) if len(P) else None
    allP = hs.all_points(R, p)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float((R @ np.array([0, 0, 1.0])) @ (R0 @ np.array([0, 0, 1.0])))))))
    info = {"plate_gaps_um": gfin, "lowest_mm": round(float(allP[:, 2].min()) * 1e3, 3), "axis_tilt_deg": round(tilt, 4),
            "iterations": _it + 1}
    return T, info


def _lid_rest(kit, c, x, R, hs, y=0.0):
    """the lid on the plate's top (z = 0) in the stable rest of its collision hull nearest to R (its outer face up),
    its footprint centred on the nest's (x, y)"""
    from .rests import stable_rests
    P = hs.all_points(np.eye(3), np.zeros(3))
    mass, com, _iq, _I = kit.inertial("M2")
    rests = stable_rests(P, com)
    down_R = R.T @ np.array([0.0, 0.0, -1.0])          # world down in the module frame at the wanted orientation
    best = max(rests, key=lambda r: float(r["normal"] @ down_R))
    # turn the wanted orientation so that the chosen facet faces down (smallest rotation)
    n_world = R @ best["normal"]
    v = np.cross(n_world, [0.0, 0.0, -1.0])
    s, cth = np.linalg.norm(v), float(n_world @ np.array([0.0, 0.0, -1.0]))
    Rc = rot_axis(v, math.atan2(s, cth)) if s > 1e-12 else np.eye(3)
    Rt = Rc @ R
    Pw = P @ Rt.T
    # the footprint's centre (the hull's xy box) on the nest's centre: the pocket is laid out around the footprint
    cxy = 0.5 * (Pw[:, :2].min(0) + Pw[:, :2].max(0))
    p = np.array([x - cxy[0], float(y) - cxy[1], -float(Pw[:, 2].min())])
    T = np.eye(4)
    T[:3, :3] = Rt
    T[:3, 3] = p
    tilt = math.degrees(math.atan2(s, cth))
    return T, {"rest_tilt_deg": round(tilt, 3), "rest_margin_mm": round(best["margin_m"] * 1e3, 3), "n_rests": len(rests)}
