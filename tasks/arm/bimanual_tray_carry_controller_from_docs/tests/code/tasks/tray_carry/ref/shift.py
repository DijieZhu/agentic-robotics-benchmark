import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
UNIT_DIR = os.path.normpath(os.path.join(TASK_DIR, "..", "..", "shared", "openarm_unit"))
if UNIT_DIR not in sys.path:
    sys.path.insert(0, UNIT_DIR)

import openarm_unit
import mujoco
import numpy as np

from openarm_unit import base
from openarm_unit import rng as urng
from openarm_unit import world as W
from openarm_unit.state import public_params

try:
    from openarm_unit.unit import UnitInvalid
    UNIT_HAS_INVALID = True
except ImportError:
    class UnitInvalid(Exception):

        def __init__(self, step, kind, kinds=None, info=None):
            super().__init__(f"the plant cannot integrate physics step {step} ({kind})")
            self.step, self.kind, self.kinds, self.info = step, kind, kinds, info
    UNIT_HAS_INVALID = False


def _load_constants():
    name = "t05_contract_constants"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(TASK_DIR, "contract", "constants.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


K = _load_constants()
SIDES = ("left", "right")
GRIPPER = {"left": 7, "right": 15}
DT = K.STEP_S
CYCLE = K.CYCLE_S
SPC = K.STEPS_PER_CYCLE
G = np.array([0.0, 0.0, -9.81])

DEV_SHIFT = (
    {"slot": 1, "job": "short", "mass_kg": 1.2, "along_m": 0.0, "label": "carry"},
    {"slot": 2, "job": "short", "mass_kg": 1.6, "along_m": +0.040, "label": "carry"},
    {"slot": 3, "job": "medium", "mass_kg": 0.9, "along_m": -0.040, "label": "carry"},
    {"slot": 4, "job": "short", "mass_kg": 2.0, "along_m": 0.0, "label": "carry"},
    {"slot": 5, "job": "lap", "mass_kg": 1.4, "along_m": +0.020, "label": "carry"},
    {"slot": 6, "job": "lap", "mass_kg": 9.5, "along_m": +0.040, "label": "decline", "only_where_named": True},
)
DEFAULT_BELOW_M = 0.030
DEFAULT_FORE_AFT_M = 0.0
DEV_SHIFT_B = (
    {"slot": 21, "name": "B1", "job": "short", "mass_kg": 1.2, "along_m": 0.0, "label": "carry"},
    {"slot": 22, "name": "B2", "job": "short", "mass_kg": 1.6, "along_m": +0.040, "label": "carry"},
    {"slot": 23, "name": "B3", "job": "lap", "mass_kg": 1.4, "along_m": +0.020, "label": "carry"},
    {"slot": 24, "name": "B4", "job": "lap", "mass_kg": 9.5, "along_m": +0.040, "label": "decline"},
    {"slot": 25, "name": "B5", "job": "medium", "mass_kg": 0.9, "along_m": -0.040, "label": "carry"},
    {"slot": 26, "name": "B6", "job": "short", "mass_kg": 2.0, "along_m": 0.0, "label": "carry"},
    {"slot": 27, "name": "c3", "job": "lap", "mass_kg": 9.0, "along_m": +0.030, "label": "decline",
     "only_where_named": True},
)
SHIFT_B = ("B1", "B2", "B3", "B4", "B5", "B6")


def dev_slots(numbers=(1, 2, 3, 4, 5), below_m=DEFAULT_BELOW_M, fore_aft_m=DEFAULT_FORE_AFT_M, label_override=None):
    if any(isinstance(n, str) or (isinstance(n, int) and n > 20) for n in numbers):
        by = {}
        for s in DEV_SHIFT + DEV_SHIFT_B:
            by[s["slot"]] = s
            if "name" in s:
                by[s["name"]] = s
        out = []
        for n in numbers:
            if n not in by:
                raise ValueError(f"no development slot {n!r}")
            d = dict(by[n])
            d["fore_aft_m"] = fore_aft_m
            d["below_m"] = below_m
            if label_override and d["slot"] in label_override:
                d["label"] = label_override[d["slot"]]
            out.append(d)
        return out
    out = []
    for s in DEV_SHIFT:
        if s["slot"] in numbers:
            d = dict(s)
            d["fore_aft_m"] = fore_aft_m
            d["below_m"] = below_m
            if label_override and s["slot"] in label_override:
                d["label"] = label_override[s["slot"]]
            out.append(d)
    return out


def quat_of_matrix(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return tuple(float(x) for x in q)


def matrix_of_quat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


def quat_xyz(rx, ry, rz):
    def ax(a, i):
        v = [math.cos(a / 2), 0.0, 0.0, 0.0]
        v[1 + i] = math.sin(a / 2)
        return tuple(v)
    return W.qmul(ax(rz, 2), W.qmul(ax(ry, 1), ax(rx, 0)))


def r_target(pitch_deg=-40.0):
    a = math.radians(pitch_deg)
    p = np.array([math.cos(a), 0.0, math.sin(a)])
    z = -p
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


class Cell:
    BENCH_TOP = K.BENCH_TOP_Z_M
    GRIP_Y = K.HANDLE_SPACING_M / 2.0
    GRIP_ABOVE_FEET = K.GRIP_ABOVE_FEET_M
    DECK_HALF = (K.TRAY_DECK_M[1] / 2, K.TRAY_DECK_M[0] / 2)
    DECK_THICK_HALF = 0.0025
    DECK_Z = -0.0515
    FOOT_HALF = (0.010, 0.010, 0.003)
    FOOT_XY = ((0.060, 0.185), (0.060, -0.185), (-0.060, 0.185), (-0.060, -0.185))
    BLOCK_WIDTH = 0.020
    CUP_CLEAR = 0.0005
    CUP_WALL = (0.002, 0.002)
    MASS = {"deck": 0.44, "foot": 0.02, "neck": 0.02, "block": 0.03}
    F_DRAWN = ((K.F_REFERENCE_POINT[0], K.F_REFERENCE_POINT[1], K.BENCH_TOP_Z_M), (1.0, 0.0, 0.0, 0.0))
    S_DRAWN = ((K.S_REFERENCE_POINT[0], K.S_REFERENCE_POINT[1], K.S_TOP_Z_M), (1.0, 0.0, 0.0, 0.0))
    S_HALF = (0.085, 0.240, 0.010)
    PITCH_DEG = -40.0

    @classmethod
    def grip_quat(cls):
        return quat_of_matrix(r_target(cls.PITCH_DEG))

    @classmethod
    def grip_frames(cls):
        q = cls.grip_quat()
        return {"left": ((0.0, cls.GRIP_Y, 0.0), q), "right": ((0.0, -cls.GRIP_Y, 0.0), q)}

    CUT_FOUND_X = (-8.892, 20.762)
    CUT_FOUND_Y = (93.480, 266.541)
    CUT_GROW_M = 0.002
    CUT_X = (-0.011, 0.023)
    CUT_Y_IN = 0.091
    FOOTING_HALF = (0.005, 0.005)
    WEB_WIDTH = 0.006

    @classmethod
    def _cell(cls, cell):
        c = CELL_DEFAULT if cell is None else cell
        if c not in CELLS:
            raise ValueError(f"cell: one of {CELLS}")
        return c

    @classmethod
    def deck_pieces(cls):
        hx, hy = cls.DECK_HALF
        x0, x1 = cls.CUT_X
        yi = cls.CUT_Y_IN
        fx, fy = cls.FOOTING_HALF
        wl = -fx - x0
        ww = cls.WEB_WIDTH
        a_mid = (2 * hx) * (2 * yi)
        a_aft = (x0 + hx) * (hy - yi)
        a_fore = (hx - x1) * (hy - yi)
        a_foot = (2 * fx) * (2 * fy)
        a_web = wl * ww
        A = a_mid + 2.0 * (a_aft + a_fore + a_foot + a_web)
        M = cls.MASS["deck"]
        m_mid = M * a_mid / A
        m_foot = M * a_foot / A
        m_web = M * a_web / A
        m_end = M * (a_aft + a_fore) / A
        xa, xf = 0.5 * (-hx + x0), 0.5 * (x1 + hx)
        xw = 0.5 * (x0 - fx)
        m_f = (-m_end * xa - m_web * xw) / (xf - xa)
        m_a = m_end - m_f
        yc = 0.5 * (yi + hy)
        out = [("deck_mid", (0.0, 0.0), (hx, yi), m_mid, True)]
        for side, sg in (("left", 1.0), ("right", -1.0)):
            out.append((f"deck_{side}_aft", (xa, sg * yc), (0.5 * (x0 + hx), 0.5 * (hy - yi)), m_a, True))
            out.append((f"deck_{side}_fore", (xf, sg * yc), (0.5 * (hx - x1), 0.5 * (hy - yi)), m_f, True))
            out.append((f"footing_{side}", (0.0, sg * cls.GRIP_Y), (fx, fy), m_foot, False))
            out.append((f"web_{side}", (xw, sg * cls.GRIP_Y), (0.5 * wl, 0.5 * ww), m_web, False))
        return out

    @classmethod
    def tray_parts(cls, cell=None):
        c = cls._cell(cell)
        m = cls.MASS
        dz = cls.DECK_Z
        if c == "precheck":
            parts = [{"part": "deck", "type": "box", "pose": ((0.0, 0.0, dz), (1, 0, 0, 0)),
                      "size": (cls.DECK_HALF[0], cls.DECK_HALF[1], cls.DECK_THICK_HALF), "mass": m["deck"]}]
        else:
            parts = []
            for p, (x, y), (sx, sy), ms, col in cls.deck_pieces():
                d = {"part": p, "type": "box", "pose": ((x, y, dz), (1, 0, 0, 0)), "size": (sx, sy, cls.DECK_THICK_HALF),
                     "mass": ms}
                if col:
                    d["cls"] = "tray.deck"
                else:
                    d["collide"] = False
                parts.append(d)
        fz = -cls.GRIP_ABOVE_FEET + cls.FOOT_HALF[2]
        for k, (x, y) in enumerate(cls.FOOT_XY):
            parts.append({"part": f"foot{k + 1}", "type": "box", "pose": ((x, y, fz), (1, 0, 0, 0)),
                          "size": cls.FOOT_HALF, "mass": m["foot"],
                          "cls": "tray.feet" if c == "precheck" else f"tray.foot{k + 1}"})
        for side, sg in (("left", 1.0), ("right", -1.0)):
            y = sg * cls.GRIP_Y
            top = dz + cls.DECK_THICK_HALF
            parts.append({"part": f"neck_{side}", "type": "box", "pose": ((0.0, y, 0.5 * (top - 0.010)), (1, 0, 0, 0)),
                          "size": (0.005, 0.005, 0.5 * (-0.010 - top)), "mass": m["neck"], "collide": False})
            parts.append({"part": f"block_{side}", "type": "box", "pose": ((0.0, y, 0.0), (1, 0, 0, 0)),
                          "size": (0.010, cls.BLOCK_WIDTH / 2, 0.010), "mass": m["block"], "collide": False})
        return parts

    @classmethod
    def empty_tray(cls, cell="precheck"):
        ms, cs = 0.0, np.zeros(3)
        for p in cls.tray_parts(cell):
            ms += p["mass"]
            cs += p["mass"] * np.array(p["pose"][0])
        return ms, cs / ms

    @classmethod
    def fixture_parts(cls):
        parts = [("base", ((0.0, 0.0, -0.002), (1, 0, 0, 0)), (0.085, 0.215, 0.002))]
        fx, fy, _ = cls.FOOT_HALF
        cw, ch = cls.CUP_WALL
        for k, (x, y) in enumerate(cls.FOOT_XY):
            ix, iy = fx + cls.CUP_CLEAR + cw, fy + cls.CUP_CLEAR + cw
            for w_, (dx, dy, sx, sy) in enumerate(((ix, 0.0, cw, iy + cw), (-ix, 0.0, cw, iy + cw),
                                                    (0.0, iy, ix - cw, cw), (0.0, -iy, ix - cw, cw))):
                parts.append((f"cup{k + 1}w{w_}", ((x + dx, y + dy, ch), (1, 0, 0, 0)), (sx, sy, ch)))
        return parts

    @classmethod
    def fixture_classes(cls, cell=None):
        c = cls._cell(cell)
        out = {}
        for p in cls.fixture_parts():
            name = p[0]
            if name == "base":
                out[name] = "F.base"
            else:
                out[name] = "F.cup" if c == "precheck" else "F." + name.split("w")[0]
        return out

    @classmethod
    def feet_in_cups(cls, tray_pose, F_pose, tol_xy=1e-4, tol_z=0.001):
        Fi = W.inverse(tuple(F_pose))
        rows, ok = [], True
        for k, (x, y) in enumerate(cls.FOOT_XY):
            c = W.compose(tuple(tray_pose), ((x, y, -cls.GRIP_ABOVE_FEET + cls.FOOT_HALF[2]), (1.0, 0.0, 0.0, 0.0)))
            p = W.compose(Fi, c)[0]
            bottom = float(p[2]) - cls.FOOT_HALF[2]
            dx, dy = float(p[0]) - x, float(p[1]) - y
            inside = abs(dx) <= cls.CUP_CLEAR + tol_xy and abs(dy) <= cls.CUP_CLEAR + tol_xy and abs(bottom) <= tol_z
            ok = ok and inside
            rows.append({"foot": k + 1, "dx_mm": 1e3 * dx, "dy_mm": 1e3 * dy, "bottom_mm": 1e3 * bottom, "in_cup": inside})
        return ok, rows

    @classmethod
    def stands_on_S(cls, tray_pose, S_pose, grow=K.S_SEAT_FRAME_GROW_M, tol_z=0.001):
        Si = W.inverse(tuple(S_pose))
        rows, ok = [], True
        fx, fy, fz = cls.FOOT_HALF
        for k, (x, y) in enumerate(cls.FOOT_XY):
            ov, hz = -1e9, -1e9
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    c = W.compose(tuple(tray_pose), ((x + sx * fx, y + sy * fy, -cls.GRIP_ABOVE_FEET), (1.0, 0.0, 0.0,
                                                                                                         0.0)))
                    p = W.compose(Si, c)[0]
                    hz = max(hz, abs(float(p[2])))
                    ov = max(ov, abs(float(p[0]) - x) - fx, abs(float(p[1]) - y) - fy)
            inside = ov <= grow and hz <= tol_z
            ok = ok and inside
            rows.append({"foot": k + 1, "outside_outline_mm": 1e3 * ov, "bottom_mm": 1e3 * hz, "in_frame": inside})
        return ok, rows

    @classmethod
    def bench_parts(cls):
        top = cls.BENCH_TOP
        fx0, fx1 = K.F_REFERENCE_POINT[0] - 0.090, K.F_REFERENCE_POINT[0] + 0.090
        fy = 0.220
        x0, x1, ym = 0.12, 0.72, 0.45
        z_layer = top - 0.002
        parts = [("slab", ((0.5 * (x0 + x1), 0.0, top - 0.004 - 0.020), (1, 0, 0, 0)), (0.5 * (x1 - x0), ym, 0.020)),
                 ("top_back", ((0.5 * (x0 + fx0), 0.0, z_layer), (1, 0, 0, 0)), (0.5 * (fx0 - x0), ym, 0.002)),
                 ("top_front", ((0.5 * (fx1 + x1), 0.0, z_layer), (1, 0, 0, 0)), (0.5 * (x1 - fx1), ym, 0.002)),
                 ("top_left", ((0.5 * (fx0 + fx1), 0.5 * (fy + ym), z_layer), (1, 0, 0, 0)),
                  (0.5 * (fx1 - fx0), 0.5 * (ym - fy), 0.002)),
                 ("top_right", ((0.5 * (fx0 + fx1), -0.5 * (fy + ym), z_layer), (1, 0, 0, 0)),
                  (0.5 * (fx1 - fx0), 0.5 * (ym - fy), 0.002))]
        return parts

    @classmethod
    def tray_in(cls, rest_pose):
        return W.compose(rest_pose, ((0.0, 0.0, cls.GRIP_ABOVE_FEET), (1.0, 0.0, 0.0, 0.0)))

    @classmethod
    def seat_angle(cls):
        return _seat_angle()


CELLS = ("corrected", "precheck")
CELL_DEFAULT = "corrected"


_SEAT_ANGLE = []


def _seat_angle():
    if not _SEAT_ANGLE:
        _SEAT_ANGLE.append(float(W.seat_angle_for_width(Cell.BLOCK_WIDTH, "left")))
    return _SEAT_ANGLE[0]


def _rot_axis(ax, a):
    c, s_ = math.cos(a), math.sin(a)
    x, y, z = ax
    Kx = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s_ * Kx + (1.0 - c) * Kx @ Kx


SWEEP_SWIVELS = (20.0, 45.0, 60.0)
SWEEP_POS_M = 0.008
SWEEP_ROT_DEG = 4.0
SWEEP_OPENINGS = 8


def cell_sweep(cell=None, swivels=SWEEP_SWIVELS, open_rad=None, footprint=True):
    from scipy.spatial import ConvexHull
    c = Cell._cell(cell)
    st, ses = nominal_unit_state()
    u = build_unit(st, ses, nominal_site(1), cell=c)
    m = u.model
    d = mujoco.MjData(m)
    kin = Kin()
    robot = set()
    for s in SIDES:
        for b in base.ARM_BODIES[s]:
            robot.add(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b))
    arm_q = {s: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_joint{i}")]
                 for i in range(1, 8)] for s in SIDES}
    fj = {s: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_finger_joint{k}") for k in (1, 2)]
          for s in SIDES}
    fq = {s: [m.jnt_qposadr[j] for j in fj[s]] for s in SIDES}
    rng_end = float(abs(m.jnt_range[fj["left"][0]]).max())
    f_end = rng_end if open_rad is None else float(open_rad)
    openings = [float(x) for x in np.linspace(_seat_angle(), f_end, SWEEP_OPENINGS)]
    tq = u.world["tray"].qadr
    tray_pos = np.array(Cell.tray_in(Cell.F_DRAWN)[0])
    ht = hand_targets(tray_pos)
    fbodies = {s: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{s}_ee_{k}_finger") for k in ("inner", "outer")]
               for s in SIDES}
    fgeoms = {s: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in fbodies[s] and m.geom_contype[g]] for s in SIDES}
    hulls = {}
    if footprint:
        for s in SIDES:
            for g in fgeoms[s]:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                V = m.mesh_vert[a:a + n].astype(np.float64)
                hulls[g] = (V, ConvexHull(V).simplices)
    zt = float(tray_pos[2] + Cell.DECK_Z + Cell.DECK_THICK_HALF)
    rec = {"cell": c, "openings_rad": openings, "finger_range_end_rad": rng_end, "deck_top_z_m": zt,
           "tray_pos": [float(x) for x in tray_pos], "swivels": {}, "pass": True}
    for sw in swivels:
        qs = {s: kin.solve(s, ht[s][0], ht[s][1], float(sw))[0] for s in SIDES}
        pairs, deep = {}, {}
        nos, hit_poses, lowest = 0, 0, 1e9
        fp = {s: [1e9, -1e9, 1e9, -1e9] for s in SIDES}
        footing_cross = 0
        for s in SIDES:
            p0, R0 = ht[s]
            sg = 1.0 if s == "left" else -1.0
            for corner in itertools.product((-1, 0, 1), repeat=6):
                dp = R0 @ (SWEEP_POS_M * np.array(corner[:3], dtype=np.float64))
                a = math.radians(SWEEP_ROT_DEG)
                Rd = _rot_axis((1, 0, 0), a * corner[3]) @ _rot_axis((0, 1, 0), a * corner[4]) \
                    @ _rot_axis((0, 0, 1), a * corner[5])
                q, _ = kin.solve(s, p0 + dp, R0 @ Rd, float(sw), qs[s])
                if q is None:
                    nos += 1
                    continue
                hit = False
                for f in openings:
                    for ss in SIDES:
                        d.qpos[arm_q[ss]] = q if ss == s else qs[ss]
                        sgn = 1.0 if ss == "left" else -1.0
                        for adr in fq[ss]:
                            d.qpos[adr] = sgn * f
                    d.qpos[tq:tq + 3] = tray_pos
                    d.qpos[tq + 3:tq + 7] = (1.0, 0.0, 0.0, 0.0)
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_collision(m, d)
                    for i in range(d.ncon):
                        cc = d.contact[i]
                        if cc.dist >= 0.0:
                            continue
                        b1, b2 = int(m.geom_bodyid[cc.geom1]), int(m.geom_bodyid[cc.geom2])
                        if b1 not in robot and b2 not in robot:
                            continue
                        if b1 in robot and b2 in robot:
                            n1, n2 = m.body(b1).name, m.body(b2).name
                            if "finger" in n1 and "finger" in n2 and n1.split("_")[1] == n2.split("_")[1]:
                                continue
                            key = f"robot:{n1}|robot:{n2}"
                        else:
                            o = cc.geom2 if b1 in robot else cc.geom1
                            key = m.geom(o).name or m.body(m.geom_bodyid[o]).name
                        pairs[key] = pairs.get(key, 0) + 1
                        deep[key] = min(deep.get(key, 0.0), float(cc.dist))
                        hit = True
                    if footprint:
                        allV = []
                        for g in fgeoms[s]:
                            V, simp = hulls[g]
                            R = d.geom_xmat[g].reshape(3, 3)
                            Wv = V @ R.T + d.geom_xpos[g]
                            allV.append(Wv)
                            lowest = min(lowest, float(Wv[:, 2].min()))
                            pts = [Wv[Wv[:, 2] < zt]]
                            e = np.concatenate([simp[:, [0, 1]], simp[:, [1, 2]], simp[:, [2, 0]]])
                            z0, z1 = Wv[e[:, 0], 2], Wv[e[:, 1], 2]
                            cr = (z0 - zt) * (z1 - zt) < 0.0
                            if np.any(cr):
                                t = ((zt - z0[cr]) / (z1[cr] - z0[cr]))[:, None]
                                pts.append(Wv[e[cr, 0]] + t * (Wv[e[cr, 1]] - Wv[e[cr, 0]]))
                            P = np.vstack(pts) - tray_pos
                            if len(P):
                                fp[s][0] = min(fp[s][0], float(P[:, 0].min()))
                                fp[s][1] = max(fp[s][1], float(P[:, 0].max()))
                                yy = sg * P[:, 1]
                                fp[s][2] = min(fp[s][2], float(yy.min()))
                                fp[s][3] = max(fp[s][3], float(yy.max()))
                        Vt = np.vstack(allV) - tray_pos
                        yy = sg * Vt[:, 1]
                        inblock = np.any((np.abs(Vt[:, 0]) <= 0.010) & (np.abs(yy - Cell.GRIP_Y) <= Cell.BLOCK_WIDTH / 2)
                                         & (np.abs(Vt[:, 2]) <= 0.010))
                        low = (Vt[:, 2] < zt - tray_pos[2]) & (np.abs(Vt[:, 0]) <= Cell.FOOTING_HALF[0]) \
                            & (np.abs(yy - Cell.GRIP_Y) <= Cell.FOOTING_HALF[1])
                        if not inblock and np.any(low):
                            footing_cross += 1
                hit_poses += hit
        met = {k: v for k, v in pairs.items()}
        rs = {"poses": 2 * 729, "nosolve": nos, "poses_with_contact": hit_poses,
              "pairs": {k: {"count": met[k], "deepest_mm": 1e3 * deep[k]} for k in sorted(met)}}
        if footprint:
            rs["footprint_below_deck_top_mm"] = {s: {"x": [1e3 * fp[s][0], 1e3 * fp[s][1]],
                                                     "abs_y": [1e3 * fp[s][2], 1e3 * fp[s][3]]} for s in SIDES}
            rs["lowest_finger_point_m"] = lowest
            rs["lowest_above_bench_mm"] = 1e3 * (lowest - Cell.BENCH_TOP)
            rs["configs_finger_below_deck_top_over_footing_no_vertex_in_block"] = footing_cross
        rec["swivels"][str(float(sw))] = rs
        if hit_poses or nos:
            rec["pass"] = False
    return rec


def cell_mass_record():
    out = {}
    for c in CELLS:
        ms, cs = Cell.empty_tray(c)
        st, ses = nominal_unit_state()
        u = build_unit(st, ses, nominal_site(1), cell=c)
        bid = u.world["tray"].bid
        m = u.model
        out[c] = {"mass_kg": ms, "centre_m": [float(x) for x in cs], "body_mass_kg": float(m.body_mass[bid]),
                  "body_ipos_m": [float(x) for x in m.body_ipos[bid]],
                  "body_inertia_kgm2": [float(x) for x in m.body_inertia[bid]],
                  "parts": [[p["part"], p["mass"], p.get("collide", True)] for p in Cell.tray_parts(c)]}
    a, b = out["precheck"], out["corrected"]
    out["difference"] = {"mass_g": 1e3 * (b["mass_kg"] - a["mass_kg"]),
                         "centre_mm": [1e3 * (y - x) for x, y in zip(a["centre_m"], b["centre_m"])],
                         "body_ipos_mm": [1e3 * (y - x) for x, y in zip(a["body_ipos_m"], b["body_ipos_m"])],
                         "body_inertia_rel": [(y - x) / x for x, y in zip(a["body_inertia_kgm2"], b["body_inertia_kgm2"])]}
    return out


def _u(key, stream, n, lo, hi):
    return [float(urng.uniform_in(key, stream, i, lo, hi)) for i in range(n)]


def draw_site(key, label, n_captures, ambient_C=None):
    prof = json.load(open(K.PROFILE_PATH, encoding="utf-8"))
    tq = {t["quantity"]: t for t in prof["task_quantities"]}
    mu = _u(key, f"site/{label}/pad_friction", 2, *tq["pad_friction"]["range"])
    out = {"label": label, "pad_friction": {"left": mu[0], "right": mu[1]}, "ambient_C": ambient_C}
    for name, q in (("F", "F_pose_offset"), ("S", "S_pose_offset")):
        r = tq[q]["range"]
        t = _u(key, f"site/{label}/{q}/m", 3, *r["m"])
        a = _u(key, f"site/{label}/{q}/deg", 3, *r["deg"])
        drawn = Cell.F_DRAWN if name == "F" else Cell.S_DRAWN
        off = (tuple(t), quat_xyz(*[math.radians(x) for x in a]))
        out[name] = {"offset_m": t, "offset_deg": a, "drawn": [list(drawn[0]), list(drawn[1])],
                     "true": [list(v) for v in W.compose(drawn, off)]}
    r = tq["seat_offset"]["range"]
    seats = {}
    for side in SIDES:
        lst = []
        for k in range(n_captures):
            t = _u(key, f"site/{label}/seat/{side}/{k}/m", 3, *r["m"])
            a = _u(key, f"site/{label}/seat/{side}/{k}/deg", 3, *r["deg"])
            lst.append([t, list(quat_xyz(*[math.radians(x) for x in a]))])
        seats[side] = lst
    out["seats"] = seats
    return out


def nominal_site(n_captures):
    ident = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    out = {"label": "nominal", "pad_friction": {"left": 0.55, "right": 0.55}, "ambient_C": None,
           "seats": {s: [ident] * n_captures for s in SIDES}}
    for name, drawn in (("F", Cell.F_DRAWN), ("S", Cell.S_DRAWN)):
        out[name] = {"offset_m": [0.0] * 3, "offset_deg": [0.0] * 3, "drawn": [list(drawn[0]), list(drawn[1])],
                     "true": [list(drawn[0]), list(drawn[1])]}
    return out


def pose_of(x):
    return (tuple(float(v) for v in x[0]), tuple(float(v) for v in x[1]))


def make_world(site, ramp_steps=int(round(K.SEAT_RAMP_S / K.STEP_S)), cell=None):
    c = Cell._cell(cell)
    w = W.World()
    w.rest("bench", parts=Cell.bench_parts(), pose=((0.0, 0.0, 0.0), (1, 0, 0, 0)),
           cls={p[0]: "bench" for p in Cell.bench_parts()})
    fp = Cell.fixture_parts()
    w.rest("F", pose=pose_of(site["F"]["true"]), parts=fp, cls=Cell.fixture_classes(c))
    sp = pose_of(site["S"]["true"])
    w.rest("S", pose=sp, parts=[("plate", ((0.0, 0.0, -Cell.S_HALF[2]), (1, 0, 0, 0)), Cell.S_HALF)],
           cls={"plate": "S"})
    w.object("tray", parts=Cell.tray_parts(c), grip_frames=Cell.grip_frames(), load=None, pose=None)
    for side in SIDES:
        seats = [pose_of(s) for s in site["seats"][side]] or [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))]
        w.grasp(f"g_{side}", arm=side, object="tray", neck=side, seats=seats, seat_angle=Cell.seat_angle(),
                trigger="clamp", ramp_steps=ramp_steps)
    return w


def load_for(mass_kg, along_m, fore_aft_m, below_m):
    m0, c0 = Cell.empty_tray()
    ml = float(mass_kg) - m0
    if ml <= 0.0:
        raise ValueError("a tray is at least its empty mass")
    c = np.array([fore_aft_m, along_m, -below_m])
    cl = (mass_kg * c - m0 * c0) / ml
    I = ml * (0.1 ** 2) / 6.0
    return ml, tuple(float(x) for x in cl), (I, I, I)


class Kin:

    def __init__(self, bases=None, stops=None):
        plant = {}
        if bases is not None:
            plant["base"] = {s: {"pos": list(bases[k][0]), "quat": list(bases[k][1])} for k, s in enumerate(SIDES)}
        if stops is not None:
            plant["joints"] = {f"openarm_{s}_joint{j + 1}": {"range": list(stops[a * 7 + j])}
                               for a, s in enumerate(SIDES) for j in range(7)}
        self.m = base.unit_model(plant or None)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jid = {s: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_joint{i}") for i in range(1, 8)]
                    for s in SIDES}
        self.qa = {s: np.array([m.jnt_qposadr[j] for j in self.jid[s]]) for s in SIDES}
        self.da = {s: np.array([m.jnt_dofadr[j] for j in self.jid[s]]) for s in SIDES}
        self.ee = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base.EE_BODIES[s]) for s in SIDES}
        self.lo = {s: m.jnt_range[self.jid[s], 0].copy() for s in SIDES}
        self.hi = {s: m.jnt_range[self.jid[s], 1].copy() for s in SIDES}
        self.fq = {s: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_finger_joint{k}")]
                       for k in (1, 2)] for s in SIDES}
        self.tool = np.array(base.TOOL_OFFSET)
        sl = [np.array([0.3, -0.3, 0.0, 1.4, 0.0, 0.3, 0.0]), np.array([-0.2, -0.6, 0.5, 1.6, 0.3, 0.0, 0.0]),
              np.array([0.6, -0.2, -0.3, 1.0, -0.3, 0.3, 0.0]), np.array([0.0, -0.9, 0.8, 1.8, 0.5, 0.0, 0.3])]
        mir = np.array([-1, -1, -1, 1, -1, -1, -1.0])
        self.mirror = mir
        self.seeds = {"left": sl, "right": [mir * q for q in sl]}
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))

    def set_arm(self, side, q):
        self.d.qpos[self.qa[side]] = q

    def tool_pose(self, side):
        d = self.d
        R = d.xmat[self.ee[side]].reshape(3, 3)
        return d.xpos[self.ee[side]] + R @ self.tool, R

    def fk(self, side, q):
        self.d.qpos[self.qa[side]] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.tool_pose(side)

    def swivel_deg(self, side, q):
        m, d = self.m, self.d
        d.qpos[self.qa[side]] = q
        mujoco.mj_kinematics(m, d)
        S = d.xanchor[self.jid[side][1]]
        E = d.xanchor[self.jid[side][3]]
        Wp = d.xpos[self.ee[side]]
        nx, ny, nz = Wp[0] - S[0], Wp[1] - S[1], Wp[2] - S[2]
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        dx, dy, dz = -nx * (-nz), -ny * (-nz), -1.0 - nz * (-nz)
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / dn, dy / dn, dz / dn
        ex, ey, ez = E[0] - S[0], E[1] - S[1], E[2] - S[2]
        en = ex * nx + ey * ny + ez * nz
        ex, ey, ez = ex - nx * en, ey - ny * en, ez - nz * en
        el = math.sqrt(ex * ex + ey * ey + ez * ez)
        if el < 1e-9:
            return 0.0
        ex, ey, ez = ex / el, ey / el, ez / el
        cx, cy, cz = dy * ez - dz * ey, dz * ex - dx * ez, dx * ey - dy * ex
        ang = math.degrees(math.atan2(cx * nx + cy * ny + cz * nz, dx * ex + dy * ey + dz * ez))
        return ang if side == "left" else -ang

    def ik(self, side, pt, Rt, sw_target, q0, margin=math.radians(2.0), iters=200, tol_p=2e-6, tol_w=2e-5):
        m, d = self.m, self.d
        q = np.array(q0, dtype=np.float64)
        lo = self.lo[side] + margin
        hi = self.hi[side] - margin
        for _ in range(iters):
            d.qpos[self.qa[side]] = q
            mujoco.mj_kinematics(m, d)
            p, R = self.tool_pose(side)
            ep = pt - p
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (Rt @ R.T).reshape(9))
            w = np.zeros(3)
            mujoco.mju_quat2Vel(w, quat, 1.0)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jac(m, d, self._jp, self._jr, p, self.ee[side])
            J = np.vstack([self._jp[:, self.da[side]], 0.2 * self._jr[:, self.da[side]]])
            err = np.concatenate([ep, 0.2 * w])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(6), err)
            sw = self.swivel_deg(side, q)
            esw = sw_target - sw
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(w) < tol_w and abs(esw) < 0.02:
                return q, True
            g = np.zeros(7)
            for k in range(7):
                qq = q.copy()
                qq[k] += 1e-5
                g[k] = (self.swivel_deg(side, qq) - sw) / 1e-5
            Nn = np.eye(7) - np.linalg.pinv(J) @ J
            gn = Nn @ g
            if gn @ gn > 1e-12:
                dq += gn * float(np.clip(esw, -8, 8)) / float(gn @ g) * 0.5
            q = np.clip(q + np.clip(dq, -0.15, 0.15), lo, hi)
        return q, False

    def solve(self, side, pt, Rt, sw, q_prev=None):
        cands = ([q_prev] if q_prev is not None else []) + list(self.seeds[side])
        best = None
        for q0 in cands:
            q, ok = self.ik(side, np.asarray(pt, float), np.asarray(Rt, float), sw, q0)
            if ok:
                mg = self.margin_deg(side, q)
                if q_prev is not None and q0 is q_prev:
                    return q, mg
                if best is None or mg > best[1]:
                    best = (q.copy(), mg)
        return best if best is not None else (None, None)

    def margin_deg(self, side, q):
        return float(np.degrees(np.min(np.minimum(q - self.lo[side], self.hi[side] - q))))

    def jac(self, side):
        p, _ = self.tool_pose(side)
        mujoco.mj_jac(self.m, self.d, self._jp, self._jr, p, self.ee[side])
        return self._jp[:, self.da[side]].copy(), self._jr[:, self.da[side]].copy()

    def rne(self, q, qd, qdd, fingers=None):
        m, d = self.m, self.d
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        for s in SIDES:
            d.qpos[self.qa[s]] = q[s]
            d.qvel[self.da[s]] = qd[s]
            d.qacc[self.da[s]] = qdd[s]
            f = _seat_angle() if fingers is None else fingers
            sg = 1.0 if s == "left" else -1.0
            for a in self.fq[s]:
                d.qpos[a] = sg * f
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        out = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 1, out)
        tau, jac = {}, {}
        for s in SIDES:
            tau[s] = out[self.da[s]] + m.dof_armature[self.da[s]] * qdd[s]
            jac[s] = self.jac(s)
        return tau, jac


def trapezoid(T):
    vp = 1.5 / T
    a = vp / (T / 3.0)

    def f(t):
        t = min(max(t, 0.0), T)
        if t < T / 3.0:
            return 0.5 * a * t * t, a * t, a
        if t < 2.0 * T / 3.0:
            return 0.5 * a * (T / 3.0) ** 2 + vp * (t - T / 3.0), vp, 0.0
        tt = T - t
        return 1.0 - 0.5 * a * tt * tt, a * tt, -a
    return f


def ramped_constant(T, ramp=0.1):
    v = 1.0 / (T - 0.5 * ramp)
    a = v / ramp

    def f(t):
        t = min(max(t, 0.0), T)
        if t < ramp:
            return 0.5 * a * t * t, a * t, a
        return 0.5 * a * ramp * ramp + v * (t - ramp), v, 0.0
    return f


def quintic(T):
    def f(t):
        t = min(max(t, 0.0), T)
        x = t / T
        return (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5, (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / T,
                (60 * x - 180 * x ** 2 + 120 * x ** 3) / (T * T))
    return f


BOX_XS = tuple(K.BOX["x"][0] + (K.BOX["x"][1] - K.BOX["x"][0]) * i / (K.BOX_GRID[0] - 1) for i in range(K.BOX_GRID[0]))
BOX_YS = tuple(K.BOX["y"][0] + (K.BOX["y"][1] - K.BOX["y"][0]) * i / (K.BOX_GRID[1] - 1) for i in range(K.BOX_GRID[1]))
BOX_ZS = tuple(K.BOX["z"][0] + (K.BOX["z"][1] - K.BOX["z"][0]) * i / (K.BOX_GRID[2] - 1) for i in range(K.BOX_GRID[2]))
GRID = tuple((round(x, 6), round(y, 6), round(z, 6)) for x in BOX_XS for y in BOX_YS for z in BOX_ZS)
SWIVELS = (45.0, 54.0, 30.0, 20.0, 60.0)


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def _grid_of(i):
    h = (halton(i, 2), halton(i, 3), halton(i, 5))
    ix = min(int(h[0] * K.BOX_GRID[0]), K.BOX_GRID[0] - 1)
    iy = min(int(h[1] * K.BOX_GRID[1]), K.BOX_GRID[1] - 1)
    iz = min(int(h[2] * K.BOX_GRID[2]), K.BOX_GRID[2] - 1)
    return (round(BOX_XS[ix], 6), round(BOX_YS[iy], 6), round(BOX_ZS[iz], 6))


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _angle_deg(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _entropy_int(entropy, name, mod):
    h = hashlib.sha256(b"t05 job generator\0" + bytes(entropy) + b"\0" + name.encode("ascii")).digest()
    return int.from_bytes(h[:8], "big") % mod


def short_or_medium_targets(kind, entropy, attempt=0):
    lo, hi = K.JOB_TARGETS[kind]
    n = lo + _entropy_int(entropy, f"{kind}/n/{attempt}", hi - lo + 1)
    k = 1 + _entropy_int(entropy, f"{kind}/halton/{attempt}", 100000)
    first, last = tuple(K.FIXED_FIRST), [tuple(x) for x in K.FIXED_LAST]
    need = K.REPEAT_TARGETS_MIN
    n_mid = n - 3
    halt = []
    kk = k
    while len(halt) < 400:
        c = _grid_of(kk)
        kk += 1
        if c != first and (not halt or c != halt[-1]):
            halt.append(c)
    budget = [20000]

    def dfs(seq, fdir, rev, hi):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        if len(seq) == n_mid:
            return seq if len(rev) >= need and _dist(seq[-1], last[0]) >= K.MOVE_MIN_M else None
        prev = seq[-1] if seq else first
        left = n_mid - len(seq)
        opts = []
        if len(rev) < need:
            for v in sorted(fdir):
                if v in rev or v == prev or _dist(prev, v) < K.MOVE_MIN_M:
                    continue
                if _angle_deg(np.array(v) - np.array(prev), fdir[v]) >= K.REPEAT_DIRECTION_MIN_DEG:
                    opts.append(("rv", v, hi))
        if left > need - len(rev):
            j = hi
            while j < len(halt) and (halt[j] in fdir or _dist(prev, halt[j]) < K.MOVE_MIN_M):
                j += 1
            if j < len(halt):
                opts.insert(0 if left > 2 * (need - len(rev)) else len(opts), ("new", halt[j], j + 1))
        for kind_, v, nh in opts:
            if kind_ == "new":
                f2 = dict(fdir)
                f2[v] = np.array(v) - np.array(prev)
                r = dfs(seq + [v], f2, rev, nh)
            else:
                r = dfs(seq + [v], fdir, rev | {v}, nh)
            if r is not None:
                return r
        return None
    mid = dfs([], {}, frozenset(), 0)
    if mid is None:
        return None, f"repeat rule not met (n {n})"
    seen, rv = set(), []
    for v in mid:
        if v in seen:
            rv.append(v)
        seen.add(v)
    revisited = set(rv)
    tg = [first]
    tg = tg + mid + last
    for a, b in zip(tg[:-2], tg[1:-1]):
        if _dist(a, b) < K.MOVE_MIN_M - 1e-12:
            return None, f"a move of {_dist(a, b):.3f} m"
    return tg, {"kind": kind, "n_targets": n, "halton_start": k, "attempt": attempt,
                "repeats": sorted([list(v) for v in revisited])}


def lap_targets(kin, rows=None):
    if rows is None:
        rows = grid_solutions(kin)
    qs = {p: rows[p] for p in GRID if rows.get(p) is not None}
    pts = sorted(qs)
    pairs, taken = [], set()
    for j in range(7):
        cand = []
        for a, b in itertools.combinations(pts, 2):
            dq = max(abs(qs[a]["q"][s][j] - qs[b]["q"][s][j]) for s in SIDES)
            cand.append((dq, a, b))
        cand.sort(key=lambda r: (-r[0], r[1], r[2]))
        c = 0
        for dq, a, b in cand:
            if (a, b) in taken:
                continue
            pairs.append((a, b, f"J{j + 1}"))
            taken.add((a, b))
            c += 1
            if c == K.LAP_PAIRS_PER_JOINT:
                break
    X0, X1 = K.BOX["x"]
    Y0, Y1 = K.BOX["y"]
    Z0, Z1 = K.BOX["z"]
    corners = [(round(x, 6), round(y, 6), round(z, 6)) for x in (X0, X1) for y in (Y0, Y1) for z in (Z0, Z1)]
    for a, b in itertools.combinations(corners, 2):
        diff = [abs(a[i] - b[i]) > 1e-9 for i in range(3)]
        if all(diff) or diff == [True, False, False] or diff == [False, True, False]:
            pairs.append((a, b, "corner"))
    adj = {}
    for a, b, _ in pairs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v in adj:
        adj[v].sort()
    unvisited = {v: list(adj[v]) for v in adj}
    nodes = sorted(adj)
    seen = set()
    groups = []
    for v in nodes:
        if v in seen:
            continue
        comp, stack = [], [v]
        seen.add(v)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        groups.append(sorted(comp))
    walk = []
    for comp in groups:
        start = min(comp, key=lambda p: _dist(p, walk[-1] if walk else K.FIXED_FIRST))
        stack, circ = [start], []
        while stack:
            x = stack[-1]
            if unvisited[x]:
                y = unvisited[x].pop(0)
                stack.append(y)
            else:
                circ.append(stack.pop())
        circ.reverse()
        walk += circ
    tg = [tuple(K.FIXED_FIRST)] + walk + [tuple(x) for x in K.FIXED_LAST]
    short = [(_dist(a, b)) for a, b in zip(tg[:-1], tg[1:]) if _dist(a, b) < K.MOVE_MIN_M]
    return tg, {"kind": "lap", "pairs": len(pairs), "groups": len(groups), "n_targets": len(tg),
                "list_moves": 2 * len(pairs), "moves_shorter_than_min": len(short)}


def grid_solutions(kin):
    pts = list(GRID) + [tuple(K.FIXED_FIRST)] + [tuple(x) for x in K.FIXED_LAST]
    out = {}
    for p in pts:
        out[p] = solve_target(kin, p)
    return out


def hand_targets(tray_pos, tray_quat=(1.0, 0.0, 0.0, 0.0), seats=None, offset=(0.0, 0.0, 0.0)):
    gf = Cell.grip_frames()
    out = {}
    for s in SIDES:
        seat = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)) if seats is None else seats[s]
        p, q = W.compose(W.compose((tuple(tray_pos), tuple(tray_quat)), gf[s]), seat)
        out[s] = (np.array(p) + np.array(offset), matrix_of_quat(q))
    return out


def solve_target(kin, tray_pos, tray_quat=(1.0, 0.0, 0.0, 0.0), seats=None, offset=(0.0, 0.0, 0.0), q_prev=None,
                 swivels=SWIVELS, min_margin=K.GENERATOR_STOP_MARGIN_DEG):
    ht = hand_targets(tray_pos, tray_quat, seats, offset)
    best = None
    for sw in swivels:
        sol, mg = {}, {}
        for s in SIDES:
            q, m_ = kin.solve(s, ht[s][0], ht[s][1], sw, None if q_prev is None else q_prev.get(s))
            if q is None and q_prev is not None:
                q, m_ = kin.solve(s, ht[s][0], ht[s][1], sw, None)
            if q is None:
                break
            sol[s], mg[s] = q, m_
        if len(sol) < 2:
            continue
        rec = {"q": sol, "swivel": sw, "margin_deg": min(mg.values())}
        if rec["margin_deg"] >= min_margin:
            return rec
        if best is None or rec["margin_deg"] > best["margin_deg"]:
            best = rec
    return best


def _no_load_speed(cls_name):
    p = public_params()
    ed = p["class_nominal_edition"]["value"][cls_name]
    v = p["editions"][ed]["no_load_speed_rpm"]
    v = v["value"] if isinstance(v, dict) else v
    return float(v) * 2.0 * math.pi / 60.0


def _peak(cls_name):
    p = public_params()
    ed = p["class_nominal_edition"]["value"][cls_name]
    v = p["editions"][ed]["T_peak_Nm"]
    return float(v["value"] if isinstance(v, dict) else v)


ARM_CLASSES = tuple(m["class"] for m in K.MODULES[:7])


def tray_wrench_shares(mass, com, acc, R=np.eye(3)):
    F = float(mass) * (np.asarray(acc, float) - G)
    c = np.asarray(com, float)
    wl = 0.5 + c[1] / (2.0 * Cell.GRIP_Y)
    rc = R @ c
    pl = R @ np.array([0.0, Cell.GRIP_Y, 0.0])
    pr = R @ np.array([0.0, -Cell.GRIP_Y, 0.0])
    Fl, Fr = wl * F, (1.0 - wl) * F
    M = np.cross(rc, F) - np.cross(pl, Fl) - np.cross(pr, Fr)
    return {"left": (Fl, 0.5 * M), "right": (Fr, 0.5 * M)}


class JobCheck:

    DECLINE_CHECKS = ("command", "need_over_scale", "peak")

    def __init__(self, kin, cell_model=None, scale=None, decline_check="command", offset=None, clamp=None):
        if decline_check not in self.DECLINE_CHECKS:
            raise ValueError(f"decline_check: one of {self.DECLINE_CHECKS}")
        self.kin = kin
        self.cm = cell_model

        def arm(v):
            return None if v is None else {"left": np.array([float(v[i]) for i in range(7)]),
                                           "right": np.array([float(v[8 + i]) for i in range(7)])}
        self.scale = arm(scale)
        self.offset = arm(offset)
        self.clamp = arm(clamp)
        self.decline_check = decline_check

    def check(self, targets, tray, plan=None, label="carry", contact_every=1, swivel=None):
        from minimal import plan_moves
        if plan is None:
            L = (K.F_REFERENCE_POINT[0], K.F_REFERENCE_POINT[1], K.F_REFERENCE_POINT[2] + K.LIFTED_POINT_M)
            if swivel is None:
                plan = plan_moves(self.kin, targets, start=L)
            else:
                sw = float(swivel)
                plan = plan_moves(self.kin, targets, start=L, swivels=[sw] * len(targets), swivel_choices=(sw,))
        if plan is None:
            return False, {"reason": "a target does not solve"}
        res = {"margin_min_deg": 1e9, "fk_err_mm": 0.0, "fk_err_deg": 0.0, "speed_share": 0.0, "torque_share": 0.0,
               "contacts": [], "tray_to_S_min_m": 1e9, "samples": 0}
        vmax = np.array([_no_load_speed(c) for c in ARM_CLASSES])
        peak = np.array([_peak(c) for c in ARM_CLASSES])
        mass, com = tray["mass_kg"], tray["com"]
        over_scale = (label == "decline" and self.scale is not None and self.decline_check == "need_over_scale")
        by_command = (label == "decline" and self.scale is not None and self.decline_check == "command")
        if over_scale:
            res["torque_rule"] = "need over the true module's scale against its peak"
        if by_command:
            res["torque_rule"] = ("the command (need - b) / k that delivers the holding torque against the module's "
                                  "clamp")
            clamp = {s: (self.clamp[s] if self.clamp is not None else peak) for s in SIDES}
            offs = {s: (self.offset[s] if self.offset is not None else np.zeros(7)) for s in SIDES}
            res["offset_given"] = self.offset is not None
            res["clamp_given"] = self.clamp is not None
        for mv in plan["moves"]:
            T = mv.T
            n = int(round(T / CYCLE))
            for i in range(n + 1):
                t = i * CYCLE
                st = mv.eval(t)
                res["samples"] += 1
                for s in SIDES:
                    q = st["q"][s]
                    res["margin_min_deg"] = min(res["margin_min_deg"], self.kin.margin_deg(s, q))
                    p, R = self.kin.fk(s, q)
                    pt, Rt = st["hands"][s]
                    res["fk_err_mm"] = max(res["fk_err_mm"], 1e3 * float(np.linalg.norm(p - pt)))
                    qq = np.zeros(4)
                    mujoco.mju_mat2Quat(qq, (Rt @ R.T).reshape(9))
                    res["fk_err_deg"] = max(res["fk_err_deg"], math.degrees(2 * math.acos(min(1.0, abs(qq[0])))))
                    res["speed_share"] = max(res["speed_share"], float(np.max(np.abs(st["qd"][s]) / vmax)))
                tau, jac = self.kin.rne(st["q"], st["qd"], st["qdd"])
                wr = tray_wrench_shares(mass, com, st["acc"])
                for s in SIDES:
                    Jp, Jr = jac[s]
                    tt = tau[s] + Jp.T @ wr[s][0] + Jr.T @ wr[s][1]
                    if by_command:
                        res["torque_share"] = max(res["torque_share"], float(np.max(
                            np.abs((tt - offs[s]) / self.scale[s]) / clamp[s])))
                    elif over_scale:
                        res["torque_share"] = max(res["torque_share"],
                                                  float(np.max(np.abs(tt) / self.scale[s] / peak)))
                    else:
                        res["torque_share"] = max(res["torque_share"], float(np.max(np.abs(tt) / peak)))
                res["tray_to_S_min_m"] = min(res["tray_to_S_min_m"], tray_to_S_distance(st["tray_pos"]))
                if self.cm is not None and i % contact_every == 0:
                    c = self.cm.contacts(st["q"], st["tray_pos"])
                    if c:
                        res["contacts"].append({"t": t, "pairs": c[:3]})
        lim = 1.0 if label == "decline" else K.GENERATOR_TORQUE_SHARE_MAX
        ok = (res["margin_min_deg"] >= K.GENERATOR_STOP_MARGIN_DEG and res["fk_err_mm"] <= 0.2
              and res["fk_err_deg"] <= 0.05 and res["speed_share"] <= K.GENERATOR_SPEED_SHARE_MAX
              and res["torque_share"] <= lim and not res["contacts"]
              and res["tray_to_S_min_m"] >= K.GENERATOR_TRAY_TO_S_MIN_M)
        res["ok"] = ok
        res["swivels"] = plan["swivels"]
        return ok, res


def tray_to_S_distance(tray_pos, S_pose=None):
    S = Cell.S_DRAWN if S_pose is None else S_pose
    tp = np.asarray(tray_pos, float)
    tlo = tp + np.array([-Cell.DECK_HALF[0], -Cell.DECK_HALF[1], -Cell.GRIP_ABOVE_FEET])
    thi = tp + np.array([Cell.DECK_HALF[0], Cell.DECK_HALF[1], Cell.DECK_Z + Cell.DECK_THICK_HALF])
    sc = np.array(S[0])
    slo = sc + np.array([-Cell.S_HALF[0], -Cell.S_HALF[1], -2 * Cell.S_HALF[2]])
    shi = sc + np.array([Cell.S_HALF[0], Cell.S_HALF[1], 0.0])
    gap = np.maximum(0.0, np.maximum(slo - thi, tlo - shi))
    return float(np.linalg.norm(gap))


class CellModel:

    def __init__(self, unit):
        self.m = unit.model
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_q = {s: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_joint{i}")]
                          for i in range(1, 8)] for s in SIDES}
        self.fq = {s: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_finger_joint{k}")]
                       for k in (1, 2)] for s in SIDES}
        tray = unit.world["tray"]
        self.tq = tray.qadr
        self.robot = set()
        for s in SIDES:
            for bname in base.ARM_BODIES[s]:
                self.robot.add(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname))
        self.d.qpos[:] = self.m.qpos0

    def contacts(self, q, tray_pos, fingers_open=False):
        m, d = self.m, self.d
        for s in SIDES:
            d.qpos[self.arm_q[s]] = q[s]
            sg = 1.0 if s == "left" else -1.0
            f = 0.45 if fingers_open else _seat_angle()
            for a in self.fq[s]:
                d.qpos[a] = sg * f
        d.qpos[self.tq:self.tq + 3] = tray_pos
        d.qpos[self.tq + 3:self.tq + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0.0:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 not in self.robot and b2 not in self.robot:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            if "finger" in n1 and "finger" in n2 and n1.split("_")[1] == n2.split("_")[1]:
                continue
            out.append((n1, n2, round(float(c.dist), 5)))
        return out


_NOMINAL_CELL_MODEL = []


def nominal_cell_model(cell=None):
    c = Cell._cell(cell)
    for cc, cm in _NOMINAL_CELL_MODEL:
        if cc == c:
            return cm
    st, ses = nominal_unit_state()
    cm = CellModel(build_unit(st, ses, nominal_site(1), cell=c))
    _NOMINAL_CELL_MODEL.append((c, cm))
    return cm


def true_kin(state):
    return Kin(bases=state.plant.base_poses, stops=state.plant.stops)


def module_clamps(unit):
    bu = unit.bus
    lp, lm = bu.env.limits(np.zeros(16))
    return tuple(float(x) for x in np.minimum(np.minimum(lp, lm), bu.t_peak))


def true_offsets(state, session):
    return tuple(float(a) + float(c) for a, c in zip(state.cal.b, session.b_step))


def generate_job(kind, entropy, kin, tray, cell_model=None, label="carry", max_attempts=20, contact_every=1,
                 true_check=None, swivel=None):
    rej = []
    chk = JobCheck(kin, cell_model)
    chk_t = None if true_check is None else JobCheck(
        true_check[0], true_check[1], scale=true_check[2] if len(true_check) > 2 else None,
        decline_check=true_check[3] if len(true_check) > 3 else "need_over_scale",
        offset=true_check[4] if len(true_check) > 4 else None, clamp=true_check[5] if len(true_check) > 5 else None)
    for att in range(max_attempts):
        if kind == "lap":
            tg, rec = lap_targets(kin)
        else:
            tg, rec = short_or_medium_targets(kind, entropy, att)
        if tg is None:
            rej.append({"attempt": att, "reason": rec})
            continue
        ok, res = chk.check(tg, tray, label=label, contact_every=contact_every, swivel=swivel)
        res_t = None
        if ok and chk_t is not None:
            ok_t, res_t = chk_t.check(tg, tray, label=label, contact_every=contact_every, swivel=swivel)
            res_t.pop("swivels", None)
            if not ok_t:
                rej.append({"attempt": att, "on": "true unit",
                            "reason": {k: res_t[k] for k in res_t if k != "contacts"},
                            "contacts": res_t.get("contacts", [])[:3]})
                if kind == "lap":
                    break
                continue
        if ok:
            sw = res.pop("swivels")
            out = {"targets": [list(t) for t in tg], "swivels": sw, "record": rec, "rejections": rej, "check": res,
                   "check_true": res_t}
            if swivel is not None:
                out["swivel_fixed"] = float(swivel)
            return out
        rej.append({"attempt": att, "on": "nominal model",
                    "reason": {k: res[k] for k in res if k not in ("contacts", "swivels")},
                    "contacts": res.get("contacts", [])[:3]})
        if kind == "lap":
            break
    return {"targets": None, "record": None, "rejections": rej, "check": None, "check_true": None}


DECLINE_CLASS = {"mass_kg": (9.0, 10.0), "along_abs_m": (0.030, 0.040), "fore_aft_m": (-0.005, 0.005),
                 "below_m": (0.0, 0.045)}
DECLINE_DRAWS_MAX = 64
NO_DECLINE_TRAY = "no decline tray of the class can be taken up"


def decline_tray_draw(entropy, k):
    key = bytes(entropy) if len(entropy) == 32 else hashlib.sha256(bytes(entropy)).digest()
    u = lambda q, lo, hi: float(urng.uniform_in(key, f"decline_tray/{q}", k, lo, hi))
    c = DECLINE_CLASS
    side = 1.0 if u("handle", 0.0, 1.0) < 0.5 else -1.0
    return {"mass_kg": u("mass_kg", *c["mass_kg"]), "along_m": side * u("along_abs_m", *c["along_abs_m"]),
            "fore_aft_m": u("fore_aft_m", *c["fore_aft_m"]), "below_m": u("below_m", *c["below_m"]), "draw": int(k)}


def clamps_of_state(state):
    from openarm_unit import bus as ubus
    from openarm_unit import registers as rg
    from openarm_unit.envelope import ModuleEnvelopes, t_lim_from_registers
    mods = ubus.pair_modules(state.editions)
    eds = tuple(m.edition for m in mods)
    R = rg.RegisterTable(state.cal.registers)
    tl = np.array([t_lim_from_registers(R.module(i), float(state.cal.imax[i])) for i in range(16)], np.float64)
    lp, lm = ModuleEnvelopes(eds, tl).limits(np.zeros(16))
    p = public_params()
    tpk = np.array([float(p["editions"][e]["T_peak_Nm"]["value"]) for e in eds], np.float64)
    return tuple(float(x) for x in np.minimum(np.minimum(lp, lm), tpk))


class TorqueBasis:

    def __init__(self, kin, targets, swivel=None):
        from minimal import plan_moves
        L = (K.F_REFERENCE_POINT[0], K.F_REFERENCE_POINT[1], K.F_REFERENCE_POINT[2] + K.LIFTED_POINT_M)
        if swivel is None:
            plan = plan_moves(kin, targets, start=L)
        else:
            sw = float(swivel)
            plan = plan_moves(kin, targets, start=L, swivels=[sw] * len(targets), swivel_choices=(sw,))
        self.ok = plan is not None
        if not self.ok:
            return
        T_0 = {s: [] for s in SIDES}
        A = {s: [] for s in SIDES}
        gy = Cell.GRIP_Y
        pl = np.array([0.0, gy, 0.0])
        E = np.eye(3)
        for mv in plan["moves"]:
            n = int(round(mv.T / CYCLE))
            for i in range(n + 1):
                st = mv.eval(i * CYCLE)
                tau, jac = kin.rne(st["q"], st["qd"], st["qdd"])
                u = np.asarray(st["acc"], float) - G
                Fx = {"left": [0.5 * u, np.zeros(3), u / (2.0 * gy), np.zeros(3)],
                      "right": [0.5 * u, np.zeros(3), -u / (2.0 * gy), np.zeros(3)]}
                Mx = [np.zeros(3), np.cross(E[0], u), np.cross(E[1], u) - np.cross(pl, u) / gy, np.cross(E[2], u)]
                for s in SIDES:
                    Jp, Jr = jac[s]
                    T_0[s].append(np.asarray(tau[s], float).copy())
                    A[s].append(np.stack([Jp.T @ Fx[s][j] + Jr.T @ (0.5 * Mx[j]) for j in range(4)], axis=1))
        self.T_0 = {s: np.asarray(T_0[s]) for s in SIDES}
        self.A = {s: np.asarray(A[s]) for s in SIDES}

    def need(self, tray):
        m = float(tray["mass_kg"])
        c = np.asarray(tray["com"], float)
        x = np.array([m, m * c[0], m * c[1], m * c[2]])
        return {s: self.T_0[s] + self.A[s] @ x for s in SIDES}

    def share(self, tray, scale=None, offset=None, clamp=None):
        peak = np.array([_peak(c) for c in ARM_CLASSES])
        nd = self.need(tray)
        out = 0.0
        for a, s in enumerate(SIDES):
            sl = slice(8 * a, 8 * a + 7)
            if scale is None:
                v = np.abs(nd[s]) / peak
            else:
                k = np.asarray(scale, float)[sl]
                b = np.zeros(7) if offset is None else np.asarray(offset, float)[sl]
                cl = peak if clamp is None else np.asarray(clamp, float)[sl]
                v = np.abs((nd[s] - b) / k) / cl
            out = max(out, float(np.max(v)))
        return out


def tray_of_draw(d):
    return {"mass_kg": d["mass_kg"], "com": (d["fore_aft_m"], d["along_m"], -d["below_m"])}


def take_up_decline_tray(entropy, basis, scale, offset, clamp, first=None, draws=DECLINE_DRAWS_MAX):
    tries = []
    cands = ([dict(first, draw="slot sheet")] if first is not None else [])
    k = 0
    while len(cands) < draws:
        cands.append(decline_tray_draw(entropy, k))
        k += 1
    for d in cands:
        sh = basis.share(tray_of_draw(d), scale, offset, clamp)
        tries.append({"draw": d["draw"], "share": sh})
        if sh <= 1.0:
            return {"tray": d, "tried": len(tries), "shares": tries, "reason": None}
    return {"tray": None, "tried": len(tries), "shares": tries, "reason": NO_DECLINE_TRAY}


CAPTURE_NOTE_SWIVELS = (20.0, 30.0, 40.0, 50.0, 60.0)


def capture_note_check(state, F_true, swivels=CAPTURE_NOTE_SWIVELS, kin_nominal=None, nominal_q=None):
    kn = Kin() if kin_nominal is None else kin_nominal
    lim_p = K.CAPTURE_BOX["m"] - K.FIT_NOTE["m"]
    lim_a = K.CAPTURE_BOX["deg"] - K.FIT_NOTE["deg"]
    lim_v = math.sin(math.radians(lim_a) / 2.0)
    tray_d = Cell.tray_in(Cell.F_DRAWN)
    if nominal_q is None:
        nominal_q = nominal_neck_angles(kn, swivels)
    tray_t = Cell.tray_in(pose_of(F_true))
    gt = hand_targets(tray_t[0], tray_t[1])
    kt = true_kin(state)
    delta = [float(x) for x in state.cal.delta]
    out = {"ok": True, "reason": None, "per_swivel": {}, "worst_axis_mm": 0.0, "worst_angle_deg": 0.0,
           "limits": {"mm": 1e3 * lim_p, "deg": lim_a}, "tray_drawn": [list(tray_d[0]), list(tray_d[1])]}
    for sw in swivels:
        row = {}
        for k, side in enumerate(SIDES):
            qn = nominal_q[(float(sw), side)]
            if qn is None:
                out["ok"] = False
                out["reason"] = f"the nominal kinematics does not solve the necks at swivel {sw}"
                continue
            q = np.asarray(qn, float) + np.array([delta[8 * k + j] for j in range(7)])
            p, R = kt.fk(side, q)
            p0, R0 = gt[side]
            e = R0.T @ (p - p0)
            qq = quat_of_matrix(R0.T @ R)
            if qq[0] < 0.0:
                qq = tuple(-x for x in qq)
            v = np.array(qq[1:4])
            ang = [math.degrees(2.0 * math.asin(min(1.0, abs(float(x))))) for x in v]
            row[side] = {"p_mm": [1e3 * float(x) for x in e], "angle_deg": ang}
            out["worst_axis_mm"] = max(out["worst_axis_mm"], 1e3 * float(np.max(np.abs(e))))
            out["worst_angle_deg"] = max(out["worst_angle_deg"], max(ang))
            if np.any(np.abs(e) > lim_p) or np.any(np.abs(v) > lim_v):
                out["ok"] = False
                out["reason"] = "fails the capture note"
        out["per_swivel"][str(float(sw))] = row
    return out


def nominal_neck_angles(kin_nominal, swivels=CAPTURE_NOTE_SWIVELS):
    tray_d = Cell.tray_in(Cell.F_DRAWN)
    ht = hand_targets(tray_d[0], tray_d[1])
    out = {}
    for sw in swivels:
        for side in SIDES:
            q, _ = kin_nominal.solve(side, ht[side][0], ht[side][1], float(sw))
            out[(float(sw), side)] = None if q is None else np.asarray(q, float)
    return out


class Monitor:
    SHELF_CARRIES_N = 2.0
    ALLOWED = frozenset(f"{a}|{b}" for r in ("F.base", "F.cup", "S", "F.cup1", "F.cup2", "F.cup3", "F.cup4")
                        for f in ("tray.feet", "tray.foot1", "tray.foot2", "tray.foot3", "tray.foot4")
                        for a, b in ((r, f), (f, r)))

    @classmethod
    def allowed(cls, key):
        return key in cls.ALLOWED

    def __init__(self, site, ot_C=K.OT_VALUE_C, shelf_exception="step", zero_end=False):
        if shelf_exception not in ("step", "latch"):
            raise ValueError("shelf_exception: 'step' or 'latch'")
        self.site = site
        self.last_tray = None
        self.mu = site["pad_friction"]
        self.ot = ot_C
        self.shelf_exception = shelf_exception
        self.zero_end = bool(zero_end)
        self.events = []
        self.breaches = []
        self._breach_on = {}
        self.load_max = {"left": 0.0, "right": 0.0}
        self.load_max_slot = {}
        self.load_on_shelf_max = {"left": 0.0, "right": 0.0}
        self.tw_max = None
        self.tilt_max_deg = 0.0
        self.stop_margin_min_deg = 1e9
        self.armed_from = {"left": None, "right": None}
        self.over = {"left": 0, "right": 0}
        self.held = {"left": False, "right": False}
        self.on_shelf = False
        self.shelf_spans = []
        self.slot = None
        self.te_step = None
        self.contacts_seen = {}
        self.unit_log = []
        self.cups = {}

    def set_slot(self, slot, te_step=None):
        self.slot = slot
        self.te_step = None if te_step is None else int(te_step)

    @staticmethod
    def shelf_normal(rec):
        d = rec.get("S.normal_N") or {}
        return float(sum(v for k, v in d.items() if "tray" in k))

    def _slot_event_step(self):
        for e in self.events:
            if e.get("slot") == self.slot:
                return e["step"]
        return None

    def _cups(self, rec, step):
        d = rec.get("F.normal_N")
        if not d or self.slot is None:
            return
        first_ev = self._slot_event_step()
        for k, v in d.items():
            if not k.startswith("F.cup") or "tray" not in k or not v > 0.0:
                continue
            e = self.cups.setdefault(self.slot, {}).setdefault(k, {
                "steps": 0, "first_step": int(step), "last_step": int(step), "max_N": 0.0, "max_step": int(step),
                "before_first_event": {"steps": 0, "first_step": None, "last_step": None, "max_N": 0.0,
                                       "max_step": None}})
            e["steps"] += 1
            e["last_step"] = int(step)
            if v > e["max_N"]:
                e["max_N"], e["max_step"] = float(v), int(step)
            if first_ev is None or step < first_ev:
                b = e["before_first_event"]
                b["steps"] += 1
                if b["first_step"] is None:
                    b["first_step"] = int(step)
                b["last_step"] = int(step)
                if v > b["max_N"]:
                    b["max_N"], b["max_step"] = float(v), int(step)

    def _breach(self, kind, step, side, active, **kw):
        key = (kind, side)
        if active and not self._breach_on.get(key):
            self.breaches.append(dict(kind=kind, slot=self.slot, side=side, step=int(step),
                                      **{k: _js(v) for k, v in kw.items()}))
        self._breach_on[key] = bool(active)

    def feed(self, rec, step):
        tw = np.asarray(rec["T_w"])
        self.tw_max = tw.copy() if self.tw_max is None else np.maximum(self.tw_max, tw)
        sm = float(np.min(rec["stop_margin"])) * 180.0 / math.pi
        self.stop_margin_min_deg = min(self.stop_margin_min_deg, sm)
        if self.zero_end:
            self._breach("E5", step, None, sm < K.STOP_MARGIN_DEG, margin_deg=sm)
        elif sm < K.STOP_MARGIN_DEG:
            self._ev("E5", step, margin_deg=sm)
        st = np.asarray(rec["status"])
        if np.any(st != 1):
            self._ev("E1", step, status=[int(x) for x in st])
        held_now = {s: int(rec[f"g_{s}.weld_state"]) > 0 for s in SIDES}
        S_N = self.shelf_normal(rec)
        if self.shelf_exception == "latch":
            if not self.on_shelf and any(held_now.values()) and S_N >= self.SHELF_CARRIES_N:
                self.on_shelf = True
                self.shelf_spans.append([int(step), None])
            if self.on_shelf and not any(held_now.values()):
                self.on_shelf = False
                self.shelf_spans[-1][1] = int(step)
            excepted = self.on_shelf
        else:
            excepted = (self.te_step is not None and step >= self.te_step and any(held_now.values())
                        and S_N >= self.SHELF_CARRIES_N)
            if excepted and not self.on_shelf:
                self.shelf_spans.append([int(step), None])
            if not excepted and self.on_shelf:
                self.shelf_spans[-1][1] = int(step)
            self.on_shelf = excepted
        any_held = False
        for s in SIDES:
            ws = int(rec[f"g_{s}.weld_state"])
            held = ws > 0
            if held and not self.held[s]:
                self.armed_from[s] = step + int(round(K.CAPACITY_RULE["armed_after_capture_s"] / DT))
            if not held and self.held[s]:
                self.armed_from[s] = None
            self.held[s] = held
            any_held = any_held or held
            if not held:
                self.over[s] = 0
                if self.zero_end:
                    self._breach("E2", step, s, False)
                continue
            w = np.asarray(rec[f"g_{s}.wrench"])
            Fc = float(rec[f"g_{s}.clamp_force"])
            Fs = math.hypot(float(w[0]), float(w[2]))
            Fn = abs(float(w[1]))
            mu = self.mu[s]
            share = max(Fs / (mu * Fc), Fn / Fc) if Fc > 0 else float("inf")
            if excepted:
                self.load_on_shelf_max[s] = max(self.load_on_shelf_max[s], share)
                self.over[s] = 0
                continue
            if self.armed_from[s] is not None and step >= self.armed_from[s]:
                self.load_max[s] = max(self.load_max[s], share)
                if self.slot is not None:
                    d = self.load_max_slot.setdefault(self.slot, {"left": 0.0, "right": 0.0})
                    d[s] = max(d[s], share)
                if share > 1.0:
                    self.over[s] += 1
                    if self.over[s] * DT >= K.CAPACITY_RULE["for_s"] - 1e-12:
                        if self.zero_end:
                            self._breach("E2", step, s, True, share=share)
                        else:
                            self._ev("E2", step, side=s, share=share)
                else:
                    self.over[s] = 0
                    if self.zero_end:
                        self._breach("E2", step, s, False)
        self.last_tray = np.asarray(rec["tray.pose"])[:3]
        tilt_on = False
        if any_held:
            q = np.asarray(rec["tray.pose"])[3:7]
            R = matrix_of_quat(q)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
            self.tilt_max_deg = max(self.tilt_max_deg, tilt)
            if tilt > K.SPILL_DEG:
                tilt_on = True
                if not self.zero_end:
                    self._ev("E3", step, tilt_deg=tilt)
            if self.zero_end:
                self._breach("E3", step, None, tilt_on, tilt_deg=tilt)
        elif self.zero_end:
            self._breach("E3", step, None, False)
        self._cups(rec, step)

    LOGGED = ("grasp_closed", "grasp_seated", "grasp_released", "block_met", "pads_left", "block_left",
              "capture_missed", "contact_begin", "fault_set", "object_placed", "object_parked",
              "plant_invalid")

    def presentation_contacts(self, contacts, step):
        for key, v in sorted(contacts.items()):
            if float(v.get("normal_N", 0.0)) <= 0.0 and int(v.get("points", 0)) == 0:
                continue
            if not self.allowed(key):
                self._ev("E4", step, pair=key, source="truth at presentation")

    def unit_events(self, evs):
        for e in evs:
            k = e.kind
            if k in self.LOGGED:
                self.unit_log.append({"kind": k, "step": int(e.step), "slot": self.slot,
                                      "fields": json.loads(json.dumps(dict(e.fields), default=_jsdefault))})
            if k == "contact_begin":
                pair = e.fields.get("pair") or e.fields.get("classes")
                key = pair if isinstance(pair, str) else "|".join(pair) if pair else json.dumps(e.fields, default=str)
                self.contacts_seen[key] = self.contacts_seen.get(key, 0) + 1
                if not self.allowed(key):
                    self._ev("E4", e.step, pair=key)
            elif k == "grasp_released" and self.last_tray is not None:
                feet = float(self.last_tray[2]) - Cell.GRIP_ABOVE_FEET
                ok = False
                for rest, half in (("S", Cell.S_HALF[:2]), ("F", (0.09, 0.215))):
                    rp = self.site[rest]["true"][0]
                    if abs(self.last_tray[0] - rp[0]) <= half[0] and abs(self.last_tray[1] - rp[1]) <= half[1] \
                            and feet - rp[2] <= K.DROP_FEET_ABOVE_REST_M:
                        ok = True
                if not ok:
                    self._ev("E2", e.step, drop=True, feet_z=feet)
            elif k in ("fault_set",):
                self._ev("E1", e.step, fields=dict(e.fields))

    def _ev(self, kind, step, **kw):
        if not self.events or self.events[-1]["kind"] != kind:
            self.events.append(dict(kind=kind, step=int(step), slot=self.slot, **{k: _js(v) for k, v in kw.items()}))


def _js(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_js(x) for x in v]
    return v


def _jsdefault(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _plain(v):
    return json.loads(json.dumps(v, default=_jsdefault))


class ReferenceAbort(RuntimeError):
    ""


TRACE_FIELDS = ("T_w", "stop_margin", "status", "tray.pose", "tray.twist", "g_left.weld_state", "g_right.weld_state",
                "g_left.wrench", "g_right.wrench", "g_left.clamp_force", "g_right.clamp_force", "S.normal_N")
CELL_TRACE_FIELDS = TRACE_FIELDS + ("F.normal_N",)


def trace_fields(cell=None):
    return CELL_TRACE_FIELDS if Cell._cell(cell) == "corrected" else TRACE_FIELDS


def supervisor_schedule(slots, jobs):
    cy = lambda t: int(round(t / CYCLE))
    out = []
    for i, sl in enumerate(slots):
        if sl["label"] == "decline":
            out.append({"slot": sl["slot"], "label": "decline", "length": None, "park": None})
            continue
        n = len(jobs[i]["targets"])
        out.append({"slot": sl["slot"], "label": sl["label"], "length": cy(K.slot_length_s(n)),
                    "park": cy(K.carry_end_s(n) + K.SLOT["retreat_end_after_Te"])})
    return out


def ambient_thermal(unit, ambient_C):
    d = unit.data
    unit.set_state(d.qpos.copy(), d.qvel.copy(), thermal=[(float(ambient_C), float(ambient_C))] * 16)


WARNINGS = ("mjWARN_BADQACC", "mjWARN_BADQVEL", "mjWARN_BADQPOS")


def warning_counts(unit):
    return {n: int(unit.data.warning[int(getattr(mujoco.mjtWarning, n))].number) for n in WARNINGS}


class IntegrationWatch:

    def __init__(self, unit):
        self.base = warning_counts(unit)

    def rebase(self, unit):
        self.base = warning_counts(unit)

    def check(self, unit, first_step, last_step):
        now = warning_counts(unit)
        rose = {k: now[k] - self.base[k] for k in now if now[k] > self.base[k]}
        for k in now:
            if now[k] < self.base[k]:
                self.base[k] = now[k]
        if not rose:
            return None
        info = {"kind": "integration", "step": int(first_step), "steps": [int(first_step), int(last_step)],
                "rose": rose, "counts": now,
                "last_info": {k: int(unit.data.warning[int(getattr(mujoco.mjtWarning, k))].lastinfo) for k in rose}}
        self.base = now
        return info


class HandProbe:
    BOX_LO = np.array([-0.075, -0.200, -0.060])
    BOX_HI = np.array([0.075, 0.200, 0.010])

    def __init__(self, unit, site):
        m = unit.model
        self.m = m
        self.d = mujoco.MjData(m)
        self.unit = unit
        self.site = site
        self.geoms = {}
        for s in SIDES:
            bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in base.ARM_BODIES[s][-3:]]
            self.geoms[s] = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bids and m.geom_contype[g]
                             and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]
        self.boxes = {}
        g = K.CLEAR_VOLUME["grow_m"]
        for rest in ("F", "S"):
            tp = Cell.tray_in(pose_of(site[rest]["true"]))
            self.boxes[rest] = (W.inverse(tp), self.BOX_LO - g, self.BOX_HI + g)

    def vertices(self, side):
        m, d = self.m, self.d
        out = []
        for g in self.geoms[side]:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            R = d.geom_xmat[g].reshape(3, 3)
            out.append(m.mesh_vert[a:a + n].astype(np.float64) @ R.T + d.geom_xpos[g])
        return np.vstack(out)

    def where(self):
        d = self.d
        d.qpos[:] = self.unit.data.qpos
        mujoco.mj_kinematics(self.m, d)
        out = {}
        for s in SIDES:
            V = self.vertices(s)
            ins = []
            for rest, (inv, lo, hi) in self.boxes.items():
                Rr = matrix_of_quat(inv[1])
                P = V @ Rr.T + np.asarray(inv[0])
                if np.any(np.all((P >= lo) & (P <= hi), axis=1)):
                    ins.append(rest)
            zmin = float(V[:, 2].min())
            out[s] = {"in_keepout": ins, "z_min_m": zmin,
                      "in_clear_volume": (not ins) and zmin >= K.CLEAR_VOLUME["z_min_m"]}
        return out


KEEPOUT_AT_REMOVAL = ("every", "missed")


def removal_contact(weld_states, where, keepout_at_removal="every", missed=False):
    held = any(h != "off" for h in weld_states)
    if keepout_at_removal == "every":
        return held or any(where[s]["in_keepout"] for s in SIDES)
    if keepout_at_removal == "missed":
        return bool(missed) and (held or not all(where[s]["in_clear_volume"] for s in SIDES))
    raise ValueError(f"keepout_at_removal: one of {KEEPOUT_AT_REMOVAL}")


def run_shift(unit, site, slots, reference, stop_at_event=True, hold_profile="A", max_cycles=None, log=None,
              jobs=None, ambient_at_presentation=None, recorder=None, shelf_exception="step", zero_end=False,
              keepout_at_removal="every"):
    import time
    from openarm_unit.state import HostProfile

    if keepout_at_removal not in KEEPOUT_AT_REMOVAL:
        raise ValueError(f"keepout_at_removal: one of {KEEPOUT_AT_REMOVAL}")
    t_wall0 = time.perf_counter()
    cpu0 = time.process_time()
    modes = reference.setup(unit, site, slots)["gripper_mode"]
    ctrl = {GRIPPER[s]: modes[s] for s in SIDES}
    hp = HostProfile.default(hold_profile, gripper_mode="force_position" if all(m == "posforce" for m in modes.values())
                             else "mit")
    th = np.zeros(16)
    th[3] = th[11] = K.HOME_POSE_RAD[3]
    th[7], th[15] = 0.45, -0.45
    replies = unit.reset(bus_angles=th, hold=hp, hold_s=K.HANDOVER_HOLD_S, ctrl_mode=ctrl)
    iw = IntegrationWatch(unit)
    r0 = unit.truth_trace()
    if recorder is not None:
        for r in r0:
            recorder.add(r)
    t_plan0 = time.perf_counter()
    reference.start(unit)
    t_plan = time.perf_counter() - t_plan0
    step0 = int(unit.bus.step_count)
    mon = Monitor(site, shelf_exception=shelf_exception, zero_end=zero_end)
    mon.unit_events(unit.events())
    hands = HandProbe(unit, site)
    jobs = reference.jobs if jobs is None else jobs
    sched = supervisor_schedule(slots, jobs)
    cy = lambda t: int(round(t / CYCLE))
    st_ = lambda t: int(round(t / DT))
    plan_len = getattr(reference, "slot_plan_lengths", lambda: None)()
    if plan_len is not None:
        for i, e in enumerate(sched):
            want = e["length"] if e["length"] is not None else cy(0.5 + K.SLOT["decline_next_s"])
            if e["length"] is not None and plan_len[i] != want:
                raise RuntimeError(f"slot {e['slot']}: the reference planned {plan_len[i]} cycles, the contract gives "
                                   f"{want}")
    next_present = cy(K.FREE_INTERVAL_S)
    i_next = 0
    t_ref = 0.0
    cyc = 0
    cur = None
    slot_start = None
    stopped = None
    figures = {"slots": []}
    presentations = []
    pending_park = None
    decline_seen = None
    carried = {}
    missed = {}
    declined_len = {}
    cap_step = {s: None for s in SIDES}
    put_back_at = None
    n_cycles = None if max_cycles is None else int(max_cycles)
    end_cycle = None if slots else next_present
    invalid = False
    cut_by_max = False
    while True:
        if n_cycles is not None and cyc >= n_cycles:
            cut_by_max = True
            break
        if end_cycle is not None and cyc >= end_cycle:
            break
        presented = False
        if i_next < len(slots) and cyc == next_present:
            i = i_next
            sl = slots[i]
            if ambient_at_presentation is not None:
                ambient_thermal(unit, ambient_at_presentation)
                iw.rebase(unit)
            n = unit_step(unit)
            ld = load_for(sl["mass_kg"], sl["along_m"], sl["fore_aft_m"], sl["below_m"])
            unit.world.set_load("tray", ld, n)
            unit.world.place("tray", Cell.tray_in(pose_of(site["F"]["true"])), n)
            cur, slot_start = i, cyc
            e = sched[i]
            te = None if e["length"] is None else n + st_(K.carry_end_s(len(jobs[i]["targets"])))
            mon.set_slot(sl["slot"], te)
            presentations.append({"slot": sl["slot"], "index": i, "cycle": cyc, "step": n,
                                  "label": sl["label"]})
            if te is not None:
                presentations[-1]["te_step"] = te
            decline_seen = None
            cap_step = {s: None for s in SIDES}
            if e["length"] is not None:
                pending_park = slot_start + e["park"]
                next_present = slot_start + e["length"]
            else:
                pending_park = None
                next_present = None
            i_next += 1
            presented = True
        if put_back_at is not None and cyc == put_back_at[0]:
            pr = presentations[put_back_at[1]]
            t = unit.truth(("world",)).world
            held = [str(t[f"g_{s}"]["weld_state"]) for s in SIDES]
            ok_f, rows = Cell.feet_in_cups(pose_of(t["tray"]["pose"]), pose_of(site["F"]["true"]))
            free = all(h == "off" for h in held)
            pr["put_back"] = {"cycle": cyc, "weld_states": held, "feet_in_cups": ok_f, "feet": rows,
                              "stands_released_in_F": bool(ok_f and free)}
            pr["declined"] = bool(pr["decline_record"]["both_captured_before"] and ok_f and free)
            put_back_at = None
        if cur is not None and slot_start is not None and cyc - slot_start == cy(K.SLOT["closing_end"]) \
                and decline_seen is None and cur not in missed and not all(cap_step[s] is not None for s in SIDES):
            missed[cur] = cy(K.SLOT["missed_next"])
            presentations[-1]["missed_capture"] = {"cycle": cyc, "captured": {s: cap_step[s] for s in SIDES}}
            pending_park = slot_start + cy(K.SLOT["missed_clear_end"])
            next_present = slot_start + cy(K.SLOT["missed_next"])
            decline_seen = cyc
            mon.set_slot(slots[cur]["slot"], None)
            presentations[-1].pop("te_step", None)
            if i_next >= len(slots):
                end_cycle = None
        if pending_park is not None and cyc == pending_park:
            t = unit.truth(("world",)).world
            held = [str(t[f"g_{s}"]["weld_state"]) for s in SIDES]
            where = hands.where()
            rem = {"cycle": cyc, "weld_states": held, "hands": where}
            rem["from"] = "F" if (cur in missed or presentations[-1].get("decline_cycle") is not None) else "S"
            rem["both_in_clear_volume"] = all(where[s]["in_clear_volume"] for s in SIDES)
            presentations[-1]["removal"] = rem
            if removal_contact(held, where, keepout_at_removal, missed=cur in missed):
                mon._ev("E4", unit_step(unit), keepout_at_removal=True, weld_states=held, removal_from=rem["from"],
                        hands={s: {"in_keepout": where[s]["in_keepout"], "z_min_m": where[s]["z_min_m"]}
                               for s in SIDES})
            if cur in missed:
                rec = slot_record(unit, slots[cur], site)
                rec["missed_capture"] = True
                figures["slots"].append(rec)
            elif cur in carried:
                figures["slots"].append(slot_record(unit, slots[cur], site, carried=True))
            else:
                figures["slots"].append(slot_record(unit, slots[cur], site))
            unit.world.park("tray", unit_step(unit))
            pending_park = None
            if i_next >= len(slots):
                end_cycle = next_present
        sig = {"tray": cur, "slot_cycle": None if cur is None else cyc - slot_start}
        if cur is not None and cur in missed:
            sig["capture_missed"] = True
        t0 = time.perf_counter()
        try:
            frames, decline = reference.step(cyc, sig, replies)
        except ReferenceAbort as ex:
            stopped = {"cycle": cyc, "error": {"name": type(ex).__name__, "message": str(ex)}}
            break
        except Exception as ex:
            if not zero_end:
                raise
            stopped = {"cycle": cyc, "error": {"name": type(ex).__name__, "message": str(ex)}}
            break
        t_ref += time.perf_counter() - t0
        if zero_end and not frames:
            stopped = {"cycle": cyc, "no_frame": True}
            break
        if cur is not None and decline_seen is None and (sched[cur]["length"] is None or decline):
            sc = cyc - slot_start
            win_end = cy(K.SLOT["decision_end"])
            if decline and sc < win_end:
                decline_seen = cyc
                pending_park = decline_seen + cy(K.SLOT["decline_clear_s"])
                next_present = decline_seen + cy(K.SLOT["decline_next_s"])
                presentations[-1]["decline_cycle"] = decline_seen
                declined_len[cur] = (decline_seen - slot_start) + cy(K.SLOT["decline_next_s"])
                mon.set_slot(slots[cur]["slot"], None)
                presentations[-1].pop("te_step", None)
                if i_next >= len(slots):
                    end_cycle = None
            elif sc >= win_end and sched[cur]["length"] is None:
                decline_seen = slot_start + win_end
                n_t = len(jobs[cur]["targets"])
                pending_park = slot_start + cy(K.carry_end_s(n_t) + K.SLOT["retreat_end_after_Te"])
                next_present = slot_start + cy(K.slot_length_s(n_t))
                presentations[-1]["decline_cycle"] = None
                presentations[-1]["carried"] = True
                carried[cur] = cy(K.slot_length_s(n_t))
                presentations[-1]["te_step"] = presentations[-1]["step"] + st_(K.carry_end_s(n_t))
                mon.set_slot(slots[cur]["slot"], presentations[-1]["te_step"])
        if decline and cur is not None and slot_start is not None and "decline_record" not in presentations[-1] \
                and cyc - slot_start < cy(K.SLOT["decision_end"]) and cur not in missed:
            n_now = unit_step(unit)
            cb = {s: (cap_step[s] is not None and cap_step[s] < n_now) for s in SIDES}
            presentations[-1]["decline_record"] = {"cycle": cyc, "step": n_now, "capture_steps": dict(cap_step),
                                                   "captured_before": cb, "both_captured_before": all(cb.values())}
            if presentations[-1].get("decline_cycle") == cyc:
                put_back_at = (cyc + cy(K.SLOT["decline_put_back_s"]), len(presentations) - 1)
        n_first = unit_step(unit)
        try:
            replies = unit.exchange(frames, steps=SPC, admit="control_only")
            recs = unit.truth_trace()
        except UnitInvalid as ex:
            mon.unit_events(unit.events())
            watch = iw.check(unit, n_first, n_first + SPC - 1)
            stopped = {"cycle": cyc + 1, "plant_invalid": {
                "kind": "plant_invalid", "step": None if getattr(ex, "step", None) is None else int(ex.step),
                "unit_kind": _plain(getattr(ex, "kind", None)), "kinds": _plain(getattr(ex, "kinds", None)),
                "info": _plain(getattr(ex, "info", None)), "message": str(ex), "watch": watch}}
            invalid = True
            cyc += 1
            break
        if recorder is not None:
            for r in recs:
                recorder.add(r)
        reference.on_trace(recs)
        for r in recs:
            if cur is not None:
                for s in SIDES:
                    if cap_step[s] is None and int(r[f"g_{s}.weld_state"]) > 0:
                        cap_step[s] = int(r["step"])
            mon.feed(r, r["step"])
        mon.unit_events(unit.events())
        if presented:
            mon.presentation_contacts(unit.truth(("contacts",)).contacts, presentations[-1]["step"])
        cyc += 1
        bad = iw.check(unit, n_first, unit_step(unit) - 1)
        if bad is not None:
            stopped = {"cycle": cyc, "integration": bad}
            break
        if mon.events and (stop_at_event or zero_end):
            stopped = {"cycle": cyc, "event": mon.events[0]}
            break
    planned = cy(K.FREE_INTERVAL_S)
    for i, e in enumerate(sched):
        if i in carried:
            planned += carried[i]
            continue
        if i in missed:
            planned += missed[i]
            continue
        if i in declined_len:
            planned += declined_len[i]
            continue
        planned += e["length"] if e["length"] is not None else cy(0.5 + K.SLOT["decline_next_s"])
    wall = time.perf_counter() - t_wall0
    cpu = time.process_time() - cpu0
    out = {"cycles": cyc, "planned_cycles": planned, "stopped": stopped, "events": mon.events,
           "load_over_capacity_max": max(mon.load_max.values()), "load_over_capacity": dict(mon.load_max),
           "load_over_capacity_by_slot": mon.load_max_slot, "load_on_shelf_max": dict(mon.load_on_shelf_max),
           "shelf_spans": mon.shelf_spans,
           "winding_max_C": None if mon.tw_max is None else [float(x) for x in mon.tw_max],
           "winding_margin_min_K": None if mon.tw_max is None else float(K.OT_VALUE_C - np.max(mon.tw_max)),
           "tilt_max_deg": mon.tilt_max_deg, "stop_margin_min_deg": mon.stop_margin_min_deg,
           "contacts_seen": mon.contacts_seen, "reference": reference.figures(), "slots": figures["slots"],
           "unit_events": mon.unit_log, "presentations": presentations, "step0": step0,
           "schedule": sched, "ambient_at_presentation": ambient_at_presentation,
           "wall_s": wall, "cpu_s": cpu, "reference_s": t_ref, "reference_plan_s": t_plan, "sim_s": cyc * CYCLE,
           "state_digest": None if invalid else unit.state_digest(),
           "motion_digest": None if invalid else unit.motion_digest()}
    if shelf_exception != "step":
        out["shelf_exception"] = shelf_exception
    if keepout_at_removal != "every":
        out["keepout_at_removal"] = keepout_at_removal
    if zero_end:
        out["zero_end"] = True
        out["breaches"] = mon.breaches
    if mon.cups:
        out["cup_walls"] = {str(k): v for k, v in mon.cups.items()}
    how = None
    if stopped is not None:
        how = next((k for k in ("plant_invalid", "integration", "event", "error", "no_frame") if k in stopped), "stop")
        if how == "event":
            how = str(stopped["event"].get("kind"))
    elif cut_by_max:
        how = "max_cycles"
    if how is not None and cur is not None:
        out["tray_in_work"] = {"slot": slots[cur]["slot"], "index": int(cur), "how": how}
    if invalid:
        out["plant_invalid"] = True
    return out


def unit_step(unit):
    return int(unit.bus.step_count)


def nominal_unit_state(unit_id="nominal-t05"):
    from openarm_unit import law as lawmod
    from openarm_unit import registers as rg
    from openarm_unit.state import (CalState, HSState, PlantState, SessionState, ThermalState, UnitState,
                                    HS_TABLE_ENTRIES, N_ARM_JOINTS)
    law = lawmod.law_params()
    p = public_params()

    def q(qid, key):
        return law["quantities"][qid][key]["value"]
    p1 = [float(x) for x in q("P1", "nominal_Nm")]
    p2 = [float(x) for x in q("P2", "nominal_Nm_s_rad")]
    p6 = float(q("P6", "nominal_Nm"))
    p7 = float(q("P7", "nominal_Nm_s_rad"))
    jr = p["conventions"]["joint_ranges_rad"]["value"]
    stops = [tuple(jr[s][j]) for s in SIDES for j in range(7)]
    ys = p["conventions"]["base_link_y_m"]["value"]
    bases = tuple(((0.0, float(ys[s]), 0.0), (1.0, 0.0, 0.0, 0.0)) for s in SIDES)
    plant = PlantState((p1 + [p6]) * 2, (p2 + [p7]) * 2, (), (), stops, bases)
    eds = dict(p["class_nominal_edition"]["value"])
    regs = rg.nominal_register_set(gripper_mode=rg.MODE_POS_FORCE, timeout=0)
    imax = [float(p["editions"][eds[m["class"]]]["Imax_A"]["value"]) for m in p["conventions"]["modules"]["value"]]
    z = [0.0] * 16
    cal = CalState(z, z, "nominal", [1] * 16, [1.0] * 16, [0.0] * 16, imax, regs)
    ones = (1.0,) * HS_TABLE_ENTRIES
    hs = HSState(0.0, (0.035,) * N_ARM_JOINTS, ((0.0, 0.0),) * N_ARM_JOINTS, (ones,) * N_ARM_JOINTS)
    st = UnitState(unit_id, law["law_version"], eds, plant, cal, hs, ThermalState([1.0] * 16))
    return st, SessionState.nominal("nominal")


def build_unit(state, session, site, noise_key=None, law=True, registry=None, policy="stage05", cell=None):
    from openarm_unit import law as lawmod
    from openarm_unit.unit import OpenArmUnit
    w = make_world(site, cell=cell)
    return OpenArmUnit.build(state, session, world=w, noise_key=noise_key,
                             law=lawmod.LawPlugins() if law else None, policy=policy,
                             truth_profile=trace_fields(cell), registry=registry)


def slot_record(unit, slot, site, carried=False):
    t = unit.truth(("world",)).world
    tp = t["tray"]["pose"]
    rest = "S" if (slot["label"] == "carry" or carried) else "F"
    seat = Cell.tray_in(pose_of(site[rest]["true"]))
    dxy = math.hypot(tp[0][0] - seat[0][0], tp[0][1] - seat[0][1])
    dz = tp[0][2] - seat[0][2]
    q = W.qmul(W.qconj(seat[1]), tuple(tp[1]))
    yaw = math.degrees(2.0 * math.atan2(q[3], q[0]))

    def normal(rest_name):
        c = t.get(rest_name, {}).get("contacts", {})
        return float(sum(v.get("normal_N", 0.0) for k, v in c.items() if "tray" in k))
    return {"slot": slot["slot"], "label": slot["label"], "rest": rest, "tray_pos": list(tp[0]),
            "seat_dxy_mm": 1e3 * dxy, "seat_dz_mm": 1e3 * dz, "seat_yaw_deg": yaw,
            "weld_states": [t[f"g_{s}"]["weld_state"] for s in SIDES],
            "clamp_N": [float(t[f"g_{s}"]["clamp_force_N"]) for s in SIDES],
            "captures": [int(t[f"g_{s}"]["captures"]) for s in SIDES],
            "S_normal_N": normal("S"), "F_normal_N": normal("F")}
