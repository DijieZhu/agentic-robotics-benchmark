"""The contact steps of the privileged reference program of task 07 (package P16; DESIGN v1.2 sections 1.1, 2.1,
2.2 H3 / H7 / H8, 2.3.2, 2.3.3, 3.3, 3.7; rulings M11, M13): the tide wait, the dropped object, the hot stab.

ContactProgram = steps_core.CoreProgram (P15: the core seven items) + four steps, so the run reaches all eleven items:
  * survey_low    the port row and the stand from the clear front (own frames): the as-built port of every port
                  plate (codes), the two lifting eyes of the stand (orange paint, the register's eye heights), the
                  stand's pose (centre = lift eye, outward normal = lift eye -> drag eye);
  * wait_window   replaces P15's: the contact work starts when the program's own free-drift measurements say the
                  panel current is <= WINDOW_GO_M_S (mean of the last WINDOW_AVG_S, at least WINDOW_MIN_N drifts),
                  or one measurement <= WINDOW_WEAK_M_S (weak flow);
  * move_debris   TRIAL LIFT on the lift eye where the stand stands (12 cm: the stand stays under the guard rail):
                  if the vehicle rose with it (its own pose + its own depth group) the stand is carried out from
                  under the rail, raised and set down in the laydown area; else it is set back, released, and the
                  drag eye is taken: pulled out along the stand's normal, then dragged along a gently turning path
                  (P02's v3) into the laydown area;
  * insert_stab   the stab's paddle in PARK-A (P15's ValveTurn approach: the panel standard + the as-built port read
                  from the program's own frames), unlatched by an outward pull and drawn out, carried to the leak
                  branch's test port and pushed in with a bounded axial force: the TRIAL PUSH.  Seated (the paddle
                  reached the seated position learnt in PARK-A) = the leak section is relieved; stopped short in the
                  seal stroke = still under pressure: wait and push again.  Latch confirmed by a gentle pull (< the
                  35 N release force) that does not move it.  Never reads a pressure.

The order after the window: close the isolation set (P15), check the plume decay (P15), move the stand (the leak
section bleeds meanwhile), insert the stab, back to the cage (P15).

Like steps_core, every DECISION is taken on the program's own data: the window on its DVL drifts, lift-or-drag on its
own vertical motion and depth-group readback, relieved-or-not on its own push travel, the ports and the eyes on its
own frames.  The true vehicle pose (moves.Cell's privileged block) only georeferences the frames and drives the servo.

The cell loads `reference.py` (run(server)); the entry of this program is `run` below (reference.run() of P15
instantiates steps_core.CoreProgram; record P2_P16 asks for the one-line switch).  Maintainer side: see main().
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))


def _paths():
    """Make moves / steps_core and the perception package importable (the cell's entry and the maintainer side;
    the same search as reference._paths)."""
    cand = [os.environ.get("SUBSEACELL_PERCEPTION_DIR"), HERE, os.path.dirname(HERE)]
    for c in cand:
        if c and os.path.isfile(os.path.join(c, "perception", "readers.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            break
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


_paths()
from perception import readers as RD  # noqa: E402

import moves as MV  # noqa: E402
import steps_core as SC  # noqa: E402

# ------------------------------------------------------------------------------------------------- public drawings
# panel standard (panel_standard.md): the port row, the port plates, the stab and its receptacles, the laydown area
PORTS_X = SC.PORTS_X
PORT_Y = SC.PORT_Y
PORT_NAMES = ("PARK-A", "TP1", "TP2", "TP3", "TP4", "TP5", "TP6", "PARK-B")       # the plate codes' tags
STAB_PADDLE_Z = 0.28            # seated stab: tip at the seat (face plane), paddle centre 0.28 m out
STAB_MOUTH_Z = 0.12             # receptacle mouth 0.12 m out of the face, insertion depth 0.12 m
LAYDOWN_X = (-1.5, 1.5)
LAYDOWN_Z = (2.5, 4.5)
LAY_TARGET_Z = 3.5
LAY_X_CLIP = 0.8                # the 1 m long stand stays inside the 3 m wide area
# dropped-object register (dropped_object_register.md): one shape for D1-D4
EYE_H = dict(drag=0.22, lift=0.8436)       # eye centres above the mudmat
RING_R, RING_TUBE = 0.045, 0.0035          # eye ring radius (inner diameter 80 mm) and its round bar
EYE_SEP = 0.33                              # lift eye (top centre) -> drag eye (outside the outer skid), horizontal
STAND_HALF = (0.5, 0.25, 0.65)              # half length, half width, height

# ------------------------------------------------------------------------------------------------- program constants
WINDOW_GO_M_S = 0.29            # mean of the recent drifts that opens the contact work (limit 0.30, DESIGN 2.1)
WINDOW_WEAK_M_S = 0.15
WINDOW_AVG_S = 150.0
WINDOW_MIN_N = 5
WINDOW_GIVE_UP_S = 1000.0
SURVEY_XS = (-1.8, -0.6, 0.6, 1.8)
SURVEY_CAM = (1.15, 1.50, -25.0)            # camera height, distance from the face, tilt (deg)
LOOK_CAM = (0.55, 1.65, 0.0)                # the stand's close look: camera height, distance, tilt
LOOK_DX = (-0.35, 0.35)
LIFT_TRIAL_M = 0.12
LIFT_OK_M = 0.08
CARRY_TCP_H = 1.35
STAB_F_TRY = 40.0               # axial push of the trial (latches up to (40 - 15) / 2.5 = 10 bar)
STAB_F_HARD = 48.0
STAB_PULL_TEST = 15.0           # < the 35 N release force
STAB_WAIT_S = 15.0
STAB_GIVE_UP_S = 360.0
ORANGE_DRG, ORANGE_DRB, ORANGE_MIN_R = 22, 30, 55
VERBOSE = SC.VERBOSE


# ======================================================================================================= small helpers
def face_point(sc: MV.Scene, cam, uv, z_off=0.0):
    """Panel-frame (X, Y) where the pixel ray through uv meets the face plane Z = z_off (None if behind)."""
    d = cam.rays(np.array([uv[0]]), np.array([uv[1]]))[0]
    Z = sc.P_R[:, 2]
    den = float(d @ Z)
    if abs(den) < 1e-6:
        return None
    s = float((sc.P_o + Z * z_off - cam.p) @ Z) / den
    if s <= 0:
        return None
    return sc.w_to_P(cam.p + s * d)[:2]


def ray_at_height(sc: MV.Scene, cam, uv, h):
    """World point where the pixel ray meets the horizontal plane h above the mudmat (None if not ahead)."""
    d = cam.rays(np.array([uv[0]]), np.array([uv[1]]))[0]
    z = float(sc.P_o[2]) + float(h)
    if abs(d[2]) < 1e-6:
        return None
    s = (z - float(cam.p[2])) / float(d[2])
    if s <= 0:
        return None
    return cam.p + s * d


# ======================================================================================================= port plates
class PortBook:
    """The program's own picture of the port row: port tag -> as-built position index (votes over frames)."""

    def __init__(self, scene: MV.Scene):
        self.sc = scene
        self.code_c = {i: np.array([PORTS_X[i] + SC.CODE_DX, PORT_Y + SC.PLATE_ABOVE]) for i in range(8)}
        self.votes = {}
        self.reads = []

    def read(self, rgb, cam, t=None):
        seen = {}
        for cid, corners in RD.decode(rgb):
            tag = RD.TAG_OF_CODE.get(int(cid))
            if tag not in PORT_NAMES:
                continue
            xy = face_point(self.sc, cam, np.asarray(corners, float).mean(0), SC.PLATE_THK)
            if xy is None:
                continue
            k, dd = min(((i, float(np.linalg.norm(c - xy))) for i, c in self.code_c.items()), key=lambda x: x[1])
            if dd > 0.15:
                continue
            v = self.votes.setdefault(tag, {})
            v[k] = v.get(k, 0) + 1
            seen[tag] = k
            self.reads.append(dict(t=t, tag=tag, pos=k, err_m=round(dd, 3)))
        return seen

    def layout(self) -> dict:
        """tag -> position index, one tag per position (more votes win); a missing tag is filled only when the
        others leave exactly one position (never guessed from the drawing)."""
        cand = sorted((-n, tag, k) for tag in self.votes for k, n in self.votes[tag].items())
        out, used = {}, set()
        for n, tag, k in cand:
            if tag in out or k in used:
                continue
            out[tag] = k
            used.add(k)
        missing = [t for t in PORT_NAMES if t not in out]
        free = [k for k in range(8) if k not in used]
        if len(missing) == 1 and len(free) == 1:
            out[missing[0]] = free[0]
        return out


# ======================================================================================================= lifting eyes
def orange_mask(rgb):
    r, g, b = (rgb[..., k].astype(np.int16) for k in range(3))
    return (r - g >= ORANGE_DRG) & (r - b >= ORANGE_DRB) & (r >= ORANGE_MIN_R)


def project_box_mask(cam, shape, centre_w, R_w, half, pad_px=6):
    """Pixels covered by the projection of a box (centre, axes R_w columns, half sizes) - a convex hull of the
    corners' projections, dilated."""
    H, W = shape
    c = np.asarray(centre_w, float)
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(c + np.asarray(R_w, float) @ (np.array([sx, sy, sz]) * np.asarray(half, float)))
    uv, z = cam.project(np.array(pts))
    if np.all(z <= 0.05):
        return np.zeros(shape, bool)
    uv = uv[z > 0.05]
    u0, u1 = int(max(0, np.floor(uv[:, 0].min() - pad_px))), int(min(W, np.ceil(uv[:, 0].max() + pad_px)))
    v0, v1 = int(max(0, np.floor(uv[:, 1].min() - pad_px))), int(min(H, np.ceil(uv[:, 1].max() + pad_px)))
    m = np.zeros(shape, bool)
    if u1 > u0 and v1 > v0:
        m[v0:v1, u0:u1] = True
    return m


class EyeBook:
    """Rays to the two eyes' ring tops from the program's own frames; least-squares eye centres with the register's
    heights as a prior.  An orange component is an eye candidate when its top ray meets that eye's ring-top plane in
    the band the register and the placement rule allow (lift: 0.30-0.95 m off the face, drag: 0.55-1.30 m)."""

    BANDS = dict(lift=(0.30, 0.95), drag=(0.55, 1.30))
    SIGMA_PX = 1.5
    SIGMA_H = 0.004

    def __init__(self, scene: MV.Scene):
        self.sc = scene
        self.rays = dict(lift=[], drag=[])          # (origin, dir, sigma_m, t)
        self.dets = []

    def top_h(self, kind):
        return EYE_H[kind] + RING_R + RING_TUBE

    def read(self, rgb, cam, t=None, exclude=None, kinds=("lift", "drag"), near=None, near_tol=0.35):
        """Orange components -> candidate rays.  `near` = {kind: world xy} keeps only candidates within near_tol."""
        m = orange_mask(rgb)
        if exclude is not None:
            m &= ~exclude
        lab, n = ndimage.label(m)
        if n == 0:
            return {}
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(np.ones_like(lab), lab, idx)
        found = {}
        H, W = m.shape
        for k, sz in zip(idx, sizes):
            if sz < 6:
                continue
            vv, uu = np.nonzero(lab == k)
            vt = int(vv.min())
            if vt <= 1:
                continue                                  # touching the top edge: the ring top may be cut
            top = uu[vv <= vt + 1]
            uv = (float(np.median(top)) + 0.5, vt + 0.5)
            rng_est = None
            for kind in kinds:
                p = ray_at_height(self.sc, cam, uv, self.top_h(kind))
                if p is None:
                    continue
                q = self.sc.w_to_P(p)
                lo, hi = self.BANDS[kind]
                if not (lo <= q[2] <= hi and abs(q[0]) <= 2.75):
                    continue
                if near is not None and kind in near and near[kind] is not None and \
                        float(np.linalg.norm(p[:2] - np.asarray(near[kind])[:2])) > near_tol:
                    continue
                rng_est = float(np.linalg.norm(p - cam.p))
                d = cam.rays(np.array([uv[0]]), np.array([uv[1]]))[0]
                cand = dict(kind=kind, uv=[round(uv[0], 1), round(uv[1], 1)], n_px=int(sz), p=p, range=rng_est,
                            o=cam.p.copy(), d=d)
                if kind not in found or found[kind]["n_px"] < sz:
                    found[kind] = cand
        for kind, c in found.items():
            sig = self.SIGMA_PX * c["range"] / float(cam.K[0, 0])
            self.rays[kind].append((c["o"], c["d"], sig, t))
            q = self.sc.w_to_P(c["p"])
            self.dets.append(dict(t=t, kind=kind, uv=c["uv"], n_px=c["n_px"], range_m=round(c["range"], 3),
                                  P=[round(float(x), 3) for x in q]))
        return found

    def estimate(self, kind):
        """Least-squares eye centre (world) from its rays and the height prior; None without rays."""
        R = self.rays[kind]
        if not R:
            return None
        A, b = [], []
        z0 = float(self.sc.P_o[2]) + EYE_H[kind]
        off = np.array([0.0, 0.0, RING_R + RING_TUBE])
        for o, d, sig, _ in R:
            P = np.eye(3) - np.outer(d, d)                # perpendicular distance of c + off from the ray
            A.append(P / sig)
            b.append(P @ (o - off) / sig)
        A.append(np.array([[0.0, 0.0, 1.0 / self.SIGMA_H]]))
        b.append(np.array([z0 / self.SIGMA_H]))
        A, b = np.vstack(A), np.concatenate(b)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        return c

    def stand(self):
        """(centre_w (xy at the mudmat), outward normal n_w (horizontal unit), lift_w, drag_w, info) or None."""
        L, D = self.estimate("lift"), self.estimate("drag")
        if L is None or D is None:
            return None
        v = (D - L)[:2]
        sep = float(np.linalg.norm(v))
        n = np.r_[v / max(sep, 1e-9), 0.0]
        Zp = self.sc.P_R[:, 2]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(n @ Zp)))))
        info = dict(sep_m=round(sep, 3), normal_vs_face_deg=round(ang, 1), n_lift=len(self.rays["lift"]),
                    n_drag=len(self.rays["drag"]), lift_P=[round(float(x), 3) for x in self.sc.w_to_P(L)],
                    drag_P=[round(float(x), 3) for x in self.sc.w_to_P(D)])
        if not (0.22 <= sep <= 0.44) or ang > 32.0:
            info["rejected"] = True
            return dict(ok=False, info=info)
        return dict(ok=True, centre=L.copy(), n=n, lift=L, drag=D, info=info)


# ======================================================================================================= the stab grasp
class PaddleGrip(MV.ValveTurn):
    """P15's ValveTurn approach on the stab's paddle (the same blade); the pre-grasp moves are straight (the caller
    has put the vehicle on the port's normal at the clear front)."""

    def prepare(self, standoff=0.5, T_settle=1.5, T_measure=4.0):
        tp, ty = self.grip_pose(self.bite)
        pre = tp - self.x_in * standoff
        self.tp, self.ty, self.pre = tp, ty, pre
        self.mark("pre")
        self.mv.go(pre - self.x_in * 0.4, ty, v=0.20, tol=0.06, tol_z=0.05, settle_s=0.5, route=False)
        self.mv.go(pre, ty, v=0.08, tol=0.02, tol_z=0.02, settle_s=T_settle, route=False)
        self.mark("measure")
        acc, n, db, yb = np.zeros(3), 0, 0.0, 0.0
        for _ in range(int(round(T_measure / 0.05))):
            self.mv.wait(0.05)
            s = self.cell.state()
            if s.get("quat_wxyz") is None or s.get("depth_m") is None:
                continue
            wg = s["wrench_groups"]
            acc += MV.quat_to_mat(s["quat_wxyz"]) @ (np.asarray(wg["depth"][:3], float) + np.asarray(wg["translation"][:3], float))
            n += 1
            tr = self.cell.truth()
            db += float(s["depth_m"]) - (-float(tr.pos[2]))
            yb += MV.wrap(float(s["heading_rad"]) - tr.yaw)
        n = max(n, 1)
        self.F_w = acc / n
        self.depth_bias, self.yaw_bias = db / n, yb / n
        self.out.update(hover_force_w=[round(float(x), 3) for x in self.F_w], depth_bias_m=round(self.depth_bias, 4),
                        yaw_bias_deg=round(math.degrees(self.yaw_bias), 3), n_measure=n)

    # ---------------------------------------------------------------- explicit servo on the TRUE pose (wrench mode)
    def servo_loop(self, ref_fn, T, axial=None, done=None, f_cap=60.0):
        """Run T seconds of the level position servo (Fx, Fy open in world axes, firmware depth / attitude hold):
        ref_fn(t) -> body xy reference; axial(t, f_ax_servo) -> the force ALONG x_in actually commanded (the servo's
        axial part is replaced by it; lateral part kept); done(t) -> True ends the loop.  Returns the time spent."""
        F_h = np.array(self.F_w, float)
        a = self.x_in[:2] / max(np.linalg.norm(self.x_in[:2]), 1e-9)
        q_lvl = MV.euler_quat(0.0, 0.0, self.ty + self.yaw_bias).tolist()
        d_t = self.d_hold
        t0 = self.cell.clock()
        ei = np.zeros(2)
        while True:
            t = self.cell.clock() - t0
            if t >= T:
                break
            tr = self.cell.truth()
            e = ref_fn(t) - tr.pos[:2]
            ei = np.clip(ei + e * MV.CTRL_S, -0.2, 0.2)
            Fs = self.SERVO_KP * e - self.SERVO_KD * tr.vel_w[:2] + self.SERVO_KI * ei
            nF = float(np.linalg.norm(Fs))
            if nF > f_cap:
                Fs *= f_cap / nF
            if axial is not None:
                f_ax = float(Fs @ a)
                Fs = Fs - f_ax * a + float(axial(t, f_ax)) * a
            F = F_h.copy()
            F[:2] = F_h[:2] + Fs
            self._call("set_wrench", force=F.tolist(), torque=[0.0, 0.0, 0.0], free=list(MV.FREE_XY), frame="world",
                       quat=q_lvl, depth=d_t)
            self._step()
            if done is not None and done(t):
                break
        return self.cell.clock() - t0


# ======================================================================================================= the program
class ContactProgram(SC.CoreProgram):
    SEQUENCE = ("sensors_first", "leave_cage", "find_structure", "inspect", "locate_leak", "place_marker",
                "survey_low", "wait_window", "close_isolation", "decay_check", "move_debris", "insert_stab",
                "return_to_cage")
    NEEDS_BRANCH = ("close_isolation", "decay_check", "insert_stab")

    def __init__(self, server, opts=None):
        super().__init__(server, opts)
        self.ports = PortBook(self.sc)
        self.eyes = EyeBook(self.sc)
        self.port_layout = {}
        self.stand_est = None
        self.a_B = np.asarray(self.md["axis_B"], float)
        self.extra_keepout = []              # panel-frame boxes (X0, X1, Z0, Z1, Ytop) the routes must avoid
        self.t_isolated = None
        self.last_mark = None
        self.out["move_timeouts"] = []
        go1 = self.mv._go1

        def _go1_logged(pos, yaw, v, tol, tol_z, settle_s, timeout, _go1=go1):
            t0 = self.cell.clock()
            ok = _go1(pos, yaw, v, tol, tol_z, settle_s, timeout)
            if not ok:
                tr = self.cell.truth()
                self.out["move_timeouts"].append(dict(step=self.last_mark, t0=round(t0, 1), dur_s=round(self.cell.clock() - t0, 1),
                                                      tol=tol, tol_z=tol_z,
                                                      err_m=round(float(np.linalg.norm(np.asarray(pos)[:2] - tr.pos[:2])), 4),
                                                      err_z_m=round(abs(float(pos[2]) - float(tr.pos[2])), 4)))
            return ok
        self.mv._go1 = _go1_logged

    def mark(self, key):
        self.last_mark = key
        super().mark(key)

    # ------------------------------------------------------------------ small motion helpers (true pose, wrench mode)
    def yaw_for_axis(self, u_w):
        return math.atan2(u_w[1], u_w[0]) - math.atan2(self.a_B[1], self.a_B[0])

    def body_for_tcp(self, p_tcp, yaw):
        return np.asarray(p_tcp, float) - MV.rz(yaw) @ self.md["tcp_B"]

    def tcp_now(self):
        return self.cell.truth().tcp.copy()

    def go_tcp(self, p_tcp, yaw, v=0.15, tol=0.03, tol_z=0.03, settle_s=0.5, timeout=None):
        return self.mv.go(self.body_for_tcp(p_tcp, yaw), yaw, v=v, tol=tol, tol_z=tol_z, settle_s=settle_s,
                          timeout=timeout, route=False)

    def path_tcp(self, p_tcp, yaw, v):
        """Move the servo's reference along a straight line to the TCP target at v (no settling wait): carrying,
        dragging, slow lifts."""
        mv = self.mv
        mv.start()
        o = self.body_for_tcp(p_tcp, yaw)
        zr = mv.z_rate
        mv.z_rate = max(min(zr, v), 0.01)
        mv.goto(o, yaw, v)
        dist = max(float(np.linalg.norm(o[:2] - mv.ref)) / max(mv._v, 1e-4), abs(o[2] - mv.z_ref) / max(mv._zr, 1e-4))
        t0 = self.cell.clock()
        while self.cell.clock() - t0 < dist + 1.0:
            self.cell.step(0.1)
        mv.z_rate = zr
        return self.cell.clock() - t0

    def front_point(self, X, h_body, Z=SC.FRONT_CLEAR):
        return self.sc.P_to_w([float(X), float(h_body), float(Z)])

    def body_h(self, p_w):
        return float(self.sc.w_to_P(p_w)[1])

    def go_front(self, X, h_body, yaw=None, Z=SC.FRONT_CLEAR, v=0.30):
        """To the clear front (Z = FRONT_CLEAR) at panel X, body height h_body: first out to the clear front at the
        current height (straight, along the normal), then over or under to the target (straight; the caller keeps
        the target above anything on the mudmat there)."""
        sc = self.sc
        yaw = sc.face_yaw if yaw is None else yaw
        tr = self.cell.truth()
        q = sc.w_to_P(tr.pos)
        if q[2] < 0.0 or abs(q[0]) > SC.SKID_HALF_W + 0.6:
            # behind the face (over the structure) or off to the side: P15's keep-out router to the clear front
            self.mv.go(sc.P_to_w([X, max(q[1], float(h_body)), Z]), yaw, v=v, tol=0.12, tol_z=0.08, settle_s=0.2)
            q = sc.w_to_P(self.cell.truth().pos)
        elif q[2] < Z - 0.05:
            self.mv.go(sc.P_to_w([q[0], q[1], Z]), yaw, v=0.15, tol=0.08, tol_z=0.06, settle_s=0.2, route=False)
            q = sc.w_to_P(self.cell.truth().pos)
        h_cross = max(float(q[1]), float(h_body))
        if self.extra_keepout:
            h_cross = max(h_cross, max(b[4] for b in self.extra_keepout) + 0.75)
        if abs(q[1] - h_cross) > 0.05:
            self.mv.go(sc.P_to_w([q[0], h_cross, max(q[2], Z)]), yaw, v=v, tol=0.1, tol_z=0.06, settle_s=0.1, route=False)
        self.mv.go(sc.P_to_w([X, h_cross, Z]), yaw, v=v, tol=0.08, tol_z=0.06, settle_s=0.2, route=False)
        return self.mv.go(sc.P_to_w([X, h_body, Z]), yaw, v=0.20, tol=0.05, tol_z=0.04, settle_s=0.3, route=False)

    def cam_body(self, X, Y, Z, tilt_deg, yaw=None):
        yaw = self.sc.face_yaw if yaw is None else yaw
        p_c, _ = self.man.camera_in_body(math.radians(tilt_deg))
        return self.sc.P_to_w([X, Y, Z]) - MV.rz(yaw) @ p_c, yaw

    def snap(self, tilt_deg, lights=1.0):
        self.sens.set_tilt(tilt_deg)
        self.sens.set_lights(lights)
        rgb, cam, info = self.sens.frame((640, 480))
        return rgb, cam, info

    # ------------------------------------------------------------------ 5b. the port row and the stand (no contact)
    def stab_paddle_box(self):
        """(centre_w, R_w, half) of the parked stab's paddle + neck, for masking its orange half (the stab sits in
        PARK-A until it is taken)."""
        k = self.port_layout.get("PARK-A")
        if k is None:
            return None
        return (self.sc.P_to_w([PORTS_X[k], PORT_Y, STAB_PADDLE_Z]), self.sc.P_R, (0.11, 0.06, 0.08))

    def survey_low(self):
        """Port plates (codes) and the stand's eyes from the clear front; then two close looks at the stand from
        just above the mudmat (both eyes in the lamps), from either side of it."""
        self.mark("survey_low")
        sc = self.sc
        h_c, Zc, tilt = SURVEY_CAM
        tr = self.cell.truth()
        xs = list(SURVEY_XS) if sc.w_to_P(tr.pos)[0] <= 0 else list(SURVEY_XS)[::-1]
        frames = []
        for i, X in enumerate(xs):
            pos, yaw = self.cam_body(X, h_c, Zc, tilt)
            if i == 0:
                self.mv.go(pos, yaw, v=0.40, tol=0.05, tol_z=0.04, settle_s=0.3)      # routed from the roof
            else:
                self.mv.go(pos, yaw, v=0.30, tol=0.04, tol_z=0.03, settle_s=0.3, route=False)
            rgb, cam, info = self.snap(tilt, 0.8)
            self.ports.read(rgb, cam, t=info["t"])
            frames.append((rgb, cam, info))
        self.port_layout = self.ports.layout()
        box = self.stab_paddle_box()
        for rgb, cam, info in frames:
            ex = None if box is None else project_box_mask(cam, rgb.shape[:2], *box)
            self.eyes.read(rgb, cam, t=info["t"], exclude=ex, kinds=("lift",))
        L = self.eyes.estimate("lift")
        rec = dict(port_layout={k: int(v) for k, v in sorted(self.port_layout.items())},
                   port_reads=len(self.ports.reads), lift_rays_survey=len(self.eyes.rays["lift"]))
        if L is not None:
            Xl = float(sc.w_to_P(L)[0])
            near = dict(lift=L[:2])
            h_c2, Z2, t2 = LOOK_CAM
            for dx in LOOK_DX:
                X = float(np.clip(Xl + dx, -2.4, 2.4))
                pos, yaw = self.cam_body(X, h_c2, Z2, t2)
                self.mv.go(pos, yaw, v=0.25, tol=0.04, tol_z=0.03, settle_s=0.4, route=False)
                rgb, cam, info = self.snap(t2, 1.0)
                ex = None if box is None else project_box_mask(cam, rgb.shape[:2], *box)
                self.eyes.read(rgb, cam, t=info["t"], exclude=ex, near=near, near_tol=0.6)
            st = self.eyes.stand()
            self.stand_est = st
            rec["stand"] = None if st is None else st["info"]
        else:
            self.note("survey_low: no lifting eye in the port-row frames")
        rec["eye_dets"] = list(self.eyes.dets)
        self.out["survey_low"] = rec
        # back up to the clear front at the survey height (the window drifts start from there)
        tr = self.cell.truth()
        q = sc.w_to_P(tr.pos)
        self.mv.go(sc.P_to_w([q[0], max(q[1], 1.6), max(q[2], SC.FRONT_CLEAR)]), sc.face_yaw, v=0.3, tol=0.1,
                   tol_z=0.08, settle_s=0.1, route=False)
        self.mark("survey_low_done")

    # ------------------------------------------------------------------ 6. contact window (own drifts)
    @staticmethod
    def window_decision(meas, t):
        """meas = [(t_i, speed_i)] of the program's own drifts.  Open when the first drift is <= WINDOW_WEAK_M_S, or
        when the mean of the drifts of the last WINDOW_AVG_S (at least WINDOW_MIN_N) is <= WINDOW_GO_M_S."""
        if len(meas) == 1 and meas[0][1] <= WINDOW_WEAK_M_S:
            return dict(open=True, estimate=float(meas[0][1]), n=1, rule="weak")
        recent = [m for m in meas if t - m[0] <= WINDOW_AVG_S]
        if len(recent) >= WINDOW_MIN_N:
            est = float(np.mean([m[1] for m in recent]))
            return dict(open=est <= WINDOW_GO_M_S, estimate=est, n=len(recent), rule="mean")
        return dict(open=False, estimate=None, n=len(recent), rule="mean")

    def wait_window(self):
        self.mark("window")
        meas = []
        dec = dict(open=False, estimate=None, n=0, rule=None)
        t = self.cell.clock()
        while True:
            sp = self.measure_current()
            t = self.cell.clock()
            if sp is not None:
                meas.append((t, sp))
            dec = self.window_decision(meas, t)
            if dec["open"]:
                self.note(f"contact window: {dec['rule']} estimate {dec['estimate']:.3f} m/s from {dec['n']} drifts")
                break
            if t >= WINDOW_GIVE_UP_S:
                self.note(f"contact window: estimate {dec['estimate']} at the give-up time; proceeding")
                break
        self.out["window"] = dict(n=len(meas), estimate=None if dec["estimate"] is None else round(dec["estimate"], 4),
                                  n_used=dec["n"], rule=dec["rule"], opened=bool(dec["open"]), t=round(t, 1))
        self.mark("window_open")

    def close_isolation(self):
        super().close_isolation()
        self.t_isolated = self.cell.clock()

    # ------------------------------------------------------------------ 9. the dropped object
    def grab_eye(self, kind, st, tries=3):
        """Approach eye `kind` along the stand's outward normal (pre-grasp 0.55 m out, one look to refine) and close
        the gripper.  True when the gripper reports an object."""
        n = st["n"]
        yaw = self.yaw_for_axis(-n)
        rec = dict(kind=kind, attempts=0)
        for k in range(tries):
            rec["attempts"] = k + 1
            e = self.eyes.estimate(kind)
            pre = e + n * 0.55
            self.go_tcp(pre, yaw, v=0.15, tol=0.02, tol_z=0.02, settle_s=0.5)
            rgb, cam, info = self.snap(0.0 if kind == "lift" else -5.0, 1.0)
            got = self.eyes.read(rgb, cam, t=info["t"], kinds=(kind,), near={kind: e[:2]}, near_tol=0.12)
            e = self.eyes.estimate(kind)
            rec.setdefault("refine_seen", []).append(kind in got)
            self.go_tcp(e + n * 0.12, yaw, v=0.10, tol=0.01, tol_z=0.01, settle_s=0.3)
            self.go_tcp(e, yaw, v=0.05, tol=0.008, tol_z=0.008, settle_s=0.6, timeout=20.0)
            tr = self.cell.truth()
            rec.setdefault("tcp_minus_est_mm", []).append([round(1000 * float(x), 1) for x in (tr.tcp - e)])
            self.cell.call("gripper_close")
            g = self.wait_gripper(("stalled_on_object", "closed_empty"), 4.0)
            rec.setdefault("gripper", []).append(g)
            if g == "stalled_on_object":
                rec["ok"] = True
                return True, rec
            self.cell.call("gripper_open")
            self.wait_gripper(("open",), 3.0)
            self.go_tcp(e + n * 0.30, yaw, v=0.08, tol=0.03, tol_z=0.03, settle_s=0.3)
        rec["ok"] = False
        return False, rec

    def wait_gripper(self, want, timeout):
        t0 = self.cell.clock()
        g = None
        while self.cell.clock() - t0 < timeout:
            self.mv.wait(0.1)
            g = self.cell.state()["gripper"]
            if g in want:
                return g
        return g

    def release_back(self, n, back=0.4, up=0.2):
        self.cell.call("gripper_open")
        self.wait_gripper(("open",), 3.0)
        p = self.tcp_now() + n * back + np.array([0.0, 0.0, up])
        self.go_tcp(p, self.mv.yaw_goal, v=0.12, tol=0.06, tol_z=0.06, settle_s=0.2, timeout=25.0)

    @staticmethod
    def lift_decision(rise_m, depth_scale, pitch_deg, aborted=None):
        """The trial lift's verdict from the program's own data: it rose with the stand (its own pose), the depth
        group was not cut by the allocator (its own state() readback), the attitude held, the trial was not cut
        short (saturation or drift)."""
        return bool(aborted is None and rise_m >= LIFT_OK_M and (depth_scale is None or depth_scale >= 0.97)
                    and abs(pitch_deg) < 8.0)

    def trial_lift(self):
        """With the lift eye held: climb LIFT_TRIAL_M slowly.  Liftable when the vehicle followed its reference with
        the depth group never cut by the allocator.  The moment the depth group saturates (the stand is too heavy:
        the allocator then cuts the horizontal forces first and the half-unloaded stand would be swept along by the
        current) the reference goes straight back down: the trial is over."""
        tr0 = self.cell.truth()
        z0 = float(tr0.pos[2])
        xy0 = tr0.pos[:2].copy()
        mv = self.mv
        mv.start()
        mv.z_rate = 0.03
        mv.goto(np.r_[mv.goal, z0 + LIFT_TRIAL_M], mv.yaw_goal)
        t0 = self.cell.clock()
        scale_min, sat, n_bad, aborted, drift = 1.0, 0, 0, None, 0.0
        while self.cell.clock() - t0 < LIFT_TRIAL_M / 0.03 + 4.0:
            self.cell.step(0.1)
            s = self.cell.state()
            gs = (s.get("group_scale") or {}).get("depth")
            if gs is not None:
                scale_min = min(scale_min, float(gs))
            bad = bool(s.get("saturated")) or (gs is not None and float(gs) < 0.97)
            sat += int(bad)
            n_bad = n_bad + 1 if bad else 0
            tr = self.cell.truth()
            drift = float(np.linalg.norm(tr.pos[:2] - xy0))
            if n_bad >= 3 or drift > 0.04:
                aborted = "depth group saturated" if n_bad >= 3 else "drifting"
                break
        tr = self.cell.truth()
        s = self.cell.state()
        rise = float(tr.pos[2]) - z0
        pitch = math.degrees(MV.rpy(s["quat_wxyz"])[1]) if s.get("quat_wxyz") is not None else 0.0
        ok = self.lift_decision(rise, scale_min, pitch, aborted)
        if not ok:
            # set it back at once: reference down to the start height
            mv.z_rate = 0.35
            mv.z_goal = mv.z_ref = z0
            mv.wait(2.0)
        mv.z_rate = 0.35
        return ok, dict(rise_m=round(rise, 4), depth_scale_min=round(scale_min, 3), saturated_samples=sat,
                        aborted=aborted, drift_m=round(drift, 4), pitch_deg=round(pitch, 2),
                        battery_V=s.get("battery_V"), t_s=round(self.cell.clock() - t0, 1), z0=z0)

    def relook_stand(self, st):
        """Fresh rays of both eyes (the trial lift may have shifted the stand by a few cm): the two close looks of
        survey_low again, the old rays dropped; keeps the old estimate if the new looks do not see both eyes."""
        sc = self.sc
        old = {k: list(v) for k, v in self.eyes.rays.items()}
        near = {k: self.eyes.estimate(k)[:2] for k in ("lift", "drag")}
        self.eyes.rays = dict(lift=[], drag=[])
        X0 = float(sc.w_to_P(np.r_[near["lift"], sc.P_o[2]])[0])
        h_c2, Z2, t2 = LOOK_CAM
        box = self.stab_paddle_box()
        for dx in LOOK_DX:
            pos, yaw = self.cam_body(float(np.clip(X0 + dx, -2.4, 2.4)), h_c2, Z2, t2)
            self.mv.go(pos, yaw, v=0.20, tol=0.04, tol_z=0.03, settle_s=0.4, route=False)
            rgb, cam, info = self.snap(t2, 1.0)
            ex = None if box is None else project_box_mask(cam, rgb.shape[:2], *box)
            self.eyes.read(rgb, cam, t=info["t"], exclude=ex, near=near, near_tol=0.25)
        st2 = self.eyes.stand()
        if st2 is None or not st2.get("ok"):
            self.eyes.rays = old
            self.out["debris"]["relook"] = dict(ok=False)
            return st
        self.out["debris"]["relook"] = dict(ok=True, info=st2["info"],
                                            shift_lift_mm=round(1000 * float(np.linalg.norm(st2["lift"][:2] - near["lift"])), 1),
                                            shift_drag_mm=round(1000 * float(np.linalg.norm(st2["drag"][:2] - near["drag"])), 1))
        return st2

    def laydown_target(self, centre_P):
        return np.array([float(np.clip(centre_P[0], -LAY_X_CLIP, LAY_X_CLIP)), LAY_TARGET_Z])

    def drag_curve(self, goal_P, v=0.10, turn_deg_s=4.0, tol=0.03, timeout=120.0):
        """P02's v3: pull the held eye towards goal_P (panel X, Z; height kept) along a path whose direction turns at
        most turn_deg_s, with the gripper axis on the pull direction (the stand trails straight behind the eye)."""
        sc, mv = self.sc, self.mv
        mv.start()
        tcp0 = sc.w_to_P(self.tcp_now())
        p = np.array([tcp0[0], tcp0[2]])
        h = float(tcp0[1])
        yaw0 = mv.yaw_ref
        a0 = math.atan2(self.a_B[1], self.a_B[0])
        a_w = np.array([math.cos(yaw0 + a0), math.sin(yaw0 + a0), 0.0])
        aP = sc.P_R.T @ a_w
        u = -np.array([aP[0], aP[2]])
        u /= np.linalg.norm(u)
        dt = 0.1
        t0 = self.cell.clock()
        while self.cell.clock() - t0 < timeout:
            d = np.asarray(goal_P, float) - p
            dist = float(np.linalg.norm(d))
            if dist < tol:
                break
            ud = d / dist
            ang = math.atan2(u[0] * ud[1] - u[1] * ud[0], u @ ud)
            mx = math.radians(turn_deg_s) * dt
            ang = max(-mx, min(mx, ang))
            c, s_ = math.cos(ang), math.sin(ang)
            u = np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]])
            p = p + u * min(v * dt, dist)
            u_w = sc.P_R @ np.array([u[0], 0.0, u[1]])
            yaw = self.yaw_for_axis(-u_w)
            o = self.body_for_tcp(sc.P_to_w([p[0], h, p[1]]), yaw)
            mv.goal, mv.ref = o[:2].copy(), o[:2].copy()
            mv.z_goal = mv.z_ref = float(o[2])
            mv.yaw_goal = mv.yaw_ref = yaw
            self.cell.step(dt)
        mv.wait(1.0)
        return u

    def move_debris(self):
        """Trial lift on the lift eye; carry (liftable) or drag (heavy) into the laydown area."""
        self.mark("debris")
        sc = self.sc
        rec = dict()
        self.out["debris"] = rec
        st = self.stand_est
        if not st or not st.get("ok"):
            # one more pair of looks from the clear front before giving up
            rec["relook"] = True
            if self.eyes.estimate("lift") is not None:
                Xl = float(sc.w_to_P(self.eyes.estimate("lift"))[0])
                for dx in (-0.6, 0.0, 0.6):
                    pos, yaw = self.cam_body(float(np.clip(Xl + dx, -2.4, 2.4)), 0.45, 1.75, -5.0)
                    self.go_front(sc.w_to_P(pos)[0], sc.w_to_P(pos)[1])
                    rgb, cam, info = self.snap(-5.0, 1.0)
                    self.eyes.read(rgb, cam, t=info["t"], near=dict(lift=self.eyes.estimate("lift")[:2]), near_tol=0.6)
                st = self.eyes.stand()
                self.stand_est = st
            if not st or not st.get("ok"):
                rec["skipped"] = "stand not located"
                self.note("debris: the stand was not located; not moved")
                return
        n = st["n"]
        L = self.eyes.estimate("lift")
        cP = sc.w_to_P(L)
        rec["stand_est"] = st["info"]
        # ---- to the lift eye's pre-grasp: out at the clear front, at the lift eye's height
        yaw = self.yaw_for_axis(-n)
        pre = L + n * 0.55
        bpre = self.body_for_tcp(pre, yaw)
        self.go_front(float(sc.w_to_P(bpre)[0]), float(sc.w_to_P(bpre)[1]), yaw=yaw)
        ok, g = self.grab_eye("lift", st)
        rec["grab_lift"] = g
        if not ok:
            rec["skipped"] = "lift eye not grasped"
            self.note("debris: the lift eye could not be grasped")
            self.go_tcp(self.tcp_now() + n * 0.5, yaw, v=0.1, tol=0.05, tol_z=0.05, settle_s=0.2)
            return
        lift_ok, lt = self.trial_lift()
        rec["trial_lift"] = lt
        rec["liftable"] = lift_ok
        tgt = self.laydown_target(cP)
        rec["target_P"] = [round(float(x), 3) for x in tgt]
        if lift_ok:
            self.mark("debris_carry")
            yaw = self.mv.yaw_goal
            p = self.tcp_now() + n * 0.70
            self.path_tcp(p, yaw, v=0.06)
            self.mv.wait(1.0)
            q = sc.w_to_P(self.tcp_now())
            self.path_tcp(sc.P_to_w([q[0], CARRY_TCP_H, q[2]]), yaw, v=0.05)
            self.mv.wait(1.0)
            self.path_tcp(sc.P_to_w([tgt[0], CARRY_TCP_H, tgt[1]]), yaw, v=0.10)
            self.mv.wait(3.0)
            self.path_tcp(sc.P_to_w([tgt[0], EYE_H["lift"] + 0.02, tgt[1]]), yaw, v=0.04)
            self.mv.wait(2.0)
            rec["mode"] = "lifted"
            self.release_back(n, back=0.4, up=0.25)
        else:
            self.mark("debris_drag")
            # (set back by trial_lift) let go, back off; fresh looks; take the drag eye
            self.release_back(n, back=0.35, up=0.05)
            st = self.relook_stand(st)
            self.stand_est = st
            n = st["n"]
            cP = sc.w_to_P(self.eyes.estimate("lift"))
            tgt = self.laydown_target(cP)
            rec["target_P"] = [round(float(x), 3) for x in tgt]
            D = self.eyes.estimate("drag")
            yaw = self.yaw_for_axis(-n)
            pre_d = D + n * 0.55
            hi = self.tcp_now()
            self.go_tcp(np.r_[pre_d[:2], hi[2]], yaw, v=0.12, tol=0.04, tol_z=0.04, settle_s=0.2)
            ok, g = self.grab_eye("drag", st)
            rec["grab_drag"] = g
            if not ok:
                rec["skipped"] = "drag eye not grasped"
                self.note("debris: the drag eye could not be grasped")
                self.go_tcp(self.tcp_now() + n * 0.5 + np.array([0, 0, 0.3]), yaw, v=0.1, tol=0.05, tol_z=0.05,
                            settle_s=0.2)
                return
            self.path_tcp(self.tcp_now() + n * 0.60, yaw, v=0.08)
            self.mv.wait(1.0)
            com = cP[[0, 2]] + np.array([float((sc.P_R.T @ n)[0]), float((sc.P_R.T @ n)[2])]) * 0.60
            d = tgt - com
            u = d / max(np.linalg.norm(d), 1e-9)
            goal = tgt + u * EYE_SEP
            u_end = self.drag_curve(goal, v=0.10, turn_deg_s=4.0)
            self.mv.wait(2.0)
            rec["mode"] = "dragged"
            n = sc.P_R @ np.array([u_end[0], 0.0, u_end[1]])
            self.release_back(n, back=0.4, up=0.25)
        # the stand now lies in the laydown area: keep the routes off it
        self.extra_keepout.append((float(tgt[0]) - 1.0, float(tgt[0]) + 1.0, float(tgt[1]) - 1.0, float(tgt[1]) + 1.0,
                                   STAND_HALF[2] + 0.25))
        rec["t_done"] = round(self.cell.clock(), 1)
        self.mark("debris_done")

    # ------------------------------------------------------------------ 10. the hot stab
    @staticmethod
    def relieved_decision(d_rem_m):
        """The trial push's verdict: the stab went home (its paddle reached the seated position learnt in PARK-A)
        -> the section is relieved.  Stopped short (the seal stroke) -> still under pressure."""
        return bool(d_rem_m is not None and d_rem_m <= 0.004)

    def stand_blocks(self, ks):
        """Which of the port positions ks the (unmoved) stand's footprint (own estimate + the register's shape)
        overlaps in the approach corridor (|X - X_port| < 0.5 m, closer than 1.0 m to the face)."""
        st = self.stand_est
        sc = self.sc
        c = sc.w_to_P(st["lift"])
        nP = sc.P_R.T @ st["n"]
        n2 = np.array([nP[0], nP[2]]) / max(np.linalg.norm([nP[0], nP[2]]), 1e-9)
        e2 = np.array([-n2[1], n2[0]])
        pts = [np.array([c[0], c[2]]) + u * STAND_HALF[0] * e2 + w * (STAND_HALF[1] + 0.08) * n2
               for u in np.linspace(-1, 1, 11) for w in np.linspace(-1, 1, 5)]
        out = []
        for k in ks:
            if any(abs(p[0] - PORTS_X[k]) < 0.5 and p[1] < 1.0 for p in pts):
                out.append(int(k))
        return out

    def port_geom(self, k):
        """Valve-like geometry of a stab paddle seated in the port at as-built position k."""
        pc = self.sc.P_to_w([PORTS_X[k], PORT_Y, STAB_PADDLE_Z])
        R = self.sc.P_R
        Rv = np.stack([-R[:, 2], -R[:, 0], R[:, 1]], 1)
        return dict(stem_w=self.sc.P_to_w([PORTS_X[k], PORT_Y, 0.0]), paddle_centre_w=pc, R_v=Rv, col=k, row=None)

    def insert_stab(self):
        self.mark("stab")
        sc = self.sc
        rec = dict()
        self.out["stab"] = rec
        lay = self.port_layout
        tp_tag = f"TP{self.branch}"
        if "PARK-A" not in lay or tp_tag not in lay:
            rec["skipped"] = f"ports not located ({sorted(lay)})"
            self.note("stab: PARK-A or the leak test port was not located; not moved")
            return
        kA, kT = int(lay["PARK-A"]), int(lay[tp_tag])
        rec.update(park=kA, test_port=kT, tp_tag=tp_tag)
        deb = self.out.get("debris") or {}
        if deb.get("mode") not in ("lifted", "dragged") and self.stand_est and self.stand_est.get("ok"):
            blk = self.stand_blocks([kA, kT])
            rec["stand_still_blocks"] = blk
            if blk:
                rec["skipped"] = "the stand still blocks " + ", ".join(PORT_NAMES[k] if k in (0, 7) else f"port {k}" for k in blk)
                self.note("stab: " + rec["skipped"])
                return
        vt = PaddleGrip(self.cell, self.mv, self.md, self.port_geom(kA))
        tp, ty = vt.grip_pose(vt.bite)
        pre_body = tp - vt.x_in * 0.9
        qb = sc.w_to_P(pre_body)
        self.go_front(float(qb[0]), float(qb[1]), yaw=ty)
        try:
            vt.prepare()
            ok = False
            for k in range(3):
                ok = vt.approach_and_close(vt.tp, vt.ty)
                if ok:
                    break
                vt._call("gripper_open")
                self.cell.step(self.md["stroke_s"] + 0.1)
                bp, _ = vt.grip_pose(-0.10)
                vt.approach_only(bp, vt.ty, T_hold=2.0)
            rec["grasp"] = dict(ok=bool(ok), attempts=k + 1, tcp_err_at_close_mm=vt.out.get("tcp_err_at_close_mm"))
            if not ok:
                rec["skipped"] = "stab paddle not grasped"
                self.mv.hold_here()
                self.go_front(float(qb[0]), float(qb[1]), yaw=ty)
                return
            vt.d_hold = float(self.cell.state()["depth_m"])
            self.cell.step(0.6)                                     # the seating glide (0.5 s)
            tr = self.cell.truth()
            T0, P0 = tr.tcp.copy(), tr.pos.copy()
            a = vt.x_in.copy()
            a2 = a[:2] / np.linalg.norm(a[:2])
            # ---- unlatch and draw out: the reference backs out along the port axis, the servo pulls
            p0 = P0[:2].copy()

            def ref_out(t, _p0=p0):
                tr_ = self.cell.truth()
                s = min(0.30, 0.04 * t)
                r = _p0 - a2 * s
                lead = r - tr_.pos[:2]
                if float(np.linalg.norm(lead)) > 0.07:
                    r = tr_.pos[:2] + lead / np.linalg.norm(lead) * 0.07
                return r
            out_ok = [False]

            def out_done(t):
                tr_ = self.cell.truth()
                gone = float((T0 - tr_.tcp) @ a)
                if gone >= 0.26:
                    out_ok[0] = True
                return out_ok[0] and t > 2.0
            self.mark("stab_extract")
            vt.servo_loop(ref_out, 20.0, done=out_done, f_cap=62.0)
            g = self.cell.state()["gripper"]
            rec["extract"] = dict(out=bool(out_ok[0]), gripper=g, t=round(self.cell.clock(), 1))
            self.mv.hold_here()
            if g != "stalled_on_object" or not out_ok[0]:
                rec["skipped"] = "stab not drawn out"
                self.note("stab: not drawn out of PARK-A")
                return
            # ---- carry along the row to the test port, on its axis 40 mm before the mouth
            dX = sc.P_R[:, 0] * (PORTS_X[kT] - PORTS_X[kA])
            T_seat = T0 + dX
            off = T0 - P0                                           # TCP - body (level, same heading)
            pre_ins = T_seat - a * 0.16
            mid = self.cell.truth().pos.copy()
            self.mv.go(np.r_[(mid + dX)[:2], mid[2]], ty, v=0.15, tol=0.02, tol_z=0.02, settle_s=0.3, route=False)
            self.mv.go(pre_ins - off, ty, v=0.05, tol=0.005, tol_z=0.005, settle_s=1.0, route=False, timeout=25.0)
            tr = self.cell.truth()
            rec["pre_insert_err_mm"] = [round(1000 * float(x), 1) for x in (tr.tcp - pre_ins)]
            vt.F_w = self.measure_hold_force(3.0)
            vt.d_hold = float(self.cell.state()["depth_m"]) + float(tr.pos[2] - (pre_ins - off)[2])
            self.mark("stab_insert")
            trials = []
            rec["trials"] = trials
            t_start = self.cell.clock()
            locked = False
            while self.cell.clock() - t_start < STAB_GIVE_UP_S:
                F_try = STAB_F_TRY if len(trials) < 3 else STAB_F_HARD
                res = self.push_trial(vt, T_seat, off, a, a2, F_try)
                trials.append(res)
                if res["seated"]:
                    lk = self.lock_check(vt, T_seat, off, a, a2)
                    res["lock_check"] = lk
                    if lk["locked"]:
                        locked = True
                        break
                elif res["d_rem_mm"] is not None and res["d_rem_mm"] > 35.0:
                    # not in the receptacle at all (the tip on the cone or the face): back out and re-align
                    res["verdict"] = "miss"
                    self.mv.hold_here()
                    self.mv.go(pre_ins - off, ty, v=0.04, tol=0.004, tol_z=0.004, settle_s=1.0, route=False,
                               timeout=40.0)
                    continue
                res.setdefault("verdict", "resisted: still under pressure")
                # hold the stab in the guide with a light push and wait
                vt.servo_loop(lambda t, r=self.cell.truth().pos[:2].copy(): r, STAB_WAIT_S, axial=lambda t, f: 5.0)
            rec["locked"] = locked
            rec["t_locked"] = round(self.cell.clock(), 1) if locked else None
            rec["since_isolation_s"] = None if self.t_isolated is None else round(self.cell.clock() - self.t_isolated, 1)
            # ---- let go and back out along the axis
            r_hold = self.cell.truth().pos[:2].copy()
            vt._call("gripper_open")
            vt.servo_loop(lambda t: r_hold, 2.0, axial=lambda t, f: 0.0)
            p1 = self.cell.truth().pos[:2].copy()
            vt.servo_loop(lambda t: p1 - a2 * min(0.40, 0.08 * t), 6.0, f_cap=40.0)
            rec["gripper_after"] = self.cell.state()["gripper"]
        except MV.ProtectiveStop:
            self.note("stab: protective stop")
            rec["protect_abort"] = True
            MV.wait_protect_clear(self.cell, self.mv)
        self.mv.hold_here()
        tr = self.cell.truth()
        q = sc.w_to_P(tr.pos)
        self.mv.go(sc.P_to_w([q[0], q[1], SC.FRONT_CLEAR]), sc.face_yaw, v=0.15, tol=0.08, tol_z=0.06, settle_s=0.2,
                   route=False)
        self.mark("stab_done")

    def measure_hold_force(self, T):
        """The servo's mean holding force over T s (the current's drag on the vehicle + the held stab), world axes,
        from the depth + translation groups' readback (as ValveTurn.prepare)."""
        acc, n = np.zeros(3), 0
        for _ in range(int(round(T / 0.05))):
            self.mv.wait(0.05)
            s = self.cell.state()
            if s.get("quat_wxyz") is None:
                continue
            wg = s["wrench_groups"]
            acc += MV.quat_to_mat(s["quat_wxyz"]) @ (np.asarray(wg["depth"][:3], float) + np.asarray(wg["translation"][:3], float))
            n += 1
        return acc / max(n, 1)

    def push_trial(self, vt, T_seat, off, a, a2, F_try):
        """One trial push: the reference runs 10 mm past the seat at 0.02 m/s; the axial force is capped at F_try.
        Ends seated (paddle within 4 mm of the seated position for 1.2 s) or stalled (no 1 mm progress in 2.5 s)."""
        self.mv.stop()
        p0 = self.cell.truth().pos[:2].copy()
        body_seat = (T_seat - off)[:2]
        L = float((body_seat - p0) @ a2) + 0.010
        hist = []
        state = dict(seated_s=0.0)

        def ref(t):
            return p0 + a2 * min(L, 0.02 * t) + ((body_seat - p0) - a2 * float((body_seat - p0) @ a2))

        def axial(t, f):
            # in the seal stroke and at the seat: the full trial force (the latch wants >= F_mate for 0.2 s)
            if state.get("last") is not None and state["last"] <= 0.025:
                return float(F_try)
            return float(np.clip(f, -10.0, F_try))

        def done(t):
            tr_ = self.cell.truth()
            d_rem = float((T_seat - tr_.tcp) @ a)
            state["last"] = d_rem
            state["seated_s"] = state["seated_s"] + MV.CTRL_S if d_rem <= 0.004 else 0.0
            if state["seated_s"] >= 1.2:
                return True
            k = int(round(t / MV.CTRL_S))
            if k % 10 == 0:
                hist.append((t, d_rem))                     # every 0.1 s
                if t > L / 0.02 + 1.0 and len(hist) > 25 and abs(hist[-26][1] - d_rem) < 0.001:
                    return True                             # no 1 mm of progress in 2.5 s: stalled
            return t > L / 0.02 + 20.0
        t0 = self.cell.clock()
        vt.servo_loop(ref, L / 0.02 + 25.0, axial=axial, done=done, f_cap=70.0)
        d_rem = state.get("last")
        seated = self.relieved_decision(d_rem) and state["seated_s"] >= 1.2
        out = dict(t=round(t0, 1), F_try=F_try, d_rem_mm=None if d_rem is None else round(1000 * d_rem, 1),
                   seated=bool(seated), dur_s=round(self.cell.clock() - t0, 1))
        if seated:
            out["verdict"] = "seated: relieved"
        return out

    def lock_check(self, vt, T_seat, off, a, a2):
        """Latched?  Release the push for 0.5 s, then pull STAB_PULL_TEST (well under the 35 N release force) for 2 s:
        a latched stab does not move out."""
        r = self.cell.truth().pos[:2].copy()
        tcp0 = self.cell.truth().tcp.copy()
        vt.servo_loop(lambda t: r, 0.5, axial=lambda t, f: 0.0)
        vt.servo_loop(lambda t: r, 2.0, axial=lambda t, f: -STAB_PULL_TEST)
        moved = float((tcp0 - self.cell.truth().tcp) @ a)
        locked = moved < 0.005
        if not locked:
            vt.servo_loop(lambda t: r, 1.0, axial=lambda t, f: 0.0)
        return dict(moved_out_mm=round(1000 * moved, 1), locked=bool(locked))

    # ------------------------------------------------------------------ the end: battery, then P15's return
    def return_to_cage(self):
        super().return_to_cage()
        s = self.cell.state()
        self.out["battery_end"] = dict(soc=s.get("battery_soc"), V=s.get("battery_V"))


# ======================================================================================================= the cell entry
def run(server):
    """Entry point (same contract as reference.run): the eleven-item program inside the cell's reference_run."""
    _paths()
    prog = ContactProgram(server)
    out = prog.run()
    summary = dict(branch=out.get("branch_chosen"), t_end=out.get("t_end"), errors=len(out.get("errors", [])),
                   valves=[(v.get("tag"), v.get("phi_end_deg")) for v in out.get("valves", [])],
                   debris=(out.get("debris") or {}).get("mode"), stab_locked=(out.get("stab") or {}).get("locked"),
                   frames=out.get("frames"), pings=out.get("pings"))
    return {"summary": summary, "program": out}


# ====================================================================================================== maintainer
# Not used by the cell.  PY = the maintainer tools interpreter (-I -B), with the software-rendering environment loaded
#   $PY steps_contact.py one SITE.tar.xz OUT.json [--seed N] [--log-dir D]     one privileged run (cell + P10 scoring)
#   $PY steps_contact.py batch OUT_DIR --cells ABCDEF --k 0 --repeat 2 --procs 8
#   $PY steps_contact.py collect OUT_DIR BUILD_JSON                             -> build/reference_dev_runs.json
#   $PY steps_contact.py audit OUT.json                                         static audit (exit 0 = pass)
# The runs go through the cell's own reference_run: the server loads <entry>/reference.py, a two-line entry that
# calls run() above (P15's reference.py still instantiates CoreProgram; record P2_P16 section 5).
ALL_ITEMS = ("M1", "M2", "H", "I1", "I2", "I3", "D1", "D2", "S1", "S2", "R")
ENTRY_SRC = '''"""Generated by steps_contact.py (maintainer runs of package P16): the cell's reference entry -> ContactProgram."""
import sys
_D = {here!r}
if _D not in sys.path:
    sys.path.insert(0, _D)
import steps_contact as _SX


def run(server):
    return _SX.run(server)
'''
BUDGET = dict(sim_s=2400.0, frame_units=600.0, pings=16000.0, w320=0.40)
ACCEPT = dict(sim_frac=0.70, frames_frac=0.65, pings_frac=0.65, soc_end=0.40, window_go=0.30, window_err=0.05,
              defl_deg=2.0)


def _maint_paths():
    DEV = os.path.normpath(os.path.join(HERE, ".."))
    SH = os.path.normpath(os.path.join(DEV, "..", "..", "shared"))
    for p in (os.path.join(DEV, "cell"), SH, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("UWKIT_SPEC_VALUES", os.path.join(SH, "spec_values.json"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return DEV, SH


def write_entry(d):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "reference.py")
    src = ENTRY_SRC.format(here=HERE)
    if not os.path.isfile(p) or open(p, encoding="utf-8").read() != src:
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
    return d


def _window_truth(sd, prog):
    """The window decision against the truth: the tide mean at the top valve row's height (3.0 m above the bed, the
    generator's definition) at the decision time, and per drift the true current (tide + gusts) over the drift's
    averaging span, at the same height (the field is horizontally uniform outside the structure)."""
    from subseacell import flow as FL
    fl = FL.StationFlow.from_site(sd)
    cur = prog.get("current") or []
    win = prog.get("window") or {}
    t_dec = (prog.get("steps") or {}).get("window_open", {}).get("t")
    P = np.array([[float(sd["pose_o"][0]) + 60.0, float(sd["pose_o"][1]), -float(sd["seabed_depth_m"]) + 3.0]])
    rows = []
    for m in cur:
        t1 = float(m["t"])
        ts = np.arange(t1 - DRIFT_SPAN_S + SC.DRIFT_AVG_AFTER_S, t1 + 1e-9, 0.1)
        sp = [float(np.linalg.norm(fl.field.velocity(P, float(t))[0][:2])) for t in ts]
        rows.append(dict(t=t1, measured=m.get("speed_m_s"), true_gust_mean=round(float(np.mean(sp)), 4),
                         tide_mean=round(fl.mean_speed_at_height(3.0, t1), 4)))
    out = dict(rows=rows, decision_t=t_dec, estimate=win.get("estimate"), rule=win.get("rule"))
    if t_dec is not None and win.get("estimate") is not None:
        tm = fl.mean_speed_at_height(3.0, float(t_dec))
        out["tide_mean_at_decision"] = round(tm, 4)
        out["error_vs_tide_mean"] = round(float(win["estimate"]) - tm, 4)
        used = [r for r in rows if float(t_dec) - r["t"] <= WINDOW_AVG_S + 1e-6 and r["t"] <= float(t_dec) + 1e-6]
        if used and win.get("rule") == "mean":
            g = float(np.mean([r["true_gust_mean"] for r in used]))
            out["true_mean_same_drifts"] = round(g, 4)
            out["error_vs_true_same_drifts"] = round(float(win["estimate"]) - g, 4)
        elif win.get("rule") == "weak" and rows:
            out["true_mean_same_drifts"] = rows[0]["true_gust_mean"]
            out["error_vs_true_same_drifts"] = round(float(win["estimate"]) - rows[0]["true_gust_mean"], 4)
    # the generator's own window opening (tide mean <= 0.30 at 3.0 m)
    t_open = None
    for t in np.arange(0.0, 2400.0, 1.0):
        if fl.mean_speed_at_height(3.0, float(t)) <= 0.30:
            t_open = float(t)
            break
    out["generator_window_open_s"] = t_open
    return out


DRIFT_SPAN_S = SC.DRIFT_S


def run_site(site_path, out_json=None, seed=None, log_dir=None, entry_dir=None):
    """One privileged run of ContactProgram through the cell's reference_run, scored with P10's scoring, plus the
    P16 acceptance quantities."""
    import hashlib
    import json
    DEV, SH = _maint_paths()
    from subseacell import server as SV, scoring as SCO, events as EVL, budget as BU, site as SITE
    import reference as REF
    w0, c0 = time.perf_counter(), time.process_time()
    entry_dir = entry_dir or os.path.join(os.path.dirname(os.path.abspath(out_json or ".")), "_entry")
    write_entry(entry_dir)
    values = BU.load_task_values(BU.default_values_path())
    S = SV.SubseaCellServer(values=values, privileged=True, render=True, reference_dir=entry_dir, log=print)
    S.reset(site_path, seed=seed)
    reply = S.handle("reference_run", {})
    prog = (S.reference_result or {}).get("program") or {}
    term = SCO.capture_terminal(S.world)
    soc_world = float(S.world.sim.battery.status()["soc"])
    sd = S.world.sd
    rec = S.finalize()
    log = S.event_log().export()
    rules = SCO.Rules.from_values(values)
    res = SCO.score_station(site_path, log, term, rules)
    st = SCO.site_truth(site_path, rules)
    names = log["meta"]["valve_names"]
    a0 = log["meta"]["angles0"]
    iso = list(st["isolation_set"])
    tcol = EVL.column(log, "t")
    defl = {n: float(np.max(np.abs(EVL.column(log, f"valve_{n}_deg") - float(a0[n])))) for n in names}
    others = {n: round(v, 4) for n, v in defl.items() if n not in iso}
    steps = {k: v["t"] for k, v in (prog.get("steps") or {}).items()}
    # valve motion during the debris step (every valve, the closed isolation set included)
    deb_defl = None
    if "debris" in steps:
        t_a = float(steps["debris"])
        t_b = float(steps.get("debris_done", steps.get("stab", tcol[-1])))
        sel = (tcol >= t_a) & (tcol <= t_b)
        if np.any(sel):
            k0 = int(np.argmax(sel))
            deb_defl = {n: round(float(np.max(np.abs(EVL.column(log, f"valve_{n}_deg")[sel]
                                                     - EVL.column(log, f"valve_{n}_deg")[k0]))), 4) for n in names}
    lp = EVL.column(log, "leak_pressure_bar")
    stab = prog.get("stab") or {}
    for tr_ in stab.get("trials") or []:
        k = int(np.clip(np.searchsorted(tcol, float(tr_["t"]) + float(tr_["dur_s"])), 0, len(tcol) - 1))
        tr_["leak_pressure_bar_at_end_truth"] = round(float(lp[k]), 3)
        tr_["mate_force_N_truth"] = round(15.0 + 2.5 * float(lp[k]), 2)
    deb = prog.get("debris") or {}
    kind = sd["debris"]["kind"]
    lay_xz = res["details"]["debris"]["laydown_xz_m"]
    tgt = deb.get("target_P")
    # contact work vs the window (grasp events of the vehicle: valve paddles, stab paddle, eyes)
    t_win = steps.get("window_open")
    grasp_ev = [e for e in log.get("events") or () if e.get("src") == "grasp" and e.get("ev") == "grasp"
                and e.get("kind") in ("paddle", "eye") and e.get("handle") != "marker"]
    first_contact = min([float(e["t"]) for e in grasp_ev] or [None]) if grasp_ev else None
    frames = rec.get("frames") or {}
    units = float(frames.get("640x480", 0)) + BUDGET["w320"] * float(frames.get("320x240", 0))
    allowed = set(SV.PUBLIC_METHODS) | {"get_state_privileged", "reference_run"}
    calls = dict(sorted(S.calls.items()))
    items = res["items"]
    out = dict(
        site=os.path.basename(str(site_path)), site_label=rec.get("site_label"), site_sha256=rec.get("site_sha256"),
        cell=str(sd.get("cell")), leak_branch=st["leak_branch"], isolation_set=iso, branch_chosen=prog.get("branch_chosen"),
        items={k: bool(items[k]) for k in ALL_ITEMS}, score=res["score"],
        points=int(sum(rules.weights_c[k] for k in ALL_ITEMS if items[k])),
        critical_errors=res["details"]["critical_errors"], misoperated=res["details"]["misoperated"],
        n_imaged=res["details"]["n_imaged"], protective_stops=res["details"]["protective_stops"],
        disabled=res["details"]["disabled"],
        protect_events=[{k: e[k] for k in sorted(e) if k != "src"} for e in log.get("events") or ()
                        if e.get("src") == "protect"][:20],
        max_deflection_outside_iso_deg=round(max(others.values()) if others else 0.0, 4),
        deflection_during_debris_deg=deb_defl,
        max_deflection_during_debris_deg=None if deb_defl is None else max(deb_defl.values()),
        final_angles_iso={k: res["details"]["final_angles"][k] for k in iso},
        marker=res["details"]["marker"], vehicle=res["details"]["vehicle"], debris_score=res["details"]["debris"],
        stab_score=res["details"]["stab"], debris_kind=kind, debris_mode=deb.get("mode"),
        debris_expected_mode="lifted" if kind in ("D1", "D2") else "dragged", debris_target_P=tgt,
        debris_final_P=lay_xz, sim_time_s=rec.get("sim_time_s"), frames=frames, frame_units=round(units, 2),
        pings=rec.get("pings"), soc_end=round(soc_world, 4), battery_end=prog.get("battery_end"),
        window=_window_truth(sd, prog), first_contact_t=first_contact, window_open_t=t_win, steps=steps,
        calls=calls, calls_ok=set(calls) <= allowed and "_unknown" not in calls, illegal_calls=rec.get("illegal_call_count"),
        internal_errors=rec.get("internal_errors"), done_reason=rec.get("done_reason"), cell_fault=rec.get("cell_fault"),
        events_sha256=rec.get("events_sha256"), fingerprint=rec.get("fingerprint"),
        program_sha256=REF._sha(REF._canon(REF._strip_wall(prog))), reply=reply, program=prog, details=res["details"],
        wall_s=round(time.perf_counter() - w0, 1), cpu_s=round(time.process_time() - c0, 1),
        cell_work_s=rec.get("cell_work_s"), entry_sha256=hashlib.sha256(open(os.path.join(entry_dir, "reference.py"), "rb").read()).hexdigest())
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        S.event_log().save(log_dir)
    S.close()
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, sort_keys=True, default=str)
    return out


def cmd_one(argv):
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--entry-dir", default=None)
    a = ap.parse_args(argv)
    r = run_site(a.site, a.out, seed=a.seed, log_dir=a.log_dir, entry_dir=a.entry_dir)
    print(json.dumps({k: r[k] for k in ("site", "leak_branch", "branch_chosen", "items", "score", "critical_errors",
                                        "protective_stops", "debris_mode", "sim_time_s", "soc_end", "wall_s")}, default=str))
    return 0


def cmd_batch(argv):
    import argparse
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--cells", default="ABCDEF")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--sites-dir", default=os.path.normpath(os.path.join(HERE, "..", "build", "dev_sites")))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    entry = write_entry(os.path.join(a.out_dir, "_entry"))
    jobs = []
    for rep in range(a.repeat):
        for c in a.cells:
            out = os.path.join(a.out_dir, f"run_{c}_{a.k}_r{rep}.json")
            if not os.path.isfile(out):
                jobs.append((c, rep, out))
    running = []
    while jobs or running:
        while jobs and len(running) < a.procs:
            c, rep, out = jobs.pop(0)
            lg = open(out.replace(".json", ".log"), "w")
            p = subprocess.Popen([sys.executable, "-I", "-B", os.path.abspath(__file__), "one",
                                  os.path.join(a.sites_dir, f"site_{c}_{a.k}.tar.xz"), out, "--entry-dir", entry],
                                 stdout=lg, stderr=subprocess.STDOUT)
            running.append((p, lg, c, rep, time.time()))
            print(f"start {c} r{rep} pid {p.pid}", flush=True)
        time.sleep(5)
        for x in list(running):
            if x[0].poll() is not None:
                x[1].close()
                running.remove(x)
                print(f"done {x[2]} r{x[3]} rc {x[0].returncode} {time.time() - x[4]:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- static audit
# the decisions (window open, liftable, relieved, the port layout) take the program's own measurements only; the eye
# estimator (EyeBook) is a measurement: rays of its own frames + the panel frame's mudmat height (a structure pose)
DECISION_FUNCS = ("ContactProgram.window_decision", "ContactProgram.lift_decision", "ContactProgram.relieved_decision",
                  "PortBook.layout")
ALLOWED_IMPORTS = {"__future__", "math", "os", "sys", "time", "numpy", "scipy", "perception", "moves", "steps_core",
                   "steps_contact", "subseacell", "reference", "argparse", "json", "subprocess", "hashlib", "glob"}
IN_CELL_IMPORTS = {"__future__", "math", "os", "sys", "time", "numpy", "scipy", "perception", "moves", "steps_core"}
MAINT_FUNCS = ("_maint_paths", "write_entry", "_window_truth", "run_site", "cmd_one", "cmd_batch", "static_audit",
               "cmd_audit", "cmd_collect", "main", "_pct")


def static_audit() -> dict:
    """AST audit of steps_contact.py's in-cell part (+ P15's audit of moves / steps_core / reference.run): imports
    (in-cell: stdlib, numpy, scipy, perception, moves, steps_core), no access to the server or the world (the server
    object is only handed to CoreProgram, i.e. moves.Cell), no _handle / handle call, no introspection or dynamic code,
    and the decision functions reference no privileged name."""
    import ast
    _maint_paths()
    import reference as REF
    rep = dict(problems=[])
    p15 = REF.static_audit()
    rep["p15_audit_ok"] = p15["ok"]
    rep["p15_problems"] = p15["problems"]
    if not p15["ok"]:
        rep["problems"].append("P15's audit of moves / steps_core / reference failed")
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    defs = dict(REF.ast_walk_defs(tree))
    maint = set(MAINT_FUNCS)
    in_cell_nodes = [n for n in tree.body if not (isinstance(n, (ast.FunctionDef,)) and n.name in maint)
                     and not (isinstance(n, ast.If))]
    imps, server_attrs, intro, dyn, handle_calls = set(), set(), [], [], []
    for root in in_cell_nodes:
        for n in ast.walk(root):
            if isinstance(n, ast.Import):
                imps |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                imps.add(n.module.split(".")[0])
            elif isinstance(n, ast.Attribute):
                if isinstance(n.value, ast.Name) and n.value.id == "server":
                    server_attrs.add(n.attr)
                if n.attr in REF.INTROSPECTION:
                    intro.append((n.attr, n.lineno))
                if n.attr in ("_handle", "handle", "world", "hooks"):
                    handle_calls.append((n.attr, n.lineno))
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in (
                    "getattr", "setattr", "vars", "globals", "locals", "eval", "exec", "__import__", "compile", "open"):
                dyn.append((n.func.id, n.lineno))
    rep["in_cell_imports"] = sorted(imps)
    if imps - IN_CELL_IMPORTS:
        rep["problems"].append(f"in-cell imports outside the allowed set: {sorted(imps - IN_CELL_IMPORTS)}")
    if imps & REF.FORBIDDEN_IMPORTS:
        rep["problems"].append(f"forbidden imports {sorted(imps & REF.FORBIDDEN_IMPORTS)}")
    rep["server_attrs"] = sorted(server_attrs)
    if server_attrs:
        rep["problems"].append(f"the server's attributes are read: {sorted(server_attrs)}")
    rep["handle_world_hooks_refs"] = handle_calls
    if handle_calls:
        rep["problems"].append(f"handle / world / hooks referenced outside moves.Cell: {handle_calls}")
    if intro:
        rep["problems"].append(f"introspection {intro}")
    if dyn:
        rep["problems"].append(f"dynamic attribute / code access {dyn}")
    dec = {}
    for fn in DECISION_FUNCS:
        node = defs[fn]
        nm = {m.id for m in ast.walk(node) if isinstance(m, ast.Name)} | \
             {m.attr for m in ast.walk(node) if isinstance(m, ast.Attribute)}
        hit = sorted(nm & (REF.PRIV_NAMES | {"cell", "tcp_now", "tr", "tr_"}))
        dec[fn] = dict(privileged_refs=hit, args=[a.arg for a in node.args.args])
        if hit:
            rep["problems"].append(f"decision function {fn} references {hit}")
    rep["decision_functions"] = dec
    # where the privileged block is used in the new steps (the vehicle pose only: Cell.truth / moves)
    priv_use = {}
    for fn, node in sorted(defs.items()):
        if fn in maint:
            continue
        nm = {m.attr for m in ast.walk(node) if isinstance(m, ast.Attribute)}
        priv_use[fn] = sorted(nm & {"truth", "tcp", "pos", "vel_w", "yaw", "scene", "sc", "P_to_w", "w_to_P"})
    rep["privileged_block_uses"] = priv_use
    rep["ok"] = not rep["problems"]
    return rep


def cmd_audit(argv):
    import json
    rep = static_audit()
    if argv:
        with open(argv[0], "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, sort_keys=True, default=str)
    print(json.dumps(dict(ok=rep["ok"], problems=rep["problems"]), indent=1))
    return 0 if rep["ok"] else 1


# ---------------------------------------------------------------------------------------------------- collect
def _pct(x, q=95.0):
    x = np.asarray([v for v in x if v is not None], float)
    return None if x.size == 0 else float(np.percentile(x, q))


def cmd_collect(argv):
    import glob
    import hashlib
    import json
    out_dir, build = argv[0], argv[1]
    runs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "run_*_r*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        b = os.path.basename(p)[:-5].split("_")
        runs.setdefault(b[1], {})[b[3]] = r
    audit = static_audit()
    A = ACCEPT
    rows, ok_all = [], True
    errs = dict(marker=[], debris_x=[], debris_z=[], valve=[])
    for cell in sorted(runs):
        R = runs[cell]
        r0 = R.get("r0")
        if r0 is None:
            continue
        r1 = R.get("r1")
        same = None if r1 is None else dict(
            events_sha256=r0["events_sha256"] == r1["events_sha256"],
            qpos_chain=r0["fingerprint"]["qpos_chain_sha256"] == r1["fingerprint"]["qpos_chain_sha256"],
            valve_hist=r0["fingerprint"]["valve_hist_sha256"] == r1["fingerprint"]["valve_hist_sha256"],
            program=r0["program_sha256"] == r1["program_sha256"])
        w = r0["window"]
        strong = cell in ("E", "F")
        stab = r0["program"].get("stab") or {}
        trials = stab.get("trials") or []
        chk = dict(
            score_100=r0["points"] == 100, all_11=all(r0["items"].values()), no_ce=len(r0["critical_errors"]) == 0,
            stops_0=r0["protective_stops"] == 0,
            sim_time=float(r0["sim_time_s"]) <= A["sim_frac"] * BUDGET["sim_s"] + 1e-6,
            frames=r0["frame_units"] <= A["frames_frac"] * BUDGET["frame_units"],
            pings=float(r0["pings"]) <= A["pings_frac"] * BUDGET["pings"],
            soc=r0["soc_end"] >= A["soc_end"],
            debris_mode=r0["debris_mode"] == r0["debris_expected_mode"],
            debris_valves=r0["max_deflection_during_debris_deg"] is not None
            and r0["max_deflection_during_debris_deg"] < A["defl_deg"],
            outside_iso=r0["max_deflection_outside_iso_deg"] < A["defl_deg"],
            stab_by_trial_push=bool(stab.get("locked")) and any(t.get("seated") for t in trials),
            calls_ok=bool(r0["calls_ok"]) and r0["illegal_calls"] == 0,
            repeat_identical=None if same is None else all(same.values()))
        if strong:
            est = w.get("estimate")
            chk["window_estimate_le_030"] = est is not None and est <= A["window_go"]
            chk["window_err_tide"] = w.get("error_vs_tide_mean") is not None and abs(w["error_vs_tide_mean"]) <= A["window_err"]
            chk["window_err_true_drifts"] = w.get("error_vs_true_same_drifts") is not None and \
                abs(w["error_vs_true_same_drifts"]) <= A["window_err"]
            chk["contact_after_window"] = r0["first_contact_t"] is not None and r0["window_open_t"] is not None and \
                float(r0["first_contact_t"]) >= float(r0["window_open_t"])
        ok = all(v for v in chk.values() if v is not None) and same is not None
        ok_all = ok_all and ok
        errs["marker"].append(r0["marker"].get("zone_distance_m"))
        if r0.get("debris_final_P") and r0.get("debris_target_P"):
            errs["debris_x"].append(abs(r0["debris_final_P"][0] - r0["debris_target_P"][0]))
            errs["debris_z"].append(abs(r0["debris_final_P"][1] - r0["debris_target_P"][1]))
        for a in r0["final_angles_iso"].values():
            errs["valve"].append(abs(90.0 - float(a)))
        rows.append(dict(cell=cell, site=r0["site_label"], site_sha256=r0["site_sha256"], leak_branch=r0["leak_branch"],
                         branch_chosen=r0["branch_chosen"], items=r0["items"], score=r0["score"], points=r0["points"],
                         critical_errors=r0["critical_errors"], protective_stops=r0["protective_stops"],
                         sim_time_s=r0["sim_time_s"], sim_frac=round(float(r0["sim_time_s"]) / BUDGET["sim_s"], 4),
                         frame_units=r0["frame_units"], frames_frac=round(r0["frame_units"] / BUDGET["frame_units"], 4),
                         pings=r0["pings"], pings_frac=round(float(r0["pings"]) / BUDGET["pings"], 4),
                         soc_end=r0["soc_end"], window=w, first_contact_t=r0["first_contact_t"],
                         window_open_t=r0["window_open_t"], debris_kind=r0["debris_kind"], debris_mode=r0["debris_mode"],
                         debris=r0["program"].get("debris"), debris_score=r0["debris_score"],
                         debris_target_P=r0["debris_target_P"], debris_final_P=r0["debris_final_P"],
                         deflection_during_debris_deg=r0["deflection_during_debris_deg"],
                         max_deflection_during_debris_deg=r0["max_deflection_during_debris_deg"],
                         max_deflection_outside_iso_deg=r0["max_deflection_outside_iso_deg"], stab=stab,
                         stab_score=r0["stab_score"], final_angles_iso=r0["final_angles_iso"], marker=r0["marker"],
                         vehicle=r0["vehicle"], steps=r0["steps"], calls=r0["calls"], survey_low=r0["program"].get("survey_low"),
                         notes=r0["program"].get("notes"), errors=r0["program"].get("errors"), wall_s=r0["wall_s"],
                         cpu_s=r0["cpu_s"], events_sha256=r0["events_sha256"],
                         qpos_chain_sha256=r0["fingerprint"]["qpos_chain_sha256"], repeat=same, checks=chk, ok=ok))
    tol = dict(marker=1.5, debris_x=None, debris_z=None, valve=10.0)
    p95 = {k: _pct(v) for k, v in errs.items()}
    # the laydown margins of the targets: the distance from each target to the area's edge (per axis), smallest
    tx = [min(r["debris_target_P"][0] - LAYDOWN_X[0], LAYDOWN_X[1] - r["debris_target_P"][0]) for r in rows if r["debris_target_P"]]
    tz = [min(r["debris_target_P"][1] - LAYDOWN_Z[0], LAYDOWN_Z[1] - r["debris_target_P"][1]) for r in rows if r["debris_target_P"]]
    tol["debris_x"] = min(tx) if tx else None
    tol["debris_z"] = min(tz) if tz else None
    tol_chk = {k: (p95[k] is not None and tol[k] is not None and 1.5 * p95[k] <= tol[k]) for k in errs}
    files = {}
    for fn in ("steps_contact.py", "steps_core.py", "moves.py", "reference.py"):
        with open(os.path.join(HERE, fn), "rb") as f:
            files[fn] = hashlib.sha256(f.read()).hexdigest()
    res = dict(schema="reference_dev_runs/1", package="P16",
               acceptance_ok=bool(ok_all and audit["ok"] and len(rows) == 6 and all(tol_chk.values())),
               tolerance_check=dict(p95=p95, tolerance=tol, ok=tol_chk, rule="1.5 x p95 <= tolerance"),
               budgets=dict(BUDGET), acceptance_limits=dict(ACCEPT),
               static_audit=audit, reference_sha256=files, stations=rows)
    with open(build, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    print(json.dumps(dict(acceptance_ok=res["acceptance_ok"], audit=audit["ok"], tolerance=tol_chk,
                          stations=[(r["cell"], r["points"], r["ok"], {k: v for k, v in r["checks"].items() if not v})
                                    for r in rows]), indent=1, default=str))
    return 0


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "one":
        return cmd_one(argv[1:])
    if cmd == "batch":
        return cmd_batch(argv[1:])
    if cmd == "audit":
        return cmd_audit(argv[1:])
    if cmd == "collect":
        return cmd_collect(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    _paths()
    sys.exit(main(sys.argv[1:]))
