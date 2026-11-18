"""Motion and sensor layer of the privileged reference program of task 07 (package P15; DESIGN v1.2 sections 1.4,
2.3.1, 3.7; records G1, G6, P2_P11).

THE PRIVILEGED BLOCK.  The reference runs inside the cell (`server.reference_run`).  Everything it does to the run
goes through `Cell`, which touches the server in exactly two ways:
  * `server.handle(name, params)` with name in the 17 public calls or the one privileged call
    `get_state_privileged` (true vehicle pose / tool point, structure, panel and cage frames, mudline: no valve
    angle, pressure, plume, flow or leak - server.get_state_privileged);
  * the world's per-control-period hook list (`server.world.hooks`, append / remove of one wrapper) = the clock of
    the program's own 100 Hz position servo; the wrapper never reads its world argument and acts only through
    `server.handle`.
Set-points come from the true poses (house style of the privileged reference: pump skid, PR #79, quadruped); every
movement is a public motion call; every DECISION of the program (steps_core.py) is taken on its own pings, frames and
state() replies.  reference.py --audit checks this statically (AST) and the runs check it dynamically (call counts).

Contents
  Cell        the adapter above (+ call log, episode end, PNG decode, ping decode, the truth trace)
  Mover       privileged position servo on the TRUE pose through set_wrench (Fx, Fy open in world axes; depth and
              attitude held by the firmware with the depth-sensor / heading offsets re-referenced against the true
              pose before every move - G6 fix 1), lead-limited reference ramps, keep-out routing around the panel
              skid and under the roof (G6 fix 3), slow landing with a force ramp (G6 fix 4 / design 1.4)
  Sensors     frames (camera pose from the true vehicle pose + the manual's mount and the tilt reading) and sonar
              scans (each ping's head pose from the true pose at the ping's time)
  ValveTurn   gate G1's close-a-valve sequence as ported to the product firmware by P11 (qc/contact_matrix.Program):
              drag measured at the pre-grasp pose under the truth servo (no DVL needed: P11 finding 3), approach
              servo on the TRUE pose with the 15 N push while the jaws close, PI turn on the AHRS attitude about the
              gripper axis, release / back-off / level with the self-righting moment cancelled; the public gripper
              state (`open` until the jaws finish closing) instead of the raw one; protective stops handled
numpy + the perception package (numpy / scipy) only: no import of the cell, the simulation kit or the physics engine.
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
import zlib

import numpy as np

from perception import vehicle as VH
from perception import pose as PO

PUBLIC_CALLS = ("mission", "clock", "step", "finish", "state", "set_nav_fix", "idle", "set_attitude", "set_velocity",
                "set_hold", "set_wrench", "camera_frame", "set_camera_tilt", "set_lights", "sonar_scan",
                "gripper_open", "gripper_close")
PRIVILEGED_CALL = "get_state_privileged"
TRUTH_KEYS = ("t", "vehicle", "tcp", "structure", "panel", "cage", "mudline_z", "physics_dt", "steps", "d_time")
CTRL_S = 0.01                       # firmware control period (the servo's tick)
STEP_MAX_S = 10.0
FREE_XY = (True, True, False, False, False, False)
FREE_T = (True, True, True, False, False, False)
FREE_R = (False, False, False, True, True, True)
FREE_ALL = (True,) * 6


class EpisodeOver(RuntimeError):
    """The run ended (finish, simulated-time cap, vehicle disabled, surfaced ...)."""


class CellError(RuntimeError):
    """A call answered an error the program does not expect (bad parameters, budget)."""


# ======================================================================================================= small math
def quat_to_mat(q):
    return VH.quat_to_mat(q)


def yaw_quat(yaw: float) -> list:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def euler_quat(roll, pitch, yaw):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def rpy(q):
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    d = float(np.clip(q0 @ q1, -1.0, 1.0))
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1.0 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return math.remainder(float(a), 2.0 * math.pi)


def png_decode(b: bytes) -> np.ndarray:
    """8-bit RGB PNG -> (H, W, 3) uint8.  The cell writes filter 0 on every row (server.png_rgb); other filters fall
    back to Pillow."""
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    i, w, h, idat = 8, 0, 0, []
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]
        tag = b[i + 4:i + 8]
        data = b[i + 8:i + 8 + n]
        if tag == b"IHDR":
            w, h, bd, ct = struct.unpack(">IIBB", data[:10])
            if bd != 8 or ct != 2:
                raise ValueError("not 8-bit RGB")
        elif tag == b"IDAT":
            idat.append(data)
        i += 12 + n
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, 1 + 3 * w)
    if np.any(raw[:, 0] != 0):
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.uint8).copy()
    return raw[:, 1:].reshape(h, w, 3).copy()


# ======================================================================================================= paths
HERE = os.path.dirname(os.path.abspath(__file__))


def manual_paths() -> dict:
    """The vehicle manual's data files the reference reads (the digest's spec values for the mounts, the gripper
    drawing): $SUBSEACELL_SPEC_VALUES / $UWKIT_SPEC_VALUES and $SUBSEACELL_GRIPPER_OUTLINE override; else the copies
    next to this folder (task layout) or the development layout."""
    cand_spec = [os.environ.get("SUBSEACELL_SPEC_VALUES"), os.environ.get("UWKIT_SPEC_VALUES"),
                 os.path.join(HERE, "data", "spec_values.json"),
                 os.path.normpath(os.path.join(HERE, "..", "..", "..", "shared", "spec_values.json"))]
    cand_out = [os.environ.get("SUBSEACELL_GRIPPER_OUTLINE"), os.path.join(HERE, "data", "gripper_outline.json"),
                os.path.normpath(os.path.join(HERE, "..", "build", "gripper_outline.json"))]
    spec = next((p for p in cand_spec if p and os.path.isfile(p)), None)
    outl = next((p for p in cand_out if p and os.path.isfile(p)), None)
    if spec is None or outl is None:
        raise FileNotFoundError("the reference needs spec_values.json and gripper_outline.json (manual data)")
    return dict(spec_values=spec, gripper_outline=outl)


def load_manual():
    """(man without marker, man with the marker in the jaws, closed-jaw outline man, raw manual dict)."""
    p = manual_paths()
    with open(p["spec_values"], encoding="utf-8") as f:
        S = json.load(f)["values"]
    with open(p["gripper_outline"], encoding="utf-8") as f:
        OUT = json.load(f)
    man = VH.Manual.from_spec_values(p["spec_values"], p["gripper_outline"])
    man_mk = VH.Manual.from_spec_values(p["spec_values"], p["gripper_outline"], with_marker=True)
    g = S["uwkit.mission.articulation"]["value"]["gripper"]
    trim = S["mission.state_sea_SP35_trim"]["value"]
    block = [pp for pp in OUT["marker_held"]["parts"] if pp.get("part") == "block"]
    blk = np.asarray(block[0]["corners_B_m"], float) if block else None
    data = dict(tcp_B=np.asarray(g["tcp_B_m"], float), tip_B=np.asarray(g["jaw_tip_B_m"], float),
                axis_B=np.asarray(g["gripper_axis_B"], float),
                tip_ahead=float((np.asarray(g["jaw_tip_B_m"]) - np.asarray(g["tcp_B_m"])) @ np.asarray(g["gripper_axis_B"])),
                WBG=float(trim["mass_kg"]) * 9.80665 * float(trim["BG_m"]),
                marker_block_B=None if blk is None else blk.mean(0), marker_bottom_B=None if blk is None else float(blk[:, 2].min()),
                seat_s=0.5, stroke_s=float(OUT.get("stroke_s", 1.6)), paths=p)
    return man, man_mk, data


# ======================================================================================================= the cell
class Truth:
    __slots__ = ("t", "pos", "R", "quat", "vel_w", "vel_b", "omega_b", "tcp", "tcp_R", "yaw")

    def __init__(self, d):
        v = d["vehicle"]
        self.t = float(d["t"])
        self.pos = np.asarray(v["pos_w"], float)
        self.quat = np.asarray(v["quat_wxyz"], float)
        self.R = quat_to_mat(self.quat)
        self.vel_b = np.asarray(v["vel_b"], float)
        self.vel_w = self.R @ self.vel_b
        self.omega_b = np.asarray(v["omega_b"], float)
        self.tcp = np.asarray(d["tcp"]["pos_w"], float)
        self.tcp_R = np.asarray(d["tcp"]["R_w"], float)
        self.yaw = math.atan2(self.R[1, 0], self.R[0, 0])


class Scene:
    """The static privileged frames: structure S (origin = mudmat top at the structure centre), panel P (X right,
    Y up, Z out of the face; origin = face plane, middle of the width, mudmat top), cage, mudline."""

    def __init__(self, d):
        self.S_o = np.asarray(d["structure"]["origin_w"], float)
        self.S_R = np.asarray(d["structure"]["R_w"], float)
        self.psi = math.atan2(self.S_R[1, 0], self.S_R[0, 0])
        self.P_o = np.asarray(d["panel"]["origin_w"], float)
        self.P_R = np.asarray(d["panel"]["R_w"], float)
        self.cage_pos = np.asarray(d["cage"]["pos_w"], float)
        self.cage_R = np.asarray(d["cage"]["R_w"], float)
        self.cage_yaw = math.atan2(self.cage_R[1, 0], self.cage_R[0, 0])
        self.deck_top = float(d["cage"]["deck_top_w"])
        self.mudline = float(d["mudline_z"])

    def S_to_w(self, p):
        return self.S_o + self.S_R @ np.asarray(p, float)

    def w_to_S(self, p):
        return self.S_R.T @ (np.asarray(p, float) - self.S_o)

    def P_to_w(self, p):
        return self.P_o + self.P_R @ np.asarray(p, float)

    def w_to_P(self, p):
        return self.P_R.T @ (np.asarray(p, float) - self.P_o)

    @property
    def face_yaw(self):
        """Heading of a vehicle facing the panel (body +x along -Z_P)."""
        z = -self.P_R[:, 2]
        return math.atan2(z[1], z[0])


class Cell:
    """The privileged reference's only handle on the run (see the module docstring)."""

    def __init__(self, server):
        self._handle = server.handle
        self._hooks = server.world.hooks          # the control-period hook list (timing only)
        self.calls = {}
        self.order = []                           # first use of each call, in order
        self.t = 0.0
        self.done = False
        self.end_reason = None
        self.refused = 0
        self._ticks = []
        self.tick_errors = []
        self.scene = Scene(self.priv())

    # ------------------------------------------------------------------ wire
    def call(self, name, **params):
        if name not in PUBLIC_CALLS:
            raise ValueError(f"not a public call: {name}")
        if name not in self.calls:
            self.order.append(name)
        self.calls[name] = self.calls.get(name, 0) + 1
        r = self._handle(name, params)
        if isinstance(r, dict) and "error" in r:
            if r["error"] == "episode_over":
                self.done = True
                raise EpisodeOver(r.get("detail", ""))
            raise CellError(f"{name}: {r}")
        if name == "step":
            self.t = float(r["t"])
            if r.get("done"):
                self.done, self.end_reason = True, r.get("end_reason")
        elif isinstance(r, dict) and r.get("accepted") is False:
            self.refused += 1
            if r.get("reason") == "episode_over":
                self.done = True
                raise EpisodeOver("refused: episode over")
        return r

    def priv(self) -> dict:
        if PRIVILEGED_CALL not in self.calls:
            self.order.append(PRIVILEGED_CALL)
        self.calls[PRIVILEGED_CALL] = self.calls.get(PRIVILEGED_CALL, 0) + 1
        r = self._handle(PRIVILEGED_CALL, {})
        if not isinstance(r, dict) or "error" in r:
            raise CellError(f"{PRIVILEGED_CALL}: {r}")
        return r

    def truth(self) -> Truth:
        return Truth(self.priv())

    def state(self) -> dict:
        return self.call("state")

    def step(self, dt: float):
        left = float(dt)
        while left > 1e-9:
            d = min(left, STEP_MAX_S)
            self.call("step", dt=d)
            if self.done:
                raise EpisodeOver(self.end_reason or "done")
            left -= d

    def clock(self) -> float:
        return self.t

    # ------------------------------------------------------------------ the servo clock
    def add_tick(self, fn):
        """Run fn(t) every control period (inside step / sonar_scan).  An exception inside a tick must not reach the
        cell (it would end the run as a cell fault): it is recorded and the tick removed."""
        def _hook(_world, t, _fn=fn):          # the world argument is never used
            try:
                _fn(float(t))
            except EpisodeOver:
                self.done = True
                self._drop(_fn)
            except Exception as e:  # noqa: BLE001
                self.tick_errors.append(repr(e)[:300])
                self._drop(_fn)
        self._ticks.append((fn, _hook))
        self._hooks.append(_hook)

    def _drop(self, fn):
        for k, (f, h) in enumerate(self._ticks):
            if f == fn:
                try:
                    self._hooks.remove(h)
                except ValueError:
                    pass
                del self._ticks[k]
                return

    def has_tick(self, fn) -> bool:
        return any(f == fn for f, _ in self._ticks)

    def remove_tick(self, fn):
        self._drop(fn)

    def clear_ticks(self):
        for f, h in list(self._ticks):
            self.remove_tick(f)


# ======================================================================================================= the servo
class Mover:
    """Privileged position servo (see the module docstring)."""

    KP, KD, KI = 800.0, 250.0, 400.0
    F_MAX = 90.0
    LEAD_MAX = 0.8                         # the reference point never runs more than this ahead of the vehicle

    def __init__(self, cell: Cell, geo=None, v_max=0.5, yaw_rate_deg=30.0, z_rate=0.35):
        self.cell = cell
        self.geo = geo                     # steps_core.Keepout (routing) or None
        tr = cell.truth()
        self.ref = tr.pos[:2].copy()
        self.goal = self.ref.copy()
        self.z_ref = float(tr.pos[2])
        self.z_goal = self.z_ref
        self.yaw_ref = tr.yaw
        self.yaw_goal = self.yaw_ref
        self.v_max, self.yaw_rate, self.z_rate = v_max, math.radians(yaw_rate_deg), z_rate
        self._v = v_max
        self._zr = z_rate
        self.ei = np.zeros(2)
        self.F_last = np.zeros(3)
        self.depth_bias = 0.0
        self.yaw_bias = 0.0
        self.press = None                  # landing: dict(F0, F1, t0, T) downward world force ramp with Fz open
        self.gain = 1.0
        self.trace = None                  # list of (t, pos, quat) while recording (sonar georeferencing)
        self.stats = dict(moves=0, timeouts=0, reroutes=0, move_s=0.0, refused=0, protect_pauses=0)
        self._last_t = None
        self.last_truth = tr

    # ------------------------------------------------------------------ offsets
    def calibrate(self):
        """Depth-sensor and heading offsets against the true pose (G1 / G6: the stand-in for referencing on the plate
        codes): the firmware holds ITS depth and heading, the servo wants true ones."""
        st = self.cell.state()
        tr = self.cell.truth()
        if st.get("depth_m") is not None:
            self.depth_bias = float(st["depth_m"]) - (-float(tr.pos[2]))
        if st.get("heading_rad") is not None:
            self.yaw_bias = wrap(float(st["heading_rad"]) - tr.yaw)

    # ------------------------------------------------------------------ on / off
    @property
    def active(self) -> bool:
        return self.cell.has_tick(self.tick)

    def start(self):
        if not self.active:
            tr = self.cell.truth()
            self.ref = tr.pos[:2].copy()
            self.z_ref = float(tr.pos[2])
            self.yaw_ref = tr.yaw
            self.goal, self.z_goal, self.yaw_goal = self.ref.copy(), self.z_ref, self.yaw_ref
            self.ei[:] = 0.0
            self.cell.add_tick(self.tick)

    def stop(self):
        if self.active:
            self.cell.remove_tick(self.tick)

    def hold_here(self):
        """Take the current true pose as the goal (servo on)."""
        self.stop()
        self.start()

    # ------------------------------------------------------------------ the 100 Hz tick
    def tick(self, t):
        cell = self.cell
        dt = CTRL_S
        tr = cell.truth()
        self.last_truth = tr
        if self.trace is not None:
            self.trace.append((tr.t, tr.pos.copy(), tr.quat.copy()))
        lead = float(np.linalg.norm(self.ref - tr.pos[:2]))
        lead_z = abs(self.z_ref - float(tr.pos[2]))
        if lead < self.LEAD_MAX and lead_z < self.LEAD_MAX:
            # xy and z advance together: the reference point moves on the straight segment of the move (the routing
            # checks straight segments; a depth change that ran ahead of the horizontal one hit the roof in a trial)
            d = self.goal - self.ref
            n = float(np.linalg.norm(d))
            step = self._v * dt
            self.ref = self.goal.copy() if n <= step else self.ref + d / n * step
            dz = self.z_goal - self.z_ref
            self.z_ref += float(np.clip(dz, -self._zr * dt, self._zr * dt))
        dy = wrap(self.yaw_goal - self.yaw_ref)
        self.yaw_ref += float(np.clip(dy, -self.yaw_rate * dt, self.yaw_rate * dt))
        e = self.ref - tr.pos[:2]
        self.ei = np.clip(self.ei + e * dt, -0.2, 0.2)
        g = self.gain
        F = np.zeros(3)
        F[:2] = g * (self.KP * e - self.KD * tr.vel_w[:2] + self.KI * self.ei)
        nF = float(np.linalg.norm(F[:2]))
        if nF > self.F_MAX:
            F[:2] *= self.F_MAX / nF
        q = yaw_quat(self.yaw_ref + self.yaw_bias)
        if self.press is not None:
            p = self.press
            u = min(1.0, max(0.0, (t - p["t0"]) / p["T"]))
            F[2] = -(p["F0"] + (p["F1"] - p["F0"]) * u)
            r = cell.call("set_wrench", force=F.tolist(), torque=[0.0, 0.0, 0.0], free=list(FREE_T), frame="world",
                          quat=q)
        else:
            r = cell.call("set_wrench", force=F.tolist(), torque=[0.0, 0.0, 0.0], free=list(FREE_XY), frame="world",
                          quat=q, depth=-self.z_ref + self.depth_bias)
        self.F_last = F
        if isinstance(r, dict) and r.get("accepted") is False:
            # protective stop (motion calls refused): no integration, the reference waits where the vehicle is
            self.stats["refused"] += 1
            self.ei[:] = 0.0
            self.ref = tr.pos[:2].copy()
            self.z_ref = float(tr.pos[2])

    # ------------------------------------------------------------------ moves
    def goto(self, pos, yaw=None, v=None):
        """New goal; the horizontal speed and the depth rate are scaled so that both reach the goal together."""
        self.goal = np.asarray(pos, float)[:2].copy()
        self.z_goal = float(pos[2])
        if yaw is not None:
            self.yaw_goal = float(yaw)
        v = self.v_max if v is None else float(v)
        dxy = float(np.linalg.norm(self.goal - self.ref))
        dz = abs(self.z_goal - self.z_ref)
        T = max(dxy / max(v, 1e-6), dz / max(self.z_rate, 1e-6))
        self._v = dxy / T if T > 1e-9 else v
        self._zr = dz / T if T > 1e-9 else self.z_rate
        self._v, self._zr = max(self._v, 1e-4), max(self._zr, 1e-4)

    def settled(self, tr, tol, tol_z, tol_yaw=6.0, vtol=0.06):
        ok_ref = (float(np.linalg.norm(self.goal - self.ref)) < 1e-6 and abs(self.z_goal - self.z_ref) < 1e-6
                  and abs(wrap(self.yaw_goal - self.yaw_ref)) < 1e-6)
        e = float(np.linalg.norm(self.goal - tr.pos[:2]))
        ez = abs(self.z_goal - tr.pos[2])
        ey = abs(math.degrees(wrap(self.yaw_goal - tr.yaw)))
        v = float(np.linalg.norm(tr.vel_w))
        return ok_ref and e < tol and ez < tol_z and ey < tol_yaw and v < vtol

    def heading_only_off(self, tr, tol, tol_z, tol_yaw=6.0, vtol=0.06):
        """The reference point has arrived and the vehicle is in position, at depth and still, but its true heading is
        not within tol_yaw of the goal."""
        ok_ref = (float(np.linalg.norm(self.goal - self.ref)) < 1e-6 and abs(self.z_goal - self.z_ref) < 1e-6
                  and abs(wrap(self.yaw_goal - self.yaw_ref)) < 1e-6)
        return (ok_ref and float(np.linalg.norm(self.goal - tr.pos[:2])) < tol and abs(self.z_goal - tr.pos[2]) < tol_z
                and float(np.linalg.norm(tr.vel_w)) < vtol and abs(math.degrees(wrap(self.yaw_goal - tr.yaw))) >= tol_yaw)

    def go(self, pos, yaw=None, v=None, tol=0.10, tol_z=0.08, settle_s=0.5, timeout=None, route=True):
        """Fly to pos (true world position of the body origin) with heading yaw; around the keep-out volumes when
        route=True.  Returns True when settled."""
        a = self.cell.truth().pos
        b = np.asarray(pos, float)
        wps = self.geo.route(a, b) if (route and self.geo is not None) else [b]
        if len(wps) > 1:
            self.stats["reroutes"] += 1
            for w in wps[:-1]:
                self._go1(w, yaw, v, 0.25, 0.20, 0.1, None)
        return self._go1(b, yaw, v, tol, tol_z, settle_s, timeout)

    def _go1(self, pos, yaw, v, tol, tol_z, settle_s, timeout):
        self.start()
        self.calibrate()                     # the heading / depth offsets drift: re-reference before every move
        self.goto(pos, yaw, v)
        tr = self.cell.truth()
        dist = float(np.linalg.norm(np.asarray(pos[:2]) - tr.pos[:2]))
        dyaw = 0.0 if yaw is None else abs(wrap(float(yaw) - tr.yaw))
        T_nom = max(dist / max(self._v, 1e-3), abs(float(pos[2]) - tr.pos[2]) / max(self._zr, 1e-3)) + dyaw / self.yaw_rate
        T = timeout or (2.0 * T_nom + 30.0)
        t0 = self.cell.clock()
        good = 0.0
        self.stats["moves"] += 1
        t_cal = t0
        while self.cell.clock() - t0 < T:
            self.cell.step(0.1)
            tr = self.cell.truth()
            if self.settled(tr, tol, tol_z):
                good += 0.1
            else:
                good = 0.0
                # P24a (decision P16-4): the firmware holds ITS heading, whose error depends on the heading itself (the
                # compass deviation of the vehicle's own iron), so a move that turns ends with the true heading 7-12 deg
                # off the goal while position, depth and speed have long arrived (every timeout of P16's runs: only the
                # 6 deg heading test failed).  Once only the heading is off, re-reference the offsets (at most 1 / s).
                if self.cell.clock() - t_cal >= 1.0 and self.heading_only_off(tr, tol, tol_z):
                    self.calibrate()
                    self.stats["recal"] = self.stats.get("recal", 0) + 1
                    t_cal = self.cell.clock()
            if good >= settle_s - 1e-9:
                self.stats["move_s"] += self.cell.clock() - t0
                return True
        self.stats["timeouts"] += 1
        self.stats["move_s"] += self.cell.clock() - t0
        return False

    def wait(self, dt):
        self.start()
        self.cell.step(dt)


# ======================================================================================================= sensors
class Sensors:
    """Frames and sonar scans of the program's own sensors, georeferenced with the true vehicle pose."""

    def __init__(self, cell: Cell, mover: Mover, man, man_marker):
        self.cell, self.mv = cell, mover
        self.man, self.man_marker = man, man_marker
        self.marker_held = True
        self.n_frames = {"640x480": 0, "320x240": 0}
        self.n_pings = 0
        self.lamp = 0.0
        self.tilt_cmd = None

    @property
    def manual(self):
        return self.man_marker if self.marker_held else self.man

    def set_lights(self, level):
        level = float(level)
        if abs(level - self.lamp) > 1e-9:
            self.cell.call("set_lights", level=level)
            self.lamp = level

    def set_tilt(self, deg, tol=0.3, max_wait=3.0):
        """Command the camera tilt and wait for the servo reading."""
        deg = float(deg)
        if self.tilt_cmd is None or abs(self.tilt_cmd - deg) > 1e-6:
            self.cell.call("set_camera_tilt", deg=deg)
            self.tilt_cmd = deg
        t0 = self.cell.clock()
        while True:
            st = self.cell.state()
            cur = st.get("camera_tilt_deg")
            if cur is not None and abs(float(cur) - deg) <= tol:
                return float(cur)
            if self.cell.clock() - t0 > max_wait:
                return None if cur is None else float(cur)
            self.mv.wait(0.05)

    def frame(self, size=(640, 480)):
        """(rgb, cam, info): one frame now; cam = the camera in WORLD axes (perception.pose.Cam) from the true vehicle
        pose, the manual's mount and the tilt servo reading."""
        st = self.cell.state()
        tilt = math.radians(float(st.get("camera_tilt_deg") or 0.0))
        tr = self.cell.truth()
        r = self.cell.call("camera_frame", size=[int(size[0]), int(size[1])])
        rgb = png_decode(base64.b64decode(r["image_png_b64"]))
        self.n_frames["%dx%d" % (rgb.shape[1], rgb.shape[0])] += 1
        cam = PO.Cam(self.manual, tilt, tr.R, rgb.shape[1], rgb.shape[0], p_body=tr.pos)
        return rgb, cam, dict(t=float(r["t"]), tilt_rad=tilt, frame_index=int(r["frame_index"]), pos=tr.pos.copy(),
                              R=tr.R.copy())

    def scan(self, start_deg, stop_deg, range_m, gain=1):
        """One clockwise scan start -> stop (equal = a full turn): list of perception pings dict(t, angle (rad,
        clockwise from the head's forward), data (uint8), range_m, gain, o, R (head pose in world, true))."""
        self.mv.start()
        self.mv.trace = []
        tr0 = self.cell.truth()
        try:
            r = self.cell.call("sonar_scan", start_deg=float(start_deg), stop_deg=float(stop_deg), range_m=float(range_m),
                               gain=int(gain))
        finally:
            trace = self.mv.trace
            self.mv.trace = None
        tr1 = self.cell.truth()
        n, ns = int(r["n_pings"]), int(r["n_samples"])
        data = np.frombuffer(base64.b64decode(r["intensity_u8_b64"]), np.uint8).reshape(n, ns)
        self.n_pings += n
        T = [(tr0.t, tr0.pos, tr0.quat)] + list(trace) + [(tr1.t, tr1.pos, tr1.quat)]
        ts = np.array([x[0] for x in T])
        p_s, R_s = self.man.sonar_in_body()
        out = []
        for k in range(n):
            tk = float(r["t"][k])
            j = int(np.clip(np.searchsorted(ts, tk), 1, len(ts) - 1))
            a, b = T[j - 1], T[j]
            u = 0.0 if b[0] - a[0] <= 1e-12 else float(np.clip((tk - a[0]) / (b[0] - a[0]), 0.0, 1.0))
            pos = a[1] + u * (b[1] - a[1])
            Rb = quat_to_mat(slerp(a[2], b[2], u))
            out.append(dict(t=tk, angle=math.radians(float(r["angle_deg"][k])), data=data[k], range_m=float(r["range_m"]),
                            gain=int(r["gain"]), o=pos + Rb @ p_s, R=Rb @ R_s))
        return out


# ======================================================================================================= the valve turn
class ValveTurn:
    """Close one valve (see the module docstring).  `valve` = dict(stem_w, paddle_centre_w, R_v) from the panel
    standard + the true panel frame + the as-built grid position READ FROM THE PROGRAM'S OWN FRAMES."""

    M_MAX, M_START, KP_RATE, KI_RATE = 12.0, 5.0, 6.0, 8.0
    BACK_N, LEVEL_RATE, BACK_SLOW_N = 15.0, 30.0, 8.0
    DEPTH_KP, DEPTH_KD = 150.0, 120.0
    BACK_M, APPROACH_V = 0.25, 0.08
    SERVO_KP, SERVO_KD, SERVO_KI = 800.0, 250.0, 400.0
    CLOSE_PUSH_N = 15.0
    RATES_DEG_S = (10.0, 5.0, 3.0)
    PADDLE_DEPTH = 0.06

    def __init__(self, cell: Cell, mover: Mover, mdata: dict, valve: dict, bite=0.040):
        self.cell, self.mv, self.md = cell, mover, mdata
        self.a_B, self.tcp_B, self.tip_ahead = mdata["axis_B"], mdata["tcp_B"], mdata["tip_ahead"]
        self.WBG = mdata["WBG"]
        self.v = valve
        self.Rv = np.asarray(valve["R_v"], float)
        self.x_in = self.Rv[:, 0]
        self.bite = float(bite)
        self.out = dict(attempts=0, phases={}, refused=0, protect=0)

    def mark(self, ph):
        self.out["phases"][ph] = round(self.cell.clock(), 3)

    def grip_pose(self, bite, dy=0.0, dz=0.0):
        """(hull position, heading) of a LEVEL vehicle with its TCP at the canonical grasp (G1 / P11 grip_pose_of)."""
        pc = np.asarray(self.v["paddle_centre_w"], float)
        tcp_w = pc + self.Rv @ np.array([-0.5 * self.PADDLE_DEPTH + bite - self.tip_ahead, dy, dz])
        psi = math.atan2(self.x_in[1], self.x_in[0]) - math.atan2(self.a_B[1], self.a_B[0])
        return tcp_w - rz(psi) @ self.tcp_B, psi

    def _call(self, name, **kw):
        r = self.cell.call(name, **kw)
        if isinstance(r, dict) and r.get("accepted") is False:
            self.out["refused"] += 1
            if r.get("reason") == "protective_stop":
                raise ProtectiveStop()
        return r

    def _step(self):
        self.cell.step(CTRL_S)

    # ------------------------------------------------------------------ 1. pre-grasp + drag
    def prepare(self, standoff=0.5, T_settle=1.5, T_measure=5.0):
        """Servo to the pre-grasp pose (standoff along the tool axis), then measure the hover force (depth +
        translation groups, body -> world: the current's drag) and the depth / heading offsets."""
        tp, ty = self.grip_pose(self.bite)
        pre = tp - self.x_in * standoff
        self.tp, self.ty, self.pre = tp, ty, pre
        self.mark("pre")
        self.mv.go(pre - self.x_in * 0.4, ty, v=0.20, tol=0.06, tol_z=0.05, settle_s=0.5)
        self.mv.go(pre, ty, v=0.08, tol=0.02, tol_z=0.02, settle_s=T_settle)
        self.mark("measure")
        acc, n, db, yb = np.zeros(3), 0, 0.0, 0.0
        for _ in range(int(round(T_measure / 0.05))):
            self.mv.wait(0.05)
            s = self.cell.state()
            if s.get("quat_wxyz") is None or s.get("depth_m") is None:
                continue
            wg = s["wrench_groups"]
            acc += quat_to_mat(s["quat_wxyz"]) @ (np.asarray(wg["depth"][:3], float) + np.asarray(wg["translation"][:3], float))
            n += 1
            tr = self.cell.truth()
            db += float(s["depth_m"]) - (-float(tr.pos[2]))
            yb += wrap(float(s["heading_rad"]) - tr.yaw)
        n = max(n, 1)
        self.F_w = acc / n
        self.depth_bias, self.yaw_bias = db / n, yb / n
        self.out.update(hover_force_w=[round(float(x), 3) for x in self.F_w], depth_bias_m=round(self.depth_bias, 4),
                        yaw_bias_deg=round(math.degrees(self.yaw_bias), 3), n_measure=n)

    # ------------------------------------------------------------------ 2. approach + close
    def approach_and_close(self, tp, yaw):
        self.mv.stop()
        self.mark("approach")
        F_h = np.array(self.F_w, float)
        q_lvl = euler_quat(0.0, 0.0, yaw + self.yaw_bias).tolist()
        d_t = float(-tp[2]) + self.depth_bias
        p0 = self.cell.truth().pos[:2].copy()
        T_ramp = float(np.linalg.norm(tp[:2] - p0)) / self.APPROACH_V
        t0 = self.cell.clock()
        ei = np.zeros(2)
        good, closing, t_close = 0.0, False, None
        while True:
            t = self.cell.clock() - t0
            u = min(1.0, t / max(T_ramp, 1e-6))
            ref = p0 + (tp[:2] - p0) * u
            tr = self.cell.truth()
            e = ref - tr.pos[:2]
            vw = tr.vel_w[:2]
            ei = np.clip(ei + e * CTRL_S, -0.2, 0.2)
            F = F_h.copy()
            F[:2] = F_h[:2] + self.SERVO_KP * e - self.SERVO_KD * vw + self.SERVO_KI * ei
            if closing:
                F[:2] += self.CLOSE_PUSH_N * self.x_in[:2]
            self._call("set_wrench", force=F.tolist(), torque=[0.0, 0.0, 0.0], free=list(FREE_XY), frame="world",
                       quat=q_lvl, depth=d_t)
            self._step()
            if not closing:
                s = self.cell.state()
                okp = (u >= 1.0 and np.linalg.norm(e) < 0.005 and np.linalg.norm(vw) < 0.01 and
                       s.get("depth_m") is not None and abs(float(s["depth_m"]) - d_t) < 0.003)
                good = good + CTRL_S if okp else 0.0
                if good >= 0.3 or (u >= 1.0 and t - T_ramp > 20.0):
                    self.out.setdefault("approach_settle_s", []).append(round(t - T_ramp, 2))
                    tr = self.cell.truth()
                    rel = self.Rv.T @ (tr.tcp - (np.asarray(self.v["paddle_centre_w"], float)
                                                 + self.Rv @ np.array([-0.5 * self.PADDLE_DEPTH + self.bite - self.tip_ahead, 0, 0])))
                    self.out.setdefault("tcp_err_at_close_mm", []).append([round(1000 * float(x), 1) for x in rel])
                    self.mark("close")
                    self._call("gripper_close")
                    closing, t_close = True, self.cell.clock()
            else:
                g = self.cell.state()["gripper"]
                if g in ("closed_empty", "stalled_on_object") or self.cell.clock() - t_close > 3.0:
                    return g == "stalled_on_object"

    def approach_only(self, tp, yaw, T_hold=2.0):
        F_h = np.array(self.F_w, float)
        q_lvl = euler_quat(0.0, 0.0, yaw + self.yaw_bias).tolist()
        d_t = float(-tp[2]) + self.depth_bias
        p0 = self.cell.truth().pos[:2].copy()
        T_ramp = float(np.linalg.norm(tp[:2] - p0)) / self.APPROACH_V
        t0 = self.cell.clock()
        ei = np.zeros(2)
        while self.cell.clock() - t0 < T_ramp + T_hold:
            u = min(1.0, (self.cell.clock() - t0) / max(T_ramp, 1e-6))
            ref = p0 + (tp[:2] - p0) * u
            tr = self.cell.truth()
            e = ref - tr.pos[:2]
            ei = np.clip(ei + e * CTRL_S, -0.2, 0.2)
            F = F_h.copy()
            F[:2] = F_h[:2] + self.SERVO_KP * e - self.SERVO_KD * tr.vel_w[:2] + self.SERVO_KI * ei
            self._call("set_wrench", force=F.tolist(), torque=[0.0, 0.0, 0.0], free=list(FREE_XY), frame="world",
                       quat=q_lvl, depth=d_t)
            self._step()

    # ------------------------------------------------------------------ 3. turn
    def _phi(self, s, q_g):
        q_rel = quat_mul(quat_conj(q_g), np.asarray(s["quat_wxyz"], float))
        v = q_rel[1:]
        return 2.0 * math.atan2(float(v @ self.a_B), float(q_rel[0])), float(np.asarray(s["rates_b_rad_s"]) @ self.a_B)

    def turn(self):
        self.mark("turn")
        s = self.cell.state()
        q_g = np.asarray(s["quat_wxyz"], float)
        self.q_grasp = q_g
        self.yaw_level = rpy(q_g)[2]
        M_i = self.M_START
        force = self.F_w.tolist()
        self._call("set_wrench", force=force, torque=(self.M_START * self.a_B).tolist(), free=list(FREE_ALL), frame="world")
        t0 = self.cell.clock()
        self.out["t_torque_start"] = round(t0, 3)
        t_seated = self.md["seat_s"] + 0.05
        still, seated, done = 0.0, False, False
        while self.cell.clock() - t0 < 60.0:
            self._step()
            s = self.cell.state()
            if s["gripper"] != "stalled_on_object":
                self.out["turn_aborted"] = s["gripper"]
                break
            phi, om = self._phi(s, q_g)
            deg = math.degrees(phi)
            r1, r2, r3 = self.RATES_DEG_S
            w_sp = math.radians(r1 if deg < 75.0 else (r2 if deg < 85.0 else r3))
            e = w_sp - om
            M_i = min(max(M_i + self.KI_RATE * e * CTRL_S, -2.0), self.M_MAX)
            M = min(max(M_i + self.KP_RATE * e, 0.0), self.M_MAX)
            self._call("set_wrench", force=force, torque=(M * self.a_B).tolist(), free=list(FREE_ALL), frame="world")
            if not seated and self.cell.clock() - t0 > t_seated:
                self.yaw_level = rpy(s["quat_wxyz"])[2]            # heading after the seating glide
                seated = True
            still = still + CTRL_S if abs(om) < math.radians(1.0) else 0.0
            if (deg >= 88.0 and still >= 0.5) or (deg >= 85.0 and still >= 1.0):
                done = True
                break
        self.out["turn_done"] = done
        self.out["phi_end_deg"] = round(math.degrees(self._phi(self.cell.state(), q_g)[0]), 2)
        self.out["turn_s"] = round(self.cell.clock() - t0, 2)
        return done

    # ------------------------------------------------------------------ 4. release, back off, level
    def _depth_pd(self, s):
        d, t = float(s["depth_m"]), float(s["t"])
        if self._dprev is None:
            self._dprev = (d, t)
        d0, t0 = self._dprev
        rate = (d - d0) / (t - t0) if t - t0 > 0.04 else self._drate
        if t - t0 > 0.04:
            self._dprev, self._drate = (d, t), rate
        return self.DEPTH_KP * (d - self.depth_hold) + self.DEPTH_KD * rate

    def release_backoff_level(self):
        a_w = self.x_in
        s = self.cell.state()
        self.depth_hold, self._dprev, self._drate = float(s["depth_m"]), None, 0.0
        self.mark("release")
        tr = self.cell.truth()
        tip_g = tr.tcp + tr.tcp_R[:, 0] * self.tip_ahead

        def ff(back, s_):
            F = np.array(self.F_w, float) - back * a_w
            F[2] += self._depth_pd(s_)
            return F.tolist()

        def tq(s_):
            ph, _ = self._phi(s_, self.q_grasp)
            return (self.WBG * math.sin(ph) * self.a_B).tolist()
        self._call("set_wrench", force=ff(0.0, s), torque=tq(s), free=list(FREE_ALL), frame="world")
        self._call("gripper_open")
        t_r = self.cell.clock()
        while self.cell.clock() - t_r < 0.25:
            self._step()
            s = self.cell.state()
            self._call("set_wrench", force=ff(0.0, s), torque=tq(s), free=list(FREE_ALL), frame="world")
        self.mark("backoff")
        t0 = self.cell.clock()
        while self.cell.clock() - t0 < 8.0:
            s = self.cell.state()
            self._call("set_wrench", force=ff(self.BACK_N, s), torque=tq(s), free=list(FREE_ALL), frame="world")
            self._step()
            tr = self.cell.truth()
            if float((tip_g - (tr.tcp + tr.tcp_R[:, 0] * self.tip_ahead)) @ a_w) >= self.BACK_M:
                break
        self.out["backoff_s"] = round(self.cell.clock() - t0, 2)
        self.mark("level")
        q0 = np.asarray(self.cell.state()["quat_wxyz"], float)
        q_level = euler_quat(0.0, 0.0, self.yaw_level)
        ang = 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q_level)))))
        T_l = max(0.5, ang / math.radians(self.LEVEL_RATE))
        t0 = self.cell.clock()
        levelled = False
        while self.cell.clock() - t0 < 25.0:
            el = self.cell.clock() - t0
            u = min(1.0, el / T_l)
            q_sp = slerp(q0, q_level, u)
            s = self.cell.state()
            r, p_, _ = rpy(s["quat_wxyz"])
            if u >= 1.0 and el > T_l + 1.0 and abs(r) < math.radians(5.0) and abs(p_) < math.radians(5.0):
                levelled = True
                break
            back = self.BACK_N if u < 1.0 else self.BACK_SLOW_N
            self._call("set_wrench", force=ff(back, s), torque=[0.0, 0.0, 0.0], free=list(FREE_T), frame="world",
                       quat=q_sp.tolist())
            self._step()
        tr = self.cell.truth()
        self.out.update(level_s=round(self.cell.clock() - t0, 2), levelled=levelled,
                        back_total_m=round(float((tip_g - (tr.tcp + tr.tcp_R[:, 0] * self.tip_ahead)) @ a_w), 3))
        self.mark("servo_after")
        self.mv.hold_here()

    # ------------------------------------------------------------------ the whole sequence
    def run(self, attempts=3):
        """Pre-grasp, drag, up to `attempts` approaches, turn, release / back off / level.  Returns self.out with
        grasped / turn_done / phi_end_deg."""
        self.prepare()
        ok = False
        for k in range(attempts):
            self.out["attempts"] = k + 1
            ok = self.approach_and_close(self.tp, self.ty)
            if ok:
                break
            self._call("gripper_open")
            self.cell.step(self.md["stroke_s"] + 0.1)
            bp, _ = self.grip_pose(-0.10)
            self.approach_only(bp, self.ty, T_hold=2.0)
        self.out["grasped"] = bool(ok)
        if not ok:
            self.mv.hold_here()
            return self.out
        self.turn()
        self.release_backoff_level()
        return self.out


class ProtectiveStop(RuntimeError):
    """A motion call was refused by a collision protective stop."""


def wait_protect_clear(cell: Cell, mover: Mover, max_s=15.0):
    """Wait until the firmware accepts motion calls again; then hold the current true pose."""
    mover.stop()
    t0 = cell.clock()
    while cell.clock() - t0 < max_s:
        st = cell.state()
        if st.get("protect_phase") is None:
            break
        cell.step(0.1)
    mover.hold_here()
