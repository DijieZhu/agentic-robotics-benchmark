"""The core seven items of the privileged reference program of task 07 (package P15; DESIGN v1.2 sections 1.1, 2.2,
2.3, 3.3, 3.7; G6 reference_lite on the product cell).

Score items reached here: M1 + M2 (marker on the leaking branch's roof zone), H (survey of the 18 valve positions, no
critical error, vehicle not disabled), I1 + I2 + I3 (the isolation set closed, nothing else moved), R (vehicle at rest in
the cage: seated on its deck after a slow descent with a force ramp) = 0.70 of the 1.00.  P16 adds the tide wait,
the dropped object and the hot stab (steps_contact.py).

DECISIONS, each from the program's own data only (moves.py is the privileged motion / sensor layer):
  * leak branch      perception.plume_fit.locate (G3 pilot v3 = P14's port) on the program's own sonar pings and
                     frames; the GA flanges as candidates;
  * tags             perception.readers (fixed readers: codes first, painted tags where a code is covered) on its own
                     frames; the as-built grid position of every tag = the frame's pixel ray (true camera pose)
                     intersected with the (true) panel face, snapped to the panel standard's grid;
  * valve states     perception.readers.read_handle (v4 AND v5 agree, own-gripper guard) on its own level frames;
  * targets          the leak branch's valves that the program READ as open (never the state table, never the
                     drawing's layout);
  * contact window   its own free-drift current measurement (perception.current.drift_fit on its DVL velocities)
                     at the top valve row's height in front of the panel (window: <= 0.30 m/s there; the program waits
                     while it measures more than WINDOW_GO_M_S);
  * isolation check  its own sonar section above the roof after the closures (echo energy vs the survey).
Set-points (where to fly) come from the true structure / panel / cage frames and the public type drawings (panel
standard, manifold general arrangement, cage drawing); the program never reads a valve angle, the leak branch, a
pressure, the flow field or the site package.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from perception import readers as RD
from perception import sonar_map as SMAP
from perception import plume_fit as PF
from perception import current as CU

import moves as MV

# ------------------------------------------------------------------------------------------------- panel standard
# (the field document panel_standard.md / panel_skid drawing; panel frame P: X right, Y up, Z out of the face,
# origin = face plane, middle of the width, mudmat top)
COL_PITCH = 0.60
COLS_X = tuple(round((c - 2.5) * COL_PITCH, 6) for c in range(6))          # -1.5 .. 1.5
ROWS_Y = (1.50, 2.10, 2.70)                                                 # stem heights above the mudmat top
PORT_Y = 0.55
PORTS_X = tuple(round(-2.1 + 0.6 * i, 6) for i in range(8))                 # PARK-A, TP-1..6, PARK-B
STEM_STANDOFF = 0.15                                                        # stem to the blade centre
PLATE_ABOVE = 0.22                                                          # plate centre above its stem
PLATE_W, PLATE_QUIET, CODE_SIZE, PLATE_THK = 0.424, 0.020, 0.160, 0.003
CODE_DX = -PLATE_W / 2 + PLATE_QUIET + CODE_SIZE / 2                        # code centre left of the stem
SKID_HALF_W, SKID_TOP = 2.71, 3.30
SKID_BEHIND = 1.2                                                           # skid depth behind the face (0.55) + margin
VALVE_TAGS = tuple(f"{k}V{i}" for k in "BPT" for i in range(1, 7))

# ------------------------------------------------------------------------------------------------- program constants
LIGHTS = 0.8
TRANSIT_ABOVE_ROOF = 3.5                   # transit / coarse scan height (G3 / G6)
SAFE_ABOVE_ROOF = 1.2                      # crossing height over the structure (vehicle origin)
FRONT_CLEAR = 1.6                          # distance from the face of the 'clear front' (routing waypoints)
DANGER_FACE = 0.95                         # body origin closer than this to the face = keep-out (G6 fix 3)
INSPECT_STANDOFF = 1.0                     # camera to the plates
HANDLE_LOOKS = ((0.0, 0.75), (-0.08, 0.70), (0.08, 0.80), (0.0, 0.65), (0.0, 0.90), (-0.12, 0.85), (0.12, 0.70))
WINDOW_GO_M_S = 0.27                       # measured current at the top row's height that opens the contact work
WINDOW_GIVE_UP_S = 1000.0                  # the field schedule guarantees a window; stop measuring after this
WINDOW_WEAK_M_S = 0.15                     # one measurement below this: weak flow, the window is open
WINDOW_AVG_S = 150.0                       # otherwise: mean of the measurements of this long ...
WINDOW_MIN_N = 4                           # ... at least this many
DRIFT_BOX = (-1.6, 1.6, 2.4, 4.4)          # panel frame X0, X1, Z0, Z1 of the drift (over the skid mudmat, DVL lock)
DRIFT_HALF_M = 1.4                         # the drift starts this far upstream of the box centre
DRIFT_S, DRIFT_AVG_AFTER_S = 8.0, 2.5      # drift length and the start of the averaging [s]
MARKER_DROP = 0.10                         # block bottom above the roof at the release
PLUME_T0 = 150.0                           # the plume survey starts no earlier (the generator's checked regime)
WALL0 = time.perf_counter
VERBOSE = bool(os.environ.get("SUBSEACELL_REFERENCE_VERBOSE"))          # maintainer runs: step marks on stderr


class Keepout:
    """Keep-out volumes for the routing (true structure and panel frames + the type drawings): the structure box
    below roof + 0.30 m (+ 0.9 m margin round), the panel skid's front band closer than DANGER_FACE below the skid
    top + 0.6 m, and anything below 0.6 m above the mudline."""

    def __init__(self, scene: MV.Scene, ga):
        self.sc, self.ga = scene, ga
        self.roof_w = ga.roof_w
        self.z_safe = ga.roof_w + SAFE_ABOVE_ROOF
        self.hx, self.hy = ga.lx / 2 + 0.9, ga.ly / 2 + 0.9
        self.unsafe_legs = []

    def in_structure(self, p):
        s = self.sc.w_to_S(p)
        return abs(s[0]) < self.hx and abs(s[1]) < self.hy and s[2] < self.ga.roof_z + 0.30

    def in_front_band(self, p, clear=DANGER_FACE):
        """In the slab in front of (and through) the panel skid: -SKID_BEHIND < Z_P < clear (behind that is the
        structure's own box)."""
        q = self.sc.w_to_P(p)
        return -SKID_BEHIND < q[2] < clear and abs(q[0]) < SKID_HALF_W + 0.6 and q[1] < SKID_TOP + 0.6

    def blocked(self, p):
        return self.in_structure(p) or self.in_front_band(p) or p[2] < self.sc.mudline + 0.6

    def direct_ok(self, a, b):
        qa, qb = self.sc.w_to_P(a), self.sc.w_to_P(b)
        if abs(qb[0] - qa[0]) < 0.12 and abs(qb[1] - qa[1]) < 0.12 and min(qa[2], qb[2]) > 0.3:
            return True                                    # a pure approach / retreat along the face normal
        for u in np.linspace(0.0, 1.0, 61):
            if self.blocked(a + u * (b - a)):
                return False
        return True

    def route(self, a, b):
        """Waypoints a -> b (the last is b): straight if clear; else back off the panel front to FRONT_CLEAR, climb to
        the crossing height, cross, descend (to FRONT_CLEAR in front of the panel when b is there), go in."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if self.direct_ok(a, b):
            return [b]
        pts, cur = [], a.copy()
        if self.in_front_band(cur, FRONT_CLEAR) and self.sc.w_to_P(cur)[2] > 0.0:
            q = self.sc.w_to_P(cur)
            cur = self.sc.P_to_w([q[0], q[1], FRONT_CLEAR])
            pts.append(cur)
        entry = b
        qb = self.sc.w_to_P(b)
        if self.in_front_band(b, FRONT_CLEAR) and qb[2] > 0.0:
            entry = self.sc.P_to_w([qb[0], qb[1], FRONT_CLEAR])
        if not self.direct_ok(cur, entry):
            zs = max(self.z_safe, cur[2], entry[2])
            pts += [np.array([cur[0], cur[1], zs]), np.array([entry[0], entry[1], zs])]
        pts.append(entry)
        if entry is not b:
            pts.append(b)
        prev = a
        for w in pts:                                   # every leg must be clear (recorded, not silently flown)
            if not self.direct_ok(prev, w):
                self.unsafe_legs.append(dict(a=[round(float(x), 2) for x in self.sc.w_to_S(prev)],
                                             b=[round(float(x), 2) for x in self.sc.w_to_S(w)]))
            prev = w
        return pts


def valve_geom(scene: MV.Scene, col: int, row: int) -> dict:
    """World geometry of the valve at the as-built grid position (col, row) from the panel standard and the true panel
    frame: stem point, blade centre, valve frame R_v = [x into the panel, y = viewer's left, z up]."""
    stem = scene.P_to_w([COLS_X[col], ROWS_Y[row], 0.0])
    R = scene.P_R
    Rv = np.stack([-R[:, 2], -R[:, 0], R[:, 1]], 1)
    return dict(stem_w=stem, paddle_centre_w=stem + R[:, 2] * STEM_STANDOFF, R_v=Rv, col=col, row=row)


# ======================================================================================================= panel book
class PanelBook:
    """The program's own picture of the panel: tag -> as-built grid position (votes), handle readings (votes)."""

    def __init__(self, scene: MV.Scene):
        self.sc = scene
        self.code_c = {(c, r): np.array([COLS_X[c] + CODE_DX, ROWS_Y[r] + PLATE_ABOVE]) for c in range(6) for r in range(3)}
        self.plate_c = {(c, r): np.array([COLS_X[c], ROWS_Y[r] + PLATE_ABOVE]) for c in range(6) for r in range(3)}
        self.pos_votes = {}
        self.handle_votes = {}
        self.reads = []
        self.n_frames = 0

    def _face_point(self, cam, uv):
        d = cam.rays(np.array([uv[0]]), np.array([uv[1]]))[0]
        Z = self.sc.P_R[:, 2]
        den = float(d @ Z)
        if abs(den) < 1e-6:
            return None
        s = float((self.sc.P_o + Z * PLATE_THK - cam.p) @ Z) / den
        if s <= 0:
            return None
        return self.sc.w_to_P(cam.p + s * d)[:2]

    @staticmethod
    def _nearest(table, xy):
        best, bd = None, 1e9
        for k in sorted(table):
            dd = float(np.linalg.norm(table[k] - xy))
            if dd < bd:
                best, bd = k, dd
        return best, bd

    def read(self, rgb, cam, man, tilt_rad, t=None) -> dict:
        """Read one frame: returns {tag: dict(pos, handle, source)} for the valve plates seen."""
        self.n_frames += 1
        ex = RD.own_gripper_mask(man, tilt_rad, rgb.shape[:2])
        recs = RD.read_panel(rgb, man.hfov_deg, roll_deg=0.0, exclude=ex)
        seen = {}
        for r in recs:
            tag = r["tag"]
            if tag not in VALVE_TAGS:
                continue
            xy = self._face_point(cam, r["corners"].mean(0))
            if xy is None:
                continue
            table = self.code_c if r["source"] == "code" else self.plate_c
            k, dd = self._nearest(table, xy)
            if dd > 0.15:
                continue
            pv = self.pos_votes.setdefault(tag, {})
            pv[k] = pv.get(k, 0) + 1
            st = r["handle"]
            if st in ("open", "closed"):
                hv = self.handle_votes.setdefault(tag, {"open": 0, "closed": 0})
                hv[st] += 1
            seen[tag] = dict(pos=k, handle=st, source=r["source"],
                             reason=None if r["handle_info"] is None else r["handle_info"].get("reason"))
            self.reads.append(dict(t=t, tag=tag, pos=list(k), via=r["source"], handle=st))
        return seen

    def position(self, tag):
        pv = self.pos_votes.get(tag)
        if not pv:
            return None
        return sorted(pv.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def layout(self) -> dict:
        """tag -> grid position, one tag per position (more votes wins a conflict)."""
        cand = []
        for tag in sorted(self.pos_votes):
            for k, n in sorted(self.pos_votes[tag].items()):
                cand.append((-n, tag, k))
        cand.sort()
        out, used = {}, set()
        for n, tag, k in cand:
            if tag in out or k in used:
                continue
            out[tag] = k
            used.add(k)
        return out

    def handle(self, tag, need=2):
        """'open' | 'closed' | None: at least `need` agreeing reads and none against."""
        v = self.handle_votes.get(tag)
        if not v:
            return None
        if v["open"] >= need and v["closed"] == 0:
            return "open"
        if v["closed"] >= need and v["open"] == 0:
            return "closed"
        return None


# ======================================================================================================= plume platform
class Platform:
    """perception.plume_fit's platform on the dynamic vehicle: moves = the privileged servo (settle at the requested
    sensor pose), pings and frames = the program's own sensors at the current time."""

    def __init__(self, prog):
        self.p = prog
        self.mv, self.sens, self.cell = prog.mv, prog.sens, prog.cell
        self.n_moves = 0

    @property
    def t(self):
        return self.cell.clock()

    @property
    def outline(self):
        return self.sens.manual.outline

    def wait(self, dt):
        if dt > 0:
            self.mv.wait(dt)

    def put_sonar(self, xy_w, z_w, yaw):
        p_s, _ = self.sens.man.sonar_in_body()
        target = np.r_[np.asarray(xy_w, float)[:2], float(z_w)] - MV.rz(float(yaw)) @ p_s
        self.n_moves += 1
        self.mv.go(target, float(yaw), tol=0.25, tol_z=0.20, settle_s=0.2)

    def scan360(self, range_m, gain=1):
        return self.sens.scan(0.0, 0.0, range_m, gain)

    def sector(self, bearing, half_deg, range_m, n_sweeps, gain=1):
        tr = self.cell.truth()
        if abs(math.degrees(MV.wrap(float(bearing) - tr.yaw))) > 3.0:
            self.mv.go(np.r_[self.mv.goal, self.mv.z_goal], float(bearing), tol=0.25, tol_z=0.20, settle_s=0.2)
        return [self.sens.scan(-float(half_deg), float(half_deg), range_m, gain) for _ in range(int(n_sweeps))]

    def put_camera(self, xyz_w, yaw, tilt_rad):
        self.sens.set_tilt(math.degrees(float(tilt_rad)))
        p_c, _ = self.sens.man.camera_in_body(float(tilt_rad))
        target = np.asarray(xyz_w, float) - MV.rz(float(yaw)) @ p_c
        self.n_moves += 1
        self.mv.go(target, float(yaw), tol=0.15, tol_z=0.12, settle_s=0.3)

    def frame(self):
        self.sens.set_lights(1.0)
        rgb, cam, _ = self.sens.frame((640, 480))
        return rgb, cam


# ======================================================================================================= the program
class CoreProgram:
    def __init__(self, server, opts=None):
        self.w0 = WALL0()
        self.opts = dict(opts or {})
        self.cell = MV.Cell(server)
        self.man, self.man_mk, self.md = MV.load_manual()
        sc = self.cell.scene
        self.sc = sc
        self.ga = SMAP.ManifoldGA(origin_w=sc.S_o, psi=sc.psi, z_bed_w=sc.mudline)
        self.keep = Keepout(sc, self.ga)
        self.mv = MV.Mover(self.cell, geo=self.keep)
        self.sens = MV.Sensors(self.cell, self.mv, self.man, self.man_mk)
        self.book = PanelBook(sc)
        self.start_truth = self.cell.truth()
        self.out = dict(steps={}, valves=[], notes=[], errors=[])
        self.branch = None
        self.plume = None

    # ------------------------------------------------------------------ bookkeeping
    def mark(self, key):
        self.out["steps"][key] = dict(t=round(self.cell.clock(), 2), wall=round(WALL0() - self.w0, 1))
        if VERBOSE:
            print(f"[reference] t {self.cell.clock():7.1f}  wall {WALL0() - self.w0:6.0f}  {key}", file=sys.stderr, flush=True)

    def note(self, s):
        self.out["notes"].append(f"t={self.cell.clock():.1f} {s}")

    # ------------------------------------------------------------------ 0. every sensor once (U1 3.1(e))
    def sensors_first(self):
        self.mark("sensors_first")
        c = self.cell
        c.call("state")
        c.call("mission")
        self.sens.set_lights(LIGHTS)
        r640 = c.call("camera_frame", size=[640, 480])
        r320 = c.call("camera_frame", size=[320, 240])
        sc = c.call("sonar_scan", start_deg=-4.5, stop_deg=4.5, range_m=5.0, gain=1)
        self.out["sensors_first"] = dict(frame640=[r640["width"], r640["height"]], frame320=[r320["width"], r320["height"]],
                                         pings=int(sc["n_pings"]), order=list(c.order))
        self.sens.n_frames["640x480"] += 1
        self.sens.n_frames["320x240"] += 1
        self.sens.n_pings += int(sc["n_pings"])

    # ------------------------------------------------------------------ 1. leave the cage
    def leave_cage(self):
        self.mark("leave_cage")
        tr = self.cell.truth()
        self.cell.call("set_nav_fix", x=float(tr.pos[0]), y=float(tr.pos[1]))
        self.mv.start()
        z_up = self.start_truth.pos[2] + 0.30
        self.mv.z_rate = 0.10
        self.mv.go(np.r_[tr.pos[:2], z_up], self.sc.cage_yaw, tol=0.05, tol_z=0.04, settle_s=0.5, route=False)
        self.mv.z_rate = 0.35
        out = self.sc.cage_pos + self.sc.cage_R @ np.array([3.0, 0.0, 0.0])
        self.mv.go(np.r_[out[:2], z_up], self.sc.cage_yaw, v=0.25, tol=0.15, tol_z=0.08, settle_s=0.3, route=False)

    # ------------------------------------------------------------------ 2. find the structure
    def find_structure(self):
        """On the way in: one 360 deg 15 m scan about 12 m short of the CHARTED position (the record of finding the
        manifold; the program then uses the as-built frame for its set-points) and one forward frame."""
        mission = self.cell.call("mission")
        ch = np.asarray(mission["chart"]["origin_xy"], float)
        tr = self.cell.truth()
        z_t = self.ga.roof_w + TRANSIT_ABOVE_ROOF
        d = ch - tr.pos[:2]
        n = float(np.linalg.norm(d))
        stop = tr.pos[:2] + d / max(n, 1e-6) * max(0.0, n - 12.0)
        yaw = math.atan2(d[1], d[0])
        self.mark("transit")
        self.mv.go(np.r_[stop, z_t], yaw, tol=0.25, tol_z=0.15, settle_s=0.2)
        self.mark("find_structure")
        pings = self.sens.scan(0.0, 0.0, 15.0)
        # the sanity record: strongest returns near the as-built footprint
        g = SMAP.Grid(float(stop[0]), float(stop[1]), 16.0, 0.25)
        S, N = SMAP.accumulate(g, pings)
        M = SMAP.smooth_map(g, S, N, 1.5)
        X, Y = g.centres()
        if np.isfinite(M).any():
            k = int(np.nanargmax(np.where(np.isfinite(M), M, -np.inf)))
            peak = np.array([X.ravel()[k], Y.ravel()[k]])
            sp = self.sc.w_to_S(np.r_[peak, self.ga.roof_w])
            self.out["find"] = dict(n_pings=len(pings), peak_S=[round(float(sp[0]), 2), round(float(sp[1]), 2)],
                                    peak_in_footprint=bool(abs(sp[0]) < self.ga.lx / 2 + 1.0 and abs(sp[1]) < self.ga.ly / 2 + 2.0),
                                    chart_offset_m=round(float(np.linalg.norm(ch - self.sc.S_o[:2])), 2))
        self.sens.set_tilt(-15.0)
        self.sens.frame((640, 480))

    # ------------------------------------------------------------------ 3. leak source
    def locate_leak(self):
        """The survey starts no earlier than PLUME_T0 (the generator checks the site's plume regime from 150 s on,
        DESIGN 2.1 / 2.5; the program waits above the structure, at the coarse scan point)."""
        self.mark("plume")
        pf = Platform(self)
        if self.cell.clock() < PLUME_T0:
            p_s, _ = self.sens.man.sonar_in_body()
            c0 = self.ga.to_w(np.array([0.0, 0.0, 0.0]))[:2]
            pf.put_sonar(c0, self.ga.origin_w[2] + self.ga.roof_z + 3.5, 0.0)
            if self.cell.clock() < PLUME_T0:
                self.mv.wait(PLUME_T0 - self.cell.clock())
        self.mark("plume_start")
        res = PF.locate(pf, self.ga)
        self.plume = res
        self.branch = res.get("branch")
        keep = {k: res[k] for k in ("branch", "slot", "flange", "exit_used", "camera_minus_sonar_m",
                                    "best_other_branch_residual", "t_used", "reason") if k in res}
        for k in ("exit_S", "s0_S", "x_cont_S", "x_ga_S", "exit_sonar_S", "exit_camera_S"):
            if res.get(k) is not None:
                keep[k] = [round(float(x), 3) for x in np.asarray(res[k]).ravel()]
        keep["n_sections"] = len(res.get("sections", []))
        keep["sections"] = [dict(z_s=round(float(x["z_s"]), 2), t0=round(float(x["t0"]), 1),
                                 c_S=None if "c_S" not in x else [round(float(v), 3) for v in x["c_S"]],
                                 area=x.get("area"), energy=x.get("energy")) for x in res.get("sections", [])]
        keep["candidates"] = [{k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in c.items()}
                              for c in res.get("candidates", [])]
        co = res.get("coarse") or {}
        keep["coarse"] = {k: [round(float(v), 3) for v in np.asarray(co[k]).ravel()] for k in ("p3_S", "p2_S", "s0_S")
                          if co.get(k) is not None}
        keep["moves"] = pf.n_moves
        self.out["plume"] = keep
        self.mark("plume_done")
        return self.branch

    # ------------------------------------------------------------------ 4. marker
    def place_marker(self):
        if self.branch is None:
            return False
        self.mark("marker")
        slot = self.ga.slots[self.branch - 1]
        zone_w = self.ga.to_w(np.array([slot[0], slot[1], self.ga.roof_z]))
        blk = self.md["marker_block_B"]
        tr = self.cell.truth()
        yaw = tr.yaw
        off = MV.rz(yaw) @ np.array([blk[0], blk[1], 0.0])
        z_rel = self.ga.roof_w + MARKER_DROP - self.md["marker_bottom_B"]
        above = np.array([zone_w[0] - off[0], zone_w[1] - off[1], self.ga.roof_w + 1.0])
        self.mv.go(above, yaw, tol=0.15, tol_z=0.12, settle_s=0.3)
        self.mv.z_rate = 0.12
        rel = np.array([zone_w[0] - off[0], zone_w[1] - off[1], z_rel])
        ok = self.mv.go(rel, yaw, v=0.10, tol=0.03, tol_z=0.03, settle_s=1.0, route=False)
        self.mv.z_rate = 0.35
        tr = self.cell.truth()
        blk_w = tr.pos + tr.R @ blk
        self.out["marker"] = dict(settled=ok, block_xy_err_m=round(float(np.linalg.norm(blk_w[:2] - zone_w[:2])), 4),
                                  block_bottom_above_roof_m=round(float(tr.pos[2] + self.md["marker_bottom_B"] - self.ga.roof_w), 3))
        self.cell.call("gripper_open")
        self.mv.wait(self.md["stroke_s"] + 1.5)
        self.sens.marker_held = False
        self.mv.go(np.r_[rel[:2], self.ga.roof_w + 1.0], yaw, tol=0.2, tol_z=0.15, settle_s=0.1, route=False)
        # look down at the block (record)
        self.sens.set_tilt(-45.0)
        self.sens.frame((640, 480))
        self.mark("marker_done")
        return True

    # ------------------------------------------------------------------ 5. survey
    def cam_target(self, X, Y, Z, tilt_deg):
        """Body-origin position that puts the camera pupil at panel point (X, Y, Z) facing the panel."""
        yaw = self.sc.face_yaw
        p_c, _ = self.man.camera_in_body(math.radians(tilt_deg))
        return self.sc.P_to_w([X, Y, Z]) - MV.rz(yaw) @ p_c, yaw

    def shoot(self, tilt_deg, size=(640, 480)):
        tilt_meas = self.sens.set_tilt(tilt_deg)
        self.sens.set_lights(LIGHTS)
        rgb, cam, info = self.sens.frame(size)
        seen = self.book.read(rgb, cam, self.sens.manual, info["tilt_rad"], t=info["t"])
        return seen, tilt_meas

    def inspect(self):
        """The as-found survey: every plate from INSPECT_STANDOFF, the camera LEVEL at each row's code height (the
        own jaws and the carried marker stay below the field of view: a 31 deg down-look at the bottom row lost it
        behind the jaws in the first trials), columns in turn, rows serpentine."""
        self.mark("inspect")
        tr = self.cell.truth()
        q = self.sc.w_to_P(tr.pos)
        cols = list(range(6)) if q[0] <= 0 else list(range(5, -1, -1))
        n = 0
        for i, c in enumerate(cols):
            X = COLS_X[c] + CODE_DX
            rows = (2, 1, 0) if i % 2 == 0 else (0, 1, 2)
            for r in rows:
                pos, yaw = self.cam_target(X, ROWS_Y[r] + PLATE_ABOVE, INSPECT_STANDOFF + PLATE_THK, 0.0)
                self.mv.go(pos, yaw, v=0.30, tol=0.04, tol_z=0.03, settle_s=0.4)
                self.shoot(0.0)
                n += 1
        lay = self.book.layout()
        missing = [t for t in VALVE_TAGS if t not in lay]
        self.out["inspection"] = dict(frames=n, tags_located=len(lay), missing_after_survey=missing)
        # second looks at grid positions without a tag (closer, level)
        if missing:
            used = set(lay.values())
            for k in [(c, r) for c in range(6) for r in range(3) if (c, r) not in used]:
                c, r = k
                pos, yaw = self.cam_target(COLS_X[c] + CODE_DX, ROWS_Y[r] + PLATE_ABOVE, 0.8, 0.0)
                self.mv.go(pos, yaw, v=0.25, tol=0.04, tol_z=0.03, settle_s=0.4)
                self.shoot(0.0)
            lay = self.book.layout()
        self.layout = lay
        self.out["inspection"].update(layout={t: list(lay[t]) for t in sorted(lay)},
                                      missing_final=[t for t in VALVE_TAGS if t not in lay],
                                      handles={t: self.book.handle(t) for t in VALVE_TAGS})
        self.mark("inspect_done")

    def look_handle(self, tag, need=2, max_looks=len(HANDLE_LOOKS)):
        """Level frames of one handle (P14: level, 0.6-1.3 m, blade clear of the own gripper) until the fixed reader
        gives `need` agreeing answers (with none against) or the looks run out."""
        k = self.layout.get(tag)
        if k is None:
            return None
        c, r = k
        res = []
        for dx, dist in HANDLE_LOOKS[:max_looks]:
            pos, yaw = self.cam_target(COLS_X[c] + dx, ROWS_Y[r] + 0.08, dist + STEM_STANDOFF, 0.0)
            self.mv.go(pos, yaw, v=0.20, tol=0.04, tol_z=0.04, settle_s=0.4)
            seen, _ = self.shoot(0.0)
            st = (seen.get(tag) or {}).get("handle")
            res.append(st)
            if self.book.handle(tag, need) is not None:
                break
        return dict(tag=tag, reads=res, state=self.book.handle(tag, need), votes=dict(self.book.handle_votes.get(tag, {})))

    # ------------------------------------------------------------------ 6. contact window (own current measurement)
    def drift_start(self, d):
        """Start of a short free drift inside DRIFT_BOX (panel frame X / Z at the top row's height): the box lies over
        the panel skid's mudmat, clear of its edges and of the dropped object, where the DVL keeps its bottom lock
        (beams over the mudmat edges, over open frame tubes or through the roof grating lose it: P11 finding 5, and
        this package's trials); the drift runs across the box's centre along the current."""
        sc = self.sc
        x0, x1, z0, z1 = DRIFT_BOX
        dP = np.array([float(d @ sc.P_R[:2, 0]), float(d @ sc.P_R[:2, 2])])
        n = float(np.linalg.norm(dP))
        dP = dP / n if n > 1e-6 else np.zeros(2)
        c = np.array([0.5 * (x0 + x1), 0.5 * (z0 + z1)])
        st = c - DRIFT_HALF_M * dP
        st = np.array([np.clip(st[0], x0, x1), np.clip(st[1], z0, z1)])
        return sc.P_to_w([st[0], ROWS_Y[2], st[1]])

    def measure_current(self, drift_s=DRIFT_S, avg_after_s=DRIFT_AVG_AFTER_S):
        """One free-drift measurement: hold, take the current's direction from the servo's holding force, start
        upstream inside DRIFT_BOX, let go horizontally (wrench mode, zero horizontal force; depth and attitude held by
        the firmware) and average the DVL velocity over ground after avg_after_s (the vehicle has caught up with the
        water by then: time constant about 1 s, P14).  perception.current.drift_fit is recorded alongside."""
        sc = self.sc
        if sc.w_to_P(self.cell.truth().pos)[2] > 6.0 or self.keep.in_structure(self.cell.truth().pos):
            self.mv.go(sc.P_to_w([0.0, ROWS_Y[2], 3.4]), sc.face_yaw, tol=0.2, tol_z=0.1, settle_s=0.3)
        F = np.zeros(2)
        for _ in range(20):                       # the servo's holding force where the vehicle is: the drag's direction
            self.mv.wait(0.1)
            F += self.mv.F_last[:2]
        F /= 20.0
        nF = float(np.linalg.norm(F))
        d = -F / nF if nF > 0.5 else np.zeros(2)
        S_ = self.drift_start(d)
        yaw = math.atan2(-d[1], -d[0]) if nF > 0.5 else sc.face_yaw          # head into the current
        go_ok = self.mv.go(S_, yaw, tol=0.15, tol_z=0.08, settle_s=0.8)
        st0 = self.cell.state()
        depth_hold = float(st0["depth_m"])
        q_lvl = MV.euler_quat(0.0, 0.0, yaw + self.mv.yaw_bias).tolist()
        self.mv.stop()
        states, aborted = [], None
        t0 = self.cell.clock()
        x0, x1, z0, z1 = DRIFT_BOX
        while self.cell.clock() - t0 < drift_s:
            self.cell.call("set_wrench", force=[0.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0], free=list(MV.FREE_XY),
                           frame="world", quat=q_lvl, depth=depth_hold)
            self.cell.step(0.1)
            s = self.cell.state()
            states.append({k: s[k] for k in ("t", "quat_wxyz", "nav_velocity_m_s", "wrench_groups", "dvl_lock")})
            q = sc.w_to_P(self.cell.truth().pos)
            if not (x0 - 1.0 <= q[0] <= x1 + 1.0 and z0 - 0.6 <= q[2] <= z1 + 1.2):
                aborted = "left the drift box"
                break
        self.mv.hold_here()
        v_fit, info = CU.drift_fit(states)
        late = [s for s in states if s["dvl_lock"] and s["nav_velocity_m_s"] is not None
                and float(s["t"]) - float(states[0]["t"]) >= avg_after_s]
        v = np.mean(np.array([s["nav_velocity_m_s"][:2] for s in late], float), axis=0) if len(late) >= 15 else None
        sp = None if v is None else float(np.linalg.norm(v))
        rec = dict(t=round(self.cell.clock(), 1), speed_m_s=None if sp is None else round(sp, 4),
                   v_w=None if v is None else [round(float(x), 4) for x in v], n_avg=len(late), n_states=len(states),
                   hold_force_N=round(nF, 2), fit_speed_m_s=None if v_fit is None else round(float(np.linalg.norm(v_fit)), 4),
                   tau_s=info.get("tau_s"), go_ok=bool(go_ok), aborted=aborted,
                   start_P=[round(float(x), 2) for x in sc.w_to_P(S_)])
        self.out.setdefault("current", []).append(rec)
        return sp

    def wait_window(self):
        """Contact work starts when the program's own current measurements say the window is open: one measurement
        <= WINDOW_WEAK_M_S (weak flow), or the mean of the measurements of the last WINDOW_AVG_S (at least
        WINDOW_MIN_N; the gusts make a single 5 s drift scatter by about 0.07 m/s) <= WINDOW_GO_M_S.  Measurements are
        taken back to back; WINDOW_GIVE_UP_S ends the wait (the field schedule guarantees a window)."""
        self.mark("window")
        meas = []
        while True:
            sp = self.measure_current()
            t = self.cell.clock()
            if sp is not None:
                meas.append((t, sp))
            recent = [m for m in meas if t - m[0] <= WINDOW_AVG_S]
            est = float(np.mean([m[1] for m in recent])) if len(recent) >= WINDOW_MIN_N else None
            if len(meas) == 1 and sp is not None and sp <= WINDOW_WEAK_M_S:
                self.note(f"contact window: weak flow, measured {sp:.3f} m/s")
                break
            if est is not None and est <= WINDOW_GO_M_S:
                self.note(f"contact window: mean of {len(recent)} measurements {est:.3f} m/s")
                break
            if t >= WINDOW_GIVE_UP_S:
                self.note(f"contact window: estimate {est} m/s at the give-up time; proceeding")
                break
        self.out["window"] = dict(n=len(meas), estimate=None if est is None else round(est, 4), t=round(t, 1))
        self.mark("window_open")

    # ------------------------------------------------------------------ 7. isolation
    def targets(self):
        b = self.branch
        if b is None:
            return []
        out = []
        for k in "BPT":
            tag = f"{k}V{b}"
            if tag not in self.layout:
                self.note(f"{tag}: not located on the panel; not touched")
                continue
            st = self.book.handle(tag)
            if st is None:
                lh = self.look_handle(tag)
                self.out.setdefault("handle_looks", []).append(lh)
                st = lh["state"] if lh else None
            out.append(dict(tag=tag, pos=self.layout[tag], read=st))
        return out

    def close_valve(self, tgt):
        c, r = tgt["pos"]
        vg = valve_geom(self.sc, c, r)
        rec = dict(tag=tgt["tag"], pos=[c, r], read=tgt["read"], t0=round(self.cell.clock(), 2))
        self.mark("close_" + tgt["tag"])
        # come in from the clear front: back off first if close to the panel and off to the side
        tr = self.cell.truth()
        q = self.sc.w_to_P(tr.pos)
        if q[2] < 1.4 and abs(q[0] - COLS_X[c]) > 0.3:
            self.mv.go(self.sc.P_to_w([q[0], q[1], 1.5]), self.sc.face_yaw, v=0.15, tol=0.08, tol_z=0.06, settle_s=0.3)
        vt = MV.ValveTurn(self.cell, self.mv, self.md, vg)
        try:
            res = vt.run()
        except MV.ProtectiveStop:
            self.note(f"{tgt['tag']}: protective stop during the close")
            MV.wait_protect_clear(self.cell, self.mv)
            res = dict(vt.out, protect_abort=True)
        rec.update({k: res.get(k) for k in ("grasped", "attempts", "turn_done", "phi_end_deg", "turn_s", "levelled",
                                            "back_total_m", "hover_force_w", "depth_bias_m", "yaw_bias_deg",
                                            "tcp_err_at_close_mm", "turn_aborted", "refused", "protect_abort")})
        rec["t1"] = round(self.cell.clock(), 2)
        tr = self.cell.truth()
        q = self.sc.w_to_P(tr.pos)
        self.mv.go(self.sc.P_to_w([q[0], q[1], 1.3]), self.sc.face_yaw, v=0.15, tol=0.08, tol_z=0.06, settle_s=0.3)
        self.out["valves"].append(rec)
        return rec

    def close_isolation(self):
        tg = self.targets()
        self.out["targets"] = [dict(t, pos=list(t["pos"])) for t in tg]
        todo = [t for t in tg if t["read"] != "closed"]
        for t in todo:
            if t["read"] is None:
                self.note(f"{t['tag']}: handle state not read; attempting the close (a closed blade cannot be gripped)")
            self.close_valve(t)
        # verification: the closed handles should now read 'closed' (fresh votes)
        ver = {}
        for t in todo:
            self.book.handle_votes.pop(t["tag"], None)
            lh = self.look_handle(t["tag"], need=1, max_looks=3)
            ver[t["tag"]] = None if lh is None else lh["state"]
        self.out["verify"] = ver

    # ------------------------------------------------------------------ 8. decay
    def decay_check(self):
        res = self.plume
        if not res or res.get("exit_S") is None or res.get("s0_S") is None:
            return
        self.mark("decay")
        pf = Platform(self)
        e, s0 = np.asarray(res["exit_S"], float), np.asarray(res["s0_S"], float)
        z = self.ga.roof_z + 1.5
        co = dict(p2_S=e + s0 * PF.G(self.ga, self.ga.roof_z + 2.0), s0_S=s0)
        before = [x for x in res.get("sections", []) if abs(x["z_s"] - z) < 1e-6 and "energy" in x]
        e0 = float(np.median([x["energy"] for x in before])) if before else None
        pred = np.median(np.array([x["c_S"] for x in before]), axis=0) if before else None
        sec = PF.section(pf, self.ga, co, z, n_sweeps=3, pred_S=pred)
        e1 = float(sec.get("energy", 0.0))
        self.out["decay"] = dict(energy_survey=e0, energy_after=e1, ratio=None if not e0 else round(e1 / e0, 3),
                                 found=bool("c_S" in sec), t=round(self.cell.clock(), 1))
        self.mark("decay_done")

    # ------------------------------------------------------------------ 9. back to the cage, seated
    def return_to_cage(self):
        self.mark("return")
        sc = self.sc
        tr = self.cell.truth()
        q = sc.w_to_P(tr.pos)
        if q[2] < FRONT_CLEAR and abs(q[0]) < SKID_HALF_W + 0.6:
            self.mv.go(sc.P_to_w([q[0], q[1], FRONT_CLEAR]), sc.face_yaw, tol=0.2, tol_z=0.15, settle_s=0.1)
        z_start = float(self.start_truth.pos[2])
        z_tr = max(z_start + 2.0, sc.deck_top + 2.6)
        ent = sc.cage_pos + sc.cage_R @ np.array([2.5, 0.0, 0.0])
        seat = self.start_truth.pos.copy()          # where the vehicle sat at power-up (the cage's seat)
        yaw_c = sc.cage_yaw
        tr = self.cell.truth()
        self.mv.go(np.r_[tr.pos[:2], max(tr.pos[2], z_tr)], None, tol=0.3, tol_z=0.2, settle_s=0.1)
        self.mv.go(np.r_[ent[:2], z_tr], yaw_c, tol=0.25, tol_z=0.15, settle_s=0.2)
        z_in = z_start + 0.25
        self.mv.go(np.r_[ent[:2], z_in], yaw_c, tol=0.08, tol_z=0.05, settle_s=0.5, route=False)
        self.mv.go(np.r_[seat[:2], z_in], yaw_c, v=0.10, tol=0.04, tol_z=0.04, settle_s=1.0, route=False)
        self.mark("in_cage")
        # slow descent to just above the deck, then a force ramp with Fz open (no impact: G6 had 400-940 N)
        self.mv.z_rate = 0.04
        self.mv.go(np.r_[seat[:2], z_start + 0.03], yaw_c, v=0.05, tol=0.03, tol_z=0.015, settle_s=1.0, route=False)
        self.mv.z_rate = 0.35
        self.mark("press")
        self.mv.press = dict(F0=2.0, F1=18.0, t0=self.cell.clock(), T=5.0)
        self.mv.gain = 0.6
        self.mv.wait(12.0)
        tr = self.cell.truth()
        self.out["landing"] = dict(seat_err_m=round(float(np.linalg.norm(tr.pos[:2] - seat[:2])), 4),
                                   z_above_start_m=round(float(tr.pos[2] - z_start), 4),
                                   speed_m_s=round(float(np.linalg.norm(tr.vel_w)), 4))
        self.mark("landed")

    # ------------------------------------------------------------------ the sequence
    # P16 extends the program by subclassing CoreProgram and overriding SEQUENCE (or run_core): the dropped object
    # and the hot stab slot in after close_isolation / decay_check, the tide wait replaces wait_window.
    SEQUENCE = ("sensors_first", "leave_cage", "find_structure", "inspect", "locate_leak", "place_marker",
                "wait_window", "close_isolation", "decay_check", "return_to_cage")
    NEEDS_BRANCH = ("wait_window", "close_isolation", "decay_check")

    def run_core(self):
        for name in self.SEQUENCE:
            if name in self.NEEDS_BRANCH and self.branch is None:
                continue
            getattr(self, name)()

    def run(self):
        c0 = time.process_time()
        try:
            self.run_core()
        except MV.EpisodeOver as e:
            self.out["errors"].append(f"episode over: {e}")
        except Exception:  # noqa: BLE001
            import traceback
            self.out["errors"].append(traceback.format_exc()[-3000:])
        try:
            self.mv.stop()
            self.cell.clear_ticks()
            if not self.cell.done:
                self.cell.call("finish")
        except Exception as e:  # noqa: BLE001
            self.out["errors"].append(f"finish: {e!r}")
        self.out.update(t_end=round(self.cell.clock(), 2), prog_cpu_s=round(time.process_time() - c0, 1),
                        wall_s=round(WALL0() - self.w0, 1), branch_chosen=self.branch,
                        calls=dict(sorted(self.cell.calls.items())), call_order=list(self.cell.order),
                        refused=self.cell.refused, tick_errors=list(self.cell.tick_errors),
                        mover=dict(self.mv.stats), unsafe_legs=list(self.keep.unsafe_legs)[:20],
                        frames=dict(self.sens.n_frames), pings=int(self.sens.n_pings),
                        reads=len(self.book.reads))
        return self.out
