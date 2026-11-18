"""The UR10e trims the vehicle: the privileged reference program, part I (the R1 segment). It weighs the base vehicle
on the positioner's three load cells, solves the trim by the PUBLIC route, sets the ballast blocks on their keel-up
sites in the rule's order and turns the vehicle deck up. Parts II and III (the guards, the vertical thrusters, the
enclosure, the whole episode) build on `Assembler` without changing this file (they subclass it).

Privileged state is used for the units' POSES only (DESIGN v1.2 section 3.2): where each block lies in its pocket
(`cell.body_T`), the block in the hand as it is held, the site's live target (`cell.live_target`: the vehicle as it is
now, its mounting offset on the carrier included). The program never reads the coupling (no capture / lock feedback)
nor any hidden quantity of the trim. The TRIM goes the public route: the work order's water -> the density chart
(`chart_density`: the function the chart is drawn from, an exact reading; a reading error can be passed in), the
averaged load-cell readings + the drawing's cell stations -> the base vehicle's mass and CG (`PublicTrim.weigh`, the
keel-up y flip), the printed trim sheet + the cell manual's hole map -> every legal layout, the maximin choice
(`PublicTrim.solve`). It never reads `tests/trim_table.json` and never imports the grader's trim function.

Sources (copied, then rewritten for this cell; none is imported):
  QASM tests/arm/assembly_quad/reference.py sha256 b3df274a136089bdcae3884c07549342532b5d4af5bfb0cb18b4efce0d1bb31d
      (the Assembler's shape: the work label, the contact monitor, plan-or-via moves, the step records)
      QASM = the quadruped full-assembly task of the benchmark's main tree (clone of 2026-11-13)
  -dev/proto/ballast/bal_prog.py sha256 3ee28c15a22fec3a41d01291d50febeb4cfd5656dc422f1b927721f93f6c288c
      (the fingertip pinch 6 mm above the bearing face aimed with the gripper model's grasp centre, the block + 14 mm
      pre-grasp opening, the +4 mm per side release, the swept path check of every segment, the transit waypoint)
  -dev/proto/ballast/probe_r1.py sha256 3b44c8c325238929d338801e18d688646e7cb336ad992207ef2e93f3b416cad3 and
  -dev/proto/ballast/order_enum.py sha256 eb4e892c9cba4e305d2682055e4ca7dbf43a15e1db3e56cc7d5eacad3e40622a
      (the R1 flow: 1 s settle, 1 000 samples, weigh, solve, the rule order: centre-line blocks first on their y faces,
      then the off-centre blocks on their x faces)
  -dev/proto/common/motion_g0.py sha256 0064bf3f284cdce98f366f20035413551431a0f281fb52c24a62f65f609a7acb through
  -dev/evidence/cell/harness/cellrun.py sha256 00eb2a16e223061453bd3f0a9ade8d1e7ddf23754a540860f6df57165b24e49b
      (follow_smooth: rest-to-rest trapezoids on straight joint segments, the tcp speed bounded on 21 samples)
  -dev/trim/v12/public_route.py sha256 4a52c9a9e9b9c3f87c6ee3e1d0969bcb43c13b5f243f0050ec69d196df23e88e
      (`PublicRoute`, embedded below as `PublicTrim` with its algorithm unchanged; the dev check
      -dev/run_reference_r1.py --selftest compares the two on random inputs)

G1's lessons, kept as rules of this program (record G1 sections 0, 5):
  * aim with the gripper model's grasp centre (`gripper_wide.pad_offset`), never with the tcp: at the ballast openings
    they are 0.57-0.59 mm apart, enough to push a block across its pocket's 0.5 mm play;
  * every posture of a move is on ONE IK family (`Assembler.ik`: elbow up, the wrist turning the tool down, the
    upper arm leaning towards the target), the aliases nearest the previous posture; a nearest-branch IK alone once swung the shoulder to
    the other branch over the ballast box and swept the forearm through a side plate (H1 552 N);
  * every segment is checked before it runs (`SweepPlanner`, a kinematic sweep on a copy of the cell: the links 10 mm,
    the fingers 2 mm and the held block 6 mm off everything else); a blocked straight segment goes through a high
    waypoint, the transit waypoint, or RRT-Connect.
Carrying (ruling M8; the cell manual's carrying rule): a held block moves with the tcp at most CARRY_TCP_MAX_M_S and
every joint at most CARRY_JOINT_ACC_MAX; the plan runs below them (CARRY_TCP_PLAN_M_S, CARRY_JOINT_ACC_PLAN: the
computed-torque loop overshoots a commanded acceleration step by about 14 %). The monitor measures both on the arm's
actual motion every tick (`Assembler.carry_log`).

The R1 flow (`Assembler.r1`, `program_r1`):
  weigh    the arm at the ready posture (clear of the vehicle and the rotation column), WEIGH_SETTLE_S, then
           WEIGH_SAMPLES samples of get_state()["fixture"]["load_N"] (one per 10 ms tick) -> mean -> PublicTrim.weigh
  solve    density = chart_density(work order) (+ the caller's reading error), PublicTrim.solve(rho, weighed) (maximin)
  order    the centre-line sites (C*) in hole-map order gripped across the block's y faces, then the off-centre sites in
           hole-map order gripped across its x faces (G1 order_enum: every one of the 1 760 legal layouts is feasible
           this way, the fingers >= 13.6 mm from every block already placed and from the side plates)
  block    i-th site <- the block of the i-th pocket. pick: the hand's yaw (0 / 180 deg) nearest the arm's posture of
           the two that pass the checks; a planned move to 50 mm above the grasp (fingers at the block + 14 mm), a
           straight line down, close (40 N), lift 50 mm; carry: a planned move (the carrying limits) to 50 mm above the
           site's release pose (the block's 180 deg turn on its site chosen for the shorter move), the block's pose in
           the hand re-read, a straight line down to HOVER_M above home, hold HOLD_BEFORE_RELEASE_S, open by 4 mm per
           side, wait, a straight line back up 60 mm
  turn     the ready posture (the column interlock), PRE_TURN_WAIT_S (the last block locks about 0.6 s after its
           release), fixture_index("deck_up"), wait for the turn to end.
Every straight line runs in joint mode on a trapezoid, one IK posture per tick (no Cartesian lag at the ends).
"""
import itertools
import json
import math
import os
import time

import numpy as np

from .coupling import _inv, quat_wxyz, rotation_angle

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---- carrying (ruling M8) and the plan's values under it
CARRY_TCP_MAX_M_S = 0.25
CARRY_JOINT_ACC_MAX = 8.0
CARRY_TCP_PLAN_M_S = 0.20
CARRY_JOINT_ACC_PLAN = 6.0
EMPTY_TCP_M_S = 0.8            # empty-hand moves are not limited by the rule; these keep them smooth
EMPTY_JOINT_ACC = 8.0
JOINT_SPEED = 1.5              # the controller's per-joint rate (rad/s)
LINE_ACC = 0.3                 # m/s^2, the straight lines' trapezoids
# ---- the ballast motions (DESIGN v1.2 section 1.2 "the three grasps": ballast; G0 trial b; G1)
Z_TRAVEL_M = 0.78              # the high waypoints' tcp height (the keel-up vehicle tops out at z 0.61 m)
PRE_M = 0.05                   # the straight approach above a grasp / a release pose
RETREAT_M = 0.06
V_APPROACH_EMPTY = 0.04
V_APPROACH_HELD = 0.03
V_LIFT = 0.06
V_RETREAT = 0.06
BLADE_ABOVE_M = 0.006          # the fingertips' end above the block's bearing face at the grasp
PRE_OPEN_M = 0.014             # the pre-grasp opening: the block's width + this
RELEASE_OPEN_M = 0.008         # the release opening: the block's width + this (4 mm per side)
GRIP_FORCE_N = 40.0
GRIP_SETTLE_S = 0.8
HOLD_BEFORE_RELEASE_S = 0.3    # the capture's rest gate is 10 ticks (0.1 s)
RELEASE_WAIT_S = 0.3
HOVER_M = 0.0003               # the block let go this far above home along the insertion (it drops onto its seat)
HELD_TOL_M, HELD_TOL_RAD = 0.001, math.radians(2.0)   # the block must stay this still in the hand while it is lifted
TRANSIT_XYZ = (0.45, 0.0, 0.90)
# ---- the weighing and the turn
WEIGH_SETTLE_S = 1.0
WEIGH_SAMPLES = 1000
PRE_TURN_WAIT_S = 1.0
# ---- the path checks (SweepPlanner)
LINK_CLEAR_M = 0.010
FINGER_CLEAR_M = 0.002
HELD_CLEAR_M = 0.006
SWEEP_STEP_RAD = 0.004
HELD_PHASES = ("lift", "carry", "approach", "hold")   # the block in the hand, the arm moving or holding it there
PUBLIC_FILES = ("trim_sheet_printed.json", "cell_public.json")


def _T(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def trapezoid(dist, a_max, v_max, dt):
    """positions s(t) every dt of a rest-to-rest trapezoid over `dist` (>= 2 samples, the last = dist)"""
    if dist <= 0:
        return np.array([0.0])
    ta = v_max / a_max
    if a_max * ta * ta >= dist:
        ta = math.sqrt(dist / a_max)
        v_max = a_max * ta
        tc = 0.0
    else:
        tc = (dist - a_max * ta * ta) / v_max
    T = 2 * ta + tc
    t = np.arange(0.0, T + dt, dt)
    s = np.where(t < ta, 0.5 * a_max * t ** 2,
                 np.where(t < ta + tc, 0.5 * a_max * ta ** 2 + v_max * (t - ta),
                          dist - 0.5 * a_max * np.maximum(T - t, 0.0) ** 2))
    s[-1] = dist
    return np.clip(s, 0.0, dist)


# ====================================================================================================== public material
def public_dir():
    """where the printed trim sheet's columns and the cell manual's trim facts are: $ROV_PUBLIC_TRIM_DIR, else <tests>/"""
    return os.environ.get("ROV_PUBLIC_TRIM_DIR") or TESTS_DIR


def chart_density(T_C, SP):
    """the density chart read exactly at the work order's water (kg/m^3): the chart is drawn from this function"""
    from .trim_eval import rho_teos10
    return float(rho_teos10(float(T_C), float(SP)))


class PublicTrim:
    """`PublicRoute` of -dev/trim/v12/public_route.py (sha256 in the module doc), unchanged: what a careful agent computes
    from the printed trim sheet and the cell manual. Reads only
      trim_sheet_printed.json   per row the printed columns (mass in air and one of volume / mass in water / net
                                buoyancy at the printed reference density), CG, CB, state; the header (reference
                                density, g, p_T, the acceptance windows, the per-unit deviation range)
      cell_public.json          the load cells' stations (deck_up projection), the index convention, the hole map, the
                                blocked holes, the block's footprint and the conflict rule
    weigh(loads, index) -> (base mass kg, CG x mm, CG y mm); solve(rho, base) -> {sites, n, dB, theta, phi, margin}.
    Choice rule 'maximin': maximise min((A - |theta|)/(A/2), (A - |phi|)/(A/2), (dB - lo)/0.5, (hi - dB)/0.5); ties
    to the first layout in the canonical order (weights ascending, then combinations in the hole map's order)."""

    MARGIN_DB = (0.7, 2.5)

    def __init__(self, sheet=None, cell=None):
        sh = self._load(sheet, PUBLIC_FILES[0])
        ce = self._load(cell, PUBLIC_FILES[1])
        self.g = float(sh['g'])
        self.rho_ref = float(sh['reference_density_kg_m3'])
        self.pT = np.array(sh['p_T_mm'], float)
        acc = sh['acceptance']
        self.db_win = (float(acc['net_buoyancy_N'][0]), float(acc['net_buoyancy_N'][1]))
        self.ang = float(acc['pitch_abs_deg'])
        if float(acc['roll_abs_deg']) != self.ang:
            raise ValueError('the margin rule assumes one angle limit for pitch and roll')
        self.dev_range = sh['per_unit_deviation']
        self.rows = []
        self.unit = None
        for r in sh['rows']:
            m_g = float(r['mass_in_air_g'])
            if r.get('displacement_cm3') is not None:
                V_cm3 = float(r['displacement_cm3'])
            elif r.get('mass_in_water_g') is not None:
                V_cm3 = (m_g - float(r['mass_in_water_g'])) / (self.rho_ref / 1000.0)
            elif r.get('net_buoyancy_g') is not None:
                V_cm3 = (m_g + float(r['net_buoyancy_g'])) / (self.rho_ref / 1000.0)
            else:
                raise ValueError(r)
            if r['cg_mm'][0] == 'hole':
                self.unit = (m_g / 1000.0, V_cm3 * 1e-6, float(r['cg_mm'][2]), float(r['cb_mm'][2]))
                continue
            self.rows.append((r['state'], m_g / 1000.0, V_cm3 * 1e-6, np.array(r['cg_mm'], float), np.array(r['cb_mm'], float)))
        if self.unit is None:
            raise ValueError('no ballast row on the sheet')
        base = [x for x in self.rows if x[0] == 'upstream']
        self.base_m = sum(x[1] for x in base)
        self.base_cg = sum(x[1] * x[3] for x in base) / self.base_m
        self.cells_deck_up = np.array([ce['load_cells_B_mm_deck_up'][k] for k in ('A', 'B', 'C')], float)
        self.holes = ce['holes_mm']
        self.names = list(self.holes)
        self.XY = np.array([self.holes[n] for n in self.names], float)
        self.blocked = tuple(ce['blocked'])
        self.lim = (float(ce['conflict_mm']['dx_below']), float(ce['conflict_mm']['dy_below']))
        self._spaces = {}

    @staticmethod
    def _load(x, fname):
        if isinstance(x, dict):
            return x
        with open(x or os.path.join(public_dir(), fname)) as f:
            return json.load(f)

    def cells_B(self, index):
        P = self.cells_deck_up.copy()
        if index == 'keel_up':
            P[:, 1] *= -1.0
        elif index != 'deck_up':
            raise ValueError(index)
        return P

    def weigh(self, loads_N, index='keel_up', assume_index=None):
        F = np.asarray(loads_N, float)
        P = self.cells_B(assume_index or index)
        s = F.sum()
        return float(s / self.g), float((F * P[:, 0]).sum() / s), float((F * P[:, 1]).sum() / s)

    def space(self, blocked=None):
        key = tuple(sorted(self.blocked if blocked is None else blocked))
        if key not in self._spaces:
            ok = [i for i in range(len(self.names)) if self.names[i] not in key]
            L = []
            for k in range(len(ok) + 1):
                for c in itertools.combinations(ok, k):
                    if all(not (abs(self.XY[a, 0] - self.XY[b, 0]) < self.lim[0] and abs(self.XY[a, 1] - self.XY[b, 1]) < self.lim[1])
                           for a, b in itertools.combinations(c, 2)):
                        L.append(c)
            SM = np.zeros((len(L), len(self.names)), bool)
            for i, c in enumerate(L):
                SM[i, list(c)] = True
            self._spaces[key] = (L, SM.sum(1).astype(float), SM @ self.XY[:, 0], SM @ self.XY[:, 1])
        return self._spaces[key]

    def layout_names(self, blocked=None):
        L = self.space(blocked)[0]
        return [tuple(self.names[k] for k in c) for c in L]

    def predict(self, rho, base, include_next=True, unit_volume=True, blocked=None):
        m_b, x_b, y_b = base
        L, NN, SX, SY = self.space(blocked)
        M = m_b
        V = 0.0
        mG = m_b * np.array([x_b, y_b, self.base_cg[2]])
        vB = np.zeros(3)
        for st, m, v, cg, cb in self.rows:
            if st == 'upstream':
                V += v
                vB += v * cb
                continue
            if st == 'next station' and not include_next:
                continue
            M += m
            V += v
            mG += m * cg
            vB += v * cb
        um, uv, uzg, uzb = self.unit
        if not unit_volume:
            uv = 0.0
        g = self.g
        W = (M + NN * um) * g
        B = rho * (V + NN * uv) * g
        dB = B - W
        k = rho * g
        dx = k * (vB[0] + uv * SX) - g * (mG[0] + um * SX) - dB * self.pT[0]
        dy = k * (vB[1] + uv * SY) - g * (mG[1] + um * SY) - dB * self.pT[1]
        dz = k * (vB[2] + uv * NN * uzb) - g * (mG[2] + um * NN * uzg) - dB * self.pT[2]
        return L, dB, np.degrees(np.arctan2(-dx, dz)), np.degrees(np.arctan2(dy, dz))

    def margin(self, dB, th, ph):
        a = self.ang
        lo, hi = self.db_win
        return np.minimum.reduce([(a - np.abs(th)) / (a / 2), (a - np.abs(ph)) / (a / 2), (dB - lo) / 0.5, (hi - dB) / 0.5])

    def choose_index(self, dB, th, ph, rule='maximin'):
        if rule == 'maximin':
            return int(np.argmax(self.margin(dB, th, ph)))
        if rule == 'dB_first':
            ok = (dB >= self.MARGIN_DB[0]) & (dB <= self.MARGIN_DB[1])
            if not ok.any():
                ok = (dB >= self.db_win[0]) & (dB <= self.db_win[1])
            if ok.any():
                return int(np.argmin(np.where(ok, np.maximum(np.abs(th), np.abs(ph)), np.inf)))
            return int(np.argmin(np.maximum(self.db_win[0] - dB, 0.0) + np.maximum(dB - self.db_win[1], 0.0)))
        raise ValueError(rule)

    def base_from(self, base, use_x=True, use_y=True):
        if isinstance(base, str) and base == 'table':
            base = (self.base_m, self.base_cg[0], self.base_cg[1])
        m_b, x_b, y_b = base
        return (float(m_b), float(x_b if use_x else self.base_cg[0]), float(y_b if use_y else self.base_cg[1]))

    def solve(self, rho, base='table', use_x=True, use_y=True, include_next=True, unit_volume=True, blocked=None, rule='maximin'):
        b = self.base_from(base, use_x, use_y)
        L, dB, th, ph = self.predict(rho, b, include_next, unit_volume, blocked)
        i = self.choose_index(dB, th, ph, rule)
        mg = self.margin(dB[i:i + 1], th[i:i + 1], ph[i:i + 1])[0]
        return dict(sites=[self.names[k] for k in L[i]], n=len(L[i]), dB=float(dB[i]), theta=float(th[i]), phi=float(ph[i]),
                    margin=float(mg), rule=rule, base=b)


def ballast_order(sites):
    """[(site, grip)]: the centre-line sites first, gripped across the block's y faces, then the off-centre ones across
    its x faces; each group in the hole map's order (as `sites` lists them)"""
    cen = [s for s in sites if s.startswith("C")]
    off = [s for s in sites if not s.startswith("C")]
    return [(s, "y") for s in cen] + [(s, "x") for s in off]


# ====================================================================================================== the path checks
def _planner_class():
    from .motion import Planner

    class SweepPlanner(Planner):
        """`motion.Planner` (a kinematic copy of the cell: straight-segment sweeps, RRT-Connect) with one addition:
        `held_allow`, bodies the HELD unit may come near (its own carrier on the last millimetres to its seat, its
        station while it is lifted out of its pocket); every other pair keeps its clearance"""

        def sync(self, held=None, group=(), allow=(), poses=None, width=None, q=None, held_allow=()):
            super(SweepPlanner, self).sync(held=held, group=group, allow=allow, poses=poses, width=width, q=q)
            self.held_allow = set(self.m.body(n).id for n in held_allow)

        def hits(self, q, tol_m=0.0005, finger_clear=None, link_clear=None, held_clear=None):
            mj, m, d = self.mj, self.m, self.d
            self.n_checks += 1
            d.qpos[self.qadr] = q
            mj.mj_kinematics(m, d)
            if self.held is not None:
                self._place_held()
                mj.mj_kinematics(m, d)
            mj.mj_collision(m, d)
            fc = self.finger_clear if finger_clear is None else finger_clear
            lc = self.clear if link_clear is None else link_clear
            hc = self.held_clear if held_clear is None else held_clear
            ha = getattr(self, "held_allow", set())
            out = []
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                r1, r2 = b1 in self.robot, b2 in self.robot
                h1, h2 = b1 in self.held_bodies, b2 in self.held_bodies
                if r1 and r2:
                    if (b1 in self.gripper and b2 in self.gripper) or c.dist >= 0.0:
                        continue
                    out.append((m.body(b1).name, m.body(b2).name, float(c.dist)))
                    continue
                if (r1 or h1) == (r2 or h2):
                    continue
                mine, other = (b1, b2) if (r1 or h1) else (b2, b1)
                if other in self.allow:
                    continue
                if mine in self.held_bodies:
                    if other in ha:
                        continue
                    if c.dist < hc - tol_m:
                        out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
                elif mine in self.fingers:
                    if c.dist < fc:
                        out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
                elif c.dist < lc:
                    out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
            return out

        def min_dist(self, qs):
            """the smallest distance of the robot / held unit to anything over the postures (the hits' rules aside)"""
            mj, m, d = self.mj, self.m, self.d
            best, pair = 1.0, None
            for q in qs:
                d.qpos[self.qadr] = q
                mj.mj_kinematics(m, d)
                if self.held is not None:
                    self._place_held()
                    mj.mj_kinematics(m, d)
                mj.mj_collision(m, d)
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    r1, r2 = b1 in self.robot, b2 in self.robot
                    h1, h2 = b1 in self.held_bodies, b2 in self.held_bodies
                    if (r1 or h1) == (r2 or h2):
                        continue
                    mine, other = (b1, b2) if (r1 or h1) else (b2, b1)
                    if other in self.allow or (mine in self.held_bodies and other in getattr(self, "held_allow", set())):
                        continue
                    if float(c.dist) < best:
                        best, pair = float(c.dist), (m.body(mine).name, m.body(other).name)
            return best, pair

    return SweepPlanner


# ====================================================================================================== the assembler
class Assembler(object):
    """the privileged reference program's core on a `cell.RovCell` with the arm (module doc)"""

    def __init__(self, cell, log=print, read_fixture=None, monitor=True, kin=None):
        from .ur_ik import URKin
        self.cell, self.m, self.d, self.ct, self.sm = cell, cell.m, cell.d, cell.ctrl, cell.sm
        self.kit, self.rb, self.log = cell.kit, cell.robot, log
        self.tick_s = float(cell.tick_s)
        self.pl = _planner_class()(cell, clearance_m=LINK_CLEAR_M, held_clearance_m=HELD_CLEAR_M,
                                   finger_clear_m=FINGER_CLEAR_M)
        self.kin = kin if kin is not None else URKin(model=self.pl.m, prefix=self.rb["prefix"], tcp_site=self.rb["tcp_site"])
        from .bench_plan import READY_Q
        self.q_ready = np.array(cell.cfg["robot_home_q"] if cell.cfg.get("robot_home_q") is not None else READY_Q, float)
        self.read_fixture = read_fixture or (lambda: cell.P.fixture_state(None))
        self.work = {"phase": "start", "unit": None, "site": None, "held": False}
        self.tcp_sid = self.m.site(self.rb["tcp_site"]).id
        self.checkpoints = []           # (name, t, cell.state_hash()) after the weighing, every block, the turn
        self._blocks = {}
        self._init_monitor()
        if monitor:
            cell.hooks.append(self._monitor)

    # ------------------------------------------------------------------ state
    def tcp_T(self):
        p, R = self.ct.tcp_actual()
        return _T(p, R)

    def tcp_cmd_T(self):
        """the tcp pose of the COMMANDED configuration: every line starts from it and every held unit's pose in the
        hand is taken against it (the arm's tracking offset under the load is then part of that pose: commanding
        the target with it puts the unit where it is aimed; a line started from the actual pose instead steps the
        command by the offset in one tick, a joint-acceleration spike of ~18 rad/s^2 for a 2 mrad offset)"""
        return self.kin.fk_T(self.ct.q_cmd.copy())

    def unit_T(self, u):
        return self.cell.body_T(u)

    def t(self):
        return float(self.d.time)

    def set_phase(self, phase, held=None, **kw):
        self.work["phase"] = phase
        if held is not None:
            self.work["held"] = bool(held)
        self.work.update(kw)
        self.cell.label["text"] = "%s %s %s" % (self.work.get("site") or "", self.work.get("unit") or "", phase)

    def checkpoint(self, name):
        self.checkpoints.append((name, round(self.t(), 3), self.cell.state_hash()))

    # ------------------------------------------------------------------ the monitor
    def _init_monitor(self):
        c = self.cell
        self.carry = {"ticks": 0, "tcp_max_m_s": 0.0, "acc_max_rad_s2": 0.0, "acc_cmd_max_rad_s2": 0.0,
                      "tcp_cmd_max_m_s": 0.0, "where_tcp": None, "where_acc": None, "where_acc_cmd": None, "over": []}
        self.carry_series = []          # (t, phase id, tcp speed, joint acc max, cmd acc max) of the held ticks
        self._prev = None
        self.contacts = []              # unplanned contacts: robot links on anything, pads on anything but the unit
        self.unit_force = {}            # per step: the moving unit's largest per-body contact sum, its 20-tick minimum
        self._uf_hist = []
        self._robot_link = (c.gclass == 1)
        self._pad = (c.gclass == 2)

    def _monitor(self):
        c, d, ct = self.cell, self.d, self.ct
        w = self.work
        t = float(d.time)
        p = d.site_xpos[self.tcp_sid].copy()
        qv = ct.dq()
        qc = ct.q_cmd.copy()
        prev = self._prev
        if prev is not None and w["held"] and w["phase"] in HELD_PHASES:
            p0, qv0, qc0, qc00 = prev
            v = float(np.linalg.norm(p - p0)) / self.tick_s
            a = float(np.max(np.abs(qv - qv0))) / self.tick_s
            ac = float(np.max(np.abs(qc - 2 * qc0 + qc00))) / (self.tick_s ** 2) if qc00 is not None else 0.0
            vc = float(np.linalg.norm(self.kin.fk_T(qc)[:3, 3] - self.kin.fk_T(qc0)[:3, 3])) / self.tick_s
            C = self.carry
            C["ticks"] += 1
            if v > C["tcp_max_m_s"]:
                C["tcp_max_m_s"], C["where_tcp"] = v, (round(t, 3), w["phase"], w.get("site"))
            if a > C["acc_max_rad_s2"]:
                C["acc_max_rad_s2"], C["where_acc"] = a, (round(t, 3), w["phase"], w.get("site"), int(np.argmax(np.abs(qv - qv0))))
            if ac > C["acc_cmd_max_rad_s2"]:
                C["acc_cmd_max_rad_s2"], C["where_acc_cmd"] = ac, (round(t, 3), w["phase"], w.get("site"))
            C["tcp_cmd_max_m_s"] = max(C["tcp_cmd_max_m_s"], vc)
            if v > CARRY_TCP_MAX_M_S or a > CARRY_JOINT_ACC_MAX:
                C["over"].append((round(t, 3), w["phase"], w.get("site"), round(v, 4), round(a, 3)))
            st = self._blocks.get(w.get("site"))
            if st is not None:
                st["carry_tcp_max_m_s"] = max(st.get("carry_tcp_max_m_s", 0.0), v)
                st["carry_acc_max_rad_s2"] = max(st.get("carry_acc_max_rad_s2", 0.0), a)
                st["carry_acc_cmd_max_rad_s2"] = max(st.get("carry_acc_cmd_max_rad_s2", 0.0), ac)
            self.carry_series.append((t, HELD_PHASES.index(w["phase"]), v, a, ac))
        self._prev = (p, qv, qc, prev[2] if prev is not None else None)
        # contacts
        nc = int(d.ncon)
        uf = 0.0
        if nc:
            g1, g2, fn = c._contact_normals(nc)
            if len(fn):
                k1, k2 = c.gclass[g1], c.gclass[g2]
                rob1, rob2 = (k1 == 1) | (k1 == 2), (k2 == 1) | (k2 == 2)
                link = ((k1 == 1) & ~rob2) | ((k2 == 1) & ~rob1)
                u = w.get("unit")
                ui = c.unit_index.get(u, -1) if u else -1
                pad_other = ((k1 == 2) & ~rob2 & (c.gunit_idx[g2] != ui)) | ((k2 == 2) & ~rob1 & (c.gunit_idx[g1] != ui))
                for i in np.nonzero(link | pad_other)[0]:
                    if fn[i] <= 0.0:
                        continue
                    a_, b_ = (g1[i], g2[i]) if (k1[i] in (1, 2)) else (g2[i], g1[i])
                    key = (w["phase"], w.get("site"), self.m.body(int(c.geom_body[a_])).name, self.m.body(int(c.geom_body[b_])).name)
                    if not self.contacts or self.contacts[-1]["key"] != key:
                        self.contacts.append({"key": key, "t": round(t, 3), "phase": w["phase"], "site": w.get("site"),
                                              "robot": key[2], "other": key[3], "force_N": round(float(fn[i]), 3),
                                              "link": bool(link[i])})
                    else:
                        self.contacts[-1]["force_N"] = max(self.contacts[-1]["force_N"], round(float(fn[i]), 3))
                if ui >= 0:
                    mine1 = (k1 == 3) & (c.gunit_idx[g1] == ui) & ~rob2
                    mine2 = (k2 == 3) & (c.gunit_idx[g2] == ui) & ~rob1
                    sums = {}
                    for i in np.nonzero(mine1 | mine2)[0]:
                        other = int(c.geom_body[g2[i]] if mine1[i] else c.geom_body[g1[i]])
                        sums[other] = sums.get(other, 0.0) + float(fn[i])
                    if sums:
                        uf = max(sums.values())
        st = self._blocks.get(w.get("site"))
        if st is not None:
            self._uf_hist.append(uf)
            if len(self._uf_hist) > 20:
                self._uf_hist.pop(0)
            st["unit_contact_max_N"] = max(st.get("unit_contact_max_N", 0.0), uf)
            if len(self._uf_hist) == 20:
                st["unit_contact_sustained_N"] = max(st.get("unit_contact_sustained_N", 0.0), min(self._uf_hist))

    def carry_log(self):
        C = dict(self.carry)
        C["limits"] = {"tcp_m_s": CARRY_TCP_MAX_M_S, "joint_acc_rad_s2": CARRY_JOINT_ACC_MAX}
        C["plan"] = {"tcp_m_s": CARRY_TCP_PLAN_M_S, "joint_acc_rad_s2": CARRY_JOINT_ACC_PLAN}
        C["ok"] = bool(C["tcp_max_m_s"] <= CARRY_TCP_MAX_M_S and C["acc_max_rad_s2"] <= CARRY_JOINT_ACC_MAX)
        C["over"] = C["over"][:20]
        return C

    # ------------------------------------------------------------------ IK on one family
    FAMILY_GAP_RAD = 0.1

    @staticmethod
    def in_family(q):
        """the coarse filter of the working family: elbow up (q3 > 0), the wrist turning the tool down (sin q5 < 0)"""
        return q[2] > 0.05 and math.sin(q[4]) < -0.3

    def ik(self, T, q_ref, margin=0.02):
        """the family's posture for the tcp pose T, its aliases nearest q_ref; None if there is none. Of the two
        elbow-up, wrist-down solutions (one per shoulder branch) the family takes the one whose upper arm leans
        towards the target (the larger q2; the other reaches over the shoulder), and only when the two are at least
        FAMILY_GAP_RAD apart (near the shoulder singularity the branches meet: no posture there)"""
        q_ref = np.asarray(q_ref, float)
        cands = []
        for q in self.kin.ik(T, q6_ref=float(q_ref[5])):
            qn = self.kin.nearest(q, q_ref, margin)
            if qn is not None and self.in_family(qn):
                cands.append(qn)
        if not cands:
            return None
        cands.sort(key=lambda x: -x[1])
        if len(cands) > 1 and cands[0][1] - cands[1][1] < self.FAMILY_GAP_RAD:
            return None
        return cands[0]

    def line_qs(self, T0, T1, q0, s_list):
        """the family's postures along the straight tcp line T0 -> T1 at the fractions s_list (orientation slerped),
        each near the previous; None if a posture is missing or a joint jumps"""
        from scipy.spatial.transform import Rotation, Slerp
        sl = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
        q = np.asarray(q0, float)
        out = []
        for s in s_list:
            p = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
            R = sl([s]).as_matrix()[0]
            q2 = self.ik(_T(p, R), q, margin=0.0)
            if q2 is None or np.max(np.abs(q2 - q)) > 0.3:
                return None
            out.append(q2)
            q = q2
        return out

    # ------------------------------------------------------------------ checks
    def sync(self, held=None, allow=(), width=None, held_allow=()):
        self.pl.sync(held=held, allow=allow, width=width, held_allow=held_allow)

    def seg_free(self, qa, qb):
        return self.pl.free(qa, qb, step=SWEEP_STEP_RAD)

    def path_free(self, path):
        return all(self.seg_free(a, b) for a, b in zip(path[:-1], path[1:]))

    def plan_path(self, q0, q1, held=None, width=None, allow=(), vias=()):
        """[q0, ..., q1] on the family, every segment swept free: straight; else through each via in turn (postures);
        else RRT-Connect. (path, how) or (None, why)"""
        self.sync(held=held, allow=allow, width=width)
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if not self.pl.ok(q1):
            return None, "goal blocked: %s" % (sorted(set(h[1] for h in self.pl.hits(q1)))[:4],)
        if self.seg_free(q0, q1):
            return [q0, q1], "straight"
        for name, vs in vias:
            if not vs or any(v is None for v in vs):
                continue
            path = [q0] + [np.asarray(v, float) for v in vs] + [q1]
            if self.path_free(path):
                return path, name
        path = self.pl.plan(q0, q1)
        if path is not None:
            return path, "rrt"
        return None, "no path (%s)" % getattr(self.pl, "why", "")

    def path_clearance(self, path, n=24):
        """(smallest distance m, pair) of the robot / held unit to everything along the path (n postures per
        segment), on the planner's current scene; (1.0, None) when nothing comes within the check margins (the links
        LINK_CLEAR_M, the held unit HELD_CLEAR_M)"""
        qs = []
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            qs += [a + (b - a) * (k / float(n)) for k in range(n + 1)]
        return self.pl.min_dist(qs)

    def high_of(self, T, q_ref):
        """the family's posture with the tcp straight above T at Z_TRAVEL_M, the same orientation"""
        Th = T.copy()
        Th[2, 3] = Z_TRAVEL_M
        return self.ik(Th, q_ref)

    def transit_of(self, R, q_ref):
        return self.ik(_T(np.array(TRANSIT_XYZ, float), R), q_ref)

    # ------------------------------------------------------------------ motion
    def tick(self, n=1):
        self.cell.tick(int(n))

    def wait(self, s):
        self.cell.tick(int(round(float(s) / self.tick_s)))

    def wait_arrived(self, tol=0.002, timeout_s=5.0):
        ct = self.ct
        for _ in range(int(round(timeout_s / self.tick_s))):
            self.cell.tick()
            if (np.max(np.abs(ct.q_goal - ct.q_cmd)) < 1e-9 and np.max(np.abs(ct.q() - ct.q_cmd)) < tol
                    and np.max(np.abs(ct.dq())) < 0.02):
                return True
        return False

    def follow_smooth(self, path, a_joint, v_tcp, v_joint=JOINT_SPEED, tol=0.002):
        """each straight joint segment on a rest-to-rest trapezoid: the largest joint <= a_joint, <= v_joint, the tcp
        <= v_tcp (|dp/ds| sampled on 41 points of the segment)"""
        ct = self.ct
        ct.set_mode("joint")
        qa = ct.q_cmd.copy()
        for qb in path[1:]:
            qb = np.asarray(qb, float)
            D = float(np.max(np.abs(qb - qa)))
            if D < 1e-9:
                continue
            u = qb - qa
            ss = np.linspace(0.0, 1.0, 41)
            P = np.array([self.kin.fk_T(qa + s_ * u)[:3, 3] for s_ in ss])
            dpds = float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1)) / (D / 40.0))
            vmax = min(float(v_joint), float(v_tcp) / max(dpds, 1e-9))
            for s_ in trapezoid(D, float(a_joint), vmax, self.tick_s)[1:]:
                ct.command_joint(qa + u * (s_ / D))
                self.cell.tick()
            qa = qb
        return self.wait_arrived(tol=tol)

    def run_line(self, qs):
        """stream a precomputed list of postures, one per tick (joint mode), then settle"""
        ct = self.ct
        ct.set_mode("joint")
        for q in qs:
            ct.command_joint(q)
            self.cell.tick()
        return self.wait_arrived(tol=0.001, timeout_s=1.0)

    def line_plan(self, T1, v_max, a_max=LINE_ACC, T0=None, q0=None):
        """(postures one per tick along the straight tcp line to T1 on a trapezoid, the fractions) or (None, None)"""
        T0 = self.tcp_cmd_T() if T0 is None else T0
        q0 = self.ct.q_cmd.copy() if q0 is None else np.asarray(q0, float)
        dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        ang = rotation_angle(T0[:3, :3].T @ T1[:3, :3])
        L = max(dist, ang * 0.1)                      # 0.1 m per rad: a pure turn runs on the same profile
        s = trapezoid(L, a_max, v_max, self.tick_s)[1:] / max(L, 1e-12) if L > 1e-9 else np.array([1.0])
        qs = self.line_qs(T0, T1, q0, list(s))
        return qs, s

    def line_check(self, qs, held=None, allow=(), width=None, held_allow=()):
        """the line's postures free (every 3rd and the last), a contact the first posture has tolerated while it
        does not get closer"""
        self.sync(held=held, allow=allow, width=width, held_allow=held_allow)
        return self.pl.line_free(list(qs[::3]) + [qs[-1]])

    def set_opening(self, width, tol_m=0.001, timeout_s=2.0):
        """the fingers to an opening (nothing in the hand) and waited for; the arm holds still meanwhile"""
        self.ct.command_width(float(width))
        target = float(self.ct.width_goal)
        for _ in range(int(round(timeout_s / self.tick_s))):
            if abs(float(self.ct.width_cmd) - target) < 1e-9 and abs(float(self.ct.width()) - target) < tol_m:
                break
            self.cell.tick()
        return float(self.ct.width())

    def grip(self, width, force_N=None, settle_s=0.3):
        if force_N is not None:
            self.ct.command_force(force_N)
        self.ct.command_width(float(width))
        self.wait(settle_s)
        return self.ct.width()

    # ------------------------------------------------------------------ the blocks
    def block_frame(self, u):
        """(R_own_in_K, centre in K, extents along the block's own axes) from the kit's collision boxes: the block's
        own sides are turned off its frame K by the vendor CAD's yaw"""
        from .bench_plan import collision_points
        cs = self.kit["bodies"][u]["collision"]
        R = _quat_R(cs[0].get("quat", [1, 0, 0, 0]))
        P, _S = collision_points(self.kit, u)
        Po = P @ R                                    # into the own axes
        lo, hi = Po.min(0), Po.max(0)
        return R, R @ (0.5 * (lo + hi)), hi - lo

    def grasp_pose(self, u, grip, flip, Tb=None):
        """(tcp pose, opening) of the fingertip pinch of block u lying on its bearing face: the tool down, the closing
        axis across the block's `grip` faces ('y': its y sides, 'x': its x sides), turned 180 deg if flip, the
        fingertips' end BLADE_ABOVE_M over the bearing face, the grasp centre (pad_offset) on the block's centre"""
        from . import gripper_wide as GW
        from .bench_plan import collision_points
        Tb = self.unit_T(u) if Tb is None else Tb
        R_own, c_K, ext = self.block_frame(u)
        Rw = Tb[:3, :3] @ R_own
        ax = Rw[:, 1] if grip == "y" else Rw[:, 0]
        w = float(ext[1] if grip == "y" else ext[0])
        yt = np.array([ax[0], ax[1], 0.0])
        yt /= np.linalg.norm(yt)
        if flip:
            yt = -yt
        zt = np.array([0.0, 0.0, -1.0])
        xt = np.cross(yt, zt)
        Rg = np.column_stack([xt, yt, zt])
        P, _S = collision_points(self.kit, u)
        z_face = float((P @ Tb[:3, :3].T + Tb[:3, 3])[:, 2].min())
        c = Tb[:3, :3] @ c_K + Tb[:3, 3]
        po = GW.pad_offset(w)
        z_tcp = z_face + BLADE_ABOVE_M + GW.fingertip_z(w)
        p = np.array([c[0], c[1], z_tcp]) - Rg @ np.array([po[0], po[1], 0.0])
        return _T(p, Rg), w

    def station_of(self, u):
        """the station body a loose unit starts in (the ballast box for a block)"""
        from .layout import KIT_BODIES
        return "ballast_box" if u in KIT_BODIES["blocks"] else None

    def pick_block(self, u, grip):
        """pick block u from where it lies; returns the record (or raises RuntimeError with the reason)"""
        rec = {"unit": u, "grip": grip, "t0_s": round(self.t(), 3)}
        q_now = self.ct.q_cmd.copy()
        T_now = self.tcp_cmd_T()
        best = None
        why = []
        for flip in (False, True):
            Tg, w = self.grasp_pose(u, grip, flip)
            Tpre = Tg.copy()
            Tpre[2, 3] += PRE_M
            q_pre = self.ik(Tpre, q_now)
            if q_pre is None:
                why.append("flip %d: no IK above the grasp" % flip)
                continue
            down, _s = self.line_plan(Tg, V_APPROACH_EMPTY, T0=Tpre, q0=q_pre)
            if down is None:
                why.append("flip %d: approach line IK" % flip)
                continue
            if not self.line_check([q_pre] + down, allow=(u,), width=w + PRE_OPEN_M):
                why.append("flip %d: approach line blocked" % flip)
                continue
            cost = float(np.max(np.abs(q_pre - q_now)))
            if best is None or cost < best[0]:
                best = (cost, flip, Tg, w, Tpre, q_pre, down)
        if best is None:
            raise RuntimeError("no pick for %s (%s)" % (u, "; ".join(why)))
        _c, flip, Tg, w, Tpre, q_pre, down = best
        rec.update({"flip": bool(flip), "width_mm": round(w * 1e3, 3)})
        # to 50 mm above the grasp. The fingers are set to the pre-grasp opening FIRST, where the hand is (clear of
        # everything): their reach along the tool axis changes with the opening (fully open the fingertips are 19 mm
        # shorter than at a block's opening), so a path checked at one opening and run while the fingers move is not
        # the checked path (a first version swept a closing finger onto the enclosure on its rack at 100 N)
        self.set_phase("to-pick", held=False, unit=u)
        width_chk = self.set_opening(w + PRE_OPEN_M)
        rec["opening_before_move_mm"] = round(width_chk * 1e3, 3)
        vias = [("over", [self.high_of(Tpre, q_pre)]),
                ("up", [self.high_of(T_now, q_now)]),
                ("up-over", [self.high_of(T_now, q_now), self.high_of(Tpre, q_pre)]),
                ("transit", [self.high_of(T_now, q_now), self.transit_of(Tpre[:3, :3], q_pre), self.high_of(Tpre, q_pre)])]
        path, how = self.plan_path(q_now, q_pre, width=width_chk, vias=vias)
        if path is None:
            raise RuntimeError("no path to the pick of %s: %s" % (u, how))
        rec["path_to_pick"] = how
        self.follow_smooth(path, EMPTY_JOINT_ACC, EMPTY_TCP_M_S)
        self.wait(0.1)
        # down, close, lift
        self.set_phase("approach-pick")
        self.run_line(down)
        self.set_phase("grip")
        self.grip(0.0, force_N=GRIP_FORCE_N, settle_s=GRIP_SETTLE_S)
        G0 = _inv(self.tcp_T()) @ self.unit_T(u)
        self.set_phase("lift", held=True)
        Tup = self.tcp_cmd_T()
        Tup[2, 3] += PRE_M
        up, _s = self.line_plan(Tup, V_LIFT)
        if up is None:
            raise RuntimeError("no lift line for %s" % u)
        rec["lift_line_free"] = bool(self.line_check([self.ct.q_cmd.copy()] + up, held=(u, G0), width=w,
                                                     held_allow=(self.station_of(u),)))
        self.run_line(up)
        G1 = _inv(self.tcp_T()) @ self.unit_T(u)
        dp = float(np.linalg.norm(G1[:3, 3] - G0[:3, 3]))
        da = rotation_angle(G0[:3, :3].T @ G1[:3, :3])
        rec.update({"t_picked_s": round(self.t(), 3), "in_hand_shift_mm": round(dp * 1e3, 4),
                    "in_hand_turn_deg": round(math.degrees(da), 4), "pad_width_mm": round(self.ct.width() * 1e3, 3)})
        if dp > HELD_TOL_M or da > HELD_TOL_RAD or self.ct.width() < 0.5 * w:
            raise RuntimeError("block %s moved in the hand (%.2f mm, %.2f deg)" % (u, dp * 1e3, math.degrees(da)))
        return rec, w

    def place_block(self, u, site, w, rec):
        """carry the held block u to `site`, set it down, let it go, back out; fills `rec`"""
        q_now = self.ct.q_cmd.copy()
        T_now = self.tcp_cmd_T()
        G = _inv(T_now) @ self.unit_T(u)                 # the block in the (commanded) tool frame, as held
        best = None
        why = []
        for k in (0, 1):
            tg = self.cell.live_target(site, u, k=k)
            ins = tg["insertion"]
            T_home = tg["T"] @ _inv(G)
            T_hov = T_home.copy()
            T_hov[:3, 3] -= ins * HOVER_M
            T_pre = T_home.copy()
            T_pre[:3, 3] -= ins * PRE_M
            q_pre = self.ik(T_pre, q_now)
            if q_pre is None:
                why.append("k %d: no IK above the site" % k)
                continue
            down, _s = self.line_plan(T_hov, V_APPROACH_HELD, T0=T_pre, q0=q_pre)
            if down is None:
                why.append("k %d: approach line IK" % k)
                continue
            if not self.line_check([q_pre] + down, held=(u, G), width=w, held_allow=(self.cell.base,)):
                why.append("k %d: approach line blocked" % k)
                continue
            cost = float(np.max(np.abs(q_pre - q_now)))
            if best is None or cost < best[0]:
                best = (cost, k, T_pre, q_pre)
        if best is None:
            raise RuntimeError("no place for %s at %s (%s)" % (u, site, "; ".join(why)))
        _c, k, T_pre, q_pre = best
        rec["k"] = int(k)
        self.set_phase("carry", held=True, site=site)
        vias = [("over", [self.high_of(T_pre, q_pre)]),
                ("up", [self.high_of(T_now, q_now)]),
                ("up-over", [self.high_of(T_now, q_now), self.high_of(T_pre, q_pre)]),
                ("transit", [self.high_of(T_now, q_now), self.transit_of(T_pre[:3, :3], q_pre), self.high_of(T_pre, q_pre)])]
        path, how = self.plan_path(q_now, q_pre, held=(u, G), width=w, vias=vias)
        if path is None:
            raise RuntimeError("no carry path for %s to %s: %s" % (u, site, how))
        rec["path_carry"] = how
        cl, pair = self.path_clearance(path)
        rec["carry_clearance_mm"] = None if pair is None else round(1e3 * cl, 2)    # None: beyond the check margins
        rec["carry_clearance_pair"] = pair
        self.follow_smooth(path, CARRY_JOINT_ACC_PLAN, CARRY_TCP_PLAN_M_S)
        # the hold as it is now, then down to just above home
        G2 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        tg = self.cell.live_target(site, u, k=k)
        ins = tg["insertion"]
        T_hov = tg["T"] @ _inv(G2)
        T_hov[:3, 3] -= ins * HOVER_M
        down, _s = self.line_plan(T_hov, V_APPROACH_HELD)
        if down is None:
            raise RuntimeError("no approach line for %s at %s" % (u, site))
        rec["carry_creep_mm"] = round(1e3 * float(np.linalg.norm(G2[:3, 3] - G[:3, 3])), 4)
        self.set_phase("approach")
        self.run_line(down)
        self.set_phase("hold")
        self.wait(HOLD_BEFORE_RELEASE_S)
        Tb = self.unit_T(u)
        tg = self.cell.live_target(site, u, k=k)
        rel = _inv(tg["T"]) @ Tb
        rec["at_release"] = {"pos_err_mm": [round(1e3 * float(v), 4) for v in rel[:3, 3]],
                             "ang_err_deg": round(math.degrees(rotation_angle(rel[:3, :3])), 4)}
        self.set_phase("release", held=False)
        self.ct.command_width(w + RELEASE_OPEN_M)
        self.wait(RELEASE_WAIT_S)
        self.set_phase("retreat")
        Tup = self.tcp_cmd_T()
        Tup[:3, 3] -= ins * RETREAT_M
        up, _s = self.line_plan(Tup, V_RETREAT)
        if up is None:
            raise RuntimeError("no retreat line at %s" % site)
        rec["retreat_line_free"] = bool(self.line_check([self.ct.q_cmd.copy()] + up, width=float(self.ct.width())))
        self.run_line(up)
        Tb = self.unit_T(u)
        tg = self.cell.live_target(site, u, k=k)
        rel = _inv(tg["T"]) @ Tb
        rec["after_retreat"] = {"pos_err_mm": [round(1e3 * float(v), 4) for v in rel[:3, 3]],
                                "ang_err_deg": round(math.degrees(rotation_angle(rel[:3, :3])), 4)}
        rec["t1_s"] = round(self.t(), 3)
        rec["block_s"] = round(rec["t1_s"] - rec["t0_s"], 3)
        return rec

    def ballast(self, hole, u, grip):
        """one block: pick u, place it on the site of `hole` (the hole map's name: C1 .. FS2; the slot bal_<hole>);
        the record (placed_by 'ur' or 'nobody' with the reason)"""
        site = hole if hole.startswith("bal_") else "bal_" + hole
        rec = {"hole": hole, "site": site, "unit": u, "grip": grip, "placed_by": "nobody"}
        self._blocks[site] = rec
        self._uf_hist = []
        self.work.update({"site": site, "unit": u})
        try:
            r, w = self.pick_block(u, grip)
            rec.update(r)
            self.place_block(u, site, w, rec)
            rec["placed_by"] = "ur"
        except RuntimeError as e:
            rec["why"] = str(e)
            self.log("  %s <- %s: %s" % (site, u, e))
            if self.work.get("held"):
                self.let_go()
        self.set_phase("idle", held=False, unit=None, site=None)
        self.log("  %-4s <- %s (%s grip, k %s) %s  %.1f s sim" % (site, u, grip, rec.get("k"), rec["placed_by"],
                                                               rec.get("block_s") or 0.0))
        return rec

    def let_go(self):
        """the hand opens where it is and backs out upwards (a block the arm could not place stays where it falls)"""
        self.set_phase("let-go", held=False)
        self.ct.command_width(self.ct.max_width)
        self.wait(0.3)
        Tup = self.tcp_cmd_T()
        Tup[2, 3] += 0.06
        up, _s = self.line_plan(Tup, V_RETREAT)
        if up is not None:
            self.run_line(up)

    def go_ready(self):
        self.set_phase("to-ready", held=False)
        q_now = self.ct.q_cmd.copy()
        vias = [("up", [self.high_of(self.tcp_cmd_T(), q_now)])]
        path, how = self.plan_path(q_now, self.q_ready, width=float(self.ct.width()), vias=vias)
        if path is None:
            path = [q_now, self.q_ready]
            how = "unchecked"
        self.follow_smooth(path, EMPTY_JOINT_ACC, EMPTY_TCP_M_S)
        return how

    # ------------------------------------------------------------------ weigh, turn
    def weigh(self, public, n=WEIGH_SAMPLES, settle_s=WEIGH_SETTLE_S):
        """the arm at the ready posture; settle; n samples of the fixture's readings, one per tick -> mean ->
        (mass, x_G, y_G) by the public route (the index from the reading)"""
        if np.max(np.abs(self.ct.q() - self.q_ready)) > 0.01:
            self.go_ready()
        self.set_phase("weigh", held=False)
        self.wait(settle_s)
        F = np.zeros(3)
        idx, moving = None, False
        t0 = self.t()
        for _ in range(int(n)):
            self.cell.tick()
            fs = self.read_fixture()
            F += np.asarray(fs["load_N"], float)
            idx = fs["index"]
            moving = moving or bool(fs.get("moving"))
        F /= float(n)
        m, x, y = public.weigh(F, idx)
        return {"loads_mean_N": [float(v) for v in F], "index": idx, "moving": moving, "samples": int(n),
                "t_s": [round(t0, 3), round(self.t(), 3)], "mass_kg": m, "cg_x_mm": x, "cg_y_mm": y}

    def turn(self, target="deck_up"):
        """the ready posture, PRE_TURN_WAIT_S, the index request, the turn waited out; the answer and the times"""
        how = self.go_ready()
        in_col = bool(self.cell.P.robot_in_column()[0])
        self.set_phase("pre-turn", held=False)
        self.wait(PRE_TURN_WAIT_S)
        t_req = self.t()
        ans = self.cell.fixture_index(target)
        self.set_phase("turn")
        P = self.cell.P
        lim = float(P.cfg["turn_s"]) + float(P.cfg["settle_s"]) + 2.0
        for _ in range(int(round(lim / self.tick_s))):
            if not P.moving():
                break
            self.cell.tick()
        self.set_phase("turned")
        return {"answer": ans, "t_request_s": round(t_req, 3), "t_done_s": round(self.t(), 3), "index": P.index(),
                "path_to_ready": how, "robot_in_column_at_request": in_col}

    # ------------------------------------------------------------------ R1
    def r1(self, work_order, rho=None, rho_read_err=0.0, sites=None, blocks=None, weigh=True, turn=True, public=None):
        """the R1 segment (module doc). work_order: describe()["work_order"]; rho: the density to use (None: the chart
        read at the work order, + rho_read_err); sites: a layout to place instead of the solved one (the campaign);
        blocks: the units to use in turn (default: the blocks in pocket order). Returns the report."""
        c = self.cell
        public = public or PublicTrim()
        rep = {"work_order": work_order, "t0_s": round(self.t(), 3)}
        t_wall = time.time()
        wo = (work_order or {}).get("delivery_site") or {}
        if rho is None and wo:
            rho = chart_density(wo["water_temperature_C"], wo["salinity_psu"]) + float(rho_read_err)
        rep["rho_used_kg_m3"] = rho
        if weigh:
            rep["weigh"] = self.weigh(public)
            base = (rep["weigh"]["mass_kg"], rep["weigh"]["cg_x_mm"], rep["weigh"]["cg_y_mm"])
            self.checkpoint("weighed")
        else:
            base = "table"
        if sites is None:
            sol = public.solve(rho, base=base)
            rep["solution"] = sol
            sites = sol["sites"]
        rep["sites"] = list(sites)
        order = ballast_order(sites)
        rep["order"] = order
        if blocks is None:
            blocks = [p["unit"] for p in c.layout["pockets"]]
        recs = []
        for (site, grip), u in zip(order, blocks):
            if c.done:
                break
            recs.append(self.ballast(site, u, grip))
            self.checkpoint("block %s" % site)
        rep["blocks"] = recs
        rep["t_ballast_done_s"] = round(self.t(), 3)
        if turn and not c.done:
            rep["turn"] = self.turn("deck_up")
            self.checkpoint("turned")
        rep["t1_s"] = round(self.t(), 3)
        rep["wall_s"] = round(time.time() - t_wall, 1)
        rep["carry"] = self.carry_log()
        rep["unplanned_contacts"] = [{k: v for k, v in x.items() if k != "key"} for x in self.contacts]
        rep["checkpoints"] = list(self.checkpoints)
        return rep


def program_r1(cell, work_order, read_fixture=None, rho=None, rho_read_err=0.0, sites=None, log=print, turn=True,
               weigh=True):
    """run the R1 segment of the privileged reference program on `cell` (a RovCell with the arm); the report and the
    Assembler (parts II / III go on with it)"""
    asm = Assembler(cell, log=log, read_fixture=read_fixture)
    asm.go_ready()
    rep = asm.r1(work_order, rho=rho, rho_read_err=rho_read_err, sites=sites, weigh=weigh, turn=turn)
    return rep, asm
