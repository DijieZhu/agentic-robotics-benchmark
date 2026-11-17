"""Jobs of the motion-controller task (`motion-jobs/1`): generated per cell at acceptance, every feasible job checked on
the TRUE unit in the TRUE cell and on the documented nominal arm in the DESCRIBED cell, and planned by the privileged
planner from the previous job's end (so the oracle can do every one).

Per cell (job counts in COUNTS):
  shelf     reach into the compartments (horizontal-ish approach along the arm's plane, a jaw axis), reach over the
            fixture top (approach down or tilted), infeasible: out-of-plane approach, out of reach, inside the post
  bins      a part is clamped in the gripper for the whole cell (its size and seat known, its mass 30-150 g and centre
            unknown); place poses inside the bins and on the fixture; infeasible as above
  dispense  the needle (in its holder) is clamped; paths over the board (needle tip 3 mm above it, needle within 8 deg
            of the board normal, 15-40 mm/s); a purge reach over the cup; infeasible: out of reach, inside the clamp
            block or the cup, an out-of-plane direction at a purge pose

A job's public form (`public(job, cell)`) is in the cell frame; its private part (targets in the base frame, the
oracle's joint solutions and plan) stays with the grader.
"""

import math

import numpy as np

from . import kin as KN
from . import metrics as M
from . import plant as P
from . import privileged as PV
from . import unit as U

JOBS_FORMAT = "motion-jobs/1"
COUNTS = {"shelf": {"reach": 11, "infeasible": 3}, "bins": {"reach": 11, "infeasible": 3},
          "dispense": {"path": 8, "reach": 2, "infeasible": 3}}
MAIN_SHARE = 0.8                 # shelf / bins: the share of reach jobs in the compartments / bins (the rest on the fixture)
GEN_CLEAR_M = 0.012              # a feasible target's clearance on both views (the specification asks 10 mm)
PLAN_CLEAR_M = 0.012             # the privileged plan's clearance between jobs (2 mm over the specification: the arm's
                                 # elastic lag while it moves)
APPROACH_TOL_DEG = 5.0
JAW_TOL_DEG = 5.0
AXIS_TOL_DEG = 8.0
GAP_MM, GAP_TOL_MM = 3.0, 1.5
NEEDLE = {"length_m": 0.045, "radius_m": 0.0012, "mass_kg": 0.020}    # the needle and its holder block
SETTLE_S = 0.8
REF_SPEED_FRAC = 0.5             # the time target: the planned joint path at half the servo's rated speed + SETTLE_S
V_RATED = 6.49 * 0.925
PATH_SAMPLE_M = 0.004
W_REG_PATH = 0.002               # a path's continuation IK keeps the joints continuous (the needle may use its cone)


def az_of(p):
    return math.atan2(-p[0], p[1])


def radial(p):
    a = az_of(p)
    return np.array([-math.sin(a), math.cos(a), 0.0])


def tilt_down(v_h, beta):
    """A unit vector beta below the horizontal direction v_h (in the vertical plane through v_h)."""
    return math.cos(beta) * np.asarray(v_h, float) + math.sin(beta) * np.array([0.0, 0.0, -1.0])


class View(object):
    """One view of a cell for job checks: a unit model (true or nominal) in a cell (true or described), with the cell's
    clamped tool compiled in (unit.unit_mjcf: the same collision geometry as the plant)."""

    def __init__(self, design_xml, spec, cell, tool=None, design=None):
        import mujoco
        import os
        self.cell = cell
        self.design = design if design is not None else U.load_design(os.path.dirname(os.path.abspath(design_xml)))
        self.tool = tool
        self.model = mujoco.MjModel.from_xml_string(U.unit_mjcf(design_xml, spec, cell, probe=True, tool=tool, design=self.design))
        self.spec = spec
        self.geom_obj = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g["name"]): g["object"] for g in cell.geoms()}
        self.geom_obj[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")] = "table"
        off = (0.0, 0.0, 0.0)
        if tool and tool["kind"] == "needle":
            seat = U.tool_seat(self.design, tool)
            off = (seat["seat_x"], 0.0, float(tool["params"]["length_m"]))
        # rigid: a job's joint solutions are LINK angles (the pose itself); the static deflection of the series
        # elasticity only maps them to the servo outputs, which is the controller's business (privileged.py for the oracle)
        self.kin = KN.Kin(self.model, self.spec, tool_off=off, deflect=False)
        self.kin.set_cell_objects(self.geom_obj)
        self.kin_tcp = KN.Kin(self.model, self.spec, tool_len=0.0, deflect=False)
        self.kin_tcp.set_cell_objects(self.geom_obj)

    def set_exempt(self, exempt):
        """exempt = [{"who": "robot" | "part" | "needle", "what": object or primitive name}]."""
        pairs = P.exempt_pairs(self.model, self.geom_obj, exempt)
        self.kin.exempt = pairs
        self.kin_tcp.exempt = pairs
        self.exempt_desc = list(exempt)


def pitch_set(view, pos, rng, grid=None):
    """{pitch_deg: q} of the approach pitches (below the horizontal, in the arm's plane through pos) the view's arm can
    take at pos (position < 0.2 mm, direction < 0.3 deg; rigid, no obstacle check): a coarse 10 deg scan from a few
    starts, then the neighbours of every hit at 5 deg with warm starts."""
    found = {}
    with view.kin.rigid():
        for b in (grid if grid is not None else range(-30, 91, 10)):
            app = tilt_down(radial(pos), math.radians(b))
            qq, e = view.kin.ik(pos, app, None, starts=2, iters=40, rng=rng)
            if e["pos_m"] < 2e-4 and e["app_deg"] < 0.3:
                found[b] = qq
        for b in sorted(found):
            for nb in (b - 5, b + 5):
                if nb in found or nb < -30 or nb > 90:
                    continue
                app = tilt_down(radial(pos), math.radians(nb))
                qq, e = view.kin.ik(pos, app, None, q0=found[b], starts=0, iters=40, rng=rng)
                if e["pos_m"] < 2e-4 and e["app_deg"] < 0.3:
                    found[nb] = qq
    return found


_PITCH_CACHE = {}


def choose_pitch(views, pos, rng, prefer="low", span=20, margin=5):
    """(pitch, q seed): a pitch the documented arm (views[-1], the nominal view) can take with `margin` deg to the
    edge of its set, drawn from the `span` deg at the preferred end (low = most horizontal, high = steepest); the true
    view is checked later by _ik_valid. (None, None) when there is none. Sets are cached on a 1 cm grid."""
    v = views[-1]
    key = (id(v), tuple(np.round(np.asarray(pos) / 0.01).astype(int)))
    if key not in _PITCH_CACHE:
        _PITCH_CACHE[key] = pitch_set(v, pos, rng)
    found = _PITCH_CACHE[key]
    common = sorted(found.keys())
    inner = [b for b in common if (b - margin in common or b - margin < -30) and (b + margin in common or b + margin > 90)]
    if not inner:
        return None, None
    lo, hi = inner[0], inner[-1]
    cand = [b for b in inner if b <= lo + span] if prefer == "low" else [b for b in inner if b >= hi - span]
    b = cand[int(rng.integers(len(cand)))]
    return float(b) + float(rng.uniform(-2.0, 2.0)), found[b]


def realise(view, pos, app, q_seed, roll_choice, rng):
    """An exactly reachable target of the documented arm near (pos, app): IK without a jaw axis, then the wrist roll
    set to put the jaw axis exactly horizontal (roll_choice 0) or exactly in the arm's plane (1): the best roll of a
    73-step grid, refined by bisection where the residual changes sign (an infeasible job's analytic jaw axis is exact
    too, so the jaw's numerical form tells nothing); returns (pos', app', jaw', q) from FK (the TCP is off the roll
    axis, so the roll moves it a little), or None."""
    with view.kin.rigid():
        q, e = view.kin.ik(pos, app, None, q0=q_seed, starts=2, rng=rng)
        if e["pos_m"] > 2e-4 or e["app_deg"] > 0.3:
            return None

        def resid(roll):
            q2 = q.copy()
            q2[4] = roll
            _, R2 = view.kin.fk(q2)
            x = R2[:, 0]
            return float(x[2]) if roll_choice == 0 else float(np.dot(x, np.cross(R2[:, 2], np.array([0.0, 0.0, 1.0]))))
        lo, hi = float(view.kin.lo[4]), float(view.kin.hi[4])
        grid = np.linspace(lo, hi, 73)
        r0 = float(min(grid, key=lambda r: abs(resid(r))))
        step = float(grid[1] - grid[0])
        for a, b in ((max(lo, r0 - step), r0), (r0, min(hi, r0 + step))):
            fa, fb = resid(a), resid(b)
            if fa * fb > 0:
                continue
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = resid(m)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            r0 = 0.5 * (a + b)
            break
        q2 = q.copy()
        q2[4] = r0
        p2, R2 = view.kin.fk(q2)
    return p2, R2[:, 2].copy(), R2[:, 0].copy(), q2


def _ik_valid(view, pos, app, jaw, clear, rng, q0=None, tries=4, app_tol_deg=0.3, exempt_min=None):
    best = None
    for t in range(tries):
        q, e = view.kin.ik(pos, app, jaw, q0=q0 if t == 0 else None, starts=6, rng=rng)
        if e["pos_m"] > 2e-4 or e["app_deg"] > app_tol_deg + (1.2 if jaw is not None else 0.0) or (jaw is not None and e.get("jaw_deg", 0) > 1.2):
            continue
        c, selfc, pair = view.kin.clearance(q)
        emin = view.kin.EXEMPT_MIN_M if exempt_min is None else exempt_min
        if selfc or c < clear or view.kin.last_exempt < emin:
            if best is None:
                best = (None, {"why": "self" if selfc else ("clearance" if c < clear else "support"), "c": c, "pair": pair})
            continue
        if view.kin.conditioning(q) < COND_GEN:
            if best is None:
                best = (None, {"why": "conditioning"})
            continue
        return q, {"clear": c, **e}
    return None, (best[1] if best else {"why": "ik"})


PRE_ENTRY_M = 0.07


PLAN_INFO = {"exit": False, "slow": []}   # of the last plan_to: whether it starts with a straight exit, and which of its
                                           # segments run slow (the straight exit and entry segments)
TRANSIT_EXEMPT_MIN_M = 0.008     # in transit (the RRT part of a plan) even the supports stay 8 mm away; only the straight
                                 # exit and entry segments may come within kin.EXEMPT_MIN_M (3 mm) of them
LINE_STEP_M = 0.014              # a straight Cartesian exit / entry is a chain of IK waypoints this far apart


def cartesian_line(k, q_from, p_from, p_to, app, rng, clear, app_tol_deg=1.0, stop_when_free=None, roll=None):
    """Joint waypoints along the straight Cartesian line p_from -> p_to with the approach `app` held within app_tol_deg
    (IK continuation from q_from, every LINE_STEP_M), each sub-edge checked at `clear` (the exempt pairs at
    kin.EXEMPT_MIN_M). With stop_when_free = d, the line ends at the first waypoint whose clearance to every cell geom,
    the supports included, is >= d (an exit: back out until free). Returns the waypoint list after q_from, or None."""
    p_from = np.asarray(p_from, float)
    p_to = np.asarray(p_to, float)
    n = max(1, int(math.ceil(np.linalg.norm(p_to - p_from) / LINE_STEP_M)))
    q = np.asarray(q_from, float).copy()
    out = []
    old = k.cond_min
    k.cond_min = 0.0
    try:
        for i in range(1, n + 1):
            target = p_from + (p_to - p_from) * i / n
            qn, e = k.ik(target, app, None, q0=q, starts=0, rng=rng, q_ref=q)
            qn[4] = q_from[4] if roll is None else roll            # the wrist roll held (the TCP is on the roll axis)
            qn[5] = q_from[5]
            if e["pos_m"] >= 5e-4 or e["app_deg"] >= app_tol_deg:
                return None
            if np.max(np.abs(qn[:5] - q[:5])) > math.radians(12.0) or not k.edge_ok(q, qn, clear):
                return None
            out.append(qn)
            q = qn
            if stop_when_free is not None:
                c, selfc, _ = k.clearance(qn)
                if not selfc and c >= stop_when_free and k.last_exempt >= stop_when_free:
                    return out
    finally:
        k.cond_min = old
    return None if stop_when_free is not None else out


ENTRY_DISTANCES_M = (0.07, 0.056, 0.042, 0.028)


def exit_ok(view, q_at, app, rng, clear=None, exempt_min=None):
    """Whether the arm can leave the pose q_at (a job's end) by a straight exit (back along `app`, else up): a target
    with no way out would block every later job of the cell."""
    clear = PLAN_CLEAR_M if clear is None else clear
    k = view.kin
    old = k.EXEMPT_MIN_M
    try:
        with k.rigid():
            p_now, _ = k.fk(np.asarray(q_at, float))
            for dirn in (-np.asarray(app, float), np.array([0.0, 0.0, 1.0])):
                if exempt_min is not None:
                    k.EXEMPT_MIN_M = exempt_min
                line = cartesian_line(k, np.asarray(q_at, float), p_now, p_now + dirn * 0.09, app, rng, min(clear, GEN_CLEAR_M - 0.004),
                                      app_tol_deg=3.0, stop_when_free=GEN_CLEAR_M)
                k.EXEMPT_MIN_M = old
                if line is not None and k.valid(line[-1], clear):
                    return True
    finally:
        k.EXEMPT_MIN_M = old
    return False


def plan_to(view, q_prev, q_goal, pos, app, rng, clear=None, exit_from=None, entry_exempt_min=None):
    """The privileged plan to a target: when the arm starts at a previous target (`exit_from` = (pos, app) of it), a
    straight Cartesian retreat PRE_ENTRY_M back along that approach first (the way it came in; else straight up; required:
    no plan otherwise), then RRT-Connect (supports kept TRANSIT_EXEMPT_MIN_M away) to a pre-entry pose PRE_ENTRY_M back
    along the new approach (the farthest of ENTRY_DISTANCES_M that works) and the straight Cartesian line into the target;
    None when any of these is impossible. PLAN_INFO["slow"] marks the straight segments."""
    clear = PLAN_CLEAR_M if clear is None else clear
    q_prev = np.asarray(q_prev, float)
    k = view.kin
    PLAN_INFO["exit"] = False
    PLAN_INFO["slow"] = []
    head, head_slow = [q_prev], []
    with k.rigid():
        if exit_from is not None:
            ppos, papp = exit_from[0], exit_from[1]
            emin_exit = exit_from[2] if len(exit_from) > 2 else None
            p_now, _ = k.fk(q_prev)
            for dirn in (-np.asarray(papp, float), np.array([0.0, 0.0, 1.0])):
                if emin_exit is not None:
                    k.EXEMPT_MIN_M = emin_exit
                line = cartesian_line(k, q_prev, p_now, p_now + dirn * 0.09, papp, rng, min(clear, GEN_CLEAR_M - 0.004),
                                      app_tol_deg=3.0, stop_when_free=GEN_CLEAR_M)
                k.EXEMPT_MIN_M = 0.003
                if line is not None and k.valid(line[-1], clear):
                    head = [q_prev] + line
                    head_slow = [True] * len(line)
                    PLAN_INFO["exit"] = True
                    break
            if not PLAN_INFO["exit"]:
                return None                  # the arm sits at a support: no plan that does not leave it straight
        start = head[-1]
        old_emin = k.EXEMPT_MIN_M
        try:
            for dist in ENTRY_DISTANCES_M:
                pre = np.asarray(pos, float) - np.asarray(app, float) * dist
                qp, e = k.ik(pre, app, None, q0=q_goal, starts=0, rng=rng, q_ref=q_goal)
                if not (e["pos_m"] < 5e-4 and e["app_deg"] < 1.0):
                    continue
                qp[5] = q_prev[5]
                k.EXEMPT_MIN_M = TRANSIT_EXEMPT_MIN_M
                ok_pre = k.valid(qp, clear)
                k.EXEMPT_MIN_M = old_emin
                if not ok_pre:
                    continue
                if entry_exempt_min is not None:
                    k.EXEMPT_MIN_M = entry_exempt_min      # a needle path starts 3 mm above its board
                qp[4] = float(np.asarray(q_goal)[4])
                tail = cartesian_line(k, qp, pre, np.asarray(pos, float), app, rng, clear, roll=float(np.asarray(q_goal)[4]))
                k.EXEMPT_MIN_M = old_emin
                if tail is None or np.max(np.abs(tail[-1][:4] - np.asarray(q_goal)[:4])) > math.radians(1.0):
                    continue
                tail[-1] = np.asarray(q_goal, float).copy()
                tail[-1][5] = q_prev[5]
                k.EXEMPT_MIN_M = TRANSIT_EXEMPT_MIN_M
                mid = k.plan(start, qp, clear, rng=rng, max_nodes=1200, step=math.radians(10.0))
                k.EXEMPT_MIN_M = old_emin
                if mid is not None:
                    plan = head[:-1] + mid + tail
                    PLAN_INFO["slow"] = head_slow + [False] * (len(mid) - 1) + [True] * len(tail)
                    return plan
                break
            return None                      # every arrival is a straight, slow entry line
        finally:
            k.EXEMPT_MIN_M = old_emin


COND_GEN = 1.25 * M.COND_MIN      # a job's targets / path samples and the privileged plan stay above this conditioning
TIME_MARGIN = 1.2
FIRST_JOB_ATTEMPTS = 30           # a layout whose first job cannot be made in this many candidates is given up


def _time_target(path_q, slow=None):
    """The job's time target: the planned joint path as synchronised point-to-point moves (transit at 0.55, the straight
    exit and entry segments at 0.2 of the slowest unit's rated speed) x TIME_MARGIN, plus SETTLE_S for latency and
    settling."""
    if path_q is None or len(path_q) < 2:
        return SETTLE_S
    return round(TIME_MARGIN * PV.ptp_duration(path_q, slow=slow) + SETTLE_S, 2)


# ------------------------------------------------------------------------------------------------- candidates
def _cand_compartment(cell, rng, views, edge=False):
    """A target in a compartment (edge: beside a side wall, where the gripper may not fit: a blocked-target
    candidate)."""
    sh = cell.object("shelf")
    reg = sh["regions"][int(rng.integers(len(sh["regions"])))]
    cx, cy, cz = reg["centre"]
    hx, hy, hz = reg["half"]
    fx = rng.uniform(0.55, 0.95) * (1 if rng.random() < 0.5 else -1) if edge else rng.uniform(-0.6, 0.6)
    loc = np.array([cx + fx * hx, cy - hy + rng.uniform(0.3, 0.75) * 2 * hy, cz + rng.uniform(-0.15, 0.15) * hz])
    pos = sh["pose"].apply(loc)
    b, qs = choose_pitch(views, pos, rng, "low", span=15)
    if b is None:
        return None
    app = tilt_down(radial(pos), math.radians(b))
    r_ = realise(views[-1], pos, app, qs, 1 if rng.random() < 0.8 else 0, rng)
    if r_ is None:
        return None
    pos, app, jaw, qs = r_
    return {"kind": "compartment", "pos": pos, "app": app, "jaw": jaw, "exempt": [{"who": "robot", "what": reg["board"]}], "region": reg,
            "q_seed": qs}


def _cand_top(cell, rng, views, obj_name, h_lo=0.012, h_hi=0.03, part_drop=0.0):
    o = cell.object(obj_name)
    reg = o["regions"][0]
    cx, cy, cz = reg["centre"]
    hx, hy = reg["half"][0], reg["half"][1]
    loc = np.array([cx + rng.uniform(-0.5, 0.5) * hx, cy + rng.uniform(-0.5, 0.5) * hy, cz + part_drop + rng.uniform(h_lo, h_hi)])
    pos = o["pose"].apply(loc)
    b, qs = choose_pitch(views, pos, rng, "high", span=20)
    if b is None:
        return None
    app = tilt_down(radial(pos), math.radians(b))
    r_ = realise(views[-1], pos, app, qs, int(rng.integers(0, 2)), rng)
    if r_ is None:
        return None
    pos, app, jaw, qs = r_
    return {"kind": "top", "pos": pos, "app": app, "jaw": jaw, "exempt": [{"who": "robot", "what": obj_name}], "region": reg, "q_seed": qs}


def _cand_bin(cell, rng, views, part, edge=False):
    """A placement in a bin (edge: by a wall, where the part or the gripper may meet it: a blocked-target
    candidate)."""
    b = cell.object("bin{}".format(int(rng.integers(1, 3))))
    reg = b["regions"][0]
    cx, cy, cz = reg["centre"]
    hx, hy = reg["half"][0], reg["half"][1]
    reach_below = part["seat_z"] + part["half"][2]      # TCP -> the part's far face along the approach
    f = [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)]            # the part's fit decides (contact, clearance)
    if edge:
        k = int(rng.integers(2))
        f[k] = rng.uniform(0.6, 1.0) * (1 if rng.random() < 0.5 else -1)
    loc = np.array([cx + f[0] * hx, cy + f[1] * hy, reg["floor_z"] + reach_below + rng.uniform(0.006, 0.014)])
    pos = b["pose"].apply(loc)
    bb, qs = choose_pitch(views, pos, rng, "high", span=15)
    if bb is None:
        return None
    app = tilt_down(radial(pos), math.radians(bb))
    r_ = realise(views[-1], pos, app, qs, int(rng.integers(0, 2)), rng)
    if r_ is None:
        return None
    pos, app, jaw, qs = r_
    return {"kind": "bin", "pos": pos, "app": app, "jaw": jaw, "exempt": [{"who": "part", "what": b["name"]}], "region": reg, "q_seed": qs}


def _cand_cup(cell, rng, views):
    """The needle's purge position over the cup: the tip 4-12 mm above the rim's centre area, the steepest needle
    direction the documented arm can take there."""
    o = cell.object("cup")
    rad, hh = o["geoms"][0]["size"]
    pos = o["pose"].apply([rng.uniform(-0.3, 0.3) * rad, rng.uniform(-0.3, 0.3) * rad, 2 * hh + rng.uniform(0.004, 0.012)])
    b, qs = choose_pitch(views, pos, rng, "high", span=10)
    if b is None:
        return None
    app = tilt_down(radial(pos), math.radians(b))
    return {"kind": "purge", "pos": pos, "app": app, "jaw": None, "exempt": [{"who": "needle", "what": "cup"}], "q_seed": qs}


def _jaw_like(app, rng):
    """A jaw axis perpendicular to the approach, as the feasible jobs carry one (horizontal or in the arm's plane)."""
    app = np.asarray(app, float)
    h = np.cross(app, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(h) < 1e-6:
        h = np.cross(app, np.array([1.0, 0.0, 0.0]))
    h /= np.linalg.norm(h)
    return h if rng.random() < 0.5 else np.cross(app, h)


# the infeasible kinds of a cell: shelf and bins always one blocked target (reachable, but the arm, the gripper or the part
# would be in the cell's geometry: only a collision check tells) and two of the others; dispense at the cup
INF_KINDS = {"shelf": ("blocked", ["out_of_plane", "inside", "blocked"]),
             "bins": ("blocked", ["out_of_plane", "inside", "blocked"]),
             "dispense": (None, ["out_of_plane", "inside", "out_of_plane"])}
# inside an object a feasible job names as its support (a taught point below the fixture's top, inside the cup), so the
# exempt name is the object's own; out of reach is not drawn (no support of a cell lies beyond the reach: such a target
# would float in free space)
INSIDE_OBJECTS = {"shelf": ("fixture",), "bins": ("fixture",), "dispense": ("cup",)}
KIND_FALLBACK = {"blocked": "out_of_plane", "inside": "out_of_plane", "out_of_plane": "inside"}
BLOCK_DEPTH_M = 0.004            # a blocked target: every solution found is at least this deep in the cell or the arm
BLOCK_TILT_DEG = 4.5             # ... also with the approach tilted this much (inside the 5 deg cone) four ways


def _pitch_deg(app):
    """The approach's angle below the horizontal (deg)."""
    return math.degrees(math.asin(max(-1.0, min(1.0, -float(np.asarray(app, float)[2])))))


def _infeasible(cell, rng, views, kind, feasible_examples, who):
    """An infeasible reach whose public fields look like a feasible job's: the exempt list and the approach's pitch of
    one of the cell's feasible reaches (inside: the object's own name, which its feasible jobs use), the jaw axis of the
    same form (_jaw_like); infeasible by physics only: the approach out of the arm's plane, or the target inside an
    object (blocked targets are made in generate). Checked on both views by _infeasible_ok."""
    exs = list(feasible_examples or [])
    if not exs:
        return None
    ex = exs[int(rng.integers(len(exs)))]
    b_ex = _pitch_deg(ex["app"])
    if kind == "out_of_plane":
        # the example's approach with its horizontal part turned about the vertical: the pitch stays the example's and
        # the approach leaves the arm's plane by g (g <= 90 deg - pitch is all a steep approach allows)
        steep = [e for e in exs if _pitch_deg(e["app"]) <= 75.0]
        if not steep:
            return None
        ex = steep[int(rng.integers(len(steep)))]
        pos = np.asarray(ex["pos"], float) + rng.uniform(-0.010, 0.010, 3) * np.array([1.0, 1.0, 0.5])
        a = np.asarray(ex["app"], float)
        b = math.asin(max(-1.0, min(1.0, -a[2])))
        hdir = np.array([a[0], a[1], 0.0]) / max(1e-9, math.hypot(a[0], a[1]))
        g_hi = min(30.0, math.degrees(math.asin(min(1.0, math.cos(b) * 0.98))))
        g = math.radians(rng.uniform(12.5, max(12.5, g_hi)))
        phi = math.asin(min(1.0, math.sin(g) / max(1e-9, math.cos(b)))) * (1 if rng.random() < 0.5 else -1)
        h2 = math.cos(phi) * hdir + math.sin(phi) * np.cross(np.array([0.0, 0.0, 1.0]), hdir)
        app = math.cos(b) * h2 - math.sin(b) * np.array([0.0, 0.0, 1.0])
        return {"kind": "out_of_plane", "pos": pos, "app": app / np.linalg.norm(app), "exempt": [dict(x) for x in ex["exempt"]]}
    if kind == "inside":
        obs = [o for o in cell.objects if o["name"] in INSIDE_OBJECTS[cell.kind]]
        if not obs:
            return None
        o = obs[int(rng.integers(len(obs)))]
        g = o["geoms"][0]
        if g["type"] == "cylinder":
            rad, hh = g["size"]
            ang = rng.uniform(0, 2 * math.pi)
            rr = 0.6 * rad * math.sqrt(rng.random())
            loc = np.array([rr * math.cos(ang), rr * math.sin(ang), rng.uniform(0.2, 0.9) * 2 * hh])
        else:
            hx, hy, hz = g["size"]
            loc = np.array([rng.uniform(-0.6, 0.6) * hx, rng.uniform(-0.6, 0.6) * hy, g["pos"][2] + rng.uniform(-0.6, 0.6) * hz])
        pos = o["pose"].apply(loc)
        app = tilt_down(radial(pos), math.radians(b_ex + rng.uniform(-3.0, 3.0)))
        own = [dict(x) for e in exs for x in e["exempt"] if x["what"] == o["name"]][:1]
        if not own:                                       # no feasible job of the cell names it: not this kind here
            return None
        return {"kind": "inside", "pos": pos, "app": app, "exempt": own}
    return None


def _infeasible_ok(views, cand, rng):
    """True when neither view can take the pose within the cones and the clearance (so refusing is right): the IK's
    best position error >= 10 mm, or its direction error >= the cone + 7 deg, or the pose in contact."""
    for v in views:
        q, e = v.kin.ik(cand["pos"], cand["app"], None, starts=8, rng=rng)
        if e["pos_m"] < 0.010 and e["app_deg"] < APPROACH_TOL_DEG + 7.0:
            c, selfc, _ = v.kin.clearance(q)
            if not selfc and c > -0.002 and v.kin.last_exempt > -0.002:
                return False
    return True


def _blocked_ok(views, cand, rng):
    """True when the target is reachable but blocked on both views: the IK reaches it (within the cones) and every
    solution found, for the approach and for it tilted BLOCK_TILT_DEG four ways, puts the arm, the gripper, the part or
    the needle at least BLOCK_DEPTH_M into the cell (exempt support included: contact always counts) or into itself."""
    app0 = np.asarray(cand["app"], float)
    u = np.cross(app0, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(app0, np.array([1.0, 0.0, 0.0]))
    u /= np.linalg.norm(u)
    w = np.cross(app0, u)
    t = math.radians(BLOCK_TILT_DEG)
    apps = [app0] + [math.cos(t) * app0 + math.sin(t) * d for d in (u, -u, w, -w)]
    for v in views:
        v.set_exempt(cand["exempt"])
        n_sol = 0
        for a in apps:
            for _ in range(3):
                q, e = v.kin.ik(cand["pos"], a, cand.get("jaw"), starts=4, rng=rng)
                if e["pos_m"] > 0.002 or e["app_deg"] > 1.0 or (cand.get("jaw") is not None and e.get("jaw_deg", 0.0) > JAW_TOL_DEG):
                    continue
                n_sol += 1
                c, selfc, _ = v.kin.clearance(q)
                if not selfc and c > -BLOCK_DEPTH_M and v.kin.last_exempt > -BLOCK_DEPTH_M:
                    return False
        if n_sol == 0:
            return False
    return True


# -------------------------------------------------------------------------------------------------------- paths
def _path_points(cell, rng):
    bd = cell.object("board")
    reg = bd["regions"][0]
    R = bd["pose"].R @ np.asarray(reg["R"], float)
    c = bd["pose"].apply(reg["centre"])
    n = bd["pose"].rot(reg["normal"])
    ex, ey = R[:, 0], R[:, 1]
    hx, hy = reg["half"][0], reg["half"][1]
    speed = float(rng.uniform(0.015, 0.040))
    if rng.random() < 0.55:
        L = float(rng.uniform(0.04, 0.10))
        ang = rng.uniform(0, math.pi)
        d = math.cos(ang) * ex + math.sin(ang) * ey
        mid = c + rng.uniform(-0.3, 0.3) * hx * ex + rng.uniform(-0.3, 0.3) * hy * ey
        T = L / speed
        ts = np.arange(0.0, T + 1e-9, 0.01)
        pts = [mid + d * (-L / 2 + speed * t) for t in ts]
        shape = {"shape": "line", "length_mm": round(L * 1e3, 1)}
    else:
        rad = float(rng.uniform(0.015, 0.035))
        sweep = math.radians(rng.uniform(90, 270))
        a0 = rng.uniform(0, 2 * math.pi)
        cen = c + rng.uniform(-0.25, 0.25) * hx * ex + rng.uniform(-0.25, 0.25) * hy * ey
        T = rad * sweep / speed
        ts = np.arange(0.0, T + 1e-9, 0.01)
        pts = [cen + rad * (math.cos(a0 + speed / rad * t) * ex + math.sin(a0 + speed / rad * t) * ey) for t in ts]
        shape = {"shape": "arc", "radius_mm": round(rad * 1e3, 1), "sweep_deg": round(math.degrees(sweep), 1)}
    pts = np.array(pts)
    # inside the board's region?
    loc = (pts - c) @ np.vstack([ex, ey]).T
    if np.any(np.abs(loc[:, 0]) > hx) or np.any(np.abs(loc[:, 1]) > hy):
        return None
    tips = pts + n * GAP_MM * 1e-3
    return {"t": ts, "tips": tips, "normal": n, "speed": speed, **shape}


def _path_check(view, path, rng, q0=None, cone_deg=AXIS_TOL_DEG - 2.0, clear=GEN_CLEAR_M, exempt_min=0.002):
    """IK along the path every ~PATH_SAMPLE_M (warm-started); the needle axis is the reachable direction closest to
    -normal (must be within cone_deg); clearance with the needle exempt against the board; joint continuity and rate."""
    tips, ts = path["tips"], path["t"]
    stride = max(1, int(round(PATH_SAMPLE_M / (path["speed"] * 0.01))))
    idx = list(range(0, len(ts), stride))
    if idx[-1] != len(ts) - 1:
        idx.append(len(ts) - 1)
    app = -np.asarray(path["normal"], float)
    qs, worst_axis, minc = [], 0.0, 1.0
    q = q0
    for k, i in enumerate(idx):
        if q is None:
            qk, info = _ik_valid(view, tips[i], app, None, clear, rng, app_tol_deg=cone_deg, exempt_min=exempt_min)
            if qk is None:
                return None, {"why": "start", **info}
            q = qk
        else:
            q, e = view.kin.ik(tips[i], app, None, q0=q, starts=0, rng=rng, q_ref=q, w_reg=W_REG_PATH)
            if e["pos_m"] > 2e-4 or e["app_deg"] > cone_deg:
                return None, {"why": "axis", "at": k, **e}
        if qs and np.max(np.abs(q[:5] - qs[-1][:5])) > math.radians(6.0):
            return None, {"why": "jump", "at": k}
        c, selfc, pair = view.kin.clearance(q)
        if selfc or c < clear:
            return None, {"why": "clearance", "at": k, "c": c, "pair": pair}
        if view.kin.last_exempt < exempt_min:
            return None, {"why": "support", "at": k, "d": view.kin.last_exempt}
        if view.kin.conditioning(q) < COND_GEN:
            return None, {"why": "conditioning", "at": k}
        _, e = q, view.kin.errors(q, tips[i], app)
        worst_axis = max(worst_axis, e["app_deg"])
        minc = min(minc, c)
        qs.append(q.copy())
    dt = (ts[idx[1]] - ts[idx[0]]) if len(idx) > 1 else 1.0
    rates = [np.max(np.abs(b[:5] - a[:5])) / dt for a, b in zip(qs[:-1], qs[1:])]
    if rates and max(rates) > 0.6 * V_RATED:
        return None, {"why": "rate", "max_rad_s": max(rates)}
    return qs, {"worst_axis_deg": worst_axis, "min_clear": minc}


# ----------------------------------------------------------------------------------------------------- generator
def part_spec(rng):
    half = [float(rng.uniform(0.010, 0.016)), float(rng.uniform(0.010, 0.016)), float(rng.uniform(0.012, 0.020))]
    seat_z = half[2] * 0.4                                # the part's centre sits this far beyond the TCP along its +z
    mass = float(rng.uniform(0.030, 0.150))
    com = [float(rng.uniform(-0.6, 0.6) * half[0]), float(rng.uniform(-0.6, 0.6) * half[1]), float(rng.uniform(-0.6, 0.6) * half[2])]
    return {"half": half, "seat_z": seat_z, "mass_kg": mass, "com_part": com}


def tool_for(kind, part=None):
    """The cell's clamped tool: the part (bins) or the needle with its holder (dispense); None in the shelf cell."""
    if kind == "dispense":
        return {"kind": "needle", "params": dict(NEEDLE)}
    if kind == "bins":
        return {"kind": "part", "params": {"mass_kg": part["mass_kg"], "com_part": part["com_part"], "half": part["half"],
                                           "seat_z": part["seat_z"]}}
    return None


def generate(design_xml, spec, nominal, cell, seed, q_start, log=None):
    """The jobs of one cell. q_start = the arm's design angles when the cell starts (home). Returns
    {"cell": kind, "tool": tool dict, "part": part or None, "jobs": [job, ...]} with each job's private fields."""
    rng = U.rng(seed, "jobs/" + cell.kind)
    _PITCH_CACHE.clear()
    part = part_spec(rng) if cell.kind == "bins" else None
    tool = tool_for(cell.kind, part)
    true_v = View(design_xml, spec, cell, tool=tool)
    nom_v = View(design_xml, nominal, cell.described(), tool=tool, design=true_v.design)
    true_v.kin.cond_min = COND_GEN                 # the privileged plans avoid near-singular configurations too
    views = (true_v, nom_v)
    jobs, examples = [], []
    q_prev = np.asarray(q_start, float)
    prev_target = None                                  # (pos, app) of the previous feasible job's end
    prev_exempt = []                                    # its support: the plan from it may stay close to it
    counts = dict(COUNTS[cell.kind])
    order = []
    for k, n in counts.items():
        order += [k] * n
    rng.shuffle(order)
    # infeasible jobs last in generation (they need feasible examples), first-come in the sequence
    feas_order = [k for k in order if k != "infeasible"]
    # the reach jobs' kinds are fixed before the attempts (the cell's main work, compartments / bins, for about 80 %;
    # the fixture's top for the rest), so a layout whose compartments are hard does not quietly fill up with tops
    n_reach = sum(1 for k in feas_order if k == "reach")
    n_main = int(round(MAIN_SHARE * n_reach)) if cell.kind in ("shelf", "bins") else n_reach
    reach_kinds = ["main"] * n_main + ["top"] * (n_reach - n_main)
    rng.shuffle(reach_kinds)
    import collections
    import time as _time
    why = collections.Counter()
    t_gen = _time.time()
    first_kind, others = INF_KINDS[cell.kind]
    inf_kinds = ([first_kind] if first_kind else []) + [others[int(i)] for i in rng.permutation(len(others))]
    blocked_pool, inf_log = [], []
    made = []
    for typ in feas_order:
        job = None
        sub = reach_kinds.pop() if typ == "reach" else None
        if not made and why.get("_first_failed"):
            break
        for attempt in range(80 if made else FIRST_JOB_ATTEMPTS):
            if typ == "path":
                path = _path_points(cell, rng)
                if path is None:
                    continue
                for v in views:
                    v.set_exempt([{"who": "needle", "what": "board"}])
                # the path points are the process program's, exact to the TRUE board; the described board is off by the
                # survey error, so on the nominal view the needle need only stay off it (no penetration)
                qs_n, info_n = _path_check(nom_v, path, rng, exempt_min=0.0)
                if qs_n is None:
                    why["path nominal " + info_n.get("why", "?")] += 1
                    continue
                qs_t, info_t = _path_check(true_v, path, rng, q0=qs_n[0])
                if qs_t is None:
                    why["path true " + info_t.get("why", "?")] += 1
                    continue
                true_v.set_exempt([{"who": "needle", "what": "board"}] + prev_exempt)
                plan = plan_to(true_v, q_prev, qs_t[0], path["tips"][0], -np.asarray(path["normal"]), rng, exit_from=prev_target,
                               entry_exempt_min=0.0015)
                if plan is None:
                    why["plan"] += 1
                    continue
                slow_saved = list(PLAN_INFO["slow"])
                true_v.set_exempt([{"who": "needle", "what": "board"}])
                if not exit_ok(true_v, qs_t[-1], -np.asarray(path["normal"], float), rng, exempt_min=0.0015):
                    why["no exit"] += 1
                    continue
                PLAN_INFO["slow"] = slow_saved
                job_end = (path["tips"][-1], -np.asarray(path["normal"], float), 0.0015)
                slow = list(PLAN_INFO["slow"])
                job = {"type": "path", "tool": "needle", "path": path, "q_true": [q.tolist() for q in qs_t],
                       "q_nominal": [q.tolist() for q in qs_n], "plan_slow": slow,
                       "plan_true": [q.tolist() for q in plan], "info": {"true": info_t, "nominal": info_n},
                       "exempt": [{"who": "needle", "what": "board"}],
                       "approach_time_target_s": _time_target(plan, slow),
                       "time_target_s": round(_time_target(plan, slow) + PV.DWELL_S + float(path["t"][-1]) + 0.5, 2)}
                q_end = qs_t[-1]
                break
            if cell.kind == "shelf":
                cand = _cand_compartment(cell, rng, views) if sub == "main" else _cand_top(cell, rng, views, "fixture", 0.022, 0.05)
            elif cell.kind == "bins":
                if sub == "main":
                    cand = _cand_bin(cell, rng, views, part)
                else:
                    cand = _cand_top(cell, rng, views, "fixture", 0.003, 0.012, part_drop=part["seat_z"] + part["half"][2])
                    if cand is not None:
                        cand["exempt"] = [{"who": "part", "what": "fixture"}]
            else:
                cand = _cand_cup(cell, rng, views)
            if cand is None:
                why["no pitch"] += 1
                continue
            for v in views:
                v.set_exempt(cand["exempt"])
            qn, inn = _ik_valid(nom_v, cand["pos"], cand["app"], cand["jaw"], GEN_CLEAR_M, rng, q0=cand.get("q_seed"))
            if qn is None:
                why["nominal " + inn.get("why", "?")] += 1
                if inn.get("why") in ("clearance", "self", "support") and len(blocked_pool) < 40:
                    blocked_pool.append(dict(cand))       # reached but in the geometry: a blocked-target candidate
                continue
            qt, it = _ik_valid(true_v, cand["pos"], cand["app"], cand["jaw"], GEN_CLEAR_M, rng, q0=qn)
            if qt is None:
                why["true " + it.get("why", "?")] += 1
                continue
            true_v.set_exempt(list(cand["exempt"]) + prev_exempt)
            plan = plan_to(true_v, q_prev, qt, cand["pos"], cand["app"], rng, exit_from=prev_target)
            if plan is None:
                why["plan"] += 1
                continue
            slow_saved = list(PLAN_INFO["slow"])
            true_v.set_exempt(list(cand["exempt"]))
            if not exit_ok(true_v, qt, cand["app"], rng):
                why["no exit"] += 1
                continue
            PLAN_INFO["slow"] = slow_saved
            slow = list(PLAN_INFO["slow"])
            job = {"type": "reach", "tool": {"shelf": "gripper", "bins": "gripper+part", "dispense": "needle"}[cell.kind],
                   "target": cand, "q_true": qt.tolist(), "q_nominal": qn.tolist(), "plan_true": [q.tolist() for q in plan],
                   "plan_slow": slow, "info": {"true": it, "nominal": inn}, "exempt": cand["exempt"],
                   "time_target_s": _time_target(plan, slow)}
            examples.append(cand)
            q_end = qt
            job_end = (cand["pos"], cand["app"])
            break
        if job is None:
            if not made:
                why["_first_failed"] = 1           # a layout in which even the first job fails: the caller redraws it
            if log:
                log("cell {}: no {} job after its attempts; rejects {}".format(cell.kind, typ, dict(why)))
            continue
        job["q_start"] = q_prev.tolist()
        job["exit_from"] = None if prev_target is None else [np.asarray(prev_target[0]).tolist(), np.asarray(prev_target[1]).tolist()] + \
            ([float(prev_target[2])] if len(prev_target) > 2 else [])
        made.append(job)
        q_prev = np.asarray(q_end, float)
        prev_target = job_end
        prev_exempt = list(job["exempt"])
        if log:
            log("  {} job {} ({:.1f} s so far) rejects {}".format(cell.kind, len(made), _time.time() - t_gen, dict(why)))
    # infeasible ones, inserted at random places; their public fields follow the feasible jobs of the cell
    who = {"shelf": "robot", "bins": "part", "dispense": "needle"}[cell.kind]
    feas_times = [j["time_target_s"] for j in made if j["type"] == "reach"] or [2.0]
    templates = list(examples)
    if not templates and cell.kind == "dispense":           # no feasible purge: the infeasible ones still sit at the cup
        for _ in range(20):
            t = _cand_cup(cell, rng, views)
            if t is not None:
                templates = [t]
                break
    for i in range(counts.get("infeasible", 0)):
        kind = inf_kinds[i % len(inf_kinds)]
        cand = None
        if kind == "blocked":
            tried = 0
            feas_names = {e["what"] for j in made for e in j.get("exempt", [])}
            while blocked_pool and cand is None:     # in the order the generator met them; on a support a feasible job uses
                c = blocked_pool.pop(0)
                if not all(e["what"] in feas_names for e in c["exempt"]):
                    continue
                tried += 1
                if _blocked_ok(views, c, rng):
                    cand = dict(c, kind="blocked")
            edge_tried = 0
            while cand is None and edge_tried < 20 and cell.kind in ("shelf", "bins"):   # by a wall, drawn on purpose
                edge_tried += 1
                c = _cand_compartment(cell, rng, views, edge=True) if cell.kind == "shelf" else _cand_bin(cell, rng, views, part, edge=True)
                if c is not None and all(e["what"] in feas_names for e in c["exempt"]) and _blocked_ok(views, c, rng):
                    cand = dict(c, kind="blocked")
            inf_log.append({"blocked_tried": tried, "edge_tried": edge_tried, "found": cand is not None})
            if cand is None:                         # none in this layout: another kind instead
                kind = KIND_FALLBACK["blocked"]
        for attempt in range(60 if cand is None else 0):
            c = _infeasible(cell, rng, views, kind, templates, who)
            if c is None:
                kind = KIND_FALLBACK.get(kind, "out_of_plane")
                continue
            for v in views:
                v.set_exempt(c["exempt"])
            if _infeasible_ok(views, c, rng):
                cand = c
                break
        if cand is None:
            continue
        if cand.get("kind") != "blocked":           # a blocked target keeps the jaw axis its candidate was realised with
            cand["jaw"] = _jaw_like(cand["app"], rng) if cell.kind in ("shelf", "bins") else None
        tt = float(feas_times[int(rng.integers(len(feas_times)))]) * float(rng.uniform(0.9, 1.1))
        job = {"type": "reach", "tool": {"shelf": "gripper", "bins": "gripper+part", "dispense": "needle"}[cell.kind],
               "target": cand, "infeasible": True, "exempt": cand["exempt"], "time_target_s": round(tt, 2)}
        made.insert(int(rng.integers(0, len(made) + 1)), job)
    for k, j in enumerate(made):
        j["id"] = "{}{:02d}".format(cell.kind[0].upper(), k + 1)
    return {"format": JOBS_FORMAT, "cell": cell.kind, "tool": tool, "part": part, "jobs": made, "infeasible_log": inf_log}


# ------------------------------------------------------------------------------------------------- public form
def _v(x, nd=5):
    return [round(float(v), nd) for v in x]


def public(job, cell):
    """The job as the controller receives it (cell frame; mm / deg where the field name says so)."""
    out = {"id": job["id"], "type": job["type"], "tool": job["tool"], "clearance_mm": 10.0, "time_target_s": job["time_target_s"],
           "exempt": [e["what"] for e in job.get("exempt", [])]}
    if job["type"] == "reach":
        t = job["target"]
        out.update({"position": _v(cell.to_cell(t["pos"])), "approach": _v(cell.dir_to_cell(t["app"]), 6),
                    "approach_tol_deg": APPROACH_TOL_DEG,
                    "jaw_axis": (None if t.get("jaw") is None else _v(cell.dir_to_cell(t["jaw"]), 6)), "jaw_tol_deg": JAW_TOL_DEG})
    else:
        p = job["path"]
        out.update({"t": [round(float(v), 3) for v in p["t"]], "points": [_v(cell.to_cell(x)) for x in p["tips"]],
                    "normal": _v(cell.dir_to_cell(p["normal"]), 6), "axis_tol_deg": AXIS_TOL_DEG, "gap_mm": GAP_MM,
                    "gap_tol_mm": GAP_TOL_MM, "speed_mm_s": round(p["speed"] * 1e3, 2), "shape": {k: p[k] for k in p if k in ("shape", "length_mm", "radius_mm", "sweep_deg")},
                    "approach_time_target_s": job["approach_time_target_s"]})
    return out
