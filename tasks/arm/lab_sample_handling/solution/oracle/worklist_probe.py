"""Lab sample handling, feasibility probes 7 and 8: the whole worklist, chained .

Worklist v1: ten capped tubes in the incoming rack (one row, A1-A10;
`worklist_scene.py`); eight to be spun, loaded into the eight-position rotor as four balanced opposite pairs (each
pair within BALANCE_G), two to named positions of the cold rack; then the lid closed. A full rotor needs turns: at
any rotor angle some positions are out of the arm's reach .

Each seed draws a unit and an episode: the turn's hidden properties from the decided ranges , the
rotor's start angle, where the handle sits in its play, which two slots go to which cold positions, the racks'
places, and the fills (the spun tubes in four levels at least FILL_GAP mL apart, the two of a level within FILL_PAIR
mL; the cold tubes anything in 3-13 mL).

The reference (probe 8: non-privileged estimates; `--privileged` takes them from the simulation, as probe 7 did):
1. locate the tubes in the incoming rack (top-down wrist images of the caps) and read their fills (the rack
   camera; `fill_read.read_rack`); pair the spun tubes by fill (sorted, neighbours);
2. locate the cold rack's holes (a top-down wrist image; `bench_estimators.locate_cold`) and place the two cold
   tubes;
3. read the rotor's angle and centre ; place every pair that fits now into an empty opposite pair of
   positions, each tube into a hole LOAD_MARGIN outside the bad sectors for its rack slot (the reach map's
   , and where the joints run out for that slot, SLOT_BAD from `--mode feasibility`), the hole nearer the robot
   first, at the hole's pose computed from the reading; each load plans its transfer by inverse kinematics and
   moves through the plan in joint space (`plan_transfer`, `pick_yaws`), the held tube taken as
   grasped where it was aimed (`centrifuge_probe.nominal_grasp`);
4. if pairs are left, turn the rotor (probe 5's learner, from the camera's readings) by the smallest angle after
   which the most pairs left fit, TURN_MARGIN inside; read again, load; repeat (at most MAX_TURNS);
5. close the lid.
Still privileged: whether the fingers hold the tube (`grasped`, from contacts) and the lid's hinge (`lid_cycle`).
Checked at the end from the simulation: every tube seated where it should be, every opposite pair within
BALANCE_G, the lid shut. Every estimate is logged against the truth.

    python -I worklist_probe.py --seed 0
    python -I worklist_probe.py --grid '{"seed": [0, 1, 2]}' --out ../outputs/lab/worklist8.jsonl
    python -I worklist_probe.py --mode feasibility           # prints SLOT_BAD
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import bench_estimators as E
import centrifuge_probe as C
import fill_read as F
import lab_sim as S
import tube_insert_probe as TP
import turn_probe as T
import worklist_scene as WS
from lab_sim import (  # noqa: F401
    FILL_GAP,
    FILL_PAIR,
    N_COLD,
    UNIT_PROPS,
    draw_episode,
    draw_unit,
    world,
)

BALANCE_G = 1.0                 # g, the largest mass difference of an opposite pair
BAD = ((57.0, 123.0), (247.0, 293.0))   # deg, a hole's world angle the arm cannot load at (the reach map's)
TURN_MARGIN = 12.0              # deg, an empty position this far inside a reachable sector after a turn
MAX_TURNS = 4
LOAD_MARGIN = 3.0               # deg, a position this far outside every bad sector is loaded without a turn
GRASP_OPEN = 0.25               # grip command approaching a tube: fingers about 30 mm apart (the rack's pitch is 45)
RELEASE_OPEN = 0.0              # grip command letting a tube go (about 40 mm apart; at 30 mm, 0.6 s, the fingers still
                                # held it and lifted it back out of an upright hole)
LIFT_EXTRA = 0.10               # m, the tube's bottom above its rack's rim when lifted (over the other tubes' caps)
CLEAR = 0.03                    # m, the held tube's bottom above the chamber's walls and the racks' caps on the way
# rad, each joint's margin to its limits a grasp yaw's plan needs, from the grasp to the push's end: joint 7 is the
# one that runs out on the way (the controller's path and the plan's differ by that much), joint 5 in the push
JOINT_NEED = np.array([0.05, 0.05, 0.05, 0.05, 0.15, 0.05, 0.3])

# deg, per incoming-rack slot (A1-A10): where no grasp with the fingers along the hole's tilt axis keeps the joints
# JOINT_NEED from their limits on the way to the hole and in the push (`--mode feasibility`: inverse kinematics every
# 5 deg); a grasp turned off that axis brings the hand onto the rings across the rotor
SLOT_BAD = {0: ((-2.5, 7.5), (182.5, 227.5), (337.5, 347.5)),
            1: ((-2.5, 7.5), (182.5, 227.5), (337.5, 347.5)),
            2: ((-2.5, 7.5), (107.5, 112.5), (182.5, 227.5), (337.5, 347.5)),
            3: ((-2.5, 7.5), (107.5, 117.5), (182.5, 227.5), (337.5, 347.5)),
            4: ((-2.5, 7.5), (107.5, 122.5), (182.5, 227.5), (337.5, 347.5)),
            5: ((-2.5, 7.5), (107.5, 127.5), (182.5, 227.5), (337.5, 347.5)),
            6: ((-2.5, 7.5), (107.5, 127.5), (182.5, 227.5), (337.5, 347.5)),
            7: ((2.5, 7.5), (107.5, 132.5), (182.5, 227.5), (337.5, 347.5)),
            8: ((2.5, 7.5), (107.5, 132.5), (182.5, 227.5), (337.5, 347.5)),
            9: ((2.5, 7.5), (107.5, 137.5), (182.5, 227.5), (337.5, 347.5))}


def bad_margin(phi, slot=None):
    """Distance (deg) of a hole's world angle from the bad sectors (the reach map's, and the source slot's if
    given): negative inside one."""
    best = 1e9
    for lo, hi in BAD + (SLOT_BAD[slot] if slot is not None else ()):
        d = (phi - (lo + hi) / 2 + 180) % 360 - 180          # from the sector's middle
        best = min(best, abs(d) - (hi - lo) / 2)
    return best


def placement(pairs, positions, theta, need):
    """Pairs to opposite positions (k, k + 4), each tube to a hole whose world angle (theta + 45 deg x hole) is
    at least `need` deg outside the bad sectors for the tube's slot: the assignment with the largest smallest
    margin among those that place as many pairs as possible, as [(tube, hole), ...] per pair; [] if none fits."""
    best, best_key = [], (0, -1e9)
    n = min(len(pairs), len(positions))
    for chosen in itertools.permutations(positions, n):
        for pr in itertools.permutations(pairs, n):
            plan, worst = [], 1e9
            for (ti, tj), k in zip(pr, chosen):
                opts = [((ti, k), (tj, k + 4)), ((tj, k), (ti, k + 4))]
                sc = [min(bad_margin(theta + 45 * h, t) for t, h in o) for o in opts]
                if max(sc) >= need:
                    plan.append(list(opts[int(np.argmax(sc))]))
                    worst = min(worst, max(sc))
            key = (len(plan), worst)
            if key > best_key:
                best, best_key = plan, key
    return best


def ik(p, q0, pos, R, iters=300):
    """Joint angles that put the grip site at (pos, R): damped least squares from q0, the start posture as the
    null-space goal (the controller's), not clamped to the joint ranges (a solution past a limit is what the
    controller would run into)."""
    m = p.model
    d = C.mujoco.MjData(m)
    d.qpos[:] = p.data.qpos
    q = np.array(q0, float)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[p.jq] = q
        C.mujoco.mj_kinematics(m, d)
        C.mujoco.mj_comPos(m, d)
        e = np.concatenate([pos - d.site_xpos[p.site], C.rotvec(R @ d.site_xmat[p.site].reshape(3, 3).T)])
        C.mujoco.mj_jacSite(m, d, jp, jr, p.site)
        J = np.vstack([jp[:, p.jv], jr[:, p.jv]])
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        q = q + Jp @ np.clip(e, -0.02, 0.02) + (np.eye(7) - Jp @ J) @ (0.05 * (C.B.HOME_Q - q))
    return q


def held_tube(p, src):
    """The tube to be taken: its tip and rotation as the controller knows them (an estimate in `src`: upright at
    the located top; else the simulation's)."""
    if "tube_top" in src:
        return np.asarray(src["tube_top"], float) - [0, 0, p.a.tube_len], np.eye(3)
    return p.tube_pos(), p.tube_R()


def plan_transfer(p, src, k, yaw, n=12):
    """Plan taking the held tube to hole k (`centrifuge_probe.target_hole`) with grasp yaw `yaw`, in joint space:
    inverse kinematics for the grasp (from above), the lift out of the source hole, then `n` poses on the straight
    line (position, and the rotation's angle) from the lifted pose to above the pre-insertion pose, the tube's
    bottom kept CLEAR above the chamber's walls and the racks' caps, the pre-insertion pose and the push's end; each
    pose solved from the one before. Returns the smallest of the joints' margins to their limits less what each
    needs (JOINT_NEED; negative: not feasible), the joints' margins, and the joint angles of the line's poses (what
    `take_to_hole` moves through)."""
    m = p.model
    jids = [m.joint(f"r_joint{i}").id for i in range(1, 8)]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    tip, Rt = held_tube(p, src)
    top = tip + Rt @ np.array([0, 0, p.a.tube_len])
    gp = top - np.array([0, 0, TP.GRASP_BELOW_TOP])
    R_down = p.down(yaw)
    x = R_down.T @ (tip - gp)
    h = C.target_hole(p, k)
    R_ee = h["R"] @ (R_down.T @ Rt).T
    pre = h["top"] + h["axis"] * TP.LIFT_CLEAR - R_ee @ x
    lifted = gp + [0, 0, src["depth"] + p.a.lift_extra]
    above = pre + [0, 0, 0.06]
    rv = C.rotvec(R_ee @ R_down.T)
    z_min = max(S.rotor_top(p.a) + C.LID_GAP, E.cap_top_z(p.a)) + CLEAR
    line = []
    for i in range(1, n + 1):
        s_ = i / n
        R = C.B.axis_angle_mat(rv * s_) @ R_down
        pos = lifted + (above - lifted) * s_
        low = pos + R @ x                                  # the tube's bottom
        if low[2] < z_min:
            pos = pos + [0, 0, z_min - low[2]]
        line.append((pos, R))
    q = p.data.qpos[p.jq].copy()
    marg = np.full(7, 1e9)
    path = []
    pushed = h["top"] - h["axis"] * p.a.held_depth - R_ee @ x          # where the push ends (the tube released)
    for i, (pos, R) in enumerate([(gp + [0, 0, 0.10], R_down), (gp, R_down), (lifted, R_down), *line, (pre, R_ee),
                                  (pushed, R_ee)]):
        q = ik(p, q, pos, R)
        marg = np.minimum(marg, np.minimum(q - lo, hi - q))
        if 3 <= i < 3 + n:
            path.append(q.copy())
    return float(np.min(marg - JOINT_NEED)), marg, path, q


def pick_yaws(p, src, k, yaw0):
    """The reference's grasp-yaw planner, each yaw's transfer planned in joint space (`take_to_hole` moves through
    the plan); the two with the most slack, the better first. A rotor hole: the two yaws with the fingers along its
    tilt axis (a half turn apart). An upright hole (the cold rack): twelve yaws 30 deg apart."""
    upright = float(C.target_hole(p, k)["axis"][2]) > 0.99
    offs = [math.radians(30 * i) for i in range(12)] if upright else [0.0, math.pi]
    yaws = [C.wrap(yaw0 + o) for o in offs]
    if upright and "rack_yaw" in src:
        # from an incoming-rack slot to an upright hole: the fingers close across the rack (pads along the rack's
        # x), so the hand's width lies across the row and not on the neighbours or the chamber's wall (v2)
        offs = [math.radians(o) for o in (0, 180, 20, -20, 200, 160)]
        yaws = [C.wrap(src["rack_yaw"] + o) for o in offs]
    plans = [plan_transfer(p, src, k, y) for y in yaws]
    order = sorted(range(len(yaws)), key=lambda i: -plans[i][0])[:2]
    p.transfer_q = {round(yaws[i], 6): plans[i][2] for i in order}
    p.last_pick = {"offsets_deg": [round(math.degrees(offs[i])) for i in order],
                   "slack": [round(plans[i][0], 2) for i in order]}
    return [yaws[i] for i in order]


def rest(p):
    """Back to the start posture between actions: up and home in the operational space, then in joint space (a
    load can leave a joint near its limit, which the operational-space controller cannot leave: joint 6 or 7 at its
    end stop made the next grasp's approach fail); the controller's null space back to the start posture."""
    p.q_null = C.B.HOME_Q.copy()
    p.home()
    p.home_joints()


ARG_DEFAULTS = {"privileged": True, "read": "truth", "read_noise": 0.3, "margin": 15.0, "max_cmd": 60.0, "past_ok": 2.5, "centre_err": 0.005, "view_dist": 0.20,
                "max_grips": 6, "stop_within": 2.0, "jobs": 1, "hole": 0, "yaw_choice": 0, "yaw_plan": None,
                "held_depth": 0.035, "lid_steps": 120, "turn_speed": 0.04, "release": "hold", "rotor_jitter": 0.0}


def reference_args(a):
    """Add the reference's settings to a world's settings `a` (an environment's: `lab_sim.sim_args`), in place."""
    for k, v in ARG_DEFAULTS.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    setup(a)
    return a


def make_args(seed, **kw):
    """The settings `main` gives a trial, for a caller outside the command line (the environment, the oracle)."""
    a = argparse.Namespace(seed=seed, **{**ARG_DEFAULTS, **kw})
    setup(a)
    return a


def setup(a):
    T.defaults(a)
    for k, v in {"grasp_open": GRASP_OPEN, "release_open": RELEASE_OPEN, "release_steps": 15,
                 "lift_extra": LIFT_EXTRA, "nominal_grasp": not a.privileged, "home_between": True,
                 "px_offset": 0.5}.items():
        setattr(a, k, v)
    a.yaw_pick = pick_yaws


def nominal_layout(a):
    """The racks' nominal layout (what the instruction gives; each rack sits a little off it per episode)."""
    slots = [np.array([WS.RACK_X, WS.RACK_Y0 + WS.RACK_PITCH * j]) for j in range(WS.N_SLOTS)]
    z = C.TABLE_Z + WS.COLD_DEPTH + 0.015
    cold = {}
    for i, r in enumerate("ABC"):
        for j in range(3):
            xy = WS.COLD_XY + WS.COLD_PITCH * np.array([i - 1, j - 1])
            cold[f"{r}{j + 1}"] = {"top": np.array([xy[0], xy[1], z])}
    return slots, cold


def trial(a):
    t0 = time.time()
    setup(a)
    unit = draw_unit(a.seed)
    p, src, names, cold, masses = world(a, unit)
    out = run_worklist(p, a, unit, src, names, cold, masses)
    out["bench"] = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(float(v), 4))
                    for k, v in a.bench.items()}
    out["bench_scale"] = a.bench_scale
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# v2 : every rotor load starts from the scale's holder; where it can reach is planned per episode
REACH_STEP = 5.0                # deg, the reach table's grid of a hole's world angle
MATCH_G = 1.0                   # g, two weighed tubes this close are a pair (the draw keeps pairs 0.2-0.8 g apart and
                                # consecutive pairs at least 1.2 g apart)
SETTLE_STEPS = 20               # control steps the scale's display is left to settle (5 lags) before it is read
READS = 5


def holder_src(a):
    """The scale's holder as a source with a tube standing in it (for planning before a tube is there)."""
    h = a.scale_hole
    tip = h["top"] - h["axis"] * (h["depth"] - 0.0005)
    return dict(h, tube_top=tip + np.array([0, 0, a.tube_len]))


def hand_clear(p, phi, qs):
    """Whether the arm at each of the joint angles `qs`, the rotor turned so hole 0 lies at world angle phi (deg),
    keeps off the rotor's other holes, hub and knob, the chamber and the lid (no contact in the simulation's
    geometry; hole 0's own ring excepted, the held tube not modelled)."""
    m = p.model
    d = C.mujoco.MjData(m)
    d.qpos[:] = p.data.qpos
    d.qpos[p.rotor_q] = math.radians(phi)
    for q in qs:
        d.qpos[p.jq] = q
        C.mujoco.mj_forward(m, d)
        if not _arm_clear(m, d):
            return False
    return True


def _arm_clear(m, d):
    for c in d.contact[:d.ncon]:
        n1, n2 = (m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
        g1, g2 = m.geom(c.geom1).name, m.geom(c.geom2).name
        robot = [n.startswith("r_") for n in (n1, n2)]
        if robot[0] == robot[1]:
            continue
        other, og = (n2, g2) if robot[0] else (n1, g1)
        if og.startswith("h0_"):
            continue
        if other in ("rotor", "handle", "lid") or og.startswith(("wall", "cabinet", "shield")):
            return False
    return True


def reach_table(p, a, src, step=REACH_STEP):
    """The bad sectors (deg intervals of a hole's world angle) for loading from `src`: per angle on the grid,
    whether either grasp yaw along the hole's tilt axis plans clear (`plan_transfer`'s slack > 0) and its path and
    push keep the arm off the rotor, the chamber and the lid (`hand_clear`). Planned from the rest posture, where
    every load starts (inverse kinematics from another posture can end on another branch of joint 7)."""
    rest(p)
    bad = []
    for phi in np.arange(0.0, 360.0, step):
        hole = E.rotor_hole(S.centre_xy(a), phi, 0, a)
        tilt = hole["R"][:, 1]
        y0 = math.atan2(tilt[1], tilt[0])
        ok = False
        for o in (0.0, math.pi):
            slack, _m, path, q = plan_transfer(p, src, hole, C.wrap(y0 + o))
            if slack > 0 and hand_clear(p, phi, [*path, q]):
                ok = True
                break
        if not ok:
            bad.append(float(phi))
    out = []
    for phi in bad:
        if out and abs(phi - step - out[-1][1]) < 1e-6:
            out[-1][1] = phi
        else:
            out.append([phi, phi])
    if len(out) > 1 and out[0][0] == 0.0 and abs(out[-1][1] - (360.0 - step)) < 1e-6:
        out[0][0] = out[-1][0] - 360.0
        out.pop()
    return [(lo - step / 2, hi + step / 2) for lo, hi in out]


def reach_margin(phi, bad):
    """Distance (deg) of a hole's world angle from the bad sectors: negative inside one."""
    best = 1e9
    for lo, hi in bad:
        d = (phi - (lo + hi) / 2 + 180) % 360 - 180
        best = min(best, abs(d) - (hi - lo) / 2)
    return best


def weigh(p, a, name, src):
    """Take tube `name` from `src` to the scale's holder, step back, let the display settle, read it (the median of
    READS readings). Returns (grams or None, the load's record)."""
    h = a.scale_hole
    got = C.load(p, name, src, dict(h, true=h), a)
    if not C.seated(p, name, h):
        return None, got
    rest(p)
    for _ in range(SETTLE_STEPS):
        p.step([0, 0, 0], [0, 0, 0], -1)
    vals = []
    for _ in range(READS):
        p.step([0, 0, 0], [0, 0, 0], -1)
        vals.append(p.scale_display())
    return float(np.median(vals)), got


def run_worklist_v2(p, a, unit, src, names, cold, masses):
    """The v2 reference  on world `p` for the episode `unit`; poses from the simulation (a token run's
    privilege), masses from the scale, the rotor's angle from the wrist camera. Returns the outcome.

    1. the cold tubes to their positions;
    2. the rotor's reach from the scale's holder, planned for this bench (`reach_table`);
    3. each spun tube: weighed in the holder, then loaded from the holder: opposite its partner if a placed tube
       weighed within MATCH_G (else into an empty opposite pair), at a hole LOAD_MARGIN outside the bad sectors,
       the rotor turned first if none is;
    4. the lid closed."""
    rng = np.random.default_rng([a.seed, 29])
    read = T.Reader(p, a, rng, fallback=unit["rotor_start"])
    turner = T.Turner("learner")
    slots = WS.slot_names()
    out = {"seed": a.seed, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in unit.items()
                              if k not in ("fills", "cold")},
           "fills": unit["fills"], "cold": {slots[i]: n for i, n in unit["cold"].items()}}
    log, placed, weights = [], {}, {}
    hsrc = a.scale_hole
    spun = list(unit.get("spin", [i for i in range(S.N_SLOTS) if i not in unit["cold"]]))
    status = "ok"
    try:
        # the lid first, if it starts closed (closed, its handle overhangs the incoming rack's end)
        if status == "ok" and p.lid_angle() < 0.5:
            rest(p)
            lid = C.lid_cycle(p, a, phases=("open",))
            log.append({"lid_open": lid.get("open"), "t": p.t})
            if not lid.get("open", {}).get("done"):
                status = "lid open"
        # 1. the cold tubes
        for tube, name in unit["cold"].items():
            rest(p)
            h = cold[name]
            got = C.load(p, names[tube], src[tube], dict(h, true=h), a)
            ok = C.seated(p, names[tube], h)
            log.append({"tube": tube, "cold": name, "status": got["status"], "seated": ok, "tries": got.get("tries"),
                        **({"fails": got["fails"]} if got.get("fails") else {}), "t": p.t})
            if not ok:
                status = "cold"
                break
        # 2. reach from the holder
        t0 = time.time()
        bad = reach_table(p, a, holder_src(a))
        out["reach_bad"] = [[round(x, 1) for x in iv] for iv in bad]
        out["reach_s"] = round(time.time() - t0, 1)
        rest(p)
        theta = read() if status == "ok" else None
        # rotor hole -> weight (g) of the tube there whose partner is still to come; the tubes already in the rotor
        # are on the worklist with their masses
        waiting = {int(h): round(a.mass * 1000 + f, 3) for h, f in unit.get("preloaded", {}).items()}
        for h in waiting:
            placed[h] = None
        turns = 0
        for tube in spun:
            if status != "ok":
                break
            rest(p)
            g, got = weigh(p, a, names[tube], src[tube])
            log.append({"tube": tube, "weigh_g": g, "true_g": round(masses[tube] * 1000, 2), "status": got["status"],
                        **({"fails": got["fails"]} if got.get("fails") else {}), "t": p.t})
            if g is None:
                status = "weigh"
                break
            weights[tube] = g
            match = [(abs(w - g), k) for k, w in waiting.items() if abs(w - g) <= MATCH_G]
            if match:
                k = min(match)[1]
                cands = [(k + 4) % 8]
            else:
                k = None
                cands = [h for h in range(8) if h not in placed and (h + 4) % 8 not in placed]
            while True:
                marg = {h: reach_margin(theta + 45 * h, bad) for h in cands}
                h = max(cands, key=lambda c: marg[c])
                if marg[h] >= LOAD_MARGIN:
                    break
                if turns >= MAX_TURNS + 4:
                    status = "turns"
                    break
                # any turn up to a half turn either way (a hole pinned opposite its partner may have its nearest
                # reachable sector more than 90 deg away), the smallest that brings a candidate TURN_MARGIN inside
                ds = [d for d in np.arange(-180.0, 180.5, 1.0)
                      if max(reach_margin(theta + d + 45 * c, bad) for c in cands) >= TURN_MARGIN]
                if not ds:
                    status = "no turn"
                    break
                goal = theta + float(min(ds, key=abs))
                rest(p)
                st, grips, theta = turner.turn_to(p, a, goal, read, theta, a.max_grips, a.stop_within)
                turns += 1
                log.append({"turn_goal_deg": round(goal - theta, 1), "status": st, "grips": grips, "t": p.t})
                if st != "ok":
                    status = "turn"
                    break
                rest(p)
            if status != "ok":
                break
            got = C.load(p, names[tube], hsrc, h, a)
            rest(p)
            ok = C.seated(p, names[tube], h)
            log.append({"tube": tube, "hole": h, "status": got["status"], "seated": ok, "tries": got.get("tries"),
                        "margin": round(marg[h], 1),
                        **{kk: got.get(kk) for kk in ("jammed", "jam_contacts", "fails") if got.get(kk) is not None},
                        "t": p.t})
            placed[h] = tube
            if not ok:
                status = "load"
                break
            if k is None:
                waiting[h] = g
            else:
                waiting.pop(k, None)
        # 4. the lid
        lid = {}
        if status == "ok":
            rest(p)
            lid = C.lid_cycle(p, a, phases=("close",))
            if not lid.get("close", {}).get("done"):
                status = "lid"
        out["lid"] = lid.get("close")
        out["turns"] = turns
    except FloatingPointError:
        status = "unstable"
    pre = getattr(p, "pre", {})

    def body(h):
        return pre[h] if placed[h] is None else names[placed[h]]

    def mass(h):
        return p.extra_mass[pre[h]] if placed[h] is None else masses[placed[h]]

    seated = {h: C.seated(p, body(h), h) for h in placed}
    cold_final = {n: C.seated(p, names[t], cold[n]) for t, n in unit["cold"].items()}
    n_rotor = len(unit.get("spin", [])) + len(pre) if "spin" in unit else 8
    pair_diff = [abs(mass(k) - mass(k + 4)) * 1000 for k in range(4) if k in placed and k + 4 in placed]
    errs = [weights[t] - masses[t] * 1000 for t in weights]
    stay = [i for i, n in enumerate(names) if n is not None and i not in unit["cold"] and i not in spun]
    foreign_ok = (all(C.seated(p, n, cold[c]) for c, n in getattr(p, "foreign", {}).items())
                  and all(C.seated(p, names[i], src[i]) for i in stay))
    out.update(status=status, placed=len(placed), all_seated=bool(len(placed) == n_rotor and all(seated.values())),
               cold_seated=bool(all(cold_final.values())), foreign_ok=bool(foreign_ok),
               balanced=bool(len(pair_diff) == n_rotor // 2 and max(pair_diff, default=0) <= BALANCE_G),
               pair_diff_g=[round(x, 2) for x in pair_diff], lid_shut=bool(p.lid_angle() < 0.02),
               weigh_err_g=[round(e, 3) for e in errs], read_failures=read.failed, log=log, steps=p.t,
               sim_min=round(p.t / 10 / 60, 1))
    out["done"] = bool(out["all_seated"] and out["cold_seated"] and out["balanced"] and out["lid_shut"]
                       and out["foreign_ok"])
    return out


# v3 : a shield with one loading port over the hole ring, a one-way rotor (ratchet)
PORT_GRIPS = 40                 # grips a turn to the port may take (a turn round after passing it is about 8)
PORT_HELD = 0.020               # m, a tube pushed this far into a rotor hole before it is let go (deeper, the fingers
                                # reach the shield beside the port); it slides the rest of the way by its weight


def port_held(a, fn, *args):
    """Call fn with the rotor's push depth (PORT_HELD) in place of the racks' (`a.held_depth`)."""
    keep = a.held_depth
    a.held_depth = PORT_HELD
    try:
        return fn(*args)
    finally:
        a.held_depth = keep


def port_reach(p, a, src):
    """Whether the port can be loaded from `src` (the scale's holder): either grasp yaw along the port hole's tilt
    axis plans clear (`plan_transfer`) and its path keeps the arm off the rotor, chamber, shield and lid."""
    rest(p)
    phi = math.degrees(S.port_az(a))
    hole = E.rotor_hole(S.centre_xy(a), phi, 0, a)
    tilt = hole["R"][:, 1]
    y0 = math.atan2(tilt[1], tilt[0])
    best = -1e9
    for o in (0.0, math.pi):
        slack, _m, path, q = plan_transfer(p, src, hole, C.wrap(y0 + o))
        if slack > 0 and hand_clear(p, phi, [*path, q]):
            return True, round(slack, 3)
        best = max(best, slack)
    return False, round(best, 3)


def port_need(theta, h, a):
    """deg the one-way rotor must turn forward to bring hole h to the port, from its angle theta (0 if there)."""
    err = (math.degrees(S.port_az(a)) - 45 * h - theta + 180) % 360 - 180
    return 0.0 if -max(a.stop_within, a.past_ok) <= err <= a.stop_within else err % 360


def run_worklist(p, a, unit, src, names, cold, masses):
    """The v3 reference  on world `p` for the episode `unit`; poses from the simulation and the rotor's
    angle as the truth plus `read_noise` (a token run's privilege), masses from the scale. Returns the outcome.

    1. the lid opened if it starts shut; the cold tubes to their positions;
    2. the port's reach from the scale's holder checked for this bench;
    3. each spun tube weighed in the holder, then given a hole: opposite its partner if a placed tube weighed
       within MATCH_G, else the empty opposite pair the forward-turning rotor brings to the port first; the rotor
       turned forward to bring that hole to the port (`turn_probe.OneWay`, the identifier: the play told from the
       give, every grip erring short), the tube loaded from the holder through the port;
    4. the lid closed."""
    rng = np.random.default_rng([a.seed, 29])
    read = T.Reader(p, a, rng, fallback=unit["rotor_start"])
    turner = T.OneWay("identifier", a.margin)
    slots = WS.slot_names()
    out = {"seed": a.seed, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in unit.items()
                              if k not in ("fills", "cold")},
           "fills": unit["fills"], "cold": {slots[i]: n for i, n in unit["cold"].items()}}
    log, placed, weights = [], {}, {}
    hsrc = a.scale_hole
    spun = list(unit["spin"])
    status, lid = "ok", {}
    grips_total, overs = 0, 0
    try:
        if p.lid_angle() < 0.5:
            rest(p)
            lid = C.lid_cycle(p, a, phases=("open",))
            log.append({"lid_open": lid.get("open"), "t": p.t})
            if not lid.get("open", {}).get("done"):
                status = "lid open"
        for tube, name in unit["cold"].items():
            if status != "ok":
                break
            rest(p)
            h = cold[name]
            got = C.load(p, names[tube], src[tube], dict(h, true=h), a)
            ok = C.seated(p, names[tube], h)
            log.append({"tube": tube, "cold": name, "status": got["status"], "seated": ok, "tries": got.get("tries"),
                        **({"fails": got["fails"]} if got.get("fails") else {}), "t": p.t})
            if not ok:
                status = "cold"
        if status == "ok":
            reach, slack = port_held(a, port_reach, p, a, holder_src(a))
            out["port_slack"] = slack
            if not reach:
                status = "port reach"
        rest(p)
        theta = read() if status == "ok" else None
        waiting = {int(h): round(a.mass * 1000 + f, 3) for h, f in unit.get("preloaded", {}).items()}
        for h in waiting:
            placed[h] = None
        for tube in spun:
            if status != "ok":
                break
            rest(p)
            g, got = weigh(p, a, names[tube], src[tube])
            log.append({"tube": tube, "weigh_g": g, "true_g": round(masses[tube] * 1000, 2), "status": got["status"],
                        **({"fails": got["fails"]} if got.get("fails") else {}), "t": p.t})
            if g is None:
                status = "weigh"
                break
            weights[tube] = g
            match = [(abs(w - g), k) for k, w in waiting.items() if abs(w - g) <= MATCH_G]
            if match:
                k = min(match)[1]
                cands = [(k + 4) % 8]
            else:
                k = None
                cands = [h for h in range(8) if h not in placed and (h + 4) % 8 not in placed]
            if not cands:
                status = "no hole"
                break
            h = min(cands, key=lambda c: port_need(theta, c, a))
            rest(p)
            goal = math.degrees(S.port_az(a)) - 45 * h
            st, grips, over, glog, theta = turner.turn_to(p, a, goal, read, theta, PORT_GRIPS)
            grips_total += grips
            overs += over
            err = (goal - T.rotor_deg(p) + 180) % 360 - 180
            log.append({"turn_hole": h, "status": st, "grips": grips, "over": over, "err_deg": round(err, 2),
                        "cmd_got": glog, "t": p.t})
            if st != "ok":
                status = "turn"
                break
            rest(p)
            got = port_held(a, C.load, p, names[tube], hsrc, h, a)
            rest(p)
            ok = C.seated(p, names[tube], h)
            log.append({"tube": tube, "hole": h, "status": got["status"], "seated": ok, "tries": got.get("tries"),
                        **{kk: got.get(kk) for kk in ("jammed", "jam_contacts", "fails") if got.get(kk) is not None},
                        "t": p.t})
            placed[h] = tube
            if not ok:
                status = "load"
                break
            if k is None:
                waiting[h] = g
            else:
                waiting.pop(k, None)
        if status == "ok":
            rest(p)
            lid = C.lid_cycle(p, a, phases=("close",))
            if not lid.get("close", {}).get("done"):
                status = "lid"
        out["lid"] = lid.get("close")
        out["grips"] = grips_total
        out["overshoots"] = overs
    except FloatingPointError:
        status = "unstable"
    pre = getattr(p, "pre", {})

    def body(h):
        return pre[h] if placed[h] is None else names[placed[h]]

    def mass(h):
        return p.extra_mass[pre[h]] if placed[h] is None else masses[placed[h]]

    seated = {h: C.seated(p, body(h), h) for h in placed}
    cold_final = {n: C.seated(p, names[t], cold[n]) for t, n in unit["cold"].items()}
    n_rotor = len(unit["spin"]) + len(pre)
    pair_diff = [abs(mass(k) - mass(k + 4)) * 1000 for k in range(4) if k in placed and k + 4 in placed]
    errs = [weights[t] - masses[t] * 1000 for t in weights]
    stay = [i for i, n in enumerate(names) if n is not None and i not in unit["cold"] and i not in spun]
    foreign_ok = (all(C.seated(p, n, cold[c]) for c, n in getattr(p, "foreign", {}).items())
                  and all(C.seated(p, names[i], src[i]) for i in stay))
    out.update(status=status, placed=len(placed), all_seated=bool(len(placed) == n_rotor and all(seated.values())),
               cold_seated=bool(all(cold_final.values())), foreign_ok=bool(foreign_ok),
               balanced=bool(len(pair_diff) == n_rotor // 2 and max(pair_diff, default=0) <= BALANCE_G),
               pair_diff_g=[round(x, 2) for x in pair_diff], lid_shut=bool(p.lid_angle() < 0.02),
               weigh_err_g=[round(e, 3) for e in errs], read_failures=read.failed, log=log, steps=p.t,
               sim_min=round(p.t / 10 / 60, 1))
    out["done"] = bool(out["all_seated"] and out["cold_seated"] and out["balanced"] and out["lid_shut"]
                       and out["foreign_ok"])
    return out


def run_worklist_v1(p, a, unit, src, names, cold, masses):
    """The reference on world `p` (in process, or a shadow driven over the wire, `lab_env/wire_world.py`) for the
    episode `unit` (`draw_unit`, or a unit's properties with `draw_episode`); `src`, `names`, `cold`, `masses` as
    `world` returns them (the true poses and masses are used for the log and the final check only). Returns the
    outcome."""
    rng = np.random.default_rng([a.seed, 29])
    read = T.Reader(p, a, rng, fallback=unit["rotor_start"])
    turner = T.Turner("learner")
    out = {"seed": a.seed, "privileged": a.privileged,
           **{k: round(v, 3) for k, v in unit.items() if k not in ("fills", "cold")},
           "fills": unit["fills"], "cold": {WS.slot_names()[i]: n for i, n in unit["cold"].items()}}
    est = {}
    log = []
    placed = {}                     # rotor hole -> tube index
    cold_seated = {}
    status = "ok"
    try:
        # 1. the tubes and their fills
        true_xy = [p.data.xpos[p.model.body(n).id][:2].copy() for n in names]
        if a.privileged:
            tube_xy = true_xy
            fill_est = list(unit["fills"])
        else:
            prior_slots, _ = nominal_layout(a)
            tube_xy = E.locate_tubes(p, rng, a, prior_slots)
            missing = [i for i, xy in enumerate(tube_xy) if xy is None]
            if missing:
                raise RuntimeError(f"tubes not found: {missing}")
            for i, xy in enumerate(tube_xy):
                src[i]["tube_top"] = np.array([xy[0], xy[1], E.cap_top_z(a)])
            rest(p)
            fill_est = F.read_rack(p, rng, tube_xy, WS.rack_mid_z(a))
            if any(f is None for f in fill_est):
                raise RuntimeError("fill not read")
        est["tube_err_mm"] = [round(1000 * float(np.linalg.norm(np.asarray(e) - t)), 2) for e, t in zip(tube_xy, true_xy)]
        est["fill_err_ml"] = [round(f - t, 3) for f, t in zip(fill_est, unit["fills"])]
        spun = [i for i in range(WS.N_SLOTS) if i not in unit["cold"]]
        order = sorted(spun, key=lambda i: fill_est[i])
        pairs = [(order[2 * i], order[2 * i + 1]) for i in range(4)]
        est["pairs_right"] = all(abs(unit["fills"][i] - unit["fills"][j]) < FILL_GAP / 2 for i, j in pairs)
        # 2. the cold tubes
        rest(p)
        if a.privileged:
            cold_est = {n: h["top"] for n, h in cold.items()}
        else:
            _, prior_cold = nominal_layout(a)
            cold_est = E.locate_cold(p, rng, a, prior_cold)
        est["cold_err_mm"] = {n: round(1000 * float(np.linalg.norm(cold_est[n][:2] - h["top"][:2])), 2)
                              for n, h in cold.items() if n in unit["cold"].values()}
        for tube, name in unit["cold"].items():
            rest(p)
            h = cold[name]
            target = dict(h, top=np.asarray(cold_est[name], float), true=h)
            got = C.load(p, names[tube], src[tube], target, a)
            cold_seated[name] = C.seated(p, names[tube], h)
            log.append({"tube": tube, "cold": name, "status": got["status"], "seated": cold_seated[name],
                        "tries": got.get("tries"), "pick": getattr(p, "last_pick", None),
                        **{k: got.get(k) for k in ("jammed", "jam_at", "radial")}, "t": p.t})
            if not cold_seated[name]:
                status = "cold"
                break
        rest(p)
        # 3-4. the rotor
        theta = read() if status == "ok" else None
        turns = 0
        while pairs and status == "ok":
            positions = [k for k in range(4) if k not in placed and k + 4 not in placed]
            for plan in placement(pairs, positions, theta, LOAD_MARGIN):
                pair = tuple(sorted(t for t, _ in plan))
                pairs = [q for q in pairs if tuple(sorted(q)) != pair]
                holes = {h: (h if a.privileged else E.rotor_hole(read.centre, theta, h, a)) for _, h in plan}
                for tube, h in sorted(plan, key=lambda th: float(C.target_hole(p, holes[th[1]])["top"][0])):
                    target = holes[h]
                    err = None
                    if isinstance(target, dict):
                        target = dict(target, true=p.hole(h))
                        err = round(E.across_mm(target, p.hole(h)), 2)
                    got = C.load(p, names[tube], src[tube], target, a)
                    rest(p)
                    ok = C.seated(p, names[tube], h)
                    log.append({"tube": tube, "hole": h, "status": got["status"], "seated": ok, "hole_err_mm": err,
                                "tries": got.get("tries"), "pick": getattr(p, "last_pick", None),
                                **{k: got.get(k) for k in ("jammed", "jam_at", "jam_contacts", "radial")},
                                "t": p.t})
                    placed[h] = tube
                    if not ok:
                        status = "load"
                        break
                if status != "ok":
                    break
            if not pairs or status != "ok":
                break
            if turns >= MAX_TURNS:
                status = "turns"
                break
            # the turn after which the most pairs left fit, TURN_MARGIN outside the bad sectors; the smallest of
            # those (the bad sectors leave too little room for a full rotor at one angle: more than one turn)
            positions = [k for k in range(4) if k not in placed and k + 4 not in placed]
            fits = {d: len(placement(pairs, positions, theta + d, TURN_MARGIN)) for d in np.arange(-90.0, 90.5, 1.0)}
            most = max(fits.values())
            cands = sorted((d for d, n in fits.items() if n == most and n > 0), key=abs)[:1]
            if not cands:
                status = "no turn"
                break
            goal = theta + float(cands[0])
            st, grips, theta = turner.turn_to(p, a, goal, read, theta, a.max_grips, a.stop_within)
            turns += 1
            log.append({"turn_goal_deg": round(float(cands[0]), 1), "status": st, "grips": grips,
                        "err_deg": round((goal - T.rotor_deg(p) + 180) % 360 - 180, 2), "t": p.t})
            if st != "ok":
                status = "turn"
                break
            rest(p)
        # 5. the lid
        lid = {}
        if status == "ok":
            lid = C.lid_cycle(p, a, phases=("close",))
            if not lid.get("close", {}).get("done"):
                status = "lid"
        out["lid"] = lid.get("close")
    except FloatingPointError:
        status = "unstable"
    except RuntimeError as e:
        status = "perception"
        out["error"] = str(e)
    # the final state, from the simulation
    seated = {h: C.seated(p, names[tube], h) for h, tube in placed.items()}
    cold_final = {n: C.seated(p, names[t], cold[n]) for t, n in unit["cold"].items()}
    pair_diff = [abs(masses[placed[k]] - masses[placed[k + 4]]) * 1000 for k in range(4)
                 if k in placed and k + 4 in placed]
    out.update(status=status, placed=len(placed), all_seated=bool(len(placed) == 8 and all(seated.values())),
               cold_seated=bool(all(cold_final.values())),
               balanced=bool(len(pair_diff) == 4 and max(pair_diff) <= BALANCE_G),
               pair_diff_g=[round(x, 2) for x in pair_diff], lid_shut=bool(p.lid_angle() < 0.02),
               turns=sum(1 for e in log if "turn_goal_deg" in e), read_failures=read.failed, est=est, log=log,
               steps=p.t, sim_min=round(p.t / 10 / 60, 1))
    out["done"] = bool(out["all_seated"] and out["cold_seated"] and out["balanced"] and out["lid_shut"])
    return out


def feasibility(a, phi):
    """For the rotor turned so hole 0 sits at world angle phi: per incoming slot, whether either grasp yaw along
    the hole's tilt axis plans clear (`plan_transfer`'s slack > 0), with the tube's true pose."""
    setup(a)
    unit = draw_unit(0)
    unit.update(rotor_start=float(phi), rotor_friction=0.3, handle_play=0.0, rotor_inertia=0.02)
    p, src, names, _cold, _m = world(a, unit)
    out = []
    for slot in range(WS.N_SLOTS):
        p.use(names[slot])
        tilt = p.hole(0)["R"][:, 1]
        yaw0 = math.atan2(tilt[1], tilt[0])
        out.append(max(plan_transfer(p, src[slot], 0, C.wrap(yaw0 + o))[0] for o in (0.0, math.pi)) > 0)
    return phi, out


def feasibility_table(a, step=5):
    from concurrent.futures import ProcessPoolExecutor
    phis = list(range(0, 360, step))
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = dict(ex.map(feasibility, [argparse.Namespace(**vars(a))] * len(phis), phis))
    table = {}
    for slot in range(WS.N_SLOTS):
        bad = [phi for phi in phis if not rows[phi][slot]]
        iv = []
        for phi in bad:
            if iv and phi - iv[-1][1] == step:
                iv[-1][1] = phi
            else:
                iv.append([phi, phi])
        if len(iv) > 1 and iv[0][0] == 0 and iv[-1][1] == 360 - step:
            iv[0][0] = iv[-1][0] - 360
            iv.pop()
        table[slot] = tuple((lo - step / 2, hi + step / 2) for lo, hi in iv)
        print(slot, "".join("." if phi in bad else "#" for phi in phis), file=sys.stderr)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("worklist", "feasibility"), default="worklist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--privileged", action="store_true", help="estimates from the simulation (probe 7's reference)")
    ap.add_argument("--read", choices=("truth", "camera"), default="truth")
    ap.add_argument("--read-noise", type=float, default=0.3)
    ap.add_argument("--centre-err", type=float, default=0.005)
    ap.add_argument("--view-dist", type=float, default=0.20)
    ap.add_argument("--max-grips", type=int, default=6)
    ap.add_argument("--stop-within", type=float, default=2.0)
    ap.add_argument("--bench-scale", type=float, default=1.0, help="the bench draw's ranges x this (0: nominal)")
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    for k, v in ARG_DEFAULTS.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    if a.mode == "feasibility":
        print(json.dumps(feasibility_table(a)))
        return
    if not a.grid:
        print(json.dumps(trial(a)))
        return
    from concurrent.futures import ProcessPoolExecutor
    grid = json.loads(a.grid)
    keys = list(grid)
    jobs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        b = argparse.Namespace(**vars(a))
        for k, val in zip(keys, combo):
            setattr(b, k.replace("-", "_"), val)
        jobs.append(b)
    with ProcessPoolExecutor(a.jobs) as ex, open(a.out, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in ex.map(trial, jobs))
    print(f"{len(jobs)} trials -> {a.out}")


if __name__ == "__main__":
    main()
