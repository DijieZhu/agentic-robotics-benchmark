"""Lab sample handling, feasibility probe 2: the centrifuge (rotor positions, balance, lid) and fill levels.

Questions: which positions of a fixed-angle rotor can
the Panda load, and can it load the two positions of an opposite pair (a balanced run needs them); can it close
and open a hinged lid; and do the tubes' fill levels (their masses, which the balance depends on) show in the
cameras.

World: tube_insert_probe's (box transport's Panda and controller, the stiff contact model). The centrifuge is
built from primitives: a cabinet, a chamber of four walls, a rotor on a vertical hinge (free to turn, damped)
with eight holes tilted 45 deg toward its axis, and a lid hinged at the back with a handle bar at the front.
Tubes start in a source rack.

    python -I centrifuge_probe.py --mode position --hole 4      # load one rotor position
    python -I centrifuge_probe.py --mode pair --hole 4          # load positions 4 and 0 (opposite)
    python -I centrifuge_probe.py --mode lid                    # close the open lid, then open it
    python -I centrifuge_probe.py --mode render --out-dir ../outputs/lab/render

Probe 4 : `--rotor-model solid` adds a visual rotor body (a bowl with bored holes, over the same
collision rings) and a knob bar on the hub; `--rotor-start` sets the rotor's angle (the reach map);

    python -I centrifuge_probe.py --mode turn --rotor-model solid --tubes 4 --hole 4 --second 2 --turn 45

loads the pair 4 and 0, turns the rotor by its knob, and loads the pair 2 and 6.

Probe 5 (`turn_probe.py`): `handle_play` (deg) puts the knob on its own hinge with that much free play,
`rotor_inertia` sets the rotor's inertia about its axis, `turn_speed` and `release` shape a grip's turn.

Positions: hole k sits at angle 2 pi k / 8 around the rotor axis, measured from +x (away from the robot);
hole 4 is the one nearest the robot.
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco
import numpy as np

import tube_insert_probe as TP
from lab_sim import (  # noqa: F401
    BORE_RGBA,
    CENTRE,
    CHAMBER,
    CONE_LIFT,
    DTH_MESH,
    HANDLE,
    HANDLE_LIMIT_TC,
    HOLE_DEPTH,
    HOLE_RADIUS_AT,
    HUB_TOP,
    KNOB,
    KNOB_YAW,
    KNOB_Z,
    LID_GAP,
    LID_LEN,
    LID_OPEN,
    MARK_R,
    MARK_RADIUS,
    MARK_RGBA,
    MAX_DQ,
    NHOLES,
    ROTOR_BASE_R,
    ROTOR_BOTTOM,
    ROTOR_BOWL_R,
    ROTOR_RGBA,
    ROTOR_RIM_R,
    ROTOR_TOP,
    SRC_DEPTH,
    SRC_XY,
    TABLE_Z,
    TILT,
    World,
    add_lid,
    build,
    fill_height,
    hole_local,
    rot_x,
    rot_z,
    rotor_mesh,
    seated,
    set_lid_mass,
    target_hole,
    wrap,
)
from tube_insert_probe import B, rotvec

# the solid rotor (`--rotor-model solid`): a visual body with bored holes over the same collision rings, and a
# knob bar on the hub to turn it by. The holes lean with their tops toward the axis (a fixed-angle rotor), so the
# top is a bowl through the entrances (each entrance faces up and in) and the side widens downward.
KNOB_GRIP = 0.018                     # m, the grip point from the hub's axis along the bar (a lever arm; 26 mm before
                                      # v3: the shield's inner edge leaves the fingers room up to about 18 mm)
TURN_TOL = math.radians(1.5)          # a turn closer than this to the command is done
KNOB_BACKLASH = math.radians(4.5)     # the fingers' give on the bar: a grip turns the rotor this much less (measured)
KNOB_OPEN = 0.25                      # grip command while on the bar: fingers about 30 mm apart (fully open, they
                                      # reach the rings of the holes beside the hub)


def nearer_j7_yaw(p, yaw, choice=0):
    """Of the finger yaws `yaw` and `yaw + pi` (a two-finger grasp is symmetric under a half turn), the one
    that keeps joint 7 nearer the middle of its range (choice 0) or the other (choice 1): a top-down hand
    turning by +d about world z moves joint 7 by about -d (box transport's rule)."""
    fw = p.ee_R() @ p.f_local
    cur = math.atan2(fw[1], fw[0])
    j7 = float(p.data.qpos[p.jq[6]])
    cands = sorted((wrap(yaw), wrap(yaw + math.pi)), key=lambda c: abs(j7 - wrap(c - cur)))
    return cands[choice]


def nominal_grasp(p, R_ee):
    """The held tube in the hand frame as a controller without the tube's pose assumes it (probe 8): upright when
    grasped with the hand at R_ee, GRASP_BELOW_TOP below its top. Returns (G, x) as `take_to_hole` uses them."""
    return R_ee.T, R_ee.T @ np.array([0, 0, -(p.a.tube_len - TP.GRASP_BELOW_TOP)])


def take_to_hole(p, src, k, yaw):
    """Grasp the tube at `yaw`, lift it out, bring it to the pre-insertion pose of hole k (`target_hole`). A
    source with `tube_top` (an estimate) is grasped there; else at the tube's true pose."""
    a = p.a
    top = src["tube_top"] if "tube_top" in src else p.tube_pos() + p.tube_R() @ np.array([0, 0, a.tube_len])
    gp = top - np.array([0, 0, TP.GRASP_BELOW_TOP])
    R_down = p.down(yaw)
    g_open = getattr(a, "grasp_open", -1)
    if not (p.servo(gp + [0, 0, 0.10], R_down, g_open, 0.025, max_steps=200) and p.servo(gp, R_down, g_open, 0.01)):
        con = robot_contacts(p)
        p.servo(gp + [0, 0, 0.10], p.ee_R(), g_open, 0.025, max_steps=60)
        return {"status": "approach", "contacts": con}
    for _ in range(8):
        p.step([0, 0, 0], [0, 0, 0], 1)
    if not p.grasped():
        return {"status": "no grasp"}
    p.servo(p.ee() + [0, 0, src["depth"] + getattr(a, "lift_extra", 0.06)], R_down, 1, 0.01, max_steps=200)
    Re, pe = p.ee_R(), p.ee()
    nominal = getattr(a, "nominal_grasp", False)
    if nominal:
        G, x = nominal_grasp(p, R_down)
    else:
        G = Re.T @ p.tube_R()
        x = Re.T @ (p.tube_pos() - pe)
    h = target_hole(p, k)
    R_ee_des = h["R"] @ G.T

    def ee_for(bottom):
        return bottom - R_ee_des @ x

    pre = h["top"] + h["axis"] * TP.LIFT_CLEAR
    out = {"R_down": R_down, "G": G, "x": x, "ee_for": ee_for}
    plan = getattr(p, "transfer_q", {}).get(round(yaw, 6))
    if plan is not None:
        # a caller planned the transfer in joint space (probe 7): move through its postures joint by joint, so
        # the arm takes the ones the plan checked against the joint limits, not those the controller would
        # drift into
        for q in plan:
            joint_move(p, q, 1, rate=0.03, settle=0)
        joint_move(p, plan[-1], 1, rate=0.03, settle=8)
        p.q_null = np.asarray(plan[-1], float).copy()   # the controller's null space holds this posture, not home
        if not nominal:
            Re, pe = p.ee_R(), p.ee()            # the tube may have shifted in the fingers: measure it again
            G = Re.T @ p.tube_R()
            x = Re.T @ (p.tube_pos() - pe)
            R_ee_des = h["R"] @ G.T

            def ee_for(bottom):
                return bottom - R_ee_des @ x

            out = {"R_down": R_down, "G": G, "x": x, "ee_for": ee_for}
    if not p.servo(ee_for(pre) + [0, 0, 0.06], R_ee_des, 1, 0.02, tol=0.005, max_steps=300):
        return dict(out, status="transfer", contacts=robot_contacts(p),
                    off_mm=round(1000 * float(np.linalg.norm(ee_for(pre) + [0, 0, 0.06] - p.ee())), 1))
    if not p.servo(ee_for(pre), R_ee_des, 1, 0.006, tol=0.0015, max_steps=200):
        return dict(out, status="pre-insert", contacts=robot_contacts(p),
                    off_mm=round(1000 * float(np.linalg.norm(ee_for(pre) - p.ee())), 1))
    return dict(out, status="ready")


def robot_contacts(p):
    """The arm's and the held tube's contacts now (geom name pairs), for a failure's record."""
    m, d = p.model, p.data
    out = set()
    for c in d.contact[:d.ncon]:
        b1, b2 = (m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
        if (b1.startswith("r_") or b2.startswith("r_") or p.tube in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])) \
                and not (b1.startswith("r_") and b2.startswith("r_")):
            out.add("/".join(sorted((m.geom(c.geom1).name, m.geom(c.geom2).name))))
    return sorted(out)[:10]


def joint_move(p, q_goal, grip, rate=0.06, settle=8):
    """Joint PD (gravity compensated, `World.joint_step`) along a straight line in joint space to q_goal, `rate`
    rad per control step at most for the joint that moves most, then `settle` steps there."""
    q0 = p.data.qpos[p.jq].copy()
    q_goal = np.asarray(q_goal, float)
    n = max(5, int(np.ceil(np.max(np.abs(q_goal - q0)) / rate)))
    for i in range(1, n + settle + 1):
        p.joint_step(q0 + (q_goal - q0) * min(1.0, i / n), grip)


def put_back(p, src, R_down):
    """Return the held tube upright into its source hole and let go."""
    above = src["top"] + [0, 0, 0.03]
    lift = p.ee() + [0, 0, 0.08]
    p.servo(lift, R_down, 1, 0.02, tol=0.005, max_steps=200)
    off = p.ee() - p.tube_pos()
    p.servo(above + off, R_down, 1, 0.02, tol=0.003, max_steps=300)
    p.servo(src["top"] - [0, 0, src["depth"] - 0.004] + off, R_down, 1, 0.005, tol=0.003, max_steps=150)
    g_open = getattr(p.a, "grasp_open", -1)
    for _ in range(8):
        p.step([0, 0, 0], [0, 0, 0], g_open)
    p.servo(p.ee() + [0, 0, 0.10], p.ee_R(), g_open, 0.02, tol=0.005, max_steps=80)


def load(p, name, src, k, a):
    """Take tube `name` out of its source hole and load rotor hole k: insert it while held until the fingers
    near the entrance, release, let it slide to the floor. Returns the outcome."""
    p.use(name)
    # grasp with the finger axis along the target hole's tilt axis (tangential to the rotor), so the hand only
    # pitches about the finger axis. Two such yaws (a half turn apart): try the one joint 7 prefers (or the
    # plan's), and if the arm cannot reach, put the tube back in its source hole and try the other.
    tilt_axis = target_hole(p, k)["R"][:, 1]
    yaw0 = math.atan2(tilt_axis[1], tilt_axis[0])
    first = a.yaw_choice if not a.yaw_plan else json.loads(a.yaw_plan).get(str(k), a.yaw_choice)
    tries = []
    yaws = [nearer_j7_yaw(p, yaw0, choice) for choice in (first, 1 - first)]
    if getattr(a, "yaw_pick", None):
        # a caller's planner picks the two grasp yaws to try (probe 7: the fingers need not lie along the tilt
        # axis; any turn about the tube's axis inserts the same)
        yaws = a.yaw_pick(p, src, k, yaw0)
    fails = []
    for yaw in yaws:
        got = take_to_hole(p, src, k, yaw)
        tries.append(got["status"])
        if got["status"] != "ready":
            fails.append({k2: got[k2] for k2 in ("status", "contacts", "off_mm") if k2 in got})
        if got["status"] == "ready":
            break
        if got["status"] in ("transfer", "pre-insert"):
            put_back(p, src, got["R_down"])
        if getattr(a, "home_between", False):
            # the failed try can leave joint 7 at its end stop, where the other yaw (a half turn away) cannot
            # be reached from: back to the start posture first (probe 7)
            p.home_joints()
    else:
        return {"status": tries[-1], "tries": tries, "fails": fails}
    G = got["G"]
    ee_for = got["ee_for"]
    # insert while held by `held_depth`, release, and let the tube slide down the 45 deg hole by its weight (a
    # deeper push brings the hand onto the rings of the holes across the rotor)
    held_depth = a.held_depth
    p.force_ref = p.wrench_force()
    p.max_force = 0.0
    p.track_force = True
    jammed, jam_at, jam_contacts = False, None, []
    n = 40
    for i in range(n + 10):
        s = min(1.0, (i + 1) / n)
        h = target_hole(p, k)                      # a rotor hole by index: followed as the rotor moves (privileged)
        bottom = h["top"] + h["axis"] * (TP.LIFT_CLEAR - s * (TP.LIFT_CLEAR + held_depth))
        e = ee_for(bottom) - p.ee()
        p.integral = np.clip(p.integral + 0.15 * e, -0.02, 0.02)
        p.step(e + p.integral, np.clip(rotvec(h["R"] @ G.T @ p.ee_R().T), -0.15, 0.15), 1)
        if float(np.linalg.norm(p.wrench_force() - p.force_ref)) > TP.FORCE_LIMIT:
            jammed = True
            h = target_hole(p, k)
            jam_at = round(float(-np.dot(p.tube_pos() - h["top"], h["axis"])), 4)
            mm, dd = p.model, p.data
            jam_contacts = sorted({"/".join(sorted((mm.geom(c.geom1).name, mm.geom(c.geom2).name)))
                                   for c in dd.contact[:dd.ncon]
                                   if p.tube in (mm.geom_bodyid[c.geom1], mm.geom_bodyid[c.geom2])
                                   or mm.body(mm.geom_bodyid[c.geom1]).name.startswith("r_")
                                   or mm.body(mm.geom_bodyid[c.geom2]).name.startswith("r_")})
            break
    p.track_force = False
    g_open = getattr(a, "release_open", getattr(a, "grasp_open", -1))
    for _ in range(getattr(a, "release_steps", 6)):
        p.step([0, 0, 0], [0, 0, 0], g_open)
    p.servo(p.ee() - p.ee_R()[:, 2] * 0.08, p.ee_R(), g_open, 0.01, max_steps=60)
    for _ in range(25):
        p.step([0, 0, 0], [0, 0, 0], g_open)
    h = target_hole(p, k) if not isinstance(k, dict) else k.get("true", k)
    rel = p.tube_pos() - h["top"]
    depth_in = float(-np.dot(rel, h["axis"]))
    radial = float(np.linalg.norm(rel + depth_in * h["axis"]))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p.tube_R()[:, 2], h["axis"]))))))
    return {"status": "ok", "tries": tries, "jammed": jammed, "jam_at": jam_at, "jam_contacts": jam_contacts[:8],
            "max_force_N": round(p.max_force, 2), "depth_in": round(depth_in, 4),
            "radial": round(radial, 4), "angle_deg": round(ang, 2),
            "seated": bool(depth_in > h.get("depth", HOLE_DEPTH) - 0.006 and radial < h["radius"] and ang < 5.0)}


def knob_world(p):
    """The knob's centre and its bar's direction in the world, now (privileged)."""
    b = p.model.geom_bodyid[p.model.geom("knob").id]
    Rr = p.data.xmat[b].reshape(3, 3)
    return p.data.xpos[b] + Rr @ np.array([0, 0, KNOB_Z]), Rr @ np.array([math.cos(KNOB_YAW), math.sin(KNOB_YAW), 0])


def turn_rotor(p, delta, a, tol=TURN_TOL, tries=3):
    """Turn the rotor by `delta` (rad): a grip-and-turn, then corrections for the part the fingers' give left
    (privileged: it reads the rotor's angle; an agent would read it from the wrist camera)."""
    q0 = float(p.data.qpos[p.rotor_q])
    out = {"turn_cmd_deg": round(math.degrees(delta), 1), "turn_grips": []}
    for _ in range(tries):
        rest = delta - (float(p.data.qpos[p.rotor_q]) - q0)
        if abs(rest) < tol:
            break
        one = turn_once(p, rest + math.copysign(KNOB_BACKLASH, rest), a)
        out["turn_grips"].append(one)
        if one["turn_status"] != "ok":
            out.update(turn_status=one["turn_status"], q=one.get("q"))
            return out
        p.home()
    turned = float(p.data.qpos[p.rotor_q]) - q0
    out.update(turn_status="ok", turned_deg=round(math.degrees(turned), 2),
               turn_err_deg=round(math.degrees(turned - delta), 2),
               turn_max_force_N=max(g["turn_max_force_N"] for g in out["turn_grips"]))
    return out


def turn_once(p, delta, a):
    """Grip the knob bar across it near one end, carry the hand along the arc about the rotor's axis by `delta`
    (rad) while turning it by the same angle, let go. The fingers push the bar's sides at KNOB_GRIP from the axis,
    so the turn does not rely on friction."""
    q0 = float(p.data.qpos[p.rotor_q])
    c, bar = knob_world(p)
    fw = p.ee_R() @ p.f_local
    cur = math.atan2(fw[1], fw[0])
    j7 = float(p.data.qpos[p.jq[6]])
    # candidate grips: either end of the bar, either finger yaw (a half turn apart); first the end nearer the
    # robot's base (less stretch) and the yaw that leaves joint 7 nearest its middle after the turn (a hand turning
    # by +d about world z moves joint 7 by about -d); if the arm cannot get there, back off and try the next
    across = math.atan2(bar[1], bar[0]) + math.pi / 2          # fingers close across the bar
    cands = sorted(((end, y) for end in (1.0, -1.0) for y in (wrap(across), wrap(across + math.pi))),
                   key=lambda ey: (float(np.linalg.norm((c + ey[0] * bar * KNOB_GRIP)[:2])),
                                   abs(j7 - wrap(ey[1] - cur) - delta)))
    out = {"turn_cmd_deg": round(math.degrees(delta), 1)}
    for tried, (end, yaw) in enumerate(cands):
        arm = end * bar * KNOB_GRIP
        R_down = p.down(yaw)
        g0 = c + arm
        if (p.servo(g0 + [0, 0, 0.10], R_down, KNOB_OPEN, 0.025, tol=0.004, max_steps=250)
                and p.servo(g0, R_down, KNOB_OPEN, 0.008, tol=0.002, max_steps=200)):
            out["grip_tries"] = tried + 1
            break
        p.servo(p.ee() + [0, 0, 0.08], p.ee_R(), -1, 0.02, max_steps=60)
        p.home_joints()
    else:
        out.update(turn_status="approach", q=np.round(p.data.qpos[p.jq], 2).tolist())
        return out
    for _ in range(8):
        p.step([0, 0, 0], [0, 0, 0], 1)
    p.force_ref = p.wrench_force()
    p.max_force = 0.0
    p.track_force = True
    p.integral = np.zeros(3)
    R0 = p.ee_R()
    speed = getattr(a, "turn_speed", 0.04)                    # rad per control step along the arc
    moving = getattr(a, "release", "hold") == "moving"
    n = max(3 if moving else 10, math.ceil(abs(delta) / speed))
    # "hold": stop at the end, hold, then open; "moving": open the fingers while the hand is still on the arc
    # (the rotor keeps what speed it has and coasts)
    steps = [(i, 1) for i in range(n + (0 if moving else 15))] + [(n + i, KNOB_OPEN) for i in range(6 if moving else 0)]
    for i, grip in steps:
        s = (i + 1) / n if moving else min(1.0, (i + 1) / n)
        Rz = B.axis_angle_mat(np.array([0.0, 0.0, delta * s]))
        e = c + Rz @ arm - p.ee()
        p.integral = np.clip(p.integral + 0.15 * e, -0.02, 0.02)
        p.step(e + p.integral, np.clip(rotvec(Rz @ R0 @ p.ee_R().T), -0.15, 0.15), grip)
    p.track_force = False
    for _ in range(0 if moving else 6):
        p.step([0, 0, 0], [0, 0, 0], KNOB_OPEN)
    p.servo(p.ee() + [0, 0, 0.10], p.ee_R(), KNOB_OPEN, 0.02, tol=0.005, max_steps=80)
    for _ in range(30):
        p.step([0, 0, 0], [0, 0, 0], -1)
    turned = float(p.data.qpos[p.rotor_q]) - q0
    out.update(turn_status="ok", turned_deg=round(math.degrees(turned), 2),
               turn_err_deg=round(math.degrees(turned - delta), 2), turn_max_force_N=round(p.max_force, 2))
    return out


def follow_lid(p, phi_to, a, steps):
    """Hold the handle and drive it along its arc to lid angle phi_to (privileged: the hinge is known)."""
    b = p.model.body("lid").id
    hinge = p.data.xpos[b].copy()
    phi0 = p.lid_angle()
    Rl = p.data.xmat[b].reshape(3, 3) @ rot_x(phi0).T     # the lid's base frame (angle 0)
    R_hold = p.ee_R()
    off = p.ee() - p.handle_world()                # the gripper site relative to the handle, held fixed
    for i in range(steps):
        phi = phi0 + (phi_to - phi0) * min(1.0, (i + 1) / (steps - 10))
        handle = hinge + Rl @ rot_x(phi) @ HANDLE  # rotation about the lid frame's x lifts the free edge
        e = handle + off - p.ee()
        p.integral = np.clip(p.integral + 0.15 * e, -0.02, 0.02)
        p.step(e + p.integral, np.clip(rotvec(R_hold @ p.ee_R().T), -0.15, 0.15), 1)


def lid_cycle(p, a, phases=("close", "open")):
    """Close the open lid by its handle, release, check; then open it again, release, check (`phases`)."""
    out = {}
    for phase, phi_to in (("close", 0.0), ("open", a.lid_open)):
        if phase not in phases:
            continue
        h = p.handle_world()
        bar_axis = p.data.xmat[p.model.body("lid").id].reshape(3, 3)[:, 0]
        yaw = math.atan2(bar_axis[1], bar_axis[0]) + math.pi / 2         # fingers across the bar
        R_down = p.down(nearer_j7_yaw(p, yaw))
        approach = h + [0, 0, 0.10]
        # fingers opened to about 30 mm (grip 0.25), enough for the 15 mm bar, so the inner finger stays off
        # the plate next to it
        if not (p.servo(approach, R_down, 0.25, 0.025, tol=0.004, max_steps=250)
                and p.servo(h - [0, 0, 0.002], R_down, 0.25, 0.008, tol=0.002, max_steps=200)):
            out[phase] = {"status": "approach", "lid_angle": round(p.lid_angle(), 3),
                          "q": np.round(p.data.qpos[p.jq], 2).tolist()}
            return out
        for _ in range(8):
            p.step([0, 0, 0], [0, 0, 0], 1)
        p.force_ref = p.wrench_force()
        p.max_force = 0.0
        p.track_force = True
        p.integral = np.zeros(3)
        follow_lid(p, phi_to, a, a.lid_steps)
        p.track_force = False
        for _ in range(6):
            p.step([0, 0, 0], [0, 0, 0], -1)
        p.servo(p.ee() + [0, 0, 0.08], p.ee_R(), -1, 0.01, max_steps=40)
        for _ in range(30):
            p.step([0, 0, 0], [0, 0, 0], -1)
        ang = p.lid_angle()
        ok = ang < 0.02 if phase == "close" else ang > a.lid_open - 0.1
        out[phase] = {"status": "ok", "done": bool(ok), "lid_angle": round(ang, 3),
                      "max_force_N": round(p.max_force, 2)}
        if not ok:
            return out
    return out


def trial(a):
    t0 = time.time()
    m, src, names = build(a)
    p = World(m, a)
    out = {"mode": a.mode, "hole": a.hole, "yaw_choice": a.yaw_choice, "held_depth": a.held_depth,
           "rotor_friction": a.rotor_friction, "lid_open": a.lid_open, "rotor_jitter": a.rotor_jitter, "lid": "none" if a.no_lid else a.lid_side, "clearance": a.clearance, "rotor_damping": a.rotor_damping,
           "lid_start": a.lid_start, "seed": a.seed,
           "rotor_model": a.rotor_model, "rotor_start": a.rotor_start, "turn": a.turn,
           "rotor_inertia": a.rotor_inertia, "handle_play": a.handle_play}
    try:
        for _ in range(10):
            p.step([0, 0, 0], [0, 0, 0], -1)
        if a.mode == "position":
            out.update(load(p, names[0], src[0], a.hole, a))
        elif a.mode == "pair":
            first = load(p, names[0], src[0], a.hole, a)
            out["first"] = first
            if first.get("seated"):
                q_before = float(p.data.qpos[p.rotor_q])
                second = load(p, names[1], src[1], (a.hole + NHOLES // 2) % NHOLES, a)
                out["second"] = second
                out["rotor_turn_deg"] = round(math.degrees(float(p.data.qpos[p.rotor_q]) - q_before), 2)
                h = p.hole(a.hole)
                p.use(names[0])
                rel = p.tube_pos() - h["top"]
                out["first_still_seated"] = bool(-np.dot(rel, h["axis"]) > HOLE_DEPTH - 0.006)
        elif a.mode == "lid":
            out.update(lid_cycle(p, a))
        elif a.mode == "turn":
            # load the opposite pair (first, first + 4), turn the rotor by --turn deg, check the pair stayed
            # seated, then load the pair (second, second + 4) that was out of reach before the turn
            k1, k2 = a.hole, a.second
            loads = {}
            for name, src_i, k in ((names[0], 0, k1), (names[1], 1, (k1 + 4) % NHOLES)):
                loads[k] = load(p, name, src[src_i], k, a)
                p.home()
            out["before"] = {str(k): r.get("seated", r.get("status")) for k, r in loads.items()}
            out.update(turn_rotor(p, math.radians(a.turn), a))
            p.home()
            out["pair_kept"] = bool(seated(p, names[0], k1) and seated(p, names[1], (k1 + 4) % NHOLES))
            after = {}
            for name, src_i, k in ((names[2], 2, k2), (names[3], 3, (k2 + 4) % NHOLES)):
                after[str(k)] = load(p, name, src[src_i], k, a)
                p.home()
            out["after"] = {k: r.get("seated", r.get("status")) for k, r in after.items()}
            out["after_tries"] = {k: r.get("tries") for k, r in after.items()}
            out["all_seated"] = bool(seated(p, names[0], k1) and seated(p, names[1], (k1 + 4) % NHOLES)
                                     and all(r.get("seated") for r in after.values()))
        out["rotor_deg"] = round(math.degrees(float(p.data.qpos[p.rotor_q])), 2)
    except FloatingPointError:
        out["status"] = "unstable"
    out["steps"] = p.t
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def render(a):
    """Three tubes at 3, 8 and 13 mL in the source rack seen from the fixed cameras and a close camera; the
    liquid column's height per millilitre in pixels."""
    from PIL import Image
    fills = [3.0, 8.0, 13.0]
    SRC_XY.insert(2, np.array([0.36, 0.38]))
    a.tubes = 3
    m, _src, _names = build(a, fills=fills)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {}
    tube = m.body("tube").id
    base = d.xpos[tube].copy()
    for cam in ("left", "right"):
        _p, _t, fovy, wdt, hgt = B.CAMERAS[cam]
        r = mujoco.Renderer(m, hgt, wdt)
        r.update_scene(d, camera=cam)
        Image.fromarray(r.render()).save(os.path.join(a.out_dir, f"{cam}.png"))
        r.close()
        cid = m.camera(cam).id
        f = (hgt / 2) / math.tan(math.radians(fovy) / 2)
        Rc, pc = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]

        def px(pt, Rc=Rc, pc=pc, f=f):
            v = Rc.T @ (pt - pc)
            return np.array([f * v[0] / -v[2], f * v[1] / -v[2]])

        per_ml = np.linalg.norm(px(base + [0, 0, fill_height(1.0, a)]) - px(base))
        out[cam] = {"px_per_mL": round(float(per_ml), 2), "dist_m": round(float(np.linalg.norm(pc - base)), 3)}
    # a close look, like the wrist camera 0.15 m from the tubes (its 75 deg field of view, 256 x 192)
    cpos = base + np.array([-0.15, 0.05, 0.05])
    look = B.look_at(cpos, base + [0, 0.05, 0.04])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fwd = look @ np.array([0, 0, -1.0])
    cam.lookat[:] = cpos + fwd * 0.15
    cam.distance = 0.15
    cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
    cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
    m.vis.global_.fovy = B.WRIST["fovy"]
    r = mujoco.Renderer(m, B.WRIST["height"], B.WRIST["width"])
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(os.path.join(a.out_dir, "close.png"))
    r.close()
    f = (B.WRIST["height"] / 2) / math.tan(math.radians(B.WRIST["fovy"]) / 2)
    out["close"] = {"px_per_mL": round(float(f * fill_height(1.0, a) / 0.15), 2), "dist_m": 0.15}
    out["mm_per_mL"] = round(fill_height(1.0, a) * 1000, 2)
    out["images"] = sorted(os.listdir(a.out_dir))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("position", "pair", "lid", "turn", "render"), default="position")
    ap.add_argument("--rotor-model", choices=("rings", "solid"), default="rings",
                    help="solid: a visual rotor body with bored holes and a knob on the hub")
    ap.add_argument("--rotor-start", type=float, default=0.0, help="deg, the rotor's start angle (before jitter)")
    ap.add_argument("--rotor-inertia", type=float, default=0.008, help="kg m^2 about the rotor's axis")
    ap.add_argument("--handle-play", type=float, default=0.0, help="deg, the knob's free play (solid rotor)")
    ap.add_argument("--turn", type=float, default=45.0, help="deg, turn mode: the rotor turn between the pairs")
    ap.add_argument("--second", type=int, default=2, help="turn mode: the second pair's first position")
    ap.add_argument("--hole", type=int, default=4, help="rotor position (4 = nearest the robot)")
    ap.add_argument("--clearance", type=float, default=0.001)
    ap.add_argument("--rotor-damping", type=float, default=0.05, help="N m s / rad")
    ap.add_argument("--rotor-friction", type=float, default=0.3, help="N m, the rotor hinge's friction loss")
    ap.add_argument("--rotor-jitter", type=float, default=0.0, help="deg, the rotor's start angle drawn in +-this")
    ap.add_argument("--yaw-plan", default=None, help='JSON {hole: yaw_choice}, overrides --yaw-choice per hole')
    ap.add_argument("--lid-open", type=float, default=LID_OPEN, help="rad, the open stop")
    ap.add_argument("--lid-start", type=float, default=None, help="rad, 0 = closed; default: open")
    ap.add_argument("--lid-mass", type=float, default=0.4)
    ap.add_argument("--lid-side", choices=("back", "right"), default="right")
    ap.add_argument("--no-lid", action="store_true", help="no lid (rotor reach only)")
    ap.add_argument("--lid-damping", type=float, default=0.3)
    ap.add_argument("--lid-friction", type=float, default=0.5, help="N m, the hinge's friction loss (holds it open)")
    ap.add_argument("--lid-steps", type=int, default=120, help="control steps for the lid's arc")
    ap.add_argument("--yaw-choice", type=int, default=0, choices=(0, 1),
                    help="of the two grasp yaws, 0 = the heuristic's pick, 1 = the other")
    ap.add_argument("--held-depth", type=float, default=0.035, help="m, inserted while held, then released")
    ap.add_argument("--tubes", type=int, default=2)
    ap.add_argument("--tube-d", type=float, default=0.017)
    ap.add_argument("--tube-len", type=float, default=0.120)
    ap.add_argument("--mass", type=float, default=0.015)
    ap.add_argument("--com-z", type=float, default=0.035)
    ap.add_argument("--friction", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="../outputs/lab/render")
    ap.add_argument("--grid", help="JSON {arg: [values]}: every combination in parallel, one line each to --out")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=48)
    a = ap.parse_args()
    if a.lid_start is None:
        a.lid_start = a.lid_open
    if a.mode == "render":
        print(json.dumps(render(a)))
        return
    if not a.grid:
        print(json.dumps(trial(a)))
        return
    import itertools
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
        f.writelines(json.dumps(row) + "\n" for row in ex.map(trial, jobs))
    print(f"{len(jobs)} trials -> {a.out}")


if __name__ == "__main__":
    main()
