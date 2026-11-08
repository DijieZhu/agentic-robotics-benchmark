"""Box skills of the reference controller (privileged; runs inside the grader's process).

erect_box(rig)   fold the four walls of the flat blank until every wall snap-locks
pack_pair(rig)   put both shoes of the current pair into the erected box, sole down, heel-to-toe
put_lid(rig)     fetch the lid tray from the infeed tray and set it over the box
ship_box(rig)    push the closed box across the outfeed line

Every motion is done with the left (gripper) arm through the Rig's controllers; the right (hook) arm is parked at its
home posture first.  The scene queries are read directly (the controller is the oracle).

Geometry used throughout (left/tcp frame: +z = approach, +y = finger closing axis, +x across the pads):
  * the fingertips reach 8.9 mm beyond the tcp and are 9 mm wide across the pads; the hand body starts 37 mm behind
    the tcp, is 0.2 m wide along the closing axis and ~38 mm thick across the pads next to the fingers;
  * the gripper attaches a handle when the fingers close with the handle inside the pinch box (|x| <= 12 mm,
    |y| <= half the opening, z in [-10, +12] mm): the "grasp" of a thin panel or of the lid plate is therefore virtual
    (the fingers close in the air just above the object and the weld snaps on).

Folding a wall: the hand is welded to the panel, so it must swing on the quarter circle about the crease while rotating
by the same angle.  With the tool straight down the hand ends up horizontal inside the box, where the wrist (link 7)
would lie across the opposite wall at the wall's height; the grasp is therefore PITCHED: the approach is tilted toward
the crease by `alpha`, so that after the fold the wrist rises inward at `alpha` above the horizontal.
"""

import math

import numpy as np

from primitives import IK_REF_OFFSETS, EpisodeOver, down_quat, link_press, rot_quat  # noqa: F401
from shoecell.control import RATE_LIMIT_RAD_S, SEED_OFFSETS, arm_collisions, ik_solve, mat_to_quat, orientation_error, quat_mul, quat_to_mat

__all__ = ["erect_box", "pack_pair", "put_lid", "ship_box", "park_hook_arm"]

FINGER_TIP_Z = 0.0089        # fingertips beyond the tcp along the approach
FINGER_TIP_X = 0.0090        # half-width of the fingertip across the pads
PINCH_Z_HI = 0.012           # attach rule: handle at most this far beyond the tcp
ATTACH_MARGIN = 0.001        # keep the handle this far inside the pinch box
CARRY_Z = 0.25               # tcp height while carrying a shoe or the lid
SHOE_GRASP_Z = 0.009         # a shoe handle sits this far beyond the tcp when the fingers close (3 mm inside the attach
                             # limit: the fingers may stop a couple of mm high on the lace tails lying around the shoe)
LEFT_BASE = np.array([-0.05, 0.30, 0.0])
# Layout (read from the scene at runtime, see _zones): the blank is delivered at (0.30, -0.22), well clear of the
# infeed tray whose -y edge is at y = 0.02.  The +y wall is folded where the blank arrived (its pitched hand leans
# toward the arm and converges better further away from it); then the blank is dragged toward the arm to BOX_X and
# to the y where the far (-y) wall and the two short walls are within reach of a pitched grasp, while the closed box
# stays >= TRAY_CLEAR from the tray's edge on its way to the outfeed line.
BOX_X = 0.16                 # floor centre x for folding the -y wall and the two short walls (a +x wall out of reach here is
                             # answered by a 2 cm corrective drag toward the arm)
BOX_Y_MAX = -0.125           # floor centre y upper bound (the -y wall's handle is then ~0.31 m from the base; its fold nudges
                             # a blank delivered yaw ~pi about 2 cm toward +y)
TRAY_CLEAR = 0.03            # the closed box keeps this much from the tray's -y edge during the push
FLOOR_Y_SHORT_MAX = -0.12    # floor centre y bound while the short walls are folded (their 0.2 m wide hand then ends
                             # >= 2 cm from the tray's -y edge at y = 0.02 and from the lid tray lying on it)
ARC_MARGIN_MIN = 0.15        # a short-wall fold arc with less joint-limit margin (rad) is not attempted: the blank is
                             # dragged 2 cm toward the arm first (a stretched arm misses the handle by 17-24 mm)
APPROACH_Z = 0.20            # every wall handle is approached from this height (lift, travel, descend)
MAX_CORRECTIVE_DRAGS = 2     # corrective drags per wall (each costs ~8 s)
MX_FOLD_X_MAX = 0.12         # the -x (near) wall is folded only with the floor centre at x <= this: folded further out
                             # its arc slides the blank 4-5 cm toward -y and ends with the hand on the +y wall's top
LID_RIM_CLEAR = 0.02         # a lid handle closer than this to the infeed tray's rim is not used (the outer finger
                             # would come down on the rim)
PUSH_GAP = 0.025             # the pitched hand descends to the push start this far from the box wall (a 12 mm gap let
                             # the hand's back brush the lid's overhanging wall on its way down), then creeps in
EDGE_MARGIN = 0.02           # the flat blank's +y free edge stays this far below the tray's -y edge
WALL_FORCE_MAX = 10.0        # a shoe placement descent stops (backs off 2 cm, re-centres, tries again) as soon as a Panda
                             # body (hand, fingers, pads, links) presses the carton or the lid harder than this
SOLE_CLEAR = 0.02            # a carried shoe enters its slot vertically from a hover with its sole this far above the walls
HAND_INSET = 0.015           # the hand body's footprint (where it is lower than the wall tops + 2 cm) stays this far inside
                             # the walls' inner faces at the placement pose
FINGER_INSET = 0.004         # same for the finger bodies (the outer finger of a heel or toe pinch sits 27 mm beyond the
                             # heel or the toe tip: only the shoe's end away from the end wall leaves room for it)
TILT_MAX_DEG = 10.0          # a shoe held so tilted that its upright placement needs the tool tilted more than this is not
                             # lowered into the box (the 0.2 m wide hand would dip onto a wall top): it is set down outside
HAND_BOX_TOOL = (np.array([-0.032, -0.106, -0.131]), np.array([0.032, 0.103, -0.0325]))   # left/hand AABB in the tcp frame
FINGER_Y_IN, FINGER_Y_OUT, FINGER_HALF_X, FINGER_Z = -0.0015, 0.0267, 0.0105, (-0.0467, 0.0098)   # a finger body, tcp
                             # frame: closing-axis extent relative to its pad face, half-width across the pads, z range
PUSH_BAD_LAG = 0.040         # a push stroke whose settled lag (0.5-1.0 s into the stroke) exceeds this runs on a bad branch
PUSH_SPLIT = 0.25            # a box farther than this from the outfeed line is pushed in two strokes (re-approach between)


# ---------------------------------------------------------------------------------------------------- small helpers
def _log(rig, msg):
    rig.log("[box] t=%.1f %s" % (rig.time, msg))


def _unit(v):
    v = np.asarray(v, float)
    return v / max(1e-12, float(np.linalg.norm(v)))


def _yaw_of(v):
    return math.atan2(float(v[1]), float(v[0]))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot_err(Rg, R):
    return float(np.linalg.norm(orientation_error(Rg, R)))


def _tilt_deg(R):
    """Angle (deg) between the tool's approach axis and straight down."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(np.asarray(R)[2, 2])))))


def _arm_box_force(rig):
    """(largest normal force (N), 'body|body') of a left-arm body (links, hand, fingers, pads) on the carton or the lid."""
    import mujoco
    m, d = rig.m, rig.d
    cache = getattr(rig, "_arm_box_cache", None)
    if cache is None:
        left = set(b for b in range(m.nbody) if m.body(b).name.startswith("left/"))
        box = set(b for b in range(m.nbody) if m.body(b).name.startswith(("box", "lid")))
        cache = (left, box, np.zeros(6))
        rig._arm_box_cache = cache
    left, box, f6 = cache
    best, what = 0.0, None
    for k in range(d.ncon):
        c = d.contact[k]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in left and b2 in box) or (b2 in left and b1 in box):
            mujoco.mj_contactForce(m, d, k, f6)
            f = abs(float(f6[0]))
            if f > best:
                best, what = f, "%s|%s" % (m.body(b1).name, m.body(b2).name)
    return best, what


def _box_corners(lo, hi):
    return [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def _footprint_margins(rig, p_tcp, R_tcp, width):
    """(hand, fingers): the smallest inward distance (m) from the carton's inner wall faces of the hand body's and the
    two finger bodies' corners (fingers opened to `width`) at the tcp pose, counting only corners lower than 2 cm above
    the wall tops (inf when none is that low).  Negative: over or across a wall."""
    L, W, H, t = _carton(rig)
    _, _, st = _floor_frame(rig)
    top = t + H + 0.02
    p_tcp = np.asarray(p_tcp, float)
    R_tcp = np.asarray(R_tcp, float)

    def margin(pts):
        best = float("inf")
        for pt in pts:
            pf = st["_to_floor"](p_tcp + R_tcp @ pt)
            if float(pf[2]) < top:
                best = min(best, L / 2 - abs(float(pf[0])), W / 2 - abs(float(pf[1])))
        return best

    hand = margin(_box_corners(*HAND_BOX_TOOL))
    fing = float("inf")
    for s in (-1.0, 1.0):
        ya, yb = s * (width / 2 + FINGER_Y_IN), s * (width / 2 + FINGER_Y_OUT)
        lo = np.array([-FINGER_HALF_X, min(ya, yb), FINGER_Z[0]])
        hi = np.array([FINGER_HALF_X, max(ya, yb), FINGER_Z[1]])
        fing = min(fing, margin(_box_corners(lo, hi)))
    return hand, fing


def _pressed(rig, name, where, ignore=()):
    """Per-tick guard: an arm link pressing something above LINK_STOP_N (pads PAD_STOP_N). On a press the arm is
    re-targeted 5 mm up from where it is (the position controller stops pressing) and the press is returned."""
    hit = link_press(rig, name, ignore=ignore)
    if hit is None:
        return None
    arm = rig.sc.arms[name]
    p, R = arm.tcp_actual()
    away = np.array([0.0, 0.0, 1.0])
    n = getattr(rig, "press_normal", None)
    if n is not None and np.linalg.norm(n) > 0.5:
        away = np.asarray(n, float) + np.array([0.0, 0.0, 0.3])
        away /= np.linalg.norm(away)
    arm.command_cartesian(p + away * 0.008, mat_to_quat(R))
    for _ in range(5):
        rig.tick()
    _log(rig, "%s arm: %s presses %s with %.0f N %s: stopped and backed off" % (name, hit[0], hit[1], hit[2], where))
    rig.box_press = (rig.time, hit)
    return hit


def _glide(rig, name, pos, quat, speed=0.08, tol=0.0015, rot_tol=0.01, timeout=6.0, settle=0.0, stall_s=None, guard=True, ignore=()):
    """Straight-line Cartesian motion of `name`'s tcp at `speed` (the target is streamed every tick), then wait until
    the measured tcp is within `tol` / `rot_tol` of the goal.  Returns the final position error (m).  With `stall_s`,
    the wait ends early once the tcp has not moved 1 mm over `stall_s` seconds (an arm stuck at a joint limit). With
    `guard`, an arm link pressing anything (but bodies named `ignore`) stops the motion (_pressed): the error is then
    returned as 1 m."""
    arm = rig.sc.arms[name]
    if arm.mode != "cartesian":
        arm.set_mode("cartesian")
    p0, _ = arm.tcp_actual()
    p1 = np.asarray(pos, float)
    quat = np.asarray(quat, float)
    n = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / max(speed, 1e-3) / rig.s.tick_s)))
    for i in range(n):
        arm.command_cartesian(p0 + (p1 - p0) * (i + 1) / n, quat)
        rig.tick()
        if guard and _pressed(rig, name, "on a glide to %s" % np.round(p1, 3), ignore):
            return 1.0
    Rg = quat_to_mat(quat)
    hist = []
    n_stall = None if stall_s is None else int(round(stall_s / rig.s.tick_s))
    for _ in range(int(timeout / rig.s.tick_s)):
        rig.tick()
        if guard and _pressed(rig, name, "on a glide to %s" % np.round(p1, 3), ignore):
            return 1.0
        p, R = arm.tcp_actual()
        if float(np.linalg.norm(p1 - p)) <= tol and _rot_err(Rg, R) <= rot_tol:
            break
        if n_stall is not None:
            hist.append(p.copy())
            if len(hist) > n_stall and float(np.linalg.norm(hist[-1] - hist[-n_stall - 1])) < 0.001:
                break
    if settle > 0:
        rig.wait(settle)
    p, _ = arm.tcp_actual()
    return float(np.linalg.norm(p1 - p))


def _ik_any(rig, pos, quat):
    """A joint solution for the left tcp pose, seeded from the current configuration, from home and from a few
    offsets of home (other branches of the arm)."""
    return rig.ik("left", pos, quat, q_ref=rig.sc.arms["left"].q(), all_branches=True)


def _ref_configs(rig, name="left"):
    """Reference configurations the planners seed from: the current one, home, and home shifted onto other branches."""
    arm = rig.sc.arms[name]
    refs = [arm.q(), arm.home_q]
    for off in IK_REF_OFFSETS:
        refs.append(np.clip(arm.home_q + np.asarray(off, float), arm.lower, arm.upper))
    return refs


def _margin(rig, q, name="left"):
    arm = rig.sc.arms[name]
    q = np.asarray(q, float)
    return float(np.min(np.minimum(q - arm.lower, arm.upper - q)))


def _zones(rig):
    """Layout facts read from the scene: the tray's -y edge, the box work spot and the on-tray parking spots."""
    sc = rig.sc
    tx, ty = sc.layout["infeed"]["centre"]
    hx, hy = sc.layout["infeed"]["half"]
    tray_edge_y = float(ty - hy)
    wo = float(sc.lid["dims"]["Wo"]) if sc.lid.get("dims") else 0.20
    box_y = min(BOX_Y_MAX, tray_edge_y - TRAY_CLEAR - wo / 2 - 0.005)
    return {"tray_edge_y": tray_edge_y, "box_x": BOX_X, "box_y": box_y, "tray_centre": (float(tx), float(ty)), "tray_half": (float(hx), float(hy))}


def _joint_path_clear(rig, q_to, steps=48, min_pen=0.001):
    """Whether the STRAIGHT joint-space line from the commanded configuration to q_to (the path _move_joint_line
    executes) is free of contacts of the arm's bodies (kinematic check, every 1/steps of the way)."""
    arm = rig.sc.arms["left"]
    qa = np.asarray(arm.q_cmd, float).copy()
    for lam in np.linspace(0.0, 1.0, steps):
        if arm_collisions(arm, qa + lam * (np.asarray(q_to, float) - qa), min_pen=min_pen):
            return False
    return True


def _move_joint_line(rig, q_to, timeout=6.0, tol=0.01):
    """Joint-space move of the left arm along the STRAIGHT line from the commanded configuration to q_to, streamed at
    90 % of the rate limit of the joint with the longest travel.  (The controller's joint mode clips each joint's step
    separately, so a single joint goal bends the executed path away from the line _joint_path_clear checked: a hand
    swept down to 0.15 m over an erected wall while the checked line stayed clear.)"""
    arm = rig.sc.arms["left"]
    arm.set_mode("joint")
    qa = np.asarray(arm.q_cmd, float).copy()
    q_to = np.clip(np.asarray(q_to, float), arm.lower, arm.upper)
    span = float(np.max(np.abs(q_to - qa)))
    n = max(1, int(math.ceil(span / (0.9 * RATE_LIMIT_RAD_S * rig.s.tick_s))))
    for k in range(n):
        arm.command_joint(qa + (q_to - qa) * (k + 1) / n)
        rig.tick()
        hit = link_press(rig, "left")
        if hit is not None:
            arm.command_joint(arm.q().copy())
            _log(rig, "joint line: %s presses %s with %.0f N, stopped" % hit)
            for _ in range(3):
                rig.tick()
            return False
    for _ in range(int(round(timeout / rig.s.tick_s))):
        if float(np.max(np.abs(arm.q() - q_to))) <= tol:
            return True
        rig.tick()
    return False


def _go_home(rig):
    """Bring the left arm back to its home posture (elbow up, tool down, high above the bench) in joint space, lifting
    the hand first so that the interpolated path stays clear of the bench and the objects."""
    arm = rig.sc.arms["left"]
    for z in (0.30, 0.40):
        _left_up(rig, z)
        if _joint_path_clear(rig, arm.home_q):
            break
    _move_joint_line(rig, arm.home_q, timeout=6.0)


def _reach(rig, pos, quat, timeout=6.0):
    """Move the left tcp to a free-space pose.  The local Cartesian controller (and rig.go's IK, seeded from the
    current configuration) stalls when the pose lies on another branch of the arm; then the arm goes back to its home
    posture first and reaches the pose from there."""
    arm = rig.sc.arms["left"]
    pos = np.asarray(pos, float)
    quat = np.asarray(quat, float)
    if rig.ik("left", pos, quat, q_ref=arm.q(), all_branches=False) is not None:
        if rig.move("left", pos, quat, timeout=timeout):
            return True
    q = _ik_any(rig, pos, quat)
    if q is None:
        _log(rig, "pose %s is out of reach" % np.round(pos, 3))
        return False
    _go_home(rig)
    if rig.move("left", pos, quat, timeout=timeout):
        return True
    # the local controller stalls from home too: travel in joint space only along a path checked against the scene
    # (an unchecked joint move can sweep the hand through the tray)
    q = _ik_any(rig, pos, quat)
    if q is not None and _joint_path_clear(rig, q):
        _move_joint_line(rig, q, timeout=timeout)
        return rig.move("left", pos, quat, timeout=2.0)
    p, _ = rig.tcp("left")
    _log(rig, "reach(%s) failed: no clear joint path (at %s)" % (np.round(pos, 3), np.round(p, 3)))
    return False


def _carry(rig, pos, quat, speed=0.2):
    """Free-space move of the left tcp at a moderate speed (a streamed straight line), with _reach as the fallback
    when the local controller does not get there."""
    e = _glide(rig, "left", pos, quat, speed=speed, tol=0.004, rot_tol=0.03, timeout=3.0)
    if e > 0.01:
        return _reach(rig, pos, quat)
    return True


def _go(rig, name, pos, quat, timeout=6.0, settle=0.0):
    ok = rig.go(name, pos, quat, timeout=timeout, settle=settle)
    if not ok:
        p, _ = rig.tcp(name)
        _log(rig, "go(%s) to %s did not converge (at %s)" % (name, np.round(pos, 3), np.round(p, 3)))
    return ok


def park_hook_arm(rig):
    """Park the hook arm at its home posture, clear of the box area (release a caught lace, lift first when low).
    The left arm is lifted first so that the two arms never press on each other."""
    arm = rig.sc.arms["right"]
    if float(np.max(np.abs(arm.q() - arm.home_q))) < 0.05:
        return True
    if rig.hook_loaded:
        rig.hook(False)
    _left_up(rig, 0.18)
    p, R = rig.tcp("right")
    if p[2] < 0.25:
        rig.move("right", p + np.array([0.0, 0.0, 0.25 - p[2]]), mat_to_quat(R), timeout=3.0)
    ok = rig.move_joint("right", arm.home_q, timeout=8.0)
    p2, _ = rig.tcp("right")
    _log(rig, "hook arm parked (%s) tcp %s t=%.1f" % (ok, np.round(p2, 3), rig.time))
    return ok


_park_right = park_hook_arm


def _upright_high(rig, quat, z=0.20):
    """After a fold the tool lies horizontal: turning it back to `quat` while the tcp travels low lets the 0.2 m wide
    hand sweep the bench, so lift to `z` first and turn there, in place."""
    p, R = rig.tcp("left")
    if _rot_err(quat_to_mat(np.asarray(quat, float)), R) < 0.1 and p[2] >= z - 0.02:
        return
    _left_up(rig, z)
    p, R = rig.tcp("left")
    rig.move("left", np.array([p[0], p[1], max(p[2], z)]), quat, tol_pos=0.01, tol_rot=0.05, timeout=3.0)


def _left_up(rig, z_min=0.18):
    """Raise the left tcp straight up to z_min when it is lower (keeps the orientation)."""
    p, R = rig.tcp("left")
    if p[2] < z_min:
        rig.move("left", np.array([p[0], p[1], z_min]), mat_to_quat(R), timeout=4.0)


def _carton(rig):
    cp = rig.sc.carton["params"]
    return float(cp["L"]), float(cp["W"]), float(cp["H"]), float(cp["thickness"])


def _floor_frame(rig):
    st = rig.sc.carton_state()
    fp = np.array(st["floor_pos"], float)
    Rf = quat_to_mat(np.array(st["floor_quat"], float))
    return fp, Rf, st


def _wall_frame(rig, w, tag="c"):
    """(handle position, edge direction, inner-face normal (up when flat), outward direction) of wall w's handle."""
    sn = [s for s in rig.sc.carton["sites"][w] if s.endswith("_" + tag)][0]
    h = rig.sc.carton_handles()[sn]
    p = np.array(h["pos"], float)
    R = quat_to_mat(np.array(h["quat"], float))
    return p, R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()


def _wall_dirs(rig):
    """World horizontal outward direction of each wall, from the floor frame (valid whatever the fold state)."""
    fp, Rf, _ = _floor_frame(rig)
    return {"front": Rf[:, 0].copy(), "back": -Rf[:, 0], "left": Rf[:, 1].copy(), "right": -Rf[:, 1]}


def _wall_facing(rig, direction):
    """The wall whose outward direction is most aligned with the world `direction`."""
    dirs = _wall_dirs(rig)
    return max(rig.sc.carton["walls"], key=lambda w: float(np.dot(dirs[w], direction)))


def _arc_poses(crease, axis, tcp0, R0, th_end, steps):
    out = []
    for k in range(steps + 1):
        Rr = quat_to_mat(rot_quat(axis, th_end * k / steps))
        out.append((crease + Rr @ (tcp0 - crease), Rr @ R0))
    return out


def _arc_hits(arm, q):
    """Kinematic check of a fold-arc configuration: an arm body (but the hand and the fingers, which hold the folding
    wall) through the lid tray or a shoe (penetration > 1 mm)."""
    for b1, b2, pen in arm_collisions(arm, q):
        mine, other = (b1, b2) if b1.startswith("left/") else (b2, b1)
        if mine in ("left/hand", "left/left_finger", "left/right_finger"):
            continue
        if other.startswith(("lid", "shoe")):
            return True
    return False


def _plan_arc(rig, pre, tcp0, q0, crease, axis, th_end):
    """A joint configuration for the pre-grasp pose `pre` from which the local Cartesian controller can follow the
    descent to tcp0 and the whole fold arc without stalling (checked by IK continuation), or None."""
    return _plan_arc_margin(rig, pre, tcp0, q0, crease, axis, th_end)[0]


def _plan_arc_margin(rig, pre, tcp0, q0, crease, axis, th_end, good_margin=0.5):
    """Like _plan_arc, but returns (q_pre, min joint-limit margin along the whole arc) of the best branch found (the
    search stops at the first branch with margin >= good_margin); (None, -1) when no branch is IK-continuable."""
    arm = rig.sc.arms["left"]
    R0 = quat_to_mat(q0)
    poses = [(np.asarray(pre, float), R0), (tcp0, R0)] + _arc_poses(crease, axis, tcp0, R0, th_end, 8)[1:]
    best = (None, -1.0)
    seen = []
    for q_ref in _ref_configs(rig):
        for off in SEED_OFFSETS:
            qs = np.clip(q_ref + np.asarray(off, float), arm.lower, arm.upper)
            q_pre, ep, er, ok = ik_solve(arm, poses[0][0], poses[0][1], q0=qs)
            if not ok or arm_collisions(arm, q_pre):
                continue
            if any(float(np.max(np.abs(q_pre - qq))) < 0.05 for qq in seen):
                continue
            seen.append(q_pre)
            q = q_pre
            margin = _margin(rig, q_pre)
            good = True
            for k, (p, R) in enumerate(poses[1:]):
                q2, ep, er, ok = ik_solve(arm, p, R, q0=q, iters=200)
                if not ok or float(np.max(np.abs(q2 - q))) > 1.2:
                    good = False
                    break
                if k >= 1 and _arc_hits(arm, q2):
                    # an arm link through the lid tray, a shoe or the bench on the way (a forearm on the lid at a
                    # 60 deg fold voided a pair): another branch, or the corrective drag
                    good = False
                    break
                q = q2
                margin = min(margin, _margin(rig, q2))
            if good and margin > best[1]:
                best = (q_pre, margin)
                if margin >= good_margin:
                    return best
    return best


def _pitched_quat(edge, up, out, alpha):
    """Tool orientation for a wall grasp: closing axis along the edge, approach = down tilted by `alpha` toward the
    crease (the hand leans outward over the free edge)."""
    z = -math.cos(alpha) * up - math.sin(alpha) * out
    y = _unit(edge)
    x = np.cross(y, z)
    return mat_to_quat(np.column_stack([x, y, z]))


def _pitched_offsets(alpha):
    """(h, u): tcp height above the panel's mid-plane and outward offset of the tcp from the handle for a grasp pitched
    by `alpha`: the fingertip's lowest corner stays >= 1.3 mm above the panel's top face and the handle sits inside the
    pinch box with ATTACH_MARGIN to spare."""
    c, s = math.cos(alpha), math.sin(alpha)
    h = 0.0015 + 0.0013 + FINGER_TIP_Z * c + FINGER_TIP_X * s
    if s < 1e-6:
        return h, 0.0
    lim = PINCH_Z_HI - ATTACH_MARGIN
    u_max = (lim - h * c) / s               # z_t = h c + u s <= lim
    u_min = (h * s - lim) / c               # x_t = u c - h s >= -lim
    if u_min > u_max:                       # cannot satisfy both with this clearance: split the difference
        return h, 0.5 * (u_min + u_max)
    return h, float(np.clip(0.0, u_min, u_max))


def _reconfigure(rig, pre, q0, q_pre):
    """Bring the arm onto the branch of `q_pre` (the configuration for the pre-grasp pose) without sweeping the hand
    through the bench: rise to a high pose on the same branch (IK continuation from q_pre), travel there in joint
    space when the interpolated path is collision-free, then descend in Cartesian mode."""
    arm = rig.sc.arms["left"]
    R0 = quat_to_mat(q0)
    high = np.array([pre[0], pre[1], max(pre[2] + 0.15, 0.30)])
    q_high, ep, er, ok = ik_solve(arm, high, R0, q0=q_pre, iters=300)
    if not ok or arm_collisions(arm, q_high):
        q_high = q_pre
        high = np.asarray(pre, float)
    _left_up(rig, 0.32)                                   # well above a closed lid (top at ~0.11) before any joint move
    if not _joint_path_clear(rig, q_high):
        _go_home(rig)
        if not _joint_path_clear(rig, q_high):
            _log(rig, "joint path to the planned branch is not clear, going there in Cartesian mode")
            rig.go("left", high, q0, timeout=5.0)
            return
    _move_joint_line(rig, q_high, timeout=6.0)
    rig.move("left", high, q0, timeout=2.0)


def _wait_until(rig, pred, timeout):
    for _ in range(int(timeout / rig.s.tick_s)):
        rig.tick()
        if pred():
            return True
    return False


def _handle_in_tool(rig, p):
    pa, Ra = rig.tcp("left")
    return Ra.T @ (np.asarray(p, float) - pa)


def _grasp(rig, label, handle_pos=None, timeout=2.5):
    """Close the fingers until a handle attaches (like Rig.grasp), with a trace of the handle's position in the tool
    frame while closing when nothing attaches."""
    arm = rig.sc.arms["left"]
    p0, _ = arm.tcp_actual()
    if arm.width() < 0.004:
        # already closed (or pushed past zero): open a little first so that the server sees a closing command
        rig.s.command("left", width=min(0.08, arm.width() + 0.02))
        rig.wait(0.4)
    rig.s.command("left", width=0.0)
    trace = []
    for k in range(int(round(timeout / rig.s.tick_s))):
        rig.tick()
        if rig.s.left_holding is not None:
            rig.wait(0.05)
            return rig.s.left_holding[1]
        if handle_pos is not None and k % 25 == 0:
            d = _handle_in_tool(rig, handle_pos)
            trace.append("%.2fs w%.3f (%.1f,%.1f,%.1f)" % (k * rig.s.tick_s, arm.width(), d[0] * 1e3, d[1] * 1e3, d[2] * 1e3))
    p1, _ = arm.tcp_actual()
    hit = rig.sc.find_left_handle()
    _log(rig, "%s: no attach (width now %.4f, closing flag %s, tcp drift %.2f mm, handle in box: %s) trace %s" % (
        label, arm.width(), rig.s._left_closing, np.linalg.norm(p1 - p0) * 1e3, None if hit is None else hit["name"], "; ".join(trace)))
    return None


# ---------------------------------------------------------------------------------------------------- erect_box
def _drag_blank(rig, dx, dy, tag="c"):
    """Drag the blank by (dx, dy) holding its -x wall at handle `tag` with a straight-down grasp.  Pulling a panel away
    from its crease or along the crease cannot fold it, so the whole blank slides along."""
    sc = rig.sc
    if abs(dx) < 0.004 and abs(dy) < 0.004:
        return True
    w = _wall_facing(rig, np.array([-1.0, 0.0, 0.0]))
    hand_line = _zones(rig)["tray_edge_y"] - 0.03         # the hand's +y edge stays below this (tray edge - 3 cm)
    # the handle furthest from the tray (lowest world y); closing axis along the edge (hand long axis along y) when
    # the hand's +y edge then stays below the line, else across the edge (hand long axis along x, 32 mm half-width
    # in y)
    cands = []
    for tg in ("c", "p", "m"):
        ph, _, _, _ = _wall_frame(rig, w, tg)
        cands.append((float(ph[1]), tg))
    cands.sort()
    low_y, low_tag = cands[0]
    p, edge, up, out = _wall_frame(rig, w, tag)
    if float(p[1]) + 0.10 + max(0.0, dy) > hand_line and low_tag != tag:
        tag = low_tag
        p, edge, up, out = _wall_frame(rig, w, tag)
    along = float(p[1]) + 0.10 + max(0.0, dy) <= hand_line
    if not along and float(p[1]) + 0.032 + max(0.0, dy) > hand_line:
        _log(rig, "drag by (%.3f, %.3f) aborted: the hand at handle %s (y %.3f) would reach over the tray line y %.3f" % (dx, dy, tag, p[1], hand_line))
        return False
    if float(p[1]) + max(0.0, dy) > -0.03:
        _log(rig, "drag by (%.3f, %.3f) aborted: the tcp at handle %s would reach y %.3f > -0.03" % (dx, dy, tag, p[1] + max(0.0, dy)))
        return False
    _log(rig, "drag the blank by (%.3f, %.3f) with wall %s handle %s, hand %s the edge (handle y %.3f, line %.3f)" % (
        dx, dy, w, tag, "along" if along else "across", p[1], hand_line))
    q = down_quat(_yaw_of(edge) if along else _yaw_of(edge) + math.pi / 2)
    h = 0.0015 + 0.0011 + FINGER_TIP_Z           # straight down: fingertips 1.1 mm above the panel top
    rig.gripper(0.03, wait=False)
    _upright_high(rig, q)
    _reach(rig, p + up * 0.08, q)
    _glide(rig, "left", p + up * h, q, speed=0.08, tol=0.001, settle=0.15)
    name = _grasp(rig, "drag grasp", p)
    if name is None:
        _log(rig, "drag grasp missed (handle in tool frame %s mm)" % np.round(_handle_in_tool(rig, p) * 1e3, 1))
        rig.release(0.2)
        _left_up(rig, 0.12)
        return False
    pa, _ = rig.tcp("left")
    _glide(rig, "left", pa + np.array([dx, dy, 0.0]), q, speed=0.06, tol=0.002, settle=0.3)
    rig.release(0.3)
    _left_up(rig, 0.12)
    fp, _, _ = _floor_frame(rig)
    _log(rig, "blank now at %s" % np.round(fp, 3))
    return True


def _edge_y(rig):
    """World y of the blank's +y free edge (when that wall is flat)."""
    w_py = _wall_facing(rig, np.array([0.0, 1.0, 0.0]))
    p_py, _, _, out_py = _wall_frame(rig, w_py)
    return float(p_py[1] + 0.012 * out_py[1])


def _lid_gap(rig, w, tag):
    """Bench-plane clearance (m) between the lid tray's footprint and the hand that grasps wall w at handle `tag` (0.2 m
    along the edge, leaning out over the free edge)."""
    lp, lq = rig.sc.lid_pose()
    Rl = quat_to_mat(np.asarray(lq, float))
    lo, wo = float(rig.sc.lid["dims"]["Lo"]) / 2, float(rig.sc.lid["dims"]["Wo"]) / 2
    p, edge, up, out = _wall_frame(rig, w, tag)
    gap = float("inf")
    for a in (-0.105, -0.05, 0.0, 0.05, 0.105):
        for b in (0.0, 0.04, 0.08):
            q = Rl.T @ (p + edge * a + out * b - np.asarray(lp, float))
            gap = min(gap, float(math.hypot(max(abs(q[0]) - lo, 0.0), max(abs(q[1]) - wo, 0.0))))
    return gap


def _wall_tag(rig, w, avoid=()):
    """The handle of wall w to fold it by: the centre, unless the hand there comes within 5 cm of the lid tray lying on
    the infeed tray (the forearm or the hand on the lid at the +x wall's pitched grasp: a voided pair, or with the
    guards a wall never folded); then the handle (centre, +-30 % along the edge) farthest from the lid."""
    tags = [t for t in ("c", "p", "m") if t not in avoid] or ["c"]
    gaps = {t: _lid_gap(rig, w, t) for t in tags}
    if "c" in gaps and gaps["c"] >= 0.05:
        return "c"
    best = max(tags, key=lambda t: gaps[t])
    if best != "c":
        _log(rig, "wall %s: the hand at the centre handle would come %.0f mm from the lid tray; folding it by handle %s (%.0f mm)" % (
            w, 1000 * gaps.get("c", 0.0), best, 1000 * gaps[best]))
    return best


def _fold_wall(rig, w, alpha_deg, q_pre=None, tag="c"):
    """Grasp wall w at its centre handle with a grasp pitched by alpha_deg, swing it about the crease to ~91 deg, wait
    for the snap-lock, release and retreat.  Returns True when the wall locked.  `q_pre`: an arc-feasible start
    configuration already planned by _plan_fold (else planned here)."""
    sc = rig.sc
    arm = sc.arms["left"]
    L, W, H, t = _carton(rig)
    alpha = math.radians(alpha_deg)
    p, edge, up, out = _wall_frame(rig, w, tag)
    crease = p - out * (H - 0.012) + up * (t / 2)          # the point of the hinge axis under the handle
    axis = -edge
    if float((quat_to_mat(rot_quat(axis, math.pi / 2)) @ out) @ up) < 0:
        axis = edge
    h, u = _pitched_offsets(alpha)
    q0 = _pitched_quat(edge, up, out, alpha)
    tcp0 = p + up * h + out * u
    rig.gripper(0.03, wait=False)
    pre = tcp0 + np.array([0.0, 0.0, 0.07])
    th_end = math.radians(90.5)
    if q_pre is None:
        q_pre = _plan_arc(rig, pre, tcp0, q0, crease, axis, th_end)
    if q_pre is not None:
        if float(np.max(np.abs(q_pre - arm.q()))) > 0.15:
            _reconfigure(rig, pre, q0, q_pre)
        _left_up(rig, APPROACH_Z)                      # lift-and-descend: never travel to a handle at wall height
        rig.move("left", pre, q0, timeout=2.0)
    else:
        _log(rig, "wall %s: no arc-feasible start configuration found, trying anyway" % w)
        _left_up(rig, APPROACH_Z)
        _reach(rig, pre, q0)
    name = None
    corr = np.zeros(3)
    for attempt in range(3):
        p, edge, up, out = _wall_frame(rig, w, tag)            # re-read: the blank may have been nudged
        crease = p - out * (H - 0.012) + up * (t / 2)
        q0 = _pitched_quat(edge, up, out, alpha)
        tcp0 = p + up * h + out * u + corr
        e = _glide(rig, "left", tcp0, q0, speed=0.06, tol=0.0008, rot_tol=0.006, timeout=4.0, settle=0.15, stall_s=0.8)
        if e >= 0.5:
            # the descent to the handle pressed something (the lid tray beside the wall): this handle is no good
            _glide(rig, "left", tcp0 + np.array([0.0, 0.0, 0.05]), q0, speed=0.06, timeout=2.0, guard=False)
            _left_up(rig, APPROACH_Z)
            return False
        d = _handle_in_tool(rig, p)
        name = None if e > 0.006 else _grasp(rig, "wall %s" % w, p)
        _log(rig, "wall %-5s alpha %2.0f: tcp err %.2f mm, handle in tool frame %s mm -> %s" % (w, alpha_deg, e * 1e3, np.round(d * 1e3, 1), name))
        if name is not None:
            break
        d = _handle_in_tool(rig, p)
        pa, Ra = arm.tcp_actual()
        lim = PINCH_Z_HI - ATTACH_MARGIN - 0.0005
        dx = d[0] + lim if d[0] < -lim else (d[0] - lim if d[0] > lim else 0.0)
        dz = d[2] - lim if d[2] > lim else 0.0
        rig.release(0.2)
        _glide(rig, "left", tcp0 + np.array([0.0, 0.0, 0.03]), q0, speed=0.08, timeout=2.0)
        rig.gripper(0.03)
        if e > 0.006 or abs(dx) > 0.006:
            # the arm is stuck at a joint limit on this branch: re-plan the arc from home and go onto that branch high
            # above the bench instead of chasing the offset
            _log(rig, "wall %s: grasp missed by %.1f mm (tool frame dx %.1f): re-planning the branch" % (w, e * 1e3, dx * 1e3))
            _go_home(rig)
            q_pre = _plan_arc(rig, pre, tcp0 - corr, q0, crease, axis, th_end)
            corr = np.zeros(3)
            if q_pre is not None:
                _reconfigure(rig, pre, q0, q_pre)
            _left_up(rig, APPROACH_Z)
            rig.move("left", pre, q0, timeout=2.0)
        else:
            # the handle missed the pinch box (|x| <= 12 mm, z <= 12 mm) by a little because the tcp did not converge
            # exactly: shift the target so that the handle lands at (+-10.5, *, <= 10.5) mm
            corr = corr + Ra @ np.array([dx, 0.0, dz])
            _log(rig, "wall %s: grasp missed, correcting the target by %s mm (tool frame dx %.1f dz %.1f)" % (w, np.round(corr * 1e3, 1), dx * 1e3, dz * 1e3))
    if name is None:
        _left_up(rig, 0.15)
        return False
    # swing about the crease from the actual grasp pose
    pa, Ra = arm.tcp_actual()
    qa = mat_to_quat(Ra)
    rate = math.radians(40.0)
    n = int(th_end / rate / rig.s.tick_s)
    stalled = False
    for i in range(n):
        th = th_end * (i + 1) / n
        qr = rot_quat(axis, th)
        tgt = crease + quat_to_mat(qr) @ (pa - crease)
        arm.command_cartesian(tgt, quat_mul(qr, qa))
        rig.tick()
        hit = link_press(rig, "left")
        if hit is not None:
            stalled = True
            _log(rig, "wall %s: arc stopped at %.0f deg: %s presses %s with %.0f N" % (w, math.degrees(th), hit[0], hit[1], hit[2]))
            break
        if i % 10 == 0:
            pc, _ = arm.tcp_actual()
            if float(np.linalg.norm(pc - tgt)) > 0.05:
                stalled = True
                _log(rig, "wall %s: arc stalled at %.0f deg (tcp lag %.0f mm)" % (w, math.degrees(th), np.linalg.norm(pc - tgt) * 1e3))
                break
    # a stalled arm drags the whole blank through the weld: let go at once instead of waiting for the lock
    locked = False if stalled else _wait_until(rig, lambda: sc.carton_state()["locked"][w], 2.5)
    st = sc.carton_state()
    pc, Rc = arm.tcp_actual()
    _log(rig, "wall %-5s at %.1f deg, locked %s, tcp err %.1f mm, floor %s (t=%.1f)" % (
        w, st["wall_angles_deg"][w], locked, np.linalg.norm(pc - tgt) * 1e3, np.round(st["floor_pos"], 3), rig.time))
    pb, Rb = arm.tcp_actual()
    if stalled:
        arm.command_cartesian(pb, mat_to_quat(Rb))
    rig.release(0.3)
    _glide(rig, "left", pb - Rb[:, 2] * 0.04, mat_to_quat(Rb), speed=0.12, tol=0.004, timeout=2.0)
    p2, R2 = arm.tcp_actual()
    _glide(rig, "left", np.array([p2[0], p2[1], max(p2[2], 0.0) + 0.12]), mat_to_quat(R2), speed=0.2, tol=0.005, timeout=2.0)
    return bool(locked)


def _tray_spot(rig, k):
    """The infeed tray spot where shoe k of a pair is delivered (a free place to park a shoe)."""
    tx, ty = rig.sc.layout["infeed"]["centre"]
    return np.array([tx - 0.115 + 0.23 * k, ty + 0.10, float(rig.sc.infeed_top_z) + 0.004])


def _clear_work_zone(rig):
    """Shoes left on the bench between the arm and the blank (the lacing skills unseat them beside the cradle, the
    second one ~0.3 m from the blank) stand where the dragged blank and the pitched hand of its -x wall go: park such a
    shoe on its infeed-tray spot first."""
    sc = rig.sc
    rec = rig.s.work.current
    if rec is None:
        return
    z = _zones(rig)
    tx, ty = z["tray_centre"]
    hx, hy = z["tray_half"]
    cradle = np.array(getattr(sc, "cradle_pos", None) or list(sc.layout["cradle"]["centre"]) + [0.0], float)[:2]
    for k, i in enumerate(rec["shoe_ids"]):
        c = sc.shoe_centre(i)
        on_tray = abs(c[0] - tx) <= hx and abs(c[1] - ty) <= hy
        in_zone = (z["box_x"] - 0.40 <= c[0] <= z["box_x"] + 0.45) and (z["box_y"] - 0.30 <= c[1] <= z["box_y"] + 0.17)
        beside_cradle = float(np.linalg.norm(np.asarray(c[:2], float) - cradle)) < 0.22
        if on_tray or not in_zone or beside_cradle or c[2] < -0.1:
            continue
        _log(rig, "shoe %d at %s is in the box work zone: parking it on the tray" % (i, np.round(c, 3)))
        if rig.s.clamp_engaged == i:
            rig.release_clamp()
        if not sc.upright(i, max_deg=15.0):
            _right_shoe(rig, i)
        spot = _tray_spot(rig, k)
        if not _grasp_shoe(rig, i, np.array([1.0, 0.0, 0.0]), spot):
            continue
        _place_shoe(rig, i, spot, np.eye(3))
        _left_up(rig, 0.2)


def erect_box(rig):
    """Fold the four walls of the blank until rig.sc.carton_state()['erected'].

    Sequence: (1) if the blank's +y free edge lies too close to the infeed tray, drag it -y first; (2) fold the +y wall
    with a slightly pitched grasp where the blank arrived (its pitched hand leans toward the arm and converges better
    further away from it); (3) drag the blank toward the arm by a side handle of the -x wall to the box work spot
    (BOX_X, box_y), where the far walls are within reach of a strongly pitched grasp and the closed box stays clear of
    the tray on its way to the outfeed; (4) fold the -y wall, then the two short walls, whose hand at 90 deg must clear
    the erected long walls (hence the strong pitch).  A wall whose fold arc is not plannable where the blank stands is
    retried after a small corrective drag toward the arm."""
    sc = rig.sc
    _left_up(rig, 0.18)
    park_hook_arm(rig)
    st = sc.carton_state()
    if st["erected"]:
        return True
    _clear_work_zone(rig)
    z = _zones(rig)
    walls = {k: _wall_facing(rig, np.array(v, float)) for k, v in (("py", (0, 1, 0)), ("my", (0, -1, 0)), ("px", (1, 0, 0)), ("mx", (-1, 0, 0)))}
    _log(rig, "erect: walls %s, floor %s, work spot (%.3f, %.3f), tray edge y %.3f" % (
        walls, np.round(st["floor_pos"], 3), z["box_x"], z["box_y"], z["tray_edge_y"]))

    def fold(w, alpha, q_pre=None):
        avoid = []
        for attempt in range(3):
            if sc.carton_state()["locked"][w]:
                return True
            tag = _wall_tag(rig, w, avoid)
            rig.box_press = None
            if _fold_wall(rig, w, alpha if attempt < 2 else alpha + 8.0, q_pre=q_pre if (attempt == 0 and tag == "c") else None, tag=tag):
                return True
            press = getattr(rig, "box_press", None)
            if press is not None and "lid" in press[1][1]:
                if not getattr(rig, "lid_shifted", False) and _shift_lid_away(rig):
                    continue                               # the lid slid away from the box: the same wall again
                avoid.append(tag)                          # that handle put the hand on the lid: another one
                if w in (walls["px"], walls["mx"], walls["py"]):
                    # a short wall or the wall facing the tray: its hand and the forearm come in from the tray's side,
                    # over the lid lying near the box; the blank 3 cm further from the tray
                    fp, _, _ = _floor_frame(rig)
                    if w == walls["py"]:
                        dy = -0.03                         # the first fold, where the blank was delivered
                    else:
                        dy = min(0.0, max(LID_DRAG_Y_MIN, float(fp[1]) - 0.03) - float(fp[1]))
                    if dy < -0.004:
                        _log(rig, "wall %s: the hand met the lid tray; dragging the blank by %.3f m in y" % (w, dy))
                        _drag_blank(rig, 0.0, dy, "c")
            _log(rig, "wall %s not locked (attempt %d)" % (w, attempt + 1))
        return False

    # (1) the flat blank must not reach the tray
    edge_max = z["tray_edge_y"] - EDGE_MARGIN
    _drag_blank(rig, 0.0, min(0.0, edge_max - _edge_y(rig)), "c")
    # (2) the +y wall where the blank stands
    fold(walls["py"], 15.0)
    # (3) toward the arm: the side handle of the -x wall that lies away from the (now standing) +y wall
    def side_tag():
        cands = []
        for tg in ("p", "m"):
            ph, _, _, _ = _wall_frame(rig, walls["mx"], tg)
            cands.append((float(ph[1]), tg))
        return min(cands)[1]

    fp, Rf, _ = _floor_frame(rig)
    _drag_blank(rig, z["box_x"] - float(fp[0]), z["box_y"] - float(fp[1]), side_tag())
    fp, Rf, _ = _floor_frame(rig)
    if abs(z["box_x"] - float(fp[0])) > 0.02 or abs(z["box_y"] - float(fp[1])) > 0.02:
        _log(rig, "drag missed the work spot by (%.3f, %.3f): one corrective drag" % (z["box_x"] - fp[0], z["box_y"] - fp[1]))
        _drag_blank(rig, z["box_x"] - float(fp[0]), z["box_y"] - float(fp[1]), side_tag())
    # (4) the far long wall, then the short walls; a corrective drag when a fold arc cannot be planned here
    for key, alpha in (("my", 45.0), ("px", 60.0), ("mx", 60.0)):
        w = walls[key]
        if sc.carton_state()["locked"][w]:
            continue
        q_pre = None
        if key == "mx":
            fp, Rf, _ = _floor_frame(rig)
            if float(fp[0]) > MX_FOLD_X_MAX:
                _log(rig, "wall %s: floor x %.3f > %.3f, dragging toward the arm before the fold" % (w, fp[0], MX_FOLD_X_MAX))
                _drag_blank(rig, MX_FOLD_X_MAX - 0.005 - float(fp[0]), 0.0, side_tag())
        for extra in range(MAX_CORRECTIVE_DRAGS + 1):
            if key != "my":
                # the short walls' hand (0.2 m along the edge) must never reach over the tray's -y edge (y = 0.02) nor
                # the lid tray lying there: keep the floor at y <= FLOOR_Y_SHORT_MAX (the -y fold and the corrective
                # drags push a blank toward +y)
                fp, Rf, _ = _floor_frame(rig)
                if float(fp[1]) > FLOOR_Y_SHORT_MAX:
                    _log(rig, "wall %s: floor y %.3f too close to the tray, dragging back to %.3f" % (w, fp[1], FLOOR_Y_SHORT_MAX - 0.006))
                    _drag_blank(rig, 0.0, FLOOR_Y_SHORT_MAX - 0.006 - float(fp[1]), side_tag())
            q_pre, margin = _plan_fold(rig, w, alpha)
            if key == "my" and (q_pre is None or margin < ARC_MARGIN_MIN):
                # a lower pitch only for the long wall: a short wall folded at 50 deg brings the hand onto the standing
                # long walls' tops at the end of its arc
                q_alt, m_alt = _plan_fold(rig, w, alpha - 10.0)
                if q_alt is not None and m_alt > max(margin, ARC_MARGIN_MIN - 1e-9):
                    _log(rig, "wall %s: pitch %.0f deg plans with margin %.2f rad (%.0f deg: %s)" % (w, alpha - 10.0, m_alt, alpha, "none" if q_pre is None else "%.2f" % margin))
                    q_pre, margin, alpha = q_alt, m_alt, alpha - 10.0
            if q_pre is not None and (key == "my" or margin >= ARC_MARGIN_MIN):
                break
            if extra == MAX_CORRECTIVE_DRAGS:
                break                                      # out of drags: fold with the last plan
            fp, Rf, _ = _floor_frame(rig)
            if key != "my" and float(fp[0]) <= MX_FOLD_X_MAX + 0.001:
                break                                      # already at the -x wall's spot: no closer (the forearm would
                                                           # reach over the lid tray); fold with the last plan
            # a short wall out of reach: one drag straight to the spot the -x wall needs anyway (saves the second drag)
            dx, dy = (min(-0.02, MX_FOLD_X_MAX - 0.005 - float(fp[0])), 0.0) if key != "my" else (0.0, 0.012)
            _log(rig, "wall %s: %s with the floor at %s, dragging by (%.3f, %.3f)" % (
                w, "no plannable fold arc" if q_pre is None else "fold arc margin %.2f rad < %.2f" % (margin, ARC_MARGIN_MIN), np.round(fp[:2], 3), dx, dy))
            if not _drag_blank(rig, dx, dy, side_tag()):
                break
        fold(w, alpha, q_pre)
    st = sc.carton_state()
    _log(rig, "erected %s angles %s floor %s t=%.1f" % (st["erected"], {k: round(v, 1) for k, v in st["wall_angles_deg"].items()}, np.round(st["floor_pos"], 3), rig.time))
    _left_up(rig, 0.2)
    return bool(st["erected"])


LID_DRAG_Y_MIN = -0.175     # a short-wall fold stopped on the lid tray moves the blank away from the tray, not beyond this
LID_SHIFT_MAX = 0.06        # a lid tray in the way of a wall fold slides this far toward +y on the infeed tray, or back
                            # onto the tray when it lies off it (LID_TRAY_IN inside the tray's near edge) ...
LID_SHOE_CLEAR = 0.02       # ... keeping this much from the shoes lying there
LID_TRAY_IN = 0.012         # the cell delivers the lid with its near wall over the tray's near rim: of 128 seeds, 45 lids
                            # rest tilted on the rim and 4 are pushed off the tray at reset, 12 cm toward the blank
                            # (seeds 1014, 1034, 1039 and 1046)
LID_LIFT = 0.035            # carried this high over the tray's rim (18 mm)


def _shift_lid_away(rig):
    """The lid tray lies so close to the blank that a wall fold's hand met it (a lid jittered toward the box or pushed
    off the infeed tray): grasp it at the middle of its wall that faces the arm across the tray's length (away from the
    shoes), lift it over the tray's rim and slide it toward +y by LID_SHIFT_MAX, or more when that leaves it off the
    tray (until its near edge is LID_TRAY_IN inside), but never closer than LID_SHOE_CLEAR to the shoes lying there,
    set it down, let go. Returns True when it moved by 1.5 cm or more."""
    sc = rig.sc
    rig.lid_shifted = True
    lp, lq = sc.lid_pose()
    lp = np.asarray(lp, float)
    Rl = quat_to_mat(np.asarray(lq, float))
    lo, wo = float(sc.lid["dims"]["Lo"]) / 2, float(sc.lid["dims"]["Wo"]) / 2
    lid_max_y = max(float((lp + Rl @ np.array([sx * lo, sy * wo, 0.0]))[1]) for sx in (-1, 1) for sy in (-1, 1))
    shoe_min_y = float("inf")
    rec = rig.s.work.current
    for i in (rec["shoe_ids"] if rec is not None else (0, 1)):
        p, R = _shoe_frame(rig, i)
        d = sc.shoes[i]["dims"]
        for sx in (-1, 1):
            for sy in (-1, 1):
                q = p + R @ np.array([sx * d["L"] / 2, sy * d["W"] / 2, 0.0])
                if abs(float(q[0]) - float(lp[0])) < lo + 0.05:
                    shoe_min_y = min(shoe_min_y, float(q[1]))
    corners = [lp + Rl @ np.array([sx * lo, sy * wo, sz], float) for sx in (-1, 1) for sy in (-1, 1)
               for sz in (0.0, -float(sc.lid["dims"]["wall_h"]))]
    lid_min_y = min(float(c[1]) for c in corners)
    lid_low_z = min(float(c[2]) for c in corners)
    cy, hy = sc.layout["infeed"]["centre"][1], sc.layout["infeed"]["half"][1]
    dy_tray = (cy - hy + LID_TRAY_IN) - lid_min_y            # back onto the tray: the near edge inside its rim
    dy = min(max(LID_SHIFT_MAX, dy_tray), shoe_min_y - LID_SHOE_CLEAR - lid_max_y)
    # where it lands: on the tray plate when its near edge ends inside the rim, else on the bench
    land_z = (sc.infeed_top_z if lid_min_y + dy > cy - hy + 0.004 else 0.0) + 0.002
    dz_land = land_z - lid_low_z
    if dy < 0.015:
        _log(rig, "lid: in the way of a wall fold, but no room to slide it away (%.0f mm)" % (1000 * max(dy, 0.0)))
        return False
    hs = sc.carton_handles()
    cands = []
    for key, sn in sc.lid["sites"].items():
        hp = np.array(hs[sn]["pos"], float)
        if abs(float(hp[1]) - float(lp[1])) < 0.5 * wo:          # a wall across the tray's length (its middle at the lid's y)
            cands.append((float(np.linalg.norm(hp[:2] - LEFT_BASE[:2])), key, sn))
    cands.sort()
    _log(rig, "lid: in the way of a wall fold, sliding it %.0f mm toward +y on the tray (handles %s)" % (1000 * dy, [c[1] for c in cands]))
    for _, key, sn in cands:
        h = hs[sn]
        p = np.array(h["pos"], float)
        R = quat_to_mat(np.array(h["quat"], float))
        normal = _unit([R[0, 1], R[1, 1], 0.0])
        for yaw in (_yaw_of(normal), _yaw_of(normal) + math.pi):
            q = down_quat(yaw)
            tcp = p + np.array([0.0, 0.0, PINCH_Z_HI - ATTACH_MARGIN - 0.0005])
            if _ik_any(rig, tcp, q) is None or _ik_any(rig, tcp + np.array([0.0, 0.03, LID_LIFT]) + np.array([0.0, dy, 0.0]), q) is None:
                continue
            rig.gripper(0.03, wait=False)
            _left_up(rig, APPROACH_Z)
            _reach(rig, np.array([tcp[0], tcp[1], APPROACH_Z]), q)
            if _glide(rig, "left", tcp, q, speed=0.1, tol=0.001, timeout=4.0, settle=0.1, stall_s=0.8) > 0.004:
                _left_up(rig, APPROACH_Z)
                continue
            if _grasp(rig, "lid %s (to slide it)" % key, p) is None:
                rig.release(0.2)
                _left_up(rig, APPROACH_Z)
                continue
            _glide(rig, "left", tcp + np.array([0.0, 0.0, LID_LIFT]), q, speed=0.04, tol=0.002, timeout=3.0)
            _glide(rig, "left", tcp + np.array([0.0, dy, LID_LIFT]), q, speed=0.03, tol=0.002, timeout=3.0 + dy / 0.03)
            _glide(rig, "left", tcp + np.array([0.0, dy, max(dz_land, 0.0) + 0.002]), q, speed=0.02, tol=0.0015, timeout=3.0, settle=0.2)
            rig.release(0.4)
            _glide(rig, "left", tcp + np.array([0.0, dy, 0.10]), q, speed=0.15, tol=0.005, timeout=3.0)
            _left_up(rig, APPROACH_Z)
            lp2, _ = sc.lid_pose()
            moved = float(lp2[1]) - float(lp[1])
            _log(rig, "lid slid by %.0f mm in y to %s (t=%.1f)" % (1000 * moved, np.round(lp2, 3), rig.time))
            return moved >= 0.015
    _log(rig, "lid: could not grasp it to slide it away")
    return False


def _plan_fold(rig, w, alpha_deg, tag=None):
    """(start configuration, min joint margin along the arc) for the fold of wall w with a grasp pitched by alpha_deg,
    (None, -1) when no IK-continuable start configuration exists."""
    L, W, H, t = _carton(rig)
    alpha = math.radians(alpha_deg)
    p, edge, up, out = _wall_frame(rig, w, tag or _wall_tag(rig, w))
    crease = p - out * (H - 0.012) + up * (t / 2)
    axis = -edge
    if float((quat_to_mat(rot_quat(axis, math.pi / 2)) @ out) @ up) < 0:
        axis = edge
    h, u = _pitched_offsets(alpha)
    q0 = _pitched_quat(edge, up, out, alpha)
    tcp0 = p + up * h + out * u
    pre = tcp0 + np.array([0.0, 0.0, 0.07])
    return _plan_arc_margin(rig, pre, tcp0, q0, crease, axis, math.radians(90.5))


# ---------------------------------------------------------------------------------------------------- pack_pair
def _shoe_frame(rig, i):
    p, q = rig.sc.shoe_pose(i)
    return np.array(p, float), quat_to_mat(np.array(q, float))


def _slerp(qa, qb, u):
    qa = np.asarray(qa, float)
    qb = np.asarray(qb, float)
    d = float(np.dot(qa, qb))
    if d < 0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + u * (qb - qa)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - u) * th) * qa + math.sin(u * th) * qb) / math.sin(th)


def _rotz(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _right_shoe(rig, i):
    """A shoe lying on its side (or tilted on something): grasp its heel counter or toe cap from above (the attach rule
    is geometric, a sideways handle qualifies), lift, rotate the hand so that the shoe's sole normal points up, set it
    down where it was and release.  The caller then grasps it the normal way.  (A push cannot tip a shoe: the friction
    at the pushing finger cancels the tipping torque and the shoe just slides.)"""
    sc = rig.sc
    arm = sc.arms["left"]
    p, R = _shoe_frame(rig, i)
    dims = sc.shoes[i]["dims"]
    zs = R[:, 2].copy()
    if float(zs[2]) > math.cos(math.radians(15.0)):
        return True
    h = R[:, 0].copy()
    h[2] = 0.0
    if np.linalg.norm(h) < 0.3:
        h = np.cross([0.0, 0.0, 1.0], zs)
    h = _unit(h)
    q = None
    for key in ("heel_grip", "toe_grip"):
        site = np.array(sc.shoe_handles(i)[key]["pos"], float)
        tcp = site + np.array([0.0, 0.0, SHOE_GRASP_Z])
        if key == "toe_grip":
            # the row-1 lace drops down the flap side just behind the toe cap of a sideways shoe: shift the hand toward
            # the toe tip so that the rear finger stays clear of it (the handle may sit anywhere within half the
            # opening along the closing axis)
            tcp = tcp + h * 0.018
        for yw in (_yaw_of(h), _yaw_of(h) + math.pi):
            qq = down_quat(yw)
            if _ik_any(rig, tcp, qq) is not None and _ik_any(rig, tcp + np.array([0.0, 0.0, 0.10]), qq) is not None:
                q = qq
                break
        if q is not None:
            break
        _log(rig, "shoe %d on its side: %s at %s out of reach" % (i, key, np.round(site, 3)))
    if q is None:
        return False
    _log(rig, "shoe %d on its side (sole normal %s): right it via a %s grasp" % (i, np.round(zs, 2), key))
    rig.gripper(0.05, wait=False)
    _reach(rig, tcp + np.array([0.0, 0.0, 0.10]), q)
    _glide(rig, "left", tcp, q, speed=0.1, tol=0.001, timeout=4.0, settle=0.1)
    name = _grasp(rig, "shoe %d sideways %s" % (i, key), site)
    if name is None:
        rig.release(0.2)
        _left_up(rig, tcp[2] + 0.1)
        return False
    _glide(rig, "left", tcp + np.array([0.0, 0.0, 0.05]), q, speed=0.04, tol=0.003, timeout=3.0)
    _glide(rig, "left", np.array([tcp[0], tcp[1], CARRY_Z]), q, speed=0.15, tol=0.004, timeout=4.0)
    support_z = float(p[2] - dims["W"] / 2) if abs(float(zs[2])) < 0.5 else max(0.0, float(p[2]) - 0.02)
    # set it down a little toward its sole side: the lace tails were laid along the side the shoe's top faced and the
    # righted shoe would otherwise stand on them
    zh = zs.copy()
    zh[2] = 0.0
    shift = -_unit(zh) * 0.03 if np.linalg.norm(zh) > 0.3 else np.zeros(3)
    pos_des = np.array([p[0] + shift[0], p[1] + shift[1], max(support_z, 0.0) + 0.004])
    pa, Ra = arm.tcp_actual()
    ps, Rs = _shoe_frame(rig, i)
    p_rel = Ra.T @ (ps - pa)
    R_rel = Ra.T @ Rs
    # the minimal rotation that brings the shoe's sole normal to world +z, then a free yaw phi
    zs = Rs[:, 2]
    ax = np.cross(zs, [0.0, 0.0, 1.0])
    ang = math.atan2(float(np.linalg.norm(ax)), float(zs[2]))
    R_fix = quat_to_mat(rot_quat(ax, ang)) if np.linalg.norm(ax) > 1e-6 else np.eye(3)
    done = False
    for phi in (0.0, math.pi, math.pi / 2, -math.pi / 2):
        R_t = _rotz(phi) @ R_fix @ Ra
        p_set = pos_des - R_t @ p_rel
        if _ik_any(rig, np.array([pa[0], pa[1], CARRY_Z]), mat_to_quat(R_t)) is None or _ik_any(rig, p_set, mat_to_quat(R_t)) is None:
            continue
        qa, qb = mat_to_quat(arm.tcp_actual()[1]), mat_to_quat(R_t)
        n = int(1.8 / rig.s.tick_s)
        arm.set_mode("cartesian")
        for k in range(n):
            arm.command_cartesian(pa, _slerp(qa, qb, (k + 1) / n))
            rig.tick()
        _glide(rig, "left", pa, qb, speed=0.1, tol=0.003, rot_tol=0.02, timeout=3.0)
        _, Rs2 = _shoe_frame(rig, i)
        if float(Rs2[2, 2]) > math.cos(math.radians(8.0)):
            done = True
            break
        _log(rig, "shoe %d: rotation to upright not reached (sole normal %s), trying another yaw" % (i, np.round(Rs2[:, 2], 2)))
    if not done:
        _log(rig, "shoe %d: no feasible righting pose, putting it back" % i)
        _glide(rig, "left", tcp + np.array([0.0, 0.0, 0.01]), q, speed=0.1, tol=0.003, timeout=4.0)
        rig.release(0.4)
        _left_up(rig, tcp[2] + 0.1)
        return False
    # set it down upright where it was
    _glide(rig, "left", p_set + np.array([0.0, 0.0, 0.03]), qb, speed=0.12, tol=0.002, timeout=5.0)
    _glide(rig, "left", p_set, qb, speed=0.04, tol=0.0015, timeout=3.0, settle=0.1)
    rig.release(0.5)
    _glide(rig, "left", p_set + np.array([0.0, 0.0, 0.12]), qb, speed=0.2, tol=0.005, timeout=3.0)
    rig.wait(0.3)
    ok = sc.upright(i, max_deg=15.0)
    p2, _ = _shoe_frame(rig, i)
    _log(rig, "shoe %d righted: upright %s pos %s t=%.1f" % (i, ok, np.round(p2, 3), rig.time))
    return ok


def _sweep_contacts(rig, q, held, p_rel, R_rel, obstacles, min_pen=0.002):
    """Contacts (as 'body1|body2' strings) of the hand, the fingers or the held shoe (carried at the pose implied by
    the tcp at q) against the obstacle bodies, on the scratch data at configuration q."""
    import mujoco
    m, sc = rig.m, rig.sc
    arm = sc.arms["left"]
    sdata = arm.scratch
    sdata.qpos[:] = rig.d.qpos
    sdata.qpos[arm.qadr] = q
    mujoco.mj_kinematics(m, sdata)
    p = sdata.site_xpos[arm.tcp].copy()
    R = sdata.site_xmat[arm.tcp].reshape(3, 3).copy()
    if held is not None:
        b = sc.shoe_body[held]
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        sdata.qpos[adr:adr + 3] = p + R @ p_rel
        sdata.qpos[adr + 3:adr + 7] = mat_to_quat(R @ R_rel)
    mujoco.mj_forward(m, sdata)
    movers = {m.body("left/hand").id, m.body("left/left_finger").id, m.body("left/right_finger").id}
    if held is not None:
        movers.add(sc.shoe_body[held])
    out = []
    for i in range(sdata.ncon):
        c = sdata.contact[i]
        if -c.dist < min_pen:
            continue
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in movers and b2 in obstacles) or (b2 in movers and b1 in obstacles):
            out.append("%s|%s" % (m.body(b1).name, m.body(b2).name))
    return out


def _carry_obstacles(rig, held):
    """Body ids the carried shoe and the hand must not touch: the carton (floor, walls, lid tail), the lid tray and the
    other shoe."""
    m, sc = rig.m, rig.sc
    ids = set()
    for b in range(m.nbody):
        name = m.body(b).name
        if name.startswith("box") or name.startswith("lid"):
            ids.add(b)
    for i, b in enumerate(sc.shoe_body):
        if i != held:
            ids.add(int(b))
    return ids


def _dry_run(rig, q0, legs, settle_ticks=150, held=None, p_rel=None, R_rel=None, obstacles=None, check_every=5, check_from=1):
    """Stream the Cartesian legs [(pos, R, speed), ...] through the cell's own controller model (ArmController.ik_step
    with the joint rate limit, from q0 on a copy of the state) and return (tcp error (m) at the end of each leg,
    first blocking contact or None).  With `held`, the shoe is swept along at its grasp pose relative to the tcp
    (p_rel, R_rel) from leg index `check_from` on, and any contact of the hand, the fingers or the shoe with the
    `obstacles` bodies (penetration > 2 mm) blocks the path."""
    arm = rig.sc.arms["left"]
    q = np.asarray(q0, float).copy()
    dq_max = RATE_LIMIT_RAD_S * rig.s.tick_s
    p, _ = arm.fk_cmd(q)
    errs = []
    blocked = None
    for li, (pos, R, speed) in enumerate(legs):
        pos = np.asarray(pos, float)
        n = max(1, int(math.ceil(float(np.linalg.norm(pos - p)) / max(speed, 1e-3) / rig.s.tick_s)))
        p_start = p.copy()
        for k in range(n):
            q, _, _ = arm.ik_step(q, p_start + (pos - p_start) * (k + 1) / n, R, dq_max=dq_max)
            if held is not None and blocked is None and li >= check_from and k % check_every == 0:
                hits = _sweep_contacts(rig, q, held, p_rel, R_rel, obstacles)
                if hits:
                    blocked = "leg %d: %s" % (li, hits[0])
        for _ in range(settle_ticks):
            q, ep, er = arm.ik_step(q, pos, R, dq_max=dq_max)
            if ep < 0.0005 and er < 0.005:
                break
        if held is not None and blocked is None and li >= check_from:
            hits = _sweep_contacts(rig, q, held, p_rel, R_rel, obstacles)
            if hits:
                blocked = "leg %d end: %s" % (li, hits[0])
        p, Rk = arm.fk_cmd(q)
        errs.append(float(np.linalg.norm(pos - p)) + 0.02 * float(np.linalg.norm(orientation_error(R, Rk))))
    return errs, blocked


def _grasp_shoe(rig, i, heading_des=None, pos_des=None, pos_des_toe=None):
    """Grasp the upright shoe at its heel counter (or toe cap) from above, closing axis along the shoe, and lift it to
    CARRY_Z.  With a destination given, the grasp (heel yaw A, heel yaw B, toe yaw A, toe yaw B) is chosen by a dry run
    of the whole streamed path (hover, grasp, lift, carry, descent) on the controller model from the current posture,
    then from home (the arm then goes home first): the first whose streamed tcp ends within 5 mm of the destination
    (and 3 mm of the grasp) wins.  Returns True (holding, lifted), False (grasp failed) or None (no feasible plan).
    The final pose of each grasp is the one _place_shoe will command (from the shoe's actual pose in the planned grasp
    frame, so a tilted shoe tilts the hand), and grasps whose hand footprint there stays HAND_INSET / FINGER_INSET inside
    the walls and whose tool stays within TILT_MAX_DEG of vertical are tried first ("clear"); the others only when no
    clear one is feasible ("tight": the guarded descent of _place_shoe then backs off from any wall contact).  A toe grasp
    goes to `pos_des_toe` (the shoe shifted toward its heel: its outer finger, 27 mm beyond the toe, would otherwise
    come down on the end wall).  The chosen destination is left in rig.shoe_dest_kind ("heel" / "toe")."""
    sc = rig.sc
    arm = sc.arms["left"]
    dims = sc.shoes[i]["dims"]
    hover = 0.10
    plan = None
    if heading_des is not None and pos_des is not None:
        h_now = sc.shoe_heading(i)
        dyaw = _wrap(_yaw_of(heading_des) - _yaw_of(h_now))
        R_des = np.column_stack([heading_des, np.cross([0.0, 0.0, 1.0], heading_des), [0.0, 0.0, 1.0]])
        cands = []
        ps_now, Rs_now = _shoe_frame(rig, i)
        L_, W_, H_, t_ = _carton(rig)
        for key in ("heel_grip", "toe_grip"):
            hd = sc.shoe_handles(i)[key]
            site = np.array(hd["pos"], float)
            Rs = quat_to_mat(np.array(hd["quat"], float))
            yaw = _yaw_of(_unit([Rs[0, 1], Rs[1, 1], 0.0]))
            tcp = site + np.array([0.0, 0.0, SHOE_GRASP_Z])
            toe = key == "toe_grip" and pos_des_toe is not None
            pdes = np.asarray(pos_des_toe, float) if toe else np.asarray(pos_des, float)
            width = float(sc.shoes[i]["grip"][key].get("pinch_width", 0.045))
            for yw in (yaw, yaw + math.pi):
                qq = down_quat(yw)
                Rg = quat_to_mat(qq)
                # the shoe's pose relative to the tcp once grasped (the grasp pose is the planned one), and the tcp pose
                # that sets it at the destination (what _place_shoe commands)
                p_rel = Rg.T @ (ps_now - tcp)
                R_rel = Rg.T @ Rs_now
                R_f = R_des @ R_rel.T
                tcp_f = pdes - R_f @ p_rel
                hm, fm = _footprint_margins(rig, tcp_f, R_f, width)
                tilt = _tilt_deg(R_f)
                clear = hm >= HAND_INSET and fm >= FINGER_INSET and tilt <= TILT_MAX_DEG
                cands.append({"key": key, "yw": yw, "qq": qq, "tcp": tcp, "site": site, "tcp_f": tcp_f, "R_f": R_f, "p_rel": p_rel,
                              "R_rel": R_rel, "kind": "toe" if toe else "heel", "hm": hm, "fm": fm, "tilt": tilt, "clear": clear})
        obstacles = _carry_obstacles(rig, i)
        tiers = [("clear", [c for c in cands if c["clear"]]),
                 ("tight", sorted([c for c in cands if not c["clear"]], key=lambda c: (c["tilt"] > TILT_MAX_DEG, -min(c["hm"], c["fm"]))))]
        for tier, group in tiers:
            for from_home in (False, True):
                q0 = arm.home_q if from_home else arm.q_cmd
                for carry_z in (CARRY_Z, CARRY_Z + 0.05):
                    for c in group:
                        key, yw, qq, tcp, site, tcp_f, Rf = c["key"], c["yw"], c["qq"], c["tcp"], c["site"], c["tcp_f"], c["R_f"]
                        Rg = quat_to_mat(qq)
                        hover_f = tcp_f + np.array([0.0, 0.0, H_ + SOLE_CLEAR])
                        legs = [(tcp + np.array([0.0, 0.0, hover]), Rg, 0.2), (tcp, Rg, 0.1), (tcp + np.array([0.0, 0.0, 0.05]), Rg, 0.04),
                                (np.array([tcp[0], tcp[1], carry_z]), Rg, 0.15), (np.array([tcp_f[0], tcp_f[1], carry_z]), Rf, 0.2),
                                (hover_f, Rf, 0.12), (tcp_f, Rf, 0.04)]
                        errs, blocked = _dry_run(rig, q0, legs, held=i, p_rel=c["p_rel"], R_rel=c["R_rel"], obstacles=obstacles, check_from=2)
                        # the sweep is advisory for the placement (logged): with the toe 4 mm from the end wall and the two
                        # shoes W/2 apart, the kinematic placement always shows 2-3 mm "penetrations" of the SHOE that the
                        # physics absorbs; the hand's own clearance there is the footprint test above. At the grasp and
                        # the lift (legs 2-3) a finger or the hand on the OTHER shoe or the lid is real: the finger beyond
                        # a toe 2-4 cm from its neighbour on the tray pressed it at 32 N (a voided pair)
                        hard = blocked is not None and blocked.startswith(("leg 2", "leg 3")) and "left/" in blocked \
                            and ("shoe" in blocked or "lid" in blocked)
                        ok = errs[1] <= 0.003 and errs[-1] <= 0.005 and not hard
                        _log(rig, "shoe %d dry run %s yaw %.2f (%s, to the %s slot spot) from %s carry z %.2f: hand/finger inset %.1f/%.1f mm, tool tilt %.0f deg, grasp err %.1f mm, destination err %.1f mm, blocked %s -> %s" % (
                            i, key, yw, tier, c["kind"], "home" if from_home else "here", carry_z, min(c["hm"], 9.99) * 1e3, min(c["fm"], 9.99) * 1e3,
                            c["tilt"], errs[1] * 1e3, errs[-1] * 1e3, blocked, "ok" if ok else "no"))
                        if ok:
                            plan = (key, yw, qq, tcp, site, from_home, carry_z, c["kind"])
                            break
                    if plan is not None:
                        break
                if plan is not None:
                    break
            if plan is not None:
                break
        if plan is None:
            _log(rig, "shoe %d: no grasp with a feasible carry to %s" % (i, np.round(pos_des, 3)))
            return None
        keys = [(plan[0], plan)]
        rig.shoe_dest_kind = plan[7]
    else:
        keys = [("heel_grip", None), ("toe_grip", None)]
    for key, pl in keys:
        for attempt in range(2):
            hd = sc.shoe_handles(i)[key]
            site = np.array(hd["pos"], float)
            Rs = quat_to_mat(np.array(hd["quat"], float))
            yaw = _yaw_of(_unit([Rs[0, 1], Rs[1, 1], 0.0]))
            tcp = site + np.array([0.0, 0.0, SHOE_GRASP_Z - 0.003 * attempt])
            if pl is not None:
                q = pl[2]
                if pl[5] and attempt == 0:
                    _go_home(rig)
            else:
                q = None
                for yw in (yaw, yaw + math.pi):
                    qq = down_quat(yw)
                    if _ik_any(rig, tcp, qq) is not None and _ik_any(rig, tcp + np.array([0.0, 0.0, hover]), qq) is not None:
                        q = qq
                        break
                if q is None:
                    _log(rig, "shoe %d: %s at %s is out of reach from above" % (i, key, np.round(site, 3)))
                    break
            rig.via_home = False
            rig.gripper(0.05, wait=False)
            _reach(rig, tcp + np.array([0.0, 0.0, hover]), q)
            e = _glide(rig, "left", tcp, q, speed=0.1, tol=0.001, timeout=4.0, settle=0.1)
            if e > 0.5:
                # the descent pressed something (a finger on the neighbouring shoe): no grasp here, back up
                rig.release(0.2)
                _left_up(rig, tcp[2] + 0.08)
                break
            name = _grasp(rig, "shoe %d %s" % (i, key), site)
            _log(rig, "shoe %d grasp at %s (err %.2f mm) -> %s" % (i, key, e * 1e3, name))
            if name is not None:
                cz = pl[6] if pl is not None else CARRY_Z
                rig.carry_z = cz
                _glide(rig, "left", tcp + np.array([0.0, 0.0, 0.05]), q, speed=0.04, tol=0.003, timeout=3.0)
                _glide(rig, "left", np.array([tcp[0], tcp[1], cz]), q, speed=0.15, tol=0.005, timeout=4.0)
                return True
            rig.release(0.2)
            _left_up(rig, tcp[2] + 0.08)
    return False


def _descend_guarded(rig, p_tcp, q_tcp, recentre=None, speed=0.05, retries=2):
    """Vertical descent of the left tcp onto p_tcp (streamed from the current height, over the target), watching the
    force of every Panda body on the carton / the lid and the tcp lag: above WALL_FORCE_MAX N (or a lag > 10 mm: the
    shoe or a finger sits on a wall top) the descent stops instead of pushing on, the hand backs off 2 cm, re-centres
    over the target (`recentre()` -> (p_tcp, q_tcp), re-read from the box's and the shoe's current poses) and descends
    again, at most `retries` times.  Returns (final position error (m), reason of the last stop or None)."""
    arm = rig.sc.arms["left"]
    why = None
    for k in range(retries + 1):
        if arm.mode != "cartesian":
            arm.set_mode("cartesian")
        pa, _ = arm.tcp_actual()
        top = np.array([p_tcp[0], p_tcp[1], max(float(pa[2]), float(p_tcp[2]) + 0.005)])
        if float(np.linalg.norm(pa[:2] - top[:2])) > 0.0015:
            _glide(rig, "left", top, q_tcp, speed=0.05, tol=0.0015, rot_tol=0.01, timeout=2.0)
        pa, _ = arm.tcp_actual()
        p0 = pa.copy()
        n = max(1, int(math.ceil(float(np.linalg.norm(p_tcp - p0)) / speed / rig.s.tick_s)))
        why = None
        for j in range(n + int(round(1.5 / rig.s.tick_s))):
            cmd = p0 + (p_tcp - p0) * min(1.0, (j + 1) / n)
            arm.command_cartesian(cmd, q_tcp)
            rig.tick()
            f, what = _arm_box_force(rig)
            pa, _ = arm.tcp_actual()
            if f > WALL_FORCE_MAX:
                why = "%s pressed at %.0f N" % (what, f)
                break
            lag = float(np.linalg.norm(pa - cmd))
            if j * rig.s.tick_s > 0.3 and lag > 0.010:
                why = "tcp lag %.0f mm" % (lag * 1e3)
                break
            if j >= n and float(np.linalg.norm(pa - p_tcp)) <= 0.001:
                break
        pa, _ = arm.tcp_actual()
        if why is None:
            rig.wait(0.1)
            pa, _ = arm.tcp_actual()
            return float(np.linalg.norm(pa - p_tcp)), None
        arm.command_cartesian(pa, q_tcp)
        _log(rig, "placement descent stopped %.1f mm above the target: %s; backing off 2 cm (try %d)" % ((pa[2] - p_tcp[2]) * 1e3, why, k + 1))
        _glide(rig, "left", pa + np.array([0.0, 0.0, 0.02]), q_tcp, speed=0.04, tol=0.003, timeout=2.0)
        if recentre is not None:
            p_tcp, q_tcp = recentre()
    pa, _ = arm.tcp_actual()
    return float(np.linalg.norm(pa - p_tcp)), why


def _place_shoe(rig, i, pos_des, R_des, park=None, slot=None):
    """Carry the held shoe (welded to the hand) so that its body frame reaches (pos_des, R_des), release, lift.  The
    shoe enters its slot vertically from a hover with its sole SOLE_CLEAR above the wall tops; the descent is guarded
    (_descend_guarded: it backs off from any wall contact above WALL_FORCE_MAX N instead of pushing on).  With `slot`
    = (floor-frame position, heading sign) the target is re-read from the box's current pose before every attempt.
    A shoe held so tilted that the tool would tilt more than TILT_MAX_DEG is not lowered into the box.  When the hand
    cannot get within a few mm of the pose (a wrist limit, a wall), it tries again through the home posture; if that
    fails too the shoe is set down at `park` (or back up high) instead of being dropped off target."""
    arm = rig.sc.arms["left"]
    state = {}
    L, W, H, t = _carton(rig)

    def refresh():
        # the shoe's pose in the hand, re-read before every attempt: a shoe pushed against a wall in a failed attempt
        # shifts in the (soft) weld, and a placement computed from the old pose then lands it in the wrong half
        ps, Rs = _shoe_frame(rig, i)
        pa, Ra = arm.tcp_actual()
        state["R_rel"] = Ra.T @ Rs
        state["p_rel"] = Ra.T @ (ps - pa)

    def target():
        if slot is None:
            return np.asarray(pos_des, float), np.asarray(R_des, float)
        local, sgn = slot
        fp, Rf, _ = _floor_frame(rig)
        heading = _unit([Rf[0, 0] * sgn, Rf[1, 0] * sgn, 0.0])
        Rd = np.column_stack([heading, np.cross([0.0, 0.0, 1.0], heading), [0.0, 0.0, 1.0]])
        return fp + Rf @ np.asarray(local, float), Rd

    def tcp_for(pos, R):
        R_tcp = R @ state["R_rel"].T
        return pos - R_tcp @ state["p_rel"], mat_to_quat(R_tcp)

    def recentre():
        refresh()
        pd, Rd = target()
        return tcp_for(pd, Rd)

    e = 1.0
    via_home = bool(getattr(rig, "via_home", False))
    carry_z = float(getattr(rig, "carry_z", CARRY_Z) or CARRY_Z)
    for attempt in range(2):
        refresh()
        pd, Rd = target()
        p_tcp, q_tcp = tcp_for(pd, Rd)
        R_tcp = quat_to_mat(q_tcp)
        tilt = _tilt_deg(R_tcp)
        hm, fm = _footprint_margins(rig, p_tcp, R_tcp, float(arm.width()))
        if tilt > TILT_MAX_DEG:
            _log(rig, "shoe %d is held tilted (the tool would be %.0f deg off vertical at the slot): not lowering it into the box" % (i, tilt))
            e = 1.0
            break
        _, _, st = _floor_frame(rig)
        lift = max(0.03, (t + H + SOLE_CLEAR) - float(st["_to_floor"](pd)[2]))
        hover = p_tcp + np.array([0.0, 0.0, lift])
        high = np.array([p_tcp[0], p_tcp[1], max(carry_z, float(hover[2]))])
        _log(rig, "shoe %d placement: hand/finger inset %.1f/%.1f mm (width %.1f mm), tool tilt %.1f deg, hover %.3f m above the target" % (
            i, min(hm, 9.99) * 1e3, min(fm, 9.99) * 1e3, arm.width() * 1e3, tilt, lift))
        if attempt == 0 and not via_home:
            _carry(rig, high, q_tcp)
        else:
            _go_home(rig)
            _go(rig, "left", high, q_tcp, timeout=6.0)
        _glide(rig, "left", hover, q_tcp, speed=0.12, tol=0.0015, rot_tol=0.01, timeout=4.0)
        e, why = _descend_guarded(rig, p_tcp, q_tcp, recentre=recentre)
        if e <= 0.006:
            break
        _log(rig, "shoe %d: placement pose missed by %.1f mm (attempt %d)%s" % (i, e * 1e3, attempt + 1, "" if why is None else ": " + why))
        pa, _ = arm.tcp_actual()
        _glide(rig, "left", np.array([pa[0], pa[1], float(high[2])]), q_tcp, speed=0.1, tol=0.005, timeout=4.0)
    if e > 0.006 and park is not None:
        refresh()
        p_park, q_park = tcp_for(park, np.eye(3))
        _log(rig, "shoe %d: parking it at %s instead" % (i, np.round(park, 3)))
        _left_up(rig, CARRY_Z)
        _carry(rig, np.array([p_park[0], p_park[1], CARRY_Z]), q_park)
        _glide(rig, "left", p_park + np.array([0.0, 0.0, 0.03]), q_park, speed=0.12, tol=0.002, timeout=4.0)
        e2 = _glide(rig, "left", p_park, q_park, speed=0.04, tol=0.001, timeout=3.0, settle=0.1)
        rig.release(0.4)
        _glide(rig, "left", p_park + np.array([0.0, 0.0, 0.12]), q_park, speed=0.25, tol=0.005, timeout=3.0)
        return e
    rig.release(0.4)
    pa, Ra = arm.tcp_actual()
    _glide(rig, "left", pa + np.array([0.0, 0.0, 0.12]), mat_to_quat(Ra), speed=0.25, tol=0.005, timeout=3.0)
    return e


def _shoe_packed(rig, i):
    sc = rig.sc
    st = sc.carton_state()
    c = sc.shoe_centre(i)
    return bool(st["erected"] and sc.inside_box(0, c) and sc.upright(i, max_deg=15.0) and not sc.resting_on_wall(0, i))


def pack_pair(rig):
    """Put the two shoes of the current pair into the erected box: sole down, heel-to-toe, one in each half."""
    sc = rig.sc
    _left_up(rig, 0.18)
    _park_right(rig)
    rec = rig.s.work.current
    shoe_ids = list(rec["shoe_ids"]) if rec is not None else [0, 1]
    L, W, H, t = _carton(rig)
    dims = sc.shoes[shoe_ids[0]]["dims"]
    shoe_len = float(dims["L"])
    # heels toward the box centre (a lid resting on the heel counters then tilts less), 4 mm of play left at the toe
    x_off = max(0.0, (L - shoe_len) / 2 - 0.004)
    y_off = W / 4
    fp, Rf, st = _floor_frame(rig)
    xb = _unit([Rf[0, 0], Rf[1, 0], 0.0])
    # slots: +x_box at +y_box and -x_box at -y_box; assign them so that the shoes turn as little as possible
    heads = [sc.shoe_heading(i) for i in shoe_ids]
    cost_a = sum(abs(_wrap(_yaw_of(sg * xb) - _yaw_of(h))) for sg, h in zip((1.0, -1.0), heads))
    cost_b = sum(abs(_wrap(_yaw_of(sg * xb) - _yaw_of(h))) for sg, h in zip((-1.0, 1.0), heads))
    signs = (1.0, -1.0) if cost_a <= cost_b else (-1.0, 1.0)
    order = list(range(len(shoe_ids)))
    deferred = set()
    swapped = set()
    while order:
        k = order.pop(0)
        i = shoe_ids[k]
        if _shoe_packed(rig, i):
            continue
        if rig.s.clamp_engaged == i:
            rig.release_clamp()
            rig.wait(0.1)
        sgn = signs[k]
        placed = False
        for attempt in range(3):
            if not sc.upright(i, max_deg=15.0):
                _right_shoe(rig, i)
                if not sc.upright(i, max_deg=15.0):
                    continue
            fp, Rf, st = _floor_frame(rig)
            heading = _unit([Rf[0, 0] * sgn, Rf[1, 0] * sgn, 0.0])
            R_des = np.column_stack([heading, np.cross([0.0, 0.0, 1.0], heading), [0.0, 0.0, 1.0]])
            local = np.array([sgn * x_off, sgn * y_off, t + 0.004])
            local_toe = np.array([-sgn * x_off, sgn * y_off, t + 0.004])    # a toe grasp: the heel goes to the end wall
            pos_des = fp + Rf @ local
            rig.shoe_dest_kind = "heel"
            g = _grasp_shoe(rig, i, heading, pos_des, fp + Rf @ local_toe)
            if g is None and not any(_shoe_packed(rig, j) for j in shoe_ids) and k not in swapped:
                # nothing placed yet: try the other slot assignment for this shoe before deferring it
                signs = tuple(-sg for sg in signs)
                swapped.add(k)
                sgn = signs[k]
                _log(rig, "shoe %d: no feasible carry to its slot, swapping the slot assignment to %s" % (i, signs))
                continue
            if g is None and k not in deferred and order:
                _log(rig, "shoe %d: deferred, the other shoe is placed first" % i)
                deferred.add(k)
                order.append(k)
                break
            if not g:
                _log(rig, "shoe %d: no grasp" % i)
                continue
            lc = local_toe if getattr(rig, "shoe_dest_kind", "heel") == "toe" else local
            e = _place_shoe(rig, i, fp + Rf @ lc, R_des, park=_tray_spot(rig, k), slot=(lc, sgn))
            rig.wait(0.3)
            placed = _shoe_packed(rig, i)
            c = sc.shoe_centre(i)
            _log(rig, "shoe %d placed (tcp err %.2f mm): packed %s, centre(floor) %s, upright %s t=%.1f" % (
                i, e * 1e3, placed, np.round(st["_to_floor"](c), 3), sc.upright(i, 15.0), rig.time))
            if placed:
                break
    _left_up(rig, 0.2)
    h0, h1 = sc.shoe_heading(shoe_ids[0]), sc.shoe_heading(shoe_ids[1])
    anti = float(np.dot(h0, h1)) <= -math.cos(math.radians(30.0))
    packed = [_shoe_packed(rig, i) for i in shoe_ids]
    if not (all(packed) and anti):
        # judge again from the scene (a shoe may have been nudged by the other placement, or the pair is not
        # heel-to-toe): one more placement of each shoe that is not packed, or of the worse-turned shoe
        _log(rig, "pack_pair check: packed %s heel_to_toe %s (headings %s %s): re-placing" % (packed, anti, np.round(h0, 2), np.round(h1, 2)))
        for k, i in enumerate(shoe_ids):
            fp, Rf, st = _floor_frame(rig)
            sgn = signs[k]
            heading = _unit([Rf[0, 0] * sgn, Rf[1, 0] * sgn, 0.0])
            own_half = float(st["_to_floor"](sc.shoe_centre(i))[1]) * sgn > 0.0
            if _shoe_packed(rig, i) and own_half and float(np.dot(sc.shoe_heading(i), heading)) >= math.cos(math.radians(30.0)):
                continue
            if rig.s.clamp_engaged == i:
                rig.release_clamp()
                rig.wait(0.1)
            if not sc.upright(i, max_deg=15.0):
                _right_shoe(rig, i)
            R_des = np.column_stack([heading, np.cross([0.0, 0.0, 1.0], heading), [0.0, 0.0, 1.0]])
            local = np.array([sgn * x_off, sgn * y_off, t + 0.004])
            local_toe = np.array([-sgn * x_off, sgn * y_off, t + 0.004])
            rig.shoe_dest_kind = "heel"
            if _grasp_shoe(rig, i, heading, fp + Rf @ local, fp + Rf @ local_toe):
                lc = local_toe if getattr(rig, "shoe_dest_kind", "heel") == "toe" else local
                _place_shoe(rig, i, fp + Rf @ lc, R_des, park=_tray_spot(rig, k), slot=(lc, sgn))
                rig.wait(0.3)
            _left_up(rig, 0.2)
        h0, h1 = sc.shoe_heading(shoe_ids[0]), sc.shoe_heading(shoe_ids[1])
        anti = float(np.dot(h0, h1)) <= -math.cos(math.radians(30.0))
    ok = all(_shoe_packed(rig, i) for i in shoe_ids) and anti
    _log(rig, "pack_pair done: %s (packed %s heel_to_toe %s) t=%.1f" % (ok, [_shoe_packed(rig, i) for i in shoe_ids], anti, rig.time))
    return ok


# ---------------------------------------------------------------------------------------------------- put_lid
def _lid_rest_height(rig, shoe_ids):
    """Height (floor frame) at which the lid's underside comes to rest: the wall tops, or the shoes' highest point."""
    L, W, H, t = _carton(rig)
    fp, Rf, st = _floor_frame(rig)
    z = t + H + 0.0015
    for i in shoe_ids:
        d = rig.sc.shoes[i]["dims"]
        p, R = _shoe_frame(rig, i)
        for lx, lz in ((d["x_hc"], d["z_hc_top"]), (-d["L"] / 2, d["z_hc_top"]), (d["x_tc"], d["z_tc_top"]), (d["L"] / 2, d["z_tc_top"])):
            for ly in (-d["W"] / 2, d["W"] / 2):
                pw = p + R @ np.array([lx, ly, lz])
                z = max(z, float(st["_to_floor"](pw)[2]))
    return z


def _grasp_lid(rig):
    """Grasp the lid tray at one of its wall-top handles (fingers closing across the wall, 10.5 mm of the wall top
    inside the pinch box), approached from z = 0.20 straight above.  Handles within LID_RIM_CLEAR of the infeed tray's
    rim are skipped (the outer finger would come down on the rim); the nearest remaining handle to the arm is tried
    first.  A descent that stalls (arm at a joint limit) is re-planned once through the home posture."""
    sc = rig.sc
    hs = sc.carton_handles()
    z = _zones(rig)
    tx, ty = z["tray_centre"]
    hx, hy = z["tray_half"]
    cands = []
    for key, sn in sc.lid["sites"].items():
        h = hs[sn]
        p = np.array(h["pos"], float)
        rim = min(hx - abs(p[0] - tx), hy - abs(p[1] - ty))     # distance to the nearest tray edge (negative: outside)
        near_rim = abs(rim) < LID_RIM_CLEAR
        cands.append((near_rim, float(np.linalg.norm(p[:2] - LEFT_BASE[:2])), key, sn, rim))
    cands.sort()
    for near_rim, _, key, sn, rim in cands:
        if near_rim:
            _log(rig, "lid handle %s is %.0f mm from the tray rim: skipped" % (key, abs(rim) * 1e3))
            continue
        h = hs[sn]
        p = np.array(h["pos"], float)
        R = quat_to_mat(np.array(h["quat"], float))
        normal = _unit([R[0, 1], R[1, 1], 0.0])                 # closing axis across the wall (the site's +y)
        for yaw in (_yaw_of(normal), _yaw_of(normal) + math.pi):
            q = down_quat(yaw)
            tcp = p + np.array([0.0, 0.0, PINCH_Z_HI - ATTACH_MARGIN - 0.0005])
            if _ik_any(rig, tcp, q) is None or _ik_any(rig, tcp + np.array([0.0, 0.0, 0.10]), q) is None:
                continue
            rig.gripper(0.03, wait=False)
            _left_up(rig, APPROACH_Z)
            _reach(rig, np.array([tcp[0], tcp[1], APPROACH_Z]), q)
            for attempt in range(2):
                e = _glide(rig, "left", tcp, q, speed=0.1, tol=0.001, timeout=4.0, settle=0.1, stall_s=0.8)
                if e > 0.004:
                    _log(rig, "lid %s: descent stalled %.1f mm short%s" % (key, e * 1e3, ", re-planning via home" if attempt == 0 else ""))
                    _glide(rig, "left", np.array([tcp[0], tcp[1], tcp[2] + 0.06]), q, speed=0.1, timeout=2.0)
                    if attempt == 0:
                        _go_home(rig)
                        _go(rig, "left", np.array([tcp[0], tcp[1], APPROACH_Z]), q, timeout=6.0)
                    continue
                name = _grasp(rig, "lid %s" % key, p)
                _log(rig, "lid grasp at %s (err %.2f mm) -> %s" % (key, e * 1e3, name))
                if name is not None:
                    _glide(rig, "left", np.array([tcp[0], tcp[1], CARRY_Z]), q, speed=0.25, tol=0.005, timeout=4.0)
                    return True
                rig.release(0.2)
                _left_up(rig, tcp[2] + 0.08)
                break
            break
    return False


def put_lid(rig):
    """Fetch the lid tray and set it over the erected box (rig.sc.lid_closed()): the lid rests on the wall tops, its
    origin t + H + 1.5 mm above the floor (the carton is taller than the shoes).  Placed once, verified, with at most
    one corrective re-place."""
    sc = rig.sc
    arm = sc.arms["left"]
    _left_up(rig, 0.18)
    park_hook_arm(rig)
    rec = rig.s.work.current
    shoe_ids = list(rec["shoe_ids"]) if rec is not None else [0, 1]
    if sc.lid_closed():
        return True
    L, W, H, t = _carton(rig)
    for attempt in range(2):
        if not _grasp_lid(rig):
            _log(rig, "lid: no grasp (attempt %d)" % (attempt + 1))
            continue
        fp, Rf, st = _floor_frame(rig)
        lp, lq = sc.lid_pose()
        Rl = quat_to_mat(lq)
        pa, Ra = arm.tcp_actual()
        R_rel = Ra.T @ Rl
        p_rel = Ra.T @ (lp - pa)
        # desired lid yaw: the box yaw or box yaw + pi, whichever needs the smaller rotation
        yaw_box = _yaw_of(Rf[:, 0])
        yaw_lid = _yaw_of(Rl[:, 0])
        cands = [yaw_box, yaw_box + math.pi]
        yaw_des = min(cands, key=lambda y: abs(_wrap(y - yaw_lid)))
        c, s_ = math.cos(yaw_des), math.sin(yaw_des)
        R_des = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        z_rest = t + H + 0.0015
        z_shoes = _lid_rest_height(rig, shoe_ids)
        if z_shoes > z_rest + 0.002:
            _log(rig, "lid: a shoe sticks out above the wall tops (%.4f > %.4f): the lid will not sit flat" % (z_shoes, z_rest))
        pos_des = fp + Rf @ np.array([0.0, 0.0, z_rest + 0.002])
        R_tcp = R_des @ R_rel.T
        q_tcp = mat_to_quat(R_tcp)
        p_tcp = pos_des - R_tcp @ p_rel
        high = np.array([p_tcp[0], p_tcp[1], CARRY_Z])
        _carry(rig, high, q_tcp)
        _glide(rig, "left", p_tcp + np.array([0.0, 0.0, 0.04]), q_tcp, speed=0.12, tol=0.0015, timeout=4.0, settle=0.2)
        e = _glide(rig, "left", p_tcp, q_tcp, speed=0.03, tol=0.0008, timeout=3.0, settle=0.2, ignore=("box",))
        lp2, _ = sc.lid_pose()
        rel = st["_to_floor"](lp2)
        _log(rig, "lid lowered (tcp err %.2f mm): lid origin in floor frame %s (rest %.4f)" % (e * 1e3, np.round(rel, 4), z_rest))
        rig.release(0.5)
        _glide(rig, "left", p_tcp + np.array([0.0, 0.0, 0.10]), q_tcp, speed=0.25, tol=0.005, timeout=3.0)
        closed = _wait_until(rig, sc.lid_closed, 1.5)
        lp3, lq3 = sc.lid_pose()
        R3 = quat_to_mat(lq3)
        _log(rig, "lid closed %s; origin(floor) %s tilt %.1f deg t=%.1f" % (closed, np.round(st["_to_floor"](lp3), 4), math.degrees(math.acos(min(1.0, R3[2, 2]))), rig.time))
        if closed:
            break
        _log(rig, "lid not closed: corrective re-place" if attempt == 0 else "lid not closed after the corrective re-place")
    _left_up(rig, 0.2)
    return bool(sc.lid_closed())


# ---------------------------------------------------------------------------------------------------- ship_box
def _push_frame(rig):
    """(floor pos, world +x-most box axis xw, yw, tool quaternion, forward/low corner offsets) for the push."""
    L, W, H, t = _carton(rig)
    fp, Rf, st = _floor_frame(rig)
    xw = Rf[:, 0] if float(Rf[0, 0]) >= 0 else -Rf[:, 0]           # the box axis that points along world +x
    xw = _unit([xw[0], xw[1], 0.0])
    yw = np.cross([0.0, 0.0, 1.0], xw)
    face = fp - xw * (L / 2 + t)                                  # outer face of the -x wall at its middle
    # fingers pitched forward by beta: the fingertip corner pushes the box wall below the lid's hanging wall while
    # the hand body leans back, clear of the lid
    beta = math.radians(35.0)
    zt = -math.cos(beta) * np.array([0.0, 0.0, 1.0]) + math.sin(beta) * xw
    xt = np.cross(yw, zt)
    R_t = np.column_stack([xt, yw, zt])
    q = mat_to_quat(R_t)
    corners = [R_t @ np.array([sx * 0.0105, 0.0, cz]) for sx in (-1.0, 1.0) for cz in (FINGER_TIP_Z, -0.045)]
    fwd = max(float(np.dot(c, xw)) for c in corners)
    low = min(float(c[2]) for c in corners)
    start = face - xw * (fwd + PUSH_GAP)
    start[2] = 0.012 - low
    return fp, xw, yw, q, start


def _plan_stroke(rig, pre, p_from, p_to, quat, steps=12, min_margin=0.3, avoid=()):
    """A start configuration for `pre` from which the descent to p_from and the straight stroke p_from -> p_to are
    IK-continuable, on the branch with the largest joint-limit margin along the stroke.  Returns (q_pre, margin).
    Configurations within 0.3 rad (every joint) of one in `avoid` (a branch whose stroke lagged) are skipped as long as
    another branch is feasible."""
    arm = rig.sc.arms["left"]
    R = quat_to_mat(quat)
    poses = [np.asarray(pre, float), np.asarray(p_from, float)] + [p_from + (p_to - p_from) * k / steps for k in range(1, steps + 1)]
    best = (None, -1.0)
    best_avoided = (None, -1.0)
    seen = []
    for q_ref in _ref_configs(rig):
        for off in SEED_OFFSETS:
            qs = np.clip(q_ref + np.asarray(off, float), arm.lower, arm.upper)
            q_pre, ep, er, ok = ik_solve(arm, poses[0], R, q0=qs)
            if not ok or arm_collisions(arm, q_pre):
                continue
            if any(float(np.max(np.abs(q_pre - qq))) < 0.05 for qq in seen):
                continue
            seen.append(q_pre)
            qq = q_pre
            margin = _margin(rig, q_pre)
            good = True
            for p in poses[1:]:
                q2, ep, er, ok = ik_solve(arm, p, R, q0=qq, iters=200)
                if not ok or float(np.max(np.abs(q2 - qq))) > 1.2:
                    good = False
                    break
                qq = q2
                margin = min(margin, _margin(rig, q2))
            if not good:
                continue
            if any(float(np.max(np.abs(q_pre - np.asarray(a, float)))) < 0.3 for a in avoid):
                if margin > best_avoided[1]:
                    best_avoided = (q_pre, margin)
                continue
            if margin > best[1]:
                best = (q_pre, margin)
                if margin >= min_margin + 0.2:
                    return best
    if best[0] is None:
        return best_avoided
    return best


def ship_box(rig):
    """Push the closed box across the outfeed line with the closed gripper, from the box's -x side.  The fingers push
    below the lid's hanging walls (tcp at z = 0.021).  Each stroke is planned first (IK continuation along the straight
    line, the branch with the largest joint margin, the branch change done high above the bench) and pushed at
    <= 0.10 m/s.  A box farther than PUSH_SPLIT from the outfeed line is pushed in two strokes: half the way, then a
    re-approach (back off, lift, re-plan from the box's new pose) and the rest.  The tcp lag is judged against the
    lag of the settled stroke (measured 0.5-1.0 s into it, the first 0.5 s of acceleration discarded): a settled lag
    above PUSH_BAD_LAG means a bad branch (lift, re-home, re-plan avoiding that branch), and a later lag 15 mm above
    the settled one, or the box not moving for 1 s, is a stall (lift, re-home, re-approach, push again).  At most 5
    strokes.  The box ships once it rests beyond the line with the lid closed: without the lid closed there is nothing
    to ship and nothing is pushed (pushing such a box from wherever a failed packing left it, near the gripper's base,
    put the hand on the bench at 295 N, seed 1004), nor is a stroke for which no branch can be planned."""
    sc = rig.sc
    arm = sc.arms["left"]
    if not _floor_frame(rig)[2]["closed"]:
        _log(rig, "ship: the lid is not closed, nothing to ship: no push (t=%.1f)" % rig.time)
        return False
    _left_up(rig, 0.18)
    park_hook_arm(rig)
    outfeed = float(sc.layout["outfeed_x"])
    goal_x = outfeed + 0.025
    rec0 = rig.s.work.current

    def shipped_now():
        return rig.s.work.current is not rec0 or rec0["shipped"]

    shipped = False
    avoid = []
    split_done = False
    for stroke in range(5):
        if shipped_now():
            shipped = True
            break
        fp, xw, yw, q, start = _push_frame(rig)
        if float(fp[0]) >= goal_x:
            shipped = _wait_until(rig, shipped_now, 3.0)
            if shipped:
                break
        dist_full = goal_x - float(fp[0]) + 0.03
        partial = (not split_done) and (outfeed - float(fp[0]) > PUSH_SPLIT)
        dist = 0.5 * (goal_x - float(fp[0])) if partial else dist_full
        pre = start + np.array([0.0, 0.0, 0.12])
        p_end = start + xw * dist
        q_pre, margin = _plan_stroke(rig, pre, start, start + xw * dist_full, q, avoid=avoid)
        _log(rig, "stroke %d: box at %s, push %.3f m%s along %s; planned branch margin %.2f rad (%s)" % (
            stroke + 1, np.round(fp, 3), dist, " (first of two)" if partial else "", np.round(xw, 2), margin, None if q_pre is None else np.round(q_pre, 2)))
        if q_pre is None:
            _log(rig, "stroke %d: no branch can be planned from here, not pushing" % (stroke + 1))
            break
        rig.gripper(0.0, wait=False)
        if q_pre is not None and float(np.max(np.abs(q_pre - arm.q()))) > 0.15:
            _reconfigure(rig, pre, q, q_pre)
        if not rig.move("left", pre, q, timeout=3.0):
            _reach(rig, pre, q)
        q_start = arm.q().copy()
        _glide(rig, "left", start, q, speed=0.12, tol=0.002, timeout=3.0, settle=0.1)
        _glide(rig, "left", start + xw * (PUSH_GAP + 0.003), q, speed=0.03, tol=0.003, timeout=2.0, ignore=("box",))
        p0, _ = arm.tcp_actual()
        target = p0 + xw * dist
        speed = 0.10
        n = int(dist / speed / rig.s.tick_s)
        lag_max = 0.0
        lag_ref = 0.0                                   # the settled lag, measured 0.5-1.0 s into the stroke
        marg_min = 10.0
        stalled = None
        bad_branch = False
        x_hist = []
        n_settle = int(round(0.5 / rig.s.tick_s))
        n_ref = int(round(1.0 / rig.s.tick_s))
        i = 0
        for i in range(n):
            cmd = p0 + (target - p0) * (i + 1) / n
            arm.command_cartesian(cmd, q)
            rig.tick()
            hit = link_press(rig, "left", ignore=("box",))     # the fingers push the box; a link on the lid is a press
            if hit is not None:
                stalled = "%s presses %s with %.0f N" % hit
                break
            pa, _ = arm.tcp_actual()
            lag = float(np.linalg.norm(pa - cmd))
            lag_max = max(lag_max, lag)
            marg_min = min(marg_min, _margin(rig, arm.q()))
            fp2, _, _ = _floor_frame(rig)
            x_hist.append(float(fp2[0]))
            if fp2[0] >= goal_x or shipped_now():
                break
            if i < n_settle:
                pass
            elif i < n_ref:
                lag_ref = max(lag_ref, lag)
            elif i == n_ref and lag_ref > PUSH_BAD_LAG:
                stalled = "settled tcp lag %.0f mm > %.0f mm: bad branch" % (lag_ref * 1e3, PUSH_BAD_LAG * 1e3)
                bad_branch = True
            elif lag > lag_ref + 0.015:
                stalled = "tcp lag %.0f mm > settled %.0f + 15 mm" % (lag * 1e3, lag_ref * 1e3)
            if len(x_hist) > n_ref and x_hist[-1] - x_hist[-n_ref - 1] < 0.003:
                stalled = "box not moving (x %.3f)" % x_hist[-1]
            if stalled:
                break
        pa, _ = arm.tcp_actual()
        arm.command_cartesian(pa, q)
        fp2, _, st2 = _floor_frame(rig)
        _log(rig, "stroke %d: box x %.3f (goal %.3f) after %.2f s, tcp lag settled %.1f / max %.1f mm, min joint margin %.2f rad%s" % (
            stroke + 1, fp2[0], goal_x, (i + 1) * rig.s.tick_s, lag_ref * 1e3, lag_max * 1e3, marg_min, "" if stalled is None else ", STALLED: " + stalled))
        if stalled is not None and _wait_until(rig, shipped_now, 1.0):
            shipped = True
            _log(rig, "box shipped during the stall at x %.3f t=%.1f" % (_floor_frame(rig)[0][0], rig.time))
            break
        if partial and stalled is None and fp2[0] < outfeed:
            # first of two strokes done: back off, lift and re-approach from the box's new pose (no re-homing)
            split_done = True
            pb, Rb = arm.tcp_actual()
            _glide(rig, "left", pb - xw * 0.03, mat_to_quat(Rb), speed=0.08, tol=0.004, timeout=2.0)
            pb, Rb = arm.tcp_actual()
            _glide(rig, "left", pb + np.array([0.0, 0.0, 0.12]), mat_to_quat(Rb), speed=0.15, tol=0.005, timeout=3.0)
            continue
        if stalled is None or fp2[0] >= outfeed:
            shipped = _wait_until(rig, shipped_now, 3.0)
            fp2, _, st2 = _floor_frame(rig)
            _log(rig, "box at x %.3f closed %s -> shipped %s t=%.1f" % (fp2[0], st2["closed"], shipped, rig.time))
            if shipped:
                break
            if fp2[0] < goal_x + 0.03 and stalled is None:
                p1, _ = arm.tcp_actual()
                _glide(rig, "left", p1 + xw * (goal_x + 0.02 - float(fp2[0])), q, speed=0.06, tol=0.003, timeout=4.0, ignore=("box",))
                shipped = _wait_until(rig, shipped_now, 3.0)
                fp2, _, st2 = _floor_frame(rig)
                _log(rig, "box at x %.3f closed %s -> shipped %s t=%.1f" % (fp2[0], st2["closed"], shipped, rig.time))
                if shipped:
                    break
        if bad_branch:
            avoid.append(q_start)
        # recover: back off, lift, re-home, and push again from the box's current pose
        split_done = True
        pb, Rb = arm.tcp_actual()
        _glide(rig, "left", pb - xw * 0.03, mat_to_quat(Rb), speed=0.08, tol=0.004, timeout=2.0)
        _left_up(rig, 0.15)
        _go_home(rig)
    _left_up(rig, 0.2)
    return bool(shipped)
