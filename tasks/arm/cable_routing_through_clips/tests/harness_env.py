"""Harness cell: a fixed Franka Panda at a bench routes a cable over a form board (no wire, no credit text, no images).

Plain MuJoCo 3.16. The Panda and its hand are robosuite 1.5.2's model files (robosuite is not imported); the
operational-space controller, the 10 Hz control step and the action layout are those of
tasks/arm/lab_sample_handling's box_transport_env.py. The cable is MuJoCo's elastic cable (`mujoco.elasticity.cable`,
40 segments) held at one end by a fixed connector.

Frames: world z up, metres, radians, seconds, newtons; quaternions (w, x, y, z). The bench top is z = 0 and the robot's
base stands on it at the origin facing +x. The form board (BOARD_HALF, BOARD_T thick) lies on the bench centred at
BOARD_C; its top is z = BOARD_T. A cable that leaves the board drops onto the bench; one that leaves the bench hangs.

Fixtures, in route order from the connector:
* clip: two steel pins, on each pin's inner top a retaining barb on a hinge about the passage axis. At rest a barb
  points inward and BARB_REST below horizontal; its joint lets it fold down only (0 .. BARB_RANGE) against a spring.
  A cable pushed down between the pins folds the barbs, passes and is held under them when they spring back: geometry,
  a spring and a hard stop, no switch.
* post: a cylinder with a cap; the cable has to pass it with the post on the stated side.

An episode is drawn from its seed by `make_episode`: the job sheet (the NOMINAL layout: what the board's print shows)
and the cell as built (every fixture and the connector moved from nominal within MOUNT_TOL / CLIP_AXIS_TOL, the
cable's stiffness, friction and density). `HarnessEnv.job_sheet()` is what a controller may be told; `truth()` is the
grader's.

    env = HarnessEnv()
    obs = env.reset(seed)
    obs, done = env.step([dx, dy, dz, drx, dry, drz, grip, cams])
    res = env.finish()          # opens the hand, lifts it, lets the cable settle, judges the settled cable
"""

import importlib.util
import math
import os

import mujoco
import numpy as np

TIMESTEP = 0.002
SUBSTEPS = 50                      # -> 10 Hz control
CONTROL_DT = TIMESTEP * SUBSTEPS
MAX_STEPS = 1500                   # control steps of an episode (150 s)
MAX_DPOS = 0.03                    # m per control step
MAX_DROT = 0.2                     # rad per control step
MAX_ARM_SPEED = 8.0                # rad/s: faster is an unstable simulation
WRENCH_NOISE = (0.3, 0.03)         # N, N m (standard deviation per axis)
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, math.pi / 4])

BENCH_X = (-0.30, 1.00)            # the bench top's extent
BENCH_Y = (-0.75, 0.75)
BENCH_H = 0.90                     # the floor is z = -BENCH_H
BOARD_T = 0.012
BOARD_C = (0.49, 0.0)
BOARD_HALF = (0.26, 0.46)          # x, y
ROUTE_HALF = (0.38, 0.17)          # the job sheet's frame: a along the board (world -y), b across (world +x)

N_SEG = 40                         # cable segments (set before reset; the resolution study varies it)
STUDY_RADIUS = 0.0                 # m; see make_episode
STUDY_LENGTH_SCALE = 1.0
LAYOUT_SEG = 40                    # the starting layout is drawn with this many steps, whatever N_SEG is
SOLVER_ITERATIONS = 100            # the constraint solver's iteration limit (MuJoCo's default)
IMPRATIO = 10.0                    # friction-to-normal constraint impedance ratio
JAW_SOLIMP_D = 0.98                # impedance of the groove faces' contact (closer to 1 = less creep of a gripped cable)
NOSLIP_ITERATIONS = 0              # the solver's no-slip pass (the resolution study decides)
JOINT_DAMPING_40 = 0.003           # cable joint damping at 40 segments; scaled with the segment count
PIN_R = 0.002
PIN_H = 0.020
BARB_REST = math.radians(35.0)
BARB_RANGE = 1.1
BARB_T = 0.0003                    # half thickness (a 0.6 mm tab)
BARB_OVERLAP = 0.0030              # m: at rest a tab reaches this far over the cable's side
CLIP_GAP_OVER = (0.0043, 0.0047)   # m: the slot's width over the cable's diameter (side play 2.15-2.35 mm)
BARB_W = 0.005                     # half width along the passage
BARB_SPRING = 0.0015               # N m / rad
LEAD_LEN = 0.007                   # a lead-in fin rises outward at LEAD_ANGLE from each pin's top (a funnel over the slot)
LEAD_ANGLE = math.radians(50.0)    # from horizontal
LEAD_T = 0.0006                    # half thickness
POST_R = 0.012
POST_H = 0.030
POST_CAP_R = 0.018
SEATED_Z = 0.013                   # a clip counts when the cable's centre line crosses between the pins below this (above the board)
POST_TOL = 0.010                   # the cable within this of the post's surface, below its top
MOUNT_TOL = 0.020                  # m: a fixture or the connector is mounted within this of its printed place
BUILT_TRIES = 20                   # draws of the mounting errors for one drawing before the drawing is dropped
CLIP_AXIS_TOL = math.radians(8.0)  # a clip's passage axis is within this of the printed one
# The strain-relief boot at the connector: the cable's fixed end is held by a flexible boot that stretches along the
# connector's axis before the conductors take the load. Stiffness from the boot itself: a TPE boot of Shore 60 A
# (Young's modulus about 3.6 MPa), 35 mm long, 1 mm wall over a 9 mm cable (section about 31 mm^2): E A / L =
# 3.6e6 * 31e-6 / 0.035 = 3.2 kN/m. It stretches at most 20 % of its length (7 mm), then the cable's conductors stop it.
BOOT_STIFFNESS = 3200.0            # N/m
BOOT_TRAVEL = 0.007                # m
BOOT_DAMPING = 15.0                # N s/m
TENSION_LIMIT = 40.0               # N
TENSION_WINDOW = 0.1               # s: the connector is damaged when the mean tension over this window passes the limit
SETTLE_S = 2.0
# The cable gripper on the Panda hand: on each finger a carrier plate under the stock pad and a V groove (two faces at
# JAW_V_ANGLE from the closing direction, JAW_V_LEN long) whose axis lies along the cable. A cable lying on the board
# is scooped into the groove as the fingers close and cannot leave it sideways, up or down; it can slide along its
# axis once the pull passes the slip force (grip force and face friction below; measured: see the datasheet).
JAW_HALF_X = 0.008                 # half length of the groove along the cable
JAW_V_ANGLE = math.radians(60.0)
JAW_V_LEN = 0.0036
JAW_V_T = 0.0012                   # half thickness of a groove face
JAW_V_DZ = 0.020                   # groove axis below the stock pad's centre (toward the fingertip)
JAW_FRICTION = 1.20                # groove faces (they govern the cable-jaw contact: priority)
GRIP_KP = 40000.0                  # N/m, finger servo (stiff enough that the limit below is what grips)
GRIP_FORCE = 60.0                  # N, finger servo limit (grip force per finger)
GRIP_DAMPING = 120.0               # N s/m, finger joints
GRASP_SITE_Z = 0.0170              # grip site above the board's top PLUS the cable's radius when the groove's axis is level with a cable lying on the board (see grasp_site_z)
JAW_APEX_IN = 0.0015               # the groove's apex stands this far inward of the stock pad's centre plane
GROOVE_BELOW_SITE = 0.017          # the groove's axis below the grip site

# Cameras: one fixed over the board looking straight down (the image's top is the far side of the board, +x), and the
# hand's own camera (robosuite's eye-in-hand). name: (model camera, width, height)
OVERHEAD_POS = (0.49, 0.0, 1.05)
OVERHEAD_FOVY = 45.0
CAMERAS = {"overhead": ("overhead", 960, 720), "wrist": ("r_eye_in_hand", 480, 360)}
CAMERA_NAMES = ("overhead", "wrist")

# Bench furniture with contact (the cable or the arm can reach it): name, type, centre (x, y, z of the centre), size
FURNITURE = (
    ("reel_blue", "cylinder", (0.88, 0.42, 0.020), (0.062, 0.020)),
    ("reel_black", "cylinder", (0.88, 0.26, 0.020), (0.050, 0.020)),
    ("bin_blue", "box", (0.88, -0.22, 0.030), (0.075, 0.050, 0.030)),
    ("bin_yellow", "box", (0.88, -0.36, 0.030), (0.075, 0.050, 0.030)),
    ("tape", "cylinder", (0.88, -0.52, 0.010), (0.034, 0.010)),
)


class EnvError(Exception):
    pass


class Episode(object):
    """Everything an episode is made of (drawn from its seed by `make_episode`)."""


def grasp_site_z(radius):
    """Grip site height above the board at which the groove's axis is level with the axis of a cable lying on it."""
    return GROOVE_BELOW_SITE + radius


def robosuite_assets():
    spec = importlib.util.find_spec("robosuite")
    if spec is None or not spec.submodule_search_locations:
        raise EnvError("robosuite assets not installed")
    return os.path.join(next(iter(spec.submodule_search_locations)), "models", "assets")


def axis_angle_mat(v):
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def to_world(p):
    """Job-sheet frame (a, b) -> world (x, y)."""
    return np.array([BOARD_C[0] + p[1], BOARD_C[1] - p[0]])


def to_world_angle(a):
    return a - math.pi / 2


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _disc(rng, r):
    ang, rad = rng.uniform(0.0, 2 * math.pi), r * math.sqrt(rng.uniform())
    return np.array([rad * math.cos(ang), rad * math.sin(ang)])


def _built_ok(conn, built, end, length):
    """Is the board as built (every mount moved within MOUNT_TOL of the print) still a fair board for its drawing?
    The passage points (a clip's place, a post's corner) in route order: every leg 9 to 23 cm; no two fixtures
    nearer than 7 cm; no leg within 3 cm of a fixture that is not at one of its ends; a post still turns the route
    to its printed side by at least 0.45 rad and a clip by at most 0.9 rad; the route still fits the cable."""
    pts = [np.asarray(conn, float)] + [np.asarray(g["corner"] if g["type"] == "post" else g["pos"], float) for g in built]
    pts.append(np.asarray(end, float))
    legs = [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
    if min(legs[1:-1] or [1.0]) < 0.09 or max(legs[1:-1] or [0.0]) > 0.23 or legs[0] < 0.10:
        return False
    if sum(legs) + 0.12 > length:
        return False
    P = np.array([g["pos"] for g in built])
    n = len(built)
    if min(np.linalg.norm(P[i] - P[j]) for i in range(n) for j in range(i + 1, n)) < 0.07:
        return False
    for i in range(len(pts) - 1):
        for j, g in enumerate(built):
            if j in (i - 1, i):
                continue
            if _seg_dist(g["pos"], pts[i], pts[i + 1]) < 0.03:
                return False
    for k, g in enumerate(built):
        a, b, c = pts[k], pts[k + 1], pts[k + 2]
        d_in, d_out = math.atan2(b[1] - a[1], b[0] - a[0]), math.atan2(c[1] - b[1], c[0] - b[0])
        turn = (d_out - d_in + math.pi) % (2 * math.pi) - math.pi
        if g["type"] == "post":
            if (turn > 0) != (g["side"] == "left") or abs(turn) < 0.45:
                return False
        elif abs(turn) > 0.9:
            return False
    return True


def make_episode(seed, n_fixtures=None):
    """The episode of `seed`. `n_fixtures`: None = 3 to 5, or a fixed count (the development boards use 3)."""
    seed = int(seed) & ((1 << 64) - 1)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    ep = Episode()
    ep.seed = seed
    ep.length = float(rng.uniform(0.70, 0.90))
    ep.radius = float(rng.uniform(0.0035, 0.0050))
    if STUDY_RADIUS > 0:                 # studies only (0 in the task): the same board with another cable
        ep.radius = STUDY_RADIUS
    ep.length *= STUDY_LENGTH_SCALE      # studies only (1 in the task): moves the segment joints along the cable
    ep.EI = float(rng.uniform(0.0008, 0.003))            # bending stiffness, N m^2
    ep.bend = ep.EI / (math.pi * ep.radius ** 4 / 4.0)   # the plugin's Young's modulus
    ep.twist = ep.bend / 2.6
    ep.friction = float(rng.uniform(0.5, 1.0))
    ep.density = float(rng.uniform(1200.0, 2000.0))
    ha, hb = ROUTE_HALF
    for _attempt in range(4000):
        n = int(rng.integers(3, 6)) if n_fixtures is None else int(n_fixtures)
        conn0 = np.array([-ha + float(rng.uniform(0.0, 0.05)), float(rng.uniform(-0.10, 0.10))])
        heading = float(rng.uniform(-0.5, 0.5))
        pts = [conn0.copy()]
        fixtures = []
        cur = conn0 + 0.02 * np.array([math.cos(heading), math.sin(heading)])
        d_in = heading
        total = 0.02
        for k in range(n):
            step = float(rng.uniform(0.12, 0.19))
            nxt = cur + step * np.array([math.cos(d_in), math.sin(d_in)])
            total += step
            kind = "clip" if (k == 0 or fixtures[-1]["type"] == "post" or rng.uniform() < 0.6) else "post"
            if kind == "clip":
                turn = float(rng.uniform(-0.6, 0.6))
            else:
                turn = float(rng.choice([-1, 1]) * rng.uniform(0.7, 1.3))
            d_out = d_in + turn
            if kind == "clip":
                gap = 2 * ep.radius + float(rng.uniform(CLIP_GAP_OVER[0], CLIP_GAP_OVER[1]))
                fixtures.append({"type": "clip", "pos": nxt.copy(), "axis": 0.5 * (d_in + d_out), "gap": gap})
            else:
                side = "left" if turn > 0 else "right"
                inner = d_in + turn / 2.0 + (math.pi / 2 if turn > 0 else -math.pi / 2)
                centre = nxt + (POST_R + ep.radius) * np.array([math.cos(inner), math.sin(inner)])
                fixtures.append({"type": "post", "pos": centre, "corner": nxt.copy(), "side": side})
            pts.append(nxt.copy())
            cur, d_in = nxt, d_out
        end = cur + 0.05 * np.array([math.cos(d_in), math.sin(d_in)])
        pts.append(end)
        total += 0.05
        if total + 0.12 > ep.length:
            continue
        P = np.array([f["pos"] for f in fixtures])
        if np.any(np.abs(P[:, 0]) > ha - 0.03) or np.any(np.abs(P[:, 1]) > hb - 0.03):
            continue
        A = np.array(pts)
        if np.any(np.abs(A[:, 0]) > ha) or np.any(np.abs(A[:, 1]) > hb):
            continue
        if min(np.linalg.norm(P[i] - P[j]) for i in range(n) for j in range(i + 1, n)) < 0.09:
            continue
        ok = True
        for i in range(len(pts) - 1):
            for j, f in enumerate(fixtures):
                if j in (i - 1, i):
                    continue
                if _seg_dist(f["pos"], pts[i], pts[i + 1]) < 0.04:
                    ok = False
        if not ok:
            continue
        ep.nominal = {"connector": {"pos": conn0, "heading": heading}, "fixtures": fixtures, "route": A}
        # the cell as built: every mount within tolerance of the print
        # (drawn again, up to BUILT_TRIES times, until the board as built is still a board a fitter could have put
        # together to this drawing: _built_ok; a drawing for which that never happens is dropped)
        for _mount in range(BUILT_TRIES):
            built = []
            for f in fixtures:
                g = dict(f)
                g["pos"] = f["pos"] + _disc(rng, MOUNT_TOL)
                if f["type"] == "clip":
                    g["axis"] = f["axis"] + float(rng.uniform(-CLIP_AXIS_TOL, CLIP_AXIS_TOL))
                else:
                    g["corner"] = f["corner"] + (g["pos"] - f["pos"])
                built.append(g)
            conn_b = conn0 + _disc(rng, MOUNT_TOL)
            ep.mount_draws = _mount + 1
            if _built_ok(conn_b, built, pts[-1], ep.length):
                break
        else:
            ep.rejected_built = getattr(ep, "rejected_built", 0) + 1
            continue
        ep.fixtures_ab = built
        ep.conn_ab = conn_b
        ep.conn_heading_ab = heading + float(rng.uniform(-CLIP_AXIS_TOL, CLIP_AXIS_TOL))
        ep.initial_ab = _initial_shape(rng, ep)
        if ep.initial_ab is not None:
            break
    else:
        raise EnvError("no route found for seed %d" % seed)
    # world frame
    ep.conn_pos = to_world(ep.conn_ab)
    ep.conn_heading = to_world_angle(ep.conn_heading_ab)
    ep.fixtures = []
    for g in built:
        w = {"type": g["type"], "pos": to_world(g["pos"])}
        if g["type"] == "clip":
            w["axis"], w["gap"] = to_world_angle(g["axis"]), g["gap"]
        else:
            w["side"] = g["side"]
        ep.fixtures.append(w)
    ep.initial = np.array([to_world(p) for p in ep.initial_ab])
    return ep


def _initial_shape(rng, ep):
    """A smooth random planar curve from the connector (as built) that keeps clear of every fixture and stays on the
    board."""
    seg = ep.length / LAYOUT_SEG
    ha, hb = BOARD_HALF[1] - 0.02, BOARD_HALF[0] - 0.02
    P = np.array([f["pos"] for f in ep.fixtures_ab])
    for _attempt in range(600):
        h = ep.conn_heading_ab
        bend_to = h + float(rng.uniform(-2.2, 2.2))
        pts = [ep.conn_ab.copy()]
        curv = 0.0
        good = True
        for i in range(LAYOUT_SEG):
            if 3 <= i < 9:
                h += (bend_to - ep.conn_heading_ab) / 6.0
            elif i >= 9:
                curv = 0.85 * curv + float(rng.normal(0.0, 0.12))
                h += curv
            nxt = pts[-1] + seg * np.array([math.cos(h), math.sin(h)])
            if abs(nxt[0]) > ha or abs(nxt[1]) > hb:
                good = False
                break
            if np.min(np.linalg.norm(P - nxt, axis=1)) < 0.040:
                good = False
                break
            if len(pts) > 4 and np.min(np.linalg.norm(np.array(pts[:-3]) - nxt, axis=1)) < 0.03:
                good = False
                break
            pts.append(nxt)
        if good:
            return resample(np.array(pts), N_SEG)
    return None


def resample(P, n):
    """The polyline P at n + 1 points equally spaced along its length."""
    a = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    s = np.linspace(0.0, a[-1], n + 1)
    return np.stack([np.interp(s, a, P[:, k]) for k in range(P.shape[1])], axis=1)


def job_sheet(ep):
    """What the board's print and the harness drawing say (world frame): nominal places, not the cell as built."""
    nom = ep.nominal
    fx = []
    for f in nom["fixtures"]:
        w = to_world(f["pos"])
        e = {"type": f["type"], "pos": [round(float(w[0]), 4), round(float(w[1]), 4)]}
        if f["type"] == "post":
            e["side"] = f["side"]
        fx.append(e)
    c = to_world(nom["connector"]["pos"])
    return {"connector": {"pos": [round(float(c[0]), 4), round(float(c[1]), 4)],
                          "heading": round(float(to_world_angle(nom["connector"]["heading"])), 4)},
            "fixtures": fx,
            "cable": {"radius": round(ep.radius, 5), "length": round(ep.length, 4)},
            "mount_tolerance": MOUNT_TOL}


def barb_length(gap, radius):
    """A clip is the size for its cable (one family, the slot and the retaining tabs scaled with the cable): the slot is
    CLIP_GAP_OVER wider than the cable, so the cable can lie up to half of that to one side; each tab, at rest, reaches
    BARB_OVERLAP over the cable's side, more than that side play, so a seated cable cannot leave past a tab however it
    lies in the slot."""
    return (gap - (2 * radius - 2 * BARB_OVERLAP)) / 2.0 / math.cos(BARB_REST)


def _clip_xml(i, f, radius):
    x, y = f["pos"]
    off = f["gap"] / 2 + PIN_R
    L = barb_length(f["gap"], radius)
    st = 'friction="0.3" solref="0.003 1" solimp="0.95 0.99 0.0005"'
    out = ['<body name="clip%d" pos="%.6f %.6f %.6f" euler="0 0 %.6f">' % (i, x, y, BOARD_T, f["axis"])]
    for sgn, nm in ((1, "a"), (-1, "b")):
        out.append('<geom name="pin%d%s" type="capsule" fromto="0 %.6f 0 0 %.6f %.6f" size="%.6f" rgba=".75 .76 .78 1" %s/>'
                   % (i, nm, sgn * off, sgn * off, PIN_H, PIN_R, st))
        out.append('<geom name="lead%d%s" type="box" size="%.6f %.6f %.6f" pos="0 %.6f %.6f" euler="%.6f 0 0" rgba=".92 .92 .88 1" %s/>'
                   % (i, nm, BARB_W, LEAD_LEN / 2, LEAD_T, sgn * (off + (LEAD_LEN / 2) * math.cos(LEAD_ANGLE)),
                      PIN_H + (LEAD_LEN / 2) * math.sin(LEAD_ANGLE), sgn * LEAD_ANGLE, st))
        out.append('<body name="barb%d%s" pos="0 %.6f %.6f" euler="%.6f 0 0">' % (i, nm, sgn * (off - PIN_R), PIN_H, sgn * BARB_REST))
        # the tab's stop is rigid: MuJoCo's default joint limit is soft in proportion to the moving inertia, and a 0.5 g
        # tab swung up past its stop under a 10 N pull (the cable left under it). A stiff limit and a little armature.
        out.append('<joint name="jb%d%s" type="hinge" axis="%d 0 0" range="0 %.3f" stiffness="%.6f" damping="0.0005" armature="1e-5" '
                   'solreflimit="0.004 1" solimplimit="0.9999 0.9999 0.001"/>' % (i, nm, sgn, BARB_RANGE, BARB_SPRING))
        out.append('<geom name="barb%d%s" type="box" size="%.6f %.6f %.6f" pos="0 %.6f 0" rgba=".92 .92 .88 1" mass="0.0005" '
                   'contype="2" conaffinity="2" priority="1" %s/>' % (i, nm, BARB_W, L / 2, BARB_T, -sgn * L / 2,
                                                          st.replace('solimp="0.95 0.99 0.0005"', 'solimp="0.999 0.9999 0.0003"').replace('friction="0.3"', 'friction="0.3 0.005 0.0001"')))
        out.append('</body>')
    out.append('</body>')
    return "\n    ".join(out)


def _xml(ep):
    R = ep.radius
    seg = ep.length / N_SEG
    ch, sh = math.cos(ep.conn_heading), math.sin(ep.conn_heading)
    verts = " ".join("%.6f %.6f 0" % (i * seg * ch, i * seg * sh) for i in range(N_SEG + 1))
    fx = []
    for i, f in enumerate(ep.fixtures):
        if f["type"] == "clip":
            fx.append(_clip_xml(i, f, R))
        else:
            x, y = f["pos"]
            st = 'friction="0.3" solref="0.003 1" solimp="0.95 0.99 0.0005"'
            fx.append('<geom name="post%d" type="cylinder" pos="%.6f %.6f %.6f" size="%.6f %.6f" rgba=".6 .5 .3 1" %s/>'
                      % (i, x, y, BOARD_T + POST_H / 2, POST_R, POST_H / 2, st))
            fx.append('<geom name="cap%d" type="cylinder" pos="%.6f %.6f %.6f" size="%.6f 0.003" rgba=".1 .1 .1 1" %s/>'
                      % (i, x, y, BOARD_T + POST_H + 0.003, POST_CAP_R, st))
    fur = []
    for name, kind, c, size in FURNITURE:
        fur.append('<geom name="%s" type="%s" pos="%.4f %.4f %.4f" size="%s" rgba=".3 .35 .6 1" friction="0.6 0.01 0.001"/>'
                   % (name, kind, c[0], c[1], c[2], " ".join("%.4f" % v for v in size)))
    bx = ((BENCH_X[1] - BENCH_X[0]) / 2, (BENCH_Y[1] - BENCH_Y[0]) / 2)
    return """
<mujoco model="harness_cell">
  <compiler angle="radian"/>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="{dt}" integrator="implicitfast" cone="elliptic" impratio="{impratio}" noslip_iterations="{noslip}" iterations="{iters}"/>
  <size memory="128M"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0.5 0 1.8" dir="0 0 -1" diffuse=".9 .9 .9" castshadow="false"/>
    <camera name="overhead" pos="{ocx} {ocy} {ocz}" xyaxes="0 -1 0 1 0 0" fovy="{ofov}"/>
    <geom name="floor" type="plane" size="4 4 .1" pos="0 0 {fz}" rgba=".5 .5 .5 1"/>
    <geom name="bench" type="box" size="{bx} {by} 0.02" pos="{bcx} {bcy} -0.02" rgba=".8 .68 .5 1" friction="0.6 0.01 0.001"/>
    <geom name="board" type="box" size="{bhx} {bhy} {bt2}" pos="{bdx} {bdy} {bt2}" rgba=".93 .92 .88 1" friction="{fr} 0.01 0.001"/>
    {furniture}
    <body name="connector" pos="{ax} {ay} {zc}">
      <geom name="connector_shell" type="box" size="0.012 0.009 0.008" pos="{cx} {cy} 0.0035" euler="0 0 {hd}" rgba=".12 .12 .14 1" contype="0" conaffinity="0"/>
      <body name="strain_relief" pos="0 0 0">
      <joint name="strain_relief" type="slide" axis="{hx} {hy} 0" range="0 {bt}" stiffness="{bk}" damping="{bd_}" armature="0.002"/>
      <geom name="boot" type="capsule" fromto="0 0 0 {b1x} {b1y} 0" size="{br}" rgba=".1 .1 .1 1" mass="0.004" contype="0" conaffinity="0"/>
      <composite prefix="c" type="cable" vertex="{verts}" initial="ball">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="{tw}"/><config key="bend" value="{bd}"/><config key="vmax" value="0"/>
        </plugin>
        <joint kind="main" damping="{jd}" armature="1e-5"/>
        <geom type="capsule" size="{R}" rgba=".9 .35 .1 1" contype="3" conaffinity="3" condim="4" friction="{fr} 0.01 0.001" density="{rho}" solref="0.004 1" solimp="0.95 0.99 0.0005"/>
      </composite>
      </body>
    </body>
    {fixtures}
  </worldbody>
</mujoco>""".format(dt=TIMESTEP, ocx=OVERHEAD_POS[0], ocy=OVERHEAD_POS[1], ocz=OVERHEAD_POS[2], ofov=OVERHEAD_FOVY, iters=int(SOLVER_ITERATIONS), impratio=IMPRATIO, noslip=NOSLIP_ITERATIONS, jd=JOINT_DAMPING_40 * N_SEG / 40.0, fz=-BENCH_H, bx=bx[0], by=bx[1], bcx=(BENCH_X[0] + BENCH_X[1]) / 2, bcy=(BENCH_Y[0] + BENCH_Y[1]) / 2,
                    bhx=BOARD_HALF[0], bhy=BOARD_HALF[1], bt2=BOARD_T / 2, bdx=BOARD_C[0], bdy=BOARD_C[1], fr=ep.friction,
                    furniture="\n    ".join(fur), ax=ep.conn_pos[0], ay=ep.conn_pos[1], zc=BOARD_T + R, cx=-0.012 * ch, cy=-0.012 * sh,
                    hd=ep.conn_heading, hx=ch, hy=sh, bt=BOOT_TRAVEL, bk=BOOT_STIFFNESS, bd_=BOOT_DAMPING, b1x=0.02 * ch, b1y=0.02 * sh, br=R + 0.001, verts=verts, tw=ep.twist, bd=ep.bend, R=R, rho=ep.density, fixtures="\n    ".join(fx))


def build_model(ep):
    w = mujoco.MjSpec.from_string(_xml(ep))
    a = robosuite_assets()
    robot = mujoco.MjSpec.from_file(os.path.join(a, "robots", "panda", "robot.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(a, "grippers", "panda_gripper.xml"))
    for cam in list(robot.cameras):
        if cam.name == "robotview":
            robot.delete(cam)
    for nm in ("finger1_pad_collision", "finger2_pad_collision"):
        pg = grip.geom(nm)
        inward = -1.0 if nm.startswith("finger1") else 1.0       # the finger body's y toward the other finger (measured: tools/probe_groove.py)
        base = nm.replace("pad_collision", "")
        plate = pg.parent.add_geom()
        plate.name = base + "jaw"
        plate.type = mujoco.mjtGeom.mjGEOM_BOX
        plate.size = [JAW_HALF_X, 0.0015 + JAW_APEX_IN / 2, 0.0115]
        plate.pos = [pg.pos[0], pg.pos[1] - inward * (0.0015 - JAW_APEX_IN / 2), pg.pos[2] + JAW_V_DZ - 0.0085]
        plate.rgba = [0.1, 0.1, 0.1, 1]
        plate.mass = 0.004
        plate.friction = [JAW_FRICTION, 0.02, 0.0001]
        for tag, sgn in (("lower", 1.0), ("upper", -1.0)):
            f = pg.parent.add_geom()
            f.name = base + "groove_" + tag
            f.type = mujoco.mjtGeom.mjGEOM_BOX
            f.size = [JAW_HALF_X, JAW_V_T, JAW_V_LEN / 2]
            f.pos = [pg.pos[0], pg.pos[1] + inward * (JAW_APEX_IN + (JAW_V_LEN / 2) * math.cos(JAW_V_ANGLE)),
                     pg.pos[2] + JAW_V_DZ + sgn * (JAW_V_LEN / 2) * math.sin(JAW_V_ANGLE)]
            th = -sgn * inward * (math.pi / 2 - JAW_V_ANGLE)
            f.quat = [math.cos(th / 2), math.sin(th / 2), 0.0, 0.0]
            f.friction = [JAW_FRICTION, 0.02, 0.0001]
            f.condim = 4
            f.priority = 1
            f.solref = [0.002, 1.0]                          # a hard face: a squeezed cable must not work through it
            f.solimp = [JAW_SOLIMP_D, max(JAW_SOLIMP_D, 0.999), 0.0003, 0.5, 2.0]
            f.rgba = [0.15, 0.15, 0.17, 1]
            f.mass = 0.001
    robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
    w.worldbody.add_frame(pos=[0.0, 0.0, 0.0]).attach_body(robot.body("base"), "r_", "")
    site = w.body("cB_first").add_site()
    site.name = "conn_site"
    s = w.add_sensor()
    s.name = "connector_force"
    s.type = mujoco.mjtSensor.mjSENS_FORCE
    s.objtype = mujoco.mjtObj.mjOBJ_SITE
    s.objname = "conn_site"
    m = w.compile()
    m.site_rgba[:, 3] = 0.0
    # the hand's grip: robosuite's finger servos (1000 N/m, 20 N) hold a 9 mm cable with under 3 N a finger; a Panda
    # hand grips with tens of newtons. Stiffer servos with a 40 N limit and joint damping (GRIP_KP, GRIP_FORCE).
    for nm in ("r_g_gripper_finger_joint1", "r_g_gripper_finger_joint2"):
        a = m.actuator(nm).id
        m.actuator_gainprm[a, 0] = GRIP_KP
        m.actuator_biasprm[a, 1] = -GRIP_KP
        m.actuator_forcerange[a] = [-GRIP_FORCE, GRIP_FORCE]
    for nm in ("r_g_finger_joint1", "r_g_finger_joint2"):
        m.dof_damping[m.joint(nm).dofadr[0]] = GRIP_DAMPING
    # the hand's colliding geoms also meet the barbs (collision bit 2)
    hand = m.body("r_g_right_gripper").id
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b > 0 and b != hand:
            b = m.body_parentid[b]
        if b == hand and (m.geom_contype[g] or m.geom_conaffinity[g]):
            m.geom_contype[g] |= 2
            m.geom_conaffinity[g] |= 2
    return m


def _crossing(P, a, b, zmax):
    """Arc length at which polyline P first crosses segment a-b in the plane below zmax."""
    s = 0.0
    for j in range(len(P) - 1):
        p, q = P[j, :2], P[j + 1, :2]
        r, w = q - p, b - a
        den = r[0] * w[1] - r[1] * w[0]
        ln = float(np.linalg.norm(r))
        if abs(den) > 1e-12:
            t = ((a[0] - p[0]) * w[1] - (a[1] - p[1]) * w[0]) / den
            u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                z = P[j, 2] + t * (P[j + 1, 2] - P[j, 2])
                if z < zmax:
                    return s + t * ln
        s += ln
    return None


def evaluate(ep, P, damaged=False, peak_mean=0.0, peak=0.0):
    """Which fixtures the settled cable satisfies, in route order. P: (N+1, 3) cable points (world)."""
    checks = []
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    for f in ep.fixtures:
        if f["type"] == "clip":
            a = f["axis"]
            n = np.array([-math.sin(a), math.cos(a)]) * (f["gap"] / 2 + PIN_R)
            s = _crossing(P, f["pos"] - n, f["pos"] + n, BOARD_T + SEATED_Z)
            checks.append({"type": "clip", "ok": s is not None, "s": s})
        else:
            c = f["pos"]
            dist = np.linalg.norm(P[:, :2] - c, axis=1)
            j = int(np.argmin(dist))
            touching = dist[j] <= POST_R + ep.radius + POST_TOL and P[j, 2] < BOARD_T + POST_H
            jj = min(max(j, 1), len(P) - 2)
            tang = P[jj + 1, :2] - P[jj - 1, :2]
            cross = tang[0] * (c[1] - P[j, 1]) - tang[1] * (c[0] - P[j, 0])
            side_ok = (cross > 0) if f["side"] == "left" else (cross < 0)
            ok = bool(touching and side_ok)
            checks.append({"type": "post", "ok": ok, "s": float(arc[j]) if ok else None})
    routed, last = 0, -1.0
    for c in checks:
        if not c["ok"] or c["s"] <= last:
            break
        routed += 1
        last = c["s"]
    n = len(ep.fixtures)
    complete = routed == n and not damaged
    return {"credit": 1.0 if complete else 0.0, "prefix_credit": 0.0 if damaged else routed / float(n), "routed": routed,
            "fixtures": n, "damaged": bool(damaged), "peak_mean_tension": float(peak_mean), "peak_tension": float(peak),
            "unstable": False, "checks": [{"type": c["type"], "ok": bool(c["ok"])} for c in checks]}


class HarnessEnv(object):
    max_steps = MAX_STEPS

    def __init__(self):
        self.model = None
        self.data = None

    # -- construction -------------------------------------------------------------------
    def reset(self, seed, n_fixtures=None):
        self.ep = make_episode(seed, n_fixtures)
        self.model = build_model(self.ep)
        self.data = mujoco.MjData(self.model)
        self.noise = np.random.default_rng(np.random.SeedSequence([self.ep.seed, 2]))
        self._index()
        # a starting layout that does not rest on the board clear of every fixture is drawn again (its own stream)
        redraw = np.random.default_rng(np.random.SeedSequence([self.ep.seed, 3]))
        self.redraws = 0
        for _attempt in range(8):
            self._init_state()
            P = self.cable_points()
            F = np.array([f["pos"] for f in self.ep.fixtures])
            clear = float(np.min(np.linalg.norm(P[:, None, :2] - F[None, :, :], axis=2)))
            if clear > 0.02 and P[:, 2].min() > BOARD_T and P[:, 2].max() < BOARD_T + 0.012:
                break
            nxt = _initial_shape(redraw, self.ep)
            if nxt is None:
                break
            self.redraws += 1
            self.ep.initial_ab = nxt
            self.ep.initial = np.array([to_world(q) for q in nxt])
        return self.observation()

    def _index(self):
        m = self.model
        self.jq = np.array([m.joint("r_joint%d" % i).qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint("r_joint%d" % i).dofadr[0] for i in range(1, 8)])
        self.act = np.array([m.actuator("r_torq_j%d" % i).id for i in range(1, 8)])
        self.gact = [m.actuator("r_g_gripper_finger_joint1").id, m.actuator("r_g_gripper_finger_joint2").id]
        self.gq = [m.joint("r_g_finger_joint1").qposadr[0], m.joint("r_g_finger_joint2").qposadr[0]]
        self.site = m.site("r_g_grip_site").id
        self.ft_force = m.sensor("r_g_force_ee").adr[0]
        self.ft_torque = m.sensor("r_g_torque_ee").adr[0]
        self.ft_site = m.site("r_g_ft_frame").id
        self.wrist_cam = m.camera("r_eye_in_hand").id
        self.seg_ids = [i for i in range(m.nbody) if m.body(i).name.startswith("cB")]
        self.seg_len = self.ep.length / N_SEG
        self.conn_adr = m.sensor("connector_force").adr[0]
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))

    def _init_state(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.jq] = HOME_Q
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = 0.04, -0.04
        # bend the straight cable into the episode's starting layout
        P0 = self.ep.initial
        heads = np.arctan2(np.diff(P0[:, 1]), np.diff(P0[:, 0]))
        mujoco.mj_forward(m, d)
        up = d.xmat[self.seg_ids[0]].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        for j in range(m.njnt):
            if not m.joint(j).name.startswith("cJ_"):
                continue
            i = self.seg_ids.index(m.jnt_bodyid[j])
            before = heads[i - 1] if i > 0 else self.ep.conn_heading
            dh = (heads[i] - before + math.pi) % (2 * math.pi) - math.pi
            adr = m.jnt_qposadr[j]
            d.qpos[adr:adr + 4] = np.concatenate([[math.cos(dh / 2)], math.sin(dh / 2) * up])
        mujoco.mj_forward(m, d)
        self.grip_target = 0.04
        self.q_null = HOME_Q.copy()
        self.window = np.zeros(int(round(TENSION_WINDOW / TIMESTEP)))
        self.wi = 0
        self.wn = 0
        self.mean_tension = 0.0
        self.peak_mean = 0.0
        self.peak = 0.0
        self.damaged = False
        self.t = 0
        for _ in range(750):                       # 1.5 s: the arm holds home, the cable comes to rest
            d.ctrl[self.act] = 200.0 * (HOME_Q - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.gact] = [0.04, -0.04]
            mujoco.mj_step(m, d)
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()

    # -- control ------------------------------------------------------------------------
    def _osc_torque(self):
        m, d = self.model, self.data
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
        J = np.vstack([self._jp, self._jr])[:, self.jv]
        mujoco.mj_fullM(m, d, self._M)
        Mq = self._M[np.ix_(self.jv, self.jv)]
        R = d.site_xmat[self.site].reshape(3, 3)
        ep = self.target_pos - d.site_xpos[self.site]
        Re = self.target_R @ R.T
        eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        v = J @ d.qvel[self.jv]
        kp, ko = 400.0, 200.0
        acc = np.concatenate([kp * ep - 2.0 * math.sqrt(kp) * v[:3], ko * eo - 2.0 * math.sqrt(ko) * v[3:]])
        Minv = np.linalg.inv(Mq)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-2 * np.eye(6))
        tau = J.T @ (Lam @ acc) + d.qfrc_bias[self.jv]
        Jbar = Minv @ J.T @ Lam
        N = np.eye(7) - J.T @ Jbar.T
        dq = d.qvel[self.jv]
        tau += N @ (5.0 * (self.q_null - d.qpos[self.jq]) - 6.0 * dq)
        tau -= 0.5 * dq
        over = np.abs(dq) - 1.5
        tau -= np.where(over > 0, 30.0 * over * np.sign(dq), 0.0)
        return tau

    def _substep(self):
        d = self.data
        d.ctrl[self.act] = self._osc_torque()
        d.ctrl[self.gact] = [self.grip_target, -self.grip_target]
        mujoco.mj_step(self.model, d)
        f = float(np.linalg.norm(d.sensordata[self.conn_adr:self.conn_adr + 3]))
        self.window[self.wi] = f
        self.wi = (self.wi + 1) % len(self.window)
        self.wn = min(self.wn + 1, len(self.window))
        self.peak = max(self.peak, f)
        if self.wn == len(self.window):
            self.mean_tension = float(self.window.mean())
            self.peak_mean = max(self.peak_mean, self.mean_tension)
            if self.mean_tension > TENSION_LIMIT:
                self.damaged = True

    def step(self, action):
        """action: [dx, dy, dz, drx, dry, drz, grip (-1 open .. 1 closed), cams]: a Cartesian step of the grip site
        (world axes) from where it is now."""
        a = np.asarray(action, float)
        d = self.data
        dp, dr = a[:3].copy(), a[3:6].copy()
        n = np.linalg.norm(dp)
        if n > MAX_DPOS:
            dp *= MAX_DPOS / n
        n = np.linalg.norm(dr)
        if n > MAX_DROT:
            dr *= MAX_DROT / n
        self.target_pos = d.site_xpos[self.site] + dp
        self.target_R = axis_angle_mat(dr) @ d.site_xmat[self.site].reshape(3, 3)
        self.grip_target = 0.04 * (1.0 - float(np.clip(a[6], -1, 1))) / 2.0
        for _ in range(SUBSTEPS):
            self._substep()
        self.t += 1
        done = self.t >= MAX_STEPS or self.unstable()
        return self.observation(), done

    def unstable(self):
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return True
        return bool(np.max(np.abs(d.qvel[self.jv])) > MAX_ARM_SPEED)

    # -- state --------------------------------------------------------------------------
    def cable_points(self):
        d = self.data
        pts = [d.xpos[i].copy() for i in self.seg_ids]
        last = self.seg_ids[-1]
        pts.append(d.xpos[last] + d.xmat[last].reshape(3, 3)[:, 0] * self.seg_len)
        return np.array(pts)

    def wrench_world(self):
        d = self.data
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        return R @ d.sensordata[self.ft_force:self.ft_force + 3], R @ d.sensordata[self.ft_torque:self.ft_torque + 3]

    def observation(self):
        """The cell's own sensors: joint encoders, the hand's opening, the grip site's pose from the encoders, the
        wrist force/torque sensor (world axes, with noise). No cable, no fixture."""
        d = self.data
        f, t = self.wrench_world()
        f = f + self.noise.normal(0.0, WRENCH_NOISE[0], 3)
        t = t + self.noise.normal(0.0, WRENCH_NOISE[1], 3)
        return {"t": self.t, "time": round(self.t * CONTROL_DT, 3),
                "q": [round(float(v), 5) for v in d.qpos[self.jq]],
                "dq": [round(float(v), 5) for v in d.qvel[self.jv]],
                "gripper_width": round(float(d.qpos[self.gq[0]] - d.qpos[self.gq[1]]), 5),
                "tcp": {"pos": [round(float(v), 5) for v in d.site_xpos[self.site]],
                        "quat": [round(float(v), 6) for v in quat_from_mat(d.site_xmat[self.site].reshape(3, 3))]},
                "wrench": {"force": [round(float(v), 3) for v in f], "torque": [round(float(v), 4) for v in t]}}

    def job_sheet(self):
        return job_sheet(self.ep)

    def truth(self):
        """Grader side only: the cell as built and the cable."""
        ep = self.ep
        fx = []
        for f in ep.fixtures:
            e = {"type": f["type"], "pos": [float(f["pos"][0]), float(f["pos"][1])]}
            if f["type"] == "clip":
                e["axis"], e["gap"] = float(f["axis"]), float(f["gap"])
            else:
                e["side"] = f["side"]
            fx.append(e)
        return {"fixtures": fx, "connector": {"pos": [float(ep.conn_pos[0]), float(ep.conn_pos[1])], "heading": float(ep.conn_heading)},
                "cable": self.cable_points().tolist(), "radius": ep.radius, "length": ep.length, "EI": ep.EI,
                "friction": ep.friction, "density": ep.density, "tension_mean": self.mean_tension, "damaged": self.damaged,
                "boot_stretch": float(self.data.qpos[self.model.joint("strain_relief").qposadr[0]])}

    # -- scoring ------------------------------------------------------------------------
    def finish(self):
        """Open the hand, lift it 15 cm, let the cable settle, judge the settled cable."""
        ep = self.ep
        if self.unstable():
            return {"credit": 0.0, "prefix_credit": 0.0, "routed": 0, "fixtures": len(ep.fixtures), "unstable": True,
                    "damaged": self.damaged, "peak_mean_tension": self.peak_mean, "peak_tension": self.peak, "checks": []}
        d = self.data
        self.grip_target = 0.04
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()
        for _ in range(int(0.4 / TIMESTEP)):
            self._substep()
        goal = self.target_pos + np.array([0.0, 0.0, 0.15])
        n = int(SETTLE_S / TIMESTEP)
        start = self.target_pos.copy()
        for k in range(n):
            self.target_pos = start + (goal - start) * min(1.0, (k + 1) / (0.6 * n))
            self._substep()
        if self.unstable():
            return {"credit": 0.0, "prefix_credit": 0.0, "routed": 0, "fixtures": len(ep.fixtures), "unstable": True,
                    "damaged": self.damaged, "peak_mean_tension": self.peak_mean, "peak_tension": self.peak, "checks": []}
        return evaluate(ep, self.cable_points(), self.damaged, self.peak_mean, self.peak)
