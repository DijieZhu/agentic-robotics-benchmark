"""ctasks: the two handling skills the grader scores (cloth_material_id_from_video).

Byte-identical copies are used by the hidden grader and shipped to you. It holds no hidden values: the grader's
hidden tests only choose poses, release distances, lifts, speeds and targets for these functions.

Every test starts with the fabric lying flat at rest at a pose (x, y, yaw) on the table and has one control value u
that the robot chooses by planning: `plan(material, mass, test)` simulates the test in cphys (the robot's simulator)
made of the given material and searches u so that the predicted outcome equals the test's target. The grader plans
with YOUR material and then executes the planned u once on the REAL fabric, which is not a cphys simulation (the
`world` argument of `outcome` and `run_test`; the grader passes its own model of the real fabric). The score of the
test depends only on how far the real outcome lands from the target (`test_quality`).

Families (outcome y, control u):
  slide     grasp the middle of the short edge that leads, lift it 15 mm, drag the fabric along the table towards
            the table's front edge, reaching speed u (0.3 to 1.1 m/s), let go while moving; y = how far the fabric's
            leading point ends up beyond the release point.                             tolerance 3 mm
  overhang  grasp the middle of the short edge nearest the table's front edge, lift it a little, pull the fabric
            along its long axis until that edge is u (0.03 to 0.13 m) beyond the table's front edge, wait, let go;
            y = the mean height of that edge above the table top once the fabric has settled (negative: it hangs
            below the table top).                                                       tolerance 6 mm
"""
import numpy as np

import cphys as W

TOL = {"slide": 0.003, "overhang": 0.006}
U_RANGE = {"slide": (0.30, 1.10), "overhang": (0.03, 0.13)}
PLAN_ITERS = 8


# ------------------------------------------------------------------------------------------------------------------
# programs
# ------------------------------------------------------------------------------------------------------------------
def fling_traj(p0, yaw, direction, v, d_acc, d_release, lift=0.015):
    """Grasp at p0, lift by `lift`, accelerate along `direction` (world yaw) to speed v over d_acc, keep the speed,
    open the jaws d_release from the start while still moving, then rise and slow down. Returns (traj, release point)."""
    u = np.array([np.cos(direction), np.sin(direction), 0.0])
    q = W.clamp_quat(yaw, 0.0)
    rows = []

    def add(p, c):
        rows.append(np.r_[p, q, c])

    for _ in range(int(0.1 / W.DT)):
        add(p0, 0)
    for _ in range(int(0.1 / W.DT)):
        add(p0, 1)
    pl = np.asarray(p0, float) + [0, 0, lift]
    n = int(0.3 / W.DT)
    for i in range(n):
        s = (i + 1) / n
        s = s * s * s * (10 - 15 * s + 6 * s * s)
        add(p0 + s * (pl - p0), 1)
    a = v * v / (2 * d_acc)
    t, x = 0.0, 0.0
    while x < d_release:
        t += W.DT
        x = 0.5 * a * t * t if t < v / a else d_acc + v * (t - v / a)
        add(pl + u * x, 1)
    pr = pl + u * x
    for i in range(int(2.0 / W.DT)):
        tt = (i + 1) * W.DT
        add(pr + u * v * 0.1 * (1 - np.exp(-tt / 0.1)) + [0, 0, 0.12 * min(1.0, tt / 0.5)], 0)
    return np.array(rows), pr


def prog_slide(test, u):
    """test: pose, side (+1/-1 short edge), d_release. The drag runs along the fabric's long axis, away from it."""
    pose = test["pose"]
    p, yaw = W.edge_grasp(pose, test["side"])
    direction = pose[2] + (0.0 if test["side"] > 0 else np.pi)
    C, pr = fling_traj(p, yaw, direction, float(u), 0.06, test["d_release"])
    return C, pr, direction


def prog_overhang(test, u):
    """test: pose (fabric centre x, y, yaw; long axis about along x, its +x short edge towards the table's front edge),
    lift (m), speed (m/s). The clamp takes the middle of the +x short edge, lifts it by `lift`, pulls the fabric
    along its long axis at `speed` until that edge's middle is u beyond the table's front edge, waits 0.4 s, opens,
    and moves away. Returns (traj, first recorded step of the settled window)."""
    pose = test["pose"]
    p, yaw = W.edge_grasp(pose, +1)
    d = np.array([np.cos(pose[2]), np.sin(pose[2]), 0.0])
    front_x = pose[0] + np.cos(pose[2]) * W.SIZE_X / 2
    dist = (W.EDGE_X + float(u) - front_x) / np.cos(pose[2])
    up = p + np.array([0.0, 0.0, test["lift"]])
    end = up + d * dist
    t1 = 0.5 + dist / test["speed"]
    away = end + d * 0.03 + np.array([0.0, 0.0, 0.08])
    keys = [(0.0, p, yaw, 0, 0), (0.1, p, yaw, 0, 1), (0.5, up, yaw, 0, 1), (t1, end, yaw, 0, 1),
            (t1 + 0.4, end, yaw, 0, 1), (t1 + 0.401, end, yaw, 0, 0), (t1 + 0.9, away, yaw, 0, 0),
            (t1 + 3.4, away, yaw, 0, 0)]
    C = W.program_to_traj(keys)
    return C, len(C) - int(0.2 / W.DT)


# ------------------------------------------------------------------------------------------------------------------
# outcomes
# ------------------------------------------------------------------------------------------------------------------
def outcome(material, mass, test, u, world=None):
    """Simulate `test` with control u in `world` (default: cphys, the robot's simulator) made of `material`.
    Returns y (float) or None if the simulation diverged."""
    fam = test["fam"]
    S = (world or W).Sim
    if fam == "slide":
        C, pr, direction = prog_slide(test, u)
        sim = S(material, mass)
        sim.place_flat(test["pose"])
        fr = sim.run(C, record_every=len(C) - 1)
        if sim.unstable or len(fr) < 2:
            return None
        d = np.array([np.cos(direction), np.sin(direction)])
        return float((fr[-1][:, :2] @ d).max() - pr[:2] @ d)
    if fam == "overhang":
        C, s0 = prog_overhang(test, u)
        sim = S(material, mass)
        sim.place_flat(test["pose"])
        fr = sim.run(C, record_every=10, start=s0)
        if sim.unstable or not len(fr):
            return None
        nx, ny = (world or W).NX, (world or W).NY
        edge = fr.reshape(len(fr), nx, ny, 3)[:, -1, :, 2]       # the pulled edge: the fabric's +x row of vertices
        return float(edge.mean())
    raise ValueError("unknown family %r" % fam)


def plan(material, mass, test, iters=PLAN_ITERS):
    """Choose u in U_RANGE[fam] so that outcome(material, mass, test, u) is as close as possible to test["target"]:
    the two ends of the range, then Illinois regula falsi. Returns (u, predicted y)."""
    fam, target = test["fam"], float(test["target"])
    lo, hi = U_RANGE[fam]
    tol = 0.25 * TOL[fam]
    pts = []

    def f(u):
        y = outcome(material, mass, test, u)
        pts.append((u, y))
        return y

    ylo, yhi = f(lo), f(hi)
    if ylo is None or yhi is None:
        good = [(u, y) for u, y in pts if y is not None]
        if not good:
            return 0.5 * (lo + hi), None
        u, y = min(good, key=lambda p: abs(p[1] - target))
        return u, y
    a, fa, b, fb = lo, ylo - target, hi, yhi - target
    if fa * fb > 0:                                   # target outside what this model can reach: the closer end
        return (lo, ylo) if abs(fa) <= abs(fb) else (hi, yhi)
    side = 0
    for _ in range(iters):
        if abs(fa) <= tol:
            return a, fa + target
        if abs(fb) <= tol:
            return b, fb + target
        c = (a * fb - b * fa) / (fb - fa)
        yc = f(c)
        if yc is None:
            c = 0.5 * (a + b)
            yc = f(c)
            if yc is None:
                break
        fc = yc - target
        if fc * fb > 0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == +1:
                fb *= 0.5
            side = +1
    u, y = min([(u, y) for u, y in pts if y is not None], key=lambda p: abs(p[1] - target))
    return u, y


def test_quality(test, y_true):
    """1 within the family tolerance of the target, falling linearly to 0 at twice the tolerance."""
    if y_true is None or not np.isfinite(y_true):
        return 0.0
    e = abs(y_true - float(test["target"]))
    return float(np.clip(2.0 - e / TOL[test["fam"]], 0.0, 1.0))


def run_test(material, true_material, mass, test, world=None):
    """Plan with `material` in cphys, execute once with `true_material` in `world`. Returns (quality, info)."""
    u, y_pred = plan(material, mass, test)
    y_true = outcome(true_material, mass, test, u, world)
    return test_quality(test, y_true), {"u": u, "predicted": y_pred, "real": y_true, "target": test["target"]}
