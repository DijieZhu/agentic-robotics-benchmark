"""The standard simulation and the probe protocol for MuJoCo environment twins.

One module (stdlib + numpy + mujoco) with every number in twin_sim_config.json next to it. The same code runs
the ground-truth twin, the graded twin and the self-check, so the physics they see is identical.

Roles in a twin (found by standardize):
- items: top-level bodies with a free joint;
- the drawer: the body carrying the one slide joint (or the body name the caller gives), together with every body
  welded to it (a front panel in a child body belongs to the drawer);
- furniture: geoms of bodies welded to the world.
A geom takes part in contacts when its contype or conaffinity is non-zero in the delivered model; visual geoms
(contype = conaffinity = 0) stay inert.

Where motion is measured. An item's displacement is the displacement of the centroid of its collision geometry (the
geom centres weighted by the volume of their bounding boxes), not of its body frame, so it does not depend on where
a twin puts the body origin. Rotations are the angle of the body's relative rotation, which is frame-independent.
Both are sampled at every step.

Driven bodies (a lifted item, the drawer, and any items the caller asks to hold) are kinematic: they get a large
armature (followed by mj_setConst, without which MuJoCo 3.16 ignores the armature of these simple bodies) and their
qpos/qvel are written before every step, so contacts never push them. No actuators are used. A held item stays put,
or (hold frame "drawer") follows the drawer's translation during drawer probes. A lift probe runs on its own copy of
the model, so the caller's model is never changed.

What standardize removes. Everything that could hold, push or slow an item without a contact is cleared: joint
stiffness (linear and polynomial), dof damping (linear and polynomial), dof friction loss and armature, gravity
compensation, geom surface velocity, adhesion and fluid coefficients, and every mjOption field (gravity, wind,
medium, overrides, flags). Elements the standard simulation cannot neutralise make standardize raise TwinSimError:
explicit contact pairs and exclusions, equality constraints, actuators, tendons, flex objects and plugins (a twin
gate must reject them before).

Main entry points:
    roles = standardize(model, drawer=None, hold=())     # hold: names, or {name: "world" | "drawer"}
    settle(model, roles, seconds=None, qpos=None)               -> per-item drift/rotation, flags, support edges
    lift_probe(model, roles, body, qpos=None, speed=1.0)         -> per-item outcome of lifting one item
    drawer_probe(model, roles, direction="out", qpos=None, speed=1.0)
    run_probes(model, roles, qpos=None, speed=1.0)               -> every lift probe plus the drawer pull-out
    support_edges_now(model, roles, data)                        -> support edges of one state (settle() averages)
    is_static(drift_mm, rot_deg, config)                         -> the static rule of the config
"""
import copy
import json
import math
import os

import mujoco
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_sim_config.json")
_CONFIG = None

_INTEGRATORS = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
_CONES = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}
_SOLVERS = {"pgs": mujoco.mjtSolver.mjSOL_PGS, "cg": mujoco.mjtSolver.mjSOL_CG, "newton": mujoco.mjtSolver.mjSOL_NEWTON}

# model elements that can hold or move an item without a contact and that cannot be switched off field by field
FORBIDDEN = (("npair", "explicit contact pairs (<contact><pair>)"),
             ("nexclude", "contact exclusions (<contact><exclude>)"),
             ("neq", "equality constraints"), ("nu", "actuators"), ("ntendon", "tendons"),
             ("nflex", "flex objects"), ("nplugin", "plugins"))


class TwinSimError(ValueError):
    """The model contains something the standard simulation does not accept."""


def load_config(path=None):
    """The parameters (cached for the default path)."""
    global _CONFIG
    if path is None:
        if _CONFIG is None:
            with open(CONFIG_PATH) as f:
                _CONFIG = json.load(f)
        return _CONFIG
    with open(path) as f:
        return json.load(f)


def _flag(kind, name):
    enum = mujoco.mjtDisableBit if kind == "disable" else mujoco.mjtEnableBit
    prefix = "mjDSBL_" if kind == "disable" else "mjENBL_"
    return int(getattr(enum, prefix + name.upper()))


def static_rule(config=None):
    """(pos_mm, rot_deg): an item is static when its displacement and rotation stay within these over a settle."""
    st = (config or load_config())["static"]
    return float(st["pos_mm"]), float(st["rot_deg"])


def is_static(drift_mm, rot_deg, config=None):
    pos_mm, rot_deg_max = static_rule(config)
    return bool(drift_mm <= pos_mm and rot_deg <= rot_deg_max)


class Roles:
    """Which bodies, joints and geoms play which part, plus the kinematic bookkeeping."""

    def __init__(self, model, drawer=None, hold=()):
        m = model
        self.model = m
        self.items = []                     # body ids of the items (top-level free bodies)
        self.item_qadr, self.item_dadr = {}, {}
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body_parentid[m.jnt_bodyid[j]] == 0:
                b = int(m.jnt_bodyid[j])
                self.items.append(b)
                self.item_qadr[b] = int(m.jnt_qposadr[j])
                self.item_dadr[b] = int(m.jnt_dofadr[j])
        self.items.sort()
        self.items_arr = np.array(self.items, int)
        slides = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
        self.drawer = None
        self.drawer_joint = None
        if drawer is not None:
            self.drawer = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, drawer)) if isinstance(drawer, str) else int(drawer)
            if self.drawer < 0:
                raise ValueError("no drawer body named %r" % drawer)
            js = [j for j in slides if m.jnt_bodyid[j] == self.drawer]
            if len(js) != 1:
                raise ValueError("the drawer body must carry exactly one slide joint")
            self.drawer_joint = js[0]
        elif len(slides) == 1:
            self.drawer_joint = slides[0]
            self.drawer = int(m.jnt_bodyid[slides[0]])
        if self.drawer_joint is not None:
            self.drawer_qadr = int(m.jnt_qposadr[self.drawer_joint])
            self.drawer_dadr = int(m.jnt_dofadr[self.drawer_joint])
        item_set = set(self.items)
        self.geom_role = np.full(m.ngeom, "", dtype=object)
        self.geom_node = [""] * m.ngeom
        self.geom_item = np.full(m.ngeom, -1, int)          # the item body a geom belongs to, else -1
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                self.geom_role[g] = "visual"
                continue
            b = int(m.geom_bodyid[g])
            root = int(m.body_rootid[b])
            if root in item_set:
                self.geom_role[g] = "item"
                self.geom_node[g] = self.name(root)
                self.geom_item[g] = root
            elif self._part_of_drawer(b):
                self.geom_role[g] = "drawer"
                self.geom_node[g] = self.name(self.drawer)
            elif m.body_weldid[b] == 0:
                self.geom_role[g] = "furniture"
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                self.geom_node[g] = "furniture:" + (nm if nm else "geom%d" % g)
            else:
                self.geom_role[g] = "other"
                self.geom_node[g] = "other:%d" % g
        # held items: {body id: "world" | "drawer"}; a "drawer" item follows the drawer's translation in drawer probes
        self.hold = {}
        pairs = hold.items() if isinstance(hold, dict) else ((h, "world") for h in hold)
        for h, frame in pairs:
            b = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, h)) if isinstance(h, str) else int(h)
            if b not in item_set:
                raise ValueError("only items can be held: %r" % (h,))
            if frame not in ("world", "drawer"):
                raise ValueError("hold frame must be 'world' or 'drawer': %r" % (frame,))
            self.hold[b] = frame
        self._measure_points()

    def _part_of_drawer(self, b):
        """True when body b is the drawer or a body welded to it (walking up the parent chain)."""
        if self.drawer is None:
            return False
        m = self.model
        if int(m.body_weldid[b]) != int(m.body_weldid[self.drawer]):
            return False
        while b > 0:
            if b == self.drawer:
                return True
            b = int(m.body_parentid[b])
        return False

    def _measure_points(self):
        """Per item: the colliding geoms whose volume-weighted centre is the item's measurement point."""
        m = self.model
        cols, rows, w = [], [], []
        self.point_geoms = []
        for i, b in enumerate(self.items):
            gs = [g for g in range(m.ngeom) if self.geom_item[g] == b]
            if not gs:
                continue
            vol = np.array([8.0 * float(np.prod(np.maximum(m.geom_aabb[g, 3:6], 0.0))) for g in gs])
            vol = np.where(np.isfinite(vol) & (vol > 0), vol, 0.0)
            if vol.sum() <= 0:
                vol = np.ones(len(gs))
            for g, v in zip(gs, vol / vol.sum()):
                rows.append(i)
                cols.append(len(self.point_geoms))
                w.append(v)
                self.point_geoms.append(g)
        self.point_geoms = np.array(self.point_geoms, int)
        self.point_W = np.zeros((len(self.items), len(self.point_geoms)))
        self.point_W[rows, cols] = w
        self.point_has_geom = self.point_W.sum(1) > 0

    def name(self, body):
        nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body))
        return nm if nm else "body%d" % int(body)

    def item_names(self):
        return [self.name(b) for b in self.items]

    def points(self, d):
        """(n_items, 3) world measurement points: the collision-geometry centroid (the COM without geoms)."""
        P = self.point_W @ d.geom_xpos[self.point_geoms] if len(self.point_geoms) else np.zeros((len(self.items), 3))
        if not self.point_has_geom.all():
            P[~self.point_has_geom] = d.xipos[self.items_arr[~self.point_has_geom]]
        return P

    def quats(self, d):
        return d.xquat[self.items_arr].copy()


def standardize(model, config=None, drawer=None, hold=()):
    """Apply the standard simulation to a compiled model in place and return its Roles.

    mjOption is reset to MuJoCo's defaults, then set from the config; every colliding geom gets the standard contact
    parameters and its class bits; every force that does not come from a contact is cleared (see the module
    docstring); the drawer joint is made unlimited and kinematic; held items become kinematic too. Body masses and
    inertias are kept as delivered. Raises TwinSimError for elements that cannot be neutralised."""
    cfg = config or load_config()
    m = model
    bad = ["%s (%d)" % (what, int(getattr(m, n))) for n, what in FORBIDDEN if int(getattr(m, n)) > 0]
    if bad:
        raise TwinSimError("the standard simulation does not accept " + ", ".join(bad))
    roles = Roles(m, drawer=drawer, hold=hold)
    default = mujoco.MjOption()
    for k in ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance",
              "density", "viscosity", "o_margin", "integrator", "cone", "jacobian", "solver", "iterations",
              "ls_iterations", "noslip_iterations", "ccd_iterations", "disableflags", "enableflags",
              "disableactuator", "sdf_initpoints", "sdf_iterations", "sleep_tolerance"):
        setattr(m.opt, k, getattr(default, k))
    for k in ("gravity", "wind", "magnetic", "o_solref", "o_solimp", "o_friction"):
        getattr(m.opt, k)[:] = getattr(default, k)
    s = cfg["sim"]
    m.opt.timestep = float(s["timestep"])
    m.opt.integrator = int(_INTEGRATORS[s["integrator"].lower()])
    m.opt.cone = int(_CONES[s["cone"].lower()])
    m.opt.impratio = float(s["impratio"])
    m.opt.solver = int(_SOLVERS[s["solver"].lower()])
    m.opt.iterations = int(s["iterations"])
    m.opt.ls_iterations = int(s["ls_iterations"])
    m.opt.tolerance = float(s["tolerance"])
    m.opt.noslip_iterations = int(s["noslip_iterations"])
    m.opt.gravity[:] = s["gravity_m_s2"]
    for f in s["disable_flags"]:
        m.opt.disableflags |= _flag("disable", f)
    for f in s["enable_flags"]:
        m.opt.enableflags |= _flag("enable", f)
    c = cfg["contact"]
    cls = cfg["classes"]
    for g in range(m.ngeom):
        role = roles.geom_role[g]
        if role == "visual":
            continue
        m.geom_condim[g] = int(c["condim"])
        m.geom_solref[g] = c["solref"]
        m.geom_solimp[g] = c["solimp"]
        m.geom_margin[g] = float(c["margin_m"])
        m.geom_gap[g] = float(c["gap_m"])
        m.geom_priority[g] = int(c["priority"])
        m.geom_solmix[g] = float(c["solmix"])
        if c["override_friction"]:
            m.geom_friction[g] = c["friction"]
        k = {"item": "item", "drawer": "drawer"}.get(role, "furniture")
        m.geom_contype[g] = int(cls[k]["contype"])
        m.geom_conaffinity[g] = int(cls[k]["conaffinity"])
    # forces without a contact: springs, dampers, friction loss, armature, gravity compensation, conveyor surfaces,
    # adhesion, fluid drag; all cleared on every element
    m.jnt_stiffness[:] = 0.0
    m.jnt_stiffnesspoly[:] = 0.0
    m.dof_damping[:] = 0.0
    m.dof_dampingpoly[:] = 0.0
    m.dof_frictionloss[:] = 0.0
    m.dof_armature[:] = 0.0
    m.body_gravcomp[:] = 0.0
    m.geom_surfacevel[:] = 0.0
    m.geom_adhesion[:] = 0.0
    m.geom_fluid[:] = 0.0
    kin = float(cfg["kinematic"]["armature"])
    for b in roles.hold:
        a = roles.item_dadr[b]
        m.dof_armature[a:a + 6] = kin
    if roles.drawer_joint is not None:
        m.jnt_limited[roles.drawer_joint] = 0
        m.dof_armature[roles.drawer_dadr] = kin
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    roles.config = cfg
    return roles


# ---------------------------------------------------------------------------------------------------------------
# helpers

def _profile(t, T, L):
    """Minimum-jerk position and velocity at time t of a move of length L over T seconds."""
    if T <= 0 or t >= T:
        return L, 0.0
    if t <= 0:
        return 0.0, 0.0
    tau = t / T
    return L * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5), L / T * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4)


def _rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def _quat_angles_deg(Q0, Q):
    """Rotation angle (deg) between unit quaternions, row by row."""
    dot = np.abs(np.sum(Q0 * Q, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


class _Kin:
    """Write the poses of kinematic bodies before each step."""

    def __init__(self, roles, d):
        self.roles = roles
        self.hold = [(roles.item_qadr[b], roles.item_dadr[b], d.qpos[roles.item_qadr[b]:roles.item_qadr[b] + 7].copy(),
                      frame) for b, frame in roles.hold.items()]
        self.dq0 = d.qpos[roles.drawer_qadr] if roles.drawer_joint is not None else None
        self.axis = d.xaxis[roles.drawer_joint].copy() if roles.drawer_joint is not None else None

    def apply(self, d, drawer_q=None, drawer_v=0.0):
        dq = 0.0 if (drawer_q is None or self.dq0 is None) else drawer_q - self.dq0
        for qa, da, q, frame in self.hold:
            d.qpos[qa:qa + 7] = q
            d.qvel[da:da + 6] = 0.0
            if frame == "drawer" and self.axis is not None:
                d.qpos[qa:qa + 3] = q[:3] + dq * self.axis
                d.qvel[da:da + 3] = drawer_v * self.axis
        if self.dq0 is not None:
            d.qpos[self.roles.drawer_qadr] = self.dq0 if drawer_q is None else drawer_q
            d.qvel[self.roles.drawer_dadr] = drawer_v


def _new_data(model, qpos):
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0 if qpos is None else qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    return d


def _item_poses(roles, d):
    """{body: (measurement point, rotation matrix)} of every item."""
    P = roles.points(d)
    return {b: (P[i].copy(), d.xmat[b].reshape(3, 3).copy()) for i, b in enumerate(roles.items)}


def _penetrations(model, roles, d, min_mm=0.0):
    """Deepest initial penetration per item (mm) and the partner node."""
    out = {}
    for i in range(d.ncon):
        con = d.contact[i]
        depth = -1000.0 * con.dist
        if depth <= min_mm:
            continue
        g1, g2 = int(con.geom[0]), int(con.geom[1])
        for ga, gb in ((g1, g2), (g2, g1)):
            if roles.geom_role[ga] == "item":
                nm = roles.geom_node[ga]
                if depth > out.get(nm, (0.0, ""))[0]:
                    out[nm] = (round(depth, 3), roles.geom_node[gb])
    return out


def _accumulate_support(model, roles, d, acc, cos_min):
    """Add the upward contact force on every dynamic (not held) item, per (supporter node, supported node). Forces
    on kinematic bodies are the drive's, not support, so they are skipped."""
    f6 = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom[0]), int(con.geom[1])
        i1, i2 = roles.geom_item[g1], roles.geom_item[g2]
        dyn1 = i1 >= 0 and i1 not in roles.hold
        dyn2 = i2 >= 0 and i2 not in roles.hold
        if not (dyn1 or dyn2):
            continue
        mujoco.mj_contactForce(model, d, i, f6)
        fr = np.asarray(con.frame).reshape(3, 3)
        F2 = fr.T @ f6[:3]                                  # world force on geom2 from geom1
        nz = fr[0, 2]                                       # normal from geom1 to geom2
        if dyn2 and nz >= cos_min:
            key = (roles.geom_node[g1], roles.geom_node[g2])
            acc[key] = acc.get(key, 0.0) + F2[2]
        if dyn1 and -nz >= cos_min:
            key = (roles.geom_node[g2], roles.geom_node[g1])
            acc[key] = acc.get(key, 0.0) - F2[2]


def support_edges(model, roles, acc, nsteps, config=None):
    """Edges (supporter, supported) whose mean up-force over nsteps is at least force_frac of the item's weight."""
    cfg = config or roles.config
    g = abs(float(model.opt.gravity[2])) or 9.81
    frac = float(cfg["support"]["force_frac"])
    mass = {roles.name(b): float(model.body_subtreemass[b]) for b in roles.items}
    out = []
    for (a, b), f in sorted(acc.items()):
        if a == b:
            continue
        mean = f / max(1, nsteps)
        w = mass[b] * g
        if mean >= frac * w:
            out.append({"supporter": a, "supported": b, "force_N": round(mean, 6), "weight_frac": round(mean / w, 4)})
    return out


def support_edges_now(model, roles, data, config=None):
    """Support edges from the contact forces of one state (no time average): the same rule as settle() uses."""
    cfg = config or roles.config
    acc = {}
    _accumulate_support(model, roles, data, acc, float(cfg["support"]["up_cos"]))
    return support_edges(model, roles, acc, 1, cfg)


# ---------------------------------------------------------------------------------------------------------------
# the standard settle

def settle(model, roles, seconds=None, qpos=None, config=None, support=True):
    """Run the standard settle from qpos (default model.qpos0) with the drawer and held items kinematic.

    Returns per item the displacement of its measurement point (mm) and its rotation (deg) at the end and their
    maxima over every step, the static flag (the maxima within the static rule), instability flags (non-finite state,
    MuJoCo warnings, an explosion inside explode_window_s), the initial penetrations, the support edges averaged over
    the last avg_last_s, and the final qpos."""
    cfg = config or roles.config
    seconds = float(cfg["settle"]["seconds"] if seconds is None else seconds)
    dt = model.opt.timestep
    n = int(round(seconds / dt))
    nwin = int(round(float(cfg["support"]["avg_last_s"]) / dt))
    nexp = int(round(float(cfg["settle"]["explode_window_s"]) / dt))
    d = _new_data(model, qpos)
    pen = _penetrations(model, roles, d)
    kin = _Kin(roles, d)
    P0, Q0 = roles.points(d), roles.quats(d)
    ni = len(roles.items)
    maxd, maxr = np.zeros(ni), np.zeros(ni)
    dr = np.zeros(ni)
    rr = np.zeros(ni)
    acc = {}
    exploded, finite = False, True
    vmax = float(cfg["settle"]["explode_speed_m_s"])
    dmax_mm = float(cfg["settle"]["explode_disp_mm"])
    dadr = np.array([roles.item_dadr[b] for b in roles.items], int)
    for k in range(n):
        kin.apply(d)
        mujoco.mj_step(model, d)
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            finite = False
            break
        dr = 1000.0 * np.linalg.norm(roles.points(d) - P0, axis=1)
        rr = _quat_angles_deg(Q0, roles.quats(d))
        np.maximum(maxd, dr, out=maxd)
        np.maximum(maxr, rr, out=maxr)
        if k < nexp and ni and not exploded:
            v = np.linalg.norm(d.qvel[dadr[:, None] + np.arange(3)[None, :]], axis=1)
            if (dr > dmax_mm).any() or (v > vmax).any():
                exploded = True
        if support and k >= n - nwin:
            _accumulate_support(model, roles, d, acc, float(cfg["support"]["up_cos"]))
    warn = {str(mujoco.mjtWarning(i)).split(".")[-1]: int(d.warning[i].number) for i in range(len(d.warning))
            if d.warning[i].number}
    items = {}
    for i, b in enumerate(roles.items):
        if finite:
            e_d, e_r, m_d, m_r = float(dr[i]), float(rr[i]), float(maxd[i]), float(maxr[i])
        else:
            e_d = e_r = m_d = m_r = float("inf")
        items[roles.name(b)] = {"drift_mm": round(e_d, 4), "rot_deg": round(e_r, 4),
                                "max_drift_mm": round(m_d, 4), "max_rot_deg": round(m_r, 4),
                                "static": is_static(m_d, m_r, cfg), "held": b in roles.hold}
    out = {"seconds": seconds, "steps": n, "finite": finite, "exploded": exploded, "warnings": warn,
           "unstable": (not finite) or exploded or bool(warn), "items": items,
           "initial_penetration_mm": {k: v[0] for k, v in pen.items()},
           "initial_penetration_partner": {k: v[1] for k, v in pen.items()},
           "qpos_end": d.qpos.copy()}
    if support:
        out["support_edges"] = support_edges(model, roles, acc, nwin, cfg) if finite else []
    return out


# ---------------------------------------------------------------------------------------------------------------
# probes

def classify(d_mm, rot_deg, driven_mm, config, lift=True):
    """carried / rides, stayed, disturbed or intermediate for one item's displacement d_mm (3,) and rotation."""
    o = config["outcome"]
    d_mm = np.asarray(d_mm, float)
    dist = float(np.linalg.norm(d_mm))
    follow_tol = o["carried_tol_mm"] if lift else o["ride_tol_mm"]
    follows = float(np.linalg.norm(d_mm - np.asarray(driven_mm, float))) <= follow_tol and rot_deg < o["follow_max_deg"]
    if lift:
        follows = follows and d_mm[2] >= o["carried_rise_mm"]
    if follows:
        return "carried" if lift else "rides"
    if dist < o["stay_mm"] and rot_deg < o["stay_deg"]:
        return "stayed"
    if dist >= o["disturbed_mm"] or rot_deg >= o["disturbed_deg"]:
        return "disturbed"
    return "intermediate"


def _outcomes(roles, d, p0, skip, driven_mm, cfg, lift):
    res = {}
    P = roles.points(d)
    for i, b in enumerate(roles.items):
        if b in skip:
            continue
        dv = 1000.0 * (P[i] - p0[b][0])
        rr = _rot_deg(p0[b][1], d.xmat[b].reshape(3, 3))
        res[roles.name(b)] = {"d_mm": [round(float(x), 3) for x in dv], "rot_deg": round(rr, 3),
                              "outcome": classify(dv, rr, driven_mm, cfg, lift), "held": b in roles.hold}
    return res


def lift_probe(model, roles, body, qpos=None, lift_mm=None, speed=1.0, config=None):
    """Lift one item kinematically by lift_mm along world +z (minimum-jerk over rise_s / speed, then hold_s);
    report every other item's displacement, rotation and outcome. Runs on a copy of the model whose lifted item has
    the kinematic armature (made effective by mj_setConst); the caller's model is not changed."""
    cfg = config or roles.config
    b = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)) if isinstance(body, str) else int(body)
    if b not in roles.items:
        raise ValueError("not an item: %r" % (body,))
    L = float(cfg["lift"]["lift_mm"] if lift_mm is None else lift_mm) / 1000.0
    T = float(cfg["lift"]["rise_s"]) / float(speed)
    Th = float(cfg["lift"]["hold_s"])
    mc = copy.copy(model)
    dt = mc.opt.timestep
    n = int(round((T + Th) / dt))
    qa, da = roles.item_qadr[b], roles.item_dadr[b]
    mc.dof_armature[da:da + 6] = float(cfg["kinematic"]["armature"])
    mujoco.mj_setConst(mc, mujoco.MjData(mc))
    d = _new_data(mc, qpos)
    kin = _Kin(roles, d)
    p0 = _item_poses(roles, d)
    ps, qs = d.qpos[qa:qa + 3].copy(), d.qpos[qa + 3:qa + 7].copy()
    finite = True
    for k in range(n):
        s, sd = _profile(k * dt, T, L)
        d.qpos[qa:qa + 3] = ps + np.array([0.0, 0.0, s])
        d.qpos[qa + 3:qa + 7] = qs
        d.qvel[da:da + 6] = [0.0, 0.0, sd, 0.0, 0.0, 0.0]
        kin.apply(d)
        mujoco.mj_step(mc, d)
        if not np.all(np.isfinite(d.qpos)):
            finite = False
            break
    s, _ = _profile(n * dt, T, L)
    d.qpos[qa:qa + 3] = ps + np.array([0.0, 0.0, s])
    d.qpos[qa + 3:qa + 7] = qs
    mujoco.mj_kinematics(mc, d)
    driven = np.array([0.0, 0.0, 1000.0 * L])
    res = _outcomes(roles, d, p0, {b}, driven, cfg, True) if finite else {}
    return {"probe": "lift:" + roles.name(b), "body": roles.name(b), "lift_mm": 1000.0 * L, "rise_s": T,
            "hold_s": Th, "finite": finite, "outcomes": res}


def drawer_centre(model, roles, d):
    """Centre of the drawer's collision geometry: the middle of the box around its geoms' bounding boxes (world, at
    d's state). "Toward photo_cam" is measured from here, not from the drawer body's origin, which may sit anywhere
    (for example right under the camera, where moving along a level slide does not change its distance)."""
    P = []
    for g in range(model.ngeom):
        if roles.geom_role[g] != "drawer":
            continue
        c, h = np.asarray(model.geom_aabb[g][:3], float), np.asarray(model.geom_aabb[g][3:], float)
        R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    P.append(np.asarray(d.geom_xpos[g], float) + R @ (c + h * np.array([sx, sy, sz])))
    if not P:
        return np.asarray(d.xpos[roles.drawer], float)
    P = np.asarray(P)
    return 0.5 * (P.min(axis=0) + P.max(axis=0))


def drawer_out_sign(model, roles, d=None):
    """+1 if increasing the drawer joint moves the drawer toward photo_cam, else -1 (+1 without a photo_cam): the sign
    of axis . (camera position - drawer_centre); when that is 0, the sign of axis . (the camera's backward axis, its
    +z: toward the viewer)."""
    if d is None:
        d = _new_data(model, None)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "photo_cam")
    if cam < 0:
        return 1.0
    axis = np.asarray(d.xaxis[roles.drawer_joint], float)
    v = float(axis @ (np.asarray(d.cam_xpos[cam], float) - drawer_centre(model, roles, d)))
    if abs(v) < 1e-9:
        v = float(axis @ np.asarray(d.cam_xmat[cam], float).reshape(3, 3)[:, 2])
    return 1.0 if v > 0 else -1.0


def drawer_probe(model, roles, direction="out", qpos=None, dist_mm=None, speed=1.0, config=None):
    """Drive the drawer kinematically out (toward photo_cam) or in by dist_mm (minimum-jerk over move_s / speed,
    then hold_s); report every item's displacement, rotation and outcome (rides / stayed / disturbed). Only the
    pull-out is part of the scored protocol."""
    cfg = config or roles.config
    if roles.drawer_joint is None:
        raise ValueError("the twin has no drawer joint")
    key = "out_mm" if direction == "out" else "in_mm"
    L = float(cfg["drawer"][key] if dist_mm is None else dist_mm) / 1000.0
    T = float(cfg["drawer"]["move_s"]) / float(speed)
    Th = float(cfg["drawer"]["hold_s"])
    dt = model.opt.timestep
    n = int(round((T + Th) / dt))
    d = _new_data(model, qpos)
    sign = drawer_out_sign(model, roles, d) * (1.0 if direction == "out" else -1.0)
    kin = _Kin(roles, d)
    p0 = _item_poses(roles, d)
    q0 = float(d.qpos[roles.drawer_qadr])
    xd0 = d.xpos[roles.drawer].copy()
    finite = True
    for k in range(n):
        s, sd = _profile(k * dt, T, L)
        kin.apply(d, drawer_q=q0 + sign * s, drawer_v=sign * sd)
        mujoco.mj_step(model, d)
        if not np.all(np.isfinite(d.qpos)):
            finite = False
            break
    s, _ = _profile(n * dt, T, L)
    d.qpos[roles.drawer_qadr] = q0 + sign * s
    mujoco.mj_kinematics(model, d)
    driven = 1000.0 * (d.xpos[roles.drawer] - xd0)
    res = _outcomes(roles, d, p0, set(), driven, cfg, False) if finite else {}
    return {"probe": "drawer:" + direction, "direction": direction, "dist_mm": 1000.0 * L, "move_s": T, "hold_s": Th,
            "drawer_d_mm": [round(float(x), 3) for x in driven], "finite": finite, "outcomes": res}


def run_probes(model, roles, qpos=None, speed=1.0, config=None, bodies=None, drawer=("out",)):
    """Every lift probe (or those in bodies) plus the drawer pull-out, from the same start state. Held items are
    never lifted."""
    out = {}
    names = roles.item_names() if bodies is None else list(bodies)
    for nm in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) in roles.hold:
            continue
        r = lift_probe(model, roles, nm, qpos=qpos, speed=speed, config=config)
        out[r["probe"]] = r
    if roles.drawer_joint is not None:
        for direction in drawer:
            r = drawer_probe(model, roles, direction, qpos=qpos, speed=speed, config=config)
            out[r["probe"]] = r
    return out
