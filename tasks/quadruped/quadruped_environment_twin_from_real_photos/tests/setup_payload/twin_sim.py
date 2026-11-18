"""The standard simulation and the probe protocol for MuJoCo site twins.

One module (stdlib + numpy + mujoco) with every number in twin_sim_config.json next to it. The same code runs
the ground-truth twin, the graded twin and the self-check, so the physics they see is identical.

Roles in a twin (found by standardize):
- units ("items"): top-level bodies with a free joint;
- gates: the bodies carrying a slide joint (or the gate bodies the caller names), each with every body welded to it;
- environment: geoms of bodies welded to the world (the floor, walls, pillars, the dock, ...).
A geom takes part in contacts when its contype or conaffinity is non-zero in the delivered model; visual geoms
(contype = conaffinity = 0) stay inert.

Where motion is measured. A unit's displacement is the displacement of the centroid of its collision geometry (the
geom centres weighted by the volume of their bounding boxes), not of its body frame, so it does not depend on where
a twin puts the body origin. Rotations are the angle of the body's relative rotation, which is frame-independent.
Both are sampled at every step.

Driven bodies (a lifted unit, every gate, and any units the caller asks to hold) are kinematic: they get a large
armature (followed by mj_setConst, without which MuJoCo 3.16 ignores the armature of these simple bodies) and their
qpos/qvel are written before every step, so contacts never push them. No actuators are used. Gates are held at their
photographed position (qpos0) during the settle and the probes. A lift probe runs on its own copy of the model, so the
caller's model is never changed.

What standardize removes. Everything that could hold, push or slow a unit without a contact is cleared: joint
stiffness (linear and polynomial), dof damping (linear and polynomial), dof friction loss and armature, gravity
compensation, geom surface velocity, adhesion and fluid coefficients, and every mjOption field (gravity, wind,
medium, overrides, flags). Elements the standard simulation cannot neutralise make standardize raise TwinSimError:
explicit contact pairs and exclusions, equality constraints, actuators, tendons, flex objects and plugins (the twin
gate rejects them before).

Main entry points:
    roles = standardize(model, gates=None, hold=())        # hold: names, or {name: "world"}
    settle(model, roles, seconds=None, qpos=None)          -> per-unit drift/rotation, flags, support edges
    lift_probe(model, roles, body, qpos=None, speed=1.0)    -> per-unit outcome of lifting one unit
    run_probes(model, roles, qpos=None, speed=1.0)          -> every lift probe (or those named)
    support_edges_now(model, roles, data)                   -> support edges of one state (settle() averages)
    is_static(drift_mm, rot_deg, config)                    -> the static rule of the config
    relax(model, roles, d)                                  -> the quasi-static relaxation settle() starts with
    gate_rule(travel, sign)                                 -> a gate's travel rule for one opening direction
    settle_report(model, gates) / probes_report(sim, ...)   -> check_twin --settle / --probes (no truth)
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

# model elements that can hold or move a unit without a contact and that cannot be switched off field by field
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
    """(pos_mm, rot_deg): a unit is static when its displacement and rotation stay within these over a settle."""
    st = (config or load_config())["static"]
    return float(st["pos_mm"]), float(st["rot_deg"])


def is_static(drift_mm, rot_deg, config=None):
    pos_mm, rot_deg_max = static_rule(config)
    return bool(drift_mm <= pos_mm and rot_deg <= rot_deg_max)


class Roles:
    """Which bodies, joints and geoms play which part, plus the kinematic bookkeeping."""

    def __init__(self, model, gates=None, hold=()):
        m = model
        self.model = m
        self.items = []                     # body ids of the units (top-level free bodies)
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
        if gates is None:
            gate_ids = sorted(set(int(m.jnt_bodyid[j]) for j in slides))
        else:
            gate_ids = []
            for gb in gates:
                b = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, gb)) if isinstance(gb, str) else int(gb)
                if b < 0:
                    raise ValueError("no gate body named %r" % (gb,))
                gate_ids.append(b)
            gate_ids = sorted(set(gate_ids))
        self.gates = gate_ids
        self.gate_joint, self.gate_qadr, self.gate_dadr = {}, {}, {}
        for b in self.gates:
            js = [j for j in slides if m.jnt_bodyid[j] == b]
            if len(js) != 1:
                raise ValueError("gate body %s must carry exactly one slide joint" % self.name(b))
            self.gate_joint[b] = js[0]
            self.gate_qadr[b] = int(m.jnt_qposadr[js[0]])
            self.gate_dadr[b] = int(m.jnt_dofadr[js[0]])
        item_set = set(self.items)
        self.geom_role = np.full(m.ngeom, "", dtype=object)
        self.geom_node = [""] * m.ngeom
        self.geom_item = np.full(m.ngeom, -1, int)          # the unit body a geom belongs to, else -1
        self.geom_gate = np.full(m.ngeom, -1, int)          # the gate body a geom belongs to, else -1
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                self.geom_role[g] = "visual"
                continue
            b = int(m.geom_bodyid[g])
            root = int(m.body_rootid[b])
            gb = self._gate_of(b)
            if root in item_set:
                self.geom_role[g] = "item"
                self.geom_node[g] = self.name(root)
                self.geom_item[g] = root
            elif gb is not None:
                self.geom_role[g] = "gate"
                self.geom_node[g] = "gate:" + self.name(gb)
                self.geom_gate[g] = gb
            elif m.body_weldid[b] == 0:
                self.geom_role[g] = "env"
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                self.geom_node[g] = "env:" + (nm if nm else "geom%d" % g)
            else:
                self.geom_role[g] = "other"
                self.geom_node[g] = "other:%d" % g
        # held units: {body id: "world"}
        self.hold = {}
        pairs = hold.items() if isinstance(hold, dict) else ((h, "world") for h in hold)
        for h, frame in pairs:
            b = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, h)) if isinstance(h, str) else int(h)
            if b not in item_set:
                raise ValueError("only units can be held: %r" % (h,))
            if frame != "world":
                raise ValueError("hold frame must be 'world': %r" % (frame,))
            self.hold[b] = frame
        self._measure_points()

    def _gate_of(self, b):
        """The gate body that body b belongs to (the gate itself or a body welded to it), else None."""
        if not self.gates:
            return None
        m = self.model
        k = b
        while k > 0:
            if k in self.gate_joint:
                return k if int(m.body_weldid[b]) == int(m.body_weldid[k]) else None
            k = int(m.body_parentid[k])
        return None

    def _measure_points(self):
        """Per unit: the colliding geoms whose volume-weighted centre is the unit's measurement point."""
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
        """(n_units, 3) world measurement points: the collision-geometry centroid (the COM without geoms)."""
        P = self.point_W @ d.geom_xpos[self.point_geoms] if len(self.point_geoms) else np.zeros((len(self.items), 3))
        if not self.point_has_geom.all():
            P[~self.point_has_geom] = d.xipos[self.items_arr[~self.point_has_geom]]
        return P

    def quats(self, d):
        return d.xquat[self.items_arr].copy()


def standardize(model, config=None, gates=None, hold=()):
    """Apply the standard simulation to a compiled model in place and return its Roles.

    mjOption is reset to MuJoCo's defaults, then set from the config; every colliding geom gets the standard contact
    parameters and its class bits; every force that does not come from a contact is cleared (see the module
    docstring); every gate joint is made unlimited and kinematic (held at qpos0); held units become kinematic too.
    Body masses and inertias are kept as delivered. Raises TwinSimError for elements that cannot be neutralised."""
    cfg = config or load_config()
    m = model
    bad = ["%s (%d)" % (what, int(getattr(m, n))) for n, what in FORBIDDEN if int(getattr(m, n)) > 0]
    if bad:
        raise TwinSimError("the standard simulation does not accept " + ", ".join(bad))
    roles = Roles(m, gates=gates, hold=hold)
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
        k = {"item": "item", "gate": "gate"}.get(role, "env")
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
    for b in roles.gates:
        m.jnt_limited[roles.gate_joint[b]] = 0
        m.dof_armature[roles.gate_dadr[b]] = kin
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
    """Write the poses of kinematic bodies (held units, every gate at its photographed position) before each step."""

    def __init__(self, roles, d):
        self.roles = roles
        self.hold = [(roles.item_qadr[b], roles.item_dadr[b], d.qpos[roles.item_qadr[b]:roles.item_qadr[b] + 7].copy())
                     for b in sorted(roles.hold)]
        self.gates = [(roles.gate_qadr[b], roles.gate_dadr[b], float(d.qpos[roles.gate_qadr[b]])) for b in roles.gates]

    def apply(self, d):
        for qa, da, q in self.hold:
            d.qpos[qa:qa + 7] = q
            d.qvel[da:da + 6] = 0.0
        for qa, da, q in self.gates:
            d.qpos[qa] = q
            d.qvel[da] = 0.0


def _new_data(model, qpos):
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0 if qpos is None else qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    return d


def _item_poses(roles, d):
    """{body: (measurement point, rotation matrix)} of every unit."""
    P = roles.points(d)
    return {b: (P[i].copy(), d.xmat[b].reshape(3, 3).copy()) for i, b in enumerate(roles.items)}


def _penetrations(model, roles, d, min_mm=0.0):
    """Deepest initial penetration per unit (mm) and the partner node."""
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


def _accumulate_support(model, roles, d, acc, cos_min, acc_pos=None):
    """Add the upward contact force on every dynamic (not held) unit, per (supporter node, supported node). Forces
    on kinematic bodies are the drive's, not support, so they are skipped. With acc_pos, also accumulate per
    (supporter ROLE, supporter node, supported node) the contact positions weighted by that force and the force
    ([weight, weight x position (3), count, position sum (3), force]), so the caller can tell WHERE a unit rests on a
    large supporter (the floor, a dock top, a kerb) and what the supporter is without trusting a name (a unit body
    may be called "env:..." by its author)."""
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
        for ok, gs, key, f in ((dyn2 and nz >= cos_min, g1, (roles.geom_node[g1], roles.geom_node[g2]), F2[2]),
                               (dyn1 and -nz >= cos_min, g2, (roles.geom_node[g2], roles.geom_node[g1]), -F2[2])):
            if not ok:
                continue
            acc[key] = acc.get(key, 0.0) + f
            if acc_pos is not None:
                rkey = (str(roles.geom_role[gs]),) + key
                a = acc_pos.get(rkey)
                if a is None:
                    a = acc_pos[rkey] = np.zeros(9)
                w = max(float(f), 0.0)
                p = np.asarray(con.pos, float)
                a[0] += w
                a[1:4] += w * p
                a[4] += 1.0
                a[5:8] += p
                a[8] += f


def support_edges(model, roles, acc, nsteps, config=None):
    """Edges (supporter, supported) whose mean up-force over nsteps is at least force_frac of the unit's weight."""
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
            out.append({"supporter": a, "supported": b, "force_N": round(float(mean), 6),
                        "weight_frac": round(float(mean / w), 4)})
    return out


def support_pairs(model, roles, acc_pos, nsteps):
    """Every (supporter, supported unit) pair with a positive mean up-force over nsteps, before any threshold: the
    supporter's role ("env", "item", "gate" or "other": from the geom, never from its name), the mean force (N), the
    supported unit's weight (N) and the mean contact point (world frame of the twin; weighted by the up-force, the
    plain mean when the force-weight is 0). The grader merges the pairs whose supporters it maps onto the same part
    (several floor tiles are one floor) and applies the force_frac rule to the merged force."""
    g = abs(float(model.opt.gravity[2])) or 9.81
    mass = {roles.name(b): float(model.body_subtreemass[b]) for b in roles.items}
    out = []
    for key in sorted(acc_pos):
        role, a, b = key
        if a == b and role == "item":
            continue
        p = acc_pos[key]
        mean = float(p[8]) / max(1, nsteps)
        if not mean > 0.0:
            continue
        if p[4] <= 0:
            pt = None
        elif p[0] > 0:
            pt = [float(v) for v in p[1:4] / p[0]]
        else:
            pt = [float(v) for v in p[5:8] / p[4]]
        out.append({"supporter": a, "supporter_role": role, "supported": b, "force_N": float(mean),
                    "weight_N": mass[b] * g, "point": pt})
    return out


def support_edges_now(model, roles, data, config=None):
    """Support edges from the contact forces of one state (no time average): the same rule as settle() uses."""
    cfg = config or roles.config
    acc = {}
    _accumulate_support(model, roles, data, acc, float(cfg["support"]["up_cos"]))
    return support_edges(model, roles, acc, 1, cfg)


# ---------------------------------------------------------------------------------------------------------------
# the standard settle: a quasi-static relaxation, then the dynamic settle

def _finite(d):
    return bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))


def relax(model, roles, d, seconds=None, config=None, kin=None, on_step=None):
    """The quasi-static relaxation (in place on d): for `seconds` of simulated time (config settle.relax_s), step the
    physics and set every velocity to zero after each step, with the gates and held units kinematic and the units
    free. A penetration is pushed out a little per step (no launch) and a small gap closes without an impact; a
    unit far above its support only sinks by about g dt^2 per step (about 2 cm in 1 s at 2 ms). on_step(d) is
    called after every step. Returns {"steps", "finite"}."""
    cfg = config or roles.config
    seconds = float(cfg["settle"].get("relax_s", 0.0) if seconds is None else seconds)
    n = int(round(seconds / model.opt.timestep))
    kin = kin or _Kin(roles, d)
    finite = True
    for _k in range(n):
        kin.apply(d)
        mujoco.mj_step(model, d)
        if not _finite(d):
            finite = False
            break
        d.qvel[:] = 0.0
        if on_step is not None:
            on_step(d)
    kin.apply(d)
    return {"steps": n, "finite": finite}


def settle(model, roles, seconds=None, qpos=None, config=None, support=True, relax_s=None):
    """The standard settle from qpos (default model.qpos0, the submitted state) with the gates and held units
    kinematic: first the quasi-static relaxation (relax(), settle.relax_s), then the dynamic settle (settle.seconds).

    Motion is measured at every step of BOTH phases against the submitted state (so what the relaxation moved counts
    as drift): per unit the displacement of its measurement point (mm) and its rotation (deg) at the end and their
    maxima, the static flag (the maxima within the static rule), and per phase: relax_drift_mm / relax_rot_deg (the
    end of the relaxation against the submitted state) and settle_drift_mm / settle_rot_deg / settle_max_drift_mm /
    settle_max_rot_deg (the dynamic phase against the relaxed state). Also: instability flags (non-finite state,
    MuJoCo warnings, an explosion inside explode_window_s of the dynamic phase), the initial penetrations (submitted
    state), the support edges and support pairs averaged over the last avg_last_s, and the final qpos."""
    cfg = config or roles.config
    seconds = float(cfg["settle"]["seconds"] if seconds is None else seconds)
    relax_s = float(cfg["settle"].get("relax_s", 0.0) if relax_s is None else relax_s)
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
    state = {"dr": np.zeros(ni), "rr": np.zeros(ni)}

    def track(dd):
        state["dr"] = 1000.0 * np.linalg.norm(roles.points(dd) - P0, axis=1)
        state["rr"] = _quat_angles_deg(Q0, roles.quats(dd))
        np.maximum(maxd, state["dr"], out=maxd)
        np.maximum(maxr, state["rr"], out=maxr)

    rx = relax(model, roles, d, seconds=relax_s, config=cfg, kin=kin, on_step=track)
    finite = rx["finite"]
    relax_d, relax_r = state["dr"].copy(), state["rr"].copy()
    P1, Q1 = roles.points(d), roles.quats(d)
    smax, srmax = np.zeros(ni), np.zeros(ni)
    sd, sr = np.zeros(ni), np.zeros(ni)
    acc, acc_pos = {}, {}
    exploded = False
    vmax = float(cfg["settle"]["explode_speed_m_s"])
    dmax_mm = float(cfg["settle"]["explode_disp_mm"])
    dadr = np.array([roles.item_dadr[b] for b in roles.items], int)
    if finite:
        for k in range(n):
            kin.apply(d)
            mujoco.mj_step(model, d)
            if not _finite(d):
                finite = False
                break
            track(d)
            sd = 1000.0 * np.linalg.norm(roles.points(d) - P1, axis=1)
            sr = _quat_angles_deg(Q1, roles.quats(d))
            np.maximum(smax, sd, out=smax)
            np.maximum(srmax, sr, out=srmax)
            if k < nexp and ni and not exploded:
                v = np.linalg.norm(d.qvel[dadr[:, None] + np.arange(3)[None, :]], axis=1)
                if (sd > dmax_mm).any() or (v > vmax).any():
                    exploded = True
            if support and k >= n - nwin:
                _accumulate_support(model, roles, d, acc, float(cfg["support"]["up_cos"]), acc_pos)
    warn = {str(mujoco.mjtWarning(i)).split(".")[-1]: int(d.warning[i].number) for i in range(len(d.warning))
            if d.warning[i].number}
    items = {}
    for i, b in enumerate(roles.items):
        if finite:
            vals = (state["dr"][i], state["rr"][i], maxd[i], maxr[i], relax_d[i], relax_r[i], sd[i], sr[i], smax[i],
                    srmax[i])
        else:
            vals = (float("inf"),) * 10
        e_d, e_r, m_d, m_r, x_d, x_r, s_d, s_r, s_m, s_mr = (float(v) for v in vals)
        items[roles.name(b)] = {"drift_mm": round(e_d, 4), "rot_deg": round(e_r, 4),
                                "max_drift_mm": round(m_d, 4), "max_rot_deg": round(m_r, 4),
                                "static": is_static(m_d, m_r, cfg), "held": b in roles.hold,
                                "relax_drift_mm": round(x_d, 4), "relax_rot_deg": round(x_r, 4),
                                "settle_drift_mm": round(s_d, 4), "settle_rot_deg": round(s_r, 4),
                                "settle_max_drift_mm": round(s_m, 4), "settle_max_rot_deg": round(s_mr, 4)}
    out = {"seconds": seconds, "steps": n, "relax_s": relax_s, "relax_steps": rx["steps"], "finite": finite,
           "exploded": exploded, "warnings": warn, "unstable": (not finite) or exploded or bool(warn), "items": items,
           "initial_penetration_mm": {k: v[0] for k, v in pen.items()},
           "initial_penetration_partner": {k: v[1] for k, v in pen.items()},
           "qpos_end": d.qpos.copy()}
    if support:
        out["support_edges"] = support_edges(model, roles, acc, nwin, cfg) if finite else []
        out["support_pairs"] = support_pairs(model, roles, acc_pos, nwin) if finite else []
    return out


# ---------------------------------------------------------------------------------------------------------------
# probes

def classify(d_mm, rot_deg, driven_mm, config, lift=True):
    """carried, stayed, disturbed or intermediate for one unit's displacement d_mm (3,) and rotation."""
    o = config["outcome"]
    d_mm = np.asarray(d_mm, float)
    dist = float(np.linalg.norm(d_mm))
    follows = float(np.linalg.norm(d_mm - np.asarray(driven_mm, float))) <= o["carried_tol_mm"] and \
        rot_deg < o["follow_max_deg"]
    if lift:
        follows = follows and d_mm[2] >= o["carried_rise_mm"]
    if follows:
        return "carried"
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
    """Lift one unit kinematically by lift_mm along world +z (minimum-jerk over rise_s / speed, then hold_s);
    report every other unit's displacement, rotation and outcome. Runs on a copy of the model whose lifted unit has
    the kinematic armature (made effective by mj_setConst); the caller's model is not changed."""
    cfg = config or roles.config
    b = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)) if isinstance(body, str) else int(body)
    if b not in roles.items:
        raise ValueError("not a unit: %r" % (body,))
    if b in roles.hold:
        raise ValueError("a held unit is never lifted: %r" % (body,))
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
            "hold_s": Th, "speed": float(speed), "finite": finite, "outcomes": res}


def gate_centre(model, roles, d, body):
    """Centre of a gate leaf's collision geometry: the middle of the box around its geoms' bounding boxes (world, at
    d's state). Directions along the rail are measured from here, never from the gate body's origin."""
    b = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)) if isinstance(body, str) else int(body)
    P = []
    for g in range(model.ngeom):
        if roles.geom_role[g] != "gate" or int(roles.geom_gate[g]) != b:
            continue
        c, h = np.asarray(model.geom_aabb[g][:3], float), np.asarray(model.geom_aabb[g][3:], float)
        R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    P.append(np.asarray(d.geom_xpos[g], float) + R @ (c + h * np.array([sx, sy, sz])))
    if not P:
        return np.asarray(d.xpos[b], float)
    P = np.asarray(P)
    return 0.5 * (P.min(axis=0) + P.max(axis=0))


def run_probes(model, roles, qpos=None, speed=1.0, config=None, bodies=None):
    """Every lift probe (or those in bodies), from the same start state. Held units are never lifted."""
    out = {}
    names = roles.item_names() if bodies is None else list(bodies)
    for nm in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) in roles.hold:
            continue
        r = lift_probe(model, roles, nm, qpos=qpos, speed=speed, config=config)
        out[r["probe"]] = r
    return out


# ---------------------------------------------------------------------------------------------------------------
# the gate travel rule in one opening direction

def gate_rule(travel, sign, rule=None, scale=1.0, true_closing_m=0.0):
    """The travel rule of a sliding gate (contract; C5's gate term) for ONE opening direction.

    travel: the measurement of the gate on the twin (twin_gate.gate_travel: limited, range_m, qpos0_m, width_m, in the
    twin's own units); sign: +1 when the gate opens toward +joint axis, -1 toward -axis; rule: min_width_ratio,
    max_width_ratio, min_open_m, max_closing_m (default: this config's gate_rule); scale: metres per twin unit (the
    grader's s*; 1 in check_twin), applied to the two lengths (the width ratio has no unit); true_closing_m: how far
    the real gate itself can travel beyond its photographed position against the opening direction, in metres (the
    grader passes the true gate's; 0 for a gate the photos show shut, and check_twin, which has no truth, uses 0).
    The rule: the slide is limited; the photographed position (qpos0) is inside the range; the total travel is
    min_width_ratio..max_width_ratio x the leaf's width along the slide (the extent of its collision geometry, never
    its body origin); the leaf can open at least min_open_m beyond qpos0 in the opening direction; and it can travel
    at most max_closing_m + true_closing_m beyond qpos0 the other way (the closing direction), so a range symmetric
    about the photographed position cannot pass in both directions.
    Returns {"ok", "reasons", "sign", "open_m" (twin units), "open_site_m", "close_m" (twin units), "close_site_m",
    "close_allowed_m", "travel_over_width"}."""
    r = rule or load_config()["gate_rule"]
    eps = 1e-6
    reasons = []
    sign = int(sign)
    allow = float(r["max_closing_m"]) + max(0.0, float(true_closing_m))
    none = {"open_m": None, "open_site_m": None, "close_m": None, "close_site_m": None, "close_allowed_m": allow,
            "travel_over_width": None}
    if sign not in (1, -1):
        return dict(none, ok=False, reasons=["no opening direction"], sign=sign)
    width = travel.get("width_m")
    if not travel.get("limited") or travel.get("range_m") is None:
        reasons.append("the gate slide has no limits")
        return dict(none, ok=False, reasons=reasons, sign=sign)
    lo, hi = (float(v) for v in travel["range_m"])
    q0 = float(travel["qpos0_m"])
    tot = hi - lo
    ratio = (tot / float(width)) if width else None
    if not (lo - eps <= q0 <= hi + eps):
        reasons.append("the photographed position (qpos0 %.3f) is outside the slide range [%.3f, %.3f]" % (q0, lo, hi))
    if ratio is None:
        reasons.append("the leaf has no width along the slide")
    else:
        if ratio < float(r["min_width_ratio"]) - eps:
            reasons.append("the travel is %.3f x the leaf's width (at least %g)" % (ratio, r["min_width_ratio"]))
        if ratio > float(r["max_width_ratio"]) + eps:
            reasons.append("the travel is %.3f x the leaf's width (at most %g)" % (ratio, r["max_width_ratio"]))
    open_m = (hi - q0) if sign > 0 else (q0 - lo)
    open_site = open_m * float(scale)
    if open_site < float(r["min_open_m"]) - eps:
        reasons.append("the gate opens only %.3f m beyond its photographed position (at least %g m)" % (
            open_site, r["min_open_m"]))
    close_m = (q0 - lo) if sign > 0 else (hi - q0)
    close_site = close_m * float(scale)
    if close_site > allow + eps:
        reasons.append("the gate travels %.3f m beyond its photographed position in the closing direction (at most "
                       "%g m%s)" % (close_site, r["max_closing_m"], (" beyond the real gate's own %.3f m" % float(
                           true_closing_m)) if float(true_closing_m) > 0 else ""))
    return {"ok": not reasons, "reasons": reasons, "sign": sign, "open_m": round(open_m, 9),
            "open_site_m": round(open_site, 9), "close_m": round(close_m, 9), "close_site_m": round(close_site, 9),
            "close_allowed_m": allow, "travel_over_width": None if ratio is None else round(ratio, 6)}


# ---------------------------------------------------------------------------------------------------------------
# back ends of check_twin --settle / --probes (no truth: the agent's own twin, exactly as the grader simulates it)

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def settle_report(model, gates=(), hold=(), config=None):
    """check_twin --settle: the standard settle of a physics model (twin_gate.physics_model(gate result): excluded
    units removed), the gates held at qpos0. Per unit: the initial penetration (mm, and into what), the displacement
    and rotation at the end of the relaxation and of the dynamic settle, the maxima over both phases against the
    submitted pose, the static flag; the support edges (the agent's own geom and body names: the grader maps the
    environment supporters onto the true floor / dock / kerb by where the unit rests). Returns (report, sim) where sim
    = (model, roles, config, settle result) feeds probes_report; the model is standardised in place."""
    cfg = config or load_config()
    roles = standardize(model, config=cfg, gates=list(gates) or None, hold=hold)
    st = settle(model, roles, config=cfg)
    pen, part = st["initial_penetration_mm"], st["initial_penetration_partner"]
    units = {}
    for nm, e in st["items"].items():
        u = dict(e)
        u["initial_penetration_mm"] = pen.get(nm, 0.0)
        u["initial_penetration_into"] = part.get(nm)
        units[nm] = u
    rep = {k: v for k, v in st.items() if k not in ("qpos_end", "items", "support_pairs")}
    rep["units"] = units
    rep["static_rule"] = {"pos_mm": static_rule(cfg)[0], "rot_deg": static_rule(cfg)[1]}
    rep["held"] = sorted(roles.name(b) for b in roles.hold)
    return _jsonable(rep), (model, roles, cfg, st)


def probes_report(sim, bodies=None, gate_travels=None, speed=1.0, rule=None):
    """check_twin --probes: from the settled state of settle_report's sim, every lift probe (or those named; held
    units are never lifted) with every other unit's outcome (stayed units summarised), and for every gate the travel
    rule in BOTH directions of its slide axis, each with its opening and its closing-side travel (open_m, close_m:
    there is no truth here, so the closing side is judged as for a gate the photos show shut; the grader takes the
    opening direction from the real gate: the side on which the rail continues beyond the leaf). gate_travels: {gate
    body: twin_gate.gate_travel dict} (the gate report's "travel" entries)."""
    model, roles, cfg, st = sim
    out = {"lift": {}, "gates": {}}
    if not st["finite"]:
        out["error"] = "the settle went non-finite: no probes"
        return _jsonable(out)
    pr = run_probes(model, roles, qpos=st["qpos_end"], speed=speed, config=cfg, bodies=bodies)
    for name, r in pr.items():
        moved = {k: o for k, o in r["outcomes"].items() if o["outcome"] != "stayed"}
        out["lift"][name] = {"body": r["body"], "finite": r["finite"], "lift_mm": r["lift_mm"],
                             "moved": moved, "n_stayed": len(r["outcomes"]) - len(moved)}
    for gb, tr in sorted((gate_travels or {}).items()):
        if not tr:
            continue
        plus, minus = gate_rule(tr, 1, rule or cfg.get("gate_rule")), gate_rule(tr, -1, rule or cfg.get("gate_rule"))
        out["gates"][gb] = {"leaf_centre": tr.get("leaf_centre"), "axis": tr.get("axis"),
                            "width_m": tr.get("width_m"), "range_m": tr.get("range_m"), "qpos0_m": tr.get("qpos0_m"),
                            "opening_plus_axis": plus, "opening_minus_axis": minus,
                            "closing_side_m": {"+axis": plus.get("close_m"), "-axis": minus.get("close_m")}}
    return _jsonable(out)
