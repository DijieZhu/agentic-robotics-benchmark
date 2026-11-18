"""Panel devices and the leak marker of task 07 (cell side; DESIGN v1.2 sections 2.3.1, 2.3.4, 2.3.6; decisions J1, K2,
K3, M13; package P05).  Product version of the gate-G1 prototype `proto/g1_valve_turn/devices.py` + `panel_scene.py`.

Valve (quarter-turn paddle handle; the panel standard's handle).  One body per valve on a hinge about the stem axis:
  * frame (valve body): x = stem axis INTO the panel, y = paddle length at 0 deg (= the viewer's left), z = up at 0 deg;
    angle 0 = OPEN (paddle horizontal), 90 deg = CLOSED (paddle vertical); closing = +rotation about x = clockwise for a
    viewer facing the panel (J1);
  * hard stops at 0 / 90 deg: joint range with stop time constant stop_tc_dt x dt (4 dt, world rule K2);
  * running torque = dry friction (`frictionloss`, friction time constant friction_tc_dt x dt = 2 dt, K2, and the
    friction impedance friction_solimp: the torque needed to start / keep a handle moving is the set value within
    2 %, and below it the handle creeps < 0.005 deg/s - with the kit's default impedance it creeps ~1 deg/s per N m);
  * breakaway: a handle that has rested within rearm.within_deg of an end stop for rearm.rest_s gets the breakaway
    torque as its friction; it drops back to the running torque as soon as it leaves that band (resists only, never
    drives);
  * detents: within detent_zone_deg of either end a constant detent_Nm towards that end (generalised force, refreshed
    every physics step).  It is smaller than the running friction, so it never pulls a released handle back into a
    stop; it adds to the torque needed to leave an end and helps over the last degrees into an end.
  Predicates: closed = angle >= 80 deg, open = angle <= 10 deg; the process model treats < 80 deg as conducting.

Ports (test ports TP-1..6 and the parking ports PARK-A / PARK-B; same receptacle): static guide cone (mouth diameter
-> bore) and a straight bore down to the seat on the face plane, as a ring of wall pieces; sites `<port>_mouth`
(x = insertion direction into the panel) and `<port>_seat`.  The hot-stab state machine is package P07.

Panel: `build_panel(spec, skid, pos_w, R_w, ...)` builds the face, skid frame, guard rail with stand-offs and legs,
the port receptacles and the 18 valves from the skid description of `gen/panel_cad.py` (one source with the drawings).
No appearance geometry (tag plates, paint) goes into the physics model; the render twin adds it (P08).

Marker (DESIGN 2.3.4, M13): a paddle handle (same blade as the valves) with a flat block hung below it by two straps
at the blade ends, behind the jaw tip; flooded housing with a steel ballast plate.  `marker_geometry()`,
`marker_mass_properties()`, `add_marker(vb, ...)` (free body + simple hydro record; held by the gripper at the start).

    from subseacell import devices as DV
    P = DV.build_panel(vb.spec, skid, pos_w, R_w, valves={"c2r0": DV.ValveSpec("BV3", 8.0, 5.0, 0.0), ...})
    sim = uwkit.UnderwaterSim(vb, ...)
    bank = DV.ValveBank(sim.model, sim.data, P["valves"].values())
    every physics step, before mj_step:  bank.tick()
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import mujoco

# Panel-standard handle and valve-mechanism values (DEV/spec/task_values.json: handle.*, valve.*, world_rules.*;
# tests/test_devices.py asserts these equal the task values).  A grader config may override them via configure().
DEFAULTS = dict(
    stem_standoff_m=0.15, paddle_m=(0.160, 0.060, 0.012),       # length, depth along the stem, thickness
    stem_radius_m=0.0125, stem_start_m=0.020,                    # stem capsule starts this far out of the face
    handle_mass_kg=0.40, damping=0.2, armature=0.01,             # N m s / rad, kg m^2 (packing / gearing)
    detent_Nm=1.5, detent_zone_deg=10.0, stops_deg=(0.0, 90.0),
    closed_min_deg=80.0, open_max_deg=10.0,
    rearm_within_deg=0.5, rearm_rest_s=0.05, rearm_rate_deg_s=1.0,
    friction_tc_dt=2.0, stop_tc_dt=4.0, timestep_s=0.002,
    # impedance of the hinge's dry-friction constraint (P05; grader-side world rule next to K2's time constants): with
    # MuJoCo's default (0.9, 0.95, ...) a soft friction constraint lets a handle creep at ~1.1 deg/s per N m of applied
    # torque BELOW the friction torque (it behaves like a damper of ~50 N m s/rad), so neither the breakaway nor the
    # running torque acts as a threshold (a 9.5 N m breakaway left the stop at 2.4 N m; measured, record P2_P05).
    # 0.9999 (MuJoCo's largest impedance): thresholds within 2 % of the set values for torque ramps of 0.1-1 N m/s,
    # creep 0.004 deg/s at 80 % of the friction torque (0.999: 0.04 deg/s, enough to creep out of the 0.5 deg
    # breakaway band under a slow ramp).
    friction_solimp=(0.9999, 0.9999, 0.001, 0.5, 2.0),
    surface_friction=(0.8, 0.02, 0.001), paddle_friction=(1.0, 0.02, 0.002),
)
VALUES = copy.deepcopy(DEFAULTS)


def configure(values: dict):
    """Override device values (keys of DEFAULTS) for this process, e.g. from the grader config."""
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown device values {sorted(unknown)}")
    VALUES.update(copy.deepcopy(values))


def _mat_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.asarray(R, float)).reshape(-1))
    return q


# ==================================================================================================== valves
@dataclass
class ValveSpec:
    name: str
    breakaway_Nm: float = 8.0          # static torque to leave an end at rest (DESIGN: 3-8 N m)
    running_Nm: float = 5.0            # sliding torque (DESIGN: 2-5 N m)
    theta0_deg: float = 0.0            # as-found angle (0 open, 90 closed)
    detent_Nm: float = None            # None: VALUES["detent_Nm"]
    zone_deg: float = None             # None: VALUES["detent_zone_deg"]


def valve_frame(R_panel_w):
    """Valve body frame in world from the panel frame P (columns X right, Y up, Z out of the face):
    x = -Z (into the panel), y = -X (viewer's left), z = Y (up)."""
    R = np.asarray(R_panel_w, float)
    return np.stack([-R[:, 2], -R[:, 0], R[:, 1]], 1)


def add_valve(spec_mj: mujoco.MjSpec, v: ValveSpec, stem_point_w, R_panel_w, dt=None, prefix=""):
    """Add one valve body (stem + paddle collision) to the worldbody.  stem_point_w = where the stem axis meets the
    panel face.  Returns the valve info (names, frame, paddle centre, and the grasp handle description)."""
    V = VALUES
    dt = float(V["timestep_s"] if dt is None else dt)
    L, Dp, T = V["paddle_m"]
    so = V["stem_standoff_m"]
    R = valve_frame(R_panel_w)
    p = np.asarray(stem_point_w, float)
    name = prefix + v.name
    body = spec_mj.worldbody.add_body(name=name, pos=list(p), quat=list(_mat_quat(R)))
    stops = list(V["stops_deg"]) if spec_mj.compiler.degree else [math.radians(x) for x in V["stops_deg"]]
    j = body.add_joint(name=f"{name}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1.0, 0.0, 0.0], range=stops,
                       limited=mujoco.mjtLimited.mjLIMITED_TRUE, frictionloss=float(v.running_Nm),
                       damping=float(V["damping"]), armature=float(V["armature"]), ref=0.0)
    j.solref_friction = [V["friction_tc_dt"] * dt, 1.0]
    j.solimp_friction = list(V["friction_solimp"])
    j.solref_limit = [V["stop_tc_dt"] * dt, 1.0]
    m_h = float(V["handle_mass_kg"])
    body.add_geom(name=f"{name}_stem", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[-V["stem_start_m"], 0, 0, -(so - 0.5 * Dp), 0, 0], size=[V["stem_radius_m"], 0, 0],
                  mass=0.3 * m_h, friction=list(V["surface_friction"]), rgba=[0.55, 0.55, 0.55, 1])
    body.add_geom(name=f"{name}_paddle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5 * Dp, 0.5 * L, 0.5 * T],
                  pos=[-so, 0, 0], mass=0.7 * m_h, friction=list(V["paddle_friction"]), condim=4,
                  rgba=[0.92, 0.25, 0.10, 1])
    centre = p + R @ np.array([-so, 0.0, 0.0])
    handle = dict(name=name, kind="valve", body=name, paddle_geom=f"{name}_paddle", geoms=[f"{name}_paddle", f"{name}_stem"],
                  centre_b=[-so, 0.0, 0.0], R_bp=np.eye(3).tolist(), joint=f"{name}_hinge",
                  paddle_m=[Dp, L, T])
    return dict(name=name, body=name, joint=f"{name}_hinge", paddle_geom=f"{name}_paddle", stem_geom=f"{name}_stem",
                spec=v, R_w=R, stem_point_w=p, paddle_centre_w=centre, handle=handle)


class ValveBank:
    """Run-time behaviour of the valves: initial angles, breakaway (stiction at rest at an end) and detents."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, infos):
        self.m, self.d = model, data
        self.v = {}
        for inf in infos:
            jid = model.joint(inf["joint"]).id
            sp = inf["spec"]
            self.v[inf["name"]] = dict(info=inf, spec=sp, jid=jid, qa=int(model.jnt_qposadr[jid]),
                                       da=int(model.jnt_dofadr[jid]), rest_t=0.0, stuck=False,
                                       theta0=math.radians(sp.theta0_deg), body=model.body(inf["body"]).id,
                                       detent=float(VALUES["detent_Nm"] if sp.detent_Nm is None else sp.detent_Nm),
                                       zone=math.radians(VALUES["detent_zone_deg"] if sp.zone_deg is None else sp.zone_deg))
        self.names = list(self.v)
        self.lo, self.hi = (math.radians(x) for x in VALUES["stops_deg"])
        self.reset()

    def reset(self):
        band = math.radians(VALUES["rearm_within_deg"])
        for s in self.v.values():
            self.d.qpos[s["qa"]] = s["theta0"]
            self.d.qvel[s["da"]] = 0.0
            at_end = min(abs(s["theta0"] - self.lo), abs(s["theta0"] - self.hi)) < band
            s["stuck"] = at_end                     # a handle that starts at an end has been sitting there
            s["rest_t"] = VALUES["rearm_rest_s"] if at_end else 0.0
            self.m.dof_frictionloss[s["da"]] = s["spec"].breakaway_Nm if at_end else s["spec"].running_Nm
            self.d.qfrc_applied[s["da"]] = 0.0

    def tick(self, dt=None):
        """Once per physics step, before mj_step: breakaway state and detent torques."""
        m, d = self.m, self.d
        dt = float(m.opt.timestep if dt is None else dt)
        band, rate = math.radians(VALUES["rearm_within_deg"]), math.radians(VALUES["rearm_rate_deg_s"])
        rest_s = VALUES["rearm_rest_s"]
        for s in self.v.values():
            sp = s["spec"]
            q, w = float(d.qpos[s["qa"]]), float(d.qvel[s["da"]])
            near = min(abs(q - self.lo), abs(q - self.hi)) < band
            s["rest_t"] = s["rest_t"] + dt if (near and abs(w) < rate) else 0.0
            if s["stuck"] and not near:
                s["stuck"] = False
            elif not s["stuck"] and near and s["rest_t"] >= rest_s:
                s["stuck"] = True
            m.dof_frictionloss[s["da"]] = sp.breakaway_Nm if s["stuck"] else sp.running_Nm
            tau = 0.0
            if q < self.lo + s["zone"]:
                tau = -s["detent"]
            elif q > self.hi - s["zone"]:
                tau = s["detent"]
            d.qfrc_applied[s["da"]] = tau

    # ---------------------------------------------------------------------------------------- readout
    def angle_deg(self, name):
        return math.degrees(float(self.d.qpos[self.v[name]["qa"]]))

    def angles_deg(self):
        return {k: math.degrees(float(self.d.qpos[s["qa"]])) for k, s in self.v.items()}

    def rate_deg_s(self, name):
        return math.degrees(float(self.d.qvel[self.v[name]["da"]]))

    def is_closed(self, name):
        return self.angle_deg(name) >= VALUES["closed_min_deg"]

    def is_open(self, name):
        return self.angle_deg(name) <= VALUES["open_max_deg"]

    def conducting(self, name):
        """Process model: a valve conducts unless it is closed (no partial opening)."""
        return not self.is_closed(name)

    def info(self, name):
        return self.v[name]["info"]

    def frame_w(self, name):
        """(paddle centre, rotation of the paddle frame) in world: x = stem axis into the panel, y = paddle length,
        z = paddle normal (vertical when open)."""
        b = self.v[name]["body"]
        R = self.d.xmat[b].reshape(3, 3).copy()
        return self.d.xpos[b] + R @ np.array([-VALUES["stem_standoff_m"], 0.0, 0.0]), R


# ==================================================================================================== panel
def _geom_from_prim(wb, pr, pos_w, R_w, prefix, friction, contact=None):
    pos_w, R_w = np.asarray(pos_w, float), np.asarray(R_w, float)
    kw = dict(friction=list(friction))
    if contact:
        kw.update({k: list(v) if isinstance(v, (list, tuple)) else v for k, v in contact.items()})
    name = prefix + pr["name"]
    if pr["kind"] == "box":
        A = np.eye(3) if pr.get("axes") is None else np.asarray(pr["axes"], float).T
        R = R_w @ A
        return wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(pr["half"]),
                           pos=list(pos_w + R_w @ np.asarray(pr["centre"])), quat=list(_mat_quat(R)),
                           rgba=[0.45, 0.50, 0.55, 1], **kw)
    a = pos_w + R_w @ np.asarray(pr["a"], float)
    b = pos_w + R_w @ np.asarray(pr["b"], float)
    gt = mujoco.mjtGeom.mjGEOM_CAPSULE if pr["kind"] == "capsule" else mujoco.mjtGeom.mjGEOM_CYLINDER
    return wb.add_geom(name=name, type=gt, size=[pr["radius"], 0, 0], fromto=list(a) + list(b),
                       rgba=[0.85, 0.70, 0.10, 1], **kw)


def build_panel(spec_mj: mujoco.MjSpec, skid: dict, pos_w, R_w, valves: dict = None, ports: dict = None,
                prefix="", contact_rules: dict = None, dt=None):
    """Build the panel skid's physics geometry into spec_mj.worldbody.

    skid        : description from gen/panel_cad.panel_skid() (also carried by the site package)
    pos_w, R_w  : pose of the panel frame P in the world (columns of R_w = X, Y, Z of P)
    valves      : {stem key 'c<j>r<r>': ValveSpec} (default: every stem, open, 8 / 5 N m, named by key)
    ports       : {port key 'p<i>': name} (default: the design labels)
    contact_rules: {role: dict(solref=..., solimp=..., condim=...)} world rules per role (e.g. rail, face)
    Returns dict(valves={name: info}, ports={name: info}, geoms={role: [names]}, frame=(pos_w, R_w), skid=skid)."""
    V = VALUES
    wb = spec_mj.worldbody
    pos_w, R_w = np.asarray(pos_w, float), np.asarray(R_w, float)
    contact_rules = contact_rules or {}
    geoms = {}
    for pr in skid["primitives"]:
        if not pr["physics"]:
            continue
        g = _geom_from_prim(wb, pr, pos_w, R_w, prefix, V["surface_friction"], contact_rules.get(pr["role"]))
        geoms.setdefault(pr["role"], []).append(g.name)
    vinfos = {}
    stems = {s["key"]: s for s in skid["stems"]}
    if valves is None:
        valves = {k: ValveSpec(k) for k in stems}
    for key, vs in valves.items():
        s = stems[key]
        inf = add_valve(spec_mj, vs, pos_w + R_w @ np.asarray(s["stem_point"]), R_w, dt=dt, prefix=prefix)
        inf["key"] = key
        vinfos[inf["name"]] = inf
    pinfos = {}
    names = ports or {p["key"]: p["design_label"] for p in skid["ports"]}
    q_port = _mat_quat(np.stack([-R_w[:, 2], -R_w[:, 0], R_w[:, 1]], 1))      # site x = insertion direction
    for p in skid["ports"]:
        nm = prefix + names[p["key"]]
        mouth = pos_w + R_w @ np.asarray(p["mouth"])
        seat = pos_w + R_w @ np.asarray(p["seat"])
        wb.add_site(name=f"{nm}_mouth", pos=list(mouth), quat=list(q_port), size=[0.01, 0, 0], group=4)
        wb.add_site(name=f"{nm}_seat", pos=list(seat), quat=list(q_port), size=[0.01, 0, 0], group=4)
        pinfos[nm] = dict(name=nm, key=p["key"], kind=p["kind"], mouth_w=mouth, seat_w=seat,
                          axis_w=-R_w[:, 2].copy(), mouth_diameter=p["mouth_diameter"], bore_diameter=p["bore_diameter"],
                          insert_depth=p["insert_depth"], geoms=[prefix + f"port{p['index']}_{k}{i:02d}"
                                                                 for k in ("cone", "bore")
                                                                 for i in range(skid["params"]["detail"]["port"]["n_staves"])])
    return dict(valves=vinfos, ports=pinfos, geoms=geoms, frame=(pos_w, R_w), skid=skid,
                handles={n: inf["handle"] for n, inf in vinfos.items()})


# ==================================================================================================== marker
# Geometry in the marker frame M = the paddle frame of its handle: origin = blade centre, x = along the tool axis
# away from the vehicle when held, y = blade length, z = blade normal (up when held level).  Block centre behind the
# blade (x < 0) and below it; straps from the blade ends down to the block's lateral faces.  Materials: blade acetal,
# straps stainless, housing flooded polyethylene (perforated), ballast plate steel on the housing floor.
MARKER = dict(
    blade=dict(size=(0.060, 0.160, 0.012), density=1410.0),
    block=dict(size=(0.120, 0.120, 0.030), centre=(-0.140, 0.0, -0.095), wall=0.002, density=960.0),
    plate=dict(size=(0.110, 0.110, 0.0059), density=7850.0),
    straps=dict(radius=0.004, density=7900.0, top=(-0.010, 0.070, -0.010), bottom=(-0.100, 0.052, -0.084)),
    hydro=dict(cd=1.0, ca_envelope=0.7, half_height=0.05),
    friction=(0.8, 0.02, 0.002),
)


def marker_geometry(params=None):
    """Collision primitives of the marker in M (dicts kind / name / centre / half or a / b / radius)."""
    P = copy.deepcopy(MARKER if params is None else params)
    bl, bk, st = P["blade"], P["block"], P["straps"]
    out = [dict(name="blade", kind="box", centre=[0.0, 0.0, 0.0], half=[x / 2 for x in bl["size"]])]
    for s, sg in (("p", 1.0), ("s", -1.0)):
        a = [st["top"][0], sg * st["top"][1], st["top"][2]]
        b = [st["bottom"][0], sg * st["bottom"][1], st["bottom"][2]]
        out.append(dict(name=f"strap_{s}", kind="capsule", a=a, b=b, radius=st["radius"]))
    out.append(dict(name="block", kind="box", centre=list(bk["centre"]), half=[x / 2 for x in bk["size"]]))
    return out


def _box_inertia(m, size):
    a, b, c = size
    return np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])


def marker_mass_properties(params=None, rho=1024.77, g=9.80665):
    """Mass, CoM, inertia about the CoM, displaced (material) volume, centre of buoyancy, wet weight [N] and the
    flooded inner water volume of the marker, in M."""
    P = MARKER if params is None else params
    bl, bk, pl, st = P["blade"], P["block"], P["plate"], P["straps"]
    parts = []                                                     # (mass, com, I_com, volume)
    V = float(np.prod(bl["size"]))
    parts.append((V * bl["density"], np.zeros(3), _box_inertia(V * bl["density"], bl["size"]), V))
    for sg in (1.0, -1.0):
        a = np.array([st["top"][0], sg * st["top"][1], st["top"][2]])
        b = np.array([st["bottom"][0], sg * st["bottom"][1], st["bottom"][2]])
        L = float(np.linalg.norm(b - a))
        Vs = math.pi * st["radius"] ** 2 * L + 4.0 / 3.0 * math.pi * st["radius"] ** 3
        ms = Vs * st["density"]
        u = (b - a) / L
        I_rod = ms * L * L / 12 * (np.eye(3) - np.outer(u, u))
        parts.append((ms, 0.5 * (a + b), I_rod, Vs))
    c = np.asarray(bk["centre"], float)
    so = np.asarray(bk["size"], float)
    si = so - 2 * bk["wall"]
    Vo, Vi = float(np.prod(so)), float(np.prod(si))
    m_o, m_i = Vo * bk["density"], Vi * bk["density"]
    parts.append((m_o - m_i, c, _box_inertia(m_o, so) - _box_inertia(m_i, si), Vo - Vi))
    Vp = float(np.prod(pl["size"]))
    cp_ = c + np.array([0.0, 0.0, -so[2] / 2 + bk["wall"] + pl["size"][2] / 2])
    parts.append((Vp * pl["density"], cp_, _box_inertia(Vp * pl["density"], pl["size"]), Vp))
    M = sum(p[0] for p in parts)
    Vt = sum(p[3] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / M
    cob = sum(p[3] * p[1] for p in parts) / Vt
    I = np.zeros((3, 3))
    for m_, r_, I_, _ in parts:
        dd = r_ - com
        I += I_ + m_ * (float(dd @ dd) * np.eye(3) - np.outer(dd, dd))
    return dict(mass_kg=M, com_m=com.tolist(), inertia_com_kgm2=I.tolist(), volume_m3=Vt, cob_m=cob.tolist(),
                wet_weight_N=(M - rho * Vt) * g, flooded_water_m3=Vi - Vp, envelope_m3=Vo,
                lowest_z_m=float(c[2] - so[2] / 2))


def add_marker(vb, name="marker", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), params=None, contact=None):
    """Add the marker as a free body (collision geoms + simple hydro record: buoyancy of the material volume, added
    mass of the flooded water + ca_envelope x the envelope, quadratic drag 0.5 rho cd A per axis).  Must be called
    before the simulator is built.  Returns the grasp handle description (subseacell.grasp)."""
    from uwkit.models.vehicle import simple_hydro_record, _apply_merged_inertia
    P = copy.deepcopy(MARKER if params is None else params)
    rho = float(vb.rho_build)
    mp = marker_mass_properties(P, rho=rho)
    s = vb.spec
    b = s.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    b.add_freejoint(name=f"{name}_root")
    kw = dict(friction=list(P["friction"]), mass=0.0, condim=4)
    for k, v in (contact or {}).items():
        kw[k] = list(v) if isinstance(v, (list, tuple)) else v
    gnames = []
    for pr in marker_geometry(P):
        gname = f"{name}_{pr['name']}"
        if pr["kind"] == "box":
            b.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_BOX, size=pr["half"], pos=pr["centre"],
                       rgba=[0.85, 0.30, 0.10, 1], **kw)
        else:
            b.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[pr["radius"], 0, 0],
                       fromto=list(pr["a"]) + list(pr["b"]), rgba=[0.6, 0.6, 0.6, 1], **kw)
        gnames.append(gname)
    hy = P["hydro"]
    bl, bk = P["blade"], P["block"]
    areas = [bk["size"][1] * bk["size"][2] + bl["size"][1] * bl["size"][2],
             bk["size"][0] * bk["size"][2] + bl["size"][0] * bl["size"][2],
             bk["size"][0] * bk["size"][1] + bl["size"][0] * bl["size"][1]]
    dq = [0.5 * rho * hy["cd"] * A for A in areas]
    Lc = mp["envelope_m3"] ** (1.0 / 3.0)
    dq_rot = [0.5 * rho * hy["cd"] * A * (0.5 * Lc) ** 3 for A in areas]
    m_add = rho * (mp["flooded_water_m3"] + hy["ca_envelope"] * mp["envelope_m3"])
    rec = simple_hydro_record(mp["volume_m3"], mp["cob_m"], mp["mass_kg"], mp["com_m"],
                              np.asarray(mp["inertia_com_kgm2"]), rho, ca=m_add / (rho * mp["volume_m3"]), ca_rot=0.0,
                              d_lin=(0,) * 6, d_quad=tuple(dq) + tuple(dq_rot), half_height=hy["half_height"])
    _apply_merged_inertia(b, rec)
    vb.meta["hydro_bodies"][name] = rec.to_dict()
    vb.meta.setdefault("free_objects", {})[name] = dict(shape="marker", mass=mp["mass_kg"], volume=mp["volume_m3"],
                                                        wet_weight_N=mp["wet_weight_N"], weld=None, contact="custom")
    if name not in vb.objects:
        vb.objects.append(name)
    vb.write_meta()
    Dp, L, T = bl["size"]
    return dict(name=name, kind="marker", body=name, paddle_geom=f"{name}_blade", geoms=gnames,
                centre_b=[0.0, 0.0, 0.0], R_bp=np.eye(3).tolist(), joint=None, paddle_m=[Dp, L, T],
                mass_properties=mp)
