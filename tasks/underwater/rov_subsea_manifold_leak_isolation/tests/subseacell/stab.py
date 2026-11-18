"""Hot stab of task 07: the stab body, the receptacle sliders and the three-state machine (cell side; DESIGN v1.2
sections 1.4 "保护行为", 2.3.2, 3.2, 3.6; package P07).  Structure after the plug-and-socket plugin of the open-source
underwater simulator described in note R5 section 2.2 (Free -> Joined -> Locked), written new for this cell.

The stab.  A free rigid body; body frame S: origin = the nose tip (the point the rules measure), x = the stab axis
pointing out of the nose (the insertion direction), y = the handle blade's length, z = the blade normal.  A sealed
stainless shaft (diameter tip_diameter, hemispherical nose, a ballast slug in the nose that sets the wet weight to
stab.wet_N), a neck, and the panel-standard paddle handle (160 x 60 x 12 mm, the valves' blade) at handle_x; parked,
the blade is horizontal and the vehicle grasps it with the paddle rule of subseacell.grasp (the stab is a free handle,
handle_info()).  The nose collision ends depth_tol_m short of the nominal tip so a seated stab does not touch the face.

The receptacles (test ports TP-1..6, parking ports PARK-A / PARK-B: the guide cone + bore of subseacell.devices) each
get, before compile, a hidden slider: a light body on a slide joint along the port axis (q = depth of the stab tip past
the mouth plane, range [slider_lo_m, insert_depth]), an inactive weld slider -> stab ("join") and an inactive joint
lock ("latch").  No geometry.  The port frame: x = into the panel (insertion), z = the panel's up, y = z x x.

State machine (StabMachine; call tick_pre() before and tick_post() after every physics step):
  Free -> Joined(port): the tip point crosses the port's mouth plane (s from < 0 to >= 0, crossing point interpolated
      between the two steps) within capture radius (cone_diameter - tip_diameter) / 2 = 25 mm of the port axis, with
      the stab axis within align_max_rad (0.15 rad) of the port axis.  A tip that is already inside the mouth joins as
      soon as it is within the funnel radius r(s) = (cone radius at depth s) - tip radius and aligned (a cone that has
      guided a compliant stab in, or a stab realigned inside the cone).  After a side-load release a port re-arms only
      when the tip has left its mouth.
      On joining the weld is switched on at the stab's current pose relative to the slider (no snap) and its target is
      then slid, as the tip goes deeper, onto the axis (quintic in depth, centred and aligned at the cone's end; the
      roll about the axis is kept): only axial sliding remains.  The stab stops colliding with the receptacles.
  Joined -> Locked: tip at the seat (depth >= insert_depth - depth_tol_m) and the axial force the stab pushes into the
      receptacle >= F_mate continuously for lock_hold_s (0.2 s), F_mate = mate_base_N + mate_per_bar_N x the port's
      pressure (test port TP-i: the pressure of branch i's leak section; parking ports: 0 -> mate_base_N).  The latch
      locks the slider.
  Joined -> Free: the tip leaves the mouth (s < 0), or the side load (force across the axis that the receptacle takes)
      > side_load_max_N continuously for side_hold_s (DESIGN 1.4 "stab 插歪": released, no damage, no penalty).
  Locked -> Joined: an outward pull >= extract_N continuously for extract_hold_s (from any port); the stab can then be
      drawn out (-> Free at the mouth).
  Measurement: the axial force and the side load are the join weld's force on the slider / on the stab (J^T f of the
  weld rows only), so they are what the stab transmits to the receptacle, whatever pushes the stab.
  Seal: in the last seal_stroke_m before the seat a pressurised receptacle pushes a joined stab back out with
      mate_per_bar_N x pressure (ramped in over seal_ramp_m): a stab pushed with less than that stops short of the seat
      ("insertion is resisted" - the formula's pressure part), between that and F_mate it seats but does not latch.
      The bore damps axial sliding (slider_damping); joint damping is integrated implicitly.

Collision bits (the vehicle's own scheme: world 1 / 1, hull 2 / 1, grasped object 4 / 1): the receptacle walls become
8 / 7 (they still meet the hull, grasped objects and free objects), a joined or locked stab 16 / 3 (16 / 1 while held
by the gripper): it meets the world and the hull but not the receptacle walls.  The machine re-applies the stab's bits
after every step (after the gripper's own pass-through handling).

    from subseacell import stab as ST
    stab = ST.add_stab(vb.spec, "stab", pos, quat, vb=vb)            # free body (+ hydro record when vb is given)
    rec = ST.add_receptacles(vb.spec, panel, "stab")                  # panel = devices.build_panel(...) result
    handles["stab"] = stab["handle"]                                  # paddle handle for subseacell.grasp
    ...compile / UnderwaterSim...
    SM = ST.StabMachine(model, data, stab, rec, pressure_bar=lambda port, t: ..., gripper=G, on_event=log.append)
    SM.reset_locked("PARK-A")
    every physics step:  ...; SM.tick_pre(); mj_step / sim.step(); ...; SM.tick_post()
    SM.state in ("Free", "Joined", "Locked"); SM.port; SM.events
"""
from __future__ import annotations

import copy
import math

import numpy as np
import mujoco

# task values stab.* (DEV/spec/task_values.json; tests/test_stab_objects.py asserts equality); configure() overrides.
# Values marked P07 are this package's choices (not in DESIGN): depth tolerance, seal stroke, bore damping, slider.
DEFAULTS = dict(
    cone_diameter_m=0.090, tip_diameter_m=0.040, align_max_rad=0.15, insert_depth_m=0.120, lock_hold_s=0.2,
    mate_base_N=15.0, mate_per_bar_N=2.5, extract_N=35.0, side_load_max_N=60.0, wet_N=3.0, initial_port="PARK-A",
    extract_hold_s=0.2, side_hold_s=0.2,                           # P07: same 0.2 s window as the latch / slip rules
    depth_tol_m=0.001,                                             # P07: "at the seat"
    seal_stroke_m=0.020, seal_ramp_m=0.002,                        # P07: seal engagement before the seat
    slider_lo_m=-0.030, slider_mass_kg=0.5, slider_damping=200.0,   # P07: load-path mass (kg), bore damping N s / m
    centre_min_span_m=0.005,                                       # P07: shortest depth span of the centring glide
    weld_solref=(0.004, 1.0), weld_solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    limit_solref=(0.004, 1.0), lock_solref=(0.004, 1.0),
)
VALUES = copy.deepcopy(DEFAULTS)
STATES = ("Free", "Joined", "Locked")

# geometry and materials in the stab frame S (metres, kg / m^3); P07 design values (DESIGN gives the tip diameter, the
# handle and the wet weight only).  The handle centre is 0.28 m behind the tip: with the canonical bite the TCP is
# then about 0.30 m off the face when the stab is seated (DESIGN 2.3.1: the vehicle 0.14-0.76 m off the panel).
STAB = dict(
    shaft=dict(back_x=-0.200, wall=0.002, density=7900.0),         # sealed stainless shell, capsule from back_x to the nose
    slug=dict(density=7900.0),                                     # ballast slug in the nose; length solved for wet_N
    neck=dict(radius=0.008, x=(-0.250, -0.200), density=7900.0),
    paddle=dict(size=(0.060, 0.160, 0.012), centre_x=-0.280, density=1410.0),   # depth, length, thickness; acetal
    hydro=dict(cd=1.0, ca=0.8, half_height=0.03),
    friction=(0.8, 0.02, 0.002),
    rgba=dict(shaft=(0.85, 0.85, 0.20, 1.0), paddle=(0.92, 0.25, 0.10, 1.0)),
)
BITS_FREE = (1, 1)
BITS_HELD_FREE = (4, 1)          # = subseacell.grasp.GRASPED_BITS
BITS_JOINED = (16, 3)
BITS_JOINED_HELD = (16, 1)
BITS_RECEPTACLE = (8, 7)


def configure(values: dict):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown stab values {sorted(unknown)}")
    VALUES.update(copy.deepcopy(values))


def mate_force_N(pressure_bar, park=False, V=None):
    """F_mate = base + per_bar x p for a test port; the base for a parking port (DESIGN 2.3.2)."""
    V = VALUES if V is None else V
    return float(V["mate_base_N"]) + (0.0 if park else float(V["mate_per_bar_N"]) * max(0.0, float(pressure_bar)))


# ---------------------------------------------------------------------------------------------------- helpers
def smootherstep(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _mat_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.asarray(R, float)).reshape(-1))
    return q


def _quat_mat(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _quat_mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _quat_conj(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    dd = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dd > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(dd)
        q = (math.sin((1.0 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def _align_x(R):
    """Rotation R' = A R where A is the smallest rotation taking R's x column onto e_x (roll about x kept)."""
    a = R[:, 0] / np.linalg.norm(R[:, 0])
    ex = np.array([1.0, 0.0, 0.0])
    v = np.cross(a, ex)
    s, c = float(np.linalg.norm(v)), float(a @ ex)
    if s < 1e-12:
        return R.copy() if c > 0 else (np.diag([-1.0, -1.0, 1.0]) @ R)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    A = np.eye(3) + s * K + (1 - c) * (K @ K)
    return A @ R


def port_frame(axis_w, up_w):
    """Port frame in world (columns): x = insertion direction, z = the panel's up, y = z x x."""
    x = np.asarray(axis_w, float)
    x = x / np.linalg.norm(x)
    z = np.asarray(up_w, float)
    z = z - (z @ x) * x
    z = z / np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


# ---------------------------------------------------------------------------------------------------- the stab body
def stab_geometry(params=None, V=None):
    """Collision primitives in S (dicts), the solved ballast slug length and the shaft radius."""
    P = STAB if params is None else params
    V = VALUES if V is None else V
    r = 0.5 * float(V["tip_diameter_m"])
    tol = float(V["depth_tol_m"])
    sh, nk, pd = P["shaft"], P["neck"], P["paddle"]
    seg = (float(sh["back_x"]) + r, -tol - r)                      # capsule segment (hemisphere centres)
    out = [dict(name="shaft", kind="capsule", a=[seg[0], 0.0, 0.0], b=[seg[1], 0.0, 0.0], radius=r),
           dict(name="neck", kind="capsule", a=[nk["x"][0], 0.0, 0.0], b=[nk["x"][1], 0.0, 0.0], radius=nk["radius"]),
           dict(name="paddle", kind="box", centre=[pd["centre_x"], 0.0, 0.0], half=[x / 2 for x in pd["size"]])]
    return out, seg, r


def stab_mass_properties(params=None, V=None, rho=1024.77, g=9.80665):
    """Mass, CoM, inertia about the CoM (S axes), displaced volume, CoB and wet weight [N]; the nose ballast slug
    length is solved so that the wet weight equals V['wet_N']."""
    P = STAB if params is None else params
    V = VALUES if V is None else V
    _, seg, r = stab_geometry(P, V)
    sh, nk, pd, sl = P["shaft"], P["neck"], P["paddle"], P["slug"]
    L = seg[1] - seg[0]
    parts = []                                                     # (mass, com, I_com (3x3), displaced volume, cob)
    # shell of the capsule (thin wall): area-weighted
    A_cyl, A_sph = 2 * math.pi * r * L, 4 * math.pi * r * r
    t = float(sh["wall"])
    m_cyl, m_sph = sh["density"] * A_cyl * t, sh["density"] * A_sph * t
    xc = 0.5 * (seg[0] + seg[1])
    Ic = np.diag([m_cyl * r * r, m_cyl * (r * r / 2 + L * L / 12), m_cyl * (r * r / 2 + L * L / 12)])
    parts.append((m_cyl, np.array([xc, 0, 0]), Ic, 0.0, None))
    for xe in seg:                                                 # two half-sphere shells at the ends
        parts.append((m_sph / 2, np.array([xe, 0, 0]), np.eye(3) * (2.0 / 3.0) * (m_sph / 2) * r * r, 0.0, None))
    Vo = math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3
    parts.append((0.0, None, np.zeros((3, 3)), Vo, np.array([xc, 0, 0])))
    # neck
    rn = float(nk["radius"])
    Ln = abs(nk["x"][1] - nk["x"][0])
    Vn = math.pi * rn * rn * Ln + 4.0 / 3.0 * math.pi * rn ** 3
    mn = nk["density"] * Vn
    xn = 0.5 * (nk["x"][0] + nk["x"][1])
    parts.append((mn, np.array([xn, 0, 0]), np.diag([mn * rn * rn / 2, mn * (Ln * Ln / 12 + rn * rn / 4),
                                                       mn * (Ln * Ln / 12 + rn * rn / 4)]), Vn, np.array([xn, 0, 0])))
    # paddle
    a, b, c = pd["size"]
    Vp = a * b * c
    mp = pd["density"] * Vp
    parts.append((mp, np.array([pd["centre_x"], 0, 0]), np.diag([mp * (b * b + c * c) / 12, mp * (a * a + c * c) / 12,
                                                                  mp * (a * a + b * b) / 12]), Vp, np.array([pd["centre_x"], 0, 0])))
    M0 = sum(p[0] for p in parts)
    Vt = sum(p[3] for p in parts)
    # ballast slug: inner radius r - t, from the nose's hemisphere centre backwards (inside the sealed shell)
    ri = r - t
    m_slug = (float(V["wet_N"]) / g + rho * Vt) - M0
    if m_slug < 0:
        raise ValueError("stab lighter than the wet weight allows without ballast")
    Ls = m_slug / (sl["density"] * math.pi * ri * ri)
    xs = seg[1] - 0.5 * Ls
    parts.append((m_slug, np.array([xs, 0, 0]), np.diag([m_slug * ri * ri / 2, m_slug * (Ls * Ls / 12 + ri * ri / 4),
                                                          m_slug * (Ls * Ls / 12 + ri * ri / 4)]), 0.0, None))
    M = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts if p[0] > 0) / M
    cob = sum(p[3] * p[4] for p in parts if p[3] > 0) / Vt
    I = np.zeros((3, 3))
    for m_, r_, I_, _, _ in parts:
        if m_ <= 0:
            continue
        dd = r_ - com
        I += I_ + m_ * (float(dd @ dd) * np.eye(3) - np.outer(dd, dd))
    return dict(mass_kg=M, com_m=com.tolist(), inertia_com_kgm2=I.tolist(), volume_m3=Vt, cob_m=cob.tolist(),
                wet_weight_N=(M - rho * Vt) * g, slug_length_m=Ls, envelope_m3=Vt)


def add_stab(spec: mujoco.MjSpec, name="stab", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), vb=None, params=None):
    """Add the stab as a free body.  With vb (a kit VehicleBuild) the body gets the kit's simple hydro record
    (buoyancy of the displaced volume, merged isotropic added mass, quadratic drag per axis) like the marker; without
    it (test benches without the water layer) the body carries its rigid inertia only.  Returns the stab info:
    body, geoms, handle (the grasp handle description), mass properties, tip / axis in S."""
    P = copy.deepcopy(STAB if params is None else params)
    rho = float(vb.rho_build) if vb is not None else 1024.77
    mp = stab_mass_properties(P, rho=rho)
    b = spec.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    b.add_freejoint(name=f"{name}_root")
    kw = dict(friction=list(P["friction"]), mass=0.0, condim=4)
    prims, _, r = stab_geometry(P)
    gnames = []
    for pr in prims:
        gname = f"{name}_{pr['name']}"
        rgba = P["rgba"]["paddle" if pr["name"] == "paddle" else "shaft"]
        if pr["kind"] == "box":
            b.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_BOX, size=pr["half"], pos=pr["centre"], rgba=list(rgba), **kw)
        else:
            b.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[pr["radius"], 0, 0],
                       fromto=list(pr["a"]) + list(pr["b"]), rgba=list(rgba), **kw)
        gnames.append(gname)
    I = np.asarray(mp["inertia_com_kgm2"])
    if vb is not None:
        from uwkit.models.vehicle import simple_hydro_record, _apply_merged_inertia
        hy = P["hydro"]
        pd = P["paddle"]
        sh_len = abs(P["shaft"]["back_x"])
        areas = [math.pi * r * r + pd["size"][1] * pd["size"][2],
                 2 * r * sh_len + pd["size"][0] * pd["size"][2],
                 2 * r * sh_len + pd["size"][0] * pd["size"][1]]
        dq = [0.5 * rho * hy["cd"] * A for A in areas]
        Lc = 0.5 * sh_len
        dq_rot = [0.5 * rho * hy["cd"] * A * Lc ** 3 for A in areas]
        rec = simple_hydro_record(mp["volume_m3"], mp["cob_m"], mp["mass_kg"], mp["com_m"], I, rho, ca=hy["ca"],
                                  ca_rot=0.0, d_lin=(0,) * 6, d_quad=tuple(dq) + tuple(dq_rot), half_height=hy["half_height"])
        _apply_merged_inertia(b, rec)
        vb.meta["hydro_bodies"][name] = rec.to_dict()
        vb.meta.setdefault("free_objects", {})[name] = dict(shape="hot_stab", mass=mp["mass_kg"], volume=mp["volume_m3"],
                                                            wet_weight_N=mp["wet_weight_N"], weld=None, contact="custom")
        if name not in vb.objects:
            vb.objects.append(name)
        vb.write_meta()
    else:
        b.explicitinertial = True
        b.mass = mp["mass_kg"]
        b.ipos = list(mp["com_m"])
        b.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    pd = P["paddle"]
    handle = dict(name=name, kind="stab", body=name, paddle_geom=f"{name}_paddle", geoms=list(gnames),
                  centre_b=[float(pd["centre_x"]), 0.0, 0.0], R_bp=np.eye(3).tolist(), joint=None,
                  paddle_m=[float(x) for x in pd["size"]])
    return dict(name=name, body=name, geoms=gnames, handle=handle, mass_properties=mp, tip_b=[0.0, 0.0, 0.0],
                axis_b=[1.0, 0.0, 0.0], radius=r, params=P)


def seated_pose(rec_port, depth=None):
    """World pose (pos, quat) of a stab seated (or at tip depth `depth`) in the receptacle, blade horizontal."""
    d = rec_port["insert_depth"] if depth is None else float(depth)
    R = np.asarray(rec_port["R_w"], float)
    return np.asarray(rec_port["mouth_w"], float) + R[:, 0] * d, _mat_quat(R)


# ---------------------------------------------------------------------------------------------------- receptacles
def fix_receptacle_walls(spec: mujoco.MjSpec, panel: dict):
    """Re-orient the receptacle wall boxes in the spec with a proper rotation.  The panel description gives each wall
    piece's axes as (axial / slant, tangential, radial) - a LEFT-handed set (det -1) - and a quaternion of an improper
    matrix is not that box: measured, a centred 40 mm stab cleared the 46 mm bore by +3.0 mm on 2 of the 12 bore staves
    and overlapped the others by up to 2.8 mm.  Flipping the tangential axis gives the same box with a right-handed
    frame.  No-op for pieces that are already right-handed (so it stays correct once the description is fixed at its
    source).  Returns the number of pieces corrected."""
    pos_w, R_w = (np.asarray(x, float) for x in panel["frame"])
    prims = {pr["name"]: pr for pr in panel["skid"]["primitives"] if pr.get("role") == "port_receptacle"}
    n = 0
    for p in panel["ports"].values():
        for gname in p["geoms"]:
            pr = next((v for k, v in prims.items() if gname.endswith(k)), None)
            if pr is None or pr.get("axes") is None:
                continue
            A = np.asarray(pr["axes"], float).T
            if np.linalg.det(A) > 0:
                continue
            A[:, 1] = -A[:, 1]
            g = spec.geom(gname)
            if g is None:
                continue
            g.quat = list(_mat_quat(R_w @ A))
            n += 1
    return n


def add_receptacles(spec: mujoco.MjSpec, panel: dict, stab_body="stab", prefix=""):
    """Add one hidden slider + join weld + latch per port of `panel` (devices.build_panel's result), before compile.
    Returns {port: info} (mouth / axis / frame in world, cone and bore radii, insert depth, kind, wall geoms, names)."""
    V = VALUES
    R_panel = np.asarray(panel["frame"][1], float)
    fix_receptacle_walls(spec, panel)
    out = {}
    for pname, p in panel["ports"].items():
        R = port_frame(p["axis_w"], R_panel[:, 1])
        mouth = np.asarray(p["mouth_w"], float)
        sname = f"{prefix}stabslide_{pname}"
        b = spec.worldbody.add_body(name=sname, pos=list(mouth), quat=list(_mat_quat(R)))
        b.explicitinertial = True
        b.mass = float(V["slider_mass_kg"])
        b.ipos = [0.0, 0.0, 0.0]
        b.inertia = [1e-6, 1e-6, 1e-6]
        jn = f"{sname}_q"
        j = b.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1.0, 0.0, 0.0],
                        range=[float(V["slider_lo_m"]), float(p["insert_depth"])], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                        damping=float(V["slider_damping"]))
        j.solref_limit = list(V["limit_solref"])
        e = spec.add_equality(name=f"{prefix}stabjoin_{pname}", type=mujoco.mjtEq.mjEQ_WELD,
                              objtype=mujoco.mjtObj.mjOBJ_BODY, name1=sname, name2=stab_body, active=False)
        e.solref, e.solimp = list(V["weld_solref"]), list(V["weld_solimp"])
        k = spec.add_equality(name=f"{prefix}stablock_{pname}", type=mujoco.mjtEq.mjEQ_JOINT,
                              objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=jn, active=False)
        k.solref = list(V["lock_solref"])
        r_mouth = 0.5 * float(p["mouth_diameter"])
        r_bore = 0.5 * float(p["bore_diameter"])
        seat = np.asarray(p["seat_w"], float)
        sk = next(x for x in panel["skid"]["ports"] if x["key"] == p["key"])
        cone_depth = float(sk["cone_depth"])
        out[pname] = dict(name=pname, kind=p["kind"], mouth_w=mouth, seat_w=seat, R_w=R, axis_w=R[:, 0].copy(),
                          r_mouth=r_mouth, r_bore=r_bore, cone_depth=cone_depth, insert_depth=float(p["insert_depth"]),
                          walls=list(p["geoms"]), slider=sname, joint=jn, weld=f"{prefix}stabjoin_{pname}",
                          lock=f"{prefix}stablock_{pname}")
    return out


# ---------------------------------------------------------------------------------------------------- the machine
class StabMachine:
    """Run-time hot-stab rules for one stab and a set of receptacles (works on a plain MjModel / MjData: the water
    layer is not needed).  pressure_bar(port_name, t) -> bar for test ports (parking ports are taken as 0)."""

    def __init__(self, model, data, stab: dict, receptacles: dict, pressure_bar=None, gripper=None, on_event=None):
        self.m, self.d = model, data
        self.V = VALUES
        m = model
        self.stab = stab
        self.name = stab["name"]
        self.bid = m.body(stab["body"]).id
        jt = int(m.body_jntadr[self.bid])
        self.qa, self.va = int(m.jnt_qposadr[jt]), int(m.jnt_dofadr[jt])
        self.geom_ids = [m.geom(g).id for g in stab["geoms"]]
        self.r_tip = float(stab["radius"])
        self.pressure_bar = pressure_bar or (lambda port, t: 0.0)
        self.gripper = gripper
        self.on_event = on_event
        self.ports = {}
        for pn, p in receptacles.items():
            pp = dict(p)
            pp["sid"] = m.body(p["slider"]).id
            jid = m.joint(p["joint"]).id
            pp["jid"] = jid
            pp["sqa"], pp["sva"] = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
            pp["weld_id"] = m.equality(p["weld"]).id
            pp["lock_id"] = m.equality(p["lock"]).id
            pp["wall_ids"] = [m.geom(g).id for g in p["walls"]]
            pp["park"] = p["kind"] == "park"
            pp["R"] = np.asarray(p["R_w"], float)
            pp["mouth"] = np.asarray(p["mouth_w"], float)
            for gi in pp["wall_ids"]:
                m.geom_contype[gi], m.geom_conaffinity[gi] = BITS_RECEPTACLE
            self.ports[pn] = pp
        self._vec = None
        self._res = np.zeros(m.nv)
        self.reset_free()

    # ------------------------------------------------------------------------------------------------ state
    def _clear(self):
        d = self.d
        for p in self.ports.values():
            d.eq_active[p["weld_id"]] = 0
            d.eq_active[p["lock_id"]] = 0
            d.qpos[p["sqa"]] = 0.0
            d.qvel[p["sva"]] = 0.0
            d.qfrc_applied[p["sva"]] = 0.0
        self.state = "Free"
        self.port = None
        self.events = []
        self.armed = {pn: True for pn in self.ports}
        self.s_prev = {pn: None for pn in self.ports}
        self.join = None                    # dict(s0, off0, q0, qT, u, t)
        self.t_lock_run = 0.0
        self.t_extract_run = 0.0
        self.t_side_run = 0.0
        self.F_ax = 0.0
        self.F_side = 0.0
        self.depth = None
        self.F_back = 0.0
        self.left_initial = False
        self.initial_port = None
        self.lock_window = None             # (t_start, F_ax at start, F_mate) of the qualifying latch window
        self.extract_window = None
        self.side_window = None
        self.peak = dict(F_ax=0.0, F_side=0.0)

    def reset_free(self, pos=None, quat=None):
        """Stab Free (optionally at a pose)."""
        self._clear()
        if pos is not None:
            self.d.qpos[self.qa:self.qa + 3] = pos
            self.d.qpos[self.qa + 3:self.qa + 7] = quat if quat is not None else (1.0, 0.0, 0.0, 0.0)
            self.d.qvel[self.va:self.va + 6] = 0.0
        self._apply_bits()

    def reset_locked(self, port=None):
        """Stab seated and latched in `port` (default: the initial port of the task values), blade horizontal."""
        self._clear()
        pn = port or self.V["initial_port"]
        p = self.ports[pn]
        d = self.d
        pos, quat = seated_pose(p)
        d.qpos[self.qa:self.qa + 3] = pos
        d.qpos[self.qa + 3:self.qa + 7] = quat
        d.qvel[self.va:self.va + 6] = 0.0
        q = p["insert_depth"]
        d.qpos[p["sqa"]] = q
        self._activate_weld(p, q, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        self.join = dict(s0=q, off0=np.zeros(3), q0=np.array([1.0, 0, 0, 0]), qT=np.array([1.0, 0, 0, 0]), u=1.0,
                         t=float(d.time))
        self.state, self.port, self.depth = "Joined", pn, q
        self._latch(p, q)
        self.initial_port = pn
        self.events = []
        self._apply_bits()
        mujoco.mj_forward(self.m, d)

    def _event(self, **e):
        e.setdefault("t", round(float(self.d.time), 4))
        self.events.append(e)
        if self.on_event is not None:
            self.on_event(e)
        return e

    # ------------------------------------------------------------------------------------------------ geometry
    def tip_axis_w(self):
        d = self.d
        R = d.xmat[self.bid].reshape(3, 3)
        return d.xpos[self.bid].copy(), R[:, 0].copy()

    def rel_port(self, pn, tip=None, axis=None):
        """(s = tip depth past the mouth plane, radial offset vector in the port's y-z, angle to the port axis)."""
        p = self.ports[pn]
        if tip is None:
            tip, axis = self.tip_axis_w()
        rel = p["R"].T @ (tip - p["mouth"])
        c = float(np.clip(axis @ p["R"][:, 0], -1.0, 1.0))
        return float(rel[0]), rel[1:].copy(), math.acos(c)

    def capture_radius(self, p, s):
        """Funnel radius for the tip centre at depth s (crossing: s = 0 -> (cone - tip) / 2)."""
        cd = p["cone_depth"]
        rc = p["r_mouth"] - (p["r_mouth"] - p["r_bore"]) * min(max(s, 0.0), cd) / cd
        return max(0.0, rc - self.r_tip)

    def f_mate(self, pn, t=None):
        p = self.ports[pn]
        t = float(self.d.time) if t is None else t
        return mate_force_N(0.0 if p["park"] else self.pressure_bar(pn, t), park=p["park"], V=self.V)

    # ------------------------------------------------------------------------------------------------ weld / latch
    def _activate_weld(self, p, q, off, qrel):
        m, d = self.m, self.d
        eq = p["weld_id"]
        m.eq_data[eq, 0:3] = 0.0                    # anchor = the tip point (origin of S)
        m.eq_data[eq, 3:6] = off                    # target of the anchor in the slider frame
        m.eq_data[eq, 6:10] = qrel
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1

    def _latch(self, p, q):
        m, d = self.m, self.d
        lk = p["lock_id"]
        m.eq_data[lk, :] = 0.0
        m.eq_data[lk, 0] = q
        d.eq_active[lk] = 1
        self.state = "Locked"

    def _do_join(self, pn, s, off_vec):
        m, d = self.m, self.d
        p = self.ports[pn]
        qS = d.xquat[self.bid].copy()
        qP = _mat_quat(p["R"])
        qrel = _quat_mul(_quat_conj(qP), qS)
        qrel /= np.linalg.norm(qrel)
        qT = _mat_quat(_align_x(_quat_mat(qrel)))
        d.qpos[p["sqa"]] = s
        v = d.qvel[self.va:self.va + 3]
        d.qvel[p["sva"]] = float(v @ p["R"][:, 0])
        off = np.array([0.0, off_vec[0], off_vec[1]])
        self._activate_weld(p, s, off, qrel)
        self.join = dict(s0=s, off0=off, q0=qrel, qT=qT, u=0.0, t=float(d.time))
        self.state, self.port, self.depth = "Joined", pn, s
        self.t_lock_run = self.t_side_run = self.t_extract_run = 0.0
        self.lock_window = self.side_window = None
        self._event(ev="join", port=pn, depth_m=round(s, 5), radial_m=round(float(np.linalg.norm(off_vec)), 5),
                    angle_rad=round(self.rel_port(pn)[2], 5))

    def _do_free(self, why, **extra):
        d = self.d
        p = self.ports[self.port]
        pn = self.port
        d.eq_active[p["weld_id"]] = 0
        d.eq_active[p["lock_id"]] = 0
        d.qfrc_applied[p["sva"]] = 0.0
        d.qpos[p["sqa"]] = 0.0
        d.qvel[p["sva"]] = 0.0
        if why == "side_load":
            self.armed[pn] = False
        if self.initial_port == pn:
            self.left_initial = True
        self.state, self.port, self.join = "Free", None, None
        self.F_ax = self.F_side = self.F_back = 0.0
        self.depth = None
        self.s_prev[pn] = None
        self._event(ev="free", port=pn, why=why, **extra)

    # ------------------------------------------------------------------------------------------------ loads
    def weld_loads(self):
        """(axial force the stab pushes into the receptacle [N, + = inwards], side load on the stab [N]) from the
        join weld rows only."""
        m, d = self.m, self.d
        p = self.ports[self.port]
        eq = p["weld_id"]
        if self._vec is None or self._vec.shape[0] != d.nefc:
            self._vec = np.zeros(d.nefc)
        vec = self._vec
        vec[:] = 0.0
        rows = np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == eq))[0]
        if rows.size == 0:
            return 0.0, 0.0
        vec[rows] = d.efc_force[rows]
        res = self._res
        mujoco.mj_mulJacTVec(m, d, res, vec)
        F_ax = float(res[p["sva"]])
        f_stab = res[self.va:self.va + 3]
        a = p["R"][:, 0]
        side = float(np.linalg.norm(f_stab - (f_stab @ a) * a))
        return F_ax, side

    # ------------------------------------------------------------------------------------------------ per step
    def tick_pre(self):
        """Before mj_step: centring glide of the join weld target and the seal force."""
        if self.state == "Free" or self.port is None:
            return
        V, d, m = self.V, self.d, self.m
        p = self.ports[self.port]
        s = float(d.qpos[p["sqa"]])
        if self.state == "Joined":
            J = self.join
            span = max(p["cone_depth"] - J["s0"], float(V["centre_min_span_m"]))
            u = smootherstep((s - J["s0"]) / span)
            if u > J["u"]:
                J["u"] = u
            eq = p["weld_id"]
            m.eq_data[eq, 3:6] = (1.0 - J["u"]) * J["off0"]
            m.eq_data[eq, 6:10] = _slerp(J["q0"], J["qT"], J["u"])
            z0 = p["insert_depth"] - float(V["seal_stroke_m"])
            pr = 0.0 if p["park"] else self.pressure_bar(self.port, float(d.time))
            ramp = min(1.0, max(0.0, (s - z0) / float(V["seal_ramp_m"])))
            self.F_back = float(V["mate_per_bar_N"]) * max(0.0, pr) * ramp
            d.qfrc_applied[p["sva"]] = -self.F_back
        else:
            self.F_back = 0.0
            d.qfrc_applied[p["sva"]] = 0.0

    def tick_post(self):
        """After mj_step: loads, transitions, collision bits."""
        V, d = self.V, self.d
        dt = float(self.m.opt.timestep)
        t = float(d.time)
        if self.state == "Free":
            self._free_step(t)
        else:
            p = self.ports[self.port]
            s = float(d.qpos[p["sqa"]])
            self.depth = s
            F_ax, side = self.weld_loads()
            self.F_ax, self.F_side = F_ax, side
            self.peak["F_ax"] = max(self.peak["F_ax"], abs(F_ax))
            self.peak["F_side"] = max(self.peak["F_side"], side)
            if self.state == "Joined":
                if s < 0.0:
                    self._do_free("withdrawn", depth_m=round(s, 5))
                else:
                    if side > V["side_load_max_N"]:
                        if self.t_side_run == 0.0:
                            self.side_window = (t, side)
                        self.t_side_run += dt
                    else:
                        self.t_side_run = 0.0
                    if self.t_side_run >= V["side_hold_s"] - 1e-9:
                        self._do_free("side_load", side_N=round(side, 3), window=self.side_window)
                    else:
                        fm = self.f_mate(self.port, t)
                        seated = s >= p["insert_depth"] - V["depth_tol_m"]
                        if seated and F_ax >= fm:
                            if self.t_lock_run == 0.0:
                                self.lock_window = (t, F_ax, fm)
                            self.t_lock_run += dt
                        else:
                            self.t_lock_run = 0.0
                        if self.t_lock_run >= V["lock_hold_s"] - 1e-9:
                            self._latch(p, s)
                            self.t_extract_run = 0.0
                            self._event(ev="lock", port=self.port, F_ax_N=round(F_ax, 3), F_mate_N=round(fm, 3),
                                        window=self.lock_window, depth_m=round(s, 5))
            elif self.state == "Locked":
                if -F_ax >= V["extract_N"]:
                    if self.t_extract_run == 0.0:
                        self.extract_window = (t, F_ax)
                    self.t_extract_run += dt
                else:
                    self.t_extract_run = 0.0
                if self.t_extract_run >= V["extract_hold_s"] - 1e-9:
                    d.eq_active[p["lock_id"]] = 0
                    self.state = "Joined"
                    self.t_lock_run = self.t_side_run = 0.0
                    if self.initial_port == self.port:
                        self.left_initial = True
                    self._event(ev="unlock", port=self.port, F_ax_N=round(F_ax, 3), window=self.extract_window)
        self._apply_bits()

    def _free_step(self, t):
        V = self.V
        tip, axis = self.tip_axis_w()
        for pn, p in self.ports.items():
            rel = p["R"].T @ (tip - p["mouth"])
            s = float(rel[0])
            sp = self.s_prev[pn]
            near = -0.3 < s < p["insert_depth"] + 0.05 and float(np.hypot(rel[1], rel[2])) < 0.2
            if not near:
                self.s_prev[pn] = None
                continue
            if s < 0.0:
                self.armed[pn] = True
                self.s_prev[pn] = (s, rel.copy())
                continue
            ang = math.acos(float(np.clip(axis @ p["R"][:, 0], -1.0, 1.0)))
            if not self.armed[pn]:
                self.s_prev[pn] = (s, rel.copy())
                continue
            if sp is not None and sp[0] < 0.0:
                # crossing of the mouth plane in this step: radial offset at s = 0 (linear between the two steps)
                w = -sp[0] / (s - sp[0])
                rc = sp[1] + w * (rel - sp[1])
                rad = float(np.hypot(rc[1], rc[2]))
                ok = rad <= self.capture_radius(p, 0.0) and ang <= V["align_max_rad"]
                if not ok:
                    self._event(ev="miss", port=pn, radial_m=round(rad, 5), angle_rad=round(ang, 5))
            else:
                rad = float(np.hypot(rel[1], rel[2]))
                ok = rad <= self.capture_radius(p, s) and ang <= V["align_max_rad"]
            self.s_prev[pn] = (s, rel.copy())
            if ok:
                self._do_join(pn, s, rel[1:])
                return

    def _apply_bits(self):
        m = self.m
        held = self.gripper is not None and self.name in getattr(self.gripper, "pass_through", {})
        if self.state == "Free":
            bits = BITS_HELD_FREE if held else BITS_FREE
        else:
            bits = BITS_JOINED_HELD if held else BITS_JOINED
        for gi in self.geom_ids:
            m.geom_contype[gi], m.geom_conaffinity[gi] = bits

    # ------------------------------------------------------------------------------------------------ readout
    def public(self):
        """Grader-side state for scoring (DESIGN 3.2 'stab 状态'): Free / Joined(port) / Locked(port)."""
        return dict(state=self.state, port=self.port, left_initial=bool(self.left_initial))

    def summary(self):
        return dict(state=self.state, port=self.port, left_initial=bool(self.left_initial), events=list(self.events),
                    peak=dict(self.peak))
