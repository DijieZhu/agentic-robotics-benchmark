"""MJCF of the duct walls: `add_ducts(spec, world, pushable)`.

    spec = venue spec that already holds the floor (the venue generator's build_spec, or scenes.venue_spec)
    add_ducts(spec, world, pushable=True)            # mode from duct_params.json, or mode="sleep" / "awake"
    model = spec.compile(); data = mujoco.MjData(model)
    ducts = DuctTracker(model, data, world)          # update() after every physics step

The piece model (`planar_pieces`, every mode): each piece of length c is a thin-walled tube of mass m = rho c with
three joints in this order, slide along its laid axis, slide across it, hinge about the vertical through its centre,
each carrying the piece's own floor friction as frictionloss (mu m g, mu m g, mu m g c / 4). Nothing touches the
floor: weight and floor drag live in the joints, so a piece can only translate and turn on the floor.
Consecutive pieces of a duct are pinned at their shared end point (`connect` equality `.../<k>/pin`); bending is a
linear spring and damper toward the laid shape on the two yaws (`joint` equality `.../<k>/bend`, params.bend_solref).
Per piece: `.../col` capsule on the axis (group 1, category duct; at a free end of the duct it stops one radius
short), `.../lidar` flat-ended cylinder exactly at the piece (group 2, no contact).

Modes (duct_params.json `mode`, or the keyword):
  pushable=False    static geoms on the world body at the world-file poses (the venue generator before the duct model; greybox equivalent)
  "awake"           every piece a body `duct/<D>/<k>`, no sleep: the reference physics (slow with many pieces)
  "sleep"           the same bodies, MuJoCo's sleep flag on, pieces start asleep (policy init)
  "active_window"   static geoms for every piece (the frozen layer, contract names) plus `window.slots` dynamic
                    chains `duct/<slot>/<k>` with collision capsules only; DuctTracker moves a duct into a slot when a
                    car, slow car or box comes near it, writes its poses back when it is frozen again, and keeps the
                    frozen layer's LiDAR cylinders on the slot's pieces meanwhile (window.py)

The joints of a piece are `duct/<D>/<k>/x`, `/y`, `/yaw` (the contract lets the duct model choose the topology; no other package
reads duct joints). Every model built here carries custom texts `ducts/params_sha256` and `ducts/mode`.
"""

import collections
import math


from sketch.examples import constants as K

from . import geom
from .params import bend_solimp, bend_solref, check_params, load_params, params_digest, piece_constants

TEXT_DIGEST = "ducts/params_sha256"
TEXT_MODE = "ducts/mode"
SLOT_PREFIX = "AWS"


def _mj():
    import mujoco
    return mujoco


def resolve_mode(params, pushable, mode=None):
    if not pushable:
        return "static"
    m = mode or params["mode"]
    if m not in ("sleep", "awake", "active_window"):
        raise ValueError("unknown duct mode %r" % m)
    return m


def duct_bits(params):
    ct, ca = K.CONTACT["duct"]
    return ct, (K.DUCT_DUCT_CONAFFINITY if params["duct_duct_contact"] else ca)


def slot_ids(world, n_slots):
    """Ids of the window's slot chains, unique against the world's duct ids."""
    taken = {d["id"] for d in world["truth"]["ducts"]}
    pre = SLOT_PREFIX
    while any(("%s%d" % (pre, i)) in taken for i in range(n_slots)):
        pre = "X" + pre
    return ["%s%d" % (pre, i) for i in range(n_slots)]


def duct_classes(world):
    """Counter of (n_pieces, piece_length rounded to 1e-6) over the world's ducts."""
    return collections.Counter((len(d["pieces"]), round(float(d["piece_length_m"]), 6)) for d in world["truth"]["ducts"])


# ------------------------------------------------------------------------------------------------ geoms
def _add_col(parent, name, lo, hi, params, bits, pose=None):
    """Collision capsule along x of the parent frame, or (pose given) of a static piece on the world body."""
    mj = _mj()
    c = params["contact"]
    if pose is None:
        ft = [lo, 0.0, 0.0, hi, 0.0, 0.0]
    else:
        x, y, yaw = pose
        ux, uy = math.cos(yaw), math.sin(yaw)
        ft = [x + lo * ux, y + lo * uy, K.DUCT_RADIUS_M, x + hi * ux, y + hi * uy, K.DUCT_RADIUS_M]
    if hi - lo < 1e-9:                                # degenerate: a sphere (a one-piece duct below 2 radii)
        cx, cy, cz = 0.5 * (ft[0] + ft[3]), 0.5 * (ft[1] + ft[4]), 0.5 * (ft[2] + ft[5])
        return parent.add_geom(name=name, type=mj.mjtGeom.mjGEOM_SPHERE, size=[K.DUCT_RADIUS_M, 0, 0], pos=[cx, cy, cz],
                               group=K.GROUP_COLLISION, contype=bits[0], conaffinity=bits[1], friction=list(c["friction"]),
                               solref=list(c["solref"]), solimp=list(c["solimp"]), condim=int(c["condim"]), mass=0.0,
                               rgba=list(params["rgba"]["collision"]))
    return parent.add_geom(name=name, type=mj.mjtGeom.mjGEOM_CAPSULE, fromto=ft, size=[K.DUCT_RADIUS_M, 0, 0],
                           group=K.GROUP_COLLISION, contype=bits[0], conaffinity=bits[1], friction=list(c["friction"]),
                           solref=list(c["solref"]), solimp=list(c["solimp"]), condim=int(c["condim"]), mass=0.0,
                           rgba=list(params["rgba"]["collision"]))


def _add_lidar(parent, name, c, params, wall_class, pose=None):
    """The LiDAR target: a flat-ended cylinder exactly at the piece."""
    mj = _mj()
    if pose is None:
        ft = [-0.5 * c, 0.0, 0.0, 0.5 * c, 0.0, 0.0]
    else:
        x, y, yaw = pose
        ux, uy = 0.5 * c * math.cos(yaw), 0.5 * c * math.sin(yaw)
        ft = [x - ux, y - uy, K.DUCT_RADIUS_M, x + ux, y + uy, K.DUCT_RADIUS_M]
    return parent.add_geom(name=name, type=mj.mjtGeom.mjGEOM_CYLINDER, fromto=ft, size=[K.DUCT_RADIUS_M, 0, 0],
                           group=K.GROUP_LIDAR, contype=0, conaffinity=0, mass=0.0, rgba=list(params["rgba"][wall_class]))


# ------------------------------------------------------------------------------------------------ dynamic pieces
def _add_piece_body(spec, name, pose, c, params, sleep_policy=None):
    """One dynamic piece: body at pose (z = duct radius), explicit tube inertia, slides x / y and the yaw hinge with
    their floor friction. Returns the body."""
    mj = _mj()
    pc = piece_constants(params, c)
    fr = params["friction_constraint"]
    x, y, yaw = pose
    b = spec.worldbody.add_body(name=name, pos=[float(x), float(y), K.DUCT_RADIUS_M], quat=geom.yaw_quat(float(yaw)))
    b.explicitinertial = True
    b.mass = pc["mass_kg"]
    b.ipos = [0.0, 0.0, 0.0]
    b.inertia = list(pc["inertia_kgm2"])
    if sleep_policy is not None:
        b.sleep = sleep_policy
    for axis_name, jtype, axis, fl in (("x", mj.mjtJoint.mjJNT_SLIDE, [1.0, 0.0, 0.0], pc["frictionloss"][0]),
                                       ("y", mj.mjtJoint.mjJNT_SLIDE, [0.0, 1.0, 0.0], pc["frictionloss"][1]),
                                       ("yaw", mj.mjtJoint.mjJNT_HINGE, [0.0, 0.0, 1.0], pc["frictionloss"][2])):
        b.add_joint(name="%s/%s" % (name, axis_name), type=jtype, axis=axis, pos=[0.0, 0.0, 0.0], frictionloss=fl,
                    solref_friction=list(fr["solref"]), solimp_friction=list(fr["solimp"]))
    return b


def _add_pin_and_bend(spec, name_a, name_b, c_a, c_b, params):
    """Pin at the head of piece a / tail of piece b, and the bending spring on their yaws."""
    mj = _mj()
    pa, pb = piece_constants(params, c_a), piece_constants(params, c_b)
    pin = params["pin"]
    spec.add_equality(name="%s/pin" % name_b, type=mj.mjtEq.mjEQ_CONNECT, objtype=mj.mjtObj.mjOBJ_BODY,
                      name1=name_a, name2=name_b, data=[0.5 * c_a, 0.0, 0.0] + [0.0] * 8,
                      solref=list(pin["solref"]), solimp=list(pin["solimp"]))
    k = 0.5 * (pa["bend_stiffness_nm_per_rad"] + pb["bend_stiffness_nm_per_rad"])
    bd = 0.5 * (pa["bend_damping_nms_per_rad"] + pb["bend_damping_nms_per_rad"])
    spec.add_equality(name="%s/bend" % name_b, type=mj.mjtEq.mjEQ_JOINT, objtype=mj.mjtObj.mjOBJ_JOINT,
                      name1="%s/yaw" % name_b, name2="%s/yaw" % name_a, data=[0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6,
                      solref=bend_solref(params, pa["inertia_kgm2"][2], pb["inertia_kgm2"][2], k, bd),
                      solimp=bend_solimp(params))


def _add_dynamic_duct(spec, did, poses, c, wall_class, params, bits, sleep_policy=None, lidar=True):
    n = len(poses)
    names = [K.duct_body(did, k) for k in range(n)]
    for k, pose in enumerate(poses):
        b = _add_piece_body(spec, names[k], pose, c, params, sleep_policy)
        lo, hi = geom.collision_extent(k, n, c)
        _add_col(b, K.duct_geom(did, k, "col"), lo, hi, params, bits)
        if lidar:
            _add_lidar(b, K.duct_geom(did, k, "lidar"), c, params, wall_class)
    for k in range(1, n):
        _add_pin_and_bend(spec, names[k - 1], names[k], c, c, params)


def _add_static_duct(spec, duct, params, bits):
    c = float(duct["piece_length_m"])
    n = len(duct["pieces"])
    for k, pose in enumerate(duct["pieces"]):
        lo, hi = geom.collision_extent(k, n, c)
        _add_col(spec.worldbody, K.duct_geom(duct["id"], k, "col"), lo, hi, params, bits, pose=pose)
        _add_lidar(spec.worldbody, K.duct_geom(duct["id"], k, "lidar"), c, params, duct["wall_class"], pose=pose)


# ------------------------------------------------------------------------------------------------ entry point
def add_ducts(spec, world, pushable=True, params=None, mode=None):
    """Add the ducts of `world` to an MjSpec. `mode` overrides duct_params.json `mode` ('sleep',
    'awake', 'active_window'); pushable=False gives static ducts whatever the mode. Sets the sleep flag of the
    shared model in mode 'sleep' (and in 'active_window' when window.slot_sleep is true): the contract's option block. Returns the mode used."""
    mj = _mj()
    p = params if params is not None else load_params()
    err = check_params(p)
    if err:
        raise ValueError("duct parameters: " + "; ".join(err))
    md = resolve_mode(p, pushable, mode)
    if md != "static" and p["duct_duct_contact"]:
        # the capsules of consecutive pieces overlap at their shared end; contact between them would need an
        # exclude per pin, and the window's slots would collide with the frozen layer: not supported
        raise ValueError("duct_duct_contact = true is not supported by the pushable modes")
    bits = duct_bits(p)
    ducts = world["truth"]["ducts"]
    if md == "static":
        for d in ducts:
            _add_static_duct(spec, d, p, bits)
    elif md in ("sleep", "awake"):
        pol = None
        if md == "sleep":
            pol = {"init": mj.mjtSleepPolicy.mjSLEEP_INIT, "allowed": mj.mjtSleepPolicy.mjSLEEP_ALLOWED,
                   "auto": mj.mjtSleepPolicy.mjSLEEP_AUTO}[p["sleep"]["policy"]]
        for d in ducts:
            _add_dynamic_duct(spec, d["id"], d["pieces"], float(d["piece_length_m"]), d["wall_class"], p, bits, pol)
    else:
        for d in ducts:
            _add_static_duct(spec, d, p, bits)
        add_window_slots(spec, world, p, bits)
    if md == "sleep" or (md == "active_window" and p["window"]["slot_sleep"]):
        spec.option.enableflags |= int(mj.mjtEnableBit.mjENBL_SLEEP)
        spec.option.sleep_tolerance = float(p["sleep"]["tolerance"])
    spec.add_text(name=TEXT_DIGEST, data=params_digest(p))
    spec.add_text(name=TEXT_MODE, data=md)
    return md


def add_window_slots(spec, world, params, bits):
    """The dynamic chains of the active window: `window.slots` chains of as many pieces as the world's longest duct,
    laid out straight at the parking place with the piece length of the world's most common duct class. They carry
    collision capsules only (the LiDAR always sees the frozen layer's cylinders, which the window keeps on them)."""
    mj = _mj()
    w = params["window"]
    ids = slot_ids(world, int(w["slots"]))
    classes = duct_classes(world)
    n_max = max(n for n, _ in classes) if classes else 1
    c0 = classes.most_common(1)[0][0][1] if classes else float(params["nominal_piece_length_m"])
    px, py = (float(v) for v in w["park_origin_m"])
    pol = mj.mjtSleepPolicy.mjSLEEP_INIT if w["slot_sleep"] else None
    for i, sid in enumerate(ids):
        poses = [[px + (k + 0.5) * c0, py + 2.0 * i, 0.0] for k in range(n_max)]
        _add_dynamic_duct(spec, sid, poses, c0, "orange_ribbed", params, bits, sleep_policy=pol, lidar=False)
    return ids
