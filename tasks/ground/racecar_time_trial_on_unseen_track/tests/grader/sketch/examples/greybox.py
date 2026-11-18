"""Reference greybox model of a world file: an executable example of the contract.

    xml   = build_xml(world, variant="track", cars={"car/": world["truth"]["start_pose"]["pose"]})
    model = mujoco.MjModel.from_xml_string(xml)

What it is: the floor, floor zone slabs and decals, tape lines, the timing line, clutter, STATIC ducts (collision
capsule + LiDAR cylinder per piece), free boxes in the `avoid` variant, and STUB cars that carry every name, site and
geom the contract makes binding. It lets a package that needs a world and a car body start before the
venue exporter, the pushable ducts and the drive model exist, and it gives `check.check_model` a model
to accept.

What it is not: the stub car has no suspension, no steering linkage, no drive line and no tuned mass distribution;
the ducts cannot be pushed; `ideal_scan_mm` is one instantaneous fan without sub-scans, noise, material range or the
floor rule. The drive model, the venue generator, the duct model and the sensor model replace these parts; their models must pass the same `check.check_model`.
"""

import math

import numpy as np

from . import constants as K

TAPE_WIDTH_M = 0.05
LINE_WIDTH_M = 0.10
RGBA = {"orange_ribbed": "0.95 0.45 0.10 1", "yellow_ribbed": "0.95 0.80 0.10 1", "black_plain": "0.08 0.08 0.08 1",
        "floor": "0.55 0.55 0.57 1", "clutter": "0.45 0.45 0.50 1", "box": "0.92 0.92 0.90 1", "car": "0.15 0.25 0.70 1",
        "tape": "0.95 0.85 0.10 1", "line": "0.95 0.95 0.95 1", "dust": "0.70 0.66 0.58 1", "metal_cover": "0.62 0.64 0.68 1",
        "carpet_tile": "0.15 0.30 0.70 1"}


def _f(*v):
    return " ".join("%.9g" % x for x in v)


def _bits(cat):
    return 'contype="%d" conaffinity="%d"' % K.CONTACT[cat]


def _yaw_quat(yaw):
    return _f(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _strip(name, a, b, width, group, rgba, z=0.001):
    """A flat marking on the floor from point a to point b."""
    cx, cy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    return ('<geom name="%s" type="box" size="%s" pos="%s" quat="%s" group="%d" %s rgba="%s"/>'
            % (name, _f(0.5 * math.dist(a, b), 0.5 * width, 0.0005), _f(cx, cy, z), _yaw_quat(yaw), group, _bits("none"), rgba))


def _is_rectangle(poly):
    if len(poly) != 4:
        return False
    v = np.asarray(poly, dtype=float)
    e = np.roll(v, -1, axis=0) - v
    return all(abs(float(e[i] @ e[(i + 1) % 4])) < 1e-3 for i in range(4))


def duct_geoms(duct):
    """The two geoms of every piece of a static duct: `.../col` (capsule, group 1) and `.../lidar` (cylinder, group 2)."""
    out, r, c, n = [], K.DUCT_RADIUS_M, duct["piece_length_m"], len(duct["pieces"])
    for k, (x, y, yaw) in enumerate(duct["pieces"]):
        ux, uy = math.cos(yaw), math.sin(yaw)
        lo, hi = -0.5 * c, 0.5 * c
        out.append('<geom name="%s" type="cylinder" fromto="%s" size="%s" group="%d" %s rgba="%s"/>'
                   % (K.duct_geom(duct["id"], k, "lidar"), _f(x + lo * ux, y + lo * uy, r, x + hi * ux, y + hi * uy, r), _f(r),
                      K.GROUP_LIDAR, _bits("none"), RGBA[duct["wall_class"]]))
        # collision capsule: the round cap of a free end stays inside the duct's end plane
        if k == 0:
            lo += r
        if k == n - 1:
            hi -= r
        out.append('<geom name="%s" type="capsule" fromto="%s" size="%s" group="%d" %s rgba="0.3 0.3 0.3 0.3"/>'
                   % (K.duct_geom(duct["id"], k, "col"), _f(x + lo * ux, y + lo * uy, r, x + hi * ux, y + hi * uy, r), _f(r),
                      K.GROUP_COLLISION, _bits("duct")))
    return out


def stub_car(prefix, pose, tail_box=None):
    """A stand-in for the drive model's car with every binding name: body `chassis` (its frame is the car frame C) on the free joint
    `root`, joint `steer`, four `<w>/wheel` bodies with `<w>/spin` joints and `<w>/tyre` geoms, the LiDAR targets on
    the chassis, and the sites `lidar`, `imu`, `ref`. `tail_box` defaults to True for every prefix but the ego car's."""
    if tail_box is None:
        tail_box = prefix != K.EGO_PREFIX
    x, y, yaw = pose
    p = prefix
    hull_lo, hull_hi = K.GROUND_CLEARANCE_M, 0.10666           # tub underside .. top of the compute unit
    g = ['<body name="%schassis" pos="%s" quat="%s">' % (p, _f(x, y, 0.0), _yaw_quat(yaw)),
         '<freejoint name="%sroot"/>' % p,
         '<inertial pos="%s" mass="%s" diaginertia="0.004 0.02 0.022"/>' % (_f(*K.COM_C), _f(K.MASS_KG - 0.1)),
         '<geom name="%schassis/col0" type="box" size="%s" pos="%s" group="%d" %s rgba="%s"/>'
         % (p, _f(0.5 * (K.X_FRONT_M - K.X_REAR_M), 0.055, 0.5 * (0.070 - hull_lo)),
            _f(0.5 * (K.X_FRONT_M + K.X_REAR_M), 0.0, 0.5 * (0.070 + hull_lo)), K.GROUP_COLLISION, _bits("car"), RGBA["car"]),
         '<geom name="%shull/lidar" type="box" size="%s" pos="%s" group="%d" %s rgba="0.2 0.2 0.2 0.2"/>'
         % (p, _f(0.5 * (K.X_FRONT_M - K.X_REAR_M), 0.5 * K.ENVELOPE_M[1], 0.5 * (hull_hi - hull_lo)),
            _f(0.5 * (K.X_FRONT_M + K.X_REAR_M), 0.0, 0.5 * (hull_hi + hull_lo)), K.GROUP_LIDAR, _bits("none")),
         '<geom name="%slidar_head/lidar" type="cylinder" size="0.0173 0.01185" pos="%s" group="%d" %s rgba="0.05 0.05 0.05 1"/>'
         % (p, _f(K.LIDAR_POS_C[0], 0.0, 0.10985), K.GROUP_LIDAR, _bits("none")),
         '<site name="%slidar" pos="%s"/>' % (p, _f(*K.LIDAR_POS_C)),
         '<site name="%simu" pos="%s"/>' % (p, _f(*K.IMU_POS_C)),
         '<site name="%sref" pos="%s"/>' % (p, _f(*K.REF_POINT_C)),
         '<body name="%srack" pos="%s"><joint name="%ssteer" type="hinge" axis="0 0 1" range="%s"/>'
         '<inertial pos="0 0 0" mass="0.004" diaginertia="1e-6 1e-6 1e-6"/></body>'
         % (p, _f(K.WHEELBASE_M, 0.0, K.TYRE_RADIUS_M), p, _f(-K.STEER_MAX_RAD, K.STEER_MAX_RAD))]
    if tail_box:
        sx, sy, sz = K.TAIL_BOX_SIZE_M
        for role, group, cat in (("col", K.GROUP_COLLISION, "car"), ("lidar", K.GROUP_LIDAR, "none")):
            g.append('<geom name="%stail_box/%s" type="box" size="%s" pos="%s" group="%d" %s mass="0" rgba="%s"/>'
                     % (p, role, _f(0.5 * sx, 0.5 * sy, 0.5 * sz), _f(*K.TAIL_BOX_CENTRE_C), group, _bits(cat), RGBA["box"]))
    for w in K.WHEELS:
        wx, wy, _ = K.WHEEL_GROUND_C[w]
        g.append('<body name="%s%s/wheel" pos="%s"><joint name="%s%s/spin" type="hinge" axis="0 1 0"/>'
                 '<geom name="%s%s/tyre" type="cylinder" size="%s" quat="0.7071068 0.7071068 0 0" mass="0.024" group="%d" %s '
                 'rgba="0.1 0.1 0.1 1"/></body>'
                 % (p, w, _f(wx, wy, K.TYRE_RADIUS_M), p, w, p, w, _f(K.TYRE_RADIUS_M, 0.5 * K.TYRE_WIDTH_M),
                    K.GROUP_COLLISION, _bits("car")))
    g.append("</body>")
    return g


def build_xml(world, variant="track", cars=None, debug_markers=False, timestep=K.PHYSICS_DT_S):
    """MJCF text of the static greybox model of `world`.

    variant  'track' (sessions practice, tt1, tt2) or 'avoid' (adds the boxes as free bodies)
    cars     {prefix: [x, y, yaw]}: stub cars to add ('car/' is the ego car, 'slow0/' ... carry a tail box); None = no car
    """
    if variant not in ("track", "avoid"):
        raise ValueError("variant must be 'track' or 'avoid'")
    t = world["truth"]
    out = ['<mujoco model="%s">' % world["world_id"],
           '<compiler angle="radian" autolimits="true"/>',
           '<option timestep="%s" cone="elliptic"/>' % _f(timestep),
           "<worldbody>",
           '<geom name="floor" type="plane" size="0 0 0.05" group="%d" %s rgba="%s"/>' % (K.GROUP_FLOOR, _bits("floor"), RGBA["floor"])]
    for p in t["floor"]["patches"]:
        if not _is_rectangle(p["polygon"]):
            continue                                   # a flush patch of any other shape has no geometry at all
        v = p["polygon"]
        cx, cy = sum(q[0] for q in v) / 4.0, sum(q[1] for q in v) / 4.0
        lx, ly = math.dist(v[0], v[1]), math.dist(v[1], v[2])
        yaw = math.atan2(v[1][1] - v[0][1], v[1][0] - v[0][0])
        h = p["edge_height_m"]
        if h > 0.0:
            out.append('<geom name="%s" type="box" size="%s" pos="%s" quat="%s" group="%d" %s rgba="%s"/>'
                       % (K.zone_geom(p["id"], "slab"), _f(0.5 * lx, 0.5 * ly, 0.5 * h), _f(cx, cy, 0.5 * h), _yaw_quat(yaw),
                          K.GROUP_FLOOR, _bits("floor"), RGBA[p["kind"]]))
        else:
            out.append('<geom name="%s" type="box" size="%s" pos="%s" quat="%s" group="%d" %s rgba="%s"/>'
                       % (K.zone_geom(p["id"], "decal"), _f(0.5 * lx, 0.5 * ly, 0.0005), _f(cx, cy, 0.0005), _yaw_quat(yaw),
                          K.GROUP_VISUAL, _bits("none"), RGBA[p["kind"]]))
    for o in t["open_walls"]:
        out.append(_strip("tape/%s" % o["id"], o["tape"][0], o["tape"][1], TAPE_WIDTH_M, K.GROUP_VISUAL, RGBA["tape"]))
    out.append(_strip("timing_line", t["timing_line"]["a"], t["timing_line"]["b"], LINE_WIDTH_M, K.GROUP_VISUAL, RGBA["line"]))
    if debug_markers:
        for c in t["checkpoints"]:
            out.append(_strip("checkpoint/%s" % c["id"], c["a"], c["b"], 0.03, K.GROUP_HIDDEN, "0.2 0.9 0.2 1"))
    for c in t["clutter"]:
        lx, ly, lz = c["size_m"]
        shape = ('type="box" size="%s"' % _f(0.5 * lx, 0.5 * ly, 0.5 * lz) if c["shape"] == "box"
                 else 'type="cylinder" size="%s"' % _f(0.5 * lx, 0.5 * lz))
        out.append('<geom name="%s" %s pos="%s" quat="%s" group="%d" %s rgba="%s"/>'
                   % (K.clutter_geom(c["id"]), shape, _f(c["pose"][0], c["pose"][1], 0.5 * lz), _yaw_quat(c["pose"][2]),
                      K.GROUP_CLUTTER, _bits("none"), RGBA["clutter"]))
    for d in t["ducts"]:
        out += duct_geoms(d)
    for prefix, pose in (cars or {}).items():
        out += stub_car(prefix, pose)
    if variant == "avoid":
        for bx in t["avoidance"]["boxes"]:
            lx, ly, lz = bx["size_m"]
            size = _f(0.5 * lx, 0.5 * ly, 0.5 * lz)
            out.append('<body name="%s" pos="%s" quat="%s"><freejoint name="%s/root"/>'
                       '<geom name="%s" type="box" size="%s" mass="%s" group="%d" %s rgba="%s"/>'
                       '<geom name="%s" type="box" size="%s" mass="0" group="%d" %s rgba="%s"/></body>'
                       % (K.box_body(bx["id"]), _f(bx["pose"][0], bx["pose"][1], 0.5 * lz), _yaw_quat(bx["pose"][2]), K.box_body(bx["id"]),
                          K.box_geom(bx["id"], "col"), size, _f(bx["mass_kg"]), K.GROUP_COLLISION, _bits("box"), RGBA["box"],
                          K.box_geom(bx["id"], "lidar"), size, K.GROUP_LIDAR, _bits("none"), RGBA["box"]))
    out += ["</worldbody>", "</mujoco>"]
    return "\n".join(out) + "\n"


def build_model(world, **kw):
    import mujoco
    return mujoco.MjModel.from_xml_string(build_xml(world, **kw))


def place_car(model, data, pose, prefix=K.EGO_PREFIX):
    """Put a car's frame C at the planar pose [x, y, yaw] on the floor, at rest (kinematic placement)."""
    import mujoco
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "root")
    qa, va = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    data.qpos[qa:qa + 7] = [pose[0], pose[1], 0.0, math.cos(0.5 * pose[2]), 0.0, 0.0, math.sin(0.5 * pose[2])]
    data.qvel[va:va + 6] = 0.0
    mujoco.mj_forward(model, data)


def ideal_scan(model, data, prefix=K.EGO_PREFIX, floor=False):
    """One instantaneous fan of the 1081 beams from the site `<prefix>lidar`: (range in metres or -1, geom id or -1).
    Targets are the LiDAR groups of the contract; the car's own targets are skipped through `bodyexclude`."""
    import mujoco
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "lidar")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "chassis")
    ang = K.ANGLE_MIN_RAD + K.ANGLE_INC_RAD * np.arange(K.N_BEAMS)
    local = np.stack([np.cos(ang), np.sin(ang), np.zeros(K.N_BEAMS)])
    vec = np.ascontiguousarray((data.site_xmat[sid].reshape(3, 3) @ local).T).reshape(-1)
    geomid = np.full(K.N_BEAMS, -1, np.int32)
    dist = np.full(K.N_BEAMS, -1.0)
    mask = np.array(K.LIDAR_FLOOR_GROUPMASK if floor else K.LIDAR_GROUPMASK, np.uint8)
    mujoco.mj_multiRay(model, data, data.site_xpos[sid].copy(), vec, mask, 1, cid, geomid, dist, None, K.N_BEAMS, K.RANGE_MAX_M)
    return dist, geomid


def to_wire_mm(dist):
    """Ranges in metres -> the wire's integers: millimetres, 0 where there is no return within 0.06 .. 10 m."""
    d = np.asarray(dist, dtype=float)
    ok = (d >= K.RANGE_MIN_M) & (d <= K.RANGE_MAX_M)
    return np.where(ok, np.rint(d * 1000.0), K.NO_RETURN).astype(np.int64).tolist()
