"""Box transport: a robot environment (the table, the Panda and its controller that lab_sim.py builds on).

Plain MuJoCo 3.16 and a fixed Panda (robosuite 1.5.2 model files; robosuite is not imported), the
operational-space controller at 10 Hz, two fixed cameras and a wrist camera, a wrist force/torque sensor,
all as in tasks/arm/kitchen_appliance_tending's appliance_env.py (itself after
tasks/arm/arm_hidden_mechanism_box/tests/box_env.py). What it loads is one **unit**: a container and an
item, both lightwheel object models from the CC-BY-4.0 dataset
`nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` (no RoboCasa code runs), each uniformly
scaled and given an explicit mass (inertia scaled with it, at fixed geometry). The item starts on the
table; the job is to load it into the container and carry the loaded container to a target marked on
the table without spilling.

Frames: world z up; the table top is z = TABLE_Z; the robot base sits on it at the origin facing +x.
Metres, radians, seconds; quaternions (w, x, y, z).

    spec = unit_spec("tupperware/Tupperware036", "marshmallow/Marshmallow005", item_mass=0.15)
    env = BoxTransportEnv(spec, assets="~/arb-data/robocasa-assets/objects/extracted")
    obs = env.reset(seed)
    obs, done, info = env.step([dx, dy, dz, drx, dry, drz, grip, cams])
    env.progress()           # the grader-side predicates (loaded, lifted, carried, delivered, spilled)

No request text, no credit, no wire.
"""

import math
import os

import mujoco
import numpy as np

TIMESTEP = 0.002
SUBSTEPS = 50                     # -> 10 Hz control
CONTROL_DT = TIMESTEP * SUBSTEPS
MAX_DPOS = 0.03
MAX_DROT = 0.2
MAX_ARM_SPEED = 8.0               # rad/s: faster is an unstable simulation (box_env's rule)
MAX_OBJECT_SPEED = 5.0            # m/s, the container or the item: a simulation blow-up, not a throw
WRENCH_NOISE = 0.3
RGB_NOISE = 2.0
VISUAL_MASS = 1e-6
IMPRATIO = 10.0                   # box_env uses 2.0; 10 stops a held container creeping out of the pads
NOSLIP_ITERATIONS = 0                # visual meshes carry no mass (as robosuite does for its objects)
# Contact model "stiff": the default model lets a held object creep in the pads under any moment about the
# grasp; torsional friction on the pads and the objects plus the no-slip solver give a friction limit instead.
STIFF_NOSLIP_ITERATIONS = 50
STIFF_TORSION = 0.003                # m, torsional friction (condim 4) of the pads and the objects
STIFF_ARENA_BYTES = 64 * 2 ** 20     # the default (about 13 MB here) overflowed in 9 of 600 trials

TABLE_Z = 0.80
TABLE_HALF = (0.45, 0.80)         # half size (x, y); the table spans x in [-0.15, 0.75], y in [-0.80, 0.80]
TABLE_CENTRE_X = 0.30
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, math.pi / 4])

# Nominal layout (x, y) on the table, robot frame; each episode jitters it (JITTER).
CONTAINER_XY = np.array([0.52, 0.20])
ITEM_XY = np.array([0.62, -0.02])
TARGET_XY = np.array([0.52, -0.22])
JITTER = {"container": (0.03, 0.03, math.radians(20)), "item": (0.03, 0.03, math.pi),
          "target": (0.03, 0.03)}

# Acceptance (experiments/physical-execution/BOX_TRANSPORT_TEST.md, unchanged)
MIN_LIFT = 0.08                   # m, container above its resting height while held and airborne
MIN_CARRY = 0.30                  # m, horizontal displacement while held, airborne and loaded
DELIVERY_TOL = 0.035              # m, container centre to target centre (xy)
REST_HEIGHT_TOL = 0.012           # m, container back at its support height
REST_SPEED = 0.025                # m/s, container and item
REST_ANG_SPEED = 0.10             # rad/s, container
GRIPPER_CLEAR = 0.25              # m, gripper site from the container at the end
STABLE_STEPS = 20                 # control steps the delivered state must hold
INTERIOR_MARGIN = 0.004           # m: the item's centre may reach this far past the interior rectangle
CENTRE_Z_SLACK = 0.01             # m: the item's centre between floor and rim, with this slack
LOADED_STEPS = 10                 # control steps the item must rest in the container, released, to count

CAMERAS = {
    # name: position, look-at point (world), fovy, width, height
    "left": ((1.15, 0.55, 1.35), (0.50, 0.0, TABLE_Z), 50.0, 320, 240),
    "right": ((1.15, -0.55, 1.35), (0.50, 0.0, TABLE_Z), 50.0, 320, 240),
}
WRIST = {"fovy": 75.0, "width": 256, "height": 192}
CAMERA_NAMES = ("left", "right", "wrist")


class EnvError(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# Small helpers (appliance_env's)
# ----------------------------------------------------------------------------------------------
def robosuite_assets():
    import importlib.util
    spec = importlib.util.find_spec("robosuite")
    if spec is None or not spec.submodule_search_locations:
        raise EnvError("robosuite assets not installed")
    return os.path.join(next(iter(spec.submodule_search_locations)), "models", "assets")


def look_at(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.column_stack([r, u, -f])          # MuJoCo camera: looks along -z, y up


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def axis_angle_mat(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ----------------------------------------------------------------------------------------------
# Unit = container + item
# ----------------------------------------------------------------------------------------------
def encode_image(img):
    import base64
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return {"dtype": "uint8", "shape": list(img.shape), "b64": base64.b64encode(img.tobytes()).decode("ascii")}


def decode_image(d):
    import base64
    return np.frombuffer(base64.b64decode(d["b64"]), np.uint8).reshape(d["shape"])


def to_wire(obs):
    """Observation with numpy arrays -> JSON-ready dict (images base64, box_env's format)."""
    out = {}
    for k, v in obs.items():
        if k in CAMERA_NAMES:
            out[k] = encode_image(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def unit_spec(container, item, container_scale=None, item_scale=1.0, container_mass=0.08, item_mass=0.10,
              container_width=0.065, item_com_offset=0.0, friction=None, item_sphere=None, rolling=0.0,
              restitution=None, container_depth_scale=1.0, contact="soft"):
    """`container` and `item` are '<category>/<Model>' folders under the assets root. The container is
    scaled so its short horizontal side is `container_width` unless `container_scale` is given.

    Hidden properties: `item_mass` (kg; inertia scaled with it at fixed geometry), `container_mass`
    (kg), `item_com_offset` (the item's centre of mass moved along its longer horizontal axis by this
    fraction of the half-length: ballast at one end), `friction` (the container's sliding friction; it
    governs every contact of the container, pads included; None keeps the model's).

    Rolling loads: `item_sphere` (m) replaces the item's collision pieces with one sphere of that
    radius at their box centre (the visual mesh keeps `item_scale`), condim 6, the item's friction governing
    its contacts; `rolling` is its rolling friction (m: resisting torque per unit normal force);
    `restitution` (0-1, None = MuJoCo's default soft contact) sets the sphere's contact damping ratio
    from a target coefficient of restitution. `container_depth_scale` scales the container's height only
    (a shallower container of the same footprint).

    `contact`: "soft" (the default model) or "stiff" (condim 4 with STIFF_TORSION on the pads, the container and
    the item, and STIFF_NOSLIP_ITERATIONS; sliding friction unchanged)."""
    if contact not in ("soft", "stiff"):
        raise EnvError(f"contact: {contact!r}")
    return {"container": container, "item": item, "container_scale": container_scale,
            "item_scale": float(item_scale), "container_mass": float(container_mass),
            "item_mass": float(item_mass), "container_width": float(container_width),
            "item_com_offset": float(item_com_offset), "friction": None if friction is None else float(friction),
            "item_sphere": None if item_sphere is None else float(item_sphere), "rolling": float(rolling),
            "restitution": None if restitution is None else float(restitution),
            "container_depth_scale": float(container_depth_scale), "contact": contact}


def load_object(folder, scale):
    """MjSpec of a lightwheel model.xml, scaled (a number, or (sx, sy, sz) for a container made shallower),
    visual meshes massless, collision geoms hidden from the cameras, a free joint on its `object` body."""
    folder = os.path.expanduser(folder)
    sv = np.broadcast_to(np.asarray(scale, float), (3,))
    u = mujoco.MjSpec.from_file(os.path.join(folder, "model.xml"))
    for mesh in u.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.join(folder, mesh.file)
        mesh.scale = np.asarray(mesh.scale) * sv
    for tex in u.textures:
        if tex.file and not os.path.isabs(tex.file):
            tex.file = os.path.join(folder, tex.file)
    for b in u.bodies:
        b.pos = np.asarray(b.pos) * sv
    for g in u.geoms:
        g.pos = np.asarray(g.pos) * sv
        if g.type == mujoco.mjtGeom.mjGEOM_BOX:
            g.size = np.asarray(g.size) * sv
        elif g.type != mujoco.mjtGeom.mjGEOM_MESH:
            g.size = np.asarray(g.size) * sv[0]
        if g.contype == 0 and g.conaffinity == 0:
            g.mass = VISUAL_MASS
            if g.type != mujoco.mjtGeom.mjGEOM_MESH:
                g.rgba = [0, 0, 0, 0]
        else:
            g.rgba = [0, 0, 0, 0]
    for s in u.sites:
        s.pos = np.asarray(s.pos) * sv
    return u


def collision_box(m, d, body):
    """Axis-aligned box over a body's collision geoms, in the body's own frame."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    Rb = d.xmat[body].reshape(3, 3)
    pb = d.xpos[body]
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        R = Rb.T @ d.geom_xmat[g].reshape(3, 3)
        c = Rb.T @ (d.geom_xpos[g] - pb) + R @ m.geom_aabb[g, :3]
        h = np.abs(R) @ m.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


def container_geometry(folder, scale):
    """The scaled container standalone: collision box, rim height, floor height and the open interior
    (x, y extents in its frame) from rays cast straight down through the mouth."""
    u = load_object(folder, scale)
    m = u.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = m.body("object").id
    lo, hi = collision_box(m, d, body)
    top = hi[2] + 0.05
    half_z = lo[2] + 0.5 * (hi[2] - lo[2])
    gid = np.zeros(1, np.int32)
    n = 41
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    inside = np.zeros((n, n), bool)
    floor = np.full((n, n), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            dist = mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0, 0, -1.0]), None, 1, -1, gid)
            if dist >= 0:
                floor[i, j] = top - dist
                inside[i, j] = floor[i, j] < half_z
    if not inside.any():
        raise EnvError(f"{folder}: no open interior found")
    # The largest axis-aligned rectangle of interior cells around the centre (shrink from the hull of
    # interior cells until every cell in it is interior).
    ii, jj = np.nonzero(inside)
    i0, i1, j0, j1 = ii.min(), ii.max(), jj.min(), jj.max()
    while not inside[i0:i1 + 1, j0:j1 + 1].all():
        widths = [(i1 - i0), (j1 - j0)]
        if widths[0] >= widths[1]:
            i0, i1 = i0 + 1, i1 - 1
        else:
            j0, j1 = j0 + 1, j1 - 1
        if i0 >= i1 or j0 >= j1:
            raise EnvError(f"{folder}: interior not rectangular")
    return {"lo": lo, "hi": hi, "rim_z": float(hi[2]),
            "floor_z": float(np.nanmax(floor[i0:i1 + 1, j0:j1 + 1])),
            "interior_lo": np.array([xs[i0], ys[j0]]), "interior_hi": np.array([xs[i1], ys[j1]])}


def resolve_scale(assets, spec):
    if spec["container_scale"]:
        return float(spec["container_scale"])
    u = load_object(os.path.join(assets, spec["container"]), 1.0)
    m = u.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = collision_box(m, d, m.body("object").id)
    return spec["container_width"] / float(min(hi[0] - lo[0], hi[1] - lo[1]))


def set_mass(m, body, mass):
    """Explicit mass at fixed geometry: mass and inertia scaled by the same factor (the method of
    loaded_tray_transport.configure_payload_mass)."""
    ratio = mass / float(m.body_mass[body])
    m.body_mass[body] *= ratio
    m.body_inertia[body] *= ratio


def sphere_item(u, centre, spec):
    """Replace the item's collision pieces with one sphere (rolling loads); returns its box (lo, hi)."""
    r = spec["item_sphere"]
    for g in list(u.geoms):
        if g.contype or g.conaffinity:
            u.delete(g)
    g = u.body("object").add_geom()
    g.name = "ball"
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [r, 0, 0]
    g.pos = list(centre)
    g.condim = 6
    g.priority = 2                     # the item's friction (sliding 1.0, rolling hidden) governs its contacts
    g.friction = [1.0, 0.002, spec.get("rolling", 0.0)]
    g.rgba = [0, 0, 0, 0]
    if spec.get("restitution") is not None:
        # solref (timeconst, dampratio): a damping ratio zeta gives a restitution of about
        # exp(-pi zeta / sqrt(1 - zeta^2)); invert it.
        e = min(max(spec["restitution"], 0.01), 0.95)
        ln = math.log(e)
        g.solref = [0.004, -ln / math.sqrt(math.pi ** 2 + ln ** 2)]
    return np.asarray(centre) - r, np.asarray(centre) + r


def build_model(assets, spec, poses):
    """Compile table + Panda + the unit. `poses` = {"container": (x, y, yaw), "item": (x, y, yaw),
    "target": (x, y)}: the container's short axis is its local y."""
    assets = os.path.expanduser(assets)
    c_scale = resolve_scale(assets, spec)
    c_sv = c_scale * np.array([1.0, 1.0, spec.get("container_depth_scale", 1.0)])
    geo = container_geometry(os.path.join(assets, spec["container"]), c_sv)
    cu = load_object(os.path.join(assets, spec["container"]), c_sv)
    iu = load_object(os.path.join(assets, spec["item"]), spec["item_scale"])
    im = iu.compile()
    idat = mujoco.MjData(im)
    mujoco.mj_forward(im, idat)
    ilo, ihi = collision_box(im, idat, im.body("object").id)
    iu = load_object(os.path.join(assets, spec["item"]), spec["item_scale"])
    if spec.get("item_sphere"):
        ilo, ihi = sphere_item(iu, (ilo + ihi) / 2, spec)

    w = mujoco.MjSpec()
    w.compiler.degree = False
    w.compiler.autolimits = True
    w.option.timestep = TIMESTEP
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w.option.impratio = IMPRATIO
    w.option.noslip_iterations = NOSLIP_ITERATIONS
    w.visual.global_.offwidth = 640
    w.visual.global_.offheight = 480
    if spec.get("contact", "soft") == "stiff":
        w.memory = STIFF_ARENA_BYTES     # condim 4 + no-slip: the default arena overflowed
    wb = w.worldbody
    light = wb.add_light()
    light.pos = [0.4, 0.0, 2.4]
    light.dir = [0.1, 0.0, -1.0]
    light.diffuse = [0.8, 0.8, 0.8]
    light.castshadow = False
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [3, 3, 0.1]
    g.rgba = [0.55, 0.52, 0.48, 1]
    g = wb.add_geom()
    g.name = "table"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [TABLE_HALF[0], TABLE_HALF[1], TABLE_Z / 2]
    g.pos = [TABLE_CENTRE_X, 0.0, TABLE_Z / 2]
    g.rgba = [0.78, 0.74, 0.68, 1]
    tx, ty = poses["target"]
    g = wb.add_geom()
    g.name = "target"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [0.03, 0.0005, 0]
    g.pos = [tx, ty, TABLE_Z + 0.0005]
    g.rgba = [0.15, 0.45, 0.85, 1]
    g.contype = 0
    g.conaffinity = 0
    for name, (p, tgt, fovy, _w, _h) in CAMERAS.items():
        c = wb.add_camera()
        c.name = name
        c.pos = list(p)
        c.quat = list(quat_from_mat(look_at(p, tgt)))
        c.fovy = fovy
    ra = robosuite_assets()
    robot = mujoco.MjSpec.from_file(os.path.join(ra, "robots", "panda", "robot.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(ra, "grippers", "panda_gripper.xml"))
    for cam in list(robot.cameras):
        if cam.name == "robotview":
            robot.delete(cam)
    robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
    wb.add_frame(pos=[0.0, 0.0, TABLE_Z]).attach_body(robot.body("base"), "r_", "")
    # container: its collision box bottom on the table; item likewise, at their poses
    cx, cy, cyaw = poses["container"]
    cb = cu.body("object")
    cb.add_freejoint().name = "free"
    # A floor slab under the interior: the scaled convex pieces of a thin floor let a dropped item
    # tunnel through (Tupperware039); the slab's top is the floor the rays found.
    fl = cb.add_geom()
    fl.name = "floor_slab"
    fl.type = mujoco.mjtGeom.mjGEOM_BOX
    flo, fhi = geo["interior_lo"], geo["interior_hi"]
    fl.size = [(fhi[0] - flo[0]) / 2, (fhi[1] - flo[1]) / 2, (geo["floor_z"] - geo["lo"][2]) / 2]
    fl.pos = [(fhi[0] + flo[0]) / 2, (fhi[1] + flo[1]) / 2, (geo["floor_z"] + geo["lo"][2]) / 2]
    fl.rgba = [0, 0, 0, 0]
    fl.mass = VISUAL_MASS
    wb.add_frame(pos=[cx, cy, TABLE_Z - geo["lo"][2] + 0.0005], quat=list(yaw_quat(cyaw))).attach_body(cb, "c_", "")
    ix, iy, iyaw = poses["item"]
    ib = iu.body("object")
    ib.add_freejoint().name = "free"
    wb.add_frame(pos=[ix, iy, TABLE_Z - ilo[2] + 0.0005], quat=list(yaw_quat(iyaw))).attach_body(ib, "i_", "")
    m = w.compile()
    m.site_rgba[:, 3] = 0.0
    set_mass(m, m.body("c_object").id, spec["container_mass"])
    ib_id = m.body("i_object").id
    set_mass(m, ib_id, spec["item_mass"])
    off = spec.get("item_com_offset", 0.0)
    if off:
        # ballast at one end: the centre of mass moves along the item's longer horizontal axis (body frame)
        axis = 0 if (ihi[0] - ilo[0]) >= (ihi[1] - ilo[1]) else 1
        centre = (ilo + ihi) / 2
        ipos = m.body_ipos[ib_id].copy()
        ipos[axis] = centre[axis] + off * (ihi[axis] - ilo[axis]) / 2
        m.body_ipos[ib_id] = ipos
    if spec.get("friction") is not None:
        # MuJoCo takes the larger friction of a pair unless one geom has priority: the container's material
        # then governs its contacts with the pads (2.0), the table and the item.
        cb_id = m.body("c_object").id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == cb_id and (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_friction[g, 0] = spec["friction"]
                m.geom_priority[g] = 1
    if spec.get("contact", "soft") == "stiff":
        m.opt.noslip_iterations = STIFF_NOSLIP_ITERATIONS
        bodies = {m.body("c_object").id, ib_id}
        pads = {m.geom("r_g_finger1_pad_collision").id, m.geom("r_g_finger2_pad_collision").id}
        for g in range(m.ngeom):
            if g in pads or (m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])):
                m.geom_condim[g] = max(4, int(m.geom_condim[g]))
                m.geom_friction[g, 1] = STIFF_TORSION
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m, {"container": geo, "container_scale": c_scale, "item_lo": ilo, "item_hi": ihi,
               "poses": poses}


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------
class BoxTransportEnv:

    def __init__(self, spec, assets, render=True):
        self.spec = spec
        self.assets = assets
        self.render_enabled = render
        self.model = None
        self._renderers = {}

    def reset(self, seed, poses=None):
        self.seed = int(seed) & (2 ** 64 - 1)
        lo32, hi32 = self.seed & 0xFFFFFFFF, self.seed >> 32
        place = np.random.default_rng([lo32, hi32, 7])
        self.rng = np.random.default_rng([lo32, hi32, 11])
        if poses is None:
            jc, ji, jt = JITTER["container"], JITTER["item"], JITTER["target"]
            poses = {
                "container": tuple(np.r_[CONTAINER_XY, 0.0] + place.uniform(-1, 1, 3) * np.asarray(jc)),
                "item": tuple(np.r_[ITEM_XY, 0.0] + place.uniform(-1, 1, 3) * np.asarray(ji)),
                "target": tuple(TARGET_XY + place.uniform(-1, 1, 2) * np.asarray(jt)),
            }
        self.model, self.layout = build_model(self.assets, self.spec, poses)
        self.data = mujoco.MjData(self.model)
        self._index()
        self._init_state()
        self.t = 0
        self.render_next = ("left", "right", "wrist")
        self._renderers = {}
        self.monitor = Monitor(self)
        return self.observation()

    def preload(self, steps=15):
        """Put the item at rest on the container's floor, at the interior centre (the carry measured
        without the loading stage). The Monitor latches loaded as it would after a drop."""
        m, d = self.model, self.data
        g = self.layout["container"]
        ic = (g["interior_lo"] + g["interior_hi"]) / 2
        centre = (self.layout["item_lo"] + self.layout["item_hi"]) / 2
        half = (self.layout["item_hi"] - self.layout["item_lo"]) / 2
        Rc = self.rot("container")
        goal = self.pos("container") + Rc @ np.array([ic[0], ic[1], g["floor_z"] + half[2] + 0.001])
        adr = self.free["item"].qposadr[0]
        d.qpos[adr + 3:adr + 7] = quat_from_mat(Rc)
        d.qpos[adr:adr + 3] = goal - Rc @ centre
        d.qvel[self.free["item"].dofadr[0]:self.free["item"].dofadr[0] + 6] = 0.0
        mujoco.mj_forward(m, d)
        obs = None
        for _ in range(steps):
            obs, _, _ = self.step([0, 0, 0, 0, 0, 0, -1, 0])
        return obs

    def _index(self):
        m = self.model
        self.jq = np.array([m.joint(f"r_joint{i}").qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint(f"r_joint{i}").dofadr[0] for i in range(1, 8)])
        self.act = np.array([m.actuator(f"r_torq_j{i}").id for i in range(1, 8)])
        self.gact = [m.actuator("r_g_gripper_finger_joint1").id, m.actuator("r_g_gripper_finger_joint2").id]
        self.gq = [m.joint("r_g_finger_joint1").qposadr[0], m.joint("r_g_finger_joint2").qposadr[0]]
        self.site = m.site("r_g_grip_site").id
        self.ft_force = m.sensor("r_g_force_ee").adr[0]
        self.ft_torque = m.sensor("r_g_torque_ee").adr[0]
        self.ft_site = m.site("r_g_ft_frame").id
        self.wrist_cam = m.camera("r_eye_in_hand").id
        self.body = {"container": m.body("c_object").id, "item": m.body("i_object").id}
        self.free = {k: m.joint(p + "free") for k, p in (("container", "c_"), ("item", "i_"))}
        gripper = m.body("r_g_right_gripper").id
        self.gripper_geoms = {g for g in range(m.ngeom) if self._under(m, m.geom_bodyid[g], gripper)}
        self.pad_geoms = (m.geom("r_g_finger1_pad_collision").id, m.geom("r_g_finger2_pad_collision").id)
        self.table_geom = m.geom("table").id
        self.container_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body["container"]}
        self.robot_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("r_")}
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))

    @staticmethod
    def _under(m, body, root):
        while body > 0:
            if body == root:
                return True
            body = m.body_parentid[body]
        return False

    def _init_state(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.jq] = HOME_Q
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        self.grip_target = 0.04
        self.q_null = HOME_Q.copy()
        for _ in range(250):                      # settle the arm and let the objects come to rest
            d.ctrl[self.act] = 200.0 * (HOME_Q - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.gact] = [0.04, -0.04]
            mujoco.mj_step(m, d)
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()

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

    def step(self, action):
        a = np.asarray(action, float)
        d = self.data
        dp, dr = a[:3], a[3:6]
        n = np.linalg.norm(dp)
        if n > MAX_DPOS:
            dp = dp * (MAX_DPOS / n)
        n = np.linalg.norm(dr)
        if n > MAX_DROT:
            dr = dr * (MAX_DROT / n)
        self.target_pos = d.site_xpos[self.site] + dp
        self.target_R = axis_angle_mat(dr) @ d.site_xmat[self.site].reshape(3, 3)
        self.grip_target = 0.04 * (1.0 - float(np.clip(a[6], -1, 1))) / 2.0
        cams = int(a[7]) if len(a) > 7 else 0
        self.render_next = tuple(n for i, n in enumerate(CAMERA_NAMES) if cams & (1 << i))
        for _ in range(SUBSTEPS):
            d.ctrl[self.act] = self._osc_torque()
            d.ctrl[self.gact] = [self.grip_target, -self.grip_target]
            mujoco.mj_step(self.model, d)
            self.monitor.substep()
        self.t += 1
        self.monitor.control_step()
        return self.observation(), False, {}

    def unstable(self):
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return True
        if np.max(np.abs(d.qvel[self.jv])) > MAX_ARM_SPEED:
            return True
        return any(np.linalg.norm(self.lin_vel(k)) > MAX_OBJECT_SPEED for k in self.body)

    # --- state the grader (and the privileged reference) reads ----------------------------------------
    def pos(self, k):
        return self.data.xpos[self.body[k]].copy()

    def rot(self, k):
        return self.data.xmat[self.body[k]].reshape(3, 3).copy()

    def lin_vel(self, k):
        return self.data.qvel[self.free[k].dofadr[0]:self.free[k].dofadr[0] + 3].copy()

    def ang_vel(self, k):
        return self.data.qvel[self.free[k].dofadr[0] + 3:self.free[k].dofadr[0] + 6].copy()

    def contacts(self, k):
        """Set of geoms touching body k (other bodies' geoms)."""
        m, d = self.model, self.data
        b = self.body[k]
        out = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == b:
                out.add(int(c.geom2))
            elif b2 == b:
                out.add(int(c.geom1))
        return out

    def grasped(self, k):
        """Both finger pads touch body k (RoboCasa's _check_grasp with the pad geoms)."""
        touching = self.contacts(k)
        return all(p in touching for p in self.pad_geoms)

    def on_table(self, k):
        return self.table_geom in self.contacts(k)

    def item_in_container(self):
        """The centre of the item's collision box, in the container frame, inside the open interior (xy,
        with INTERIOR_MARGIN) and between the floor and the rim (each with CENTRE_Z_SLACK), and the item
        touching nothing but the container and the gripper. Box corners would be too strict: a rounded
        item leaning on a tapered wall has corners outside the walls while it is plainly inside."""
        lay = self.layout
        centre = (lay["item_lo"] + lay["item_hi"]) / 2
        world = self.pos("item") + self.rot("item") @ centre
        local = (world - self.pos("container")) @ self.rot("container")
        g = lay["container"]
        if not (np.all(local[:2] >= g["interior_lo"] - INTERIOR_MARGIN)
                and np.all(local[:2] <= g["interior_hi"] + INTERIOR_MARGIN)
                and g["floor_z"] - CENTRE_Z_SLACK <= local[2] <= g["rim_z"] + CENTRE_Z_SLACK):
            return False
        allowed = self.container_geoms | self.gripper_geoms
        return all(gid in allowed for gid in self.contacts("item"))

    def tilt(self, k):
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.rot(k)[2, 2])))))

    def truth(self):
        """Everything the privileged reference is given (token runs only, in a task)."""
        lay = self.layout
        g = lay["container"]
        return {"container_pos": self.pos("container").tolist(), "container_R": self.rot("container").tolist(),
                "item_pos": self.pos("item").tolist(), "item_R": self.rot("item").tolist(),
                "target": list(lay["poses"]["target"]),
                "container_lo": g["lo"].tolist(), "container_hi": g["hi"].tolist(),
                "interior_lo": g["interior_lo"].tolist(), "interior_hi": g["interior_hi"].tolist(),
                "floor_z": g["floor_z"], "rim_z": g["rim_z"],
                "item_lo": lay["item_lo"].tolist(), "item_hi": lay["item_hi"].tolist(),
                "item_mass": self.spec["item_mass"], "container_mass": self.spec["container_mass"],
                "container_com": self.data.xipos[self.body["container"]].tolist(),
                "item_com": self.data.xipos[self.body["item"]].tolist(),
                "item_com_offset": self.spec.get("item_com_offset", 0.0), "friction": self.spec.get("friction"),
                "rolling": self.spec.get("rolling", 0.0), "restitution": self.spec.get("restitution"),
                "grasped_item": self.grasped("item"), "grasped_container": self.grasped("container"),
                "item_on_table": self.on_table("item"), "item_in_container": self.item_in_container(),
                "item_speed": float(np.linalg.norm(self.lin_vel("item"))),
                "loaded": bool(self.monitor.loaded), "rest_z": float(self.monitor.rest_z)}

    def progress(self):
        return self.monitor.metrics()

    # --- observation ---------------------------------------------------------------------------------
    def wrench_world(self):
        d = self.data
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        return np.concatenate([R @ d.sensordata[self.ft_force:self.ft_force + 3],
                               R @ d.sensordata[self.ft_torque:self.ft_torque + 3]])

    def render(self, cam):
        m, d = self.model, self.data
        if cam == "wrist":
            w, h, cid = WRIST["width"], WRIST["height"], self.wrist_cam
        else:
            _p, _t, _f, w, h = CAMERAS[cam]
            cid = m.camera(cam).id
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(m, h, w)
        r = self._renderers[key]
        r.update_scene(d, camera=cid)
        img = r.render().astype(np.float32)
        img += self.rng.normal(0, RGB_NOISE, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def camera_matrix(self, cam):
        """3x3 pinhole intrinsics of `cam` (left, right, wrist) at its image size; no distortion."""
        if cam == "wrist":
            w, h, cid = WRIST["width"], WRIST["height"], self.wrist_cam
        else:
            _p, _t, _f, w, h = CAMERAS[cam]
            cid = self.model.camera(cam).id
        f = (h / 2.0) / math.tan(math.radians(float(self.model.cam_fovy[cid])) / 2.0)
        return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])

    def fixed_pose(self, cam):
        """4x4 world pose of a fixed camera, OpenCV convention (x right, y down, z forward)."""
        d, cid = self.data, self.model.camera(cam).id
        T = np.eye(4)
        T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = d.cam_xpos[cid]
        return T

    def wrist_pose(self, d=None):
        """4x4 world pose of the wrist camera, OpenCV convention."""
        d = self.data if d is None else d
        T = np.eye(4)
        T[:3, :3] = d.cam_xmat[self.wrist_cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = d.cam_xpos[self.wrist_cam]
        return T

    def perturbed_observation(self, obs, rng):
        """A probe copy for the grader's lock (box_env's rule): joint readings offset by up to 0.001 rad,
        the gripper pose and wrist-camera pose recomputed consistently, and a small wrench offset."""
        f = mujoco.MjData(self.model)
        f.qpos[:] = self.data.qpos
        q = np.asarray(obs["q"], float) + rng.uniform(-0.001, 0.001, 7)
        f.qpos[self.jq] = q
        mujoco.mj_kinematics(self.model, f)
        mujoco.mj_camlight(self.model, f)
        out = dict(obs)
        out["q"] = np.round(q, 6)
        out["ee_pos"] = np.round(f.site_xpos[self.site], 6)
        out["ee_quat"] = np.round(quat_from_mat(f.site_xmat[self.site].reshape(3, 3)), 6)
        out["T_world_wrist"] = np.round(self.wrist_pose(f), 6)
        w = np.asarray(obs["wrench"], float).copy()
        w[:3] += rng.uniform(-0.2, 0.2, 3)
        out["wrench"] = np.round(w, 4)
        return out

    def observation(self):
        d = self.data
        R = d.site_xmat[self.site].reshape(3, 3)
        obs = {
            "t": self.t,
            "target": np.asarray(self.layout["poses"]["target"], float),
            "T_world_wrist": self.wrist_pose(),
            "T_world_left": self.fixed_pose("left"),
            "T_world_right": self.fixed_pose("right"),
            "K_left": self.camera_matrix("left"),
            "K_right": self.camera_matrix("right"),
            "K_wrist": self.camera_matrix("wrist"),
            "q": d.qpos[self.jq].copy(),
            "dq": d.qvel[self.jv].copy(),
            "ee_pos": d.site_xpos[self.site].copy(),
            "ee_quat": quat_from_mat(R),
            "gripper": float(d.qpos[self.gq[0]] - d.qpos[self.gq[1]]),
            "wrench": self.wrench_world()
            + self.rng.normal(0, 1, 6) * np.array([WRENCH_NOISE] * 3 + [WRENCH_NOISE / 10] * 3),
        }
        for cam in CAMERA_NAMES:            # an image the previous action did not ask for is empty
            ask = self.render_enabled and cam in self.render_next
            obs[cam] = self.render(cam) if ask else np.zeros((0, 0, 3), np.uint8)
        return obs

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}


class Monitor:
    """The grader-side predicates, sampled at every physics substep (containment) and every control step
    (stages). Stages are latched in order: loaded -> lifted -> carried -> delivered. A spill (the item
    leaving the container's interior at any substep after it was loaded) is latched too."""

    def __init__(self, env):
        self.env = env
        self.rest_z = env.pos("container")[2]
        self.start_xy = env.pos("container")[:2]
        self.loaded_count = 0
        self.loaded = False
        self.lifted = False
        self.carried = False
        self.delivered = False
        self.spills = 0
        self.first_spill_t = None
        self.stable = 0
        self.max_lift = 0.0
        self.max_carry = 0.0
        self.max_tilt = 0.0
        self.substeps = 0
        self.final = {}

    def substep(self):
        self.substeps += 1
        if self.loaded and not self.env.item_in_container():
            self.spills += 1
            if self.first_spill_t is None:
                self.first_spill_t = self.env.t

    def control_step(self):
        e = self.env
        inside = e.item_in_container()
        if not self.loaded:
            resting = (inside and not e.grasped("item") and not e.on_table("item")
                       and np.linalg.norm(e.lin_vel("item")) < REST_SPEED)
            self.loaded_count = self.loaded_count + 1 if resting else 0
            if self.loaded_count >= LOADED_STEPS:
                self.loaded = True
                self.rest_z = e.pos("container")[2]
                self.start_xy = e.pos("container")[:2]
            return
        p = e.pos("container")
        held = e.grasped("container")
        airborne = not e.on_table("container")
        lift = float(p[2] - self.rest_z)
        self.max_tilt = max(self.max_tilt, e.tilt("container"))
        clean = self.spills == 0
        if held and airborne and clean:
            self.max_lift = max(self.max_lift, lift)
            if lift >= MIN_LIFT:
                self.lifted = True
        if self.lifted and held and airborne and clean:
            dist = float(np.linalg.norm(p[:2] - self.start_xy))
            self.max_carry = max(self.max_carry, dist)
            if dist >= MIN_CARRY:
                self.carried = True
        self.final = {
            "carried_clean": bool(self.carried and clean), "inside": inside,
            "at_target": bool(np.linalg.norm(p[:2] - np.asarray(e.layout["poses"]["target"])) < DELIVERY_TOL),
            "rest_height": bool(abs(p[2] - self.rest_z) < REST_HEIGHT_TOL),
            "on_table": e.on_table("container"), "item_off_table": not e.on_table("item"),
            "released": bool(not held and not e.grasped("item")),
            "gripper_clear": bool(np.linalg.norm(e.data.site_xpos[e.site] - p) > GRIPPER_CLEAR),
            "still": bool(np.linalg.norm(e.lin_vel("container")) < REST_SPEED
                          and np.linalg.norm(e.lin_vel("item")) < REST_SPEED
                          and np.linalg.norm(e.ang_vel("container")) < REST_ANG_SPEED)}
        ok = all(self.final.values())
        self.stable = self.stable + 1 if ok else 0
        if self.stable >= STABLE_STEPS:
            self.delivered = True

    def metrics(self):
        e = self.env
        p = e.pos("container")
        return {"loaded": self.loaded, "lifted": self.lifted, "carried": self.carried,
                "delivered": self.delivered, "spills": self.spills, "first_spill_t": self.first_spill_t,
                "max_lift_m": round(self.max_lift, 4), "max_carry_m": round(self.max_carry, 4),
                "max_tilt_deg": round(self.max_tilt, 2), "stable_steps": self.stable,
                "delivery_error_m": round(float(np.linalg.norm(p[:2] - np.asarray(e.layout["poses"]["target"]))), 4),
                "item_in_container": e.item_in_container(), "substeps": self.substeps,
                "final_failed": sorted(k for k, v in self.final.items() if not v)}
