"""Native free-base Go1 and a separate camera-only aerial survey."""
import base64
import io
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image

import hall_render
from scene_assets import asset_root, geom, materials, robot_meshes, vec
from props import ENCODER_BIAS_RAD, ENCODER_NOISE_RAD, GRAVITY_NOISE, GYRO_BIAS_RAD_S, GYRO_NOISE_RAD_S
from mobility import BEVEL, CAGE_TOP, DUCT, PROTECTED, SUPPORT, TERRAIN_BIT, add_fittings, add_load


LEGS = ("FR", "FL", "RR", "RL")
JOINTS = tuple(f"{leg}_{part}" for leg in LEGS for part in ("hip", "thigh", "calf"))
WALL_Z, CAMERA_Z, FOV, PIXELS = .32, 1.65, 78, 256
MAX_SCOUT, MAX_TICKS, MAX_HOLD = 80, 48000, 40
DOCK_RADIUS = .17
POST_Z, POST_R = .72, .105
MAST_Z, MAST_FOV, MAST_PITCH, MAX_LOOKS = .20, 70, math.radians(36), 24
LAUNCH_BIT = 1 << 33
ROUTE_BITS = 3 << 14
# What changes between the survey and the deployment, by seed bits alone:
PALLET_BIT = 1 << 30        # which opening of the middle rack the pallet closes at survey time
MOVED_BIT = 1 << 27         # the pallet was moved to the other opening afterwards
DOCK_BIT = 1 << 43          # the dock (with its beacon) was moved afterwards
POSTS_BIT = 1 << 37         # one bollard was taken away or one was added afterwards
CHANGE_BITS = PALLET_BIT | MOVED_BIT | DOCK_BIT | POSTS_BIT
# The barrier: a 1.2 x 0.8 x 0.144 m pallet stood on its short edge inside the opening it closes, held by a foot
# bracket near each end: a floor plate under the pallet and an upright on each of its faces.
PALLET_Y, PALLET_Z, PALLET_X = .80, 1.20, .144
PALLET_FEET = tuple((f"pallet_foot{i}", (0., s*.30, .004), (.11, .05, .004)) for i, s in enumerate((-1, 1)))+tuple(
    (f"pallet_upright{i}{j}", (sx*.075, s*.30, .04), (.003, .05, .04))
    for i, s in enumerate((-1, 1)) for j, sx in enumerate((-1, 1)))
# The dock mat (4 mm) and the strip at its east edge (to 6 mm): walked surfaces.
DOCK_SURFACES = ("dock", "dock_nose")
BEACON = (.42, .50)         # the beacon's offset from the dock's centre


@dataclass(frozen=True)
class Rack:
    near: float
    far: float
    wide: tuple
    shortcut: tuple | None = None
    other: tuple | None = None      # the middle rack's second opening (closed by the pallet)

    def choice(self, y):
        if self.wide[0] <= y <= self.wide[1]:
            return "wide"
        if self.shortcut and self.shortcut[0] <= y <= self.shortcut[1]:
            return "shortcut"
        return "outside"


def layout(seed):
    rng = np.random.default_rng(seed & ~(LAUNCH_BIT | ROUTE_BITS | TERRAIN_BIT | CHANGE_BITS))
    # Independent uniform seed bits; the fixed evaluation seeds cover each pair twice.
    sides = tuple(1 if (seed >> bit) & 1 else -1 for bit in (14, 15))
    bound = float(rng.uniform(1.86, 1.98))
    goal = np.array([rng.uniform(6.25, 6.43), rng.uniform(-.20, .20)])
    walls = {
        "back": (-1.14, -1.00, -bound-.14, bound+.14),
        "front": (7.06, 7.20, -bound-.14, bound+.14),
        "south": (-1.00, 7.06, -bound-.14, -bound),
        "north": (-1.00, 7.06, bound, bound+.14),
    }
    racks = []
    for i, nominal in enumerate((1.39, 3.13, 4.87)):
        x = nominal + float(rng.uniform(-.09, .09))
        depth = float(rng.uniform(.23, .32))
        center = (float(rng.uniform(-.12, .12)) if i == 1 else
                  float(sides[i//2]*rng.uniform(1.00, 1.14)))
        half = float(rng.uniform(.47, .55))
        wide = (center-half, center+half)
        shortcut = None
        if i != 1:
            center = float(rng.uniform(-.15, .15))
            half = float(rng.uniform(.080, .108))
            shortcut = (center-half, center+half)
        rack = Rack(x-depth/2, x+depth/2, wide, shortcut)
        gaps = sorted((wide, shortcut)) if shortcut else [wide]
        if i == 1:
            # Two openings; a pallet on edge stands in one of them (which one can change after the survey).
            south = (-float(rng.uniform(.66, .80)), float(rng.uniform(.42, .46)))
            north = (float(rng.uniform(.66, .80)), float(rng.uniform(.42, .46)))
            south, north = ((c-h, c+h) for c, h in (south, north))
            rack = Rack(x-depth/2, x+depth/2, north, None, south)
            gaps = [south, north]
        edges = [-bound, *(edge for gap in gaps for edge in gap), bound]
        for j, (lo, hi) in enumerate(zip(edges[::2], edges[1::2])):
            walls[f"rack{i+1}_{j}"] = (rack.near, rack.far, lo, hi)
        racks.append(rack)
    return rng, racks, goal, walls


def fixtures_and_launch(seed, walls):
    # Independent streams: reflecting the launch changes neither rack, fixtures nor servo gains.
    site = seed & ~(LAUNCH_BIT | ROUTE_BITS | TERRAIN_BIT | CHANGE_BITS)
    fixtures = np.random.default_rng([site, 3701])
    start = np.random.default_rng([site, 3702])
    hand = 1 if seed & LAUNCH_BIT else -1
    bank = fixtures.choice((-1, 1))
    rail = sum(walls["north"][2:])/2
    back = sum(walls["back"][:2])/2
    posts = [(back, bank*fixtures.uniform(-1.40, -.93)), (back, bank*fixtures.uniform(.22, .74)),
             (fixtures.uniform(-.08, .33), bank*rail), (fixtures.uniform(1.45, 1.95), bank*rail),
             (fixtures.uniform(.50, 1.00), -bank*rail)]
    launch = [start.uniform(-.06, .10)+hand*.12, hand*start.uniform(.54, .85),
              -hand*math.radians(start.uniform(42., 70.))]
    # Bollards along both rails of the hall: what the Go1 re-localises on along its way.
    for nominal in (2.30, 3.95, 5.60):
        for side in (-1, 1):
            posts.append((nominal+fixtures.uniform(-.30, .30), side*rail))
    return posts, launch


def site_states(seed, racks, goal, posts, walls):
    """The site as surveyed and as the Go1 finds it: (racks, goal, posts, pallet rectangle) twice.

    Between the two, by seed bits and an independent stream: the pallet may have moved to the middle
    rack's other opening, the dock (with its beacon) may have been moved 0.10-0.25 m west and as far north or south, and
    one bollard may have been taken away or one added.
    """
    site = seed & ~(LAUNCH_BIT | ROUTE_BITS | TERRAIN_BIT | CHANGE_BITS)
    rng = np.random.default_rng([site, 3706])
    middle = racks[1]
    openings = {"north": middle.wide, "south": middle.other}
    closed = "south" if seed & PALLET_BIT else "north"
    shift = np.array([-rng.uniform(.10, .25), rng.choice((-1, 1))*rng.uniform(.10, .25)])     # west; north or south
    take = int(rng.integers(len(posts)))
    add = bool(rng.integers(2))
    rail = sum(walls["north"][2:])/2
    extra = None
    while extra is None or min(math.dist(extra, post) for post in posts) < .60:
        extra = (float(rng.uniform(.6, 6.0)), float(rng.choice((-1, 1))*rail))

    def state(closed_now, goal_now, posts_now):
        other = "north" if closed_now == "south" else "south"
        rack = Rack(middle.near, middle.far, openings[other], None, openings[closed_now])
        lo, hi = openings[closed_now]
        x, y = (middle.near+middle.far)/2, (lo+hi)/2
        pallet = (x-PALLET_X/2, x+PALLET_X/2, y-PALLET_Y/2, y+PALLET_Y/2)
        return [racks[0], rack, racks[2]], goal_now, posts_now, pallet

    survey = state(closed, goal, list(posts))
    closed_then = ("north" if closed == "south" else "south") if seed & MOVED_BIT else closed
    posts_then = list(posts)
    if seed & POSTS_BIT:
        if add:
            posts_then.append(extra)
        else:
            posts_then[take] = None       # keeps the others' names; a missing post is not built
    mission = state(closed_then, goal+shift if seed & DOCK_BIT else goal, posts_then)
    return survey, mission


def rotation2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def scene(root, walls, goal, posts, fittings, pallet=None, beacon=None):
    world = root.find("worldbody")
    materials(root)
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 5 .1",
                  material="floor_mat", friction="1 .02 .001")
    ET.SubElement(world, "light", name="worklight", directional="true",
                  pos="0 0 5", dir="0 0 -1", diffuse=".60 .60 .60", castshadow="false")
    for name, (x0, x1, y0, y1) in walls.items():
        size = ((x1-x0)/2, (y1-y0)/2, WALL_Z/2)
        pos = ((x0+x1)/2, (y0+y1)/2, WALL_Z/2)
        geom(world, name, pos=pos, size=size, material="rust_mat", contype=1, conaffinity=1)
        # Painted caps mark the exact footprints; no geometry or depth is sent separately.
        geom(world, name+"_paint", pos=(*pos[:2], WALL_Z-.001),
             size=(*size[:2], .0015), rgba=(.90, .18, .045, 1))
    if goal is not None:        # the commissioning pad has no dock
        geom(world, "dock", pos=(*goal, .002), size=(.29, .28, .002), rgba=(.035, .28, .89, 1),
             contype=1, conaffinity=1, friction="1 .02 .001")
        geom(world, "dock_nose", pos=(goal[0]+.261, goal[1], .005), size=(.026, .26, .001),
             rgba=(.88, .81, .09, 1), contype=1, conaffinity=1, friction="1 .02 .001")
        beacon = (goal[0]+BEACON[0], goal[1]+BEACON[1])
    if beacon is not None:
        # The dock's beacon: a blue cap at the bollards' height, at a fixed offset from the dock's centre.
        bx, by = beacon
        geom(world, "dock_beacon_stem", "cylinder", (bx, by, POST_Z/2), (.06, POST_Z/2),
             rgba=(.46, .47, .45, 1), contype=1, conaffinity=1)
        geom(world, "dock_beacon_cap", "sphere", (bx, by, POST_Z), (POST_R,),
             rgba=(.06, .25, .92, 1), contype=1, conaffinity=1)
    if pallet is not None:
        x0, x1, y0, y1 = pallet
        geom(world, "pallet_barrier", pos=((x0+x1)/2, (y0+y1)/2, PALLET_Z/2),
             size=((x1-x0)/2, (y1-y0)/2, PALLET_Z/2), rgba=(.80, .10, .62, 1), contype=1, conaffinity=1)
        for name, offset, size in PALLET_FEET:
            geom(world, name, pos=((x0+x1)/2+offset[0], (y0+y1)/2+offset[1], offset[2]), size=size,
                 rgba=(.25, .26, .27, 1), contype=1, conaffinity=1)
    add_fittings(root, world, fittings)
    for i, post in enumerate(posts):
        if post is None:
            continue
        x, y = post
        geom(world, f"bollard{i}_stem", "cylinder", (x, y, POST_Z/2), (.06, POST_Z/2),
             rgba=(.46, .47, .45, 1), contype=1, conaffinity=1)
        geom(world, f"bollard{i}_cap", "sphere", (x, y, POST_Z), (POST_R,),
             rgba=(.19, .84, .065, 1), contype=1, conaffinity=1)
    return world


def look_scene(walls, goal, posts, fittings, pallet=None, beacon=None, station_y=None):
    """What hall_render draws for one state of a site (the numbers the collision scene is built from)."""
    north = walls.get("north")
    if goal is not None:
        beacon = (goal[0]+BEACON[0], goal[1]+BEACON[1])
    return {
        "walls": {name: [float(v) for v in box] for name, box in walls.items()},
        "fittings": [[f.kind, f.x, f.lo, f.hi, f.length, f.clearance, f.rise] for f in fittings],
        "posts": [[float(v) for v in post] for post in posts if post is not None],
        "goal": None if goal is None else [float(v) for v in goal],
        "beacon": None if beacon is None else [float(v) for v in beacon],
        "pallet": None if pallet is None else [float(v) for v in pallet],
        "bound": float(north[2]) if north else 0.,
        "lane_y": float(north[3]) if north else None,
        "station_y": None if station_y is None else float(station_y),
    }


class Views:
    """The cameras' pictures of one site: a renderer process, started at the first picture and ended with the site."""

    def __init__(self, scenes):
        self.scenes, self.cameras = scenes, None

    def image(self, scene, **camera):
        if self.cameras is None:
            self.cameras = hall_render.Cameras(self.scenes)
        return self.cameras.image(scene, **camera)

    def close(self):
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None


def image64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Survey:
    def __init__(self, walls, goal, posts, fittings, pallet=None, scale=1., views=None):
        self.views = views
        root = ET.fromstring('<mujoco><asset/><worldbody/></mujoco>')
        world = scene(root, walls, goal, posts, fittings, pallet)
        self.scale = scale
        uav = ET.SubElement(world, "body", name="uav", mocap="true",
                            pos=vec((0, 0, CAMERA_Z+.12)))
        robot_meshes(root, uav, "drone")
        ET.SubElement(uav, "camera", name="down", pos="0 0 -.12", xyaxes="1 0 0 0 1 0",
                      fovy=str(FOV))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.cached = None
        self.xy = np.zeros(2)

    def move(self, delta):
        self.xy = np.clip(self.xy+delta, [-.75, -1.55], [6.80, 1.55])
        self.cached = None

    def observe(self):
        if self.cached is None:
            # The camera hangs 0.12 m under the UAV and looks straight down; image up is north (+y).
            self.cached = image64(self.views.image("survey", eye=(self.xy[0], self.xy[1], CAMERA_Z), forward=(0., 0., -1.),
                                                   up=(0., 1., 0.), fovy=FOV, near=.05))
        # The UAV's own tracking over-reads or under-reads its displacement by the site's factor.
        return {"tracking_xy_m": (self.scale*self.xy).round(4).tolist(), "images": [self.cached]}

    def close(self):
        self.cached = None


class Go1:
    def __init__(self, rng, walls, goal, racks, posts, launch, fittings, props=None, budget=MAX_TICKS,
                 pallet=None, sense_seed=0, beacon=None, views=None, uav=None):
        spec = asset_root()/"unitree_go1/go1.xml"
        root = ET.parse(spec).getroot()
        root.find("compiler").set("meshdir", str(spec.parent/"assets"))
        scene(root, walls, goal, posts, fittings, pallet, beacon)
        has_dock = goal is not None or beacon is not None
        if goal is None:
            goal = np.array([1e3, 0.])      # no dock: the dock test can never hold
        trunk = root.find(".//body[@name='trunk']")
        add_load(trunk, props)
        for i in range(8):
            angle = i*math.pi/4
            # The mast looks up at fixtures. Roll/pitch follow the real trunk.
            image_up = (-math.sin(MAST_PITCH)*math.cos(angle), -math.sin(MAST_PITCH)*math.sin(angle),
                        math.cos(MAST_PITCH))
            ET.SubElement(trunk, "camera", name=f"mast{i}", pos=vec((0, 0, MAST_Z)),
                          xyaxes=vec((math.sin(angle), -math.cos(angle), 0, *image_up)), fovy=str(MAST_FOV))
        model = self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        if (tuple(model.actuator(i).name for i in range(model.nu)) != JOINTS
                or model.nq != 19 or model.nv != 18 or model.neq or model.nmocap):
            raise RuntimeError("expected twelve servos and an unstabilized free-base Go1")
        gain = np.asarray(props.servo_gain) if props else rng.uniform(.90, 1.10, size=model.nu)
        model.actuator_gainprm[:, 0] *= gain
        model.actuator_biasprm[:, 1] *= gain
        self.budget = int(budget)
        self.trunk = model.body("trunk").id
        self.floor = model.geom("floor").id
        obstacles = [*walls, *(f"bollard{i}_{part}" for i, post in enumerate(posts) if post is not None
                               for part in ("cap", "stem"))]
        if has_dock:
            obstacles.extend(("dock_beacon_stem", "dock_beacon_cap"))
        if pallet is not None:
            obstacles.extend(("pallet_barrier", *(name for name, _, _ in PALLET_FEET)))
        if any(f.kind == "duct" for f in fittings):
            obstacles.extend(DUCT)
        self.wall_ids = {model.geom(name).id: name for name in obstacles}
        self.support_ids = {self.floor: "floor"}
        if goal[0] < 1e2:
            self.support_ids.update({model.geom(name).id: name for name in DOCK_SURFACES})
        if any(f.kind == "deck" for f in fittings):
            self.support_ids.update({model.geom(name).id: name for name in SUPPORT})
        if props:
            model.actuator_forcerange[:] *= props.torque_scale
            # The walked surface sets the friction of a foot's contact (geom priority), not the foot.
            for gid in self.support_ids:
                model.geom_friction[gid, 0] = props.friction
                model.geom_priority[gid] = 1
        self.protected = {model.geom(name).id: name for name in PROTECTED}
        self.guard_ids = [model.geom(name).id for name in CAGE_TOP]
        self.fittings = fittings
        self.fitting_entered = set()
        self.fitting_cleared = set()
        self.deck_feet = set()
        self.min_duct_gap = None
        self.contact_part = None
        self.feet = [model.geom(leg).id for leg in LEGS]
        self.foot_modes = [set() for _ in LEGS]
        self.data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.data, model.key("home").id)
        self.data.qpos[:2] = launch[:2]
        yaw = launch[2]
        self.data.qpos[3:7] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
        mujoco.mj_forward(model, self.data)
        mujoco.mj_step(model, self.data, nstep=200)  # settle the native home pose, before mission time
        mujoco.mj_kinematics(model, self.data)
        self.origin = self.xy.copy()
        self.origin_yaw = self.heading
        self.world_to_odom = rotation2(-self.origin_yaw)
        # Sensing: leg odometry that drifts (a yaw-rate error and a distance factor, constant per site),
        # encoders and an IMU with noise and constant biases. The stream is the site's, not the layout's.
        self.sense = np.random.default_rng([int(sense_seed) & ((1 << 63)-1), 3705])
        self.encoder_bias = self.sense.uniform(-ENCODER_BIAS_RAD, ENCODER_BIAS_RAD, size=12)
        self.gyro_bias = self.sense.uniform(-GYRO_BIAS_RAD_S, GYRO_BIAS_RAD_S, size=3)
        self.odom_drift = math.radians(props.odom_yaw_drift_deg_s) if props else 0.
        self.odom_scale = props.odom_scale if props else 1.
        self.odom_xy = np.zeros(2)
        self.odom_yaw = 0.
        self.views, self.uav = views, uav
        self.mast_ids = [model.camera(f"mast{i}").id for i in range(8)]
        self.drawn = {name: model.body(name).id for name in hall_render.BODIES}
        self.looks = 0
        self.goal = goal
        self.ticks = self.dwell = self.tipped = 0
        self.actions = 0
        self.outcome = "running"
        self.max_x = float(self.xy[0])
        self.travel = 0.
        self.racks = racks
        self.choices = [None]*len(racks)
        self.crossed = [False]*len(racks)
        self.lateral_margin = [None]*len(racks)
        self.contact_wall = None
        self.dock_error = float(np.linalg.norm(self.xy-goal))
        self.dock_zone_entered = False
        self.contacts = self.ground_contacts()

    @property
    def xy(self):
        return self.data.xpos[self.trunk, :2]

    @property
    def rotation(self):
        return self.data.xmat[self.trunk].reshape(3, 3)

    @property
    def heading(self):
        return math.atan2(self.rotation[1, 0], self.rotation[0, 0])

    def ground_contacts(self):
        supported = set()
        for a, b in self.data.contact.geom:
            if b in self.support_ids or b in self.wall_ids:
                a, b = b, a
            if a in self.support_ids:
                supported.add(int(b))
                if self.support_ids[a] == "deck_table" and b in self.feet:
                    self.deck_feet.add(int(b))
                if b in self.protected:
                    self.contact_wall, self.contact_part = self.support_ids[a], self.protected[b]
            elif a in self.wall_ids:
                self.contact_wall = self.wall_ids[a]
                self.contact_part = self.model.geom(b).name or self.model.body(self.model.geom_bodyid[b]).name
        return supported

    def record_fittings(self):
        x, y = self.xy
        for f in self.fittings:
            half = f.length/2 + (BEVEL if f.kind == "deck" else 0.)
            if abs(x-f.x) < half and f.lo < y < f.hi:
                self.fitting_entered.add(f.kind)
                if f.kind == "duct" and abs(x-f.x) < f.length/2-.07:
                    # highest point of the guard frame's top bars (boxes) at the trunk's attitude
                    z = max(self.data.geom_xpos[it, 2]+abs(self.data.geom_xmat[it, 6:]) @ self.model.geom_size[it]
                            for it in self.guard_ids)
                    gap = float(f.clearance-z)
                    self.min_duct_gap = gap if self.min_duct_gap is None else min(gap, self.min_duct_gap)
            if f.kind in self.fitting_entered and x > f.x+half+.30:
                self.fitting_cleared.add(f.kind)

    def observe(self):
        noise = self.sense.normal
        gravity = -self.rotation[2]+noise(0., GRAVITY_NOISE, size=3)
        return {
            "odom_xy_m": self.odom_xy.round(3).tolist(),
            "odom_yaw_rad": round(math.atan2(math.sin(self.odom_yaw), math.cos(self.odom_yaw)), 4),
            "gravity_body_unit": (gravity/np.linalg.norm(gravity)).round(5).tolist(),
            "gyro_body_rad_s": (self.data.qvel[3:6]+self.gyro_bias+noise(0., GYRO_NOISE_RAD_S, size=3)).round(4).tolist(),
            "joint_rad": (self.data.qpos[7:]+self.encoder_bias+noise(0., ENCODER_NOISE_RAD, size=12)).round(4).tolist(),
            "joint_rad_s": self.data.qvel[6:].round(4).tolist(),
            "foot_contact": [foot in self.contacts for foot in self.feet],
            "ticks_remaining": self.budget-self.ticks,
        }

    def look(self, detent):
        if type(detent) is not int or not 0 <= detent < 8:
            raise ValueError("camera detent must be an integer from 0 to 7")
        if self.looks >= MAX_LOOKS:
            raise ValueError("ground camera exposure budget spent")
        image = self.picture(detent)
        self.looks += 1
        return {"images": [image64(image)]}

    def picture(self, detent):
        """The mast camera's image at this state (uint8, 256 x 256 x 3); look() is the budgeted call."""
        mujoco.mj_camlight(self.model, self.data)       # the cameras' poses at this state
        cam = self.mast_ids[detent]
        axes = self.data.cam_xmat[cam].reshape(3, 3)
        bodies = {name: (self.data.xpos[b].copy(), self.data.xmat[b].copy()) for name, b in self.drawn.items()}
        return self.views.image("found", eye=self.data.cam_xpos[cam], forward=-axes[:, 2], up=axes[:, 1], fovy=MAST_FOV,
                                near=.02, bodies=bodies, uav=self.uav)

    def close(self):
        pass

    def record_passages(self):
        x, y = self.xy
        for i, rack in enumerate(self.racks):
            if self.choices[i] is None and x >= rack.near-.34:
                self.choices[i] = rack.choice(y)
            if rack.near-.32 <= x <= rack.far+.32:
                # Lateral trunk-center slack, not a claim about the leg envelope.
                lo, hi = rack.wide if self.choices[i] != "shortcut" else rack.shortcut
                margin = float(min(y-lo, hi-y))
                old = self.lateral_margin[i]
                self.lateral_margin[i] = margin if old is None else min(margin, old)
            if x >= rack.far+.34:
                self.crossed[i] = True

    def act(self, action):
        if not isinstance(action, dict) or set(action) != {"target_rad", "ticks"}:
            raise ValueError("expected target_rad (12 angles) and ticks")
        raw, ticks = action["target_rad"], action["ticks"]
        if type(ticks) is not int or not 1 <= ticks <= min(MAX_HOLD, self.budget-self.ticks):
            raise ValueError("invalid tick count")
        if not isinstance(raw, list) or len(raw) != 12 or any(type(v) not in (int, float) for v in raw):
            raise ValueError("expected twelve numeric angles")
        target = np.asarray(raw, dtype=float)
        lo, hi = self.model.actuator_ctrlrange.T
        if not np.isfinite(target).all() or (target < lo).any() or (target > hi).any():
            raise ValueError("target outside native joint limits")
        self.actions += 1
        self.data.ctrl[:] = target
        for _ in range(ticks):
            last, last_heading = self.xy.copy(), self.heading
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_kinematics(self.model, self.data)
            self.ticks += 1
            self.travel += float(np.linalg.norm(self.xy-last))
            # Leg odometry: the body-frame step, scaled, laid along a heading that drifts.
            turn = self.heading-last_heading
            self.odom_yaw += math.atan2(math.sin(turn), math.cos(turn))+self.odom_drift*self.model.opt.timestep
            step = rotation2(-last_heading) @ (self.xy-last)
            self.odom_xy += self.odom_scale*(rotation2(self.odom_yaw) @ step)
            self.max_x = max(self.max_x, float(self.xy[0]))
            self.record_passages()
            self.record_fittings()
            self.contacts = self.ground_contacts()
            for modes, foot in zip(self.foot_modes, self.feet):
                modes.add(foot in self.contacts)
            upright = self.rotation[2, 2] > .80 and self.data.xpos[self.trunk, 2] > .16
            quiet = np.linalg.norm(self.data.qvel[:3]) < .09 and np.linalg.norm(self.data.qvel[3:6]) < .35
            self.dock_error = float(np.linalg.norm(self.xy-self.goal))
            self.dock_zone_entered |= self.dock_error <= DOCK_RADIUS
            parked = self.dock_error <= DOCK_RADIUS and abs(self.heading) <= math.radians(16)
            self.dwell = self.dwell+1 if upright and quiet and parked else 0
            self.tipped = 0 if upright else self.tipped+1
            dt = self.model.opt.timestep
            if self.contact_wall:
                self.outcome = "contact"
            elif self.tipped*dt >= .20:
                self.outcome = "fell"
            elif self.dwell*dt >= .45:
                self.outcome = "docked"
            elif self.ticks >= self.budget:
                self.outcome = "budget"
            if self.outcome != "running":
                break
