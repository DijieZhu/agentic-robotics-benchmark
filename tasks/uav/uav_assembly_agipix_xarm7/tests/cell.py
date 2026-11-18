"""Grader-owned xArm/AGiPix physics. No object pose writes after initialization.

Small screws are represented by at-rest couplings. A locked part keeps the exact
relative pose at fastening: capture never teleports it to a nominal target.
"""
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

DT = .01
HOME_Q = np.array([0, -.247, 0, .909, 0, 1.15644, 0])


def text(v):
    return " ".join(f"{float(x):.10g}" for x in np.asarray(v).ravel())


def quat(R):
    q = Rotation.from_matrix(R).as_quat()
    return np.roll(q, 1)


def transform(pos, rot=None):
    T = np.eye(4)
    T[:3, 3] = pos
    if rot is not None:
        T[:3, :3] = rot
    return T


def body_pose(d, bid):
    return transform(d.xpos[bid], d.xmat[bid].reshape(3, 3))


def orientation_error(rot, symmetry):
    """Smallest mating error for the documented physical axial symmetry."""
    if symmetry == 0:
        return math.acos(float(np.clip(rot[2, 2], -1, 1)))
    return min(Rotation.from_matrix(
        Rotation.from_euler("z", -2*math.pi*k/symmetry).as_matrix() @ rot
    ).magnitude() for k in range(symmetry))


def fixture_targets(graph, kit):
    root = transform(graph["fixture_pos_m"])
    poses = {"base": root}
    for name, p in graph["instances"].items():
        poses[name] = root @ transform(p["target_pos_m"], Rotation.from_euler(
            "xyz", p["target_euler_xyz_deg"], degrees=True).as_matrix())
        if p["kind"] == "motor":
            rotor = kit["parts"]["motor"]["rotor"]
            poses[name + "/rotor"] = poses[name] @ transform(rotor["hinge_pos_m"])
    return poses


# Staging stands: bench equipment, not kit parts. Each thin part (no captive
# feet) lies on its own small printed stand, as before 26 mm above the bench so
# the fingers reach under it: a base plate on the bench with posts and ribs
# whose flat tops meet the part's collision material at its staged pose. Part
# frame (m); every stand lies inside its part's bench footprint and keeps clear
# of the finger paths of the part's bench grasps. Static world geoms, placed
# with the part's staged pose (incl. its yaw).
#   plate: ("disc", x, y, radius) or ("box", x, y, half x, half y, corner radius)
#   posts: (x, y, radius); ribs: (x, y, half x, half y)
#   collar: width of the printed fillet ring at the foot of each post (2 mm high)
STAND_RGBA = ".86 .87 .84 1"
STAND_PLATE_M = .003
STAND_COLLAR_H_M = .002
STANDS = {
    # Pedestal under the flat hub ring (bore stays open); blades free.
    "prop_cw": dict(plate=("disc", 0., 0., .024), posts=((0., 0., .006),), ribs=(), collar=.002),
    # C plate: posts under the spine's top and bottom, and a narrow rib under
    # each arm, between the jaws of a rim pinch across that arm.
    "sensor_cheek": dict(plate=("box", -.0055, .001, .0205, .0395, .005),
                         posts=((.010, .0355, .0035), (.00975, -.0315, .0035)),
                         ribs=((-.015, .03725, .007, .0015), (-.015, -.03475, .007, .0015)), collar=.002),
    # C arc: both ends and both flanks, clear of a pinch across the arc's middle.
    "sensor_cap": dict(plate=("box", .002, 0., .017, .0315, .005),
                       posts=((-.011, .02125, .0022), (-.0115, -.02125, .0022), (.00725, .029, .0022),
                              (.01575, -.029, .0022)),
                       ribs=(), collar=.0008),
}
# The CCW propeller is the source mesh's mirror image (y -> -y).
STANDS["prop_ccw"] = dict(plate=STANDS["prop_cw"]["plate"][:2] + (-STANDS["prop_cw"]["plate"][2],) + STANDS["prop_cw"]["plate"][3:],
                          posts=tuple((x, -y, r) for x, y, r in STANDS["prop_cw"]["posts"]),
                          ribs=tuple((x, -y, hx, hy) for x, y, hx, hy in STANDS["prop_cw"]["ribs"]),
                          collar=STANDS["prop_cw"]["collar"])
_STAND_TOPS = {}


def _stand_top(assets, kit, kind, region):
    """Lowest collision-material height (part frame z) above a support region.

    Vertical rays from below over the region against the union of the part's
    convex collision pieces: the support's flat top meets the part's lowest
    point there exactly at the staged pose.
    """
    key = (str(assets), kind, region)
    if key not in _STAND_TOPS:
        import trimesh
        part = kit["parts"][kind]
        mesh = trimesh.util.concatenate([trimesh.load_mesh(Path(assets) / "given/kit/collision" / fn)
                                         for fn in part["collision"]])
        if region[0] == "post":
            _, x, y, r = region
            samples = [(x, y)] + [(x + f*r*math.cos(a), y + f*r*math.sin(a))
                                  for f, n in ((1/3, 8), (2/3, 16), (1., 24))
                                  for a in np.linspace(0, 2*math.pi, n, endpoint=False)]
        else:
            _, x, y, hx, hy = region
            samples = [(x + u, y + v) for u in np.linspace(-hx, hx, 17) for v in np.linspace(-hy, hy, 5)]
        origins = np.array([[u, v, -.05] for u, v in samples])
        hits, rays, _ = mesh.ray.intersects_location(origins, np.tile([0., 0., 1.], (len(origins), 1)),
                                                     multiple_hits=False)
        if len(set(rays.tolist())) < len(origins) / 2:
            raise ValueError(f"{kind} stand support {region} is not under the part's material")
        _STAND_TOPS[key] = float(hits[:, 2].min())
    return _STAND_TOPS[key]


def stand_geoms(assets, kit, kind, pos, q):
    """World geoms (name suffix, MJCF attributes) of the staging stand under a part at pos/q."""
    design = STANDS[kind]
    R = Rotation.from_quat(np.roll(q, -1)).as_matrix()
    if abs(R[2, 2] - 1) > 1e-9:
        raise ValueError(f"{kind} stand needs an upright staged part")
    pos = np.asarray(pos, float)
    yaw_q = text(quat(R))

    def at(x, y, z):
        return text([*(pos[:2] + R[:2, :2] @ [x, y]), z])
    t = STAND_PLATE_M
    plate = design["plate"]
    geoms = []
    if plate[0] == "disc":
        geoms.append(("plate", dict(type="cylinder", size=text([plate[3], t/2]), pos=at(plate[1], plate[2], t/2))))
    else:
        # Rounded rectangle: two crossed boxes and four corner discs.
        _, x, y, hx, hy, c = plate
        geoms.append(("plate", dict(type="box", size=text([hx, hy-c, t/2]), pos=at(x, y, t/2), quat=yaw_q)))
        geoms.append(("plate_x", dict(type="box", size=text([hx-c, hy, t/2]), pos=at(x, y, t/2), quat=yaw_q)))
        for k, (u, v) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            geoms.append((f"plate_corner{k}", dict(type="cylinder", size=text([c, t/2]),
                                                   pos=at(x+u*(hx-c), y+v*(hy-c), t/2))))
    for k, (x, y, r) in enumerate(design["posts"]):
        top = pos[2] + _stand_top(assets, kit, kind, ("post", x, y, r))
        geoms.append((f"post{k}", dict(type="cylinder", size=text([r, (top-t)/2]), pos=at(x, y, (top+t)/2))))
        h = STAND_COLLAR_H_M
        geoms.append((f"collar{k}", dict(type="cylinder", size=text([r+design["collar"], h/2]), pos=at(x, y, t+h/2))))
    for k, (x, y, hx, hy) in enumerate(design["ribs"]):
        top = pos[2] + _stand_top(assets, kit, kind, ("rib", x, y, hx, hy))
        geoms.append((f"rib{k}", dict(type="box", size=text([hx, hy, (top-t)/2]), pos=at(x, y, (top+t)/2), quat=yaw_q)))
    return geoms


def default_layout(graph, kit, seed, jitter=True):
    """Draw the WHOLE layout again on overlap. No per-part fallback collisions."""
    rng = np.random.default_rng(seed)
    large = {"guard": [-.15, -.40], "compute_cover": [-.24, .42], "battery": [-.07, .46],
             "lidar": [-.48, .42]}
    large.update({f"leg_{i+1}": [.08, -.25-.05*i] for i in range(4)})
    small = sorted([n for n in graph["instances"] if n not in large], key=lambda n:
                   -float(np.prod(np.ptp(kit["parts"][graph["instances"][n]["kind"]]["bounds_m"],axis=0)[:2])))
    slots = np.array([[x, y] for y in np.arange(.22, .54, .025) for x in np.arange(-.31, .26, .025)] +
                     [[x, y] for y in np.arange(-.42, -.21, .025) for x in np.arange(.14, .31, .025)])
    slots = slots[np.linalg.norm(slots - [-.4, 0], axis=1) < .69]
    # Parts installed after the guard need an approach outside its eventual
    # overhang. Otherwise a valid early assembly can block a later bench pick.
    late = {"mount_guard"}
    for mate in graph["mates"]:
        if any(r in late for r in mate["requires"]):
            late.add(mate["id"])
    late_parts = {m["part"] for m in graph["mates"] if m["id"] in late and m["part"] != "guard"}
    # Keep propellers accessible regardless of the chosen assembly order. The
    # guard is not a mechanical prerequisite for mounting a propeller.
    late_parts.update(n for n, p in graph["instances"].items() if p["kind"].startswith("prop_"))
    shadow = np.asarray(kit["parts"]["guard_deck"]["bounds_m"])[:, :2] + np.asarray(graph["fixture_pos_m"][:2])
    shadow += [[-.05, -.05], [.05, .05]]
    # The intact sensor cable and source-sized electronics occupy more of the
    # bench. Retry complete randomized packings; never accept an overlap.
    for attempt in range(512):
        layout, bounds = {}, []
        for name in list(large) + small:
            inst = graph["instances"][name]
            kind = inst["kind"]
            part = kit["parts"][kind]
            bb = np.array(part.get("rest_bounds_m", part["bounds_m"]))
            scale = 1 if jitter else 0
            yaw = rng.uniform(-.2, .2) * scale
            yaw_R = Rotation.from_euler("z", yaw).as_matrix()
            R = yaw_R @ np.asarray(part.get("rest_rotation", np.eye(3)))
            corners = np.array([[x, y, 0] for x in bb[:, 0] for y in bb[:, 1]]) @ yaw_R.T
            choices = np.array([large[name]]) if name in large else slots[rng.permutation(len(slots))] if jitter else slots
            for xy in choices:
                xy = xy + rng.uniform(-.004, .004, 2) * scale
                low, high = xy + corners[:, :2].min(0), xy + corners[:, :2].max(0)
                if name in late_parts and np.all(high > shadow[0]) and np.all(shadow[1] > low):
                    continue
                if not any(np.all(high + .008 > a) and np.all(b + .008 > low) for a, b in bounds):
                    break
            else:
                break
            bounds.append((low, high))
            # Thin parts rest on a printed staging stand (STANDS) inside their footprint.
            supported = kind in STANDS
            z = (.026 if supported else .001) - bb[0, 2]
            layout[name] = dict(pos=[*xy, z], quat=quat(R).tolist(), supported=supported)
        else:
            return layout
    raise RuntimeError("Cannot draw a non-overlapping bench layout; fix the bench plan")


class Cell:
    def __init__(self, assets, graph_path=None, seed=0, jitter=True, layout=None):
        self.assets = Path(assets)
        self.graph = json.loads(Path(graph_path or Path(__file__).with_name("assembly_graph.json")).read_text())
        setup_rng = np.random.default_rng(seed ^ 0xACA7)
        if jitter:
            self.graph["fixture_pos_m"][:2] = (np.asarray(self.graph["fixture_pos_m"][:2]) + setup_rng.uniform(-.01, .01, 2)).tolist()
        self.kit = json.loads((self.assets / "given/kit/kit.json").read_text())
        self.nominal = fixture_targets(self.graph, self.kit)
        self.layout = layout or default_layout(self.graph, self.kit, seed, jitter)
        root = ET.parse(self.assets / "given/xarm7/xarm7.xml").getroot()
        root.find("compiler").set("meshdir", str(self.assets / "given/xarm7/assets"))
        root.find("option").set("timestep", ".002")
        root.find("option").set("iterations", "80")
        root.find("option").set("cone", "elliptic")
        root.remove(root.find("keyframe"))
        ET.SubElement(root, "visual")
        ET.SubElement(root.find("visual"), "global", offwidth="1280", offheight="960")
        asset, world, eq = root.find("asset"), root.find("worldbody"), root.find("equality")
        world.find("body").set("pos", "-.4 0 0")
        ET.SubElement(world, "light", pos="-.3 -.3 1.8", dir="0 0 -1", diffuse=".8 .8 .8")
        ET.SubElement(world, "light", pos=".6 .5 1.4", dir="-.2 -.2 -1", diffuse=".5 .5 .5")
        ET.SubElement(world, "geom", name="table", type="box", size=".8 .75 .025", pos="0 0 -.025",
                      rgba=".6 .46 .3 1", friction=".8 .01 .001")
        ET.SubElement(world, "geom", name="fixture", type="box", size=".025 .025 .09", pos=text([*self.graph["fixture_pos_m"][:2], .09]),
                      rgba=".62 .72 .8 1")
        ET.SubElement(world, "camera", name="top", pos="-.04 0 1.65", xyaxes="1 0 0 0 1 0", fovy="55")
        ET.SubElement(world, "camera", name="review", pos="1.05 -1.4 1.1", xyaxes=".8 .6 0 -.3 .4 .866", fovy="50")
        hand = world.find(".//body[@name='xarm_gripper_base_link']")
        ET.SubElement(hand, "camera", name="wrist", pos=".055 0 .04", xyaxes="0 1 0 .96 0 .28", fovy="65")
        ET.SubElement(hand, "site", name="ft_site", pos="0 0 0", size=".002")
        sensor = ET.SubElement(root, "sensor")
        ET.SubElement(sensor, "force", name="wrist_force", site="ft_site")
        ET.SubElement(sensor, "torque", name="wrist_torque", site="ft_site")
        # Physical ruler squares: visible calibration cues, no camera matrix API.
        for i in range(11):
            ET.SubElement(world, "geom", type="box", size=".01 .01 .0003", pos=f"{-.34+i*.02} .62 .0003",
                          rgba=".05 .05 .05 1" if i % 2 else ".95 .95 .95 1", contype="0", conaffinity="0")
        render_components = dict(self.kit["parts"])
        for name, p in self.kit["parts"].items():
            if "rotor" in p:
                render_components[name+"_rotor"] = p["rotor"]
        for name, p in render_components.items():
            for i, visual in enumerate(p.get("visual_parts", [{"mesh": p["mesh"]}])):
                # A material region is an open surface, not a mass volume.
                # Dynamics uses the explicit body inertia and convex colliders.
                ET.SubElement(asset, "mesh", name=f"vis_{name}_{i}",
                              file=str(self.assets / "given/kit/meshes" / visual["mesh"]), inertia="shell")
            for i, fn in enumerate(p["collision"]):
                ET.SubElement(asset, "mesh", name=f"col_{name}_{i}", file=str(self.assets / "given/kit/collision" / fn))
        self._body_xml(world, "base", "base", self.graph["fixture_pos_m"], [1, 0, 0, 0], fixed=True)
        for name, inst in self.graph["instances"].items():
            rest = self.layout[name]
            self._body_xml(world, name, inst["kind"], rest["pos"], rest["quat"])
            if rest["supported"]:
                # A printed staging stand under the part's actual collision material.
                for suffix, attributes in stand_geoms(self.assets, self.kit, inst["kind"], rest["pos"], rest["quat"]):
                    ET.SubElement(world, "geom", name=f"stand_{name}_{suffix}", rgba=STAND_RGBA, **attributes)
        for mate in self.graph["mates"]:
            ET.SubElement(eq, "weld", name=mate["id"], body1="part_" + mate["carrier"],
                          body2="part_" + mate["part"], active="false", solref=".004 1", solimp=".95 .99 .001")
        self.xml = ET.tostring(root, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        if jitter:
            for camera, offset, angular in (("top", .02, 1.5), ("wrist", .002, 1)):
                i = self.m.camera(camera).id
                self.m.cam_pos[i] += setup_rng.uniform(-offset, offset, 3)
                nominal_R = Rotation.from_quat(np.roll(self.m.cam_quat[i], -1))
                delta = Rotation.from_euler("xyz", setup_rng.uniform(-angular, angular, 3), degrees=True)
                self.m.cam_quat[i] = np.roll((nominal_R * delta).as_quat(), 1)
                self.m.cam_fovy[i] *= setup_rng.uniform(.98, 1.02)
        self.d = mujoco.MjData(self.m)
        self.arm_qadr = np.array([self.m.jnt_qposadr[self.m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dadr = np.array([self.m.jnt_dofadr[self.m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.d.qpos[self.arm_qadr] = HOME_Q
        self.d.ctrl[:7] = HOME_Q
        mujoco.mj_forward(self.m, self.d)
        self.ids = {n: self.m.body("part_" + n).id for n in self.nominal}
        self.coupling = Coupling(self)
        self.violations, self.events, self.frames = [], [], []
        self.recording = False
        self.max_force = 0
        self.max_nonpad_force = 0
        self.nonpad_contacts = {}
        self.pad_contacts = {}
        self.over_force = 0
        self.rng = np.random.default_rng(seed ^ 8171)
        self.ft_bias = self.rng.normal(0, .15, 6)
        self.start_time = time.monotonic()
        self.tick_count = 0
        self.sim_cap_s = math.inf
        # Every initialization settle tick is physical; exclude startup transients
        # from safety, but do not permit couplings to activate during settling.
        for _ in range(300):
            mujoco.mj_step(self.m, self.d)
        self.d.time = 0

    def _body_xml(self, world, name, kind, pos, q, fixed=False):
        p = self.kit["parts"][kind]
        b = ET.SubElement(world, "body", name="part_" + name, pos=text(pos), quat=text(q))
        if not fixed:
            ET.SubElement(b, "freejoint", name="free_" + name)
        bb = np.array(p["bounds_m"])
        sz = bb[1] - bb[0]
        inertia = p["mass_kg"] / 12 * np.array([sz[1]**2+sz[2]**2, sz[0]**2+sz[2]**2, sz[0]**2+sz[1]**2])
        if "inertial" in p:
            properties = p["inertial"]
            inertia = np.asarray(properties["inertia_kg_m2"], dtype=float)
            if inertia.shape != (3, 3) or not np.all(np.linalg.eigvalsh(inertia) > 0):
                raise ValueError(f"{kind} has invalid imported inertia")
            ET.SubElement(b, "inertial", mass=str(p["mass_kg"]),
                          pos=text(properties["center_m"]),
                          fullinertia=text(inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]]))
        elif kind.startswith("prop_") or kind in ("sensor_cap", "sensor_cheek"):
            # Homogeneous manufactured parts use their actual material volume.
            # A box centre is especially misleading for a C plate or propeller.
            import trimesh
            volume = trimesh.load_mesh(self.assets/"given/kit/meshes"/p["mesh"])
            volume.merge_vertices(digits_vertex=7)
            volume.update_faces(volume.unique_faces())
            volume.update_faces(volume.nondegenerate_faces())
            volume.remove_unreferenced_vertices()
            if not volume.is_volume:
                raise RuntimeError(f"{kind} mass-property mesh is not a closed volume")
            inertia = volume.moment_inertia * p["mass_kg"] / volume.volume
            ET.SubElement(b,"inertial",mass=str(p["mass_kg"]),pos=text(volume.center_mass),
                          fullinertia=text(inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]]))
        else:
            ET.SubElement(b, "inertial", mass=str(p["mass_kg"]), pos=text(bb.mean(0)), diaginertia=text(np.maximum(inertia, 1e-7)))
        for i, visual in enumerate(p.get("visual_parts", [{"mesh": p["mesh"], "rgba": p["rgba"]}])):
            ET.SubElement(b, "geom", name=f"visual_{name}_{i}", type="mesh", mesh=f"vis_{kind}_{i}",
                          rgba=text(visual.get("rgba", p["rgba"])),
                          contype="0", conaffinity="0", group="2")
        for i, _ in enumerate(p["collision"]):
            ET.SubElement(b, "geom", name=f"collision_{name}_{i}", type="mesh", mesh=f"col_{kind}_{i}",
                          rgba=text(p["rgba"]), group="3", friction=".8 .02 .002", solref=".004 1", condim="4")
        if kind == "motor":
            definition = p["rotor"]
            rotor = ET.SubElement(b, "body", name="part_"+name+"/rotor",
                                  pos=text(definition["hinge_pos_m"]))
            ET.SubElement(rotor, "joint", name="rotor_"+name, type="hinge", axis="0 0 1", damping=".0002", frictionloss=".001")
            inertia = np.asarray(definition["inertial"]["inertia_kg_m2"])
            ET.SubElement(rotor, "inertial", mass=str(definition["mass_kg"]),
                          pos=text(definition["inertial"]["center_m"]),
                          fullinertia=text(inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]]))
            for i, visual in enumerate(definition["visual_parts"]):
                ET.SubElement(rotor, "geom", name=f"visual_{name}_rotor_{i}", type="mesh", mesh=f"vis_motor_rotor_{i}",
                              rgba=text(visual["rgba"]), contype="0", conaffinity="0", group="2")
            for i, _ in enumerate(definition["collision"]):
                ET.SubElement(rotor, "geom", name=f"collision_{name}_rotor_{i}", type="mesh", mesh=f"col_motor_rotor_{i}",
                              rgba=text(definition["rgba"]), group="3", friction=".8 .02 .002", solref=".004 1", condim="4")
        return b

    def pose(self, name):
        return body_pose(self.d, self.ids[name])

    def tick(self):
        if self.tick_count * DT >= self.sim_cap_s - 1e-8:
            raise TimeoutError("Simulation budget exhausted")
        for _ in range(5):
            mujoco.mj_step(self.m, self.d)
        self.tick_count += 1
        self.coupling.tick()
        self._watch()
        if self.recording and self.tick_count % 4 == 0:
            self.frames.append(self.d.qpos.copy())

    def _watch_tables(self):
        """Per-geom/body classification for the contact monitor, computed once.

        Only bookkeeping is cached; every tick still inspects every contact.
        """
        m = self.m
        bodies = [m.body(i).name for i in range(m.nbody)]
        geoms = [m.geom(i).name or "" for i in range(m.ngeom)]
        parts = sorted({n.partition("/")[0] for n in bodies if n.startswith("part_")})
        index = {n: i for i, n in enumerate(parts)}
        self._w = dict(
            bodies=bodies, geoms=geoms, parts=parts,
            geom_body=np.asarray(m.geom_bodyid, dtype=np.int64).copy(),
            body_part=np.array([index[n.partition("/")[0]] if n.startswith("part_") else -1 for n in bodies]),
            body_arm=np.array([not n.startswith("part_") and n != "world" for n in bodies]),
            geom_pad=np.array(["finger_pad" in n for n in geoms]),
            link_base=m.body("link_base").id, world=m.body("world").id,
            parent=np.asarray(m.body_parentid).copy(),
            free_dof=np.array([m.jnt_dofadr[m.joint("free_" + n).id] for n in self.graph["instances"]]),
            instances=list(self.graph["instances"]))

    def _watch(self):
        w = getattr(self, "_w", None)
        if w is None:
            self._watch_tables()
            w = self._w
        part_forces, robot_forces, self_forces, pad_forces = {}, {}, {}, {}
        ncon = int(self.d.ncon)
        if ncon:
            con = self.d.contact
            g1 = np.asarray(con.geom1[:ncon], dtype=np.int64)
            g2 = np.asarray(con.geom2[:ncon], dtype=np.int64)
            adr = np.asarray(con.efc_address[:ncon], dtype=np.int64)
            dim = np.asarray(con.dim[:ncon], dtype=np.int64)
            # mj_contactForce for elliptic cones: the contact-frame force is the
            # constraint force itself (normal, then tangential components).
            efc = self.d.efc_force
            f = np.zeros(ncon)
            ok = adr >= 0
            normal = np.where(ok, efc[np.where(ok, adr, 0)], 0.)
            tangent = ok & (dim >= 3)
            t1 = np.where(tangent, efc[np.where(tangent, adr+1, 0)], 0.)
            t2 = np.where(tangent, efc[np.where(tangent, adr+2, 0)], 0.)
            f = np.sqrt(normal*normal + t1*t1 + t2*t2)
            b1, b2 = w["geom_body"][g1], w["geom_body"][g2]
            p1, p2 = w["body_part"][b1], w["body_part"][b2]
            p2 = np.where(p2 == p1, -1, p2)  # one physical part counts once per contact
            idx = np.stack([p1, p2], axis=1).ravel()
            weight = np.repeat(f, 2)
            use = idx >= 0
            sums = np.bincount(idx[use], weights=weight[use], minlength=len(w["parts"]))
            for k in np.flatnonzero(np.bincount(idx[use], minlength=len(w["parts"]))):
                part_forces[w["parts"][k]] = float(sums[k])
            arm1, arm2 = w["body_arm"][b1], w["body_arm"][b2]
            pad1, pad2 = w["geom_pad"][g1], w["geom_pad"][g2]
            bodies, geoms, parent = w["bodies"], w["geoms"], w["parent"]
            for ci in np.flatnonzero(arm1 | arm2 | pad1 | pad2):
                fc = float(f[ci])
                body_ids = (int(b1[ci]), int(b2[ci]))
                names = [bodies[b] for b in body_ids]
                gnames = [geoms[int(g1[ci])], geoms[int(g2[ci])]]
                arm = [bool(arm1[ci]), bool(arm2[ci])]
                mounted_base = w["link_base"] in body_ids and w["world"] in body_ids
                for side, other in ((0, 1), (1, 0)):
                    if "finger_pad" in gnames[side] and not arm[other]:
                        surface = names[other] if names[other] != "world" else "world:" + gnames[other]
                        key = gnames[side] + "|" + surface
                        pad_forces[key] = pad_forces.get(key, 0) + fc
                if any(arm) and not all(arm) and not mounted_base and not any("finger_pad" in n for n in gnames):
                    for n, is_arm in zip(names, arm):
                        if is_arm:
                            robot_forces[n] = robot_forces.get(n, 0)+fc
                if all(arm):
                    a, b = body_ids
                    if parent[a] != b and parent[b] != a and a != b:
                        pair = tuple(sorted(names))
                        self_forces[pair] = self_forces.get(pair, 0) + fc
        for pair, f in self_forces.items():
            self._contact_evidence("self:" + "|".join(pair), f)
            if f > 30:
                self.violations.append(dict(t=float(self.d.time),kind="robot_self_collision",force_N=float(f),bodies=pair))
        for n,f in robot_forces.items():
            self._contact_evidence("external:" + n, f)
            if f > 30:
                self.violations.append(dict(t=float(self.d.time),kind="robot_collision",force_N=float(f),body=n))
        for key, f in pad_forces.items():
            self._record_contact(self.pad_contacts, key, f)
        part_max = max(part_forces.values(),default=0)
        self.max_force = max(self.max_force, part_max)
        self.over_force = self.over_force + 1 if part_max > 150 else 0
        if self.over_force == 20:
            self.violations.append(dict(t=float(self.d.time), kind="sustained_part_force", force_N=float(part_max)))
        if not np.isfinite(self.d.qpos).all():
            raise RuntimeError("Nonfinite physical state")
        speeds = np.linalg.norm(self.d.qvel[w["free_dof"][:, None] + np.arange(3)], axis=1)
        for k in np.flatnonzero(speeds > 3):
            self.violations.append(dict(t=float(self.d.time), kind="part_speed", part=w["instances"][k]))

    def _contact_evidence(self, key, force):
        """Owner-only contact evidence; this never changes controls or scoring."""
        self.max_nonpad_force = max(self.max_nonpad_force, float(force))
        self._record_contact(self.nonpad_contacts, key, force)

    def _record_contact(self, events, key, force):
        if force <= .1:
            return
        event = events.setdefault(key, dict(first_s=float(self.d.time), ticks=0, peak_N=0.))
        event["last_s"] = float(self.d.time)
        event["ticks"] += 1
        event["peak_N"] = max(event["peak_N"], float(force))

    def result(self):
        achieved = [n for n, s in self.coupling.state.items() if s == "LOCKED"]
        score = sum(m["weight"] for m in self.graph["mates"] if m["id"] in achieved) / self.graph["total_weight"]
        return dict(score=score if not self.violations else 0, raw_score=score, achieved=achieved,
                    states=dict(self.coupling.state), safety_ok=not self.violations, violations=self.violations[:50],
                    sim_time_s=float(self.d.time), max_force_N=self.max_force,
                    max_nonpad_robot_force_N=self.max_nonpad_force,
                    nonpad_contact_evidence={k: dict(v) for k, v in self.nonpad_contacts.items()},
                    pad_contact_evidence={k: dict(v) for k, v in self.pad_contacts.items()},
                    coupling_events=[dict(event) for event in self.events])


class Coupling:
    def __init__(self, cell):
        self.c = cell
        self.state = {m["id"]: "WAITING" if m["requires"] else "FREE" for m in cell.graph["mates"]}
        self.hold = {m["id"]: 0 for m in cell.graph["mates"]}
        self.assigned = {}
        self.candidate = {}
        self.relative = {m["id"]: np.linalg.inv(cell.nominal[m["carrier"]]) @ cell.nominal[m["part"]]
                         for m in cell.graph["mates"]}

    def target(self, mate):
        return self.c.pose(self.carrier(mate)) @ self.relative[mate["id"]]

    def carrier(self, mate):
        name, _, suffix = mate["carrier"].partition("/")
        name = self.assigned.get("mount_" + name, name)
        return name + ("/" + suffix if suffix else "")

    def tick(self):
        c = self.c
        for m in c.graph["mates"]:
            key = m["id"]
            if self.state[key] in ("LOCKED", "LOCKED_OFF"):
                continue
            if not all(self.state[x] in ("LOCKED", "LOCKED_OFF") for x in m["requires"]):
                continue
            target = self.target(m)
            kind = c.graph["instances"][m["part"]]["kind"]
            candidates = [n for n, p in c.graph["instances"].items()
                          if p["kind"] == kind and n not in self.assigned.values()]
            if not candidates:
                continue
            physical = min(candidates, key=lambda n: np.linalg.norm(c.pose(n)[:3,3]-target[:3,3]))
            if self.candidate.get(key) != physical:
                self.hold[key] = 0
                self.candidate[key] = physical
            actual = c.pose(physical)
            error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
            if error > m["capture_m"]:
                # Outside the capture distance the state is FREE whatever the angle.
                self.state[key], self.hold[key] = "FREE", 0
                continue
            rot = target[:3, :3].T @ actual[:3, :3]
            angle = orientation_error(rot, m["axial_symmetry"])
            if angle > math.radians(8):
                self.state[key], self.hold[key] = "FREE", 0
                continue
            if error > m["position_tolerance_m"]:
                self.state[key], self.hold[key] = "ENGAGED", 0
                continue
            # Free-joint angular qvel is body-local. Compare both velocities in
            # world axes at body-frame origins, including the carrier's rigid
            # rotational velocity at the part's location (not its own COM).
            part_v, carrier_v = np.zeros(6), np.zeros(6)
            carrier_id = c.ids[self.carrier(m)]
            mujoco.mj_objectVelocity(c.m, c.d, mujoco.mjtObj.mjOBJ_XBODY, c.ids[physical], part_v, 0)
            mujoco.mj_objectVelocity(c.m, c.d, mujoco.mjtObj.mjOBJ_XBODY, carrier_id, carrier_v, 0)
            offset = actual[:3, 3] - c.d.xpos[carrier_id]
            relative_speed = np.linalg.norm(part_v[3:] - carrier_v[3:] - np.cross(carrier_v[:3], offset))
            relative_angular_speed = np.linalg.norm(part_v[:3] - carrier_v[:3])
            still = relative_speed < .005 and relative_angular_speed < math.radians(2)
            if not still:
                self.state[key], self.hold[key] = "ENGAGED", 0
            else:
                self.state[key] = "HOLDING"
                self.hold[key] += 1
                if self.hold[key] * DT >= m["hold_s"]:
                    e = c.m.equality(key).id
                    rel = np.linalg.inv(c.pose(self.carrier(m))) @ actual
                    c.m.eq_obj1id[e] = c.ids[self.carrier(m)]
                    c.m.eq_obj2id[e] = c.ids[physical]
                    c.m.eq_data[e, 3:6] = rel[:3, 3]
                    c.m.eq_data[e, 6:10] = quat(rel[:3, :3])
                    c.d.eq_active[e] = 1
                    self.assigned[key] = physical
                    self.state[key] = "LOCKED" if angle <= math.radians(m["angle_tolerance_deg"]) else "LOCKED_OFF"
                    c.events.append(dict(t=float(c.d.time), mate=key, state=self.state[key], error_m=float(error),
                                         angle_deg=math.degrees(float(angle)), relative_speed_m_s=float(relative_speed),
                                         relative_angular_speed_deg_s=math.degrees(float(relative_angular_speed))))
