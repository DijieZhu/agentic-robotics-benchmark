"""Licensed robot meshes and authored workcells used by both live and review cameras.

The navigation/flight collision and control abstractions are deliberately unchanged.
Meshes are not evidence of hardware-accurate dynamics or learned legged locomotion.
"""
from functools import lru_cache
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from pbr_scene import upgrade


def asset_root():
    candidates = [Path(os.environ.get("VISUAL_ROBOTICS_ASSETS", "/opt/visual_robotics/assets"))]
    candidates.extend(p/"runs/visual-suite/visual-assets" for p in Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate/"provenance.json").exists():
            return candidate
    raise FileNotFoundError("Run prepare_visual_assets.py or fetch_visual_assets.py before rendering")


def vec(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def geom(parent, name, kind="box", pos=(0, 0, 0), size=(.1, .1, .1),
         rgba=(.5, .5, .5, 1), **kwargs):
    attrs = dict(name=name, type=kind, pos=vec(pos), size=vec(size), rgba=vec(rgba),
                 contype="0", conaffinity="0", mass="0", group="2")
    attrs.update({k: str(v) for k, v in kwargs.items()})
    return ET.SubElement(parent, "geom", attrs)


@lru_cache(maxsize=2)
def nominal_meshes(kind):
    folder, filename = (("unitree_go1", "go1.xml") if kind == "ground"
                        else ("bitcraze_crazyflie_2", "cf2.xml"))
    source = asset_root()/folder/filename
    root = ET.parse(source).getroot()
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    if kind == "ground":
        for i in range(model.njnt):
            name = model.joint(i).name
            if name.endswith("thigh_joint"):
                data.qpos[model.jnt_qposadr[i]] = .8
            elif name.endswith("calf_joint"):
                data.qpos[model.jnt_qposadr[i]] = -1.6
    mujoco.mj_forward(model, data)
    root_position = data.xpos[1].copy()
    files = {m.get("name", Path(m.get("file")).stem): source.parent/"assets"/m.get("file")
             for m in root.findall("asset/mesh")}
    rows, points = [], []
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[i] != 2:
            continue
        mid = int(model.geom_dataid[i])
        rotation = data.geom_xmat[i].reshape(3, 3)
        mr = np.zeros(9)
        mujoco.mju_quat2Mat(mr, model.mesh_quat[mid])
        raw_rotation = rotation@mr.reshape(3, 3).T
        raw_pos = data.geom_xpos[i]-root_position-raw_rotation@model.mesh_pos[mid]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, raw_rotation.ravel())
        rgba = model.geom_rgba[i].copy()
        if model.geom_matid[i] >= 0:
            rgba = model.mat_rgba[model.geom_matid[i]].copy()
        rows.append(dict(file=files[model.mesh(mid).name], pos=raw_pos, quat=quat, rgba=rgba))
        begin, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[begin:begin+count]@rotation.T + data.geom_xpos[i]-root_position
        points.append(vertices)
    points = np.concatenate(points)
    return rows, points.min(axis=0), points.max(axis=0)


def robot_meshes(root, body, kind, radius=.13):
    asset = root.find("asset")
    rows, lower, upper = nominal_meshes(kind)
    scale = (2*radius/max((upper-lower)[:2])) if kind == "ground" else 4.8
    z_shift = -.23-lower[2]*scale+.008 if kind == "ground" else -.01
    for i, row in enumerate(rows):
        mesh_name = f"real_{kind}_{i}"
        ET.SubElement(asset, "mesh", name=mesh_name, file=str(row["file"]), scale=vec([scale]*3))
        rgba = row["rgba"]
        if kind == "ground":
            rgba = (.68, .71, .73, 1) if "trunk" in str(row["file"]) else (.23, .26, .28, 1)
        geom(body, mesh_name, "mesh", row["pos"]*scale+[0, 0, z_shift], (), rgba,
             mesh=mesh_name, quat=vec(row["quat"]))
    if kind == "ground":
        geom(body, "robot_marker", pos=(0, 0, upper[2]*scale+z_shift+.004),
             size=(radius*.64, radius*.25, .004), rgba=(.95, .025, .035, 1))
    else:
        # A visible tracking sticker, not a privileged state overlay.
        geom(body, "robot_marker", "ellipsoid", pos=(0, 0, .05), size=(.09, .07, .045), rgba=(.95, .025, .035, 1))


def materials(root):
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1=".24 .29 .32", rgb2=".55 .60 .63", width="512", height="2048")
    ET.SubElement(asset, "texture", name="workfloor", type="2d", file=str(asset_root()/"textures/concrete_floor_02.png"))
    ET.SubElement(asset, "texture", name="asphalt", type="2d", file=str(asset_root()/"textures/asphalt_02.png"))
    ET.SubElement(asset, "material", name="road_mat", texture="asphalt", texrepeat="2 2", texuniform="true",
                  reflectance="0", specular=".05", shininess=".1")
    ET.SubElement(asset, "material", name="floor_mat", texture="workfloor", texrepeat=".5 .5",
                  texuniform="true", reflectance="0", specular=".12", shininess=".2")
    ET.SubElement(asset, "material", name="indoor_mat", rgba=".62 .63 .64 1",
                  reflectance="0", specular=".18", shininess=".35")
    ET.SubElement(asset, "material", name="brushed", rgba=".55 .59 .62 1", specular=".65", shininess=".7")
    ET.SubElement(asset, "material", name="dark_metal", rgba=".10 .13 .15 1", specular=".5", shininess=".55")
    ET.SubElement(asset, "texture", name="river", type="2d", builtin="gradient", width="512", height="512",
                  rgb1=".04 .28 .64", rgb2=".05 .30 .67")
    ET.SubElement(asset, "material", name="water_mat", texture="river", texrepeat="40 9",
                  reflectance=".2", specular=".8", shininess=".9")
    for name, filename, repeat in (("brick", "brick_wall_001.png", ".67 .67"),
                                    ("rust", "rusty_metal_02.png", "1 1")):
        ET.SubElement(asset, "texture", name=name, type="2d", file=str(asset_root()/"textures"/filename))
        ET.SubElement(asset, "material", name=name+"_mat", texture=name, texrepeat=repeat,
                      texuniform="true", specular=".1", shininess=".15")
    return asset


def gantry(world, agent, obj):
    # A complete Cartesian industrial gantry: rails, moving crossbeam, Y carriage,
    # telescoping Z ram and suction tool. Its three real slide axes are unchanged.
    old = world.find("geom[@name='gantry']")
    if old is not None:
        old.set("rgba", "0 0 0 0")
    for x in (-2.9, 2.9):
        for y in (-2.85, 2.85):
            geom(world, f"column_{x}_{y}", pos=(x, y, .8), size=(.1, .1, .8), material="brushed")
            geom(world, f"foot_{x}_{y}", pos=(x, y, .035), size=(.19, .19, .035), material="dark_metal")
    for y in (-2.85, 2.85):
        geom(world, f"xrail_{y}", pos=(0, y, 1.7), size=(3.05, .09, .12), material="brushed")
        geom(world, f"xtrack_{y}", pos=(0, y, 1.83), size=(2.95, .035, .025), material="dark_metal")
    cross = ET.SubElement(world, "body", name="visual_crossbeam", mocap="true", pos="0 0 2")
    geom(cross, "crossbeam", pos=(0, 0, 0), size=(.085, 2.95, .1), material="brushed")
    geom(cross, "cross_track", pos=(.09, 0, .03), size=(.018, 2.8, .028), material="dark_metal")
    carriage = ET.SubElement(world, "body", name="visual_carriage", mocap="true", pos="0 0 2")
    geom(carriage, "y_carriage", size=(.2, .24, .11), material="dark_metal")
    geom(carriage, "drive_motor", "cylinder", (.21, 0, .07), (.09, .14), material="brushed")
    ram = ET.SubElement(world, "body", name="visual_ram", mocap="true", pos="0 0 1")
    geom(ram, "ram", size=(.045, .045, .8), material="brushed")
    for i in range(5):
        geom(agent, f"bellow_{i}", "cylinder", (0, 0, .07+i*.025), (.065, .008), rgba=(.07, .09, .1, 1))
    geom(agent, "tool_mount", "cylinder", (-.28, 0, .23), (.08, .045), material="brushed")
    geom(agent, "cantilever", pos=(-.15, 0, .22), size=(.15, .035, .025), material="brushed")
    geom(agent, "suction_lip", "cylinder", (0, 0, -.025), (.095, .009), rgba=(.12, .13, .14, 1))
    for g in agent.findall("geom"):
        if g.get("type") == "capsule":
            g.set("rgba", "0 0 0 0")
    fragile = obj.find("geom[@name='fragile']")
    fragile.set("rgba", "0 0 0 0")
    geom(obj, "ceramic_body", "cylinder", (0, 0, -.015), (.105, .095), rgba=(.06, .65, .18, 1))
    geom(obj, "ceramic_base", "cylinder", (0, 0, -.112), (.09, .008), rgba=(.07, .32, .12, 1))
    geom(obj, "ceramic_lid", "cylinder", (0, 0, .09), (.11, .012), rgba=(.03, .9, .10, 1))
    geom(obj, "lid_knob", "sphere", (0, 0, .11), (.026,), rgba=(.08, .6, .12, 1))
    # Table top coincides with the original contact plane; no additional collision.
    geom(world, "workbench", pos=(0, 0, -.075), size=(3.2, 3.1, .076), rgba=(.50, .53, .55, 1))
    tray = world.find("geom[@name='tray']")
    tx, ty, _ = map(float, tray.get("pos").split())
    for j, (dx, dy, sx, sy) in enumerate(((.28, 0, .025, .29), (-.28, 0, .025, .29),
                                         (0, .28, .29, .025), (0, -.28, .29, .025))):
        geom(world, f"tray_rim{j}", pos=(tx+dx, ty+dy, .065), size=(sx, sy, .035), rgba=(.035, .12, .7, 1))


def building(world, name, x, y, sx=.4, sy=.4, height=.8):
    # Metre-scale industrial cabinets, not miniature multi-storey buildings.
    geom(world, f"{name}_panel", pos=(x, y-sy-.005, height/2),
         size=(sx*.85, .012, height*.42), material="rust_mat")
    for j, z in enumerate(np.linspace(.46, .65, 6)):
        geom(world, f"{name}_vent{j}", pos=(x+.1, y-.42, z), size=(.14, .01, .009), rgba=(.08, .1, .11, 1))
    geom(world, f"{name}_roof", pos=(x, y, height+.02), size=(sx+.03, sy+.03, .035), material="rust_mat")
    geom(world, f"{name}_handle", pos=(x-.2, y-.435, .4), size=(.018, .025, .09), rgba=(.12, .13, .14, 1))
    geom(world, f"{name}_roof_vent", pos=(x+.16, y+.13, .9), size=(.12, .11, .055), material="brushed")


def site_context(world, kind, rng):
    """Architectural context outside the bounded task area; no hidden collision changes."""
    if kind == "bridge":
        for side in (-1, 1):
            geom(world, f"riverbank_{side}", pos=(0, side*2.8, -.2), size=(8, .8, .25), material="floor_mat")
            for x in np.arange(-7, 8, 1.0):
                geom(world, f"quay_post_{side}_{x}", pos=(x, side*2.5, .05),
                     size=(.08, .09, .4), material="rust_mat")
            geom(world, f"quay_rail_{side}", pos=(0, side*2.5, .4), size=(8, .035, .035), material="dark_metal")
        return
    # Walls/windows are in the real scene, but deliberately outside [-3.8,3.8]^2.
    for i, x in enumerate((-5.0, 5.0)):
        geom(world, f"site_wall_{i}", pos=(x, .5, 2.1), size=(.18, 6, 2.1), material="brick_mat")
        for j, y in enumerate((-3.8, -1.2, 1.4, 4.0)):
            geom(world, f"site_column_{i}_{j}", pos=(x-np.sign(x)*.2, y, 2.6),
                 size=(.11, .12, 2.6), material="rust_mat")
            for z in (1.9, 3.1):
                geom(world, f"site_window_{i}_{j}_{z}", pos=(x-np.sign(x)*.19, y+.7, z),
                     size=(.016, .52, .45), rgba=(.17, .25, .28, 1))
    geom(world, "site_back", pos=(0, 5.4, 2.1), size=(5, .18, 2.1), material="brick_mat")
    for j, x in enumerate((-4, -2, 0, 2, 4)):
        geom(world, f"site_beam_{j}", pos=(x, 0, 5.1), size=(.09, 5.5, .12), material="rust_mat")
        geom(world, f"site_back_pillar_{j}", pos=(x, 5.15, 2.6), size=(.12, .12, 2.6), material="dark_metal")
    for j, x in enumerate((-3.4, -1.7, 0, 1.7, 3.4)):
        height = float(rng.uniform(.55, 1.25))
        geom(world, f"site_crate_{j}", pos=(x, 4.5, height/2), size=(.52, .42, height/2), material="rust_mat")
        for dx in (-.38, .38):
            geom(world, f"site_crate_strap_{j}_{dx}", pos=(x+dx, 4.06, height/2),
                 size=(.025, .014, height/2), material="brushed")


def texture_box_faces(root):
    """Give each textured box face metric UVs instead of MuJoCo's stretched box projection."""
    asset = root.find("asset")
    tiles = {"brick_mat": 2., "rust_mat": 1., "floor_mat": 3., "road_mat": 3.}
    for parent in list(root.find("worldbody").iter()):
        for g in list(parent.findall("geom")):
            material = g.get("material")
            if g.get("type") != "box" or material not in tiles or g.get("group") == "3":
                continue
            half = np.fromstring(g.get("size"), sep=" ")
            vertices, uv, faces = [], [], []
            for axis in range(3):
                for sign in (-1, 1):
                    normal = np.eye(3)[axis]*sign
                    # Wall U runs horizontally, not vertically. Keep V upright.
                    u = (np.array([0., sign, 0.]) if axis == 0 else
                         np.array([-sign, 0., 0.]) if axis == 1 else np.array([1., 0., 0.]))
                    v = np.cross(normal, u)
                    du, dv = float(np.abs(u)@half), float(np.abs(v)@half)
                    start = len(vertices)
                    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                        vertices.append(normal*half[axis]+u*a*du+v*b*dv)
                        uv.append((a*du/tiles[material], 1-b*dv/tiles[material]))
                    faces.extend(((start, start+1, start+2), (start, start+2, start+3)))
            name = g.get("name")+"_textured"
            ET.SubElement(asset, "mesh", name=name, vertex=vec(np.ravel(vertices)),
                          texcoord=vec(np.ravel(uv)), face=" ".join(map(str, np.ravel(faces))))
            attrs = {k: v for k, v in g.attrib.items() if k not in ("name", "type", "size", "mass")}
            attrs.update(name=name, type="mesh", mesh=name, contype="0", conaffinity="0",
                         mass="0", group="2", rgba="1 1 1 1")
            ET.SubElement(parent, "geom", attrs)
            g.set("group", "3")


def decorate(xml, kind, radius=.13, seed=0):
    root = ET.fromstring(xml)
    materials(root)
    world = root.find("worldbody")
    rng = np.random.default_rng(seed)
    visual = root.find("visual")
    visual.find("global").set("offwidth", "1600")
    visual.find("global").set("offheight", "1000")
    visual.find("quality").set("shadowsize", "2048")
    visual.find("headlight").set("ambient", ".55 .55 .55")
    visual.find("headlight").set("diffuse", ".30 .30 .30")
    for light in world.findall("light"):
        light.set("diffuse", ".65 .65 .65")
        light.set("castshadow", "true")
        light.set("pos", "-3 -4 9")
    ET.SubElement(world, "light", pos="5 3 6", dir="-1 -1 -2", diffuse=".25 .25 .25", castshadow="false")
    floor = world.find("geom[@name='floor']")
    indoor = kind in ("flight", "arm", "icl", "cross")
    floor.set("rgba", ".62 .63 .64 1" if indoor else ".95 .95 .95 1")
    floor.set("material", "floor_mat" if indoor else "road_mat")
    agent = world.find("body[@name='agent']")
    if kind == "arm":
        gantry(world, agent, world.find("body[@name='object']"))
    else:
        for g in agent.findall("geom"):
            g.set("rgba", "0 0 0 0")
        robot_meshes(root, agent, "ground" if kind in ("icl", "cross") else "drone", radius)
    if kind in ("flight", "arm", "icl", "cross"):
        for j, x in enumerate((-4.3, 4.3)):
            geom(world, f"hall_wall{j}", pos=(x, 0, 1.6), size=(.12, 4.6, 1.6), rgba=(.48, .52, .55, 1))
            for k, y in enumerate((-3.5, 0, 3.5)):
                geom(world, f"hall_column{j}_{k}", pos=(x-.03, y, 1.6), size=(.16, .12, 1.6), material="brushed")
        geom(world, "hall_back", pos=(0, 4.5, 1.6), size=(4.4, .12, 1.6), rgba=(.38, .42, .45, 1))
        for i, x in enumerate((-3, -1.5, 0, 1.5, 3)):
            geom(world, f"back_window{i}", pos=(x, 4.36, 2), size=(.52, .018, .6), rgba=(.13, .25, .31, 1))
        for g in world.findall("geom"):
            if g.get("name", "").startswith("hall_") and "column" not in g.get("name", ""):
                g.set("material", "brick_mat")
                g.set("rgba", "1 1 1 1")
    site_context(world, kind, rng)
    sun = world.find("light")
    sun.set("pos", vec((rng.uniform(-4, -2), rng.uniform(-5, -3), rng.uniform(8, 10))))
    sun.set("diffuse", vec([rng.uniform(.58, .72)]*3))
    if kind == "flight":
        for g in list(world.findall("geom")):
            if g.get("name", "").startswith("gate_"):
                g.set("rgba", "1 .72 .05 1")
        gate = world.find("geom[@name='gate_top']")
        gx, gy, _ = map(float, gate.get("pos").split())
        for y in (gy-.8, gy+.8):
            geom(world, f"gate_foot{y}", pos=(gx, y, .04), size=(.22, .20, .04), rgba=(.12, .14, .16, 1))
    elif kind in ("icl", "cross"):
        for g in list(world.findall("geom")):
            if not g.get("name", "").startswith("wall"):
                continue
            g.set("rgba", ".42 .49 .58 1")
            x, y, z = map(float, g.get("pos").split())
            sx, sy, sz = map(float, g.get("size").split())
            geom(world, g.get("name")+"_cap", pos=(x, y, z+sz+.012), size=(sx+.015, sy, .012), rgba=(.42, .49, .58, 1))
            for k, yy in enumerate(np.arange(y-sy+.12, y+sy, .5)):
                geom(world, g.get("name")+f"_post{k}", pos=(x-.125, yy, .32), size=(.016, .025, .32), material="brushed")
    elif kind == "bridge":
        water = world.find("geom[@name='water']")
        water.set("rgba", "1 1 1 1")
        water.set("material", "water_mat")
        for name in ("deck_left", "deck_right"):
            g = world.find(f"geom[@name='{name}']")
            x, cy, z = map(float, g.get("pos").split())
            sx, sy, _ = map(float, g.get("size").split())
            g.set("rgba", ".50 .53 .55 1")
            g.set("material", "road_mat")
            for y in (cy-sy+.03, cy+sy-.03):
                geom(world, name+f"_rail{y}", pos=(x, y, .88), size=(sx, .025, .025), material="brushed")
                for j, xx in enumerate(np.arange(x-sx+.08, x+sx-.02, .37)):
                    geom(world, name+f"_post{y}_{j}", pos=(xx, y, .73), size=(.018, .018, .17), material="brushed")
            for j, xx in enumerate(np.arange(x-sx+.15, x+sx-.1, .55)):
                geom(world, name+f"_lane{j}", pos=(xx, cy, z+.061), size=(.11, .018, .0015), rgba=(.91, .9, .82, 1))
            geom(world, name+"_girder", pos=(x, cy, .40), size=(sx, sy*.75, .07), material="rust_mat")
        for i, y in enumerate((-1.75, 1.75)):
            geom(world, f"embankment{i}", pos=(0, y, -.02), size=(4, .33, .10), rgba=(.26, .32, .25, 1))
    elif kind == "rescue":
        for item in list(world.findall("geom")):
            if item.get("name", "").startswith("building"):
                x, y, z = map(float, item.get("pos").split())
                sx, sy, sz = map(float, item.get("size").split())
                item.set("material", "rust_mat")
                building(world, item.get("name"), x, y, sx, sy, 2*sz)
        human = world.find("body")
        human = next(b for b in world.findall("body") if b.find("geom[@name='victim']") is not None)
        victim = human.find("geom[@name='victim']")
        victim.set("rgba", "1 .38 .025 1")
        for i, yy in enumerate((-.075, .075)):
            geom(human, f"human_leg{i}", "capsule", (-.33, yy, -.02), (.045, .16),
                 rgba=(.12, .17, .22, 1), quat=".7071068 0 .7071068 0")
            geom(human, f"human_boot{i}", pos=(-.51, yy, -.03), size=(.07, .045, .04), rgba=(.09, .10, .11, 1))
            geom(human, f"human_arm{i}", "capsule", (.015, yy*2, -.015), (.035, .16),
                 rgba=(.95, .39, .035, 1), quat=".7071068 0 .7071068 0")
            geom(human, f"vest_stripe{i}", pos=(-.02, yy*.6, .081), size=(.1, .009, .002), rgba=(.91, .91, .85, 1))
        geom(human, "helmet", "sphere", (.23, 0, .02), (.085,), rgba=(.92, .88, .65, 1))
        for g in human.findall("geom"):
            for attr in ("pos", "size", "fromto"):
                if g.get(attr):
                    g.set(attr, vec(np.array(list(map(float, g.get(attr).split())))*1.65))
    upgrade(root, kind, asset_root(), rng)
    texture_box_faces(root)
    return ET.tostring(root, encoding="unicode")


def sync_visuals(model, data):
    """Position non-contact gantry structural followers from its actual slide axes."""
    for name in ("visual_crossbeam", "visual_carriage", "visual_ram"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        mid = model.body_mocapid[bid]
        x, y, z = data.qpos[:3]
        if name == "visual_crossbeam":
            data.mocap_pos[mid] = [x-.35, 0, 1.98]
        elif name == "visual_carriage":
            data.mocap_pos[mid] = [x-.35, y, 1.98]
        else:
            data.mocap_pos[mid] = [x-.28, y, z+.9]
    mujoco.mj_forward(model, data)
