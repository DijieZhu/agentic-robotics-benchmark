"""Open-asset industrial scenery and physically based material setup.

All additions are visible to the same cameras and have zero collision/mass.
Existing scored collision proxies remain authoritative and retain their shape.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def numbers(v):
    return " ".join(f"{float(x):.8g}" for x in v)


def add(parent, name, pos, size, material="painted_steel", **attrs):
    defaults = dict(name=name, type="box", pos=numbers(pos), size=numbers(size),
                    material=material, rgba="1 1 1 1", contype="0", conaffinity="0",
                    mass="0", group="2")
    defaults.update(attrs)
    return ET.SubElement(parent, "geom", defaults)


def prop(root, assets, name, instance, pos, size=None, yaw=0):
    folder = assets/"baked"/name
    info = json.loads((folder/"model.json").read_text())
    scale = np.ones(3) if size is None else np.asarray(size)/info["size"]
    body = ET.SubElement(root.find("worldbody"), "body", name=instance,
                         pos=numbers(pos), euler=f"0 0 {yaw}")
    asset = root.find("asset")
    for i, part in enumerate(info["parts"]):
        material = f"scan_{name}_{i}"
        if asset.find(f"material[@name='{material}']") is None:
            mat = ET.SubElement(asset, "material", name=material,
                                rgba=numbers(part["rgba"]), roughness=str(part["roughness"]),
                                metallic=str(part["metallic"]))
            for role, filename in part["textures"].items():
                texname = material+"_"+role
                ET.SubElement(asset, "texture", name=texname, type="2d", file=str(folder/filename),
                              colorspace="sRGB" if role == "rgb" else "linear")
                ET.SubElement(mat, "layer", role=role, texture=texname)
        meshname = f"{instance}_{i}"
        ET.SubElement(asset, "mesh", name=meshname, file=str(folder/part["mesh"]),
                      scale=numbers(scale), inertia="shell")
        add(body, meshname, (0, 0, 0), (), material, type="mesh", mesh=meshname)


def upgrade(root, kind, assets, rng):
    asset, world = root.find("asset"), root.find("worldbody")
    # Explicit colour spaces and maps prevent linear-space photo textures from
    # washing out; roughness/normal maps are data, never sRGB colour images.
    for texture in asset.findall("texture"):
        if texture.get("file"):
            texture.set("colorspace", "sRGB")
    for material, surface in (("floor_mat", "concrete_floor_02"), ("road_mat", "asphalt_02"),
                              ("brick_mat", "brick_wall_001"), ("rust_mat", "rusty_metal_02")):
        mat = asset.find(f"material[@name='{material}']")
        mat.set("roughness", "1")
        mat.set("metallic", "0")
        mat.set("texuniform", "false")
        mat.set("texrepeat", "1 1")
        rgb = mat.attrib.pop("texture")
        ET.SubElement(mat, "layer", role="rgb", texture=rgb)
        for role, suffix in (("normal", "nor_gl"), ("roughness", "rough")):
            texname = f"{material}_{role}"
            ET.SubElement(asset, "texture", name=texname, type="2d", colorspace="linear",
                          file=str(assets/"textures"/f"{surface}_{suffix}.png"))
            ET.SubElement(mat, "layer", role=role, texture=texname)
    for name, color, rough, metal in (
        ("painted_steel", (.33, .39, .40, 1), .54, .3),
        ("navigation_panel", (.22, .31, .52, 1), .86, 0),
        ("warm_plaster", (.65, .66, .63, 1), .9, 0),
        ("trim", (.16, .19, .20, 1), .46, .6),
        ("safety_paint", (.71, .54, .12, 1), .8, 0),
        ("window_glass", (.24, .37, .43, 1), .23, .2),
        ("lamp", (.95, .91, .77, 1), .4, 0),
        ("tracking_red", (.48, .001, .003, 1), 1, 0),
    ):
        ET.SubElement(asset, "material", name=name, rgba=numbers(color),
                      roughness=str(rough), metallic=str(metal), emission="1.2" if name == "lamp" else "0")
    for item in world.iter("geom"):
        if item.get("name") in ("robot_marker", "cup"):
            item.set("material", "tracking_red")
            item.set("rgba", "1 1 1 1")
    for name, rough, metal in (("brushed", .3, .8), ("dark_metal", .48, .65), ("water_mat", .17, .12)):
        material = asset.find(f"material[@name='{name}']")
        material.set("roughness", str(rough))
        material.set("metallic", str(metal))
    # One coherent building, not two nested walls and floating oversized beams.
    for item in list(world):
        if item.tag == "light" or item.get("name", "").startswith(("hall_", "site_", "back_window")):
            world.remove(item)
    root.find("visual/headlight").set("active", "0")
    ET.SubElement(world, "light", name="daylight", type="directional",
                  pos="-4 -4 8", dir=numbers((rng.uniform(.25, .45), .45, -1)),
                  diffuse="1 .96 .88", intensity=str(rng.uniform(22000, 28000)), castshadow="true")
    # The collision floor remains a plane. The finite render surface has metric UVs.
    floor = world.find("geom[@name='floor']")
    floor.set("group", "3")
    add(world, "site_ground", (0, 0, -.025), (12, 12, .025),
        "road_mat" if kind in ("bridge", "rescue") else "floor_mat")
    if kind != "bridge":
        for i, x in enumerate((-4.7, 4.7)):
            add(world, f"architecture_wall{i}", (x, .8, 2), (.13, 5.8, 2), "warm_plaster")
            add(world, f"architecture_base{i}", (x, .8, .55), (.15, 5.8, .55), "brick_mat")
            for j, y in enumerate((-3.8, -1.3, 1.2, 3.7)):
                add(world, f"architecture_column{i}_{j}", (x-np.sign(x)*.17, y, 2.15),
                    (.08, .10, 2.15), "painted_steel")
                for k in range(3):
                    add(world, f"architecture_window{i}_{j}_{k}",
                        (x-np.sign(x)*.145, y+.34+k*.58, 2.65), (.01, .26, .72), "window_glass")
                    add(world, f"architecture_window_frame{i}_{j}_{k}",
                        (x-np.sign(x)*.16, y+.63+k*.58, 2.65), (.018, .018, .76), "trim")
        add(world, "architecture_rear", (0, 5.1, 2), (4.7, .15, 2), "brick_mat")
        add(world, "loading_door", (.6, 4.91, 1.4), (1.5, .04, 1.4), "painted_steel")
        for j in range(15):
            add(world, f"door_slat{j}", (.6, 4.86, .1+j*.18), (1.48, .013, .014), "trim")
        for x in (-4.5, 4.5):
            add(world, f"cable_tray{x}", (x, 0, 3.75), (.09, 5, .035), "brushed")
            for j, y in enumerate((-2.8, .5, 3.8)):
                add(world, f"wall_lamp{x}_{j}", (x*.98, y, 3.35), (.045, .5, .055), "lamp")
        # Real scans stay outside the traversable square unless they replace an
        # existing obstacle, so they cannot create invisible collision changes.
        prop(root, assets, "utility_box_01", "back_cabinet", (-3.25, 4.1, 0), (.85, .55, 1.65))
        prop(root, assets, "portable_generator", "back_generator", (2.9, 4.0, 0), (1.15, .68, .9), yaw=-18)
        for x in (-3.6, 3.6):
            add(world, f"floor_line{x}", (x, 0, .002), (.018, 3.7, .001), "safety_paint")
    else:
        # A service bridge over an engineered channel, with approaches that meet
        # the bridge rather than decorative buildings at miniature scale.
        for side in (-1, 1):
            add(world, f"channel_bank{side}", (0, side*3.2, .02), (10, 1.0, .25), "floor_mat")
            add(world, f"retaining_wall{side}", (0, side*2.2, -.3), (10, .15, .55), "brick_mat")
            for j in range(4):
                prop(root, assets, "concrete_road_barrier", f"barrier{side}_{j}",
                     (-5+j*3.3, side*3.0, .27), (2.7, .42, .85))
        prop(root, assets, "utility_box_01", "bridge_switchgear", (4.1, 3.6, .27), (.8, .48, 1.5))
    if kind == "rescue":
        for item in list(world.findall("geom")):
            name = item.get("name", "")
            if not name.startswith("building"):
                continue
            if name.removeprefix("building").isdigit():
                x, y, z = np.fromstring(item.get("pos"), sep=" ")
                sx, sy, sz = np.fromstring(item.get("size"), sep=" ")
                item.set("group", "3")
                prop(root, assets, "utility_box_01", name+"_scan", (x, y, 0), (2*sx, 2*sy, 2*sz))
            else:
                world.remove(item)
    # Neutral-looking materials for the navigation barriers retain their actual
    # topology. Their stepped bases and caps make them constructed partitions.
    if kind in ("icl", "cross"):
        for item in world.findall("geom"):
            if item.get("name", "").startswith(("wall", "boundary")) and "_post" not in item.get("name", ""):
                item.set("material", "navigation_panel")
                item.set("rgba", ".42 .49 .58 1")
    if kind == "arm":
        world.find("geom[@name='workbench']").set("material", "brushed")
        # Matte measurement block avoids confusing the estimator with metallic
        # highlights, while keeping the original physical collision primitive.
        block = world.find("geom[@name='obstacle']")
        block.set("rgba", ".82 .82 .82 1")
