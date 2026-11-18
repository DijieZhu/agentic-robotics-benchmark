"""The MJCF of a site through MjSpec: materials and procedural textures, the floor plane, every primitive as a
named geom in a named static body, lights, viewing cameras, and the instruments (ip_models adapter).

Geom groups: 0 structure and equipment (collision), 1 decor (no collision), 2 ceiling and overhead decor (no
collision; hide it in top-down views), 3 traversal elements (collision). condim 3 everywhere, reflectance 0.
"""

from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from . import textures as T
from .util import quat_from_mat

# name: (rgba, specular, shininess, texture-key or None, texrepeat, texuniform, emission)
MATS = {
    "floor_base": ((0.62, 0.62, 0.6, 1), 0.1, 0.1, "concrete", (0.5, 0.5), True, 0),
    "floor_concrete": ((0.66, 0.66, 0.63, 1), 0.1, 0.1, "concrete", (0.5, 0.5), True, 0),
    "floor_corridor": ((0.70, 0.71, 0.70, 1), 0.2, 0.3, "concrete_light", (0.4, 0.4), True, 0),
    "floor_airlock": ((0.62, 0.66, 0.70, 1), 0.2, 0.3, "concrete_light", (0.5, 0.5), True, 0),
    "floor_epoxy_green": ((0.50, 0.62, 0.54, 1), 0.3, 0.4, "concrete_light", (0.4, 0.4), True, 0),
    "floor_epoxy": ((0.64, 0.66, 0.66, 1), 0.3, 0.4, "concrete_light", (0.4, 0.4), True, 0),
    "floor_lino": ((0.68, 0.64, 0.56, 1), 0.2, 0.3, "concrete_light", (0.5, 0.5), True, 0),
    "bund_floor": ((0.52, 0.56, 0.55, 1), 0.3, 0.4, "concrete_light", (0.5, 0.5), True, 0),
    "wall_paint": ((0.88, 0.88, 0.85, 1), 0.05, 0.1, "blockwork", (0.6, 0.6), True, 0),
    "block_wall": ((0.70, 0.70, 0.68, 1), 0.05, 0.1, "blockwork_grey", (1.0, 1.0), True, 0),
    "shield_concrete": ((0.72, 0.71, 0.68, 1), 0.05, 0.1, "shield", (0.5, 0.5), True, 0),
    "skirting": ((0.32, 0.33, 0.35, 1), 0.1, 0.1, None, None, False, 0),
    "roof": ((0.75, 0.75, 0.74, 1), 0.0, 0.0, None, None, False, 0),
    "door_leaf": ((0.42, 0.50, 0.58, 1), 0.3, 0.3, None, None, False, 0),
    "door_leaf_fire": ((0.66, 0.20, 0.16, 1), 0.3, 0.3, None, None, False, 0),
    "door_leaf_heavy": ((0.38, 0.40, 0.44, 1), 0.3, 0.3, None, None, False, 0),
    "door_frame": ((0.28, 0.30, 0.33, 1), 0.3, 0.3, None, None, False, 0),
    "door_frame_fire": ((0.50, 0.14, 0.12, 1), 0.3, 0.3, None, None, False, 0),
    "glass": ((0.62, 0.74, 0.80, 0.45), 0.6, 0.8, None, None, False, 0),
    "concrete_plinth": ((0.70, 0.69, 0.65, 1), 0.05, 0.1, "concrete", (2, 2), True, 0),
    "steel_dark": ((0.24, 0.25, 0.27, 1), 0.4, 0.4, None, None, False, 0),
    "steel": ((0.62, 0.64, 0.67, 1), 0.6, 0.5, None, None, False, 0),
    "galv": ((0.66, 0.68, 0.68, 1), 0.4, 0.4, None, None, False, 0),
    "steel_plate": ((0.48, 0.49, 0.50, 1), 0.4, 0.4, "checker", (6, 6), True, 0),
    "ramp_steel": ((0.55, 0.56, 0.57, 1), 0.4, 0.4, "checker", (3, 3), True, 0),
    "motor_blue": ((0.16, 0.32, 0.58, 1), 0.4, 0.4, None, None, False, 0),
    "motor_blue_dark": ((0.10, 0.20, 0.36, 1), 0.4, 0.4, None, None, False, 0),
    "pump_casing": ((0.26, 0.40, 0.56, 1), 0.4, 0.4, None, None, False, 0),
    "pump_blue": ((0.20, 0.36, 0.60, 1), 0.4, 0.4, None, None, False, 0),
    "guard_orange": ((0.92, 0.52, 0.10, 1), 0.3, 0.3, None, None, False, 0),
    "tank_grey": ((0.80, 0.81, 0.80, 1), 0.3, 0.3, None, None, False, 0),
    "cabinet": ((0.78, 0.78, 0.74, 1), 0.2, 0.2, None, None, False, 0),
    "cabinet_door": ((0.74, 0.74, 0.70, 1), 0.2, 0.2, None, None, False, 0),
    "cabinet_white": ((0.92, 0.92, 0.90, 1), 0.2, 0.2, None, None, False, 0),
    "panel_grey": ((0.66, 0.68, 0.69, 1), 0.3, 0.3, None, None, False, 0),
    "base_station": ((0.22, 0.27, 0.32, 1), 0.3, 0.3, None, None, False, 0),
    "locker_blue": ((0.36, 0.46, 0.62, 1), 0.3, 0.3, None, None, False, 0),
    "bench_wood": ((0.62, 0.46, 0.30, 1), 0.1, 0.1, None, None, False, 0),
    "lab_bench": ((0.82, 0.82, 0.80, 1), 0.2, 0.2, None, None, False, 0),
    "rack_blue": ((0.18, 0.34, 0.66, 1), 0.3, 0.3, None, None, False, 0),
    "rack_orange": ((0.90, 0.45, 0.10, 1), 0.3, 0.3, None, None, False, 0),
    "carton": ((0.72, 0.56, 0.36, 1), 0.05, 0.1, None, None, False, 0),
    "box_grey": ((0.50, 0.51, 0.52, 1), 0.1, 0.1, None, None, False, 0),
    "box_blue": ((0.30, 0.40, 0.62, 1), 0.1, 0.1, None, None, False, 0),
    "drum_small": ((0.30, 0.52, 0.32, 1), 0.2, 0.2, None, None, False, 0),
    "drum_yellow": ((0.93, 0.75, 0.10, 1), 0.3, 0.3, None, None, False, 0),
    "drum_lid": ((0.30, 0.30, 0.30, 1), 0.3, 0.3, None, None, False, 0),
    "drum_band": ((0.12, 0.12, 0.12, 1), 0.3, 0.3, None, None, False, 0),
    "pallet_wood": ((0.64, 0.51, 0.33, 1), 0.05, 0.1, None, None, False, 0),
    "crate_wood": ((0.60, 0.47, 0.30, 1), 0.05, 0.1, None, None, False, 0),
    "ahu_grey": ((0.72, 0.74, 0.74, 1), 0.3, 0.3, None, None, False, 0),
    "ahu_panel": ((0.62, 0.64, 0.64, 1), 0.3, 0.3, None, None, False, 0),
    "machine_green": ((0.36, 0.52, 0.42, 1), 0.3, 0.3, None, None, False, 0),
    "gas_green": ((0.20, 0.46, 0.26, 1), 0.3, 0.3, None, None, False, 0),
    "red_fire": ((0.80, 0.10, 0.08, 1), 0.3, 0.3, None, None, False, 0),
    "screen": ((0.08, 0.10, 0.13, 1), 0.6, 0.6, None, None, False, 0),
    "rubber_black": ((0.30, 0.30, 0.31, 1), 0.05, 0.05, None, None, False, 0),
    "rubber_blue": ((0.20, 0.30, 0.52, 1), 0.05, 0.05, None, None, False, 0),
    "actuator_yellow": ((0.92, 0.72, 0.10, 1), 0.3, 0.3, None, None, False, 0),
    "tray": ((0.62, 0.64, 0.64, 1), 0.3, 0.3, None, None, False, 0),
    "seam": ((0.40, 0.40, 0.40, 1), 0.1, 0.1, None, None, False, 0),
    "hose_red": ((0.72, 0.15, 0.10, 1), 0.2, 0.2, None, None, False, 0),
    "hose_black": ((0.10, 0.10, 0.11, 1), 0.2, 0.2, None, None, False, 0),
    "cable_protector": ((1, 1, 1, 1), 0.2, 0.2, "protector", (5, 5), True, 0),
    "hazard_stripe": ((1, 1, 1, 1), 0.2, 0.2, "hazard", (5, 5), True, 0),
    "zone_tape": ((1, 1, 1, 1), 0.1, 0.1, "zone_tape", (5, 5), True, 0),
    "walkway_yellow": ((0.92, 0.80, 0.12, 1), 0.2, 0.2, None, None, False, 0),
    "walkway_green": ((0.25, 0.60, 0.32, 1), 0.2, 0.2, None, None, False, 0),
    "pad_paint": ((1, 1, 1, 1), 0.2, 0.2, "pad", (1, 1), False, 0),
    "grating": ((1, 1, 1, 1), 0.3, 0.3, "grating", (4, 4), True, 0),
    "puddle": ((0.25, 0.28, 0.30, 1.0), 0.9, 0.9, None, None, False, 0),
    "lamp_on": ((1.0, 0.97, 0.88, 1), 0.0, 0.0, None, None, False, 1.0),
    "lamp_dim": ((0.85, 0.82, 0.70, 1), 0.0, 0.0, None, None, False, 0.45),
    "led_green": ((0.1, 1.0, 0.2, 1), 0.0, 0.0, None, None, False, 1.0),
    "led_off": ((0.25, 0.05, 0.05, 1), 0.0, 0.0, None, None, False, 0.0),
    "pipe_cw": ((0.27, 0.56, 0.34, 1), 0.4, 0.4, None, None, False, 0),
    "pipe_dw": ((0.20, 0.50, 0.56, 1), 0.4, 0.4, None, None, False, 0),
    "pipe_st": ((0.78, 0.78, 0.78, 1), 0.5, 0.5, None, None, False, 0),
    "pipe_rw": ((0.82, 0.63, 0.20, 1), 0.4, 0.4, None, None, False, 0),
    "pipe_rw_station": ((0.46, 0.30, 0.52, 1), 0.4, 0.4, None, None, False, 0),   # = dress/decor.py's radwaste colour
    "pipe_ia": ((0.52, 0.72, 0.88, 1), 0.4, 0.4, None, None, False, 0),
    "pipe_fw": ((0.76, 0.14, 0.10, 1), 0.4, 0.4, None, None, False, 0),
    "pipe_dr": ((0.22, 0.20, 0.18, 1), 0.3, 0.3, None, None, False, 0),
    "pipe_oil": ((0.52, 0.34, 0.20, 1), 0.4, 0.4, None, None, False, 0),
    "placeholder": ((0.85, 0.15, 0.70, 1), 0.3, 0.3, None, None, False, 0),
    "placeholder_face": ((1, 1, 1, 1), 0.2, 0.2, None, None, False, 0),
    "needle": ((0.85, 0.05, 0.05, 1), 0.2, 0.2, None, None, False, 0),
    "liquid": ((0.20, 0.45, 0.85, 0.85), 0.4, 0.4, None, None, False, 0),
    "lamp_red_on": ((1.0, 0.15, 0.10, 1), 0.2, 0.2, None, None, False, 0.9),
    "lamp_green_on": ((0.15, 1.0, 0.25, 1), 0.2, 0.2, None, None, False, 0.9),
    "lamp_amber_on": ((1.0, 0.70, 0.10, 1), 0.2, 0.2, None, None, False, 0.9),
    "lamp_white_on": ((1.0, 1.0, 0.95, 1), 0.2, 0.2, None, None, False, 0.9),
    "lamp_off": ((0.22, 0.22, 0.20, 1), 0.4, 0.5, None, None, False, 0.0),
}

LIGHT_LEVEL = {"bright": 0.62, "mid": 0.46, "dim": 0.30, "dark": 0.16}


def _tex_image(key, assets):
    fn = os.path.join(assets, "%s.png" % key)
    if os.path.exists(fn):
        return fn
    img = {
        "concrete": lambda: T.concrete((160, 160, 156), 512, 1),
        "concrete_light": lambda: T.concrete((215, 215, 212), 512, 4, var=14, speckle=0.004),
        "blockwork": lambda: T.blockwork((236, 236, 230), 512, 2),
        "blockwork_grey": lambda: T.blockwork((175, 175, 170), 512, 5, rows=5, cols=2),
        "shield": lambda: T.shield_concrete(512, 3),
        "hazard": lambda: T.stripes((238, 192, 18), (28, 28, 28), 256, 64, 45),
        "protector": lambda: T.stripes((238, 192, 18), (28, 28, 28), 256, 128, 0),
        "zone_tape": lambda: T.stripes((238, 200, 20), (190, 30, 160), 256, 64, 45),
        "grating": lambda: T.grating(base=(120, 122, 124), hole=(70, 72, 74)),
        "checker": lambda: T.checker_plate(),
        "pad": lambda: T.pad(512),
    }[key]()
    return T.write_png(img, fn)


class MatBank:
    def __init__(self, spec, assets):
        self.spec = spec
        self.assets = assets
        self.made = set()

    def get(self, name, sign=None) -> str:
        if name in self.made:
            return name
        spec = self.spec
        if sign is not None:
            fn = os.path.join(self.assets, "%s.png" % name.replace("/", "_"))
            st = sign["style"]
            if st == "room":
                img = T.room_sign(sign["text"], *sign["size"])
            elif st == "trefoil":
                img = T.trefoil(sign["size"][0], sign["text"])
            elif st == "notice":
                img = T.notice(sign["size"][0], sign["text"])
            elif st == "tag":
                img = T.tag_plate(sign["text"], *sign["size"])
            elif st == "fiducial":
                img = T.fiducial(sign["id"], sign["text"])
            elif st == "floor_marker":
                img = T.floor_marker(sign["text"])
            elif st == "room_plate":                    # version 2: the plant lettering
                img = T.room_plate(sign["number"], sign["name"])
            elif st == "survey_mark":                   # version 2: the plant lettering
                img = T.survey_mark(sign["text"])
            elif st == "dial":
                img = T.dial_placeholder(*sign["dial"])
            else:
                img = T.room_sign(sign["text"])
            T.write_png(img, fn)
            tn = "tex/" + name
            spec.add_texture(name=tn, type=mujoco.mjtTexture.mjTEXTURE_2D, file=os.path.basename(fn))
            m = spec.add_material(name=name, rgba=[1, 1, 1, 1], specular=0.15, shininess=0.2, reflectance=0.0)
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tn
            self.made.add(name)
            return name
        rgba, sp, sh, tex, rep, uni, em = MATS.get(name, MATS["steel"])
        m = spec.add_material(name=name, rgba=list(rgba), specular=sp, shininess=sh, reflectance=0.0, emission=em)
        if tex:
            fn = _tex_image(tex, self.assets)
            tn = "tex/" + tex
            if tn not in self.made:
                spec.add_texture(name=tn, type=mujoco.mjtTexture.mjTEXTURE_2D, file=os.path.basename(fn))
                self.made.add(tn)
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tn
            m.texrepeat = list(rep)
            m.texuniform = bool(uni)
        self.made.add(name)
        return name


GEOM_T = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
          "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
          "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}


def new_spec(cfg, assets, name="site", extent=40.0, center=(14.0, 0.0, 1.0)):
    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.compiler.degree = False
    spec.compiler.texturedir = assets.rstrip("/") + "/"
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    R = cfg["render"]
    spec.visual.global_.offwidth = int(max(R["topdown_px"][0], R["dollhouse_px"][0], 2560))
    spec.visual.global_.offheight = int(max(R["topdown_px"][1], R["dollhouse_px"][1], 1920))
    spec.visual.global_.fovy = 60.0
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.28, 0.28, 0.28]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.stat.extent = float(extent)
    spec.stat.center = list(center)
    spec.visual.map.znear = 0.01 / extent
    spec.visual.map.zfar = 200.0 / extent
    spec.visual.map.haze = 0.0
    return spec


def build_scene(W, ipw, out_dir, cameras=None):
    """Build the MjSpec of world W (+ the instruments decided in ipw), write assets; returns (spec, info)."""
    cfg = W.cfg
    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    bb = W.lay.bbox()
    cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
    ext = math.hypot(bb[2] - bb[0], bb[3] - bb[1])
    spec = new_spec(cfg, assets, "plant_site", extent=ext, center=(cx, cy, 1.0))
    mats = MatBank(spec, assets)
    wb = spec.worldbody
    # floor plane
    fm = mats.get("floor_base")
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[(bb[2] - bb[0]) / 2 + 4, (bb[3] - bb[1]) / 2 + 4,
                0.1], pos=[cx, cy, 0.0], material=fm, friction=[0.9, 0.005, 0.0001], condim=3, group=0)
    # bodies and geoms
    bodies = {}
    for p in W.P.prims:
        b = bodies.get(p.body)
        if b is None:
            b = wb.add_body(name=p.body)
            bodies[p.body] = b
        sign = W.signs.get(p.mat)
        mname = mats.get(p.mat, sign)
        coll = p.group in (0, 3)
        mu = p.friction if p.friction is not None else 0.8
        size = list(p.size) + [0.0] * (3 - len(p.size))
        g = b.add_geom(name=p.name, type=GEOM_T[p.type], pos=list(p.pos), quat=list(p.quat), size=size,
                       group=int(p.group), material=mname, contype=1 if coll else 0, conaffinity=1 if coll else 0,
                       condim=3, friction=[mu, 0.005, 0.0001])
        if p.priority:
            g.priority = int(p.priority)
        if p.rgba is not None:
            g.rgba = list(p.rgba)
    # lights: two dim directional fills first, then the room lights sorted by distance from the pad
    fills = [((cx - 8, cy - 10, 30.0), (0.25, 0.3, -1.0), 0.32), ((cx + 8, cy + 10, 30.0), (-0.2, -0.25, -1.0), 0.18)]
    for k, (pos, d, lv) in enumerate(fills):
        l = wb.add_light(name="light/fill%d" % k, pos=list(pos), dir=list(d))
        l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        l.diffuse = [lv, lv, lv * 0.97]
        l.specular = [0.05, 0.05, 0.05]
        l.ambient = [0.0, 0.0, 0.0]
        l.castshadow = (k == 0)
    lights = sorted(W.lights, key=lambda L: math.hypot(L["pos"][0], L["pos"][1]))
    for k, L in enumerate(lights):
        lv = LIGHT_LEVEL.get(L["level"], 0.4)
        l = wb.add_light(name="light/%s/%d" % (L["room"], k), pos=list(L["pos"]), dir=[0, 0, -1])
        l.type = mujoco.mjtLightType.mjLIGHT_SPOT
        l.diffuse = [lv, lv * 0.98, lv * 0.92]
        l.specular = [0.1, 0.1, 0.1]
        l.ambient = [0.0, 0.0, 0.0]
        l.cutoff = 75.0 if not spec.compiler.degree else 75.0
        l.exponent = 1.0
        l.attenuation = [0.6, 0.12, 0.03]
        l.castshadow = False
        L["name"] = "light/%s/%d" % (L["room"], k)
    # instruments (bricks or placeholders)
    ip_info = ipw.build(spec, mats, bodies, assets) if ipw is not None else {}
    # instruments are visual targets: no contacts (their valve bodies sit inside our pipe runs, and the map does
    # not count them as obstacles either)
    for g in spec.geoms:
        if g.name.startswith(("ip/", "inst/", "marker/")):
            g.contype = 0
            g.conaffinity = 0
    # (the earlier lever-valve gravity workaround, gravcomp on every instrument body, is gone: the bricks targets set
    # gravcomp on their moving bodies at the source; a development test checks that nothing drifts while stepping)
    # cameras
    cam_info = add_cameras(spec, W, cameras or [])
    return spec, {"assets": assets, "ip_models": ip_info, "cameras": cam_info, "bbox": bb}


def _look_quat(pos, target, up=(0, 0, 1)):
    """Camera quaternion: MuJoCo cameras look along their local -z with +y up."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    if abs(np.dot(f, u)) > 0.999:
        u = np.array([0.0, 1.0, 0.0])
    x = np.cross(f, u)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    R = np.stack([x, y, -f], 1)
    return quat_from_mat(R)


def add_cameras(spec, W, extra):
    bb = W.lay.bbox()
    cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    wb = spec.worldbody
    out = []
    R = W.cfg["render"]
    aspect = R["topdown_px"][0] / R["topdown_px"][1]
    c = wb.add_camera(name="cam/topdown", pos=[cx, cy, 45.0], quat=[1, 0, 0, 0])
    c.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    c.fovy = float(max(h, w / aspect) * 1.04)
    out.append({"name": "cam/topdown", "kind": "topdown"})
    for k, (dx, dy) in enumerate(((-0.55, -1.0), (0.75, 1.0))):
        pos = [cx + dx * w * 0.55, cy + dy * h * 0.95, 0.85 * max(w, h)]
        tgt = [cx + 0.05 * w * dx, cy, 0.0]
        c = wb.add_camera(name="cam/dollhouse%d" % (k + 1), pos=pos, quat=_look_quat(pos, tgt))
        c.fovy = 42.0
        out.append({"name": "cam/dollhouse%d" % (k + 1), "kind": "dollhouse"})
    for e in extra:
        c = wb.add_camera(name=e["name"], pos=list(e["pos"]), quat=_look_quat(e["pos"], e["target"]))
        c.fovy = float(e.get("fovy", 70.0))
        out.append(dict(e))
    return out


def write_xml(spec, out_dir, assets):
    xml = spec.to_xml()
    a = assets.rstrip("/") + "/"
    xml = xml.replace('texturedir="%s"' % a, 'texturedir="assets/"')
    xml = xml.replace(a, "")
    path = os.path.join(out_dir, "scene.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path
