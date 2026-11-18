"""Components with a surface temperature for the thermal camera.

Kinds (face frame: x out, y right, z up; the origin is the hot spot's centre on its surface, except 'puddle', whose
origin is the puddle centre on the floor and whose face is +z):
  bearing_housing  a pump + motor on a baseplate; the drive-end bearing housing is the hot geom
  flange           a pipe spool with a flanged joint; the flange pair is hot (a leaking gasket), the pipe warm
  cabinet          an electrical cabinet; a breaker-sized hot spot on the door, the door around it warm
  puddle           a leaked warm fluid puddle on the floor (thin ellipse), wet-dark
Ground truth: names.state['temps'] = {geom name: temperature in deg C} and names.state['emissivity']; every geom the
table does not list is at the ambient temperature of the thermal sensor. `set_state(model, data, names, t_hot)`
rescales the table (warm geoms keep their fraction of the hot excess) and returns it; it does not touch MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..common import QUAT_Z_TO_Y, Pose, TargetNames, color_material
from . import tag as tagmod

__all__ = ["ThermalSourceSpec", "KINDS", "add_to_spec", "set_state", "temperatures", "public_hint", "read"]

KINDS = ("bearing_housing", "flange", "cabinet", "puddle")


@dataclass
class ThermalSourceSpec:
    name: str = "TS-0001"
    kind: str = "bearing_housing"
    t_hot: float = 85.0              # deg C of the hot geom(s)
    t_ambient: float = 22.0
    size: float = 1.0                # scale factor of the component
    emissivity: float = 0.92         # painted surfaces ~0.9-0.95, bare steel ~0.3, water ~0.96
    tag: tagmod.TagSpec | None = None
    tag_offset: tuple | None = None


def add_to_spec(spec, parent, pose: Pose, ts: ThermalSourceSpec, assets_dir: str) -> TargetNames:
    n = ts.name
    G = mujoco.mjtGeom
    s = ts.size
    body = parent.add_body(name=n, pos=list(pose.pos), quat=list(pose.quat))
    names = TargetNames(name=n, kind="thermal_" + ts.kind, body=n, bodies=[n])
    frac = {}                    # geom -> fraction of the hot excess

    def geom(nm, f=None, **kw):
        body.add_geom(name=nm, **kw)
        names.geoms.append(nm)
        if f is not None:
            frac[nm] = f
        return nm

    if ts.kind == "bearing_housing":
        blue = color_material(spec, "bricks/pump_blue", (0.15, 0.32, 0.60, 1), specular=0.35)
        motor = color_material(spec, "bricks/motor_grey", (0.30, 0.33, 0.36, 1), specular=0.3)
        base = color_material(spec, "bricks/base_dark", (0.20, 0.21, 0.22, 1), specular=0.2)
        # the hot bearing housing: a cylinder about the shaft (+y), its front at x = 0
        r = 0.07 * s
        geom(n + "/bearing", f=1.0, type=G.mjGEOM_CYLINDER, size=[r, 0.05 * s, 0], pos=[-r, 0, 0],
             quat=list(QUAT_Z_TO_Y), material=blue)
        geom(n + "/casing", f=0.45, type=G.mjGEOM_CYLINDER, size=[0.15 * s, 0.07 * s, 0], pos=[-r, -0.15 * s, 0],
             quat=list(QUAT_Z_TO_Y), material=blue)
        geom(n + "/shaft", f=0.6, type=G.mjGEOM_CYLINDER, size=[0.02 * s, 0.09 * s, 0], pos=[-r, 0.13 * s, 0],
             quat=list(QUAT_Z_TO_Y), material=motor)
        geom(n + "/motor", f=0.35, type=G.mjGEOM_CYLINDER, size=[0.11 * s, 0.18 * s, 0], pos=[-r, 0.40 * s, 0],
             quat=list(QUAT_Z_TO_Y), material=motor)
        geom(n + "/base", f=0.08, type=G.mjGEOM_BOX, size=[0.16 * s, 0.45 * s, 0.03 * s],
             pos=[-r, 0.1 * s, -0.17 * s], material=base)
        names.face_geoms = [n + "/bearing"]
    elif ts.kind == "flange":
        pipe = color_material(spec, "bricks/pipe_grey", (0.58, 0.60, 0.60, 1), specular=0.35, shininess=0.4)
        rp, rf = 0.057 * s, 0.10 * s
        geom(n + "/flange_a", f=1.0, type=G.mjGEOM_CYLINDER, size=[rf, 0.012 * s, 0], pos=[-rf, -0.013 * s, 0],
             quat=list(QUAT_Z_TO_Y), material=pipe)
        geom(n + "/flange_b", f=1.0, type=G.mjGEOM_CYLINDER, size=[rf, 0.012 * s, 0], pos=[-rf, 0.013 * s, 0],
             quat=list(QUAT_Z_TO_Y), material=pipe)
        for k, f in enumerate((0.7, 0.45, 0.25)):
            for sg in (-1, 1):
                y0 = sg * (0.025 + 0.15 * k) * s
                y1 = sg * (0.025 + 0.15 * (k + 1)) * s
                geom(n + "/pipe%+d_%d" % (sg, k), f=f, type=G.mjGEOM_CYLINDER, size=[rp, abs(y1 - y0) / 2, 0],
                     pos=[-rf, (y0 + y1) / 2, 0], quat=list(QUAT_Z_TO_Y), material=pipe)
        names.face_geoms = [n + "/flange_a", n + "/flange_b"]
    elif ts.kind == "cabinet":
        grey = color_material(spec, "bricks/cabinet_grey", (0.76, 0.77, 0.75, 1), specular=0.15)
        dark = color_material(spec, "bricks/cabinet_dark", (0.25, 0.26, 0.27, 1), specular=0.2)
        W, H, D = 0.6 * s, 1.2 * s, 0.35 * s
        # origin = hot spot on the door; the door's centre is 0.25 s below and left of it
        geom(n + "/cabinet", f=0.05, type=G.mjGEOM_BOX, size=[D / 2, W / 2, H / 2],
             pos=[-D / 2 - 0.003, -0.1 * s, -0.2 * s], material=grey)
        geom(n + "/door", f=0.25, type=G.mjGEOM_BOX, size=[0.0015, W / 2 - 0.01, H / 2 - 0.01],
             pos=[-0.0015, -0.1 * s, -0.2 * s], material=grey)
        geom(n + "/hotspot", f=1.0, type=G.mjGEOM_BOX, size=[0.0008, 0.05 * s, 0.07 * s], pos=[0.0001, 0, 0],
             material=grey)
        geom(n + "/handle", f=0.05, type=G.mjGEOM_BOX, size=[0.012, 0.01, 0.06 * s],
             pos=[0.012, -0.1 * s + W / 2 - 0.05, -0.2 * s], material=dark)
        names.face_geoms = [n + "/hotspot"]
    elif ts.kind == "puddle":
        # a wet, slightly darker film over the floor (semi-transparent so the floor shows through)
        wet = color_material(spec, "bricks/puddle_wet", (0.20, 0.19, 0.14, 0.55), specular=0.9, shininess=0.95)
        # origin on the floor; the puddle's face is +z of the pose frame (use Pose with the default orientation)
        geom(n + "/puddle", f=1.0, type=G.mjGEOM_ELLIPSOID, size=[0.35 * s, 0.22 * s, 0.002], pos=[0, 0, 0.0005],
             material=wet, contype=0, conaffinity=0, group=1)
        geom(n + "/puddle_edge", f=0.5, type=G.mjGEOM_ELLIPSOID, size=[0.42 * s, 0.28 * s, 0.0012],
             pos=[0.03 * s, 0.01 * s, 0.0003], material=wet, contype=0, conaffinity=0, group=1)
        names.face_geoms = [n + "/puddle"]
    else:
        raise ValueError("unknown thermal source kind %r" % ts.kind)
    names.state = {"spec": ts, "frac": frac, "temps": {}, "emissivity": {g: ts.emissivity for g in frac}}
    set_state(None, None, names, ts.t_hot)
    if ts.tag is not None:
        tw, th = ts.tag.plate_size_m()
        ty, tz = ts.tag_offset if ts.tag_offset is not None else (0.0, -0.25 * s - th / 2)
        names.tag = tagmod.add_to_spec(spec, body, Pose((0.002, ty, tz)), ts.tag, assets_dir, name=n + "/tag")
    return names


def set_state(model, data, names: TargetNames, t_hot: float):
    """Set the hot temperature (deg C); returns the per-geom table {geom: deg C}."""
    ts = names.state["spec"]
    ts.t_hot = float(t_hot)
    names.state["temps"] = {g: ts.t_ambient + f * (ts.t_hot - ts.t_ambient) for g, f in names.state["frac"].items()}
    return names.state["temps"]


def temperatures(names: TargetNames) -> dict:
    return dict(names.state["temps"])


def public_hint(ts: ThermalSourceSpec) -> dict:
    return {"type": "thermal", "component": ts.kind, "emissivity": ts.emissivity}


def read(thermal_c: np.ndarray, hint: dict):
    """Reference reader on a radiometric image (deg C): the hottest compact region's peak temperature, emissivity
    corrected with the hint's emissivity (grey-body, reflected = ambient). Returns (deg C, confidence, debug)."""
    from ..sensors.thermal import emissivity_correct, hottest_region
    reg = hottest_region(thermal_c)
    if reg is None:
        return None, 0.0, {}
    eps = float(hint.get("emissivity", 1.0))
    amb = float(hint.get("ambient_c", np.median(thermal_c)))
    t = emissivity_correct(reg["t_peak"], eps, amb)
    conf = float(np.clip(reg["n_px"] / 9.0, 0, 1))
    reg["t_corrected"] = t
    return float(t), conf, reg
