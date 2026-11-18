"""Convenience wiring: the reference vehicle's sensor suite + navigation firmware, driven on the simulation clock.

    world = SensorWorld(water=water.COASTAL, ambient=water.DARK, column=WaterColumn(10, 34), magnetic=MagneticWorld())
    suite = SensorSuite(m, d, world, np.random.default_rng(seed))
    every physics step:   new = suite.step(t)            # dict: name -> list of new readings (incl. 'nav')
    on demand:            frame = suite.capture(t)       # camera frame (or None if faster than max fps)

Determinism: one parent Generator; each sensor gets its own child stream for noise and one for its unit-specific
hidden parameters (biases, scale errors), spawned in a fixed order, so adding camera frames never changes the DVL
noise and the same seed reproduces every reading bit for bit.

Public vs hidden: `public_specs()` returns the datasheet-level tables (what the agent's manual may say); the
*Hidden dataclasses and SensorWorld are grader-side only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import mujoco

from . import water as W
from .acoustics import WaterColumn, SEAWATER_TEMPERATE
from .materials import MaterialMap
from .camera import UnderwaterCamera, CameraConfig, public_spec as camera_public_spec
from .sonar import (ScanningSonar, ScanningSonarConfig, MultibeamSonar, MultibeamSonarConfig,
                    scanning_sonar_public_spec, multibeam_public_spec)
from .dvl import DVL, DVLConfig, DVLHidden, dvl_public_spec
from .depth import DepthSensor, DepthConfig, DepthHidden, depth_public_spec
from .ahrs import AHRS, AHRSConfig, AHRSHidden, MagneticWorld, ahrs_public_spec
from .usbl import USBL, USBLConfig, USBLHidden, usbl_public_spec
from .altimeter import Altimeter, AltimeterConfig, altimeter_public_spec
from ._util import dataclass_to_dict


def _reference_meta(m: mujoco.MjModel) -> dict:
    """The 'reference' block of the uwkit model metadata (empty for models not built from the digest)."""
    try:
        from ..dynamics.hydro import read_uwkit_meta
        return read_uwkit_meta(m).get("reference") or {}
    except (KeyError, ImportError):
        return {}


@dataclass
class SensorWorld:
    """Environment of the site as the sensors see it (water optics, light, water column, magnetics, materials)."""
    water: W.WaterOptics = W.COASTAL
    ambient: W.AmbientLight = W.DARK
    column: WaterColumn = SEAWATER_TEMPERATE
    magnetic: MagneticWorld = field(default_factory=MagneticWorld)
    material_overrides: dict = field(default_factory=dict)
    usbl_transceiver: tuple = (0.0, 0.0, -1.0)
    water_velocity: tuple = (0.0, 0.0, 0.0)


@dataclass
class SuiteConfig:
    camera: CameraConfig = field(default_factory=lambda: CameraConfig(hide_geomgroups=(3,)))
    scanning_sonar: Optional[ScanningSonarConfig] = field(default_factory=ScanningSonarConfig)
    multibeam: Optional[MultibeamSonarConfig] = None          # optional payload
    dvl: DVLConfig = field(default_factory=DVLConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    ahrs: AHRSConfig = field(default_factory=AHRSConfig)
    usbl: Optional[USBLConfig] = field(default_factory=USBLConfig)
    altimeter: Optional[AltimeterConfig] = None              # optional payload
    acoustic_geomgroup: Sequence[int] = (1, 1, 1, 0, 1, 1)  # vehicle geoms in group 3 are invisible to acoustics
    with_camera: bool = True
    sample_unit_params: bool = True
    exclude_bodies: dict = field(default_factory=dict)       # sensor -> body name its rays ignore (default: root)
    # 0.3.0: what the acoustic rays see of the vehicle itself (cost per sonar ping on this host, load ~90, 20 m range,
    # seabed 4 m below: 2.48 / 2.37 / 1.15 ms):
    #   'visual'    the visual meshes (groups 1, 2): exact self-occlusion (default)
    #   'collision' the collision pieces (group 3; a few mm fatter; needs collision='mesh')
    #   'none'      the vehicle is invisible to the acoustic rays: for the reference vehicle the sonar's self-returns all
    #               lie inside 0.5 m (the ring-down zone) and the DVL cone is clear, so beyond 0.5 m nothing changes
    # 'collision' and 'none' assume the task's world geometry is not in groups 1-3 (world = group 0 by convention).
    acoustic_self: str = "visual"

    @classmethod
    def for_vehicle(cls, m: mujoco.MjModel, **overrides) -> "SuiteConfig":
        """Configuration matched to a model built by uwkit.models.build_vehicle (reference vehicle): camera port,
        FOV, tilt range / rate and hidden geom groups from the model metadata; each acoustic sensor ignores only
        its own housing body, so the rest of the vehicle can occlude it (self-occlusion is physical)."""
        from ..dynamics.hydro import read_uwkit_meta
        meta = read_uwkit_meta(m)
        ref = meta.get("reference") or {}
        ct = meta.get("camera_tilt") or {}
        cam = CameraConfig(hide_geomgroups=(1, 3) if ref else (3,))
        if ct:
            lo, hi = ct.get("range_rad", (-math.pi / 2, math.pi / 2))
            cam = replace(cam, tilt_min_deg=math.degrees(lo), tilt_max_deg=math.degrees(hi),
                          tilt_rate_deg_s=math.degrees(ct.get("rate_rad_s", math.radians(60.0))))
            if ct.get("hfov_deg"):
                cam = replace(cam, hfov_air_deg=float(ct["hfov_deg"]))
            port = ct.get("port") or {}
            if port.get("type") == "dome":
                cam = replace(cam, port="dome", port_clearance_m=float(port["clearance_m"]))
        ex = {}
        for sensor, body in (ref.get("acoustic_self_bodies") or {}).items():
            if sensor == "sonar":
                ex["scanning_sonar"] = body
                ex["multibeam"] = body
            else:
                ex[sensor] = body
        cfg = cls(camera=cam, exclude_bodies=ex)
        return replace(cfg, **overrides) if overrides else cfg


class SensorSuite:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, world: SensorWorld, rng: np.random.Generator,
                 config: Optional[SuiteConfig] = None, t0: float = 0.0, renderer: Optional[mujoco.Renderer] = None):
        from ..nav.dead_reckoning import DeadReckoning
        self.m, self.d = m, d
        self.world = world
        meta_ref = _reference_meta(m)
        self.cfg = config or (SuiteConfig.for_vehicle(m) if meta_ref else SuiteConfig())
        c = self.cfg
        if not c.exclude_bodies and meta_ref:
            # reference vehicle: each acoustic sensor ignores only its own housing (an explicit config without
            # exclude_bodies would otherwise let the housing block its own beams)
            c = replace(c, exclude_bodies=SuiteConfig.for_vehicle(m).exclude_bodies)
            self.cfg = c
        names = ["camera", "scanning_sonar", "multibeam", "dvl", "depth", "ahrs", "usbl", "altimeter"]
        kids = rng.spawn(2 * len(names))
        noise = {n: kids[2 * i] for i, n in enumerate(names)}
        unit = {n: kids[2 * i + 1] for i, n in enumerate(names)}
        mat = MaterialMap(m, overrides=world.material_overrides)
        gg = list(c.acoustic_geomgroup)
        if c.acoustic_self == "collision":
            if meta_ref and meta_ref.get("collision") != "mesh":
                raise ValueError("acoustic_self='collision' needs a vehicle built with collision='mesh' (the merged "
                                 "hulls put the sonar / DVL housings on the hull body, they would occlude their own beams)")
            gg[1], gg[2], gg[3] = 0, 0, 1
        elif c.acoustic_self == "none":
            gg[1], gg[2], gg[3] = 0, 0, 0
        elif c.acoustic_self != "visual":
            raise ValueError("acoustic_self must be 'visual', 'collision' or 'none'")
        smp = c.sample_unit_params
        exb = {k: m.body(v).id for k, v in c.exclude_bodies.items()}
        self.camera = (UnderwaterCamera(m, world.water, noise["camera"], ambient=world.ambient, config=c.camera,
                                        water_velocity=world.water_velocity, renderer=renderer)
                       if c.with_camera else None)
        self.sonar = (ScanningSonar(m, mat, noise["scanning_sonar"], world.column, config=c.scanning_sonar,
                                    geomgroup=gg, exclude_body=exb.get("scanning_sonar"), t0=t0)
                      if c.scanning_sonar is not None else None)
        self.multibeam = (MultibeamSonar(m, mat, noise["multibeam"], world.column, config=c.multibeam, geomgroup=gg,
                                         exclude_body=exb.get("multibeam"), t0=t0) if c.multibeam is not None else None)
        self.dvl = DVL(m, mat, noise["dvl"], world.column, config=c.dvl,
                       hidden=DVLHidden.sample(unit["dvl"]) if smp else None, geomgroup=gg,
                       exclude_body=exb.get("dvl"), t0=t0)
        self.depth = DepthSensor(m, noise["depth"], world.column, config=c.depth,
                                 hidden=DepthHidden.sample(unit["depth"]) if smp else None, t0=t0)
        self.ahrs = AHRS(m, noise["ahrs"], world.magnetic, config=c.ahrs,
                         hidden=AHRSHidden.sample(unit["ahrs"]) if smp else None, t0=t0)
        if c.usbl is not None:
            ucfg = replace(c.usbl, transceiver_pos=tuple(world.usbl_transceiver))
            self.usbl = USBL(m, noise["usbl"], world.column, config=ucfg,
                             hidden=USBLHidden.sample(unit["usbl"]) if smp else None, geomgroup=gg, t0=t0)
        else:
            self.usbl = None
        self.altimeter = (Altimeter(m, mat, noise["altimeter"], world.column, config=c.altimeter, geomgroup=gg,
                                    t0=t0) if c.altimeter is not None else None)
        self.nav = DeadReckoning(m, d, t0=t0)

    def step(self, t: float) -> dict:
        m, d = self.m, self.d
        out = {}
        out["dvl"] = self.dvl.update(t, m, d)
        out["depth"] = self.depth.update(t, d)
        out["ahrs"] = self.ahrs.update(t, d)
        if self.sonar is not None:
            out["scanning_sonar"] = self.sonar.update(t, d)
        if self.multibeam is not None:
            fr = self.multibeam.update(t, d)
            out["multibeam"] = [fr] if fr is not None else []
        if self.usbl is not None:
            out["usbl"] = self.usbl.update(t, d)
        if self.altimeter is not None:
            out["altimeter"] = self.altimeter.update(t, d)
        out["nav"] = self.nav.update(t, out["dvl"], out["ahrs"], out["depth"])
        return out

    def capture(self, t: float, **kw):
        return None if self.camera is None else self.camera.capture(self.d, t, **kw)

    def public_specs(self) -> dict:
        c = self.cfg
        out = {"camera": [dataclass_to_dict(x) for x in camera_public_spec(c.camera)],
               "dvl": [dataclass_to_dict(x) for x in dvl_public_spec(c.dvl)],
               "depth": [dataclass_to_dict(x) for x in depth_public_spec(c.depth)],
               "ahrs": [dataclass_to_dict(x) for x in ahrs_public_spec(c.ahrs)]}
        if c.scanning_sonar is not None:
            out["scanning_sonar"] = [dataclass_to_dict(x) for x in scanning_sonar_public_spec(c.scanning_sonar)]
        if c.multibeam is not None:
            out["multibeam"] = [dataclass_to_dict(x) for x in multibeam_public_spec(c.multibeam)]
        if c.usbl is not None:
            out["usbl"] = [dataclass_to_dict(x) for x in usbl_public_spec(c.usbl)]
        if c.altimeter is not None:
            out["altimeter"] = [dataclass_to_dict(x) for x in altimeter_public_spec(c.altimeter)]
        return out
