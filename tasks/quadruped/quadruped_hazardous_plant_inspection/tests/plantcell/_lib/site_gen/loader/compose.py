"""Compose a stored site with the robot: the site's MJCF (from the package, textures from the package and the
shared directory) with the robot's MJCF attached at the deployment pad, the robot's physics options on the final
spec, every instrument in its true state.

    spec, model, info = compose(pkg_or_path, robot_xml=..., start_pose=(x, y, yaw))
    data = info["data"]      # qpos: the robot's 'stand' key at the start pose, every instrument in its true state
"""

from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from .package import open_site
from .states import apply_truth_states, instrument_records

DEFAULT_ROBOT_XML = os.environ.get("SITE_GEN_ROBOT_XML")      # no built-in path: the caller names the robot's MJCF
STANDIN = {"footprint": (0.55, 0.33), "height": 0.12, "mass": 3.0}


def quat_yaw(yaw: float) -> list:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _copy_options(src, dst):
    """Every settable field of an MjSpec option struct from src to dst."""
    for k in dir(src):
        if k.startswith("_"):
            continue
        try:
            v = getattr(src, k)
        except Exception:  # noqa: BLE001
            continue
        if callable(v):
            continue
        try:
            setattr(dst, k, v.copy() if hasattr(v, "copy") else v)
        except Exception:  # noqa: BLE001  (read-only fields)
            pass


def standin_robot(footprint=None):
    """A free box (3 kg) of the robot's footprint, for tests without the robot's MJCF."""
    s = mujoco.MjSpec()
    L, Wd = footprint or STANDIN["footprint"]
    H = float(STANDIN["height"])
    m = float(STANDIN["mass"])
    b = s.worldbody.add_body(name="base", pos=[0, 0, 0])
    b.explicitinertial = True
    b.mass = m
    b.ipos = [0.0, 0.0, 0.0]
    b.iquat = [1.0, 0.0, 0.0, 0.0]
    b.inertia = [m / 12 * (Wd ** 2 + H ** 2), m / 12 * (L ** 2 + H ** 2), m / 12 * (L ** 2 + Wd ** 2)]
    b.add_freejoint(name="root")
    b.add_geom(name="body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, Wd / 2, H / 2], rgba=[0.95, 0.55, 0.1, 1],
               friction=[0.9, 0.005, 0.0001], condim=3, group=0)
    b.add_site(name="head", pos=[L / 2 - 0.05, 0, H / 2 + 0.02], size=[0.01, 0.01, 0.01])
    return s


def compose(site, robot_xml=None, start_pose=None, prefix="robot/", use_robot_options=True, return_spec=True,
            remap_robot_visual_group=True, shared_dir=None):
    """site: a SitePackage, an archive path or a site folder. robot_xml: the robot's MJCF (default
    DEFAULT_ROBOT_XML when it exists, else a stand-in box). Returns (spec, model, info): info holds the robot's
    root body / free joint names, its qpos address, the start qpos ('stand' key when the robot has one), the count
    of instruments set to their true state, the options used and `data`."""
    pkg = open_site(site, shared_dir=shared_dir)
    spec = pkg.spec()
    x, y, yaw = start_pose if start_pose is not None else (0.0, 0.0, 0.0)
    if robot_xml is None and DEFAULT_ROBOT_XML and os.path.exists(DEFAULT_ROBOT_XML):
        robot_xml = DEFAULT_ROBOT_XML
    info = {"robot_xml": robot_xml or "stand-in box", "prefix": prefix, "package": pkg.source}
    if robot_xml:
        rspec = mujoco.MjSpec.from_file(robot_xml)
        root = rspec.worldbody.first_body()
        z0 = float(root.pos[2])
        opts = {k: getattr(rspec.option, k) for k in ("timestep", "integrator", "cone", "solver", "iterations",
                                                      "ls_iterations", "impratio", "noslip_iterations")}
        keys = {k.name: (np.array(k.qpos, float), np.array(k.ctrl, float)) for k in rspec.keys}
    else:
        rspec = standin_robot(pkg.site.get("robot_footprint"))
        root = rspec.worldbody.first_body()
        z0 = float(STANDIN["height"]) / 2 + 0.0005
        opts = {"timestep": 0.002, "integrator": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                "cone": mujoco.mjtCone.mjCONE_ELLIPTIC, "impratio": 10.0}
        keys = {}
    rname = root.name
    if use_robot_options:
        # the robot's options on the site's spec BEFORE the attach: MuJoCo compares the two and, when they differ,
        # warns and writes MUJOCO_LOG.TXT into the current folder (2026-11-14)
        _copy_options(rspec.option, spec.option)
        for k, v in opts.items():
            setattr(spec.option, k, v)
    frame = spec.worldbody.add_frame(pos=[x, y, z0], quat=quat_yaw(yaw))
    root.pos = [0.0, 0.0, 0.0]
    body = frame.attach_body(root, prefix, "")
    if use_robot_options:                       # the attach drops the child's options: set them on the final spec
        for k, v in opts.items():
            setattr(spec.option, k, v)
    if remap_robot_visual_group:
        for g in spec.geoms:
            if g.name.startswith(prefix) and g.group == 2 and g.contype == 0 and g.conaffinity == 0:
                g.group = 1
    from ..dress import maybe_dress                   # version 2: the site's look; SITE_GEN_DRESS=0 turns it off
    info["dress"] = maybe_dress(spec, pkg.site)       # appends after the robot: every site and robot geom keeps its id
    model = spec.compile()
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "root")
    qadr = int(model.jnt_qposadr[jid])
    q0 = model.qpos0.copy()
    if "stand" in keys:
        kq = keys["stand"][0]
        q0[qadr:qadr + len(kq)] = kq
    q0[qadr:qadr + 3] = [x, y, z0 if "stand" not in keys else float(keys["stand"][0][2])]
    q0[qadr + 3:qadr + 7] = quat_yaw(yaw)
    data.qpos[:] = q0
    n_inst = apply_truth_states(model, data, instrument_records(pkg.site))
    mujoco.mj_forward(model, data)
    info.update({"root_body": prefix + rname, "root_joint": prefix + "root",
                 "qpos_adr": qadr, "qpos0": data.qpos.copy(), "instruments_set": n_inst,
                 "options": {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in opts.items()},
                 "data": data, "site": pkg.site, "prior": pkg.prior})
    del body
    return (spec, model, info) if return_spec else (model, info)


def compile_scene(site, shared_dir=None):
    """The site alone (no robot), compiled, every instrument in its true state: (model, data, package)."""
    pkg = open_site(site, shared_dir=shared_dir)
    m, d = pkg.model()
    apply_truth_states(m, d, instrument_records(pkg.site))
    return m, d, pkg
