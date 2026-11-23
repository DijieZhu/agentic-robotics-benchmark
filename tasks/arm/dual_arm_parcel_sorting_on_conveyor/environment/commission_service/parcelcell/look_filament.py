"""The ``filament`` look of the cell's cameras: MuJoCo's Filament renderer (physically based, on Vulkan; Mesa's
lavapipe on the CPU in the task image) with light distilled from the station's path-traced look.

    look = FilamentLook(model, cameras, world=world, options=cfg["look"])
    rgb = look.rgb(data, CameraView("head_cam", 640, 400))     # uint8 H x W x 3, rows top-down, no sensor noise
    look.close()                                                # frees the shift's renderer resources

What is drawn (the physics model is never touched):
  * the station's surroundings, the conveyor (belt, rails, lip, frame, drive, guards), the station's hardware, the
    hall as far as the cameras can see it: the render bundle's static set (``static.npz``), drawn only; the
    belt, the rails and the frame lie on the cell's collision boxes (within a millimetre);
  * the robot and its grippers with the station's materials (``robot.npz``: each drawn part on its MuJoCo body, posed
    from the shift's state at every frame), the head camera's housing; the wrist cameras' housings from the cell's
    own boxes;
  * every parcel of the shift with its own mesh, maps and waybill (the parcel visuals the world was built with);
  * lit by the look's image light (``ibl.ktx``: the hall seen from the work area, everything that moves taken out)
    and its spot lamps (where the station's luminaires are), with shadows for what moves; ambient occlusion,
    exposure and tone mapping from the look file.
Per shift, a render twin is compiled from a copy of the world's MjSpec: the same bodies, joints and cameras; every
geom the cell draws or collides with is taken out except the parcels' visuals and the wrist cameras' housings, so a
frame copies the state's positions into the twin and runs forward kinematics only. The camera is the cell's own:
its position and axes in the state (the physics model's camera), its configured vertical field of view, the
principal point at the image centre: the same pinhole as the classic renderer. Each image is drawn at ``ssaa`` times
the camera's size and averaged over ssaa x ssaa blocks in linear light (same pixel centres, same intrinsics).
Depth stays the classic renderer's (look.make_depth).

The look file's keys: render (ssaa, lavapipe_threads, clear_color, near_m, far_m, timeout_s for the start and a
shift's first frame, frame_timeout_s for any later frame, undither, shadow_type pcf | vsm | dpcf | pcss | pcfd), view
(filament.* fields MuJoCo reads: numeric filament.ao.enabled, filament.shadows.map_size; text
filament.cg.tone_mapping), exposure (default: the factor of every light; per_camera: each camera's own factor), ibl
(file, intensity), lamps (spot lamps: name, pos, dir, candela, cutoff_deg, softness, shadow, shadow_map, bulb_m, color,
range_m), static (albedo_gain, emissive_scale, classes belt | conveyor | station | floor | hall | luminaire: gain,
roughness_scale, emissive_scale, cast_shadows, hide), robot (albedo_gain, classes: gain, roughness, metallic), parcels
(gain, roughness, metallic, specular), housings (rgba, roughness, metallic). Gains are linear, one number or r, g, b.

Files: the look file (``look_config/look.json`` beside this module, or options["look_file"]) and the render bundle
(options["render_dir"], else $PARCELCELL_RENDER_DIR, else /opt/task-assets/render): manifest.json (sha256 of every
file, checked once per process), static.npz / static.json, robot.npz / robot.json, ibl.ktx. Every failure is a
``RenderError``: an error of the cell (the camera call answers camera_unavailable and the frame is not charged).
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import mujoco
import numpy as np

from .filament_engine import Engine, RenderError, engine_info

__all__ = ["FilamentLook", "load_look", "check_bundle", "render_dir", "LOOK_FILE", "downsample", "build_twin"]

HERE = os.path.dirname(os.path.abspath(__file__))
LOOK_FILE = os.path.join(HERE, "look_config", "look.json")
DEFAULT_RENDER_DIR = "/opt/task-assets/render"
BUNDLE_FILES = ("manifest.json", "static.npz", "static.json", "robot.npz", "robot.json")

_LIN8 = np.where(np.arange(256) / 255.0 <= 0.04045, (np.arange(256) / 255.0) / 12.92,
                 ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4).astype(np.float32)
_x = np.arange(4096) / 4095.0
_SRGB12 = np.round(255.0 * np.where(_x <= 0.0031308, _x * 12.92, 1.055 * np.power(_x, 1.0 / 2.4) - 0.055)
                   ).astype(np.uint8)


def downsample(img, s):
    """uint8 (H*s) x (W*s) x 3 -> H x W x 3: the mean of each s x s block in linear light."""
    if s == 1:
        return img
    h, w = img.shape[0] // s, img.shape[1] // s
    lin = _LIN8[img].reshape(h, s, w, s, 3).mean(axis=(1, 3))
    return _SRGB12[np.clip(np.round(lin * 4095.0), 0, 4095).astype(np.int32)]


def render_dir(options=None):
    o = options or {}
    return os.path.abspath(o.get("render_dir") or os.environ.get("PARCELCELL_RENDER_DIR") or DEFAULT_RENDER_DIR)


_LOOKS = {}


def load_look(path=None):
    """The look file: ``path`` (relative paths are taken from this package's folder), else look_config/look.json."""
    p = path or LOOK_FILE
    if not os.path.isabs(p):
        p = os.path.join(HERE, p)
    p = os.path.abspath(p)
    if p not in _LOOKS:
        try:
            with open(p) as fh:
                _LOOKS[p] = json.load(fh)
        except (OSError, ValueError) as exc:
            raise RenderError("renderer: cannot read the look file %s: %s" % (p, exc))
    return _LOOKS[p]


_CHECKED = {}


def check_bundle(rdir, look=None):
    """The bundle's manifest, every file checked against its sha256 (once per process and folder)."""
    if rdir in _CHECKED:
        return _CHECKED[rdir]
    missing = [f for f in BUNDLE_FILES if not os.path.isfile(os.path.join(rdir, f))]
    ibl = (look or {}).get("ibl", {}).get("file", "ibl.ktx")
    if not os.path.isfile(os.path.join(rdir, ibl)):
        missing.append(ibl)
    if missing:
        raise RenderError("renderer: the render bundle in %s lacks %s" % (rdir, ", ".join(missing)))
    try:
        with open(os.path.join(rdir, "manifest.json")) as fh:
            man = json.load(fh)
        for rel, digest in man["files"].items():
            h = hashlib.sha256()
            with open(os.path.join(rdir, rel), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                raise RenderError("renderer: render bundle file changed: %s" % rel)
    except RenderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RenderError("renderer: render bundle unreadable: %s: %s" % (type(exc).__name__, exc))
    _CHECKED[rdir] = man
    return man


# ------------------------------------------------------------------------------------------------ the render twin
def _is_robot_body(name):
    return not (name == "world" or name.startswith("parcel") or name.startswith("belt") or name.endswith("_body"))


def build_twin(world, look):
    """The shift's render twin: (MjModel, {robot body name: id}, record). Kept: the bodies, joints and cameras; the
    parcels' visual geoms (their materials made physically based, the look's parcel gain in their maps); the wrist
    cameras' housing boxes (drawn, the look's housing material). Taken out: every other geom, every site, light,
    unused mesh, texture and material."""
    spec = world.spec.copy()
    m0 = world.model
    pl = look.get("parcels", {})
    hl = look.get("housings", {"rgba": [0.05, 0.05, 0.055, 1.0], "roughness": 0.45, "metallic": 1.0})
    rec = {"kept_geoms": 0, "deleted_geoms": 0}
    for li in list(spec.lights):
        spec.delete(li)
    spec.visual.headlight.active = 0
    for x in list(spec.sensors):                         # (they name sites; the twin needs neither)
        spec.delete(x)
    for x in list(spec.actuators):
        spec.delete(x)
    for s in list(spec.sites):
        spec.delete(s)
    kept_mesh, kept_mat = set(), set()
    hmat = spec.add_material(name="look_housing")
    hmat.rgba = list(hl["rgba"])
    hmat.roughness, hmat.metallic, hmat.specular, hmat.reflectance = float(hl["roughness"]), float(hl["metallic"]), 0.5, 0.0
    sgs = list(spec.geoms)
    if len(sgs) != m0.ngeom:
        raise RenderError("renderer: the world's spec and model disagree (%d / %d geoms)" % (len(sgs), m0.ngeom))
    parcel_mats = set()
    for g, sg in enumerate(sgs):
        nm = m0.geom(g).name or ""
        body = m0.body(int(m0.geom_bodyid[g])).name
        if nm.endswith("_wrist_cam_housing"):
            sg.group, sg.contype, sg.conaffinity = 0, 0, 0
            sg.material = "look_housing"
            sg.rgba = list(hl["rgba"])
            kept_mat.add("look_housing")
            rec["kept_geoms"] += 1
            continue
        if body.startswith("parcel") and int(m0.geom_group[g]) == 0 and not int(m0.geom_contype[g]):
            sg.group = 0
            if sg.meshname:
                kept_mesh.add(sg.meshname)
            if sg.material:
                kept_mat.add(sg.material)
                parcel_mats.add(sg.material)
            rec["kept_geoms"] += 1
            continue
        spec.delete(sg)
        rec["deleted_geoms"] += 1
    # the bodies keep the physics model's mass and inertia explicitly (their geoms are gone; kinematics only)
    sbs = list(spec.bodies)
    if len(sbs) != m0.nbody:
        raise RenderError("renderer: the world's spec and model disagree (%d / %d bodies)" % (len(sbs), m0.nbody))
    for b, sb in enumerate(sbs):
        if b == 0:
            continue
        sb.explicitinertial = True
        sb.fullinertia = [float("nan")] * 6
        sb.mass = float(max(m0.body_mass[b], 1e-3))
        sb.inertia = [float(max(x, 1e-6)) for x in m0.body_inertia[b]]
        sb.ipos = [float(x) for x in m0.body_ipos[b]]
        sb.iquat = [float(x) for x in m0.body_iquat[b]]
    kept_tex = set()
    for mt in list(spec.materials):
        if mt.name not in kept_mat:
            spec.delete(mt)
            continue
        for t in mt.textures:
            if t:
                kept_tex.add(t)
    for me in list(spec.meshes):
        if me.name not in kept_mesh:
            spec.delete(me)
    gain = np.broadcast_to(np.asarray(pl.get("gain", 1.0), float), (3,))
    from .filament_engine import gain_srgb8
    for t in list(spec.textures):
        if t.name not in kept_tex:
            spec.delete(t)
            continue
        # base colour maps: sRGB, the parcel gain in their data; normal / ORM maps: linear data
        if int(t.colorspace) == int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB) and not np.allclose(gain, 1.0):
            h, w, c = int(t.height), int(t.width), int(t.nchannel)
            img = np.frombuffer(bytes(t.data), np.uint8).reshape(h, w, c)
            out = img.copy()
            out[..., :3] = gain_srgb8(img[..., :3], gain)
            t.data = np.ascontiguousarray(out).tobytes()
    for mt in spec.materials:
        if mt.name in parcel_mats:
            mt.roughness = float(pl.get("roughness", 0.8))
            mt.metallic = float(pl.get("metallic", 0.0))
            mt.specular = float(pl.get("specular", 0.5))
            mt.reflectance = 0.0
    try:
        model = spec.compile()
    except Exception as exc:  # noqa: BLE001
        raise RenderError("renderer: the render twin did not compile: %s" % exc)
    if model.nq != m0.nq or model.nbody != m0.nbody or model.ncam != m0.ncam:
        raise RenderError("renderer: render twin out of step with the cell (nq %d / %d)" % (model.nq, m0.nq))
    for c in range(m0.ncam):
        model.cam_fovy[c] = m0.cam_fovy[c]
    model.vis.map.znear, model.vis.map.zfar = m0.vis.map.znear, m0.vis.map.zfar
    bodies = {model.body(b).name: b for b in range(model.nbody) if _is_robot_body(model.body(b).name)}
    rec.update(ngeom=int(model.ngeom), nmesh=int(model.nmesh), ntex=int(model.ntex), nmat=int(model.nmat))
    return model, bodies, rec


# ------------------------------------------------------------------------------------------------ the backend
class FilamentLook(object):
    name = "filament"

    def __init__(self, model, cameras=None, world=None, options=None):
        if world is None:
            raise RenderError("renderer: the filament look needs the shift's world")
        self.model = model
        self.world = world
        self.options = dict(options or {})
        self.look = load_look(self.options.get("look_file"))
        self.rdir = render_dir(self.options)
        rv = self.look.get("render", {})
        self.ssaa = max(1, int(rv.get("ssaa", 2)))
        self.near = float(rv.get("near_m", getattr(world, "NEAR_CLIP_M", 0.02)))
        self.far = float(rv.get("far_m", 120.0))
        ex = self.look.get("exposure", {})
        self.exposure_default = float(ex.get("default", 1.0))
        self.exposure_by_cam = {}
        names = {}
        for pub, spec in (cameras or {}).items():
            v = spec.get("view") if isinstance(spec, dict) else None
            if v is not None:
                names[v.model_camera] = pub
        for mc, pub in names.items():
            self.exposure_by_cam[mc] = float(ex.get("per_camera", {}).get(pub, ex.get("per_camera", {}).get(mc, 1.0)))
        self.light_factor = float(self.options.get("light_factor", 1.0))
        self.engine = None
        self.sid = None
        self.twin = self.tdata = None
        self.record = {}
        self._cid = {}
        self._drawn = False
        self._start_error = None

    def _ensure(self):
        if self.sid is not None:
            return
        if self._start_error is not None:            # the renderer failed closed in this process: no new build
            raise RenderError(self._start_error)
        t0 = time.time()
        man = check_bundle(self.rdir, self.look)
        try:
            twin, bodies, rec = build_twin(self.world, self.look)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: building the render twin: %s: %s" % (type(exc).__name__, exc))
        try:
            self.engine = Engine.instance(self.rdir, self.look)
        except RenderError as exc:
            self._start_error = str(exc)
            raise
        self.twin, self.tdata = twin, mujoco.MjData(twin)
        self.sid = self.engine.call(self.engine.new_session, twin, bodies)
        rec["twin_build_s"] = round(time.time() - t0, 3)
        rec["bundle"] = {"built": man.get("built"), "note": man.get("note")}
        self.record = rec

    def rgb(self, data, cam) -> np.ndarray:
        """The camera's colour image at the state ``data`` (a render-side MjData of the cell's model with its
        kinematics computed)."""
        self._ensure()
        try:
            td = self.tdata
            td.qpos[:] = data.qpos
            mujoco.mj_kinematics(self.twin, td)
            cid = self._cid.get(cam.model_camera)
            if cid is None:
                cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam.model_camera)
                if cid < 0:
                    raise RenderError("no camera %r" % cam.model_camera)
                self._cid[cam.model_camera] = cid
            R = np.asarray(data.cam_xmat[cid], float).reshape(3, 3)
            pos = np.asarray(data.cam_xpos[cid], float)
            fovy = float(self.model.cam_fovy[cid])
            e = self.exposure_default * self.exposure_by_cam.get(cam.model_camera, 1.0) * self.light_factor
            s = self.ssaa
            args = (self.sid, td, pos, -R[:, 2], R[:, 1], fovy, int(cam.width) * s, int(cam.height) * s, e,
                    self.near, self.far)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: camera %r: %s: %s" % (cam.model_camera, type(exc).__name__, exc))
        first = not self._drawn
        img = self.engine.call(self.engine.draw, *args,
                               timeout=self.engine.timeout if first else self.engine.frame_timeout)
        self._drawn = True
        return downsample(img, s)

    def crush(self, slot, seed, kind) -> None:
        """Parcel ``slot`` is drawn crushed from now on: the render twin's base colour texture of the parcel rewritten
        (parcelkit.crush, over the look's parcel gain) and the session's renderables made again at the next frame."""
        self._ensure()
        from .look import crush_model_rgba, crush_model_texture
        if not crush_model_texture(self.twin, slot, seed, kind):
            crush_model_rgba(self.twin, slot)
        self.engine.call(self.engine.invalidate, self.sid)
        self.record["crushed"] = int(self.record.get("crushed", 0)) + 1

    def info(self):
        return {"record": dict(self.record), "engine": engine_info() if self.engine is not None else None}

    def close(self):
        if self.sid is not None and self.engine is not None and not self.engine.broken:
            try:
                self.engine.call(self.engine.free_session, self.sid)
            except RenderError:
                pass
        self.sid = None
        self.twin = self.tdata = None
