"""The renderer behind the cell's cameras: one interface, two backends.

    look = make_look(name, model, cameras, world=None)   # name: "classic" | "filament"
    rgb = look.rgb(data, cam)                     # uint8 H x W x 3 at data's state
    look.close()

``cam`` is a ``CameraView`` (model camera name, width, height). ``data`` is a
render-side MjData whose kinematics the caller has computed (the camera ring
copies a recorded state into it); a backend never touches the physics data.

* ``classic``: MuJoCo's classic OpenGL renderer offscreen on the cell's own
  model (geom groups 0-2 drawn: station, parcels, robot, grippers; collision
  geoms and sites hidden); one GL context and one render context serve every
  camera (``ClassicGL``). Headless through ``MUJOCO_GL``: ``egl`` with a GPU,
  ``osmesa`` on the CPU (the task image).
* ``filament``: MuJoCo's Filament renderer (physically based, Vulkan; Mesa's
  lavapipe on the CPU in the task image) with the light distilled from the
  station's path-traced look: ``look_filament`` (the backend: render twin,
  cameras, supersampling) and ``filament_engine`` (the process's renderer
  thread, the render bundle's static set and robot, the lamps and the image
  light); the look's numbers in ``look_config/look.json``, its meshes and maps
  in the render bundle (configuration ``look.render_dir``, else
  ``$PARCELCELL_RENDER_DIR``, else ``/opt/task-assets/render``).

Depth always comes from ``ClassicDepth`` (the classic renderer on the same
geometry, plus the depth twin: the render bundle's near meshes of the
station's surroundings, drawn in depth only), planar distance along the
optical axis in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = ["CameraView", "ClassicGL", "ClassicLook", "ClassicDepth", "RenderError", "make_look",
           "make_depth", "BACKENDS"]

BACKENDS = ("classic", "filament")


class RenderError(RuntimeError):
    """The renderer could not produce a frame (the cell's failure)."""


@dataclass(frozen=True)
class CameraView:
    model_camera: str
    width: int
    height: int


DEPTH_TWIN_GROUP = 5


def _scene_option(depth: bool = False) -> mujoco.MjvOption:
    """Geom groups 0-2 (station, parcels, robot, grippers); a depth pass also draws the depth twin's group (the
    station's surroundings near the cameras, ``ShiftWorld.DEPTH_GROUP``)."""
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if (g in (0, 1, 2) or (depth and g == DEPTH_TWIN_GROUP)) else 0
        opt.sitegroup[g] = 0
    return opt


class ClassicGL:
    """One GL context, one MuJoCo render context and one scene for every camera
    of a model (one upload of the meshes and textures). Colour and depth are
    two passes, as MuJoCo's own ``Renderer`` draws them (depth with the
    segmentation flags, reversed-Z depth map, the same conversion to metres)."""

    def __init__(self, model):
        self.model = model
        w, h = int(model.vis.global_.offwidth), int(model.vis.global_.offheight)
        try:
            self.gl = mujoco.GLContext(w, h)
            self.gl.make_current()
            self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        except Exception as e:  # noqa: BLE001 - no GL context on this machine
            raise RenderError("classic renderer unavailable: %s" % e) from e
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.con)
        self.con.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self.scn = mujoco.MjvScene(model=model, maxgeom=10000)
        self.option = _scene_option()
        self.depth_option = _scene_option(depth=True)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._ids = {}

    def _scene(self, data, cam: CameraView, option=None):
        cid = self._ids.get(cam.model_camera)
        if cid is None:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam.model_camera)
            if cid < 0:
                raise RenderError("no camera %r" % cam.model_camera)
            self._ids[cam.model_camera] = cid
        self.cam.fixedcamid = cid
        self.gl.make_current()
        mujoco.mjv_updateScene(self.model, data, option or self.option, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.scn)
        return mujoco.MjrRect(0, 0, int(cam.width), int(cam.height))

    def rgb(self, data, cam: CameraView) -> np.ndarray:
        rect = self._scene(data, cam)
        out = np.empty((cam.height, cam.width, 3), np.uint8)
        mujoco.mjr_render(rect, self.scn, self.con)
        mujoco.mjr_readPixels(out, None, rect, self.con)
        return np.ascontiguousarray(np.flipud(out))

    def depth(self, data, cam: CameraView) -> np.ndarray:
        rect = self._scene(data, cam, self.depth_option)
        flags = self.scn.flags.copy()
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        out = np.empty((cam.height, cam.width), np.float32)
        try:
            mujoco.mjr_render(rect, self.scn, self.con)
            mujoco.mjr_readPixels(None, out, rect, self.con)
        finally:
            np.copyto(self.scn.flags, flags)
        extent = self.model.stat.extent
        zfar = np.float32(self.model.vis.map.zfar * extent)
        znear = np.float32(self.model.vis.map.znear * extent)
        c = -(zfar + znear) / (zfar - znear)
        d = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c = np.float32(-0.5) * c - np.float32(0.5)
        d = np.float32(-0.5) * d
        z = (d / (out.astype(np.float64) + c)).astype(np.float32)
        return np.ascontiguousarray(np.flipud(z))

    def close(self):
        for x in ("con", "gl"):
            obj = getattr(self, x, None)
            if obj is not None:
                try:
                    obj.free()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, x, None)


def crush_model_texture(model, slot, seed, kind) -> bool:
    """Write parcel ``slot``'s crushed look into ``model``'s base colour texture (``parcel<i>_kit_rgb``,
    parcelkit.crush). False when the parcel has no such texture."""
    from .parcelkit.crush import crush_rgb
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "parcel%d_kit_rgb" % int(slot))
    if tid < 0:
        return False
    adr, h, w, c = (int(model.tex_adr[tid]), int(model.tex_height[tid]), int(model.tex_width[tid]),
                    int(model.tex_nchannel[tid]))
    img = np.asarray(model.tex_data[adr:adr + h * w * c]).reshape(h, w, c)
    model.tex_data[adr:adr + h * w * c] = crush_rgb(img, seed, kind).reshape(-1)
    return True


def crush_model_rgba(model, slot) -> int:
    """The plain visuals' crushed look: the parcel's drawn geoms (group 0) darkened to the crushed tone."""
    from .parcelkit.crush import TONE
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parcel%d" % int(slot))
    n = 0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == bid and int(model.geom_group[g]) == 0 and not int(model.geom_contype[g]):
            model.geom_rgba[g, :3] = model.geom_rgba[g, :3] * TONE * 0.8
            n += 1
    return n


class ClassicLook:
    name = "classic"

    def __init__(self, model, cameras=None, gl=None, world=None):
        self.gl = gl or ClassicGL(model)

    def rgb(self, data, cam: CameraView) -> np.ndarray:
        return self.gl.rgb(data, cam)

    def crush(self, slot, seed, kind) -> None:
        """Parcel ``slot`` is drawn crushed from now on: its base colour texture rewritten and uploaded again (the
        plain visuals: its colours darkened). Drawing only: no physics field changes."""
        m = self.gl.model
        if crush_model_texture(m, slot, seed, kind):
            tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "parcel%d_kit_rgb" % int(slot))
            self.gl.gl.make_current()
            mujoco.mjr_uploadTexture(m, self.gl.con, tid)
        else:
            crush_model_rgba(m, slot)

    def close(self):
        self.gl.close()


class ClassicDepth:
    """Depth from the classic renderer; shares the classic look's context when
    there is one."""

    def __init__(self, model, gl=None):
        self.own = gl is None
        self.gl = gl or ClassicGL(model)

    def depth(self, data, cam: CameraView) -> np.ndarray:
        return self.gl.depth(data, cam)

    def close(self):
        if self.own:
            self.gl.close()


def make_look(name: str, model, cameras=None, world=None, options=None):
    """``cameras``: {public name: spec dict with "view" (CameraView)}; ``world``:
    the ShiftWorld (a backend that builds its own render twin reads
    ``world.spec``, ``world.visuals`` and the parcel bodies from it);
    ``options``: the configuration's ``look`` block (the filament backend reads
    ``look_file`` and ``render_dir`` from it)."""
    if name == "classic":
        return ClassicLook(model, cameras, world=world)
    if name == "filament":
        try:
            from .look_filament import FilamentLook
        except ImportError as e:
            raise RenderError("the filament backend is not installed: %s" % e) from e
        return FilamentLook(model, cameras, world=world, options=options)
    raise RenderError("unknown look backend %r (backends: %s)" % (name, ", ".join(BACKENDS)))


def make_depth(model, look=None):
    """The depth renderer: the classic look's own context, or a classic one."""
    return ClassicDepth(model, gl=getattr(look, "gl", None) if isinstance(look, ClassicLook) else None)
