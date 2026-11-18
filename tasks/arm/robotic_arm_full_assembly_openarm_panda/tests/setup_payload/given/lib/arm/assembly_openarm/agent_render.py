"""The renderer of the cell's cameras, for your own twin of the OpenArm pair cell: MuJoCo's Filament renderer (physically
based, Vulkan; on the CPU through Mesa's lavapipe) in the baked factory hall with the cell's look. This is the code the
cell renders its camera frames with (the full-arm cell's session, `arm/assembly_full/agent_render.py`, on the pair's
bench); what the cell draws per run (the hall's brightness) is an argument here.

    from arm.assembly_openarm import agent_render
    session = agent_render.open_session(spec)          # your mujoco.MjSpec of the cell, in the cell frame (copied)
    session.sync_from(model, data)                     # your physics state, copied by joint name
    raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # raw.rgb uint8 H x W x 3, raw.depth float32 metres
    res = session.render([cam], photo=photo, seed=k, outputs=("rgb", "sensor_depth"))   # through render's photo pipeline
    session.close()

`cam` is a render.spec.CameraSpec (pixels fx fy cx cy, world_T_cam in OpenCV axes: x right, y down, z forward; lens
distortion only through the photo pipeline). One session per scene and one open session per process (close it before
opening the next): it is compiled once (about 12-30 s on 8 CPUs for a twin of the whole cell, up to about 45 s on a busy
machine, the AgX display transform's 8-bit lookup table, about 2 s, included) and holds about 5.5-6.5 GB while open;
until you give it a state it shows your twin's own pose (qpos0). The first frame after Filament's context is made takes
about 7-11 s, later ones about 0.45-0.65 s (1280 x 960) and 0.25-0.35 s (640 x 480) with depth (longer on a busy
machine). Closing the last open session (or view) of the process releases Filament's context and its memory, so the next
one's first frame takes that long again. The render model is your MjSpec (bodies and joint names as in your physics)
without the MuJoCo-only scenery (geoms named `floor`, `backdrop_*`, `bench_leg_*`, `bench_apron_*`, `bench_shelf`, and
every geom in a group above 2), plus the steel workbench frame under the 1.6 x 1.4 m bench top, the factory room bundle
(Cycles-baked lightmaps on the hall, its lamps lighting the live objects, its HDR image light) and the look
(`agent_look.json` + the material classes below; the scanned oak maps on the bench top geom `table_top` of material
`mat_bench`). The datum boards (a checkerboard on the bench, one on the gallows' foot plate, a plate on each Panda's
hand: physical facts the cell manual documents) are part of your twin: `render.scale_refs.checkerboard` + `add_to_spec`
draw them, their materials named `<board>__*` (`bench_board`, `fixture_board`, `hand_plate_left`, `hand_plate_right`).
Your scene's own lights are replaced by the hall's. RGB from Filament as an exact pinhole, depth from the MuJoCo backend
on the same geometry; lens distortion, noise, blur, vignetting, colour jitter and the depth-sensor model in
render/postprocess.py. `exposure_factor` scales the exposure (the cell's hall is up to about +-20 % brighter or darker
from run to run).
"""
import json
import math
import os

import numpy as np

from ..assembly_full import agent_render as FA

__all__ = ["open_session", "available", "make_look", "PairSession"]

LOOK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_look.json")
TOP_HALF = (0.8, 0.7)
_SPECS = {}


def available():
    return FA.available()


def make_look(exposure_factor=1.0, render_dir=None):
    from render.filament.backend import Look
    look = Look(name="factory-agent-openarm", version="1")
    look.materials = [
        (None, "mat_alu", dict(roughness=0.38, metallic=1.0)),          # machined / anodised aluminium plates
        (None, "mat_steel", dict(roughness=0.32, metallic=1.0)),        # steel shafts, the gallows' column
        (None, "mat_dark", dict(roughness=0.5, metallic=0.0)),          # the motors' black cases, the gripper's prints
        (None, "mat_rotor", dict(roughness=0.3, metallic=1.0)),         # the motors' output flanges
        (None, "mat_mark", dict(roughness=0.5, metallic=0.0)),          # the flanges' index ticks
        (None, "mat_riser", dict(roughness=0.4, metallic=1.0)),         # the aluminium riser blocks
        (None, "pl/*", dict(roughness=0.38, metallic=0.0)),             # the Pandas' painted shells
        (None, "pr/*", dict(roughness=0.38, metallic=0.0)),
        (None, "mat_bench", dict(roughness=1.0, metallic=0.0)),         # x the oak's roughness map
        (None, "bench_steel", dict(roughness=0.4, metallic=1.0)),
        (None, "bench_shelf", dict(roughness=0.55, metallic=1.0)),
        (None, "*_board__*", dict(roughness=0.8, metallic=0.0)),
        (None, "hand_plate*", dict(roughness=0.8, metallic=0.0)),
        (None, None, dict(roughness=0.6, metallic=0.0)),
    ]
    look.surface_material, look.surface_geom, look.surface_map_m = "mat_bench", "table_top", 1.83
    look.surface_albedo_gain = 1.0
    look.override({k: v for k, v in json.load(open(LOOK_FILE)).items() if not k.startswith("_")})
    oak = os.path.join(render_dir or FA._render_dir(), "oak")
    look.surface_dir = oak if os.path.isdir(oak) else None
    ibl = os.path.join(FA._room_dir(render_dir), "ibl", "ibl.ktx")
    look.room_ibl_ktx = ibl if os.path.isfile(ibl) else None
    look.exposure_scale = float(look.exposure_scale) * float(exposure_factor)
    return look


def _base_spec(spec):
    """A copy of your twin without the MuJoCo-only scenery, with the steel workbench under the pair's bench (no room yet)."""
    spec = spec.copy()
    for g in list(spec.geoms):
        nm = g.name or ""
        if nm == "floor" or nm.startswith(FA._DROP) or int(g.group) > 2:
            spec.delete(g)
    FA.TOP_HALF = TOP_HALF            # the pair's bench: 1.6 x 1.4 m
    FA._workbench(spec)
    return spec


def depth_scene(key):
    return _SPECS[key].copy()


def _mjrf():
    """the Filament module, with MuJoCo 3.14's names on 3.17 (ModelObjects -> ModelResources; a texture's
    `set_data` -> `upload`, in the room's image light)"""
    import mujoco._render_filament as mjrf
    if not hasattr(mjrf, "ModelObjects") and hasattr(mjrf, "ModelResources"):
        mjrf.ModelObjects = mjrf.ModelResources
    if not hasattr(mjrf.Texture, "set_data"):
        from render.filament import backend as FB

        def _ktx_ibl(self, ctx, scn, path=None, intensity=None):
            raw = open(path, "rb").read() if path else self._ibl_bytes()
            tc = mjrf.TextureConfig()
            tc.width, tc.height, tc.format = len(raw), 0, mjrf.PixelFormat.PIXEL_FORMAT_KTX
            tc.sampler_type, tc.color_space = mjrf.SamplerType.TEXTURE_CUBE, mjrf.ColorSpace.COLORSPACE_LINEAR
            tex = ctx.create_texture(tc)
            td = mjrf.TextureData()
            td.set_image(raw)
            tex.upload(td)
            lp = mjrf.LightParams()
            lp.type, lp.texture, lp.color = mjrf.LightType.LIGHT_IMAGE, tex, [1.0, 1.0, 1.0]
            lp.intensity = float(self.look.ibl_intensity if intensity is None else intensity)
            ibl = ctx.create_light(lp)
            scn.add_light(ibl)
            return ibl, tex, td
        FB.FilamentBackend._ktx_ibl = _ktx_ibl
    return mjrf


class PairSession(FA.FilamentAgentSession):
    """the full-arm task's Filament session; MuJoCo 3.17's readback (ReadRequest + a caller's buffer) where 3.14 had
    ReadPixelsRequest (alloc / buffer)"""

    def _target(self, w, h):
        mjrf = self._mjrf
        if hasattr(mjrf, "ReadPixelsRequest"):
            return FA.FilamentAgentSession._target(self, w, h)
        if (w, h) not in self._targets:
            t = self.ctx.create_render_target(mjrf.RenderTargetConfig(color_format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8))
            t.resize(w, h)
            rd = mjrf.ReadRequest()
            rd.target = t
            buf = np.zeros(w * h * 3, np.uint8)
            rd.set_buffer(buf)
            self._targets[(w, h)] = (t, rd, buf)
        return self._targets[(w, h)]

    def _rgb(self, cam):
        if hasattr(self._mjrf, "ReadPixelsRequest"):
            return FA.FilamentAgentSession._rgb(self, cam)
        from render.filament.backend import gl_camera
        mjrf = self._mjrf
        target, rd, buf = self._target(cam.width, cam.height)
        rq = mjrf.RenderRequest(camera=gl_camera(cam, self.near, self.far), draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT)
        rq.scene, rq.target = self.scn, target
        rq.viewport.left, rq.viewport.bottom, rq.viewport.width, rq.viewport.height = 0, 0, cam.width, cam.height
        self.ctx.wait_for_frame(self.ctx.render([rq], [rd]))
        rgb = buf.reshape(cam.height, cam.width, 3)
        if self.lut is None:
            return rgb.copy()
        idx = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
        return self.lut[idx]

    def close(self):
        """The full-arm session's close (Filament's context goes with the process's last open session or view), then
        the pair's registered depth twin."""
        was_open = self.scn is not None
        FA.FilamentAgentSession.close(self)
        if was_open:
            _SPECS.clear()


def open_session(spec, exposure_factor=1.0, render_dir=None, content_files=(), lod="full"):
    """A render session of your twin `spec` (a mujoco.MjSpec of the pair cell in the cell frame; copied, never changed):
    Filament in the factory hall, depth from MuJoCo on the same geometry.

    exposure_factor  the hall's brightness (1.0 = nominal; the cell's own varies by up to about +-20 % between runs)
    render_dir       the scenery folder with room/ and oak/ (default: the full-arm module's _render_dir())
    content_files    files or folders whose bytes enter the frames' provenance digest (e.g. your meshes)
    lod              "full" (the meshes as they are) or "policy" (the render layer's decimated meshes, render/lod.py)
    """
    from render.spec import SceneSpec
    FA.check_one_session()               # one Filament session per process at a time: close the open one first
    FA._KEY[0] += 1
    key = "twin-%d-%d" % (os.getpid(), FA._KEY[0])
    dspec = _base_spec(spec)
    FA._room_geometry(dspec, FA._room_dir(render_dir))
    content = [str(p) for p in content_files] + [os.path.join(FA._room_dir(render_dir), "room.json")]
    _SPECS.clear()
    _SPECS[key] = dspec
    sc = SceneSpec(builder="arm.assembly_openarm.agent_render:depth_scene", builder_args={"key": key}, content_files=content,
                   shadows=False, lod=str(lod), near_m=0.01)
    # the full-arm session builds its Filament twin with `_base_spec` and its look with `make_look`: the pair's
    old_base, old_look = FA._base_spec, FA.make_look
    FA._base_spec, FA.make_look = _base_spec, make_look
    try:
        _mjrf()
        be = FA.FilamentAgentBackend()
        return PairSession(be, spec, sc, float(exposure_factor), render_dir=render_dir)
    finally:
        FA._base_spec, FA.make_look = old_base, old_look
