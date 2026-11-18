"""Filament backend: MuJoCo's own physically based renderer (`mujoco._render_filament`, MuJoCo >= 3.14) behind the
renderer-agnostic photo layer (`render/backends/base.py` is normative).

What it does per request:
 1. Resolves the scene exactly as the MuJoCo backend does (`mujoco_scene.build_spec`), then applies a LOOK to the MjSpec:
    physically based materials per object class (roughness / metallic, matched by geom / material name), light
    intensities in candela with a bulb radius for soft shadows.  Geometry, state, cameras and textures are untouched, so
    every photo stays pixel-registered with the MuJoCo backend's photo of the same request.
 2. Builds a Filament scene from the compiled model (ModelObjects / ModelLights / ModelRenderables) and adds an image
    light (IBL) from the HDR environment that ships with MuJoCo (experimental/studio/assets/ibl.ktx, Apache-2.0).
 3. Renders each raw camera as an exact off-axis pinhole (the same frustum as mujoco_backend.set_scene_camera) into an
    RGB8 target; rows are top-down.  Lens distortion, exposure, white balance, vignetting, noise, colour jitter and
    JPEG stay in postprocess.py, identical for every backend.
 4. Segmentation and depth, when requested, come from the MuJoCo backend for the same raw cameras (exact geom ids and
    metric depth; the look does not move any geom).

Needs a GPU with Vulkan (default) or an X/EGL-capable OpenGL; authoring-time only, never inside a task image.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from render.backends import mujoco_scene as MS
from render.backends.base import BackendCapabilities, RawRender, RawView, RenderBackend, raw_outputs
from render.spec import CameraSpec, RenderRequest

BACKEND_VERSION = "filament-backend/0.1.0"


@dataclass
class Look:
    """Appearance knobs that change pixels only (never geometry).  Every field goes into the backend version."""
    name: str = "lab"
    version: str = "2"
    # (glob on geom name, glob on material name) -> PBR parameters; first match wins; None = no constraint
    materials: List[Tuple[Optional[str], Optional[str], Dict[str, float]]] = field(default_factory=lambda: [
        ("vis_P*_reprint", None, dict(roughness=0.34, metallic=0.0)),  # a PETG reprint (glossier)
        ("vis_P*", None, dict(roughness=0.72, metallic=0.0)),          # FDM PLA prints (matte)
        ("vis_S*", None, dict(roughness=0.5, metallic=0.0)),           # servo cases (black ABS / PA)
        (None, "bench_wood", dict(roughness=1.0, metallic=0.0)),       # satin wood top (x the scanned roughness map)
        ("tape_*", None, dict(roughness=0.7, metallic=0.0)),           # paper / vinyl tape
        ("front_arrow*", None, dict(roughness=0.5, metallic=0.0)),     # vinyl sticker
        ("tote_*", None, dict(roughness=0.55, metallic=0.0)),          # PP tote
        ("object_*", None, dict(roughness=0.2, metallic=0.0)),         # glazed ceramic / enamel / can objects
        (None, "mat__*", dict(roughness=0.88, metallic=0.0)),          # PVC cutting mat
        (None, "ruler__*", dict(roughness=0.32, metallic=1.0)),        # stainless steel rule
        (None, "checker__*", dict(roughness=0.8, metallic=0.0)),       # printed card
        (None, "can__wall", dict(roughness=0.38, metallic=0.55)),      # printed aluminium wall
        (None, "can__*", dict(roughness=0.3, metallic=1.0)),           # aluminium lid / tab / bottom
        (None, "card__*", dict(roughness=0.38, metallic=0.0)),         # PVC card
        (None, None, dict(roughness=0.6, metallic=0.0)),               # anything else
    ])
    cd_per_diffuse: float = 240000.0    # spot / point light candela per unit of MuJoCo diffuse (mean of rgb)
    lux_per_diffuse: float = 60000.0    # directional light lux per unit of MuJoCo diffuse
    shadow_bulb_radius_m: float = 0.03  # soft shadows from the shadow-casting lights (a softbox, not a bare bulb)
    ibl_intensity: float = 5000.0       # image light (HDR environment) intensity
    ibl_file: str = "experimental/studio/assets/ibl.ktx"   # relative to the mujoco package
    # an equirectangular environment (HDRI as float32 .npy, linear): background + image light; replaces the ktx IBL
    env_file: Optional[str] = None      # default: $ARB_FILAMENT_ENV
    env_intensity: float = 22000.0
    env_yaw_deg: float = -108.0         # poly_haven_studio: its window (az -148, el 50) onto the key light's azimuth (-40)
    env_ev: float = 0.0
    env_wb: float = 0.5                 # half-way white balance of the environment (5000 K room -> towards neutral)
    env_face_px: int = 1536
    fill_scale_with_env: float = 0.35   # the non-shadow lights (fill, back) under an environment that already fills
    key_scale_with_env: float = 0.75    # the shadow-casting light under an environment (no clipped highlights)
    key_bulb_with_env_m: float = 0.12   # a softer key (window / softbox) under an environment
    # Filament view options as custom fields of the model: MuJoCo's Filament reads "filament.*" from <numeric> (numbers,
    # booleans as 0 / 1, enums as their index) and from <text> (e.g. filament.cg.tone_mapping)
    filament_numeric: Dict[str, float] = field(default_factory=lambda: {
        "filament.shadows.map_size": 4096.0, "filament.shadows.type": 3.0,   # PCSS: contact-hardening soft shadows
        "filament.ao.ssct": 1.0, "filament.bloom.enabled": 1.0})             # cone-traced contact shadows, lens bloom
    filament_text: Dict[str, str] = field(default_factory=dict)
    # scanned PBR maps (one *_diff_ / *_nor_gl_ / *_arm_ set covering surface_map_m metres) for the bench top
    surface_dir: Optional[str] = None   # default: $ARB_FILAMENT_SURFACE
    surface_material: str = "bench_wood"
    surface_geom: str = "bench_top"
    surface_map_m: float = 1.83
    surface_roughness_scale: float = 1.0
    surface_albedo_gain: float = 1.35   # oak_veneer_01 is a darker veneer than a natural oak bench top
    # a room bundle in the scene (geoms room_*, lamps room_light_*; room/__init__.py): the lamps light the bench, a baked
    # light probe of the room (equirectangular .npy in the bench frame) is the image light, env_file is only the sky
    # seen through the windows
    room_probe_file: Optional[str] = None   # default: $ARB_FILAMENT_ROOM_PROBE
    room_probe_intensity: float = 22000.0
    room_probe_px: int = 256
    room_light_scale: float = 500.0     # room lamp intensity (bundle units) -> candela
    room_bulb_m: float = 0.25           # tube / panel lamps (1.4 m tubes): soft shadows
    room_sky_ev: float = 2.0            # the outdoor sky is brighter than the room (windows read blown out)
    bench_lights_in_room: float = 0.0   # the scene's own key / fill lights when a room lights the bench
    # a baked room (render_cycles/bake_room.py: Cycles lightmaps + a Cycles panorama as the probe), calibrated against the
    # Cycles photos of the same shots (region luminance of room / arm / bench within ~15 %)
    room_baked_emission: float = 1.38   # emission of its lightmaps
    room_baked_light_scale: float = 1300.0      # the lamps, lighting only the live objects (arm, bench top, props)
    room_baked_probe_intensity: float = 56000.0
    # an HDR image light (make_ktx_ibl.py: cmgen KTX of the Cycles panorama) instead of the 8-bit probe cube
    room_ibl_ktx: Optional[str] = None  # default: $ARB_FILAMENT_ROOM_IBL_KTX
    room_ibl_ktx_intensity: float = 10000.0
    exposure_scale: float = 1.0         # one factor on every light, image light and baked emission (a camera exposure)
    # the display transform: None = Filament's own tone mapper; "agx_punchy" / "agx" = render with tone mapping "linear"
    # at exposure_scale (headroom for highlights) and apply AgX (agx.py, as Blender / Cycles) with post_tonemap_gain
    post_tonemap: Optional[str] = None
    post_tonemap_gain: float = 1.0

    def digest(self) -> str:
        env = env_sha(self.env_file)
        surf = surface_sha(self.surface_dir)
        return hashlib.sha256(repr((self.name, self.version, self.materials, self.cd_per_diffuse, self.lux_per_diffuse,
                                    self.shadow_bulb_radius_m, self.ibl_intensity, self.ibl_file, env, self.env_intensity,
                                    self.env_yaw_deg, self.env_ev, self.env_wb, self.env_face_px, self.fill_scale_with_env,
                                    self.key_scale_with_env, self.key_bulb_with_env_m, sorted(self.filament_numeric.items()),
                                    sorted(self.filament_text.items()), surf, self.surface_material, self.surface_geom,
                                    self.surface_map_m, self.surface_roughness_scale, self.surface_albedo_gain,
                                    env_sha(self.room_probe_file), self.room_probe_intensity, self.room_probe_px,
                                    self.room_light_scale, self.room_bulb_m, self.room_sky_ev,
                                    self.bench_lights_in_room, self.room_baked_emission, self.room_baked_light_scale,
                                    self.room_baked_probe_intensity, env_sha(self.room_ibl_ktx),
                                    self.room_ibl_ktx_intensity, self.exposure_scale, self.post_tonemap,
                                    self.post_tonemap_gain)).encode()).hexdigest()[:12]

    def override(self, spec: Dict[str, Any]) -> None:
        """Set fields from a dict (ARB_LOOK_JSON); "materials" maps a geom / material glob of an existing rule to new
        parameters, "filament_numeric" / "filament_text" are merged."""
        for k, v in spec.items():
            if k == "materials":
                self.materials = [(gp, mp, dict(p, **v.get(gp or mp, {}))) for gp, mp, p in self.materials]
            elif k in ("filament_numeric", "filament_text"):
                getattr(self, k).update(v)
            elif hasattr(self, k):
                setattr(self, k, type(getattr(self, k))(v) if getattr(self, k) is not None else v)
            else:
                raise KeyError("Look has no field %r" % k)


def env_sha(path):
    if not path:
        return None
    from .env import file_digest
    return file_digest(path)


def surface_sha(directory):
    if not directory:
        return None
    from .surfaces import set_digest
    return set_digest(directory)


def _match(name: str, pattern: Optional[str]) -> bool:
    return pattern is None or fnmatch.fnmatchcase(name or "", pattern)


def _albedo(rgba, scale):
    """rgba (sRGB-encoded, as MuJoCo / Filament read it) with its colour scaled by `scale` in linear light."""
    if scale == 1.0:
        return rgba
    out = []
    for c in rgba[:3]:
        lin = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        lin = min(1.0, lin * scale)
        out.append(12.92 * lin if lin <= 0.0031308 else 1.055 * lin ** (1 / 2.4) - 0.055)
    return out + list(rgba[3:])


def apply_look(spec, look: Look) -> Dict[str, Any]:
    """Edit an MjSpec in place: a PBR material on every visible geom, physical light intensities.  Returns a record."""
    import mujoco
    mats = {m.name: m for m in spec.materials}
    made, counts = {}, {}
    room = any((g.name or "").startswith("room_") for g in spec.geoms)
    baked_room = any((m.name or "").startswith("room_baked_") for m in spec.materials)
    light_scale = look.room_baked_light_scale if baked_room else look.room_light_scale
    probe_intensity = look.room_baked_probe_intensity if baked_room else look.room_probe_intensity
    for g in spec.geoms:
        gname, mname = g.name or "", g.material or ""
        if gname.startswith("room_"):                  # a room bundle brings its own PBR materials
            continue
        params = next((p for gp, mp, p in look.materials if _match(gname, gp) and _match(mname, mp)), None)
        if params is None:
            continue
        key = "%s|%s" % (mname, tuple(sorted(params.items())))
        if mname and mname in mats:
            if mname not in made:                      # an authored material: add PBR terms in place (textures kept)
                m = mats[mname]
                m.roughness, m.metallic = float(params["roughness"]), float(params["metallic"])
                m.rgba = _albedo(list(m.rgba), float(params.get("albedo_scale", 1.0)))
                made[mname] = mname
        else:                                          # rgba-only geom: give it its own PBR material of the same colour
            nm = "look_%d" % len(made)
            m = spec.add_material(name=nm)
            m.rgba = _albedo(list(g.rgba), float(params.get("albedo_scale", 1.0)))
            m.roughness, m.metallic = float(params["roughness"]), float(params["metallic"])
            g.material = nm
            made[key + nm] = nm
        counts[str(params)] = counts.get(str(params), 0) + 1
    lights = []
    for l in spec.lights:
        if int(l.type) == int(mujoco.mjtLightType.mjLIGHT_IMAGE):
            continue
        if (l.name or "").startswith("room_light_"):   # a room lamp: its own colour, intensity in bundle units
            l.intensity = float(l.intensity) * light_scale
            if l.castshadow:
                l.bulbradius = look.room_bulb_m
            lights.append({"name": l.name, "intensity": round(float(l.intensity), 1), "bulbradius": float(l.bulbradius)})
            continue
        d = float(np.mean(l.diffuse))
        directional = int(l.type) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        l.intensity = d * (look.lux_per_diffuse if directional else look.cd_per_diffuse)
        if room:
            l.intensity *= look.bench_lights_in_room
            l.castshadow = bool(l.castshadow and look.bench_lights_in_room > 0)
        elif l.castshadow:
            l.bulbradius = look.key_bulb_with_env_m if look.env_file else look.shadow_bulb_radius_m
            if look.env_file:
                l.intensity *= look.key_scale_with_env
        elif look.env_file:
            l.intensity *= look.fill_scale_with_env
        lights.append({"name": l.name, "intensity": round(float(l.intensity), 1), "bulbradius": float(l.bulbradius)})
    baked = [m for m in spec.materials if (m.name or "").startswith("room_baked_")]
    for m in baked:                                    # a baked room shows its lightmaps; the lamps light only the live
        m.emission = float(look.room_baked_emission)  # objects (black base colour, no reflectance)
    rec = {"materials": len(made), "lights": lights, "room": room, "baked_room_materials": len(baked)}
    if look.env_file:
        from .env import add_environment
        probe = look.room_probe_file if room else None
        rec["environment"] = add_environment(spec, look.env_file, probe_intensity if probe else look.env_intensity,
                                             n=look.env_face_px, yaw_deg=look.env_yaw_deg,
                                             ev=look.env_ev + (look.room_sky_ev if probe else 0.0), wb=look.env_wb,
                                             ibl_path=probe, ibl_n=look.room_probe_px,
                                             image_light=not (room and look.room_ibl_ktx))
    if look.surface_dir and look.surface_material in mats:
        from .surfaces import apply_surface
        rec["surface"] = apply_surface(spec, look.surface_material, look.surface_geom, look.surface_dir, look.surface_map_m,
                                       roughness_scale=look.surface_roughness_scale, albedo_gain=look.surface_albedo_gain)
    if look.exposure_scale != 1.0:
        for l in spec.lights:
            l.intensity = float(l.intensity) * look.exposure_scale
        for m in baked:
            m.emission = float(m.emission) * look.exposure_scale
    for k, v in sorted(look.filament_numeric.items()):
        spec.add_numeric(name=k, data=[float(v)])
    texts = dict(look.filament_text)
    if look.post_tonemap:                              # the display transform is applied after the render (agx.py)
        texts["filament.cg.tone_mapping"] = "linear"
    for k, v in sorted(texts.items()):
        spec.add_text(name=k, data=str(v))
    rec["filament"] = dict(look.filament_numeric, **look.filament_text)
    return rec


def compile_with_look(scene, look: Look):
    """mujoco_scene.compile_scene plus the look edits between build_spec and compile."""
    import mujoco
    spec, info = MS.build_spec(scene)
    info["look"] = apply_look(spec, look)
    model = spec.compile()
    if scene.lod != "full":
        info["lod"] = MS.apply_lod(spec, model, scene)
        model = spec.compile()
    ext = float(model.stat.extent)
    model.vis.map.znear = scene.near_m / ext
    model.vis.map.zfar = scene.far_m / ext
    data = mujoco.MjData(model)
    MS.apply_state(model, data, scene.state)
    return MS.CompiledScene(model=model, data=data, spec=spec, labels=MS._labels(model, info), info=info)


_CTX: Dict[str, Any] = {}


def _context(api: str):
    import mujoco._render_filament as mjrf
    from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
    if api not in _CTX:
        pci = os.environ.get("ARB_FILAMENT_PCI")          # e.g. 0000:41:00.0: expose only this GPU to Vulkan
        if pci and api == "vulkan":                         # (Mesa device-select layer; read at instance creation)
            os.environ.setdefault("DRI_PRIME", "pci-" + pci.replace(":", "_").replace(".", "_"))
            os.environ.setdefault("MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE", "1")
        gapi = {"vulkan": mjrf.GraphicsApi.GRAPHICS_API_VULKAN, "opengl": mjrf.GraphicsApi.GRAPHICS_API_OPENGL}[api]
        _CTX[api] = mjrf.Context(mjrf.ContextConfig(graphics_api=gapi))
    return _CTX[api]


def gl_camera(cam: CameraSpec, near: float, far: float):
    """The exact off-axis pinhole of mujoco_backend.set_scene_camera as an mjrf.Camera."""
    import mujoco._render_filament as mjrf
    R = cam.R
    eye, fwd, up = cam.position, R[:, 2], -R[:, 1]
    W, H = cam.width, cam.height
    left, right = (-0.5 - cam.cx) / cam.fx * near, (W - 0.5 - cam.cx) / cam.fx * near
    top, bottom = (cam.cy + 0.5) / cam.fy * near, -(H - 0.5 - cam.cy) / cam.fy * near
    return mjrf.Camera(pos=[float(x) for x in eye], forward=[float(x) for x in fwd], up=[float(x) for x in up],
                       frustum_center=0.5 * (left + right), frustum_width=0.5 * (right - left),
                       frustum_bottom=bottom, frustum_top=top, frustum_near=near, frustum_far=far)


class FilamentBackend(RenderBackend):
    name = "filament"

    def __init__(self, look: Optional[Look] = None, api: Optional[str] = None):
        self.look = look or Look()
        for env, attr in (("ARB_LOOK_IBL", "ibl_intensity"), ("ARB_LOOK_CD", "cd_per_diffuse"), ("ARB_LOOK_BULB", "shadow_bulb_radius_m"),
                          ("ARB_LOOK_ENV_INTENSITY", "env_intensity"), ("ARB_LOOK_ENV_YAW", "env_yaw_deg"), ("ARB_LOOK_ENV_EV", "env_ev"),
                          ("ARB_LOOK_FILL", "fill_scale_with_env"), ("ARB_LOOK_ENV_WB", "env_wb"),
                          ("ARB_LOOK_KEY", "key_scale_with_env"), ("ARB_LOOK_KEY_BULB", "key_bulb_with_env_m")):
            if os.environ.get(env):
                setattr(self.look, attr, float(os.environ[env]))
        if self.look.env_file is None and os.environ.get("ARB_FILAMENT_ENV"):
            self.look.env_file = os.environ["ARB_FILAMENT_ENV"]
        if self.look.room_probe_file is None and os.environ.get("ARB_FILAMENT_ROOM_PROBE"):
            self.look.room_probe_file = os.environ["ARB_FILAMENT_ROOM_PROBE"]
        if self.look.room_ibl_ktx is None and os.environ.get("ARB_FILAMENT_ROOM_IBL_KTX"):
            self.look.room_ibl_ktx = os.environ["ARB_FILAMENT_ROOM_IBL_KTX"]
        if self.look.surface_dir is None and os.environ.get("ARB_FILAMENT_SURFACE"):
            self.look.surface_dir = os.environ["ARB_FILAMENT_SURFACE"]
        if os.environ.get("ARB_LOOK_JSON"):            # a JSON object, or the path of a preset file (presets/*.json)
            import json
            v = os.environ["ARB_LOOK_JSON"]
            spec = json.load(open(v)) if os.path.isfile(v) else json.loads(v)
            self.look.override({k: x for k, x in spec.items() if not k.startswith("_")})
        self.api = api or os.environ.get("ARB_FILAMENT_API", "vulkan")

    def _ibl_bytes(self) -> bytes:
        import mujoco
        with open(os.path.join(os.path.dirname(mujoco.__file__), self.look.ibl_file), "rb") as f:
            return f.read()

    def version(self) -> str:
        import mujoco
        ibl = hashlib.sha256(self._ibl_bytes()).hexdigest()[:8]
        return "%s mujoco-%s api-%s look-%s/%s-%s ibl-%s" % (BACKEND_VERSION, mujoco.__version__, self.api, self.look.name,
                                                             self.look.version, self.look.digest(), ibl)

    def capabilities(self) -> BackendCapabilities:
        try:
            import mujoco._render_filament  # noqa: F401
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
        return BackendCapabilities(name=self.name, implemented=ok, outputs=["rgb", "depth", "segmentation"],
                                   native_lens_distortion=False, sessions=False,
                                   deterministic="to be measured: same mujoco build, Vulkan driver and GPU",
                                   notes="MuJoCo Filament PBR renderer; segmentation / depth from the MuJoCo backend",
                                   lod_levels=["full", "policy"])

    def environment(self, session=None) -> Dict[str, str]:
        import mujoco
        return {"mujoco": mujoco.__version__, "filament_api": self.api, "look": self.look.name + "/" + self.look.version}

    def render_raw(self, request: RenderRequest, raw_cameras: List[CameraSpec]) -> RawRender:
        import mujoco._render_filament as mjrf
        t0 = time.perf_counter()
        cs = compile_with_look(request.scene, self.look)
        model, data = cs.model, cs.data
        ctx = _context(self.api)
        objs = mjrf.ModelObjects(ctx, model)
        scn = ctx.create_scene(mjrf.SceneParams())
        scn.configure_from_model(model)
        lights = mjrf.ModelLights(scn, objs)
        rends = mjrf.ModelRenderables(scn, objs)
        lights.update(data)
        rends.update(data)
        ibl = ibl_tex = td = None
        if cs.info["look"].get("room") and self.look.room_ibl_ktx:      # a room's HDR image light (make_ktx_ibl.py)
            ibl, ibl_tex, td = self._ktx_ibl(ctx, scn, self.look.room_ibl_ktx,
                                             self.look.room_ibl_ktx_intensity * self.look.exposure_scale)
        elif not self.look.env_file:                  # no environment: the HDR studio IBL that ships with MuJoCo
            ibl, ibl_tex, td = self._ktx_ibl(ctx, scn)
        ext = float(model.stat.extent)
        near, far = float(model.vis.map.znear) * ext, float(model.vis.map.zfar) * ext
        t1 = time.perf_counter()
        views = self._views(ctx, scn, raw_cameras, near, far)
        if self.look.post_tonemap:
            from .agx import regrade
            look = "punchy" if self.look.post_tonemap == "agx_punchy" else "none"
            for v in views:
                v.rgb = regrade(v.rgb, self.look.post_tonemap_gain / self.look.exposure_scale, look)
        t2 = time.perf_counter()
        if ibl is not None:
            scn.remove_light(ibl)
        del rends, lights, objs, scn, ibl, ibl_tex, td
        return self._finish(request, raw_cameras, cs, model, views, t0, t1, t2)

    def _ktx_ibl(self, ctx, scn, path=None, intensity=None):
        import mujoco._render_filament as mjrf
        if path:
            with open(path, "rb") as fh:
                raw = fh.read()
        else:
            raw = self._ibl_bytes()
        tc = mjrf.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, mjrf.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = mjrf.SamplerType.TEXTURE_CUBE, mjrf.ColorSpace.COLORSPACE_LINEAR
        ibl_tex = ctx.create_texture(tc)
        td = mjrf.TextureData()
        td.set_image(raw)
        ibl_tex.set_data(td)
        lp = mjrf.LightParams()
        lp.type, lp.texture, lp.color = mjrf.LightType.LIGHT_IMAGE, ibl_tex, [1.0, 1.0, 1.0]
        lp.intensity = float(self.look.ibl_intensity if intensity is None else intensity)
        ibl = ctx.create_light(lp)
        scn.add_light(ibl)
        return ibl, ibl_tex, td

    def _views(self, ctx, scn, raw_cameras, near, far):
        import mujoco._render_filament as mjrf
        views = []
        for cam in raw_cameras:
            target = ctx.create_render_target(mjrf.RenderTargetConfig(color_format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8))
            target.resize(cam.width, cam.height)
            rd = mjrf.ReadPixelsRequest()
            rd.target = target
            rd.alloc(cam.width * cam.height * 3)
            rq = mjrf.RenderRequest(camera=gl_camera(cam, near, far), draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT)
            rq.scene, rq.target = scn, target
            rq.viewport.left, rq.viewport.bottom, rq.viewport.width, rq.viewport.height = 0, 0, cam.width, cam.height
            ctx.wait_for_frame(ctx.render([rq], [rd]))
            rgb = np.frombuffer(rd.buffer(), np.uint8).reshape(cam.height, cam.width, 3).copy()
            views.append(RawView(rgb=rgb))
            del rq, rd, target
        return views

    def _finish(self, request, raw_cameras, cs, model, views, t0, t1, t2):
        outs = [o for o in raw_outputs(request.outputs, request.photo) if o in ("depth", "segmentation")]
        labels = cs.labels
        if outs:
            from render.backends.mujoco_backend import MujocoBackend
            with MujocoBackend().open(request.scene) as s:
                mv = s.render_raw_views(raw_cameras, outs)
                labels = s.labels
            for v, m in zip(views, mv):
                v.seg, v.depth = m.seg, m.depth
        facts = {"t_compile_s": round(t1 - t0, 4), "t_render_s": round(t2 - t1, 4), "look": cs.info.get("look")}
        return RawRender(views=views, labels=labels, resolved_scene_digest=MS.resolved_digest(model),
                         environment=self.environment(), facts=facts, lod=cs.info.get("lod"))


def register() -> None:
    """Register the backend under the name "filament" (idempotent)."""
    from render.backends import register_backend
    register_backend("filament", lambda **kw: FilamentBackend(**kw))
