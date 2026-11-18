"""Registered RGB-D views of the physical xArm/AGiPix assembly cell.

RGB uses MuJoCo Filament on CPU Vulkan; optical-axis depth uses OSMesa on
exactly the same compiled visible model and pinhole. The render copy never
steps physics or changes the source cell. Capture returns raw sensor arrays;
the server owns range limits, quantization, noise, dropouts and frame budgets.

Set MUJOCO_GL=osmesa before importing MuJoCo. No classic RGB fallback exists.
The factory/oak bundle must be present in <cell.assets>/render (or render_dir).
"""
from __future__ import annotations

import atexit
import gc
import glob
import math
import os
from pathlib import Path
import time

import numpy as np

SIZES = {"top": (1280, 960), "wrist": (640, 480), "review": (1280, 960)}
_OPEN = set()
_LUT = None


def _headless_cpu():
    for key in ("DISPLAY", "WAYLAND_DISPLAY"):
        os.environ.pop(key, None)
    candidates = sorted(glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json"))
    if not candidates:
        raise RuntimeError("Filament CPU rendering requires Mesa lavapipe (mesa-vulkan-drivers)")
    os.environ["VK_ICD_FILENAMES"] = candidates[0]
    os.environ["VK_DRIVER_FILES"] = candidates[0]


def _shutdown():
    for session in list(_OPEN):
        session.close()
    gc.collect()
    from render_filament import backend
    backend._CTX.clear()
    gc.collect()


atexit.register(_shutdown)


def _lut(look):
    global _LUT
    if _LUT is None:
        from render_filament.agx import regrade
        gain = look.post_tonemap_gain / look.exposure_scale
        v = np.arange(256, dtype=np.uint8)
        _LUT = np.empty((256 ** 3, 3), np.uint8)
        for red in range(256):
            rgb = np.stack(np.broadcast_arrays(np.uint8(red), v[:, None], v[None, :]), -1).reshape(-1, 3)
            _LUT[red * 65536:(red + 1) * 65536] = regrade(rgb, gain, "punchy").reshape(-1, 3)
    return _LUT


def _bench_frame(spec):
    import mujoco
    # Render-only steel structure below the physical 1.6 x 1.5 m tabletop.
    mat = spec.add_material(name="render_bench_steel", rgba=[.26, .29, .32, 1])
    def box(name, pos, size):
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=pos, size=size, material=mat.name,
                                contype=0, conaffinity=0, group=2)
    for i, x in enumerate((-.73, .73)):
        for j, y in enumerate((-.68, .68)):
            box(f"render_bench_leg_{i}{j}", [x, y, -.52], [.025, .025, .474])
    for i, y in enumerate((-.68, .68)):
        box(f"render_bench_rail_{i}", [0, y, -.82], [.73, .025, .025])


def _make_model(cell, render_dir):
    import mujoco
    from room import add_room
    from render_filament.backend import Look, apply_look
    spec = mujoco.MjSpec.from_string(cell.xml)
    # Collision geoms are excluded identically from both image streams.
    for geom in list(spec.geoms):
        if int(geom.group) >= 3:
            spec.delete(geom)
    used_meshes = {g.meshname for g in spec.geoms if g.meshname}
    for mesh in list(spec.meshes):
        if mesh.name not in used_meshes:
            spec.delete(mesh)
    table = spec.geom("table")
    table_mat = spec.add_material(name="render_table_oak", rgba=[1, 1, 1, 1])
    table.material = table_mat.name
    table.rgba = [1, 1, 1, 1]
    _bench_frame(spec)
    rec = add_room(spec, str(render_dir / "room"), bench_xy=(0., 0.))
    lamps = sorted([l for l in spec.lights if (l.name or "").startswith("room_light_")],
                   key=lambda l: math.hypot(l.pos[0], l.pos[1]))
    keep = {l.name for l in lamps[:6]}
    for light in list(spec.lights):
        if light.name not in keep:
            spec.delete(light)
        else:
            light.range = 40.
    look = Look(name="agipix-factory", version="1")
    # Preserve the imported component's material regions in both human and
    # agent views. Mesh and contact geometry remain independent.
    look.materials = []
    for name, inst in {"base": {"kind": "base"}, **cell.graph["instances"]}.items():
        part = cell.kit["parts"][inst["kind"]]
        for i, visual in enumerate(part.get("visual_parts", [])):
            look.materials.append((f"visual_{name}_{i}", None,
                                   dict(roughness=visual.get("roughness", .52),
                                        metallic=visual.get("metallic", 0.))))
        for i, visual in enumerate(part.get("rotor", {}).get("visual_parts", [])):
            look.materials.append((f"visual_{name}_rotor_{i}", None,
                                   dict(roughness=visual.get("roughness", .38),
                                        metallic=visual.get("metallic", .6))))
    look.materials += [
        ("visual_prop_*", None, dict(roughness=.38, metallic=0.)),
        ("visual_*", None, dict(roughness=.52, metallic=0.)),
        ("fixture", None, dict(roughness=.45, metallic=.15)),
        ("stand_*", None, dict(roughness=.55, metallic=0.)),
        (None, "render_table_oak", dict(roughness=1., metallic=0.)),
        (None, "render_bench_steel", dict(roughness=.4, metallic=.7)),
        (None, "white", dict(roughness=.38, metallic=0.)),
        (None, "gray", dict(roughness=.3, metallic=.7)),
        (None, None, dict(roughness=.6, metallic=0.)),
    ]
    # Same calibrated baked-room look as the prior physical assembly task.
    look.exposure_scale = .35
    look.post_tonemap, look.post_tonemap_gain = "agx_punchy", 1.9
    look.room_baked_emission, look.room_baked_light_scale = .51, 6500.
    look.room_ibl_ktx_intensity, look.room_bulb_m = 12000., .6
    look.surface_dir = str(render_dir / "oak")
    look.surface_material, look.surface_geom = table_mat.name, "table"
    look.surface_albedo_gain = .9
    look.room_ibl_ktx = str(render_dir / "room/ibl/ibl.ktx")
    apply_look(spec, look)
    model = spec.compile()
    # Exact common clip planes, independent of the surrounding factory extent.
    model.vis.map.znear = .01 / float(model.stat.extent)
    model.vis.map.zfar = 80. / float(model.stat.extent)
    model.vis.global_.offwidth = max(w for w, h in SIZES.values())
    model.vis.global_.offheight = max(h for w, h in SIZES.values())
    return model, look, rec


class CameraSession:
    """One reusable render session per episode. `review` is developer-only.

    `capture(name)` returns `{rgb, depth_m}`; neither intrinsics, object poses,
    segmentation labels nor contact/mate states are exposed by that method.
    """
    def __init__(self, cell, render_dir=None):
        _headless_cpu()
        import mujoco
        import mujoco._render_filament as filament
        from render_filament.backend import _context, FilamentBackend
        self.cell, self._mj, self._filament = cell, mujoco, filament
        self.scene = None
        self._targets, self._depth = {}, {}
        self.ibl = self.objects = self.lights = self.renderables = None
        self.closed = False
        _OPEN.add(self)  # exception-safe cleanup if compilation/resource creation fails
        render_dir = Path(render_dir or cell.assets / "render")
        for path in (render_dir / "room/room.json", render_dir / "room/ibl/ibl.ktx"):
            if not path.is_file():
                raise FileNotFoundError(f"Required camera rendering asset missing: {path}")
        started = time.perf_counter()
        self.model, self.look, self.room_record = _make_model(cell, render_dir)
        self.data = mujoco.MjData(self.model)
        self._joint_map = []
        for joint in range(cell.m.njnt):
            name = cell.m.joint(joint).name
            target = self.model.joint(name).id
            width = {0: 7, 1: 4, 2: 1, 3: 1}[int(cell.m.jnt_type[joint])]
            self._joint_map.append((int(cell.m.jnt_qposadr[joint]), int(self.model.jnt_qposadr[target]), width))
        self.context = _context("vulkan")
        self.objects = filament.ModelResources(self.context, self.model)
        self.scene = self.context.create_scene(filament.SceneParams())
        self.scene.configure_from_model(self.model)
        self.lights = filament.ModelLights(self.scene, self.objects)
        self.renderables = filament.ModelRenderables(self.scene, self.objects)
        self.ibl = FilamentBackend(look=self.look)._ktx_ibl(
            self.context, self.scene, self.look.room_ibl_ktx,
            self.look.room_ibl_ktx_intensity * self.look.exposure_scale)
        self.lut = _lut(self.look)
        self.sync()
        self.open_seconds = time.perf_counter() - started
        self.last_timing = {}

    def sync(self):
        if self.closed:
            raise RuntimeError("Camera session is closed")
        for source, target, width in self._joint_map:
            self.data.qpos[target:target + width] = self.cell.d.qpos[source:source + width]
        self._mj.mj_kinematics(self.model, self.data)
        self._mj.mj_comPos(self.model, self.data)
        self._mj.mj_camlight(self.model, self.data)
        self.lights.update(self.data)
        self.renderables.update(self.data)

    def _camera(self, name):
        from render.spec import CameraSpec
        if name not in SIZES:
            raise ValueError(f"Unknown camera {name!r}")
        # Read source-model camera commissioning draws at capture time. This
        # supports hidden mount/FOV changes made after XML compilation without
        # copying stale source camera world transforms from the last mj_step.
        source = self.cell.m.camera(name).id
        if int(self.cell.m.cam_mode[source]) != 0:
            raise ValueError("Only body-fixed cameras are supported")
        body = self.cell.m.body(int(self.cell.m.cam_bodyid[source])).name
        target = self.model.body(body).id if body else 0
        body_rotation = self.data.xmat[target].reshape(3, 3)
        local_rotation = np.empty(9)
        self._mj.mju_quat2Mat(local_rotation, self.cell.m.cam_quat[source])
        rotation = body_rotation @ local_rotation.reshape(3, 3)
        position = self.data.xpos[target] + body_rotation @ self.cell.m.cam_pos[source]
        pose = np.eye(4)
        pose[:3, :3] = rotation @ np.diag([1., -1., -1.])
        pose[:3, 3] = position
        width, height = SIZES[name]
        focal = .5 * height / math.tan(math.radians(float(self.cell.m.cam_fovy[source])) / 2)
        return CameraSpec(name=name, width=width, height=height, fx=focal, fy=focal,
                          cx=(width - 1) / 2, cy=(height - 1) / 2,
                          world_T_cam=pose.tolist())

    def _rgb(self, camera):
        from render_filament.backend import gl_camera
        f = self._filament
        key = camera.width, camera.height
        if key not in self._targets:
            target = self.context.create_render_target(f.RenderTargetConfig(color_format=f.PixelFormat.PIXEL_FORMAT_RGB8))
            target.resize(*key)
            pixels = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
            read = f.ReadRequest(target=target)
            read.set_buffer(pixels)
            self._targets[key] = target, read, pixels
        target, read, raw = self._targets[key]
        request = f.RenderRequest(camera=gl_camera(camera, .01, 80.), draw_mode=f.DrawMode.DRAW_MODE_DEFAULT)
        request.scene, request.target = self.scene, target
        request.viewport.left, request.viewport.bottom = 0, 0
        request.viewport.width, request.viewport.height = key
        self.context.wait_for_frame(self.context.render([request], [read]))
        index = (raw[..., 0].astype(np.int32) << 16) | (raw[..., 1].astype(np.int32) << 8) | raw[..., 2]
        return self.lut[index]

    def _depth_m(self, camera):
        from render.backends.mujoco_backend import set_scene_camera
        key = camera.width, camera.height
        if key not in self._depth:
            renderer = self._mj.Renderer(self.model, height=camera.height, width=camera.width,
                                        max_geom=max(10000, self.model.ngeom + 100))
            renderer.enable_depth_rendering()
            self._depth[key] = renderer
        renderer = self._depth[key]
        renderer.update_scene(self.data, camera=camera.name)
        set_scene_camera(renderer.scene, camera, self.model)
        result = renderer.render().copy()
        # No-hit sentinel is zero; sensor limits/noise are applied by the server.
        result[(result >= 79.99) | ~np.isfinite(result)] = 0.
        return result

    def capture(self, name):
        started = time.perf_counter()
        self.sync()
        camera = self._camera(name)
        rgb = self._rgb(camera)
        middle = time.perf_counter()
        depth = self._depth_m(camera)
        ended = time.perf_counter()
        self.last_timing = dict(camera=name, rgb_s=middle - started, depth_s=ended - middle,
                                total_s=ended - started)
        return {"rgb": rgb, "depth_m": depth}

    def close(self):
        if self.closed:
            return
        self.closed = True
        _OPEN.discard(self)
        for renderer in self._depth.values():
            renderer.close()
        self._depth.clear()
        if self.ibl is not None and self.scene is not None:
            self.scene.remove_light(self.ibl[0])
        self.ibl = None
        self._targets.clear()
        self.renderables = self.lights = self.scene = self.objects = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
