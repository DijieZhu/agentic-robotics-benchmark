"""Image build check: MuJoCo's Filament renderer starts on this image's Vulkan driver (Mesa's lavapipe, on the CPU)
for the account that runs it, draws a lit sphere and reads the picture back; the render bundle is readable.

    python3 -I render_check.py [bundle dir]

The cameras' renderer (door_render.py, beside the simulator in the agent's home and in the grader's folder) uses the
same calls. A Vulkan backend that cannot start aborts the process inside Filament: the build then fails here, not at
an episode's first image.
"""
import os
import sys

import numpy as np

for k in ("DISPLAY", "WAYLAND_DISPLAY"):
    os.environ.pop(k, None)
import mujoco  # noqa: E402
import mujoco._render_filament as F  # noqa: E402
from mujoco.experimental.studio import window  # noqa: E402,F401  registers Filament's resource providers


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DOOR_RENDER_DIR", "/opt/task-assets/payload/render")
    for name in ("textures.npz", "ibl.ktx", "room.npz", "room.json"):
        with open(os.path.join(bundle, name), "rb") as fh:
            assert fh.read(16), name
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size=".2" rgba=".8 .3 .2 1"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    W, H = 64, 48
    ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
    scn = ctx.create_scene(F.SceneParams())
    lp = F.LightParams()
    lp.type, lp.color, lp.intensity, lp.cast_shadow = F.LightType.LIGHT_DIRECTIONAL, [1.0, 1.0, 1.0], 100000.0, False
    light = ctx.create_light(lp)
    light.set_pose(np.array([0, 0, 2], np.float32), np.array([0, 0, -1], np.float32))
    scn.add_light(light)
    res = F.ModelResources(ctx, m)
    rends = F.ModelRenderables(scn, res)
    rends.update(d)
    buf = np.zeros((H, W, 3), np.uint8)
    tgt = ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
    tgt.resize(W, H)
    read = F.ReadRequest()
    read.target = tgt
    read.set_buffer(buf)
    cam = F.Camera(pos=np.array([0, 0, 1.5], np.float32), forward=np.array([0, 0, -1], np.float32),
                   up=np.array([0, 1, 0], np.float32), frustum_center=0.0, frustum_width=0.02, frustum_bottom=-0.015,
                   frustum_top=0.015, frustum_near=0.05, frustum_far=10.0)
    req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT, post_processing=True, shadows=False, reflections=False)
    req.scene, req.target = scn, tgt
    req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, W, H
    ctx.wait_for_frame(ctx.render([req], [read]))
    img = buf.copy()
    centre, corner = img[H // 2, W // 2].astype(int), img[1, 1].astype(int)
    assert centre[0] > 60 and centre[0] > centre[2] + 20 and int(np.abs(centre - corner).sum()) > 60, (centre, corner)
    print("[image] uid %d: Filament on Vulkan draws (centre %s, corner %s); render bundle readable" % (os.getuid(), centre.tolist(), corner.tolist()), flush=True)
    del rends, res
    scn.remove_light(light)
    del light, read, tgt, scn
    import gc
    gc.collect()
    del ctx
    os._exit(0)


if __name__ == "__main__":
    main()
