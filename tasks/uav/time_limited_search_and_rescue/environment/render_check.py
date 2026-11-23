"""Image build check: MuJoCo's Filament renderer starts on this image's Vulkan driver (Mesa's lavapipe, on the CPU),
lights a textured rectangle with the render bundle's image light and reads the picture back.

    python3 -I render_check.py [bundle dir]

The aircraft's cameras (rescue_render.py, in the aircraft service's root-only folder and in the grader's) use the same
calls. A Vulkan backend that cannot start aborts the process inside Filament: the build then fails here, not at a
site's first image.
"""
import gc
import os
import sys

import numpy as np

for k in ("DISPLAY", "WAYLAND_DISPLAY"):
    os.environ.pop(k, None)
os.environ["MESA_SHADER_CACHE_DISABLE"] = "true"
import mujoco  # noqa: E402,F401
import mujoco._render_filament as F  # noqa: E402
from mujoco.experimental.studio import window  # noqa: E402,F401  registers Filament's resource providers
from PIL import Image  # noqa: E402


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESCUE_RENDER_DIR", "/opt/task-assets/payload/render")
    names = sorted(os.listdir(bundle))
    assert "ibl.ktx" in names and len([n for n in names if n.endswith(".png")]) >= 10, names
    ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
    scn = ctx.create_scene(F.SceneParams())
    with open(os.path.join(bundle, "ibl.ktx"), "rb") as fh:
        raw = fh.read()
    tc = F.TextureConfig()
    tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
    tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_CUBE, F.ColorSpace.COLORSPACE_LINEAR
    ibl_tex = ctx.create_texture(tc)
    td = F.TextureData()
    td.set_image(raw)
    ibl_tex.upload(td)
    lp = F.LightParams()
    lp.type, lp.texture, lp.color, lp.intensity = F.LightType.LIGHT_IMAGE, ibl_tex, [1.0, 1.0, 1.0], 30000.0
    ibl = ctx.create_light(lp)
    scn.add_light(ibl)
    img = np.ascontiguousarray(np.asarray(Image.open(os.path.join(bundle, "paint_b830db.png")).convert("RGB")))
    tc = F.TextureConfig()
    tc.width, tc.height, tc.format = img.shape[1], img.shape[0], F.PixelFormat.PIXEL_FORMAT_RGB8
    tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_2D, F.ColorSpace.COLORSPACE_SRGB
    tex = ctx.create_texture(tc)
    td = F.TextureData()
    td.set_image(img.tobytes())
    tex.upload(td)
    A, Us, Ty = F.VertexAttribute, F.VertexAttributeUsage, F.VertexAttributeType
    cfg = F.MeshConfig(max_vertices=4, max_indices=6, interleaved=False, index_type=F.IndexType.INDEX_TYPE_U32,
                       primitive_type=F.MeshPrimitiveType.MESH_PRIMITIVE_TYPE_TRIANGLES,
                       attributes=[A(usage=Us.VERTEX_ATTRIBUTE_USAGE_POSITION, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                                   A(usage=Us.VERTEX_ATTRIBUTE_USAGE_NORMAL, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                                   A(usage=Us.VERTEX_ATTRIBUTE_USAGE_UV, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT2)])
    mesh = ctx.create_mesh(cfg)
    V = np.array([(-.3, -.3, 0), (.3, -.3, 0), (.3, .3, 0), (-.3, .3, 0)], np.float32)
    N = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
    UV = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], np.float32)
    md = F.MeshData(compute_bounds=True)
    md.set_vertices(cfg, np.concatenate([V.ravel(), N.ravel(), UV.ravel()]).tobytes())
    md.set_indices(cfg, np.array([0, 1, 2, 0, 2, 3], np.uint32).tobytes())
    mesh.upload(md)
    mat = F.Material(color=np.ones(4, np.float32), roughness=0.7, metallic=0.0, specular=0.1, glossiness=0.05,
                     reflectance=0.0, emissive=0.0, color_texture=tex)
    rend = ctx.create_renderable(F.RenderableParams(cast_shadow=False, receive_shadow=True))
    rend.set_mesh(mesh, 0, 6)
    rend.set_material(mat)
    scn.add_renderable(rend)
    W = H = 64
    buf = np.zeros((H, W, 3), np.uint8)
    tgt = ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
    tgt.resize(W, H)
    read = F.ReadRequest()
    read.target = tgt
    read.set_buffer(buf)
    cam = F.Camera(pos=np.array([0, 0, 1.0], np.float32), forward=np.array([0, 0, -1], np.float32),
                   up=np.array([0, 1, 0], np.float32), frustum_center=0.0, frustum_width=0.02, frustum_bottom=-0.02,
                   frustum_top=0.02, frustum_near=0.05, frustum_far=10.0)
    req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT, post_processing=True, shadows=False,
                          reflections=False)
    req.scene, req.target = scn, tgt
    req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, W, H
    ctx.wait_for_frame(ctx.render([req], [read]))
    centre = buf[H // 2, W // 2].astype(int)
    # the violet floor paint under the bundle's light: red and blue well over green
    assert centre[0] > centre[1] + 40 and centre[2] > centre[1] + 40 and centre.max() > 120, centre
    print("[image] uid %d: Filament on Vulkan draws with the render bundle (%d files; centre %s)"
          % (os.getuid(), len(names), centre.tolist()), flush=True)
    scn.remove_renderable(rend)
    scn.remove_light(ibl)
    del rend, mat, mesh, tex, ibl, ibl_tex, read, tgt, scn
    gc.collect()
    del ctx
    os._exit(0)


if __name__ == "__main__":
    main()
