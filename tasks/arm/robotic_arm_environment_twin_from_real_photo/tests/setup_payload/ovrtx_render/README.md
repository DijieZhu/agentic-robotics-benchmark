# render_twin.py: render a MuJoCo scene with OVRTX path tracing

`render_twin.py` renders an MJCF scene through one of its cameras with NVIDIA OVRTX (RTX path tracing) on this
machine's GPU. The image has the camera's resolution and exact pinhole intrinsics, so you can lay it over the photo
and compare. It renders; it does not check or score anything.

```
python3 render_twin.py scene.xml --out render.png                      # through photo_cam, 64 samples per pixel
python3 render_twin.py scene.xml --out render.png --blend photo.png    # also writes render_blend.png (50 % mix)
python3 render_twin.py --warmup                                        # fill the shader cache (no image kept)
python3 render_twin.py --help
```

| option | meaning |
|---|---|
| `--camera NAME` | the camera to render through (default `photo_cam`) |
| `--spp N` | path-traced samples per pixel (default 64; 256 is smoother and about 1.5x slower) |
| `--size WxH` | render at another resolution instead of the camera's own (MuJoCo's rule for that size) |
| `--groups 0,1,2` | the geom groups to draw (default 0, 1 and 2, as MuJoCo's viewer) |
| `--dome-lux L` | uniform sky light in lux on an upward-facing surface (default 60; 0 = off; it is also the background) |
| `--exposure-ev S` | exposure offset in stops (+1 = twice as bright) |
| `--tonemap srgb\|ovrtx` | `srgb` (default): the linear exposed image, sRGB-encoded and clipped; `ovrtx`: OVRTX's own tone mapping |
| `--denoise` | run OVRTX's denoiser |
| `--blend PHOTO` | also write `<out>_blend.png`, a 50 % mix of the render and PHOTO (resized if the sizes differ) |
| `--keep DIR` | keep the USD stage (`scene.usdc`, `textures/`) and the raw outputs (`hdr.npy`: linear RGBA float32 after exposure; `ldr.npy`: OVRTX's 8-bit output) in DIR |
| `--timeout S` | seconds for the whole call, waiting for another render included (default 1800) |
| `--json` | print a one-line JSON summary at the end (timings, camera, mean colour, clipped share, warnings) |

## What is rendered

- The scene is compiled with MuJoCo and drawn in its initial state (`qpos0`, after `mj_forward`), not after a
  simulation.
- Every geom in the drawn groups whose colour alpha is above zero: planes, boxes, spheres, ellipsoids, cylinders,
  capsules and meshes, at their world poses. Meshes keep their own normals and texture coordinates (MuJoCo's
  `texrepeat` applies to them as in MuJoCo); primitives get generated texture coordinates.
- Materials: colour (`rgba`, or the geom's own `rgba` when it differs from the default), 2D textures for colour
  (`rgb` / `rgba` role or the `texture` attribute), normal, ORM, roughness, metallic and emissive maps, `roughness`,
  `metallic` (MuJoCo's `shininess` is converted when `roughness` is not set), `emission`, and alpha below 1 as
  opacity.
- Lights: spot and point lights emit `intensity` candela along their axis; directional lights give `intensity` lux.
  A light without `intensity` gets 1000 cd (spot, point) or 400 lux (directional) per unit of its brightest
  `diffuse` component. Spot cones follow `cutoff` and `exponent`; `castshadow` is honoured. Plus the dome light.
- The camera exposes like ISO 100, f/2, 1/25 s, shifted by `--exposure-ev`: about 0.01 linear units per nit
  (cd/m^2). A white matte surface lit by the default 60 lux dome alone comes out at about sRGB 120; it clips above
  about 330 lux.
- The camera: `fx`, `fy`, `cx`, `cy` exactly as MuJoCo computes them for that camera (with `sensorsize`:
  `fx = focalpixel_x`, `cx = (W - 1)/2 - principalpixel_x`, likewise for y; without it, `fovy` and a centred
  principal point). Pixel (0, 0) is the centre of the top-left pixel. Measured against an exact pinhole projection:
  within 0.05 px, also with fx != fy, an off-centre principal point and a rolled camera.

Not rendered, each with a note in the output: height-field and SDF geoms, flex objects and skins; cube and skybox
textures; occlusion, opacity and user-role material textures (the colour's alpha sets the opacity); image lights;
MuJoCo's headlight and ambient light (they have no physical counterpart: light the scene with MJCF lights or the
dome). A mesh without texture coordinates is drawn in its material's colour only. Planes of size 0 (infinite) are
drawn 100 m wide.

## Time, memory, one render at a time

- **First render.** With an empty OVRTX cache, the first render compiles shaders and materials for about 10 to
  25 minutes (on the CPU while the stage loads; longer when the machine is busy) and needs up to about 13 GB of
  memory meanwhile. Setup runs one warm-up render before you start, so the cache is normally warm already:
  `/tmp/render_twin_warmup.json` then says `"status": "ok"`. If it says anything else, start your first render in
  the background (`python3 render_twin.py scene.xml --out first.png > first.log 2>&1 &`) and keep working; it prints
  a progress line every 30 s. Once the cache is warm, a scene's own new materials add only seconds. Renders are
  headless (the OVRTX process gets no display), so the warm cache serves any shell of this account.
- **Later renders** of a 1920 x 1080 image take about 15 s at 64 spp and about 22 s at 256 spp, of which about 4 s is
  the renderer start.
- **GPU memory:** about 3 GB per render.
- **One render at a time.** Two OVRTX processes on one GPU can hang it, so each render holds a lock file
  (`/tmp/render_twin.lock`) from just before OVRTX starts until the OVRTX process has exited. A second render waits
  (it prints that it is waiting) and counts the wait against its `--timeout`.
- **Stopping it.** The OVRTX process belongs to `render_twin.py`: Ctrl-C, SIGTERM, SIGKILL of `render_twin.py`, or a
  signal to its process group (a tool timeout) stops the renderer too, and the lock is released once the renderer is
  really gone. Nothing of a render keeps running after `render_twin.py` has ended.
- **Determinism:** the same scene with the same options renders to the same PNG, byte for byte.

## Exit codes

| code | meaning | what to do |
|---|---|---|
| 0 | rendered | |
| 2 | the scene cannot be rendered: it does not compile (MuJoCo's message is printed), the camera is missing, orthographic or has no resolution, or the arguments are wrong | fix the scene or the call |
| 3 | OVRTX cannot start here: no NVIDIA GPU device, a driver without the Vulkan graphics OVRTX needs, or a Python without the helper's modules | retrying will not help; work without renders |
| 4 | the render failed (the message and the end of OVRTX's log are printed) | look at the message; a retry may help |
| 5 | `--timeout` ran out, while waiting for another render or while rendering (the renderer was stopped) | wait for the other render, or allow more time |
| 130, 143 | stopped by Ctrl-C or SIGTERM | |

`render_twin.py` needs a Python with numpy, MuJoCo, usd-core and Pillow. Started with another `python3`, it runs
itself again with the task's Python (`/opt/venv/bin/python3`, or `$RENDER_TWIN_PYTHON`), and exits 3 if there is none.

## Files and settings

- `render_twin.py`: the command; process 1 (the task's Python) compiles the scene and writes the USD stage.
- `ovrtx_render/camera.py`: MuJoCo camera to pinhole intrinsics. `shapes.py`: triangle meshes of the geoms.
  `export_usd.py`: the USD stage (geometry, materials, lights, camera, render settings). `ovrtx_job.py`: process 2,
  run with the OVRTX Python, which path-traces the stage and saves the raw outputs. `warmup.py`: the warm-up scene.
- `ARB_OVRTX_PYTHON`: the OVRTX Python (default `/opt/ovrtx-venv/bin/python`). `RENDER_TWIN_PYTHON`: the task's
  Python. `RENDER_TWIN_LOCK`: the lock file.
- Keep renders and `--keep` folders outside the submission folder.
