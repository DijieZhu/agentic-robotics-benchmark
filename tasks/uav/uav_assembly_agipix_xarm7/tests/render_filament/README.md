# Task camera integration

The `render`, `render_filament`, and `room` helpers are vendored from the SO-101
Panda assembly task's rendering library. `../rendering.py` is the UAV task's
runtime entry point; it uses MuJoCo 3.17's `ModelResources`, `ReadRequest`, and
`Texture.upload` APIs.

Before importing MuJoCo, set `MUJOCO_GL=osmesa` and
`PYOPENGL_PLATFORM=osmesa`. System packages must include Mesa's Vulkan lavapipe
driver and OSMesa. The Python runtime needs MuJoCo 3.17, NumPy, and Pillow.
`CameraSession` removes `DISPLAY` and selects the lavapipe ICD explicitly. A
missing Filament module, driver, or room asset is an error; there is no classic
RGB fallback.

```python
from rendering import CameraSession

with CameraSession(cell) as camera:
    frame = camera.capture("top")
    # frame["rgb"]: uint8 H x W x 3
    # frame["depth_m"]: float32 H x W optical-axis metres; 0 is no hit
```

Both streams use the same compiled visible model and exact pinhole camera.
Collision geoms with group >= 3 and their unused meshes are removed from the
render copy. Joint state is copied by name without stepping or modifying the
source cell. Body-fixed camera mount position, quaternion and field of view
are read from `cell.m` on every capture, including commissioning draws applied
after compilation. Top is 1280 x 960; wrist is 640 x 480. `review` is a developer
view that the agent-facing server must not expose.

The required `<cell.assets>/render/room` and `render/oak` bundle is reused from
the SO-101 task payload in the Hugging Face dataset
`grounded-intelligence-initiative/agentic-robotics-last-exam`, revision
`33bb1eb0a87b0aaf3f29875898d03dc35078c6ae`, under
`tasks/arm/robotic_arm_full_assembly_so101_panda/payload/render`.
The 133 source files were SHA-256 verified when retrieved.
The runtime emits raw sensor arrays. The server owns sensor range,
noise, quantization, dropout and capture budgets.
