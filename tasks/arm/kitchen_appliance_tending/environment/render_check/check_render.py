"""Build-time check of the cameras' renderer (environment/Dockerfile runs it as root, as the agent account and as the
lab's account): practice unit p0 (stovetops/Stovetop018, from the task assets) is reset with a fixed seed and its
three images are drawn from the room bundle (KITCHEN_RENDER_DIR, default /opt/task-assets/payload/render) by the
task's own modules beside this file (copies of environment/setup/payload's appliance_env.py, kitchen_render.py,
kitchen_fx.py, kitchen_scene.py and kitchen_look.json). Exits non-zero, and so fails the build, when an image is missing, has the
wrong shape, is flat or is far too dark or too bright.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import appliance_env as E  # noqa: E402

UNITS = "/opt/task-assets/payload/units"
SHAPES = {"left": (240, 320, 3), "right": (240, 320, 3), "wrist": (192, 256, 3)}


def main():
    t0 = time.monotonic()
    spec = E.unit_spec("stovetops/Stovetop018", files=E.read_unit_files(os.path.join(UNITS, "Stovetop018")),
                       quirks=["push_to_turn"])
    env = E.ApplianceEnv(spec, None, render=True)
    obs = env.reset(1)
    report = []
    for cam, shape in sorted(SHAPES.items()):
        img = obs[cam]
        mean, std = float(img.mean()), float(img.std())
        if tuple(img.shape) != shape or img.dtype != E.np.uint8 or std < 8.0 or not 20.0 <= mean <= 235.0:
            print("[image] kitchen renderer check FAILED: {} image {} {} mean {:.1f} std {:.1f}".format(
                cam, img.shape, img.dtype, mean, std), file=sys.stderr)
            return 1
        report.append("{} mean {:.0f} std {:.0f}".format(cam, mean, std))
    env.close()
    print("[image] kitchen renderer OK as uid {}: p0's three images from {} ({}; {:.1f} s)".format(
        os.getuid(), E._KR.RENDER_DIR, ", ".join(report), time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
