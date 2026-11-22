"""Render a mission video from a trajectory record of the rubble cell (out/traj_<tag>.npz under /opt/arb-verify of a
kept verifier sandbox: time, qpos and eq_active samples at 4 Hz). Maintainers' review tool; nothing runs it.

usage: python3 tests/review/replay.py TRAJ.npz OUT.mp4 [--title TEXT] [--size 1600x900] [--fps 24] [--every 1]
                                       [--intro 6] [--crf 24]
       (needs mujoco with its Filament module, numpy, Pillow, ffmpeg and a Vulkan driver: a GPU's, or Mesa's lavapipe
       at a few seconds a frame; the look's texture sets are made on first use, see rubblecell/pbr.py)

The grader saves each episode's state at 4 Hz (time, qpos, the welds' activity); this rebuilds the site from its seed,
dresses it with the look's video quality (look.SceneLook: the render twin unbatched, so the storeys over the robot can
be cut away) and films it with MuJoCo's Filament renderer: an establishing crane shot down the street, then a chase
camera that follows the robot, the storeys above it cut away while it is inside, the robot's mast camera inset, the
frame letterboxed with a vignette and a light film grain. One video frame per state sample: 4 Hz of simulated time at
24 frames per second plays 6x faster than the mission.
"""
import argparse
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # tests/ (rubblecell)
for k in ("DISPLAY", "WAYLAND_DISPLAY"):
    os.environ.pop(k, None)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rubblecell.look import SceneLook  # noqa: E402
from rubblecell.plan import draw_plan  # noqa: E402
from rubblecell.scene import build_scene  # noqa: E402


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def look_at(pos, target, roll_up=(0.0, 0.0, 1.0)):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    up = np.asarray(roll_up, float)
    up = up - f * up.dot(f)
    up /= np.linalg.norm(up)
    return pos, f, up


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--intro", type=float, default=6.0)
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    W, H = (int(x) for x in a.size.split("x"))
    z = np.load(a.traj, allow_pickle=True)
    traj = z["traj"]
    seed, kind = int(z["seed"]), str(z["kind"])
    plan = draw_plan(seed, kind)
    sc = build_scene(plan)
    m = sc.model
    d = mujoco.MjData(m)
    nq, neq = m.nq, m.neq
    t0 = time.time()
    look = SceneLook(sc, quality="video")
    print("[film] look built in %.1f s" % (time.time() - t0), flush=True)
    tw = look.twins
    tm, td = tw.rgb_model, tw.rgb_data
    S = sc.site
    ch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    # the static geoms the cut-away may hide: bottom height of each (the twin at rest)
    mujoco.mj_kinematics(tm, td)
    static = np.array([int(tm.body_rootid[tm.geom_bodyid[i]]) == 0 for i in range(tm.ngeom)])
    bottom = np.empty(tm.ngeom)
    inside_fp = np.zeros(tm.ngeom, bool)
    for i in range(tm.ngeom):
        R = td.geom_xmat[i].reshape(3, 3)
        half = np.asarray(tm.geom_aabb[i][3:], float)
        c = td.geom_xpos[i] + R @ np.asarray(tm.geom_aabb[i][:3], float)
        bottom[i] = c[2] - float(np.abs(R[2]) @ half)
        inside_fp[i] = (-0.4 < c[0] < S.W + 0.4) and (-0.4 < c[1] < S.D + 0.4)
    st = plan["start"]
    yaw0 = float(st["yaw"])
    cam_pos = None
    cam_tgt = None
    frames = list(range(0, len(traj), a.every))
    n_intro = int(a.intro * a.fps)
    total = n_intro + len(frames)
    if a.max_frames:
        total = min(total, a.max_frames)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    bar = int((H - W / 2.2) / 2)
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1.0 - 0.32 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.4
    vig = np.clip(vig, 0.55, 1.0)[..., None].astype(np.float32)
    grain_rng = np.random.default_rng(7)
    f_title, f_cap = font(int(H * 0.032)), font(int(H * 0.022))
    iw, ih = int(W * 0.26), int(W * 0.26 * 0.75)
    tstart = time.time()
    for n in range(total):
        k = frames[0] if n < n_intro else frames[min(n - n_intro, len(frames) - 1)]
        row = traj[k]
        d.qpos[:] = row[1:1 + nq]
        if neq:
            d.eq_active[:] = row[1 + nq:1 + nq + neq] > 0.5
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        tw.sync(d)
        p = d.xpos[ch].copy()
        R = d.xmat[ch].reshape(3, 3)
        hd = R[:, 0].copy()
        hd[2] = 0.0
        hd /= max(1e-6, np.linalg.norm(hd))
        side = np.array([-hd[1], hd[0], 0.0])
        inside = (0.15 < p[0] < S.W - 0.15) and (0.15 < p[1] < S.D - 0.15)
        # the camera: the crane down the street at first, then the chase
        if inside:
            want_pos = p - hd * 2.4 + side * 0.9 + np.array([0, 0, 3.6])
            want_tgt = p + hd * 0.6 + np.array([0, 0, 0.15])
        else:
            want_pos = p - hd * 3.4 + side * 1.4 + np.array([0, 0, 1.9])
            want_tgt = p + hd * 1.0 + np.array([0, 0, 0.35])
        if n < n_intro:
            u = smoothstep(n / max(1, n_intro - 1))
            crane = np.array([p[0] - 9.0, -9.5, 9.0])
            crane_t = np.array([p[0] + 2.0, 6.0, 4.0])
            cam_pos = crane * (1 - u) + want_pos * u
            cam_tgt = crane_t * (1 - u) + want_tgt * u
        else:
            al = 0.10
            cam_pos = cam_pos * (1 - al) + want_pos * al
            cam_tgt = cam_tgt * (1 - al) + want_tgt * al
        # the cut-away: the storeys (and a dropped slab) over the robot while it is inside the building
        if inside:
            cut = static & inside_fp & (bottom > p[2] + 0.45)
            td.geom_xmat[cut] = 0.0
        pos, fwd, up = look_at(cam_pos, cam_tgt)
        img = look.renderer.view(d, pos, fwd, up, 52.0, W, H, ssaa=1, gain=1.0 if not inside else 1.4)
        inset = look.renderer.camera(d, "mast_rgbd", iw, ih)
        frame = img.astype(np.float32) * vig
        frame += grain_rng.normal(0.0, 2.2, frame.shape).astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame[:bar] = 0
        frame[H - bar:] = 0
        im = Image.fromarray(frame)
        x0, y0 = W - iw - int(W * 0.02), H - bar - ih - int(H * 0.03)
        dr = ImageDraw.Draw(im)
        dr.rectangle([x0 - 3, y0 - 3, x0 + iw + 2, y0 + ih + 2], outline=(235, 235, 235), width=2)
        im.paste(Image.fromarray(inset), (x0, y0))
        dr.text((x0 + 8, y0 + 6), "mast camera", fill=(240, 240, 240), font=f_cap)
        if a.title:
            dr.text((int(W * 0.03), bar + int(H * 0.025)), a.title, fill=(245, 245, 245), font=f_title)
        dr.text((int(W * 0.03), H - bar + int(H * 0.012)), "t = %5.1f s  (x%d)" % (row[0], max(1, round(a.fps / 4 * a.every))),
                fill=(210, 210, 210), font=f_cap)
        ff.stdin.write(np.asarray(im).tobytes())
        if n % 120 == 0:
            el = time.time() - tstart
            print("[film] frame %d/%d  %.2f s/frame" % (n, total, el / max(1, n)), flush=True)
    ff.stdin.close()
    ff.wait()
    look.close()
    print("[film] wrote %s %d frames, %.1f MB" % (a.out, total, os.path.getsize(a.out) / 1e6), flush=True)


if __name__ == "__main__":
    main()
