"""Generator of the agent's recordings for cloth_material_id_from_video (maintainers' side; in no image layer).

Run once by the maintainers; the recordings are published in the asset repository and the task image fetches them
by checksum (environment/assets.sha256). For every fabric (P, A ... J) and clip it simulates the clip on the real
fabric (creal: the real fabric's model, which the agent does not get; its cphys is the robot's planning simulator),
solves the arm's joint angles (carm), lays the drawn fabric on the real model's vertices (csheet), draws the three
calibrated cameras with the Filament renderer of the approved scene (crender, cscene and the render bundle at
CLOTH_RENDER_DIR; lavapipe, no GPU), degrades the pictures like a real camera and encodes H.264 with ffmpeg.
Output tree (OUT):

  data/cameras.json
  data/fabrics/<ID>/fabric.json                         size, grid, mass, number of clips
  data/fabrics/<ID>/clipNN/{clip.json, clamp_500hz.npy, arm_q.npy, cam0.mp4, cam1.mp4, cam2.mp4}

Everything here is deterministic: fixed seeds, fixed clip programs, one clip per process. A fabric's look is a
function of its id alone (cscene.fabric_looks), never of its material; every fabric is drawn 1.5 mm thick.
usage: python3 render_clips.py OUT [--workers N] [--only P,A] [--frames-max K]      (workers: cores / 4)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cphys as W  # noqa: E402
import creal as R  # noqa: E402
import ctasks as T  # noqa: E402
import csheet as CS  # noqa: E402

W_IMG, H_IMG = 640, 480
FOCAL = 640.0
CAMERAS = [  # name, position, look-at target (world)
    ("cam0", (0.60, 0.70, 0.65), (0.10, -0.10, 0.10)),
    ("cam1", (0.85, 0.32, 0.22), (0.20, 0.0, 0.0)),
    ("cam2", (-0.70, 0.55, 0.62), (-0.12, 0.0, 0.12)),
]


# ----------------------------------------------------------------------------------------------------------------
# cameras
# ----------------------------------------------------------------------------------------------------------------
def look_at(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_w2c = np.stack([x, y, z])
    return R_w2c, -R_w2c @ pos


def cameras_json():
    cams = []
    for name, pos, tgt in CAMERAS:
        Rw, t = look_at(pos, tgt)
        K = [[FOCAL, 0.0, W_IMG / 2], [0.0, FOCAL, H_IMG / 2], [0.0, 0.0, 1.0]]
        cams.append(dict(name=name, width=W_IMG, height=H_IMG, K=K, dist=[0, 0, 0, 0, 0], R_w2c=Rw.tolist(),
                         t_w2c=t.tolist(), position_world=list(map(float, pos))))
    return {"cameras": cams, "model": "OpenCV pinhole, pixel ~ K (R_w2c X + t_w2c), no distortion; world z up, metres"}


# ----------------------------------------------------------------------------------------------------------------
# procedural textures (no external assets)
# ----------------------------------------------------------------------------------------------------------------
def _noise(rng, n, scale):
    g = rng.random((n // scale + 2, n // scale + 2))
    from scipy.ndimage import zoom
    return zoom(g, scale, order=3)[:n, :n]


def fabric_texture(kind, seed, n=512):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    if kind == "plaid":            # P: red / cream plaid
        a = (np.sin(2 * np.pi * 9 * xx) > 0.55).astype(float) + 0.6 * (np.sin(2 * np.pi * 27 * xx) > 0.8)
        b = (np.sin(2 * np.pi * 5 * yy) > 0.55).astype(float) + 0.6 * (np.sin(2 * np.pi * 21 * yy) > 0.8)
        m = np.clip(0.5 * a + 0.5 * b, 0, 1)[..., None]
        img = (1 - m) * np.array([0.93, 0.88, 0.78]) + m * np.array([0.72, 0.12, 0.14])
    elif kind == "floral":         # A: light blue with scattered blots
        img = np.ones((n, n, 3)) * np.array([0.70, 0.82, 0.93])
        for _ in range(140):
            cx, cy, r = rng.random(), rng.random(), rng.uniform(0.012, 0.04)
            col = np.array([[0.95, 0.95, 0.98], [0.30, 0.40, 0.75], [0.95, 0.75, 0.30]][rng.integers(3)])
            d = np.hypot(xx - cx, yy - cy)
            img[d < r] = col
    elif kind == "twill":          # B: dark blue twill with a woven diagonal and blotches, white stitch frame
        diag = 0.5 + 0.5 * np.sin(2 * np.pi * 60 * (xx + yy))
        blot = _noise(rng, n, 32)
        v = 0.55 + 0.25 * diag + 0.35 * (blot - 0.5)
        img = v[..., None] * np.array([0.22, 0.32, 0.55])
        for _ in range(60):
            cx, cy, r = rng.random(), rng.random(), rng.uniform(0.01, 0.03)
            img[np.hypot(xx - cx, yy - cy) < r] = [0.85, 0.80, 0.55]
        e = (np.minimum(np.minimum(xx, 1 - xx), np.minimum(yy, 1 - yy)) < 0.02)
        img[e] = [0.9, 0.75, 0.35]
    else:                          # C: green / grey knit stripes with speckles
        s = (np.sin(2 * np.pi * 14 * xx) > 0).astype(float)
        img = (1 - s[..., None]) * np.array([0.35, 0.55, 0.35]) + s[..., None] * np.array([0.78, 0.78, 0.75])
        sp = rng.random((n, n)) < 0.01
        img[sp] = [0.1, 0.1, 0.1]
        for _ in range(50):
            cx, cy, r = rng.random(), rng.random(), rng.uniform(0.01, 0.025)
            img[np.hypot(xx - cx, yy - cy) < r] = [0.95, 0.55, 0.15]
    img = img * (0.9 + 0.2 * _noise(rng, n, 8)[..., None])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def wood_texture(seed, n=512):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    ring = 0.5 + 0.5 * np.sin(2 * np.pi * (25 * xx + 4 * _noise(rng, n, 64)))
    v = 0.75 + 0.2 * ring + 0.1 * (_noise(rng, n, 4) - 0.5)
    return (np.clip(v[..., None] * np.array([0.62, 0.45, 0.30]), 0, 1) * 255).astype(np.uint8)


def floor_texture(seed, n=512):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    tiles = ((np.floor(xx * 8) + np.floor(yy * 8)) % 2)
    v = 0.45 + 0.08 * tiles + 0.1 * (_noise(rng, n, 16) - 0.5)
    return (np.clip(v[..., None] * np.array([0.9, 0.9, 0.95]), 0, 1) * 255).astype(np.uint8)


def png_bytes(img):
    import cv2
    ok, buf = cv2.imencode(".png", img[:, :, ::-1])
    assert ok
    return buf.tobytes()


# The print on a fabric says nothing about its material: the four prints are dealt to the ids in a fixed cycle.
FABRIC_IDS = "PABCDEFGHIJ"
TEXTURE_KIND = {fid: ("plaid", "floral", "twill", "knit")[i % 4] for i, fid in enumerate(FABRIC_IDS)}


# ----------------------------------------------------------------------------------------------------------------
# clip programs (observation clips; see clip_programs)
# ----------------------------------------------------------------------------------------------------------------
def _jitter_grasp(p, yaw, pose, rng):
    """The robot's grasp is not placed exactly: shift the clamp along the fabric edge by up to 2 cm, in or out by up to
    4 mm and turn it by up to 5 degrees, so the clamp's commanded pose does not reveal the fabric's start pose."""
    c, s_ = np.cos(pose[2]), np.sin(pose[2])
    along, inward = float(rng.uniform(-0.02, 0.02)), float(rng.uniform(-0.004, 0.004))
    d = np.array([-s_ * along + c * inward, c * along + s_ * inward, 0.0])
    return p + d, yaw + float(rng.uniform(-np.radians(5), np.radians(5)))


def clip_programs(rng):
    """List of (kind, fabric_start pose, clamp trajectory): seven clips per fabric.

    0-2  drag_release    the handling the slide tests use, at three speeds inside their range (one in each of 0.35-0.55,
                         0.60-0.80 and 0.85-1.05 m/s, in a drawn order), with a lift, an acceleration distance and a
                         release distance of their own;
    3-4  pull_over_edge  the handling the overhang tests use, to two overhangs inside their range (0.04-0.07 and
                         0.08-0.12 m), with a lift and a pulling speed of their own;
    5    lift_hang, 6 shake   other handling of the same fabric.
    The grasp of every clip is placed with scatter (_jitter_grasp), so none is a scored test program and the
    clamp's motion does not give the fabric's start pose away. Every pose is inside the arm's working area (the same
    ranges as the hidden tests: tests/review/make_set.py, and tests/review/clip_check.py for the arm)."""
    out = []
    bands = [(0.35, 0.55), (0.60, 0.80), (0.85, 1.05)]
    for k in rng.permutation(3):
        pose = (float(rng.uniform(-0.24, -0.14)), float(rng.uniform(-0.20, 0.0)), float(rng.uniform(-0.25, 0.25)))
        p, yaw = _jitter_grasp(*W.edge_grasp(pose, +1), pose, rng)
        v = float(rng.uniform(*bands[k]))
        C, _ = T.fling_traj(p, yaw, pose[2], v, float(rng.uniform(0.05, 0.08)), float(rng.uniform(0.08, 0.15)),
                            lift=float(rng.uniform(0.010, 0.020)))
        out.append(("drag_release", pose, C))
    for lo, hi in ((0.04, 0.07), (0.08, 0.12)):
        pose = (float(rng.uniform(0.10, 0.13)), float(rng.uniform(-0.20, 0.0)), float(rng.uniform(-0.12, 0.12)))
        p, yaw = _jitter_grasp(*W.edge_grasp(pose, +1), pose, rng)
        d = np.array([np.cos(pose[2]), np.sin(pose[2]), 0.0])
        u = float(rng.uniform(lo, hi))
        dist = (W.EDGE_X + u - (pose[0] + np.cos(pose[2]) * W.SIZE_X / 2)) / np.cos(pose[2])
        up = p + np.array([0.0, 0.0, float(rng.uniform(0.010, 0.022))])
        end = up + d * dist
        t1 = 0.6 + dist / float(rng.uniform(0.07, 0.15))
        away = end + d * 0.03 + np.array([0.0, 0.0, 0.08])
        keys = [(0.0, p, yaw, 0, 0), (0.2, p, yaw, 0, 1), (0.6, up, yaw, 0, 1), (t1, end, yaw, 0, 1), (t1 + 0.5, end, yaw, 0, 1),
                (t1 + 0.501, end, yaw, 0, 0), (t1 + 1.0, away, yaw, 0, 0), (t1 + 3.2, away, yaw, 0, 0)]
        out.append(("pull_over_edge", pose, W.program_to_traj(keys)))
    # lift by a corner of the edge nearest the table's front edge, hang, lower partway
    pose = (float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.18, -0.04)), float(rng.uniform(-0.3, 0.3)))
    p, yaw = _jitter_grasp(*W.corner_grasp(pose, +1, int(rng.choice([-1, 1]))), pose, rng)
    top = np.array([p[0], p[1], 0.40])
    low = np.array([p[0] + 0.03, p[1], float(rng.uniform(0.31, 0.35))])
    keys = [(0.0, p, yaw, 0, 0), (0.2, p, yaw, 0, 1), (1.4, top, yaw, 0, 1), (3.2, top, yaw, 0, 1),
            (4.0, low, yaw, 0, 1), (5.6, low, yaw, 0, 1)]
    out.append(("lift_hang", pose, W.program_to_traj(keys)))
    # shake: lift by that short edge, three quick vertical strokes, hold
    pose = (float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.18, -0.04)), float(rng.uniform(-0.3, 0.3)))
    p, yaw = _jitter_grasp(*W.edge_grasp(pose, +1), pose, rng)
    top = np.array([p[0] + 0.03, p[1], 0.38])
    keys = [(0.0, p, yaw, 0, 0), (0.2, p, yaw, 0, 1), (1.3, top, yaw, 0, 1), (2.0, top, yaw, 0, 1)]
    t = 2.0
    for k in range(3):
        keys += [(t + 0.18, top + [0, 0, 0.05], yaw, 0, 1), (t + 0.36, top - [0, 0, 0.02], yaw, 0, 1)]
        t += 0.36
    keys += [(t + 0.2, top, yaw, 0, 1), (t + 2.2, top, yaw, 0, 1)]
    out.append(("shake", pose, W.program_to_traj(keys)))
    return out


# ----------------------------------------------------------------------------------------------------------------
# render scene
# ----------------------------------------------------------------------------------------------------------------
def render_xml(material, mass, cams_xml, arm_assets, arm_body):
    xml = R.scene_xml(material, mass, texture="fabric_tex")
    extra_assets = ('<texture name="fabric_tex" type="2d" file="fabric.png"/>'
                    '<texture name="wood" type="2d" file="wood.png"/><material name="wood" texture="wood" texrepeat="2 2"/>'
                    '<texture name="floor" type="2d" file="floor.png"/><material name="floor" texture="floor" texrepeat="6 6"/>'
                    '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.78 0.82" rgb2="0.35 0.36 0.4" width="256" height="256"/>'
                    + arm_assets)
    tx, ty = W.TABLE_HALF
    cx = W.EDGE_X - tx
    legs = "".join('<geom type="box" size="0.025 0.025 0.37" pos="%g %g -0.41" rgba="0.3 0.25 0.2 1" contype="0" conaffinity="0" group="1"/>'
                   % (cx + sx * (tx - 0.05), sy * (ty - 0.05)) for sx in (-1, 1) for sy in (-1, 1))
    import carm as _carm
    bx, by = float(_carm.ROBOT_BASE[0]), float(_carm.ROBOT_BASE[1])
    pedestal = ('<geom type="box" size="0.11 0.11 0.39" pos="%g %g -0.39" rgba="0.32 0.33 0.36 1" contype="0" conaffinity="0" group="1"/>' % (bx, by))
    world = ('<light pos="0.6 -0.4 1.6" dir="-0.35 0.25 -1" diffuse="0.75 0.73 0.7" specular="0.2 0.2 0.2" castshadow="true"/>'
             '<light pos="-0.8 0.8 1.4" dir="0.5 -0.5 -1" diffuse="0.35 0.37 0.42" castshadow="false"/>'
             '<geom type="plane" size="3 3 0.1" pos="0 0 -0.78" material="floor" contype="0" conaffinity="0" group="1"/>'
             '<geom type="box" size="0.02 2.5 1.2" pos="-1.6 0 0.4" rgba="0.82 0.8 0.76 1" contype="0" conaffinity="0" group="1"/>'
             '<geom name="probe" type="sphere" size="0.0015" pos="0 0 -5" contype="0" conaffinity="0" rgba="0 0 0 0" group="3"/>'
             + legs + pedestal + cams_xml + arm_body)
    xml = xml.replace('<asset>', '<asset>' + extra_assets, 1)
    xml = xml.replace('rgba="0.55 0.45 0.35 1" group="1"/>', 'material="wood" group="1"/>', 1)
    xml = xml.replace('<visual><global offwidth="1280" offheight="960"/>',
                      '<visual><global offwidth="1280" offheight="960"/><headlight ambient="0.35 0.35 0.35" diffuse="0.2 0.2 0.2"/>', 1)
    return xml.replace("</worldbody>", world + "</worldbody>", 1)


# ----------------------------------------------------------------------------------------------------------------
# camera degradation
# ----------------------------------------------------------------------------------------------------------------
def camera_look(rng):
    return dict(wb=rng.uniform(0.88, 1.12, 3), gain=rng.uniform(0.85, 1.15), blur=rng.uniform(0.6, 1.1),
                noise=rng.uniform(2.0, 4.0), vignette=rng.uniform(0.15, 0.3))


def degrade(img_sub, look, rng):
    """img_sub: list of float RGB renders within one exposure (motion blur), each (H, W, 3) in [0, 1]."""
    import cv2
    img = np.mean(img_sub, axis=0)
    img = cv2.GaussianBlur(img, (0, 0), look["blur"])
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    img = img * (1 - look["vignette"] * r2)[..., None] * look["gain"] * look["wb"][None, None]
    img = np.clip(img, 0, 1) ** (1 / 1.05)
    img = img * 255 + rng.normal(0, look["noise"], img.shape) + rng.normal(0, 0.7 * look["noise"], img.shape) * np.sqrt(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def encode(frames, path):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "%dx%d" % (W_IMG, H_IMG), "-r", str(W.FPS), "-i", "-", "-c:v", "libx264",
                          "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "1", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0, path


# ----------------------------------------------------------------------------------------------------------------
# the drawn fabric: one closed shell per frame (csheet), not the model's own facets
# ----------------------------------------------------------------------------------------------------------------
DRAW_NX, DRAW_NY = 4 * (R.NX - 1) + 1, 4 * (R.NY - 1) + 1
DRAW_THICKNESS = 0.0015          # one drawn thickness for every fabric: what is drawn says nothing about the material
TABLE_BOX = ([W.EDGE_X - W.TABLE_HALF[0], 0.0, -0.02], [W.TABLE_HALF[0], W.TABLE_HALF[1], 0.02])


def arm_check(m, d, rows, q, frames):
    """Maintainers' record for one clip: how well the arm follows the clamp (IK error, the smallest margin to a joint
    limit) and how close the arm comes to the table and to the fabric. The clamp's own parts are left out of both
    distances: its jaws close on the fabric where it lies, the lower one in the plane of the table top."""
    import mujoco
    import carm
    sid = m.site("r_eef_site").id
    jid = [m.joint(n).id for n in carm.ARM_JOINTS]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    clamp_body = m.body("r_clamp_vis").id
    arm_bodies = [b for b in range(m.nbody) if m.body(b).name.startswith("r_") and b != clamp_body]
    arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies]
    table = m.geom("table").id
    ft = np.zeros(6)
    out = dict(ik_err_mm=0.0, ik_rot_deg=0.0, joint_margin_rad=9.0, arm_table_mm=1e9, arm_fabric_mm=1e9)
    for k in range(len(rows)):
        carm.set_arm(m, d, q[k])
        mujoco.mj_forward(m, d)
        p_t, R_t = carm.eef_target(rows[k])
        R = d.site_xmat[sid].reshape(3, 3)
        out["ik_err_mm"] = max(out["ik_err_mm"], 1000 * float(np.linalg.norm(d.site_xpos[sid] - p_t)))
        out["ik_rot_deg"] = max(out["ik_rot_deg"], float(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_t) - 1) / 2, -1, 1)))))
        out["joint_margin_rad"] = min(out["joint_margin_rad"], float(np.min(np.minimum(q[k] - lo, hi - q[k]))))
        if k % 2:
            continue
        P = np.asarray(frames[k])
        for g in arm_geoms:
            dt = mujoco.mj_geomDistance(m, d, g, table, 0.5, ft)
            out["arm_table_mm"] = min(out["arm_table_mm"], 1000 * float(dt))
            c, r = d.geom_xpos[g], float(m.geom_rbound[g])
            near = P[np.linalg.norm(P - c, axis=1) < r + 0.05]
            if not len(near):
                out["arm_fabric_mm"] = min(out["arm_fabric_mm"], 1000 * float(np.linalg.norm(P - c, axis=1).min() - r))
                continue
            # distance from the geom to each nearby fabric vertex, through a probe sphere placed at the vertex
            pid = m.geom("probe").id
            for X in near[::2]:
                d.geom_xpos[pid] = X
                dv = mujoco.mj_geomDistance(m, d, g, pid, 0.1, ft)
                out["arm_fabric_mm"] = min(out["arm_fabric_mm"], 1000 * float(dv))
    return out


def do_clip(args):
    import mujoco
    import carm
    fid, ci, kind, pose, C, material, mass, out_dir, frames_max = args
    seed = ord(fid) * 1000 + ci
    rng = np.random.default_rng(seed)
    d = os.path.join(out_dir, "data", "fabrics", fid, "clip%02d" % ci)
    os.makedirs(d, exist_ok=True)
    sim = R.Sim(material, mass)
    sim.place_flat(pose)
    rec = sim.run(C, record_every=W.STEPS_PER_FRAME // 2)       # frame states and the half-frame states between
    assert not sim.unstable, (fid, ci)
    frames, half = rec[0::2], rec[1::2]                       # half[k]: half a frame after frame k (motion blur)
    n = min(len(frames), frames_max or len(frames))
    rows = C[::W.STEPS_PER_FRAME][:n]
    a_assets, a_body = carm.arm_xml_parts()
    cams = carm.cameras_xml(os.path.join(out_dir, "data", "cameras.json"))
    xml = render_xml(material, mass, cams, a_assets, a_body)
    assets = {"fabric.png": png_bytes(fabric_texture(TEXTURE_KIND[fid], 7 + ord(fid))),
              "wood.png": png_bytes(wood_texture(11)), "floor.png": png_bytes(floor_texture(13))}
    m = mujoco.MjModel.from_xml_string(xml, assets)
    dd = mujoco.MjData(m)
    q = carm.solve_ik(m, dd, rows, q0=carm.first_pose(m, dd, rows[0]))
    qh = carm.solve_ik(m, dd, C[W.STEPS_PER_FRAME // 2::W.STEPS_PER_FRAME][:n], q0=q[0])
    if os.environ.get("CLIP_CHECK_DIR"):
        chk = arm_check(m, dd, rows, q, frames[:n])
        os.makedirs(os.environ["CLIP_CHECK_DIR"], exist_ok=True)
        json.dump(dict(fabric=fid, clip=ci, kind=kind, **chk), open(os.path.join(os.environ["CLIP_CHECK_DIR"], "%s_clip%02d.json" % (fid, ci)), "w"))
    import crender
    import cscene
    look = cscene.fabric_looks(FABRIC_IDS, 12)[fid]          # the fabric's look: a function of the ids alone
    clip = crender.open_clip(m, cameras_json()["cameras"], look)
    half_rows = C[W.STEPS_PER_FRAME // 2::W.STEPS_PER_FRAME]
    looks = [camera_look(rng) for _ in carm_cams()]
    out = {c: [] for c in carm_cams()}
    digest = hashlib.sha256()

    def draw(P, qa, row):
        sh = CS.shell(P, (R.NX, R.NY), R.RADIUS, DRAW_THICKNESS, TABLE_BOX, row, W.JAW_W / 2)
        carm.set_arm(m, dd, qa)
        mujoco.mj_forward(m, dd)                     # the renderer reads the arm's body poses
        imgs = {}
        for c in carm_cams():
            im = clip.draw(c, sh, dd, row, thickness=DRAW_THICKNESS)
            digest.update(np.ascontiguousarray(im).tobytes())
            imgs[c] = im.astype(np.float32) / 255.0
        return imgs

    for k in range(n):
        A = draw(frames[k], q[k], rows[k])
        B = (draw(half[k], qh[k], half_rows[k]) if k < len(half) and k < len(qh) and k < len(half_rows) else A)
        for i, c in enumerate(carm_cams()):
            out[c].append(degrade([A[c], B[c]], looks[i], rng))
    if os.environ.get("CLIP_CHECK_DIR"):          # the digest of every picture before the camera effects (determinism record)
        open(os.path.join(os.environ["CLIP_CHECK_DIR"], "%s_clip%02d.digest" % (fid, ci)), "w").write(digest.hexdigest() + "\n")
    for c in carm_cams():
        encode(out[c], os.path.join(d, c + ".mp4"))
    np.save(os.path.join(d, "clamp_500hz.npy"), C.astype(np.float64))
    np.save(os.path.join(d, "arm_q.npy"), q.astype(np.float64))
    info = dict(kind=kind, fps=W.FPS, n_frames=int(n), clamp_dt=W.DT, clamp_rows=int(len(C)),
                frame_k_row="k * %d (frame k shows the fabric when the clamp is at row 20 k)" % W.STEPS_PER_FRAME,
                fabric_start="flat and at rest on the table; its pose is not given")
    json.dump(info, open(os.path.join(d, "clip.json"), "w"), indent=1)
    return fid, ci, n


def carm_cams():
    return [c[0] for c in CAMERAS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=",".join(FABRIC_IDS))
    ap.add_argument("--frames-max", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    json.dump(cameras_json(), open(os.path.join(a.out, "data", "cameras.json"), "w"), indent=1)
    fab = json.load(open(os.path.join(HERE, "fabrics.json")))["fabrics"]
    jobs = []
    for fid in a.only.split(","):
        progs = clip_programs(np.random.default_rng(1000 + ord(fid)))
        fd = os.path.join(a.out, "data", "fabrics", fid)
        os.makedirs(fd, exist_ok=True)
        json.dump(dict(id=fid, size_m=[W.SIZE_X, W.SIZE_Y], grid=[W.NX, W.NY], spacing_m=W.SPACING,
                       mass_kg=round(fab[fid]["mass_kg"], 6), n_clips=len(progs),
                       texture_face="+z (up when flat)"), open(os.path.join(fd, "fabric.json"), "w"), indent=1)
        for ci, (kind, pose, C) in enumerate(progs):
            jobs.append((fid, ci, kind, pose, C, fab[fid]["material"], fab[fid]["mass_kg"], a.out, a.frames_max))
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as ex:      # one clip per process (crender)
        for fid, ci, n in ex.map(do_clip, jobs):
            print("[gen] %s clip%02d %d frames" % (fid, ci, n), flush=True)


if __name__ == "__main__":
    main()
