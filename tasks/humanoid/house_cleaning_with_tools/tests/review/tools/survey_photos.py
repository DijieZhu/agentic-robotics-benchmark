"""Survey photos (example camera frames of development scenes, never the hidden ones) and the observability evidence
(crumbs and grease seen in those frames). Run inside the task image: survey_photos.py STAGE OUT SEED [SEED ...]"""
import base64, io, json, math, os, sys
import numpy as np
stage, out = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
sys.path.insert(0, stage)
sys.dont_write_bytecode = True
import mujoco
from PIL import Image
from house_cell.server import CellServer
from house_cell import room as RM
from house_cell import cleaning as CL
cfg = json.load(open(os.path.join(stage, "grader_config.json")))

# head views: (label, base x, y, yaw, waist pitch) - None = the start pose
HEAD_VIEWS = [("head_start", None), ("head_dining_south", (2.85, 1.70, math.pi / 2, 0.35)),
              ("head_floor_south", (2.85, 1.20, math.pi / 2, 0.10)),
              ("head_coffee_west", (3.95, 4.15, 0.0, 0.35)),
              ("head_bins", (1.00, 0.85, math.pi, 0.20)), ("head_sink", (0.95, 2.65, math.pi, 0.35))]
CEILING = ("ceiling_kitchen", "ceiling_dining", "ceiling_living")


def teleport(cell, x, y, yaw, pitch):
    m, d, r = cell.m, cell.d, cell.robot
    d.qpos[r.base_q[:3]] = (x, y, yaw)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    cell.ctrl.reset_targets(d)
    if cell._q_kin is not None:
        cell._q_kin[:] = d.qpos
    if cell.walker is not None:
        cell.walker.reset(cell.sim_time, d.qpos[cell.walker.base_adr])
    cell.command({"waist": [0.0, 0.0, pitch]})
    for _ in range(150):
        cell._tick()


def save_frame(rep, folder, label):
    os.makedirs(folder, exist_ok=True)
    rgb = base64.b64decode(rep["rgb_png_b64"]); dep = base64.b64decode(rep["sensor_depth_u16_mm_b64"])
    im = Image.open(io.BytesIO(rgb)); im.save(os.path.join(folder, label + "_rgb.png"), optimize=True)
    dm = Image.open(io.BytesIO(dep)); dm.save(os.path.join(folder, label + "_depth_mm.png"), optimize=True)
    meta = {k: v for k, v in rep.items() if not k.endswith("_b64")}
    meta["files"] = {"rgb": label + "_rgb.png", "depth_mm": label + "_depth_mm.png"}
    json.dump(meta, open(os.path.join(folder, label + ".json"), "w"), indent=1)


def project(cell, mjcam, P):
    """pixel (u, v) and depth of world points P (N x 3) in camera mjcam of the cell's render state."""
    d = cell.render_state()
    cid = mujoco.mj_name2id(cell.m, mujoco.mjtObj.mjOBJ_CAMERA, mjcam)
    pos, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
    # MuJoCo camera: looks along -z, y up; image: x right, y down
    c = (P - pos) @ R
    z = -c[:, 2]
    w, h = cell.cams.sizes[mjcam]
    f = (h / 2) / math.tan(math.radians(float(cell.m.cam_fovy[cid])) / 2)
    u = w / 2 + f * c[:, 0] / z
    v = h / 2 - f * c[:, 1] / z
    return u, v, z, f


def visibility(cell, mjcam):
    """Per crumb: its projected diameter (px) and the pixels whose colour changes by more than 6 sigma of the pixel
    noise when the crumbs are taken out of the scene (clean renders; noise sigma 0.008 = 2 grey levels)."""
    d = cell.render_state()
    cr = cell.process.crumbs
    with_c = cell.cams.rgb_clean(d, mjcam).astype(np.int16)
    keep = d.mocap_pos[cr.mocap].copy()
    cell.d.mocap_pos[cr.mocap] = keep - np.array([0.0, 0.0, 5.0])
    d = cell.render_state()
    without = cell.cams.rgb_clean(d, mjcam).astype(np.int16)
    cell.d.mocap_pos[cr.mocap] = keep
    cell.render_state()
    diff = np.abs(with_c - without).max(axis=2)
    u, v, z, f = project(cell, mjcam, cr.P)
    hgt, wid = diff.shape
    out = []
    for i in range(cr.n):
        if not (z[i] > 0.05 and 0 <= u[i] < wid and 0 <= v[i] < hgt):
            out.append(None); continue
        dpx = 2 * cr.r[i] * f / z[i]
        rad = int(math.ceil(dpx / 2 + 2))
        u0, v0 = int(round(u[i])), int(round(v[i]))
        win = diff[max(0, v0 - rad):v0 + rad + 1, max(0, u0 - rad):u0 + rad + 1]
        out.append({"i": i, "on": str(cr.on[i]), "diam_px": round(float(dpx), 2), "dist_m": round(float(z[i]), 3),
                    "px_changed": int((win > 12).sum()), "max_change": int(win.max()) if win.size else 0})
    return out


def grease_visibility(cell, mjcam):
    d = cell.render_state()
    g = cell.process.grease
    with_g = cell.cams.rgb_clean(d, mjcam).astype(np.int16)
    keep = g.g.copy()
    g.g[:] = 0.0
    g.sync(force=True)
    d = cell.render_state()
    without = cell.cams.rgb_clean(d, mjcam).astype(np.int16)
    g.g[:] = keep
    g.sync(force=True)
    diff = np.abs(with_g - without).max(axis=2)
    return {"px_changed_gt12": int((diff > 12).sum()), "px_changed_gt30": int((diff > 30).sum()),
            "grease_cells_dirty": int((keep > 0.03).sum()), "cell_m": float(g.cell if hasattr(g, "cell") else 0.02)}


for seed in seeds:
    srv = CellServer(seed, privileged=False, nominal=False, config=cfg, log=print)
    cell = srv.cell
    folder = os.path.join(out, "scene_%d" % seed)
    ev = {"seed": seed, "frames": {}}
    for cam in CEILING:
        rep = srv.handle("get_camera", {"name": cam})
        save_frame(rep, folder, cam)
        ev["frames"][cam] = {"crumbs": visibility(cell, cam), "grease": grease_visibility(cell, cam)}
    for label, pose in HEAD_VIEWS:
        if pose is not None:
            teleport(cell, *pose)
        rep = srv.handle("get_camera", {"name": "head"})
        save_frame(rep, folder, label)
        mj = "a/head_cam"
        ev["frames"][label] = {"crumbs": visibility(cell, mj), "grease": grease_visibility(cell, mj)}
    json.dump(ev, open(os.path.join(out, "evidence_%d.json" % seed), "w"), indent=1)
    srv.close()
    print("scene", seed, "done", flush=True)
