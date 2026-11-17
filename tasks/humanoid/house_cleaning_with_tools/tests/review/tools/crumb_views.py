"""Observability of the crumbs and the grease from viewpoints the robot can reach: the robot placed (not walked) at
free stances on a ring round the dining set, squatting and bent over, its head camera rendered with and without the
crumbs (clean colour, no sensor noise); a crumb counts as seen from a view when >= MIN_PX of its pixels change by more
than 12 grey levels (6 sigma of the cameras' pixel noise). Usage: crumb_views.py STAGE OUT SEED [SEED ...]"""
import json, math, os, sys
import numpy as np
stage, out = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
sys.path.insert(0, stage)
sys.dont_write_bytecode = True
import mujoco
from house_cell.server import CellServer
from house_cell import room as RM
cfg = json.load(open(os.path.join(stage, "grader_config.json")))
MIN_PX = 3
STAND, PITCH = float(os.environ.get("CV_STAND", 0.55)), float(os.environ.get("CV_PITCH", 0.55))
D = RM.DINING
(dx, dy), (lx, ly) = D["center"], D["size"]


def views():
    if os.environ.get("CV_VIEWS"):
        return [tuple(v) for v in json.loads(os.environ["CV_VIEWS"])]
    out = []
    for x in np.arange(dx - 0.75, dx + 0.751, 0.25):
        out.append((float(x), dy - ly / 2 - 0.42, math.pi / 2))          # south side, facing north
        out.append((float(x), dy + ly / 2 + 0.42, -math.pi / 2))         # north side, facing south
        out.append((float(x), dy - ly / 2 - 0.95, math.pi / 2))          # further out
        out.append((float(x), dy + ly / 2 + 0.95, -math.pi / 2))
    for y in np.arange(dy - 0.6, dy + 0.61, 0.3):
        out.append((dx - lx / 2 - 0.85, float(y), 0.0))                   # west, beyond the chair
        out.append((dx + lx / 2 + 0.85, float(y), math.pi))               # east
    return out


def robot_collides(cell):
    m, d, r = cell.m, cell.d, cell.robot
    rb = set(int(b) for b in r.body_ids)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in rb) != (b2 in rb) and c.dist < -0.002:
            return True
    return False


def place(cell, x, y, yaw):
    m, d, r = cell.m, cell.d, cell.robot
    d.qpos[r.base_q[:3]] = (x, y, yaw)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    cell.ctrl.reset_targets(d)
    if cell._q_kin is not None:
        cell._q_kin[:] = d.qpos
    if cell.walker is not None:
        cell.walker.reset(cell.sim_time, d.qpos[cell.walker.base_adr])
    cell.command({"stand_height": STAND, "waist": [0.0, 0.0, PITCH]})
    for _ in range(250):
        cell._tick(record=False)
    mujoco.mj_forward(m, d)
    return not robot_collides(cell)


def seen(cell, mj="a/head_cam"):
    cr = cell.process.crumbs
    d = cell.render_state()
    a = cell.cams.rgb_clean(d, mj).astype(np.int16)
    keep = cell.d.mocap_pos[cr.mocap].copy()
    cell.d.mocap_pos[cr.mocap] = keep - np.array([0.0, 0.0, 5.0])
    d = cell.render_state()
    b = cell.cams.rgb_clean(d, mj).astype(np.int16)
    cell.d.mocap_pos[cr.mocap] = keep
    diff = np.abs(a - b).max(axis=2)
    d = cell.render_state()
    cid = mujoco.mj_name2id(cell.m, mujoco.mjtObj.mjOBJ_CAMERA, mj)
    pos, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
    c = (cr.P - pos) @ R
    z = -c[:, 2]
    w, h = cell.cams.sizes[mj]
    f = (h / 2) / math.tan(math.radians(float(cell.m.cam_fovy[cid])) / 2)
    u, v = w / 2 + f * c[:, 0] / np.maximum(z, 1e-6), h / 2 - f * c[:, 1] / np.maximum(z, 1e-6)
    res = {}
    for i in range(cr.n):
        if z[i] < 0.05 or not (0 <= u[i] < w and 0 <= v[i] < h):
            continue
        rad = int(math.ceil(cr.r[i] * f / z[i] + 2))
        u0, v0 = int(round(u[i])), int(round(v[i]))
        win = diff[max(0, v0 - rad):v0 + rad + 1, max(0, u0 - rad):u0 + rad + 1]
        res[i] = (int((win > 12).sum()), round(float(2 * cr.r[i] * f / z[i]), 2), round(float(z[i]), 3))
    return res, float(pos[2])


summary = {}
for seed in seeds:
    srv = CellServer(seed, privileged=False, nominal=False, config=cfg, log=print)
    cell = srv.cell
    srv.handle("get_camera", {"name": "head"})            # opens the renderer
    cr = cell.process.crumbs
    best = {i: (0, 0.0, 0.0) for i in range(cr.n)}
    used = []
    for (x, y, yaw) in views():
        if not place(cell, x, y, yaw):
            continue
        res, camz = seen(cell)
        used.append((round(x, 2), round(y, 2), round(yaw, 2), round(camz, 3)))
        for i, t in res.items():
            if t[0] > best[i][0]:
                best[i] = t
    floor = [i for i in range(cr.n) if cr.on[i] == "floor"]
    table = [i for i in range(cr.n) if cr.on[i] != "floor"]
    rec = {"seed": seed, "stand_height": STAND, "waist_pitch": PITCH, "views_used": used,
           "crumbs": {int(i): {"on": str(cr.on[i]), "xy": [round(float(cr.P[i, 0]), 3), round(float(cr.P[i, 1]), 3)],
                               "r_m": round(float(cr.r[i]), 4), "best_px_changed": best[i][0],
                               "diam_px_there": best[i][1], "dist_m_there": best[i][2]} for i in range(cr.n)},
           "floor_seen": sum(1 for i in floor if best[i][0] >= MIN_PX), "floor_total": len(floor),
           "table_seen": sum(1 for i in table if best[i][0] >= MIN_PX), "table_total": len(table)}
    json.dump(rec, open(os.path.join(out, "crumb_views_%d.json" % seed), "w"), indent=1)
    summary[seed] = {k: rec[k] for k in ("floor_seen", "floor_total", "table_seen", "table_total")}
    summary[seed]["views"] = len(used)
    print(seed, summary[seed], flush=True)
    srv.close()
json.dump(summary, open(os.path.join(out, "crumb_views_summary.json"), "w"), indent=1)
