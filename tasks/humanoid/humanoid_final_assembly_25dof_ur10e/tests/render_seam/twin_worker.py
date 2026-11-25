"""The render WORKER behind the cell's given-layer seam (arm/assembly_humanoid/render_iface.py given_layer_session):
one process that holds one Filament view of one episode and renders the cell's camera frames on request.

It renders the GIVEN render twin (render/twin/ of the root-owned copy of the givens) through the GIVEN render layer
(lib/arm/assembly_humanoid/agent_render.py, lib/render/), the very code and files a program renders its own twin with;
the only hidden input is the map from the cell's bodies to the twin's bodies (gt/render/twin_map.json): every unit
body's constant transform into its unit's delivered frame, every fixture body's nominal place. At "open" it builds the
episode's kinematic skeleton (the cell's own data path at the episode's layout, no collision pieces) to turn the cell's
joint state into the twin's body poses.

Its file descriptor 1 is not the server's: the server starts it with stdout and stderr to a log file (Filament writes
its native log to descriptor 1); requests come on stdin, one JSON line each; every reply goes back on a separate pipe
(the descriptor number in argv[1]); frames are .npz files in the worker's private folder (argv[2]).
  {"op": "open", "layout": {...}}                                 -> {"ok", "t_open_s", "rss_kb", "twin": {...}}
  {"op": "frame", "camera": {name, width, height, fx, fy, cx, cy, world_T_cam}, "state": {joint: [qpos...]},
   "depth": bool, "index": k}                                     -> {"ok", "npz", "t_frame_s", "rss_kb"}
  {"op": "close"}                                                 -> {"ok", "vm_hwm_kb"}
Environment: ARB_TWIN_STAGE (the stage folder: arm/ and cell_roots.py), ARB_TWIN_GT (gt/), ARB_GIVEN_DIR (the
root-owned givens), ARB_TWIN_LIB (the given library), ARB_TWIN_SAVE_STATE=1 (also write each frame's twin state).
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True


def status_kb(key):
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith(key + ":"):
                return int(ln.split()[1])
    return None


def quat_of(R):
    t = float(np.trace(R))
    if t > -0.99:
        w = math.sqrt(max(0.0, 1.0 + t)) / 2.0
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        k = int(np.argmax(np.diag(R)))
        i, j = (k + 1) % 3, (k + 2) % 3
        r = math.sqrt(max(0.0, 1.0 + R[k, k] - R[i, i] - R[j, j]))
        v = [0.0, 0.0, 0.0]
        v[k] = r / 2.0
        v[i] = (R[i, k] + R[k, i]) / (2 * r)
        v[j] = (R[j, k] + R[k, j]) / (2 * r)
        q = [(R[j, i] - R[i, j]) / (2 * r)] + v
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def body_T(m, d, bid):
    T = np.eye(4)
    T[:3, :3] = d.xmat[bid].reshape(3, 3)
    T[:3, 3] = d.xpos[bid]
    return T


class Skeleton(object):
    """the episode's cell model at its layout without collision pieces: the cell's joint state -> body poses"""

    def __init__(self, layout, given_dir):
        import mujoco
        from arm.assembly_humanoid import episode as EP
        from arm.assembly_humanoid import fixtures_h as FH
        from arm.assembly_humanoid import kit_adapter as KA
        from arm.assembly_humanoid import scene as SC
        self.mj = mujoco
        inputs = EP.load_inputs()
        cfg, des, kit, fr = inputs["cfg"], inputs["des"], inputs["kit"], inputs["frozen"]
        lay = dict(layout)
        if lay.get("top_camera_T") is not None:
            lay["top_camera_T"] = (np.asarray(lay["top_camera_T"][0], float), np.asarray(lay["top_camera_T"][1], float))
        pl = FH.module_rest_placements(des, kit, fr, lay)
        pl["base"] = FH.base_placement(des, lay)
        fx = FH.fixture_records(des, lay, visual=False)
        data, arrays = KA.cell_data(cfg=cfg, frozen=fr, placements=pl, fixtures=fx, collision=False,
                                    rest_bodies=FH.REST_BODIES)
        cell = SC.build_cell(data, cfg, arrays=arrays, robot=True, robot_models_dir=os.path.join(given_dir, "robot"),
                             monitor=False, settle=False)
        self.m = cell.m
        self.d = mujoco.MjData(self.m)
        del cell

    def set(self, state):
        m, d = self.m, self.d
        for jn, q in state.items():
            j = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_JOINT, jn)
            if j < 0:
                continue
            a = int(m.jnt_qposadr[j])
            n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
            d.qpos[a:a + n] = np.asarray(q, float).ravel()[:n]
        self.mj.mj_kinematics(m, d)


def twin_state(sk, tmap):
    """the twin's free joints {"<body>/pose": [x y z qw qx qy qz]} from the skeleton's body poses"""
    m, d, mj = sk.m, sk.d, sk.mj
    out = {}
    for name, rec in tmap["bodies"].items():
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, rec["cell_body"])
        if bid < 0:
            raise KeyError("the cell has no body %r" % rec["cell_body"])
        T = body_T(m, d, bid) @ np.asarray(rec["T_cell_body_from_twin"], float)
        out[name + "/pose"] = [float(v) for v in T[:3, 3]] + [float(v) for v in quat_of(T[:3, :3])]
    return out


def given_layer(lib):
    """the GIVEN render layer (lib/arm/assembly_humanoid/agent_render.py), loaded from its file: the cell's own package
    of the same name (arm/assembly_humanoid in this stage folder) has no render layer"""
    import importlib.util
    p = os.path.join(lib, "arm", "assembly_humanoid", "agent_render.py")
    spec = importlib.util.spec_from_file_location("given_agent_render", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    fd = int(sys.argv[1])
    folder = sys.argv[2]
    out = os.fdopen(fd, "w", buffering=1)
    stage = os.environ["ARB_TWIN_STAGE"]
    gt = os.environ["ARB_TWIN_GT"]
    given = os.environ["ARB_GIVEN_DIR"]
    lib = os.environ.get("ARB_TWIN_LIB") or os.path.join(given, "lib")
    for p in (lib, stage):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cell_roots
    cell_roots.bind(gt, given)
    with open(cell_roots.check()["twin_map"]) as f:
        tmap = json.load(f)
    AR = given_layer(lib)
    save_state = os.environ.get("ARB_TWIN_SAVE_STATE") == "1"
    v = sk = None
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["op"] == "open":
                t = time.perf_counter()
                sk = Skeleton(req["layout"], given)
                twin = AR.load_twin()
                v = AR.open_view(twin)
                out.write(json.dumps({"ok": True, "t_open_s": round(time.perf_counter() - t, 3), "rss_kb": status_kb("VmRSS"),
                                      "twin": {"bodies": len(twin["bodies"]), "meshes": len(twin["meshes"])}}) + "\n")
            elif req["op"] == "frame":
                sk.set(req["state"])
                st = twin_state(sk, tmap)
                cam = dict(req["camera"])
                t = time.perf_counter()
                if req.get("depth"):
                    rgb, dep = v.render(st, cam, depth=True)
                else:
                    rgb, dep = v.render(st, cam), np.zeros((0,), np.float32)
                tf = time.perf_counter() - t
                p = os.path.join(folder, "frame_%03d_%s.npz" % (int(req.get("index", 0)), cam["name"]))
                np.savez_compressed(p, rgb=rgb, depth=dep)
                if save_state:
                    with open(p[:-4] + "_state.json", "w") as f:
                        json.dump({"camera": cam, "state": st, "depth": bool(req.get("depth"))}, f)
                out.write(json.dumps({"ok": True, "npz": p, "t_frame_s": round(tf, 3), "rss_kb": status_kb("VmRSS")}) + "\n")
            elif req["op"] == "close":
                if v is not None:
                    v.close()
                out.write(json.dumps({"ok": True, "vm_hwm_kb": status_kb("VmHWM")}) + "\n")
                break
        except Exception as e:  # noqa: BLE001 - every failure is a reply, never a hang
            out.write(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300]),
                                  "trace": traceback.format_exc()[-2000:]}) + "\n")
    out.close()


if __name__ == "__main__":
    main()
