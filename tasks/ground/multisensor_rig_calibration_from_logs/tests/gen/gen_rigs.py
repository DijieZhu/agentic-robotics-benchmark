#!/usr/bin/env python3
"""Generate rigs of a set, once, offline.

    <python 3.12 with numpy 2.5.4, scipy, opencv, pyyaml, embreex, usd-core> -I -B gen_rigs.py --set practice --index 0 --name rig_a
        --hall <hall.usdc> --props <props dir> --husky <husky stl dir> --cache <dir> --out <root> [--blender blender] [--only-seq 0] [--keep-src]

Writes <root>/<set>/<name>/ (the agent-visible rig), <root>/<set>_truth/<name>/ (ground_truth.yaml, faults.json,
heldout.npz, site.json) and <root>/<set>_truth/<name>/sha256.json (every file of the rig folder)."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rigsim, rigsite

ap = argparse.ArgumentParser()
for k in ("set", "index", "name", "hall", "props", "husky", "people", "cache", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--odom", type=int, default=1); ap.add_argument("--knock", type=int, default=0)
ap.add_argument("--blender", default="blender"); ap.add_argument("--only-seq", type=int, default=-1); ap.add_argument("--keep-src", action="store_true")
ap.add_argument("--no-camera", action="store_true"); ap.add_argument("--reuse-camera", action="store_true", help="keep the rig folder's camera frames, write everything else again"); ap.add_argument("--lamp-w", type=float, default=330.0)
a = ap.parse_args()
t0 = time.time()
site = rigsite.Site(a.hall, a.props, a.cache)
rigsite.PEOPLE_DIR = os.path.abspath(a.people)
rig = rigsim.make_rig(rigsim.rig_seed(a.set, a.index), site, has_odom=bool(a.odom), knock=bool(a.knock))
site.build(rig.props)
sr, sc, sn = rigsite.self_returns(a.husky, rig.T_body_lidar, rigsim.lidar_ray_dirs())
rig.self_range, rig.self_cos, rig.self_nir = sr, sc, sn
rd = os.path.join(a.out, a.set, a.name); td = os.path.join(a.out, a.set + "_truth", a.name)
if not a.reuse_camera:
    shutil.rmtree(rd, ignore_errors=True)
shutil.rmtree(td, ignore_errors=True); os.makedirs(td)
if a.only_seq >= 0:
    os.makedirs(rd, exist_ok=True)
    c = rig.given[a.only_seq]
    r, p = rigsim.write_sequence(rig, site, c, os.path.join(rd, "seq_%02d" % a.only_seq)); records, plans = r, {c: p}
else:
    records, plans = rigsim.write_rig(rig, site, a.name, rd)
print("[gen] %s: lidar, imu, odometry written, %.0f s" % (a.name, time.time() - t0))
# movers of every drive as one list for the renderer; a shot names the ones of its drive
moff, mlist = {}, []
for c in sorted(plans):
    moff[c] = len(mlist)
    for m in rig.movers[c]:
        if m["kind"] == "vehicle":
            mlist.append({"kind": "vehicle", "usd": os.path.abspath(os.path.join(a.props, m["name"] + ".usdc"))})
        else:
            mlist.append({"kind": "person", "npz": os.path.join(os.path.abspath(a.people), m["name"] + ".npz"),
                          "tex": os.path.join(os.path.abspath(a.people), "palette.png")})
scene = {"hall": {"usd": os.path.abspath(a.hall), "t": [float(x) for x in rigsite.HALL_T]},
         "props": [{"usd": os.path.abspath(os.path.join(a.props, n + ".usdc")), "pos": [x, y, 0.0], "yaw": yaw} for n, x, y, yaw in rig.props],
         "movers": mlist, "lamps": rigsite.hall_lamps(site), "lamp_w": a.lamp_w, "shadow_res": 0.15, "world": 0.03}
json.dump(scene, open(os.path.join(td, "site.json"), "w"), indent=1)
if not a.no_camera and not a.reuse_camera:
    src = rigsim.camera_source(rig); sd = os.path.join(a.out, "_src", a.set, a.name); shutil.rmtree(sd, ignore_errors=True); os.makedirs(sd)
    shots = []
    for c, p in plans.items():
        os.makedirs(os.path.join(sd, "seq_%02d" % (c - 1)))
        for s_ in p["shots"]:
            shots.append(dict(s_, name="seq_%02d/%s" % (c - 1, s_["name"]), movers=[[mv[0] + moff[c]] + mv[1:] for mv in s_["movers"]]))
    json.dump({"w": src["w"], "h": src["h"], "fx": src["fx"], "fy": src["fy"], "cx": src["cx"], "cy": src["cy"], "shots": shots}, open(os.path.join(sd, "shots.json"), "w"))
    t1 = time.time()
    r = subprocess.run([a.blender, "-b", "--python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bl_render.py"), "--",
                        "--scene", os.path.join(td, "site.json"), "--shots", os.path.join(sd, "shots.json"), "--out", sd],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    open(os.path.join(sd, "blender.log"), "w").write(r.stdout)
    assert r.returncode == 0 and "[bl_render] %d shots" % len(shots) in r.stdout, r.stdout[-2000:]
    print("[gen] %s: %d frames rendered (%d x %d), %.0f s" % (a.name, len(shots), src["w"], src["h"], time.time() - t1))
    for c, p in plans.items():
        rigsim.finish_camera(rig, site, c, os.path.join(rd, "seq_%02d" % (c - 1)), p, os.path.join(sd, "seq_%02d" % (c - 1)))
    if not a.keep_src:
        shutil.rmtree(sd)
rigsim._dump_yaml(rigsim.truth_dict(rig), os.path.join(td, "ground_truth.yaml"))
json.dump({"faults": records}, open(os.path.join(td, "faults.json"), "w"), indent=1)
rigsim.save_heldout(rig, site, os.path.join(td, "heldout.npz"))
H = {}
for root, _, files in os.walk(rd):
    for f in sorted(files):
        p = os.path.join(root, f); H[os.path.relpath(p, rd)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(dict(sorted(H.items())), open(os.path.join(td, "sha256.json"), "w"), indent=0)
print("[gen] %s done: %d files, %.0f s" % (a.name, len(H), time.time() - t0))
