"""Render the MuJoCo views of a site: the top-down view with the roof hidden, the two dollhouse views, and the
eye-level stills (0.33 m) at the cameras the generator placed (pad, chase hatch, a high sill, a floor pipe, the
bund, a gauge from 1.5 m, the valve gallery, the drum rows, the pump sets).

  MUJOCO_GL=osmesa python -m site_gen.render_site <site_dir> [--out <dir>] [--views topdown,eye_chase]

The classic renderer draws at most 8 lights (headlight included), so each view activates the lights that matter
for it (light_active); use lights_for_view() the same way for robot cameras.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


from .loader.view import lights_for_view, scene_option  # noqa: E402,F401  (the loader's; re-exported)


def render(site_dir, out_dir=None, views=None, size=(2400, 1700), eye_size=(1280, 905)):
    site = json.load(open(os.path.join(site_dir, "site.json")))
    out_dir = out_dir or os.path.join(site_dir, "renders")
    os.makedirs(out_dir, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(os.path.join(site_dir, "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    from .ip_models import apply_truth_states
    apply_truth_states(model, data, site["ips"] + site["decoys"] + site.get("plain_instruments", []))
    W, H = size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    r = mujoco.Renderer(model, H, W)
    opt = scene_option()
    out = []
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    for cam in site["cameras"]:
        nm = cam["name"]
        key = nm.split("/")[-1]
        if views and key not in views:
            continue
        if nm not in names:
            continue
        t = time.time()
        kind = cam.get("kind", "eye")
        if kind in ("topdown", "dollhouse"):
            for i in range(model.nlight):
                model.light_active[i] = 1 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or "").startswith(
                    "light/fill") else 0
            model.vis.headlight.ambient[:] = [0.42, 0.42, 0.42]
            model.vis.headlight.diffuse[:] = [0.30, 0.30, 0.30]
            opt.geomgroup[2] = 0
        else:
            cid = names.index(nm)
            lights_for_view(model, model.cam_pos[cid], n=7)
            model.vis.headlight.ambient[:] = [0.30, 0.30, 0.30]
            model.vis.headlight.diffuse[:] = [0.22, 0.22, 0.22]
            opt.geomgroup[2] = 1
        r.update_scene(data, camera=nm, scene_option=opt)
        img = r.render()
        im = Image.fromarray(img)
        if kind not in ("topdown", "dollhouse"):
            im = im.resize(eye_size, Image.LANCZOS)
        p = os.path.join(out_dir, "%s.png" % key)
        im.save(p)
        out.append({"view": key, "path": p, "what": cam.get("what", kind), "s": round(time.time() - t, 2)})
    r.close()
    if site.get("layout") == "ladder_v1" and (not views or any(v.startswith(("slot_", "inspect_")) for v in views)):
        out += render_slots(site, model, data, out_dir, views)
    with open(os.path.join(out_dir, "renders.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def render_slots(site, model, data, out_dir, views=None, size=(1280, 960)):
    """Ladder sites: for every item, a context view (60 deg) and the inspection camera's view (18 deg horizontal) from
    its observation pose, aimed at the instrument (free camera; the robot is not in the scene)."""
    W, H = size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    rr = mujoco.Renderer(model, H, W)
    opt = scene_option()
    cams = ((site.get("ladder") or {}).get("config") or {}).get("cameras") or {
        "inspect": {"width": 1280, "height": 960, "hfov_deg": 18.0}, "thermal": {"width": 160, "height": 120,
                                                                              "hfov_deg": 50.0}}

    def vfov(name):
        c = cams[name]
        return math.degrees(2 * math.atan(math.tan(math.radians(float(c["hfov_deg"]) / 2)) * float(c["height"]) /
                                          float(c["width"])))
    vfov_inspect = vfov("inspect")
    fovy0 = float(model.vis.global_.fovy)
    out = []
    for ip in site["ips"]:
        slot = ip.get("slot")
        op = ip.get("observation_pose")
        if slot is None or op is None:
            continue
        eye = np.array([op["pos"][0], op["pos"][1], op.get("head_height", 0.33)])
        if slot in ("M", "X"):
            tgt = np.array([ip["pose"]["pos"][0], ip["pose"]["pos"][1], 0.0])
        else:
            n = np.array([math.cos(ip["pose"]["yaw"]), math.sin(ip["pose"]["yaw"]), 0.0])
            tgt = np.asarray(ip["pose"]["pos"], float) + (0.42 if ip["type"] == "valve_handwheel" else 0.0) * \
                np.array([0, 0, 1.0]) + 0.02 * n
        # the inspection view uses the camera the acceptance file names for the type (thermal: the 50 deg imager)
        for kind, fovy in (("slot", 60.0), ("inspect", vfov("thermal") if ip["type"] == "thermal" else vfov_inspect)):
            key = "%s_%s" % (kind, slot)
            if views and key not in views:
                continue
            if kind == "inspect" and slot in ("M", "X"):
                continue
            t = time.time()
            d = tgt - eye
            dist = float(np.linalg.norm(d))
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = tgt
            cam.distance = dist
            cam.azimuth = math.degrees(math.atan2(d[1], d[0]))
            cam.elevation = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
            model.vis.global_.fovy = fovy
            lights_for_view(model, eye, n=7)
            model.vis.headlight.ambient[:] = [0.30, 0.30, 0.30]
            model.vis.headlight.diffuse[:] = [0.22, 0.22, 0.22]
            rr.update_scene(data, camera=cam, scene_option=opt)
            img = rr.render()
            p = os.path.join(out_dir, "%s.png" % key)
            Image.fromarray(img).save(p)
            out.append({"view": key, "path": p, "what": "%s %s %s from its observation pose%s" % (
                kind, slot, ip["tag"], " (thermal camera field of view)" if kind == "inspect" and
                ip["type"] == "thermal" else ""), "s": round(time.time() - t, 2)})
    model.vis.global_.fovy = fovy0
    rr.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--views", default=None, help="comma list of view keys (topdown, dollhouse1, eye_chase, ...)")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=1700)
    a = ap.parse_args(argv)
    res = render(a.site_dir, a.out, a.views.split(",") if a.views else None, (a.width, a.height))
    for x in res:
        print(json.dumps(x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
