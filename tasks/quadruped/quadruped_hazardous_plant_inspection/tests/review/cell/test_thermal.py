"""The thermal camera renders each temperature point with the emissivity of its surface type.

    MUJOCO_GL=osmesa python -I -B tests/review/cell/test_thermal.py SITE [SITE ...] [--json OUT]

Per site package (the training hall has one temperature point of each of the plant's three surface types):
  1. table: every unit of every temperature point renders with the emissivity that mission()["catalog"] publishes for
     the surface type the round sheet names for its item (decoys: the surface of their own truth); a decoy without a
     unit list renders its hot face at its own truth.temp_C (seen from in front of its face);
  2. render: the robot is placed (kinematically, nothing is stepped) at the point's observation pose, facing the hot
     unit, the head tilted onto it; one thermal frame (the cell's own capture, noise included). Over the pixels that see
     only the unit's face (all 3 x 3 sub-pixels of the segmentation render, eroded by one pixel for the optics' blur),
     the median apparent temperature must equal the grey-body value for the unit's own emissivity,
        T_app = (eps T^4 + (1 - eps) T_refl^4)^(1/4)  (kelvin; T_refl = the ambient, 22 degC),
     within 0.15 K; where the unit's emissivity is not the default 0.93 and the two predictions differ by 0.4 K or more,
     the median must also be at least 0.3 K away from the 0.93 prediction (the old render);
  3. what a program does: emissivity_correct(median, catalogue eps, the frame's median as the reflected temperature)
     gives back the unit's true temperature (reported, with its error);
  4. coverage: over the sites given, every surface type of the catalogue was measured, and each non-default one at
     least once where the test can tell the emissivities apart.
"""

import argparse
import json
import math
import os
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)

import numpy as np  # noqa: E402

from plantcell.server import PlantCellServer  # noqa: E402

import mujoco  # noqa: E402
import bricks.sensors.thermal as bth  # noqa: E402

DEFAULT_EPS = 0.93
SIGMA = bth.SIGMA
K0 = 273.15


def grey(t_c, eps, t_refl_c):
    L = eps * (t_c + K0) ** 4 + (1.0 - eps) * (t_refl_c + K0) ** 4
    return L ** 0.25 - K0


def place(w, xy, target):
    """Robot body at xy facing `target`, head pan 0, tilt onto the target (kinematics only)."""
    m, d = w.model, w.data
    jr = m.joint("robot/root")
    a = int(m.jnt_qposadr[jr.id])
    yaw = math.atan2(target[1] - xy[1], target[0] - xy[0])
    d.qpos[a:a + 2] = xy
    d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    jp, jt = m.joint("robot/head_pan"), m.joint("robot/head_tilt")
    d.qpos[int(m.jnt_qposadr[jp.id])] = 0.0
    cid = w.cam_ids["thermal"]
    tilt = 0.0
    for _ in range(4):
        d.qpos[int(m.jnt_qposadr[jt.id])] = tilt
        mujoco.mj_kinematics(m, d)
        c = d.cam_xpos[cid]
        tilt = float(np.clip(math.atan2(c[2] - target[2], math.hypot(target[0] - c[0], target[1] - c[1])),
                             -1.0, 0.8))
    d.qpos[int(m.jnt_qposadr[jt.id])] = tilt
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    return yaw, tilt


def site_rows(path, problems):
    srv = PlantCellServer(config={}, mode="hidden")
    srv.reset(path, seed=3, kind="hidden")
    w = srv.world
    m = w.model
    cat = srv.robot.mission()["catalog"]["thermal"]["emissivity_by_surface"]
    wo = {it["tag"]: it for it in srv.robot.mission()["work_order"]["items"]}
    amb = float(w.cfg["sensors"]["thermal"]["ambient_c"])
    rec = {}
    orig = bth.segmentation

    def seg(*a, **k):
        ids, typ = orig(*a, **k)
        rec["ids"], rec["typ"] = ids, typ
        return ids, typ
    bth.segmentation = seg
    rows = []
    try:
        for ip in w.site["ips"] + w.site.get("decoys", []):
            if ip.get("type") != "thermal":
                continue
            tr = ip.get("truth") or {}
            surface = (wo.get(ip["tag"]) or {}).get("surface") or tr.get("surface")
            want_eps = cat.get(surface)
            units = tr.get("units") or []
            hot = tr.get("geom") or (ip.get("params") or {}).get("hot_geom")
            if not units and tr.get("temp_C") is not None and hot:
                # a decoy temperature point (truth.temp_C on its hot face, no unit list): it must render at its own
                # true temperature too, not at the ambient
                units = [{"geom": hot, "label": "%s hot face (unlisted)" % ip["tag"], "temp_C": float(tr["temp_C"])}]
            for u in units:
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, u["geom"])
                if gid < 0:
                    continue
                eps = float(w.ttable.eps[gid])
                row = {"site": os.path.basename(path), "tag": ip["tag"], "unit": u.get("label"), "surface": surface,
                       "eps_catalogue": want_eps, "eps_rendered": eps, "T_true_C": float(u["temp_C"])}
                if want_eps is None or abs(eps - float(want_eps)) > 1e-12:
                    problems.append("%s %s: renders with emissivity %s, the catalogue says %s for %s" % (
                        row["site"], ip["tag"], eps, want_eps, surface))
                op = ip.get("observation_pose") or {}
                mujoco.mj_kinematics(m, w.data)
                target = np.array(w.data.geom_xpos[gid], float)
                if op.get("pos"):
                    views = [(np.asarray(op["pos"][:2], float), None)]
                else:
                    # a decoy has no observation pose: in front of its face along the hot face's normal (the box's
                    # thinnest axis, either side), 2.0 m first, then 1.5, 2.5, 3.0, 1.0 m; the first view with the
                    # face's plateau in sight counts (the camera is placed kinematically: a view inside a wall or a
                    # cubicle sees nothing and the next one is tried)
                    Rg = np.asarray(w.data.geom_xmat[gid], float).reshape(3, 3)
                    nrm = Rg[:, int(np.argmin(np.asarray(m.geom_size[gid][:3], float)))]
                    nh = nrm[:2] / max(float(np.linalg.norm(nrm[:2])), 1e-9)
                    views = [(target[:2] + dist * sgn * nh, "%.1f m in front of the face (%s normal)" % (
                        dist, "+" if sgn > 0 else "-")) for dist in (2.0, 1.5, 2.5, 3.0, 1.0) for sgn in (1.0, -1.0)]
                ss = int(w.cam_models["thermal"].supersample)
                best = None
                for at, how in views:
                    place(w, at, target)
                    fid, img, K, T = w.capture_thermal()
                    ids, typ = rec["ids"], rec["typ"]
                    hit = ((ids == gid) & (typ == int(mujoco.mjtObj.mjOBJ_GEOM))).reshape(img.shape[0], ss,
                                                                                          img.shape[1], ss).all(axis=(1, 3))
                    inner = hit.copy()
                    inner[1:-1, 1:-1] = hit[1:-1, 1:-1] & hit[:-2, 1:-1] & hit[2:, 1:-1] & hit[1:-1, :-2] & hit[1:-1, 2:]
                    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
                    n = int(inner.sum())
                    if best is None or n > best[0]:
                        best = (n, img, inner, how)
                    if n >= 20:
                        break
                n, img, inner, how = best
                if how:
                    row["render_from"] = how
                eps_doc = float(want_eps) if want_eps is not None else eps      # what the documents say
                t_own = grey(row["T_true_C"], eps_doc, amb)
                t_old = grey(row["T_true_C"], DEFAULT_EPS, amb)
                row.update({"n_px": n, "T_app_expected_C": round(t_own, 3), "T_app_if_0.93_C": round(t_old, 3)})
                if n < 20:
                    problems.append("%s %s: only %d face pixels in view" % (row["site"], ip["tag"], n))
                    rows.append(row)
                    continue
                med = float(np.median(img[inner]))
                refl = float(np.median(img))
                back = bth.emissivity_correct(med, float(want_eps), refl)
                row.update({"T_app_measured_C": round(med, 3), "err_own_K": round(med - t_own, 3),
                            "err_vs_0.93_K": round(med - t_old, 3), "frame_median_C": round(refl, 3),
                            "program_recovers_C": round(back, 3), "program_error_K": round(back - row["T_true_C"], 3),
                            "discriminating": abs(t_own - t_old) >= 0.4})
                if abs(med - t_own) > 0.15:
                    problems.append("%s %s: apparent %.3f C, expected %.3f C for the catalogue's eps %.2f" % (
                        row["site"], ip["tag"], med, t_own, eps_doc))
                if row["discriminating"] and abs(med - t_old) < 0.3:
                    problems.append("%s %s: apparent %.3f C looks like eps 0.93 (%.3f C)" % (
                        row["site"], ip["tag"], med, t_old))
                rows.append(row)
    finally:
        bth.segmentation = orig
        srv.close()
    return rows, cat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    problems, rows, cat = [], [], {}
    for s in a.sites:
        r, cat = site_rows(s, problems)
        rows += r
    measured = {r["surface"] for r in rows if "T_app_measured_C" in r}
    for sfc, eps in cat.items():
        if sfc not in measured:
            problems.append("surface type %s (eps %s) not measured on the sites given" % (sfc, eps))
        elif abs(eps - DEFAULT_EPS) > 1e-9 and not any(r.get("discriminating") for r in rows if r["surface"] == sfc):
            problems.append("surface type %s: no unit where eps %s and 0.93 can be told apart" % (sfc, eps))
    for r in rows:
        print(json.dumps(r))
    for p in problems:
        print("PROBLEM", p)
    print("thermal emissivity: %s (%d units, %d problems)" % ("OK" if not problems else "FAIL", len(rows),
                                                              len(problems)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"rows": rows, "catalogue": cat, "problems": problems}, f, indent=1)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
