"""World: a W2 site package in the simulation base, the per-site floor friction, and the set-down.

Site package (generator w2gen, tests/review/sitegen): site.json (all coordinates and the truth) + site.xml (the world only: floor
plane with the site's friction and priority 1, static boxes, 2 mm ground patches with priority 2, visual-only
chips, the search camera, the site's key light). The children of its <worldbody> and <asset> are inserted into the
simulation base's model through ModelConfig.world_xml / asset_xml with floor=False, so the site's own floor replaces
the base's floor. The generator itself is not imported here: the cell needs only the package files.

Set-down (every episode): the robot in its rest shape (the base's stand(): six short sensors at
200 mm, settled) is turned rigidly so that a seeded rollable face points down, given a seeded yaw, put with its
lowest point 0.05 m above the floor with its planar centroid at the pad plus a seeded error <= 0.10 m, and let go
with the motors braked; it settles; the three lowest end caps must be one of the six rollable faces and touch the
floor. Otherwise the next sub-seed is drawn (fresh simulator) and the rejected attempt is recorded.
"""
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.normpath(os.path.join(HERE, "..", ".."))
YALE_DIR = os.environ.get("YSIM_PARENT") or os.path.normpath(os.path.join(DEV_DIR, "..", "..", "yale"))

DROP_HEIGHT_M = 0.05          # lowest point of the robot above the floor when it is let go
SETDOWN_ERROR_M = 0.10        # max distance of the landed centroid target from the pad centre
SETTLE_VTOL = 5e-4            # m/s: settled when every node is slower than this for 0.5 s (the base's stand())
SETTLE_T_MAX = 20.0           # s
ON_FLOOR_Z = 0.025            # m: an end cap touches the floor when its centre is below this (cap radius 17.5 mm)
MAX_SUBSEEDS = 12


def ysim():
    """Read-only import of the simulation base (package parent on sys.path); returns the module namespace."""
    if YALE_DIR not in sys.path:
        sys.path.insert(0, YALE_DIR)
    import ysim as _ys                                   # noqa: WPS433
    from ysim import gaits, metrics, model, sim, params  # noqa: F401,WPS433
    return _ys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- site package
def load_site(pkg_dir):
    """(site dict, site.xml text, provenance) of a site package directory."""
    pj, px = os.path.join(pkg_dir, "site.json"), os.path.join(pkg_dir, "site.xml")
    with open(pj, encoding="utf-8") as fh:
        site = json.load(fh)
    with open(px, encoding="utf-8") as fh:
        xml = fh.read()
    prov = dict(dir=os.path.abspath(pkg_dir), site_json_sha256=sha256_file(pj), site_xml_sha256=sha256_file(px),
                generator=site.get("generator"), seed=site.get("seed"))
    return site, xml, prov


def fragments(site_xml):
    """(worldbody children, asset children) of a standalone site MJCF, as XML text for ModelConfig."""
    root = ET.fromstring(site_xml)
    wb = root.find("worldbody")
    asset = root.find("asset")
    w = "\n".join(ET.tostring(c, encoding="unicode").strip() for c in (list(wb) if wb is not None else []))
    a = "\n".join(ET.tostring(c, encoding="unicode").strip() for c in (list(asset) if asset is not None else []))
    return w, a


def model_config(site_xml=None, visual=False, mesh_res="dec", lights=True, offscreen=(1280, 720), **kw):
    """ysim ModelConfig for a site (or the base's flat floor, mu 0.8, when site_xml is None)."""
    ys = ysim()
    if site_xml is None:
        return ys.model.ModelConfig(visual=visual, mesh_res=mesh_res, lights=lights, offscreen=offscreen, **kw)
    w, a = fragments(site_xml)
    return ys.model.ModelConfig(visual=visual, mesh_res=mesh_res, world_xml=w, asset_xml=a, floor=False,
                                lights=lights, offscreen=offscreen, **kw)


def new_simulator(site_xml=None, visual=False, **kw):
    ys = ysim()
    return ys.sim.Simulator(model_config(site_xml, visual=visual, **kw))


def floor_geom_ids(S):
    """Geom ids of the site's floor plane and ground patches (name 'floor', 'patch__*')."""
    import mujoco
    out = {}
    for g in range(S.m.ngeom):
        nm = mujoco.mj_id2name(S.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm == "floor" or nm.startswith("patch__"):
            out[g] = nm
    return out


# ----------------------------------------------------------------------------------------------- rigid moves
def _quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(-1))
    return q


def rigid_move(S, R, about, to):
    """Turn the three struts rigidly by R about the point `about` and move `about` to `to`; velocities zeroed.
    A rigid move keeps every tendon length, so the cable rest lengths stay as they are."""
    import mujoco
    m, d = S.m, S.d
    qR = _quat(R)
    about, to = np.asarray(about, float), np.asarray(to, float)
    for k in range(3):
        adr = S.ids.jnt_qpos[k]
        p = d.qpos[adr:adr + 3].copy()
        d.qpos[adr:adr + 3] = to + R @ (p - about)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qR, d.qpos[adr + 3:adr + 7].copy())
        d.qpos[adr + 3:adr + 7] = q / np.linalg.norm(q)
        dof = S.ids.jnt_dof[k]
        d.qvel[dof:dof + 6] = 0.0
    d.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(m, d)


def lowest_point(S):
    return min(S._geom_zrange(g)[0] for g in S.ids.col_geoms)   # noqa: SLF001 (read-only helper of the base)


def _rot_a_to_b(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, p)
        v /= np.linalg.norm(v)
        return 2.0 * np.outer(v, v) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def face_down_rotation(nodes, face):
    """Rotation that turns the outward normal of the hull triangle `face` (three node indices) to -z."""
    N = np.asarray(nodes, float)
    i, j, k = face
    n = np.cross(N[j] - N[i], N[k] - N[i])
    if n @ (N[[i, j, k]].mean(axis=0) - N.mean(axis=0)) < 0:
        n = -n
    return _rot_a_to_b(n, [0.0, 0.0, -1.0])


# ----------------------------------------------------------------------------------------------- set-down
def draw_setdown(seed, sub, face=None, pad_xy=(0.0, 0.0), err_max=SETDOWN_ERROR_M):
    """The seeded draw of one set-down attempt: face (uniform over the six rollable faces unless given), yaw,
    position error (uniform in the disc of radius err_max)."""
    ys = ysim()
    r = np.random.default_rng([int(seed), 7, int(sub)])
    faces = ys.gaits.ROLLABLE_FACES
    f = tuple(face) if face is not None else tuple(faces[int(r.integers(0, len(faces)))])
    yaw = float(r.uniform(0.0, 2.0 * math.pi))
    e, a = err_max * math.sqrt(float(r.random())), float(r.uniform(0.0, 2.0 * math.pi))
    xy = (float(pad_xy[0]) + e * math.cos(a), float(pad_xy[1]) + e * math.sin(a))
    return dict(face=list(f), yaw_rad=yaw, target_xy=list(xy), error_m=e)


def set_down(make_sim, pad_xy, seed, face=None, require_face=False, err_max=SETDOWN_ERROR_M, z_floor=0.0):
    """Set the robot down near pad_xy. make_sim() returns a fresh Simulator (robot at the base's initial pose).
    Returns (S, info); info['attempts'] lists every rejected sub-seed with its reason. require_face=True (for
    measurements) also rejects a landing on another rollable face than the drawn one."""
    ys = ysim()
    from ysim import capability as cap      # noqa: WPS433 (placement helper of the base, read-only)
    attempts = []
    for sub in range(MAX_SUBSEEDS):
        dr = draw_setdown(seed, sub, face, pad_xy, err_max)
        S = make_sim()
        cap.place(S, pad_xy, 90.0, z_support=z_floor)
        ok_stand = S.stand()
        n = S.nodes()
        c3 = n.mean(axis=0)
        R = _rz(dr["yaw_rad"]) @ face_down_rotation(n, dr["face"])
        rigid_move(S, R, c3, [dr["target_xy"][0], dr["target_xy"][1], c3[2]])
        n = S.nodes()
        c2 = n[:, :2].mean(axis=0)
        dz = z_floor + DROP_HEIGHT_M - lowest_point(S)
        rigid_move(S, np.eye(3), [c2[0], c2[1], 0.0], [dr["target_xy"][0], dr["target_xy"][1], dz])
        t0 = S.time
        S.settle(SETTLE_T_MAX, vtol=SETTLE_VTOL)
        t_settle = S.time - t0
        b, rollable = S.bottom()
        z = np.sort(S.nodes()[:, 2])
        on_floor = bool(z[2] - z_floor < ON_FLOOR_Z)
        why = None
        if not ok_stand:
            why = "stand() timed out"
        elif not rollable:
            why = "landed on %s (not a rollable face)" % "".join(map(str, b))
        elif not on_floor:
            why = "three lowest end caps not all on the floor (z3 %.4f m)" % z[2]
        elif require_face and tuple(b) != tuple(dr["face"]):
            why = "landed on %s, drawn %s" % ("".join(map(str, b)), "".join(map(str, dr["face"])))
        if why is None:
            c, th = ys.metrics.se2(S.nodes())
            info = dict(seed=int(seed), sub_seed=sub, drawn=dr, face_landed=list(b),
                        centroid_xy=[float(c[0]), float(c[1])], heading_rad=float(th),
                        dist_from_pad_m=float(np.hypot(c[0] - pad_xy[0], c[1] - pad_xy[1])),
                        settle_s=round(t_settle, 4), sim_time_at_start_s=round(S.time, 6), attempts=attempts,
                        rule="rest shape (stand) turned so the drawn rollable face points down, seeded yaw, lowest "
                             "point %.2f m above the floor, centroid at pad + error <= %.2f m, braked, settled "
                             "(nodes < %.1f mm/s for 0.5 s, <= %.0f s); landing face must be rollable with its "
                             "three caps on the floor" % (DROP_HEIGHT_M, err_max, SETTLE_VTOL * 1000, SETTLE_T_MAX))
            return S, info
        attempts.append(dict(sub_seed=sub, drawn=dr, reason=why))
    raise RuntimeError("set-down: no acceptable landing in %d sub-seeds: %s" % (MAX_SUBSEEDS, attempts))
