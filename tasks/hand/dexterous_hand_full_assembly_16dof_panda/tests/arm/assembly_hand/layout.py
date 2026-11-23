"""Per-episode layout of the hand assembly cell: exactly what is drawn, within the bounds the documents state.

  the fixture with the palm      +-10 mm in x and y, +-2 deg about the vertical
  each tray                      +-10 mm, +-5 deg
  each finger module's chain     its ab/ad angle in its nest, uniform in +-10 deg, per nest
  the thumb's chain              its side angle in its nest, uniform in +-8 deg
  upper shell, cover, hub, cap   nominal or turned 180 deg about the vertical, each with probability one half
  the electronics stack          nominal (`stack_turn`: turned only once the cameras read its connector end)
  which finger nest holds which module: not drawn (the three are identical)

A draw is VALID when the three pieces of equipment (the fixture with its deck and the trays) are at least 15 mm apart
in plan and on the bench, and every nest lies inside the working annulus 0.30-0.62 m from the Panda's axis. An
invalid draw redraws the WHOLE layout; the jitter halves every 50 tries; after `max_tries` the nominal
layout. The server re-checks reach on top (server.py). The nominal layout's chain angles are one more fixed draw
(seed `nominal_seed`), so no fixed hidden constant shows in a film of the reference.

    episode_layout(seed, kit, cfg=None, nominal=False)      the layout dict (poses in the cell frame, m)
    equipment_polygons(lay, kit, cfg)                       {"fixture": (n, 2), "digit_tray": (4, 2), "parts_tray": (4, 2)}
"""
import copy
import math

import numpy as np

from . import fixture as FX
from . import trays as TR

DEFAULT_LAYOUT = {
    "fixture": {"jitter_m": 0.010, "yaw_jitter_deg": 2.0},
    # the trays' nominal places: the best layout found for the placement r 0.44 m, yaw 0; each
    # tray's x along its long side, its yaw = the angle of that side in the cell
    # (the parts tray 25 mm further out along its radius than that place: on its near row a part 0.33 m
    # from the Panda's axis lost its top-down approach in 28 of 300 draws)
    "trays": {"digit_tray": {"nominal_xy": [-0.181, -0.269], "yaw_deg": 45.0},
              "parts_tray": {"nominal_xy": [-0.189, 0.322], "yaw_deg": 141.0}},
    "tray_jitter_m": 0.010, "tray_yaw_jitter_deg": 5.0,
    "abad_deg": 10.0, "thumb_side_deg": 8.0,
    "flip_kinds": ["PT", "PC", "HB", "CC"], "stack_turn": False,
    "assign": {"nest_f1": "finger_module_1", "nest_f2": "finger_module_2", "nest_f3": "finger_module_3", "nest_thumb": "thumb_module",
               "nest_stack": "electronics_stack", "nest_top": "upper_shell", "nest_cover": "dorsal_cover", "nest_hub": "power_hub",
               "nest_cap": "press_cap"},
    "bench_xy": [[-0.8, 0.8], [-0.5, 0.5]], "panda_xy": [-0.45, 0.0], "annulus_m": [0.30, 0.62], "min_gap_m": 0.015,
    "max_tries": 200, "halve_every": 50, "nominal_seed": 20261120,
}


def defaults():
    return copy.deepcopy(DEFAULT_LAYOUT)


def merged(cfg=None):
    out = defaults()
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            for kk, vv in v.items():
                if isinstance(vv, dict) and isinstance(out[k].get(kk), dict):
                    out[k][kk].update(vv)
                else:
                    out[k][kk] = vv
        else:
            out[k] = v
    return out


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def planar_T(xy, yaw_rad, z=0.0):
    T = np.eye(4)
    T[:3, :3] = rz(yaw_rad)
    T[:3, 3] = [float(xy[0]), float(xy[1]), float(z)]
    return T


def fixture_T(fx):
    """the fixture frame F in the cell: P's +y along the cell's +x at yaw 0 (F = Rz(yaw) . Rz(-90 deg))"""
    return planar_T(fx["xy"], math.radians(float(fx["yaw_deg"])) - math.pi / 2)


def tray_T(tp, plate_t):
    return planar_T(tp["xy"], math.radians(float(tp["yaw_deg"])), plate_t)


def _hull(P):
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float)
    h = ConvexHull(P)
    return P[h.vertices]


def poly_gap(A, B):
    """a lower bound of the plan distance between two convex polygons (SAT over both edge sets; <= 0 if they
    overlap)"""
    best = -1e9
    for P, Q in ((A, B), (B, A)):
        n = len(P)
        for i in range(n):
            e = P[(i + 1) % n] - P[i]
            nn = np.array([e[1], -e[0]])
            nn /= np.linalg.norm(nn)
            sep = max((Q @ nn).min() - (P @ nn).max(), (P @ nn).min() - (Q @ nn).max())
            best = max(best, sep)
    return best


def fixture_plan_points(cfg=None):
    """every vertex of the fixture's pieces in plan (F, m): the plate, the pedestal, the jaws, the deck, the pad"""
    pts = []
    for row in FX.fixture_geoms(cfg):
        if row[1] == "box":
            c, h = np.array(row[2]), np.array(row[3])
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append([c[0] + sx * h[0], c[1] + sy * h[1]])
        else:
            pts += [list(p[:2]) for p in row[2]]
    return np.array(pts)


def equipment_polygons(lay, cfg_cell=None):
    cc = cfg_cell or {}
    Tf = fixture_T(lay["fixture"])
    F = fixture_plan_points(cc.get("fixture"))
    Fw = (np.c_[F, np.zeros(len(F)), np.ones(len(F))] @ Tf.T)[:, :2]
    out = {"fixture": _hull(Fw)}
    trays = TR.trays_of(cc.get("tray"))
    for tn, tp in lay["trays"].items():
        px, py, _pt = trays[tn]["plate"]
        Tt = tray_T(tp, 0.0)
        C = np.array([[-px / 2, -py / 2, 0, 1], [px / 2, -py / 2, 0, 1], [px / 2, py / 2, 0, 1], [-px / 2, py / 2, 0, 1]])
        out[tn] = (C @ Tt.T)[:, :2]
    return out


def validity(lay, cfg=None, cfg_cell=None):
    c = merged(cfg)
    polys = equipment_polygons(lay, cfg_cell)
    (bx0, bx1), (by0, by1) = c["bench_xy"]
    on_bench = all(bool(np.all((P[:, 0] >= bx0) & (P[:, 0] <= bx1) & (P[:, 1] >= by0) & (P[:, 1] <= by1))) for P in polys.values())
    names = sorted(polys)
    gaps = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            gaps["%s|%s" % (names[i], names[j])] = float(poly_gap(polys[names[i]], polys[names[j]]))
    base = np.array(c["panda_xy"], float)
    r0, r1 = c["annulus_m"]
    nest_r = {}
    trays = TR.trays_of((cfg_cell or {}).get("tray"))
    for tn, tp in lay["trays"].items():
        Tt = tray_T(tp, 0.0)
        for nest, _k, nx, ny in trays[tn]["nests"]:
            p = Tt @ np.array([nx, ny, 0.0, 1.0])
            nest_r[nest] = float(np.linalg.norm(p[:2] - base))
    in_ann = all(r0 <= r <= r1 for r in nest_r.values())
    ok = on_bench and in_ann and min(gaps.values()) >= float(c["min_gap_m"])
    return {"ok": bool(ok), "on_bench": bool(on_bench), "nests_in_annulus": bool(in_ann), "gaps_m": gaps, "nest_r_m": nest_r}


def _draw(rng, c, amp):
    fx = {"xy": [float(v) for v in FX.merged().get("nominal_xy")], "yaw_deg": float(FX.merged().get("nominal_yaw_deg"))}
    j = float(c["fixture"]["jitter_m"]) * amp
    fx["xy"] = [fx["xy"][0] + float(rng.uniform(-j, j)), fx["xy"][1] + float(rng.uniform(-j, j))]
    fx["yaw_deg"] += float(rng.uniform(-1, 1)) * float(c["fixture"]["yaw_jitter_deg"]) * amp
    trays = {}
    for tn in sorted(c["trays"]):
        t = c["trays"][tn]
        jt = float(c["tray_jitter_m"]) * amp
        trays[tn] = {"xy": [float(t["nominal_xy"][0]) + float(rng.uniform(-jt, jt)), float(t["nominal_xy"][1]) + float(rng.uniform(-jt, jt))],
                     "yaw_deg": float(t["yaw_deg"]) + float(rng.uniform(-1, 1)) * float(c["tray_yaw_jitter_deg"]) * amp}
    return fx, trays


def _chain_draws(rng, c):
    ang = {}
    for nest in ("nest_f1", "nest_f2", "nest_f3"):
        ang[nest] = float(rng.uniform(-1, 1)) * float(c["abad_deg"])
    ang["nest_thumb"] = float(rng.uniform(-1, 1)) * float(c["thumb_side_deg"])
    turned = {}
    for nest, unit in c["assign"].items():
        kind = {"nest_top": "PT", "nest_cover": "PC", "nest_hub": "HB", "nest_cap": "CC", "nest_stack": "BS"}.get(nest)
        if kind is None:
            continue
        if kind == "BS":
            turned[nest] = bool(c.get("stack_turn")) and bool(rng.uniform() < 0.5)
        else:
            turned[nest] = bool(kind in c["flip_kinds"]) and bool(rng.uniform() < 0.5)
    return ang, turned


def unit_poses(lay, kit, cfg_cell=None, cache=None):
    """{unit: 4x4 pose (m) of its root link in the cell} for every loose unit at rest in its nest, before the settle;
    and {unit: {joint: angle rad}}"""
    cc = cfg_cell or {}
    trays = TR.trays_of(cc.get("tray"))
    poses, joints = {}, {}
    for tn, tp in lay["trays"].items():
        Tt = tray_T(tp, trays[tn]["plate"][2])
        for nest, kind, _nx, _ny in trays[tn]["nests"]:
            u = lay["assign"][nest]
            Tn, _k = TR.nest_unit_T(kit, cc.get("tray"), tn, nest, cache=cache, turned=bool(lay["turned"].get(nest)))
            poses[u] = Tt @ Tn
            if kind == "FM":
                joints[u] = {"abad": math.radians(float(lay["chain_deg"][nest]))}
            elif kind == "TM":
                joints[u] = {"side": math.radians(float(lay["chain_deg"][nest]))}
    return poses, joints


def episode_layout(seed, kit=None, cfg=None, nominal=False, cfg_cell=None):
    c = merged(cfg)
    tries = 0
    amp = 1.0
    if nominal:
        rng = np.random.default_rng([int(c["nominal_seed"]) & 0xFFFFFFFF, 0x4A4E])
        ang, turned = _chain_draws(rng, c)
        fx = {"xy": list(FX.merged().get("nominal_xy")), "yaw_deg": float(FX.merged().get("nominal_yaw_deg"))}
        trays = {tn: {"xy": list(t["nominal_xy"]), "yaw_deg": float(t["yaw_deg"])} for tn, t in sorted(c["trays"].items())}
        lay = {"seed": int(seed), "nominal": True, "fixture": fx, "trays": trays, "chain_deg": ang, "turned": turned,
               "assign": dict(c["assign"]), "jitter": {"tries": 0, "amplitude": 0.0}}
        lay["validity"] = validity(lay, c, cfg_cell)
        return lay
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 0x4A4D])
    while tries < int(c["max_tries"]):
        tries += 1
        if tries > 1 and (tries - 1) % int(c["halve_every"]) == 0:
            amp *= 0.5
        fx, trays = _draw(rng, c, amp)
        ang, turned = _chain_draws(rng, c)
        lay = {"seed": int(seed), "nominal": False, "fixture": fx, "trays": trays, "chain_deg": ang, "turned": turned,
               "assign": dict(c["assign"]), "jitter": {"tries": tries, "amplitude": amp}}
        v = validity(lay, c, cfg_cell)
        lay["validity"] = v
        if v["ok"]:
            return lay
    lay = episode_layout(seed, kit, cfg, nominal=True, cfg_cell=cfg_cell)
    lay["jitter"]["fallback_nominal"] = True
    return lay
