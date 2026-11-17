"""Per-episode physical draws of the final-assembly cell (grader side): where the five stations stand on the bench, how
the kit's loose units lie in them, which thruster is in which nest (and how its mounting face and its propeller are
turned), how the enclosure lies on its V blocks, the vehicle's mounting offset on the positioner's carrier and the
load cells' zero offsets. The episode seed and the stratum in, a layout out; a pure function of its inputs
(`layout_digest` fingerprints the result).

Source (the form, rewritten for this cell): docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/
assembly_car/layout.py sha256 e96f3f2725ca1427dcd68d585fa1b3362f5f417be546090ed6ec9702d3ab9960 (seeded streams per draw,
the digest's canonical rounding, check_layout, the 2D convex helpers).

`episode_layout(seed, fx=None, cfg=None, stratum=None, nominal=False)` ->
  {"format", "seed", "stratum", "dome_deg", "nominal", "lmag",
   "stations": {name: {xy, yaw_deg}}, "redraws": {name: n},
   "parts": {kit body: (pos, quat wxyz)}      (world pose of the body's frame: T / E / G / K; a lift above the support)
   "nests": [{nest, unit, product, mount_deg, prop_deg}], "pockets": [{pocket, unit, dx_mm, dy_mm, dyaw_deg}],
   "enclosure": {roll_deg, dome_deg, axial_mm}, "guards": {GRD1 | GRD2: {rack, dx_mm, dy_mm, dyaw_deg}},
   "vehicle_mount": {"t_mm": [3], "rv_deg": [3]}, "cell_bias_N": [3], "boards": {name: 4x4 world}}

The draws (DESIGN v1.2 section 1.3; amplitudes from the frozen cell layout and fixtures.PLAYS):
* the thruster tray +-40 mm in x and in y, +-10 deg; the enclosure rack +-20 mm, +-5 deg; each guard rack +-20 mm,
  +-5 deg; the ballast box fixed (the L-mag lever, OFF by default, rulings J8 / M10: +-20 mm, +-5 deg and a board);
  each station from its own stream (switching L-mag on changes nothing but the box);
* the thrusters: which two nests hold the thruster_a pair (6 ways), which unit of a pair goes where, each mounting
  face towards the tray's +y or -y, each propeller at a uniform angle;
* the enclosure: the dome end = the stratum ("dome0" / "dome180": an INPUT, never drawn from the seed when given),
  the roll about its axis uniform in +-roll_range_deg, the axial place on the V blocks +-1 mm;
* each block in its pocket +-0.4 mm, +-0.3 deg (redrawn until its own outline is >= 0.02 mm off the pocket's walls:
  the 0.5 mm play per side does not hold both extremes at once); each guard on its rack +-1 mm, +-0.3 deg;
* the vehicle on the carrier (pins and indexing): each translation +-0.3 mm, each rotation-vector component +-0.1 deg;
* the load cells' zero offsets, each +-0.02 N.
A drawn set of station poses that breaks a validity rule (check_layout's station rules) is redrawn, only the stations
that break it, from the next draw of their own streams (at most `redraw_tries`); with the frozen layout and L-mag off
no draw breaks a rule (the stations' jitter envelopes are >= 56 mm apart), with L-mag on the box can come within
30 mm of the enclosure rack or the positioner's beam, and is redrawn.

nominal=True (the privileged reference program's fallback, FB-L): every station at its nominal pose, every play 0, no mounting
offset, no zero offsets, the thrusters THA1 THA2 THB1 THB2 in nests N1..N4 with mount 0, propellers at 0, roll 0;
the stratum's dome end still applies.

`check_layout(lay, fx=None, cfg=None)` -> (ok, problems, measures): the rules re-checked on a drawn layout (the dev
check over many seeds, -dev/tools/layout_validity.py): every station (its boxes and the units in it) on the bench,
>= station_gap_m from every other station, the bench board and the positioner's static boxes, >= arm_base_clear_m from
the arm's base axis, clear of the vehicle's swept cylinder, inside the nominal top camera's image; the units inside
their plays and their kit bodies each placed once; the nests 2 + 2; the stratum's dome end.
"""
import copy
import hashlib
import json
import math

import numpy as np

from . import fixtures as FX

__all__ = ["FORMAT", "STRATA", "KIT_BODIES", "DEFAULT_LAYOUT", "parse_stratum", "episode_layout", "layout_digest",
           "layout_digest_of", "check_layout", "T_world_station", "station_points", "station_footprints",
           "nest_products", "mount_T", "top_camera", "project"]

FORMAT = "rov-asm-layout/1"
STRATA = ("dome0", "dome180")
KIT_BODIES = {"enclosure": "ENC", "thruster_a": ("THA1", "THA2"), "thruster_b": ("THB1", "THB2"), "guards": ("GRD1", "GRD2"),
              "blocks": tuple("BAL%02d" % k for k in range(1, 13))}
GUARD_RACK = {"GRD1": "port", "GRD2": "stbd"}
DEFAULT_LAYOUT = {
    "lmag": False,
    "station_gap_m": 0.030,             # stations (with their units) apart, and from the bench board / positioner base
    "arm_base_clear_m": 0.20,           # every station point this far from the arm's base axis
    "sweep_margin_m": 0.010,            # clear of the vehicle's swept cylinder (radius from the fixture facts) by this
    "fov_margin_px": 0.0,               # inside the nominal top camera's image by this
    "redraw_tries": 200,
    "vehicle_mount": {"t_mm": 0.3, "rv_deg": 0.1},
    "cell_bias_max_N": 0.02,
    "top_camera": {"eye": [0.45, 0.0, 1.6], "target": [0.45, 0.0, 0.0], "up": [0.0, 1.0, 0.0], "size": [1280, 960],
                   "hfov_deg": 70.0},
}
# stream tags (one numpy stream per draw)
_TAG = {"stratum": 0x5157, "thruster_tray": 0x7A40, "enclosure_rack": 0xE4C0, "guard_rack_port": 0x6A9F,
        "guard_rack_stbd": 0x6A95, "ballast_box": 0xB0C5, "nests": 0x4E57, "enclosure": 0xE4C1, "blocks": 0xB10C,
        "guards": 0x6A90, "mount": 0x3017, "bias": 0xB1A5}


def _merge(cfg):
    out = copy.deepcopy(DEFAULT_LAYOUT)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = dict(out[k])
            out[k].update(v)
        else:
            out[k] = v
    return out


def _rng(seed, tag, k=0):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, int(tag), int(k)])


def parse_stratum(s):
    """the dome end (0 / 180 deg) of a stratum: 'dome0' / 'dome180' / 0 / 180"""
    if s in ("dome0", 0, 0.0):
        return "dome0", 0.0
    if s in ("dome180", 180, 180.0):
        return "dome180", 180.0
    raise ValueError("unknown stratum %r (have %s)" % (s, ", ".join(STRATA)))


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pose(T):
    return ([float(v) for v in T[:3, 3]], [float(v) for v in FX._quat_wxyz(T[:3, :3])])


def _T_pose(p):
    pos, quat = p
    return FX.T_of(_quat_R(quat), pos)


def _rotvec_R(rv):
    rv = np.asarray(rv, float)
    a = float(np.linalg.norm(rv))
    if a < 1e-15:
        return np.eye(3)
    k = rv / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def T_world_station(pose):
    """4x4 of a station frame from its {xy, yaw_deg}"""
    return FX.T_of(FX.rotz(float(pose["yaw_deg"])), [float(pose["xy"][0]), float(pose["xy"][1]), 0.0])


def mount_T(lay):
    """4x4 (m, B frame) of the vehicle's mounting offset on the carrier"""
    vm = lay.get("vehicle_mount") or {}
    return FX.T_of(_rotvec_R(np.radians(vm.get("rv_deg", [0.0, 0.0, 0.0]))), np.asarray(vm.get("t_mm", [0.0, 0.0, 0.0]), float) / 1000.0)


def nest_products(lay):
    """[product of the thruster in N1, N2, N3, N4] ('thruster_a' / 'thruster_b'): the order the hidden-set rule (d)
    compares with the slots' design parts"""
    return [n["product"] for n in sorted(lay["nests"], key=lambda n: n["nest"])]


# ------------------------------------------------------------------ the draws
def _station_pose(fx, name, seed, k, nominal):
    _s, (x, y, yaw), (j, jy) = fx.station_spec(name)
    if nominal or (j <= 0 and jy <= 0):
        return {"xy": [float(x), float(y)], "yaw_deg": float(yaw)}
    r = _rng(seed, _TAG[name], k)
    return {"xy": [float(x + r.uniform(-j, j)), float(y + r.uniform(-j, j))], "yaw_deg": float(yaw + r.uniform(-jy, jy))}


def _fixtures(fx, L):
    if fx is None:
        return FX.Fixtures(FX.load_cell_layout(), FX.load_facts(), lmag=bool(L["lmag"]))
    if bool(fx.lmag) != bool(L["lmag"]):
        raise ValueError("the fixtures' L-mag (%s) is not the layout config's (%s)" % (fx.lmag, L["lmag"]))
    return fx


def episode_layout(seed, fx=None, cfg=None, stratum=None, nominal=False):
    """the episode's layout (module docstring). fx: fixtures.Fixtures (built from the default files when None; its
    L-mag must equal cfg's); stratum: 'dome0' / 'dome180' / 0 / 180 (None: drawn from the seed)"""
    L = _merge(cfg)
    fx = _fixtures(fx, L)
    seed = int(seed)
    if stratum is None:
        stratum = STRATA[int(_rng(seed, _TAG["stratum"]).integers(0, len(STRATA)))]
    sname, dome = parse_stratum(stratum)
    lay = {"format": FORMAT, "seed": seed, "stratum": sname, "dome_deg": dome, "nominal": bool(nominal), "lmag": bool(fx.lmag)}
    # ---- stations: own streams; redraw only the stations that break a station rule
    att = {n: 0 for n in FX.STATIONS}
    st = {n: _station_pose(fx, n, seed, 0, nominal) for n in FX.STATIONS}
    for _ in range(int(L["redraw_tries"])):
        bad = _station_problems(fx, L, st, _parts_for(fx, st, lay, seed, nominal, dome), names_only=True)
        if not bad:
            break
        movable = [n for n in bad if fx.station_spec(n)[2][0] > 0 or fx.station_spec(n)[2][1] > 0]
        if nominal or not movable:
            raise ValueError("seed %d: the nominal layout breaks a station rule: %s" % (seed, sorted(bad)))
        for n in movable:
            att[n] += 1
            st[n] = _station_pose(fx, n, seed, att[n], nominal)
    else:
        raise ValueError("seed %d: no valid station draw in %d tries" % (seed, L["redraw_tries"]))
    lay["stations"] = st
    lay["redraws"] = att
    parts = _parts_for(fx, st, lay, seed, nominal, dome)
    lay.update(parts["meta"])
    lay["parts"] = {u: _pose(T) for u, T in sorted(parts["T"].items())}
    # ---- the vehicle's mounting offset and the cells' zero offsets
    if nominal:
        lay["vehicle_mount"] = {"t_mm": [0.0, 0.0, 0.0], "rv_deg": [0.0, 0.0, 0.0]}
        lay["cell_bias_N"] = [0.0, 0.0, 0.0]
    else:
        r = _rng(seed, _TAG["mount"])
        vm = L["vehicle_mount"]
        lay["vehicle_mount"] = {"t_mm": [float(v) for v in r.uniform(-vm["t_mm"], vm["t_mm"], 3)],
                                "rv_deg": [float(v) for v in r.uniform(-vm["rv_deg"], vm["rv_deg"], 3)]}
        lay["cell_bias_N"] = [float(v) for v in _rng(seed, _TAG["bias"]).uniform(-L["cell_bias_max_N"], L["cell_bias_max_N"], 3)]
    # ---- the boards in the world
    bd = {}
    for n, b in fx.boards().items():
        Tf = np.eye(4) if b["fixture"] == "bench" else T_world_station(st[b["fixture"]])
        bd[n] = (Tf @ b["T_rest"]).tolist()
    lay["boards"] = bd
    return lay


def _parts_for(fx, st, lay, seed, nominal, dome):
    """the kit bodies' world poses (4x4) for the station poses st, and the per-unit draws"""
    lift = float(FX.PLAYS["settle_lift_m"])
    T = {}
    meta = {}
    # blocks
    Tb = T_world_station(st["ballast_box"])
    pb = FX.PLAYS["ballast_block"]
    r = _rng(seed, _TAG["blocks"])
    own = fx.der["ballast_block_own_mm"]
    play = float(fx.st["ballast_box"]["play_per_side_mm"])
    pk = []
    for k, ((pn, _c), u) in enumerate(zip(fx.pockets(), KIT_BODIES["blocks"])):
        dx = dy = dw = 0.0
        if not nominal:
            # uniform in the play's box, redrawn until the block's own outline is >= BLOCK_WALL_MARGIN_MM off the walls
            for _t in range(100):
                dx, dy = float(r.uniform(-pb["xy_mm"], pb["xy_mm"])), float(r.uniform(-pb["xy_mm"], pb["xy_mm"]))
                dw = float(r.uniform(-pb["yaw_deg"], pb["yaw_deg"]))
                if _block_gap_mm(own, play, dx, dy, dw) >= BLOCK_WALL_MARGIN_MM:
                    break
            else:
                dx = dy = dw = 0.0
        T[u] = Tb @ fx.pocket_pose(k, dx, dy, dw, lift)
        pk.append({"pocket": pn, "unit": u, "dx_mm": dx, "dy_mm": dy, "dyaw_deg": dw})
    meta["pockets"] = pk
    # thrusters
    Tt = T_world_station(st["thruster_tray"])
    names = [n for n, _c in fx.nests()]
    if nominal:
        occ = ["THA1", "THA2", "THB1", "THB2"]
        mounts, props = [0.0] * 4, [0.0] * 4
    else:
        r = _rng(seed, _TAG["nests"])
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        ia = pairs[int(r.integers(0, len(pairs)))]
        a_units = list(KIT_BODIES["thruster_a"])
        b_units = list(KIT_BODIES["thruster_b"])
        if r.integers(0, 2):
            a_units.reverse()
        if r.integers(0, 2):
            b_units.reverse()
        occ = [None] * 4
        for k in range(4):
            occ[k] = a_units.pop(0) if k in ia else b_units.pop(0)
        mounts = [180.0 if r.integers(0, 2) else 0.0 for _ in range(4)]
        props = [float(r.uniform(0.0, 360.0)) for _ in range(4)]
    ns = []
    for k in range(4):
        u = occ[k]
        prod = "thruster_a" if u in KIT_BODIES["thruster_a"] else "thruster_b"
        T[u] = Tt @ fx.nest_pose(k, mounts[k], lift)
        ns.append({"nest": names[k], "unit": u, "product": prod, "mount_deg": mounts[k], "prop_deg": props[k]})
    meta["nests"] = ns
    # enclosure
    Te = T_world_station(st["enclosure_rack"])
    rr = float(FX.PLAYS["enclosure_roll_deg"] if FX.PLAYS["enclosure_roll_deg"] is not None
               else fx.st["enclosure_rack"]["roll_range_deg"])
    if nominal:
        roll, ax = 0.0, 0.0
    else:
        r = _rng(seed, _TAG["enclosure"])
        roll = float(r.uniform(-rr, rr))
        ax = float(r.uniform(-FX.PLAYS["enclosure_axial_mm"], FX.PLAYS["enclosure_axial_mm"]))
    T[KIT_BODIES["enclosure"]] = Te @ fx.enclosure_pose(roll, dome, ax, lift)
    meta["enclosure"] = {"roll_deg": roll, "dome_deg": float(dome), "axial_mm": ax}
    # guards
    pg = FX.PLAYS["guard"]
    r = _rng(seed, _TAG["guards"])
    gd = {}
    for u in KIT_BODIES["guards"]:
        side = GUARD_RACK[u]
        dx, dy, dw = (0.0, 0.0, 0.0) if nominal else (float(r.uniform(-pg["x_mm"], pg["x_mm"])), float(r.uniform(-pg["y_mm"], pg["y_mm"])),
                                                       float(r.uniform(-pg["yaw_deg"], pg["yaw_deg"])))
        T[u] = T_world_station(st["guard_rack_" + side]) @ fx.guard_pose(side, dx, dy, dw, lift)
        gd[u] = {"rack": "guard_rack_" + side, "dx_mm": dx, "dy_mm": dy, "dyaw_deg": dw}
    meta["guards"] = gd
    return {"T": T, "meta": meta}


# ------------------------------------------------------------------ geometry for the checks
_KIND = {"ENC": "enclosure", "GRD1": "guard", "GRD2": "guard"}


def _kind(u):
    if u in _KIND:
        return _KIND[u]
    if u.startswith("BAL"):
        return "ballast"
    return "thruster_a" if u in KIT_BODIES["thruster_a"] else "thruster_b"


_STATION_UNITS = {"ballast_box": KIT_BODIES["blocks"], "thruster_tray": KIT_BODIES["thruster_a"] + KIT_BODIES["thruster_b"],
                  "enclosure_rack": (KIT_BODIES["enclosure"],), "guard_rack_port": ("GRD1",), "guard_rack_stbd": ("GRD2",)}


def station_points(fx, name, pose, part_T):
    """(n, 3) world points of a station: its boxes' corners and the hull points of the units resting in it"""
    Tw = T_world_station(pose)
    P = [FX.box_corners(c, R, h) @ Tw[:3, :3].T + Tw[:3, 3] for _bn, c, R, h, _p in fx.station_boxes(name)]
    for u in _STATION_UNITS[name]:
        if u in part_T:
            T = part_T[u]
            P.append(fx.hull(_kind(u)) @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(P)


def _hull2(P):
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float)[:, :2]
    if len(P) <= 4:
        c = P.mean(0)
        return P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]
    return P[ConvexHull(P).vertices]


def _overlap(A, B):
    for P, Q in ((A, B), (B, A)):
        E = np.roll(P, -1, axis=0) - P
        N = np.c_[E[:, 1], -E[:, 0]]
        pa, pb = P @ N.T, Q @ N.T
        if np.any(pa.min(0) > pb.max(0)) or np.any(pb.min(0) > pa.max(0)):
            return False
    return True


def _pt_seg(P, A, B):
    AB = B - A
    Ln = np.maximum((AB * AB).sum(1), 1e-18)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / Ln[None], 0.0, 1.0)
    C = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - C, axis=2)


def poly_dist(A, B):
    """distance of two convex polygons (0 when they overlap)"""
    if _overlap(A, B):
        return 0.0
    return float(min(_pt_seg(A, B, np.roll(B, -1, axis=0)).min(), _pt_seg(B, A, np.roll(A, -1, axis=0)).min()))


def station_footprints(fx, stations, part_T):
    """{name: (convex plan hull (n, 2), all points (m, 3))} of every station with its units"""
    out = {}
    for n in FX.STATIONS:
        P = station_points(fx, n, stations[n], part_T)
        out[n] = (_hull2(P), P)
    return out


def _bench_board_poly(fx):
    P = fx.board_plate_corners("bench")
    return _hull2(P), P


def _static_polys(fx):
    return [(nm, _hull2(FX.box_corners(c, np.eye(3), h))) for nm, c, h in fx.positioner_boxes()["static"]]


def top_camera(L):
    """(R_cam->world 3x3 (x right, y down, z forward), eye (3,), f px, cx, cy, W, H) of the nominal top camera"""
    c = L["top_camera"]
    eye = np.asarray(c["eye"], float)
    z = np.asarray(c["target"], float) - eye
    z /= np.linalg.norm(z)
    up = np.asarray(c["up"], float)
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    W, H = (int(v) for v in c["size"])
    f = 0.5 * W / math.tan(math.radians(float(c["hfov_deg"])) / 2.0)
    return np.c_[x, y, z], eye, f, (W - 1) / 2.0, (H - 1) / 2.0, W, H


def project(L, P):
    """(u, v, depth) px of world points in the nominal top camera (pinhole, no distortion)"""
    R, eye, f, cx, cy, _W, _H = top_camera(L)
    Q = (np.asarray(P, float) - eye) @ R
    return cx + f * Q[:, 0] / Q[:, 2], cy + f * Q[:, 1] / Q[:, 2], Q[:, 2]


def _fov_margin(L, P):
    R, eye, f, cx, cy, W, H = top_camera(L)
    u, v, z = project(L, P)
    if np.any(z <= 0):
        return -1e9
    return float(min(u.min(), (W - 1) - u.max(), v.min(), (H - 1) - v.max()))


def _sweep_clear(fx, P):
    """smallest (distance to the hinge axis - swept radius) of the points within the vehicle's x range (m); +inf if none"""
    o = fx.vehicle_origin()
    lo, hi = fx.vehicle_box_B()
    R = float(fx.F["vehicle"]["sweep_radius_full_mm"]) / 1000.0
    sel = (P[:, 0] >= o[0] + lo[0]) & (P[:, 0] <= o[0] + hi[0])
    if not np.any(sel):
        return float("inf")
    d = np.hypot(P[sel, 1] - o[1], P[sel, 2] - o[2])
    return float(d.min() - R)


def _station_problems(fx, L, st, parts, names_only=False):
    """the station rules: on the bench, apart (>= station_gap_m) from each other / the bench board / the positioner's
    static boxes, clear of the arm's base axis and of the vehicle's swept cylinder, inside the top camera's image.
    names_only: the set of station names that break a rule; else (problems, measures)"""
    fp = station_footprints(fx, st, parts["T"])
    g = float(L["station_gap_m"])
    (x0, x1), (y0, y1) = FX.TABLE["x_m"], FX.TABLE["y_m"]
    bad, probs = set(), []
    meas = {"pair_gap_m": {}, "table_margin_m": {}, "board_gap_m": {}, "positioner_gap_m": {}, "arm_base_m": {},
            "sweep_clear_m": {}, "fov_margin_px": {}}
    names = list(FX.STATIONS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd = poly_dist(fp[names[i]][0], fp[names[j]][0])
            meas["pair_gap_m"]["%s|%s" % (names[i], names[j])] = dd
            if dd < g - 1e-12:
                bad |= {names[i], names[j]}
                probs.append("%s and %s %.1f mm apart" % (names[i], names[j], dd * 1e3))
    bb, bbP = _bench_board_poly(fx)
    stat = _static_polys(fx)
    for n in names:
        H2, P = fp[n]
        m = float(min(H2[:, 0].min() - x0, x1 - H2[:, 0].max(), H2[:, 1].min() - y0, y1 - H2[:, 1].max()))
        meas["table_margin_m"][n] = m
        if m < 0:
            bad.add(n)
            probs.append("%s off the bench (%.1f mm)" % (n, m * 1e3))
        db = poly_dist(H2, bb)
        meas["board_gap_m"][n] = db
        if db < g - 1e-12:
            bad.add(n)
            probs.append("%s %.1f mm from the bench board" % (n, db * 1e3))
        dp = min(poly_dist(H2, Q) for _nm, Q in stat)
        meas["positioner_gap_m"][n] = dp
        if dp < g - 1e-12:
            bad.add(n)
            probs.append("%s %.1f mm from the positioner's base" % (n, dp * 1e3))
        da = float(np.min(np.hypot(P[:, 0], P[:, 1])))
        if _overlap(H2, _hull2(np.array([[0.0, 0.0], [1e-9, 0.0], [0.0, 1e-9]]))):
            da = 0.0
        meas["arm_base_m"][n] = da
        if da < float(L["arm_base_clear_m"]):
            bad.add(n)
            probs.append("%s %.0f mm from the arm's base axis" % (n, da * 1e3))
        sc = _sweep_clear(fx, P)
        meas["sweep_clear_m"][n] = sc
        if sc < float(L["sweep_margin_m"]):
            bad.add(n)
            probs.append("%s in the vehicle's swept cylinder (%.1f mm)" % (n, sc * 1e3))
        fm = _fov_margin(L, P)
        meas["fov_margin_px"][n] = fm
        if fm < float(L["fov_margin_px"]):
            bad.add(n)
            probs.append("%s outside the top camera's image (%.1f px)" % (n, fm))
    # the bench board itself: on the bench, in the image, clear of the positioner
    if names_only:
        return bad
    mb = float(min(bb[:, 0].min() - x0, x1 - bb[:, 0].max(), bb[:, 1].min() - y0, y1 - bb[:, 1].max()))
    meas["bench_board"] = {"table_margin_m": mb, "fov_margin_px": _fov_margin(L, bbP),
                           "positioner_gap_m": min(poly_dist(bb, Q) for _nm, Q in stat)}
    if mb < 0 or meas["bench_board"]["fov_margin_px"] < float(L["fov_margin_px"]) or meas["bench_board"]["positioner_gap_m"] < g:
        probs.append("the bench board breaks a rule: %s" % meas["bench_board"])
    return probs, meas


# ------------------------------------------------------------------ the fingerprint
def _canon(x, nd):
    if isinstance(x, dict):
        return {str(k): _canon(v, nd) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_canon(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        v = round(float(x), nd)
        return 0.0 if v == 0 else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _canon(x.tolist(), nd)
    return x


DIGEST_KEYS = ("format", "stratum", "dome_deg", "nominal", "lmag", "stations", "parts", "nests", "pockets", "enclosure",
               "guards", "vehicle_mount", "cell_bias_N")


def layout_digest(lay):
    """sha256 (hex) of the layout's physical content (DIGEST_KEYS), floats rounded to 1e-7 (m, quaternion components,
    deg, mm, N): the same seed, stratum, fixtures and config give the same digest in any process"""
    body = {k: lay.get(k) for k in DIGEST_KEYS}
    s = json.dumps(_canon(body, 7), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def layout_digest_of(seed, fx=None, cfg=None, stratum=None, nominal=False):
    return layout_digest(episode_layout(seed, fx=fx, cfg=cfg, stratum=stratum, nominal=nominal))


# ------------------------------------------------------------------ the dev check
def check_layout(lay, fx=None, cfg=None):
    """(ok, problems, measures) of a drawn layout (module docstring)"""
    L = _merge(cfg)
    if bool(lay.get("lmag")) != bool(L["lmag"]):
        L["lmag"] = bool(lay.get("lmag"))
    fx = _fixtures(fx, L)
    T = {u: _T_pose(p) for u, p in lay["parts"].items()}
    probs, meas = _station_problems(fx, L, lay["stations"], {"T": T})
    # every kit body placed once
    want = set(KIT_BODIES["blocks"]) | set(KIT_BODIES["thruster_a"]) | set(KIT_BODIES["thruster_b"]) | set(KIT_BODIES["guards"]) | {KIT_BODIES["enclosure"]}
    if set(T) != want:
        probs.append("kit bodies placed: %s (want %s)" % (sorted(T), sorted(want)))
    # the stations inside their jitter (or nominal)
    for n in FX.STATIONS:
        _s, (x, y, yaw), (j, jy) = fx.station_spec(n)
        p = lay["stations"][n]
        if lay.get("nominal"):
            j = jy = 0.0
        if abs(p["xy"][0] - x) > j + 1e-12 or abs(p["xy"][1] - y) > j + 1e-12 or abs(p["yaw_deg"] - yaw) > jy + 1e-9:
            probs.append("%s outside its jitter" % n)
    # nests: 2 + 2, mounts 0 / 180, the units where the nest record says
    prods = nest_products(lay)
    if sorted(prods) != ["thruster_a", "thruster_a", "thruster_b", "thruster_b"]:
        probs.append("nests are not 2 + 2: %s" % prods)
    Tt = T_world_station(lay["stations"]["thruster_tray"])
    lift = float(FX.PLAYS["settle_lift_m"])
    for k, n in enumerate(sorted(lay["nests"], key=lambda n: n["nest"])):
        if n["mount_deg"] not in (0.0, 180.0):
            probs.append("%s mount %s" % (n["nest"], n["mount_deg"]))
        if n["unit"] not in T:
            continue
        if not np.allclose(T[n["unit"]], Tt @ fx.nest_pose(k, n["mount_deg"], lift), atol=1e-9):
            probs.append("%s is not in %s as recorded" % (n["unit"], n["nest"]))
    # the thrusters' ducts apart (plan circles of the lip radius)
    lip = float(fx.F["thruster_T_mm"]["lip_r"]) / 1000.0
    ax = [T[n["unit"]][:2, 3] for n in lay["nests"] if n["unit"] in T]
    dmin = min((float(np.linalg.norm(ax[i] - ax[j])) - 2 * lip for i in range(len(ax)) for j in range(i + 1, len(ax))), default=float("inf"))
    meas["duct_gap_min_m"] = dmin
    if dmin <= 0:
        probs.append("two thruster ducts overlap")
    # plays
    pb = FX.PLAYS["ballast_block"]
    for q in lay["pockets"]:
        if abs(q["dx_mm"]) > pb["xy_mm"] + 1e-9 or abs(q["dy_mm"]) > pb["xy_mm"] + 1e-9 or abs(q["dyaw_deg"]) > pb["yaw_deg"] + 1e-9:
            probs.append("%s outside its pocket play" % q["unit"])
    meas["block_wall_gap_min_mm"] = _block_wall_gap(fx, lay)
    if meas["block_wall_gap_min_mm"] < 0:
        probs.append("a block reaches into its pocket's wall")
    pg = FX.PLAYS["guard"]
    for u, q in lay["guards"].items():
        if abs(q["dx_mm"]) > pg["x_mm"] + 1e-9 or abs(q["dy_mm"]) > pg["y_mm"] + 1e-9 or abs(q["dyaw_deg"]) > pg["yaw_deg"] + 1e-9:
            probs.append("%s outside its play" % u)
    e = lay["enclosure"]
    rr = float(fx.st["enclosure_rack"]["roll_range_deg"])
    if abs(e["roll_deg"]) > rr + 1e-9 or abs(e["axial_mm"]) > FX.PLAYS["enclosure_axial_mm"] + 1e-9:
        probs.append("enclosure outside its roll / axial range")
    sname, dome = parse_stratum(lay["stratum"])
    if float(e["dome_deg"]) != dome or float(lay["dome_deg"]) != dome:
        probs.append("dome end %s against the stratum %s" % (e["dome_deg"], sname))
    vm = lay["vehicle_mount"]
    if max(abs(v) for v in vm["t_mm"]) > L["vehicle_mount"]["t_mm"] + 1e-12 or max(abs(v) for v in vm["rv_deg"]) > L["vehicle_mount"]["rv_deg"] + 1e-12:
        probs.append("vehicle mount outside its bounds")
    if max(abs(v) for v in lay["cell_bias_N"]) > L["cell_bias_max_N"] + 1e-12:
        probs.append("cell zero offset outside its bound")
    meas["min_pair_gap_m"] = min(meas["pair_gap_m"].values())
    meas["min_table_margin_m"] = min(meas["table_margin_m"].values())
    meas["min_fov_margin_px"] = min(meas["fov_margin_px"].values())
    meas["min_sweep_clear_m"] = min(meas["sweep_clear_m"].values())
    meas["min_arm_base_m"] = min(meas["arm_base_m"].values())
    meas["min_board_gap_m"] = min(meas["board_gap_m"].values())
    meas["min_positioner_gap_m"] = min(meas["positioner_gap_m"].values())
    return not probs, probs, meas


BLOCK_WALL_MARGIN_MM = 0.02


def _block_gap_mm(own, play, dx, dy, dyaw):
    """the plan gap (mm) between a block's own outline (own[0] x own[1], turned by dyaw, shifted by dx, dy) and its
    pocket's walls (the outline + play per side)"""
    a = math.radians(dyaw)
    c, s = math.cos(a), math.sin(a)
    ex = abs(c) * own[0] / 2 + abs(s) * own[1] / 2
    ey = abs(s) * own[0] / 2 + abs(c) * own[1] / 2
    return min(own[0] / 2 + play - (abs(dx) + ex), own[1] / 2 + play - (abs(dy) + ey))


def _block_wall_gap(fx, lay):
    """smallest plan gap (mm) between a block's own outline and its pocket's walls (box frame)"""
    s = fx.st["ballast_box"]
    own = fx.der["ballast_block_own_mm"]
    pw = own[0] / 2 + s["play_per_side_mm"]
    pd = own[1] / 2 + s["play_per_side_mm"]
    out = float("inf")
    for (pn, c), q in zip(fx.pockets(), lay["pockets"]):
        a = math.radians(q["dyaw_deg"])
        corners = np.array([[sx * own[0] / 2, sy * own[1] / 2] for sx in (-1, 1) for sy in (-1, 1)])
        Rz = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        P = corners @ Rz.T + np.array([q["dx_mm"], q["dy_mm"]])
        out = min(out, float(pw - np.abs(P[:, 0]).max()), float(pd - np.abs(P[:, 1]).max()))
    return out
