"""The ladder's proofs (their definitions, with the numbers over the seeds, are kept with the generator's notes).

Everything here works on a site record (site.json), its documents (prior.json), its traversability map (trav.npz) and
the compiled scene (scene.xml with the instruments in their true states), so the generator (in memory, before it
admits a sub-seed) and the maintainers' check of saved sites (from the files) run the same code.

Cells are camera positions (the pan-tilt centre) at the centres of the 0.05 m grid. "Reachable without crossing E":
the robot's footprint graph with the cells of E made solid for every mode. The head can be up to head.reach_m from
the body centre, so the cells a robot's camera can occupy are the reachable body cells dilated by that distance.
"""

from __future__ import annotations

import json
import math
import os

import mujoco
import numpy as np
from scipy import ndimage

from . import radiation as RAD
from .accept import (Acceptance, RayCaster, camera, camera_from_body, evgeoms, feature_of,  # noqa: F401
                     generous_region, heights, los, margin_region, nominal_head, tag_seen)
from .config import load_config
from .trav import MODES, TravMap
from .util import runs_from_mask

INSTR = ("A", "B-step", "B-kerb", "B-low", "C-sealed", "C-moved", "C-relocated", "C-annex")
SLOT_MODES = {"A": ("walk",), "B-step": ("walk", "high_step"), "B-kerb": ("walk", "high_step"), "B-low": MODES,
              "C-sealed": MODES, "C-moved": MODES, "C-relocated": MODES, "C-annex": MODES, "M": MODES, "X": MODES,
              "HALL": MODES}
CUT_SLOTS = ("B-step", "B-kerb", "B-low", "C-sealed", "C-annex")    # C-annex since 2026-11-14


# ============================================================================================ context
class Ctx:
    def __init__(self, site, prior, trav, model, data, cfg, site_dir=None, fields=None):
        self.site, self.prior, self.trav, self.model, self.data, self.cfg = site, prior, trav, model, data, cfg
        self.dir = site_dir
        self.acc = Acceptance(cfg)
        self.grid = trav.g
        self.L = cfg["ladder"]
        self.res = trav.g.res
        self.recs = {r["tag"]: r for r in site["ips"]}
        self.by_slot = {r["slot"]: r for r in site["ips"]}
        self.elements = {e["id"]: e for e in site["elements"]}
        self.doors = {d["id"]: d for d in site["doors"]}
        self.walls = {w["id"]: w for w in site["walls"]}
        self.rooms = {r["id"]: r for r in site["rooms"]}
        self.regions = {}
        self.fields = fields or {}
        self.pad = trav.node(0.0, 0.0, "walk")
        self._reach = {}
        self.inside = trav.inside.astype(bool)
        # version 2: the missing sheet (site_gen.sheet.Missing) or None
        ms = site.get("missing_sheet")
        self.miss = None
        if isinstance(ms, dict) and ms.get("place"):
            from .sheet import Missing
            self.miss = Missing.from_json(ms, site["building"]["rect"])
        self.v2 = bool((site.get("ladder") or {}).get("v2"))

    def part_mask(self):
        """Cells of the missing part (missing rooms' rects and cut-off corridor parts), or all False."""
        if getattr(self, "_part", None) is None:
            X, Y = self.grid.centers()
            m = np.zeros(self.grid.shape, bool)
            if self.miss is not None:
                for r in self.miss.areas():
                    m |= (X >= r[0]) & (X <= r[2]) & (Y >= r[1]) & (Y <= r[3])
            self._part = m
        return self._part

    def on_sheet(self, rec):
        """Is a listed instrument's true place on the missing sheet (site_gen.ladder.slot_on_sheet)?"""
        if self.miss is None:
            return False
        from .ladder import slot_on_sheet
        L = self.site["ladder"]
        return slot_on_sheet(self.miss, rec["room"], rec["pose"]["pos"][:2], L["annex"], L["annex_host"])

    # ---------------------------------------------------------------- reach
    def reach_nodes(self, modes=MODES, blocked=None, key=None):
        k = (tuple(modes), key)
        if k not in self._reach:
            enter = self.trav.enter_blocked(blocked) if blocked is not None else None
            self._reach[k] = self.trav.reach_from(self.pad, modes, enter)
        return self._reach[k]

    def reach_cells(self, modes=MODES, blocked=None, key=None):
        return self.trav.cells(self.reach_nodes(modes, blocked, key))

    def reach_mode(self, mode, modes=MODES, blocked=None, key=None):
        return self.trav.cells_mode(self.reach_nodes(modes, blocked, key), mode)

    def stand_crouch(self, modes):
        st = np.zeros(self.grid.shape, bool)
        cr = np.zeros(self.grid.shape, bool)
        for m in modes:
            c = self.reach_mode(m, modes)
            if m == "crouch_walk":
                cr |= c
            else:
                st |= c
        return st, cr & ~st

    def dilate(self, mask):
        r = float(self.L["head"]["reach_m"])
        d = ndimage.distance_transform_edt(~mask)
        return (d * self.res <= r + 1e-9) & self.inside

    # ---------------------------------------------------------------- cut masks
    def elem_mask(self, ids):
        m = np.zeros(self.grid.shape, np.uint8)
        for i in ids:
            if i in self.elements:
                self.grid.fill_poly(m, self.elements[i]["footprint"], 1)
        return m.astype(bool)

    def door_mask(self, did, extra=0.15):
        d = self.doors[did]
        w = self.walls[d["wall"]]
        a, b = d["center"] - d["width"] / 2 - 0.05, d["center"] + d["width"] / 2 + 0.05
        if w["axis"] == "x":
            r = [w["coord"] - w["t_neg"] - extra, a, w["coord"] + w["t_pos"] + extra, b]
        else:
            r = [a, w["coord"] - w["t_neg"] - extra, b, w["coord"] + w["t_pos"] + extra]
        m = np.zeros(self.grid.shape, np.uint8)
        self.grid.fill_poly(m, [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], 1)
        return m.astype(bool)

    def cut(self, slot):
        Ld = self.site["ladder"]
        if slot == "sill":
            return self.elem_mask([Ld["corridor_sill"]])
        if slot == "B-step":
            return self.elem_mask(Ld["b_step_elements"])
        if slot == "B-kerb":
            return self.elem_mask(Ld["b_kerb_elements"])
        if slot == "B-low":
            return self.elem_mask(Ld["b_low_elements"])
        if slot == "C-sealed":
            return self.door_mask(Ld["as_built_opening"])
        if slot == "C-annex":
            return self.door_mask(Ld["annex_door"])
        raise KeyError(slot)

    def a_side(self):
        return self.reach_cells(MODES, self.cut("sill"), "sill")

    def link_base(self):
        if "link_base_cells" not in self.fields:
            if "link_base" in self.fields:
                cov, x0, y0, res = self.fields["link_base"], *self.fields["link_origin"], self.fields["link_res"]
            else:
                f = np.load(os.path.join(self.dir, "fields.npz"))
                cov, (x0, y0), res = f["link_base"], f["link_origin"], float(f["link_res"])
            X, Y = self.grid.centers()
            iy = np.clip(np.floor((Y - y0) / res).astype(int), 0, cov.shape[0] - 1)
            ix = np.clip(np.floor((X - x0) / res).astype(int), 0, cov.shape[1] - 1)
            self.fields["link_base_cells"] = cov[iy, ix]
        return self.fields["link_base_cells"]

    def tight_free(self, modes):
        """Per mode the cells reachable (in `modes`) with every tight element (narrow gap, door left ajar) solid, or
        None when the site has none (observation poses off tight passages)."""
        ids = [e["id"] for e in self.site["elements"] if e["class"] == "tight"]
        if not ids:
            return None
        union = self.elem_mask(ids)
        return {m: self.reach_mode(m, modes, union, ("tight", tuple(modes))) for m in modes}

    def link3d(self, posture="walk", base_only=True, region=None):
        """Cells (of `region`; default the airlock side of the corridor sill and the pad circle's neighbourhood) where
        the link is up as the cell decides it (site_gen.linkray: rays from the lidar site to a working node against
        group-0 geometry, the open door leaves included), the robot's lidar at the posture's height."""
        from . import linkray as LR
        key = ("link3d", posture, base_only)
        if region is not None or key not in self.fields:
            lk = LR.cell_link_config(self.cfg)
            L = self.site["link"]
            nodes = LR.nodes_of(L["base_station"], [] if base_only else L.get("repeaters", []), lk)
            reg = region if region is not None else \
                (self.a_side() | LR.disc(self.grid, (0.0, 0.0), 1.6)) & self.inside
            m = LR.link_mask(self.model, self.data, self.grid, reg, nodes, float(lk["end_margin_m"]),
                             LR.lidar_heights()[posture])
            if region is not None:
                return m
            self.fields[key] = m
        return self.fields[key]

    def body_id(self, rec):
        b = rec.get("body")
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) if b else -1

    def feature(self, rec):
        """The instrument's evidence geometry from the cell's EvidenceModel on the compiled site."""
        return self.evgeoms()[rec["tag"]]

    def evgeoms(self):
        if getattr(self, "_evg", None) is None:
            recs = [r for r in self.site["ips"] if r["type"] not in ("dose_point", "hotspot")]
            self._evg = evgeoms(self.model, self.data, recs, self.acc.ev)
        return self._evg

    @property
    def caster(self):
        if getattr(self, "_caster", None) is None:
            self._caster = RayCaster(self.model, self.data)
        return self._caster

    def camera_cells(self):
        """Where the robot's camera can be: the cells reachable in any mode, dilated by the head's reach."""
        return self.dilate(self.reach_cells(MODES))

    def rate(self, pts, only=None, exclude=None):
        srcs = [s for s in self.site["sources"] if (only is None or s["id"] in only) and
                (exclude is None or s["id"] not in exclude)]
        r = self.site["radiation"]
        bg = r["background_uSv_h"] if only is None else 0.0
        return RAD.dose_rate(np.asarray(pts, float), srcs, [tuple(b) for b in r["boxes"]], self.cfg, bg)


def load(site_dir, cfg=None, dressed=False):
    """A Ctx from a saved site folder. dressed: the scene as the task runs it (version 2,
    the dressing's site_gen.dress over the stored scene; the bundle from SITE_GEN_DRESS_BUNDLE), so that the
    rows are made on the dressed scene's geometry."""
    site = json.load(open(os.path.join(site_dir, "site.json")))
    prior = json.load(open(os.path.join(site_dir, "prior.json")))
    if cfg is None:
        lc = site["ladder"]["config"]
        cfg = load_config(overrides={"layout": "ladder_v1", "capability": site["capability_bands"],
                                     "ladder": dict(lc, acceptance_file=site["ladder"]["acceptance_file"])})
    if dressed:
        from .dress import dress_spec
        spec = mujoco.MjSpec.from_file(os.path.join(site_dir, "scene.xml"))
        dress_spec(spec, site)
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(os.path.join(site_dir, "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    from .ip_models import apply_truth_states
    apply_truth_states(model, data, site["ips"] + site["decoys"] + site.get("plain_instruments", []))
    trav = TravMap.load(os.path.join(site_dir, "trav.npz"), cfg)
    return Ctx(site, prior, trav, model, data, cfg, site_dir)


# ============================================================================================ regions
BACK_EPS_M = 1e-6      # a cell centre ON the far edge is not beyond it (the far edge often lies on a row of cell
                       # centres; without the margin float noise decided it on a development site)


def back_part_mask(grid, back_part):
    """Cells whose centre lies in a variant room's back part (site.json ladder.v2_obstacles[].zones.back_part: from
    the strip's or bridge's far edge to the back wall), strictly beyond the far edge (by more than BACK_EPS_M)."""
    X, Y = grid.centers()
    bp = [float(v) for v in back_part]
    e = BACK_EPS_M
    return (X > bp[0] + e) & (X < bp[2] - e) & (Y > bp[1] + e) & (Y < bp[3] - e)


def b_low_back_mask(ctx):
    """Version 2, the rack variant: the cells of B-low's room behind the
    pipe bridge where its reading place may lie, body centre at least variants.RACK_READ_CLEAR_M (0.30 m) beyond the
    bridge's far edge (the whole body behind it), or None in the chase variant and in version 1."""
    if not ctx.v2:
        return None
    from .variants import rack_reading_rect
    for o in ctx.site["ladder"].get("v2_obstacles", []):
        if o.get("slot") == "B-low":
            return back_part_mask(ctx.grid, rack_reading_rect(o["zones"]))
    return None


def compute_regions(ctx: Ctx):
    """Per instrument: the generous region (camera cells; a set that holds every accepted pose), the margin regions
    (BODY cells standing / crouching; subset), from the cell's evidence geometry (accept.py). Version 2, the
    rack variant: B-low's margin regions (both postures) are clipped to the back part from 0.30 m beyond
    the bridge's far edge (b_low_back_mask), so that its reading place, its stored readable_region and its
    observation pose lie behind the rack with the whole body."""
    cfg = ctx.cfg
    t = ctx.trav
    st_cells = t.enter["walk"] | t.enter["high_step"]
    cr_cells = t.enter["crouch_walk"] & ~st_cells
    X, Y = ctx.grid.centers()
    cam_cells = ctx.camera_cells()
    hs, hc = nominal_head(cfg, "stand"), nominal_head(cfg, "crouch")
    bl_back = b_low_back_mask(ctx)
    for rec in ctx.site["ips"]:
        if rec["slot"] in ("M", "X"):
            continue
        eg = ctx.feature(rec)
        gen = generous_region(ctx.acc, ctx.caster, ctx.grid, eg, cam_cells)
        # one cell wider (the region holds every accepted pose at cell centres and sampled heights only; 1,168 of 70,108
        # accepted camera cells of 16 fresh sites lay just outside it); the exclusion proofs use the wider region
        gen = ndimage.binary_dilation(gen, structure=np.ones((3, 3), bool)) & cam_cells
        ms = margin_region(ctx.acc, ctx.caster, ctx.grid, eg, st_cells, hs)
        mc = margin_region(ctx.acc, ctx.caster, ctx.grid, eg, cr_cells, hc)
        if bl_back is not None and rec["slot"] == "B-low":
            ms, mc = ms & bl_back, mc & bl_back          # the reading place lies behind the rack
        ctx.regions[rec["tag"]] = {"feat": eg, "generous": gen, "stand": ms, "crouch": mc}
    m = ctx.by_slot["M"]
    p = m["pose"]["pos"]
    rr = float(ctx.acc.A["mark"]["evidence"]["within_m"])
    d2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2
    ctx.regions[m["tag"]] = {"generous": (d2 <= rr * rr) & ctx.inside, "stand": (d2 <= (rr - 0.05) ** 2) & st_cells,
                             "crouch": (d2 <= (rr - 0.05) ** 2) & cr_cells}
    _x_region(ctx)


def _x_source(ctx):
    """The source of the item in slot X (the unlisted source; the training hall's check source)."""
    sid = (ctx.by_slot.get("X", {}).get("truth") or {}).get("source")
    for s in ctx.site["sources"]:
        if s["id"] == sid:
            return s
    return [s for s in ctx.site["sources"] if not s["documented"]][0]


def _x_region(ctx):
    D = ctx.L["dose"]
    f = float(D["unit_factor"])
    thr = float(D["evidence_gyh"]) / f
    t = ctx.trav
    X, Y = ctx.grid.centers()
    xs = _x_source(ctx)
    near = ((X - xs["pos"][0]) ** 2 + (Y - xs["pos"][1]) ** 2 <= 8.0 ** 2) & ctx.inside
    iy, ix = np.nonzero(near)
    r = np.zeros(ctx.grid.shape)
    if len(iy):
        r[iy, ix] = ctx.rate(np.stack([X[iy, ix], Y[iy, ix]], 1))
    st = t.enter["walk"] | t.enter["high_step"]
    ctx.regions["X"] = {"generous": (r >= thr) & ctx.inside, "stand": (r >= thr * float(D["evidence_margin"])) & st,
                        "crouch": (r >= thr * float(D["evidence_margin"])) & t.enter["crouch_walk"] & ~st, "rate": r}


def store_regions(ctx: Ctx):
    """Write the regions into site.json (run-length masks) and the readable_region the earlier consumers read
    (the margin region: an accepted frame with margin from there)."""
    gi = ctx.grid.info()
    for rec in ctx.site["ips"]:
        R = ctx.regions.get(rec["tag"] if rec["slot"] != "X" else "X")
        if R is None:
            continue
        rec["readable_region"] = {"grid": gi, "semantics": "body",
                                  "window": {"rule": "acceptance_v1.yaml (mission/evidence.py's record) with the ladder "
                                                     "margins; body centre in the cell facing the instrument",
                                             "margin": ctx.L["margin"], "camera": _camera_model(ctx.cfg)},
                                  "runs_stand": runs_from_mask(R["stand"]), "runs_crouch": runs_from_mask(R["crouch"]),
                                  "cells_stand": int(R["stand"].sum()), "cells_crouch": int(R["crouch"].sum())}
        rec["acceptance_region"] = {"grid": gi, "semantics": "camera", "runs_generous": runs_from_mask(R["generous"]),
                                    "cells_generous": int(R["generous"].sum()),
                                    "definition": "generous: camera cells from which some pose of the robot could "
                                                  "give an accepted frame (mission/evidence.py's conditions relaxed: "
                                                  "no in-frame test, camera heights over the robot's range, aim over "
                                                  "the head and body-pitch range, one of the k+1 key points clear)"}


# ============================================================================================ reading place
def _camera_model(cfg):
    H = cfg["ladder"]["head"]
    return {"stand_height_m": nominal_head(cfg, "stand"), "crouch_height_m": nominal_head(cfg, "crouch"),
            "mast_x_m": float(H["mast_x"]), "cam_fwd_m": float(H["cam_fwd"]), "cam_lat_m": dict(H["cam_lat"])}


def reading_place_cells(cfg, grid, feat, mask, item, height=None):
    """BODY cells of `mask` that are a reading place, with the camera where the robot's camera is when
    it stands there facing the instrument: 1.5-3 m from the face within 30 deg of head-on (level gauge: 2.5-3 m
    within 15 deg); a dial above 1.4 m needs a horizontal stand-off of 1.0 m (63 mm: 1.5 m); a 63 mm dial must not
    hang above 1.3 m. feat: an accept.EvGeom."""
    L = cfg["ladder"]
    RP, MT = L["reading_place"], L["mount"]
    typ = item["type"]
    case = int((item.get("params") or {}).get("case_mm", 0))
    z = float(feat.c[2])
    if typ == "gauge" and case == 63 and z > float(MT["gauge_63_max_z"]) + 1e-6:
        return np.zeros(grid.shape, bool)
    if typ == "level":
        d0, d1 = RP["level_d"]
        th = float(RP["level_off_deg"])
    else:
        d0, d1 = RP["d"]
        th = float(RP["off_deg"])
    h = nominal_head(cfg, "stand") if height is None else float(height)
    X, Y = grid.centers()
    iy, ix = np.nonzero(mask)
    out = np.zeros(grid.shape, bool)
    if not len(iy):
        return out
    C = camera_from_body(cfg, np.stack([X[iy, ix], Y[iy, ix]], 1), feat.aim, h, feat.sensor)
    v = C - feat.c[None, :]
    d = np.linalg.norm(v, axis=1)
    n = feat.n if feat.n is not None else -v[0] / max(np.linalg.norm(v[0]), 1e-9)
    off = np.degrees(np.arccos(np.clip((v @ n) / np.maximum(d, 1e-9), -1, 1)))
    ok = (d >= float(d0)) & (d <= float(d1)) & (off <= th)
    if typ == "gauge" and z > float(MT["dial_high_z"]):
        so = float(MT["gauge_63_high_standoff"] if case == 63 else MT["dial_high_standoff"])
        ok &= np.linalg.norm(v[:, :2], axis=1) >= so
    out[iy[ok], ix[ok]] = True
    return out


# ============================================================================================ route
def _obs_pose(ctx, rec, dist_pad, extra_mask=None):
    """The observation state of an item: a reachable BODY cell of its margin region in the slot's modes, preferring
    a reading place, the ideal camera distance, head-on, walk, no tight or slow cells, near the pad. The pose gives
    the body centre, the heading toward the instrument, the camera height and the head tilt from the camera."""
    t = ctx.trav
    g = t.g
    N = g.nx * g.ny
    X, Y = g.centers()
    slot = rec["slot"]
    R = ctx.regions[rec["tag"] if slot != "X" else "X"]
    modes = SLOT_MODES[slot]
    feat = R.get("feat")
    best = None
    pen = {"walk": 0.0, "high_step": 0.35, "crouch_walk": 0.6}
    off_tight = ctx.tight_free(modes)
    for m in modes:
        mask = (R["crouch"] if m == "crouch_walk" else R["stand"]) & t.enter[m]
        mask &= ctx.reach_mode(m, modes)
        if slot == "A":
            mask &= ctx.a_side() & ctx.link3d("walk", base_only=True)
        if extra_mask is not None:
            mask &= extra_mask
        if off_tight is not None and (mask & off_tight[m]).any():
            mask &= off_tight[m]               # a pose reachable without any tight passage
        if not mask.any():
            continue
        iy, ix = np.nonzero(mask)
        nodes = MODES.index(m) * N + iy * g.nx + ix
        tp = dist_pad[nodes]
        fin = np.isfinite(tp)
        if not fin.any():
            continue
        iy, ix, nodes, tp = iy[fin], ix[fin], nodes[fin], tp[fin]
        hh = nominal_head(ctx.cfg, "crouch" if m == "crouch_walk" else "stand")
        B = np.stack([X[iy, ix], Y[iy, ix]], 1)
        if feat is None:
            c = np.asarray(rec["pose"]["pos"][:2], float)
            q = np.hypot(B[:, 0] - c[0], B[:, 1] - c[1]) / 0.3
            if slot == "X":
                q = tp / 300.0
            Cm = np.c_[B, np.full(len(B), hh)]
        else:
            Cm = camera_from_body(ctx.cfg, B, feat.aim, hh, feat.sensor)
            v = Cm - feat.c[None, :]
            d = np.linalg.norm(v, axis=1)
            n = feat.n if feat.n is not None else np.array([1.0, 0.0, 0.0])
            ang = np.degrees(np.arccos(np.clip((v @ n) / np.maximum(d, 1e-9), -1, 1)))
            di = 2.75 if rec["type"] == "level" else 2.0
            sub = np.zeros(g.shape, bool)
            sub[iy, ix] = True
            rp = reading_place_cells(ctx.cfg, g, feat, sub, rec, hh)[iy, ix]
            q = np.abs(d - di) / 1.5 + 0.8 * ang / 45.0 + 1.5 * (~rp)
        q = q + pen[m] + 0.3 * t.tight[m][iy, ix] + 0.15 * t.slow[iy, ix] + tp / 900.0
        k = int(np.argmin(q))
        if rec["type"] == "valve_lever" and feat is not None:
            # no red object behind the valve in the view from its observation pose (the
            # readers' test case: a red fire door behind a lever merges with its red grip): the best cell
            # without visible red geometry within RED_BEHIND_DEG of the line of sight behind the valve
            for kk in np.argsort(q, kind="stable")[:200]:
                if not red_behind(ctx, Cm[kk], feat.aim, rec):
                    k = int(kk)
                    break
        if best is None or q[k] < best[0]:
            best = (float(q[k]), int(nodes[k]), m, float(B[k, 0]), float(B[k, 1]), hh, Cm[k].copy())
    if best is None:
        return None, None
    _, node, m, x, y, hh, cm = best
    P = feat.aim if feat is not None else np.array(list(rec["pose"]["pos"][:2]) + [0.0])
    yaw = math.atan2(P[1] - y, P[0] - x)
    horiz = math.hypot(P[0] - cm[0], P[1] - cm[1])
    tilt = math.atan2(P[2] - cm[2], max(horiz, 1e-6)) if feat is not None else -0.6
    return node, {"pos": [x, y], "yaw": yaw, "mode": m, "head_height": hh, "head_tilt": tilt,
                  "distance": float(np.linalg.norm(P - cm)) if feat is not None else math.hypot(horiz, P[2] - hh),
                  "camera": [round(float(v), 4) for v in cm], "semantics": "body"}


def reference_route(ctx: Ctx):
    """Observation poses, the unlisted source's strength inside its band so that the route stays inside the dose
    limit, and the reference route (TSP over the poses)."""
    from scipy.sparse.csgraph import dijkstra
    from .route import plan
    cfg = ctx.cfg
    D = ctx.L["dose"]
    f = float(D["unit_factor"])
    rating = float(D["rating_gy"]) / f
    lim = float(D["route_fraction_max"]) * rating
    t = ctx.trav
    G = t.build_graph(MODES)
    dist_pad = dijkstra(G, directed=True, indices=[ctx.pad])[0]
    xs = [s for s in ctx.site["sources"] if not s["documented"]][0]
    S_lo, S_hi = [float(v) / f for v in D["source_gyh_at_1m"]]
    dw = cfg["ips"]["dwell_s"]
    poses = {}
    for rec in ctx.site["ips"]:
        if rec["slot"] == "X":
            continue
        node, pose = _obs_pose(ctx, rec, dist_pad)
        if node is None:
            from .ladder import LadderReject
            raise LadderReject("observable_%s" % rec["slot"], "no observation pose for %s" % rec["tag"])
        poses[rec["tag"]] = (node, pose)
    S = float(xs["strength"])
    route = None
    for _ in range(4):
        xs["strength"] = round(S, 2)
        _x_region(ctx)
        xr = ctx.by_slot["X"]
        dose_cost = _dose_graph(ctx, G)
        dd = dijkstra(dose_cost, directed=True, indices=[ctx.pad])[0]
        node, pose = _obs_pose(ctx, xr, dd)
        if node is None:
            from .ladder import LadderReject
            raise LadderReject("X_evidence", "no reachable evidence place at %.0f uSv/h @ 1 m" % S)
        poses["X"] = (node, pose)
        obs = []
        for rec in ctx.site["ips"]:
            tag = rec["tag"] if rec["slot"] != "X" else "X"
            n_, p_ = poses[tag]
            rec["observation_pose"] = p_
            dwell = float(D["mark_dwell_s"]) if rec["slot"] == "M" else (float(D["source_dwell_s"]) if
                                                                         rec["slot"] == "X" else float(dw[rec["type"]]))
            obs.append((rec["tag"], n_, dwell))
        route = plan(t, G, ctx.pad, obs, cfg, dose_parts={
            "other": lambda xy: ctx.rate(xy, exclude=[xs["id"]]),
            "x_unit": lambda xy: ctx.rate(xy, only=[xs["id"]]) / max(float(xs["strength"]), 1e-9)})
        dwell_other, dwell_x = 0.0, 0.0
        for (tag, n_, dwl), rec in zip(obs, ctx.site["ips"]):
            p_ = np.array([rec["observation_pose"]["pos"]])
            dwell_other += float(ctx.rate(p_, exclude=[xs["id"]])[0]) * dwl / 3600.0
            dwell_x += float(ctx.rate(p_, only=[xs["id"]])[0]) / max(float(xs["strength"]), 1e-9) * dwl / 3600.0
        D0 = route["dose_parts_uSv"]["other"] + dwell_other
        D1 = route["dose_parts_uSv"]["x_unit"] + dwell_x
        S_dose = (0.95 * lim - D0) / max(D1, 1e-12)
        if S <= S_dose + 1e-6:
            break
        if S_dose < S_lo:
            S = S_lo
            xs["strength"] = round(S, 2)
            break
        S = max(S_lo, S_dose)
    xs["strength"] = round(S, 2)
    _x_region(ctx)
    D0 = route["dose_parts_uSv"]["other"] + dwell_other
    D1 = route["dose_parts_uSv"]["x_unit"] + dwell_x
    tot = D0 + S * D1
    route["dose_travel_uSv"] = route["dose_parts_uSv"]["other"] + S * route["dose_parts_uSv"]["x_unit"]
    route["dose_dwell_uSv"] = dwell_other + S * dwell_x
    route["dose_total_uSv"] = tot
    route["dose_fraction_of_rating"] = tot / rating
    route["dose_total_gy"] = tot * f
    # energy at the mode powers (as in the first layout)
    M = cfg["modes"]
    en = 0.0
    for leg in route["legs"]:
        for s in leg["segments"]:
            en += float(M[s["mode"]]["power_W"]) * s["time_s"]
    route["energy_Wh"] = (en + float(M["idle_power_W"]) * route["dwell_s"]) / 3600.0
    route["modes_used"] = sorted({s["mode"] for leg in route["legs"] for s in leg["segments"]})
    rooms = []
    for leg in route["legs"]:
        for s in leg["segments"]:
            for x, y in s["waypoints"]:
                for r in ctx.site["rooms"]:
                    if r["rect"][0] <= x <= r["rect"][2] and r["rect"][1] <= y <= r["rect"][3]:
                        if not rooms or rooms[-1] != r["id"]:
                            rooms.append(r["id"])
    route["rooms_visited"] = sorted(set(rooms))
    route["n_rooms_visited"] = len(route["rooms_visited"])
    route["crossed_elements"] = _crossed(ctx, route)
    ctx._route_cells = route.pop("_cells", [])
    route.pop("_cost_matrix", None)
    for s in ctx.site["sources"]:
        if s["id"] == xs["id"]:
            s["strength"] = xs["strength"]
    ctx.site["ladder"]["dose"]["x_strength"] = xs["strength"]
    ctx.site["ladder"]["dose"]["x_strength_gyh_at_1m"] = xs["strength"] * f
    return route


def _dose_graph(ctx, G):
    """The time graph re-weighted by the dose rate at the arrival cell (uSv per edge)."""
    t = ctx.trav
    X, Y = t.g.centers()
    xs = [s for s in ctx.site["sources"] if not s["documented"]][0]
    N = t.g.nx * t.g.ny
    rate = np.full(N, float(ctx.site["radiation"]["background_uSv_h"]))
    near = ((X - xs["pos"][0]) ** 2 + (Y - xs["pos"][1]) ** 2 <= 14.0 ** 2) & ctx.inside
    iy, ix = np.nonzero(near)
    rate[iy * t.g.nx + ix] = ctx.rate(np.stack([X[iy, ix], Y[iy, ix]], 1))
    rn = np.tile(rate, 3)
    G2 = G.copy().tocsr()
    G2.data = G2.data * rn[G2.indices] / 3600.0 + 1e-9
    return G2


def _crossed(ctx, route):
    """The traversal elements the route's cells lie on, with the modes used there."""
    t = ctx.trav
    out = {}
    for xy, modes, _T in route.get("_cells", []):
        for (x, y), m in zip(xy, modes):
            iy, ix = t.g.cell(x, y)
            e = int(t.elem[iy, ix])
            if e < 0:
                continue
            eid = t.elem_ids[e]
            el = ctx.elements.get(eid)
            if el is None or el["class"] in ("free",):
                continue
            rec = out.setdefault(eid, {"kind": el["kind"], "class": el["class"], "param": el["param"], "modes": set()})
            rec["modes"].add(m)
    return {k: dict(v, modes=sorted(v["modes"])) for k, v in sorted(out.items())}


# ============================================================================================ rows
def _row(ok, value=None, detail="", required=True):
    return {"ok": bool(ok), "value": value, "detail": detail, "required": bool(required)}


def run_rows(ctx: Ctx):
    rows = {}
    cfg = ctx.cfg
    area_min = float(ctx.L["margin"]["min_area_m2"])
    a2 = ctx.res ** 2
    X, Y = ctx.grid.centers()
    site = ctx.site
    # ---------------------------------------------------------------- the site-wide rule
    A_side = ctx.a_side()
    xs_ = float(site["ladder"]["corridor_sill_x"])
    al = ctx.rooms["A1"]["rect"]
    c1 = ctx.rooms["C1"]["rect"]
    allowed = ((X >= al[0]) & (X <= al[2]) & (Y >= al[1]) & (Y <= al[3])) | \
        ((X >= c1[0]) & (X <= xs_ + 0.1) & (Y >= c1[1]) & (Y <= c1[3]))
    walk_only = ctx.reach_cells(("walk",))
    # boundary tolerance 0.15 m: the element footprint that is blocked is a little wider than the geometry
    near_A = ndimage.distance_transform_edt(~A_side) * ctx.res <= 0.15
    near_allowed = ndimage.distance_transform_edt(~allowed) * ctx.res <= 0.15
    l1 = A_side & ~near_allowed
    l2 = walk_only & ~near_A
    leak = int(l1.sum()) + int(l2.sum())
    ex = []
    for mk in (l1, l2):
        if mk.any():
            iy, ix = np.nonzero(mk)
            ex.append([round(float(X[iy[0], ix[0]]), 2), round(float(Y[iy[0], ix[0]]), 2)])
    rows["site_rule"] = _row(leak <= 4, {"a_side_cells": int(A_side.sum()), "outside_cells": leak,
                                         "a_side_outside": int(l1.sum()), "walk_beyond": int(l2.sum()), "ex": ex},
                             "airlock side of the corridor sill = airlock + slot-A stretch; walking alone stays there")
    dA = ctx.dilate(A_side)
    seen = []
    for slot in INSTR[1:]:
        rec = ctx.by_slot[slot]
        if (ctx.regions[rec["tag"]]["generous"] & dA).any():
            seen.append(slot)
    rows["A_side_only_A"] = _row(not seen, seen, "no other item has an accepted frame from the airlock side")
    wo = []
    dW = ctx.dilate(walk_only)
    for slot in INSTR:
        rec = ctx.by_slot[slot]
        if (ctx.regions[rec["tag"]]["generous"] & dW).any():
            wo.append(slot)
    rows["walk_only_items"] = _row(wo in ([], ["A"]), wo, "items with an accepted frame reachable by walking alone "
                                                         "(only A may be)")
    # ---------------------------------------------------------------- cuts and observability
    obs_area = {}
    for slot in INSTR:
        rec = ctx.by_slot[slot]
        R = ctx.regions[rec["tag"]]
        st, cr = ctx.stand_crouch(SLOT_MODES[slot])
        m = (R["stand"] & st) | (R["crouch"] & cr)
        if slot == "A":
            m &= A_side & ctx.link3d("walk", base_only=True)
        area = float(m.sum()) * a2
        obs_area[slot] = round(area, 3)
        rp = reading_place_cells(cfg, ctx.grid, R["feat"], m, rec)
        rows["%s_observable" % slot] = _row(area >= area_min and rp.any(),
                                            {"area_m2": round(area, 3), "reading_place_cells": int(rp.sum())},
                                            "reachable margin region >= %.2f m2 with a reading place%s" % (
                                                area_min, " (walk, airlock side, base link)" if slot == "A" else ""))
        if slot in CUT_SLOTS:
            rw = ctx.reach_cells(MODES, ctx.cut(slot), slot)
            hit = R["generous"] & ctx.dilate(rw)
            n = int(hit.sum())
            ex = None
            if n:
                iy, ix = np.nonzero(hit)
                ex = [round(float(X[iy[0], ix[0]]), 2), round(float(Y[iy[0], ix[0]]), 2)]
            rows["%s_cut" % slot] = _row(n == 0, {"cells": n, "example": ex},
                                         "no accepted frame (generous region) from any pose reachable without "
                                         "crossing %s" % {"B-step": "the room's high-step sills",
                                                          "B-kerb": "the bund kerb or its ramp",
                                                          "B-low": "a needs_crouch element",
                                                          "C-sealed": "the undrawn as-built opening",
                                                          "C-annex": "the undrawn annex door"}[slot])
    rows["observable_areas"] = _row(True, obs_area, "", required=False)
    # ---------------------------------------------------------------- lighting
    dark = [s for s in INSTR if ctx.rooms[ctx.by_slot[s]["room"]]["light"] != "bright"]
    rows["lighting"] = _row(not dark, dark, "every instrument's room (its reading place) lit to bright")
    # ---------------------------------------------------------------- types, decoys, deviations
    rows.update(_types_rows(ctx))
    rows.update(_deviation_rows(ctx))
    # ---------------------------------------------------------------- drawing follower
    rows.update(_drawing_rows(ctx))
    # ---------------------------------------------------------------- findability and search budget
    rows.update(_find_rows(ctx))
    # ---------------------------------------------------------------- documents' facts
    rows.update(_fact_rows(ctx))
    # ---------------------------------------------------------------- dose
    rows.update(_dose_rows(ctx))
    # ---------------------------------------------------------------- elements on the route, access rules
    rows.update(_element_rows(ctx))
    rows.update(_access_rows(ctx))
    # ---------------------------------------------------------------- the rows added after a review
    rows.update(_fixpass_rows(ctx))
    # ---------------------------------------------------------------- version 2
    if ctx.v2:
        rows.update(_v2_rows(ctx))
    if ctx.acc.plate_rule is not None:
        # version 2 (text plates): plate_readable, tags; plates_clear covers every listed instrument's text plate
        # besides version 1's level scale plates and lever tag plates
        pr = _plate_rows(ctx)
        rows["plate_readable"] = pr["plate_readable"]
        rows["tags"] = pr["tags"]
        v1pc = rows.get("plates_clear")
        rows["plates_clear"] = _row((v1pc is None or v1pc["ok"]) and pr["plates_clear"]["ok"],
                                    {"scale_and_lever_plates": v1pc and v1pc["value"],
                                     "text_plates": pr["plates_clear"]["value"]},
                                    "nothing in front of a level gauge's scale plate, a lever valve's tag plate or any "
                                    "listed instrument's text plate from the margin cells")
    # ---------------------------------------------------------------- leak
    errs = leak_errors(site, ctx.prior)
    if ctx.v2:
        errs += ["missing sheet: %s" % e for e in sheet_nothing_drawn(site, ctx.prior)]
    rows["leak"] = _row(not errs, errs[:6], "prior.json carries nothing that only the ground truth knows")
    # guarantees in the first layout's format
    need_c, need_h = [], []
    full = ctx.reach_nodes(MODES)
    noc = ctx.reach_nodes(("walk", "high_step"))
    noh = ctx.reach_nodes(("walk", "crouch_walk"))
    N = ctx.grid.nx * ctx.grid.ny
    for rec in site["ips"]:
        R = ctx.regions.get(rec["tag"] if rec["slot"] != "X" else "X")
        if R is None:
            continue
        nodes = np.concatenate([np.flatnonzero(R["stand"] & ctx.trav.enter["walk"]),
                                N + np.flatnonzero(R["stand"] & ctx.trav.enter["high_step"]),
                                2 * N + np.flatnonzero(R["crouch"] & ctx.trav.enter["crouch_walk"])])
        if not len(nodes) or not full[nodes].any():
            continue
        if not noc[nodes].any():
            need_c.append(rec["tag"])
        if not noh[nodes].any():
            need_h.append(rec["tag"])
        rec["reach"] = {"all": bool(full[nodes].any()), "without_crouch": bool(noc[nodes].any()),
                        "without_high_step": bool(noh[nodes].any()), "readable_cells": int(len(nodes))}
    site["guarantees"] = {"ok": True, "needs_crouch": need_c, "needs_high_step": need_h,
                          "behind_discrepancy": [ctx.by_slot[s]["tag"] for s in INSTR if s.startswith("C")],
                          "blind_fail": rows["drawing_fails_C"]["value"], "unreachable": []}
    return rows


def _types_rows(ctx):
    rows = {}
    want = sorted(ctx.L["types"])
    have = sorted(ctx.by_slot[s]["ladder_type"] for s in INSTR)
    rows["types"] = _row(have == want, dict(zip(INSTR, [ctx.by_slot[s]["ladder_type"] for s in INSTR])),
                         "the eight instrument types once each")
    nd = len(ctx.site["decoys"])
    lo, hi = ctx.L["decoys"]
    mv = ctx.by_slot["C-moved"]
    dec = [d for d in ctx.site["decoys"] if d.get("at_drawn_place_of") == mv["tag"]]
    okd = bool(dec) and dec[0]["type"] == mv["type"] and \
        math.hypot(dec[0]["pose"]["pos"][0] - mv["documented"]["pos"][0],
                   dec[0]["pose"]["pos"][1] - mv["documented"]["pos"][1]) < 0.05
    rows["decoys"] = _row(lo <= nd <= hi and okd, {"decoys": nd, "c_moved_decoy": dec[0]["tag"] if dec else None},
                          "6-12 decoys with their own tags; a decoy of the same type at the C-moved item's drawn place")
    return rows


def _deviation_rows(ctx):
    site = ctx.site
    kinds = [d["kind"] for d in site["deviations"]]
    need = ["door_sealed", "instrument_moved", "instrument_relocated", "annex_added", "zoning_outdated", "repeater_dead",
            "opening_as_built"]
    opt = [k for k in kinds if k in ("opening_added", "partition_added")]
    if ctx.v2:
        # version 2: exactly one sheet missing, never the airlock's
        from .sheet import place_of_sheet
        ms = site.get("missing_sheet") or {}
        okms = bool(ms) and place_of_sheet(site["building"]["rect"], int(ms.get("sheet", 0))) in \
            ("along", "across", "diagonal") and ms.get("place") == place_of_sheet(site["building"]["rect"],
                                                                                  int(ms["sheet"]))
    else:
        okms = not site.get("missing_sheet")
    ok = all(kinds.count(k) == 1 for k in need) and len(opt) <= 2 and okms
    sd = ctx.doors[site["ladder"]["sealed_door"]]
    ok &= sd["state"] == "closed"
    undrawn = [r for r in site["rooms"] if not r["on_drawing"]]
    ok &= len(undrawn) == 1 and undrawn[0]["type"] == "annex"
    dead = [r for r in site["link"]["repeaters"] if not r["working"]]
    ok &= len(dead) == 1
    gm = site["grid_markers"]
    n_doc = len([m for m in gm if m["documented"]])
    n_pres = len([m for m in gm if m["present"]])
    want = int(round(float(ctx.L["grid_markers_present"]) * n_doc))
    an = undrawn[0]["rect"] if undrawn else None
    in_undrawn = [m["id"] for m in gm if an and an[0] <= m["pos"][0] <= an[2] and an[1] <= m["pos"][1] <= an[3]]
    on_missing = []
    if ctx.miss is not None:
        # version 2 (row grid_markers): none listed or present on the missing part (a plate within 0.2 m of a
        # corridor cut counts as on it)
        on_missing = [m["id"] for m in gm if (m["documented"] or m["present"]) and
                      (ctx.miss.in_cut(m["pos"][0], m["pos"][1], margin=-0.2) or
                       ctx.miss.in_missing_room(m["pos"][0], m["pos"][1]))]
    okm = n_pres == want and not in_undrawn and not on_missing
    return {"deviations": _row(ok, kinds, "exactly one sealed door, one moved, one relocated instrument, the undrawn "
                                          "annex, the unlisted source, one dead repeater; 0-2 optional; %s" %
                               ("exactly one sheet missing, not the airlock's" if ctx.v2 else "no missing sheet")),
            "grid_markers": _row(okm, {"documented": n_doc, "present": n_pres, "in_undrawn": in_undrawn,
                                       "on_missing_part": on_missing},
                                 "75 %% of the documented grid markers exist, none in undrawn space%s" %
                                 (" and none on the missing part" if ctx.v2 else ""))}


# ---------------------------------------------------------------- the drawing follower
def drawn_map(ctx):
    """Enterable cells on the drawing (drawn walls with the drawn door openings, drawn equipment; the footprint)."""
    g = ctx.grid
    pr = ctx.prior
    solid = np.zeros(g.shape, np.uint8)
    inside = np.zeros(g.shape, np.uint8)
    for r in pr["rooms"]:
        x0, y0, x1, y1 = r["rect"]
        g.fill_poly(inside, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 1)
    for w in pr["walls"]:
        (ax, ay), (bx, by) = w["a"], w["b"]
        t = w["thickness"]
        off = w.get("offset", 0.0)
        if abs(ax - bx) < 1e-9:
            r = [ax + off - t / 2, min(ay, by) - 0.12, ax + off + t / 2, max(ay, by) + 0.12]
        else:
            r = [min(ax, bx) - 0.12, ay + off - t / 2, max(ax, bx) + 0.12, ay + off + t / 2]
        g.fill_poly(solid, [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], 1)
    for d in pr["doors"]:
        if d["kind"] == "fire_door":
            continue
        x, y = d["pos"]
        wd = d["width"]
        if d["axis"] == "x":
            r = [x - 0.45, y - wd / 2, x + 0.45, y + wd / 2]
        else:
            r = [x - wd / 2, y - 0.45, x + wd / 2, y + 0.45]
        g.fill_poly(solid, [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], 0)
    for e in pr["equipment"]:
        if e.get("kind") == "deployment_pad":
            continue
        if e["shape"] == "circle":
            cx, cy = e["center"]
            rr = e["radius"]
            poly = [[cx + rr * math.cos(a), cy + rr * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16,
                                                                                      endpoint=False)]
        else:
            poly = e["outline"]
        g.fill_poly(solid, poly, 1)
    s = solid.astype(bool) | ~inside.astype(bool)
    ctx.drawn_solid = s
    d = ndimage.distance_transform_edt(~s)
    en = (~s) & ((d - 0.5) * g.res >= ctx.trav.th["r_in"])
    lab, _ = ndimage.label(en, ndimage.generate_binary_structure(2, 2))
    iy, ix = g.cell(0.0, 0.0)
    k = lab[iy, ix]
    reach = (lab == k) if k > 0 else np.zeros(g.shape, bool)
    return en, reach


def hard_map(ctx):
    """Structure that stops any robot in reality: walls (with the real openings of passable doors), partitions,
    closed and sealed doors."""
    g = ctx.grid
    m = np.zeros(g.shape, np.uint8)
    ops = {}
    for d in ctx.site["doors"]:
        if d.get("kind") == "fire" or not d.get("wall"):
            continue
        ops.setdefault(d["wall"], []).append(d)
    for w in ctx.site["walls"]:
        cur = w["lo"] - 0.12
        segs = []
        for d in sorted(ops.get(w["id"], []), key=lambda d: d["center"]):
            a, b = d["center"] - d["width"] / 2, d["center"] + d["width"] / 2
            if d["state"] == "closed" or d["kind"] in ("stair", "airlock_outer"):
                continue
            segs.append((cur, a))
            cur = b
        segs.append((cur, w["hi"] + 0.12))
        for a, b in segs:
            if b - a < 1e-3:
                continue
            if w["axis"] == "x":
                r = [w["coord"] - w["t_neg"], a, w["coord"] + w["t_pos"], b]
            else:
                r = [a, w["coord"] - w["t_neg"], b, w["coord"] + w["t_pos"]]
            g.fill_poly(m, [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], 1)
    for p in ctx.site.get("partitions", []):
        legr = ctx.rooms[p["leg"]]["rect"]
        x = p["x"]
        r = [x - p["thickness"] / 2, legr[1], x + p["thickness"] / 2, legr[3]]
        g.fill_poly(m, [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], 1)
    return m.astype(bool)


def _drawn_reading_cells(ctx, en, reach, loc, typ, n_max=None):
    """Reading places in front of a drawn location, as the drawing shows them, best first: the design's window
    (1.5-3 m within 30 deg; the level gauge 2.5-3 m within 15 deg), then the acceptance file's window with the
    margin (min_dist to 3 m, max_off_axis less 5 deg), then the whole acceptance window (min_dist to max_dist_all,
    max_off_axis). A cell counts only when the drawn location is in view on the drawing (the segment to 0.25 m short
    of it crosses no drawn wall or drawn equipment). Returns [(iy, ix, C, p)]."""
    RP = ctx.L["reading_place"]
    g = ctx.grid
    X, Y = g.centers()
    p = np.array([loc["pos"][0], loc["pos"][1], float(loc.get("height") or 1.0)])
    ph = math.radians(float(loc.get("facing_deg", 0.0)))
    n = np.array([math.cos(ph), math.sin(ph), 0.0])
    rule = ctx.acc.type_rule(typ)
    dmin = max(float(rule.get("min_dist", 0.4)), 0.8)
    windows = [((RP["level_d"][0], RP["level_d"][1], RP["level_off_deg"]) if typ == "level" else
                (RP["d"][0], RP["d"][1], RP["off_deg"])),
               (dmin, 3.0, float(rule["max_off_axis"]) - float(ctx.L["margin"]["angle_deg"])),
               (dmin, ctx.acc.max_dist, float(rule["max_off_axis"]))]
    h = nominal_head(ctx.cfg, "stand")
    sel = en & reach & ((X - p[0]) ** 2 + (Y - p[1]) ** 2 <= (ctx.acc.max_dist + 0.1) ** 2)
    iy, ix = np.nonzero(sel)
    if not len(iy):
        return []
    C = np.stack([X[iy, ix], Y[iy, ix], np.full(len(iy), h)], 1)
    v = C - p[None, :]
    d = np.linalg.norm(v, axis=1)
    off = np.degrees(np.arccos(np.clip((v @ n) / np.maximum(d, 1e-9), -1, 1)))
    seen = np.ones(len(iy), bool)
    solid = getattr(ctx, "drawn_solid", None)
    if solid is not None:
        t = np.linspace(0.0, 1.0, 64)[None, :]
        stop = np.clip(1.0 - 0.25 / np.maximum(d, 1e-9), 0.0, 1.0)[:, None]
        sx = C[:, 0:1] + (p[0] - C[:, 0:1]) * t * stop
        sy = C[:, 1:2] + (p[1] - C[:, 1:2]) * t * stop
        jx = np.clip(np.floor((sx - g.x0) / g.res).astype(int), 0, g.nx - 1)
        jy = np.clip(np.floor((sy - g.y0) / g.res).astype(int), 0, g.ny - 1)
        seen = ~solid[jy, jx].any(axis=1)
    di = 2.75 if typ == "level" else 2.0
    out, used = [], np.zeros(len(iy), bool)
    for d0, d1, th in windows:
        ok = seen & ~used & (d >= float(d0)) & (d <= float(d1)) & (off <= float(th))
        q = np.where(ok, np.abs(d - di) + off / 30.0, np.inf)
        for k in np.argsort(q)[:int(ok.sum())]:
            out.append((iy[k], ix[k], C[k], p))
        used |= ok
    return out[:n_max] if n_max else out


def _drawn_reading_cell(ctx, en, reach, loc, typ):
    c = _drawn_reading_cells(ctx, en, reach, loc, typ, 1)
    return c[0] if c else None


def _drawn_tree(en, start):
    """Shortest-path tree (8-connected, metric) on the drawing's enterable cells from `start`: predecessors."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    ny, nx = en.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        ys0, ys1 = max(0, -dy), ny - max(0, dy)
        xs0, xs1 = max(0, -dx), nx - max(0, dx)
        ok = en[ys0:ys1, xs0:xs1] & en[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        rows.append(idx[ys0:ys1, xs0:xs1][ok])
        cols.append(idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][ok])
        vals.append(np.full(int(ok.sum()), math.hypot(dx, dy)))
    G = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx, ny * nx))
    s = start[0] * nx + start[1]
    _, pred = dijkstra(G, directed=True, indices=[s], return_predecessors=True)
    return s, pred[0]


def _drawn_path(tree, goal, nx):
    s, pred = tree
    g = goal[0] * nx + goal[1]
    if g != s and pred[g] < 0:
        return None
    path = []
    c = g
    while c != s and c >= 0:
        path.append(c)
        c = int(pred[c])
    return path[::-1]


def _in_frame(cam, C, p, q):
    """Is point q inside the frame of camera `cam` at C aimed at p (upright)?"""
    f = p - C
    f = f / max(np.linalg.norm(f), 1e-9)
    right = np.cross(f, [0, 0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, f)
    v = q - C
    z = float(v @ f)
    return z > 0 and abs(float(v @ right) / z) <= cam["tan_h"] and abs(float(v @ up) / z) <= cam["tan_v"]


def follow_drawing(ctx, rec, en, reach, hard, tree=None, max_replans=6, n_try=4, spacing=0.5):
    """Follow the drawing alone to an item: plan on the drawing to a reading place in front of the drawn location,
    replay the path against the real structure; where real structure (a wall, a partition, a closed or sealed door)
    stops it, add that structure to its own map and plan again on the drawing (it never learns an undrawn opening);
    at the reading place aim at the drawn location. It reads when the reading place is in the true instrument's
    generous region and the instrument (its body or its readable feature) is inside the aimed frame; it tries the
    n_try best reading places it can get to, at least `spacing` apart. Returns (success, why)."""
    loc = None
    for it in ctx.prior["round_sheet"]["items"]:
        if it["tag"] == rec["tag"]:
            loc = it["location"]
    if loc is None or not loc.get("recorded", loc.get("documented", False)) or not loc.get("pos"):
        return False, "no recorded location"
    cands = _drawn_reading_cells(ctx, en, reach, loc, rec["type"])
    if not cands:
        return False, "no reading place on the drawing"
    g = ctx.grid
    pad = g.cell(0.0, 0.0)
    if tree is None:
        tree = _drawn_tree(en, pad)
    R = ctx.regions.get(rec["tag"])
    cam = camera(ctx.cfg, ctx.acc.type_rule(rec["type"])["camera"])
    body = np.asarray(rec["pose"]["pos"], float)
    hard_flat = hard.ravel()
    en_k = en
    replans = 0
    why = "no path on the drawing"
    while True:
        tried, blocked, at = 0, None, []
        for iy, ix, C, p in cands:
            if not en_k[iy, ix] or any(math.hypot(C[0] - a[0], C[1] - a[1]) < spacing for a in at):
                continue
            path = _drawn_path(tree, (iy, ix), g.nx)
            if path is None:
                continue
            hit = np.asarray(path)[hard_flat[np.asarray(path)]]
            if len(hit):
                blocked = hit
                break
            tried += 1
            at.append(C)
            if R is None or not R["generous"][iy, ix]:
                why = "no accepted frame of the true instrument from the drawn reading place"
            elif not (_in_frame(cam, C, p, body) or _in_frame(cam, C, p, R["feat"].aim)):
                why = "the true instrument is not where the drawing says"
            else:
                return True, "read" + (" (after %d re-plans)" % replans if replans else "")
            if tried >= n_try:
                return False, why
        if blocked is None:
            return False, why if tried else ("blocked on the way, no other drawn way" if replans else
                                             "no path on the drawing")
        if replans >= max_replans:
            return False, "blocked on the way (%d re-plans)" % replans
        # the robot meets the structure: everything hard within 0.8 m of where the path hit it joins its map
        hy, hx = np.divmod(blocked, g.nx)
        near = np.zeros(g.shape, bool)
        near[hy, hx] = True
        near = ndimage.binary_dilation(near, iterations=int(round(0.8 / g.res))) & hard
        dist = ndimage.distance_transform_edt(~near)
        en_k = en_k & ((dist - 0.5) * g.res >= ctx.trav.th["r_in"])
        tree = _drawn_tree(en_k, pad)
        replans += 1


def _drawing_rows(ctx):
    en, reach = drawn_map(ctx)
    hard = hard_map(ctx)
    tree = _drawn_tree(en, ctx.grid.cell(0.0, 0.0))
    res = {}
    for slot in INSTR:
        ok, why = follow_drawing(ctx, ctx.by_slot[slot], en, reach, hard, tree)
        res[slot] = {"success": ok, "why": why}
    c_fail = [s for s in INSTR if s.startswith("C") and not res[s]["success"]]
    # version 2: A and each B item whose room is drawn are read; every item on the missing sheet fails
    on = [s for s in INSTR if ctx.on_sheet(ctx.by_slot[s])]
    ab = [s for s in INSTR if not s.startswith("C") and s not in on]
    ab_ok = [s for s in ab if res[s]["success"]]
    rows = {"drawing_fails_C": _row(len(c_fail) == 4, [ctx.by_slot[s]["tag"] for s in c_fail],
                                    "following the drawing alone fails for all four C slots: %s" %
                                    {s: res[s]["why"] for s in INSTR if s.startswith("C")}),
            "drawing_reads_A_B": _row(len(ab_ok) == len(ab), {s: res[s]["why"] for s in ab},
                                      "the same drawing follower (with legs) reads A and the B items at their drawn "
                                      "places%s" % (" (those whose room is drawn)" if ctx.v2 else ""))}
    if ctx.v2:
        bad = [s for s in on if res[s]["success"]]
        rows["drawing_fails_missing"] = _row(not bad, {"on_sheet": {s: res[s]["why"] for s in on}, "read": bad},
                                             "the drawing follower fails every item whose true place is on the "
                                             "missing sheet")
    return rows


# ---------------------------------------------------------------- findability and search budget
def _tag_geom(ctx, rec):
    """Marker centre and normal of an instrument's tag plate in the compiled model (bricks: site <body>/tag/marker);
    version 2: the text plate's frame (accept.plate_compiled: the plate body <body>/tag)."""
    if ctx.acc.plate_rule is not None:
        from .accept import plate_compiled
        return plate_compiled(ctx.model, ctx.data, rec["body"])
    sid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, "%s/tag/marker" % rec["body"])
    if sid >= 0:
        p = ctx.data.site_xpos[sid].copy()
        R = ctx.data.site_xmat[sid].reshape(3, 3)
        return {"marker": p, "normal": R[:, 2].copy(), "side": 0.060}
    return ctx.regions[rec["tag"]]["feat"].tag


def viewpoints(ctx, en_draw, reach_draw):
    """Doorway and corridor points that the drawing shows reachable, and doorway points of the openings that a lidar
    scan shows (undrawn openings and doors); all really reachable. (N, 3) at the standing head height each."""
    g = ctx.grid
    X, Y = g.centers()
    h = nominal_head(ctx.cfg, "stand")
    real = ctx.reach_cells(MODES)
    pts_d, pts_l = [], []
    for d in ctx.site["doors"]:
        if d.get("kind") in ("fire", "stair", "airlock_outer") or not d.get("wall") or d["state"] == "closed":
            continue
        w = ctx.walls[d["wall"]]
        for s in (-1, 1):
            off = (w["t_pos"] if s > 0 else w["t_neg"]) + 0.45
            for a in (-0.25, 0.0, 0.25):
                c = d["center"] + a * d["width"]
                x, y = (w["coord"] + s * off, c) if w["axis"] == "x" else (c, w["coord"] + s * off)
                iy, ix = g.cell(x, y)
                if not (0 <= iy < g.ny and 0 <= ix < g.nx) or not real[iy, ix]:
                    continue
                if d["on_drawing"] and reach_draw[iy, ix]:
                    pts_d.append((x, y, h))
                elif not d["on_drawing"]:
                    pts_l.append((x, y, h))
    corr = np.zeros(g.shape, bool)
    for r in ctx.prior["rooms"]:
        if r["type"] == "corridor":
            x0, y0, x1, y1 = r["rect"]
            corr |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    sub = np.zeros(g.shape, bool)
    sub[::6, ::6] = True
    cm = corr & reach_draw & real & sub
    iy, ix = np.nonzero(cm)
    pts_c = [(float(X[a, b]), float(Y[a, b]), h) for a, b in zip(iy, ix)]
    return np.array(pts_d + pts_c, float).reshape(-1, 3), np.array(pts_l, float).reshape(-1, 3)


LIDAR_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)
# since 2026-11-16: the geom groups the cell's planar lidar sees (mission/world.py builds it with
# blidar.Lidar(..., geomgroup=(1, 1, 1, 0, 0, 0)): groups 0-2, the walls, structure, equipment and the dressing's
# visuals; not group 3, the tagged floor elements, sills, hatch lintels and rack beams the manual says the scan does not
# show). Before that date it used groups 0, 1 and 3 here (a development test compares this with the cell's own call).


def _annex_lidar(ctx, reach_draw):
    """Cells of the annex host's floor (drawn, reachable) from which the annex door spans >= lidar_beams beams of
    the planar lidar (the cell's lidar groups, LIDAR_GROUPS). Returns (area m2, best beam count)."""
    F = ctx.L["findable"]
    g = ctx.grid
    X, Y = g.centers()
    an = ctx.site["ladder"]["annex"]
    host = ctx.site["ladder"]["annex_host"]
    d = ctx.doors[ctx.site["ladder"]["annex_door"]]
    w = ctx.walls[d["wall"]]
    hx0, hy0, hx1, hy1 = ctx.rooms[host]["inner"]
    real = ctx.reach_cells(MODES)
    if ctx.miss is not None and ctx.miss.room_missing(host):
        reach_draw = np.ones(g.shape, bool)     # version 2: the host's reachable floor, drawn or not
    sel = real & reach_draw & (X >= hx0) & (X <= hx1) & (Y >= hy0) & (Y <= hy1)
    sub = np.zeros(g.shape, bool)
    sub[::2, ::2] = True
    iy, ix = np.nonzero(sel & sub)
    zl = float(F["lidar_height"])
    stp = math.radians(float(F["lidar_step_deg"]))
    a0, a1 = d["center"] - d["width"] / 2, d["center"] + d["width"] / 2
    good = 0
    best = 0
    # the far face of the door's wall (on the annex side)
    s_an = 1 if w["rooms"][1] == an else -1
    plane = w["coord"] + s_an * (w["t_pos"] if s_an > 0 else w["t_neg"])
    for a, b in zip(iy, ix):
        x, y = float(X[a, b]), float(Y[a, b])
        if w["axis"] == "x":
            e1, e2 = (plane, a0), (plane, a1)
        else:
            e1, e2 = (a0, plane), (a1, plane)
        t1 = math.atan2(e1[1] - y, e1[0] - x)
        t2 = math.atan2(e2[1] - y, e2[0] - x)
        lo, hi = sorted((t1, t2))
        if hi - lo > math.pi:
            lo, hi = hi, lo + 2 * math.pi
        k0 = int(math.ceil(lo / stp))
        k1 = int(math.floor(hi / stp))
        if k1 < k0:
            continue
        angs = np.arange(k0, k1 + 1) * stp
        vec = np.stack([np.cos(angs), np.sin(angs), np.zeros(len(angs))], 1)
        # one mj_ray per beam (mj_multiRay misses geoms of static bodies with several geoms in MuJoCo
        # 3.14, which counted beams that hit such a body as passing through the door); the cell's own ray test
        # (mission.evidence.first_hits), the cell lidar's own groups 0-2, static geoms included
        from .accept import first_hits
        gid, dist = first_hits(ctx.model, ctx.data, np.array([x, y, zl]), vec, LIDAR_GROUPS, -1)
        rng_max = float(F["lidar_range"][1])
        gid = np.where((gid >= 0) & (dist > rng_max), -1, gid)        # beyond the lidar's range: nothing seen
        dplane = np.abs((plane - (x if w["axis"] == "x" else y)) / np.where(
            np.abs(vec[:, 0 if w["axis"] == "x" else 1]) < 1e-9, 1e-9, vec[:, 0 if w["axis"] == "x" else 1]))
        through = (gid < 0) | (dist > dplane + 0.3)
        cnt = int(through.sum())
        best = max(best, cnt)
        if cnt >= int(F["lidar_beams"]):
            good += 1
    return good * (2 * g.res) ** 2, best


def room_doorways(ctx, rid, h=None):
    """Doorway points (0.45 m from both wall faces, at -0.25 / 0 / +0.25 of the width) of every passable opening of
    room `rid` as built (drawn or not), really reachable: the viewpoints of a search on the missing part, where no
    door is drawn."""
    g = ctx.grid
    h = nominal_head(ctx.cfg, "stand") if h is None else h
    real = ctx.reach_cells(MODES)
    out = []
    for d in ctx.site["doors"]:
        if rid not in d.get("rooms", []) or d.get("kind") in ("fire", "stair", "airlock_outer") or \
                not d.get("wall") or d["state"] == "closed":
            continue
        w = ctx.walls[d["wall"]]
        for s in (-1, 1):
            off = (w["t_pos"] if s > 0 else w["t_neg"]) + 0.45
            for a in (-0.25, 0.0, 0.25):
                c = d["center"] + a * d["width"]
                x, y = (w["coord"] + s * off, c) if w["axis"] == "x" else (c, w["coord"] + s * off)
                iy, ix = g.cell(x, y)
                if 0 <= iy < g.ny and 0 <= ix < g.nx and real[iy, ix]:
                    out.append((x, y, h))
    return np.array(out, float).reshape(-1, 3)


def _find_rows(ctx):
    en, reach = drawn_map(ctx)
    Vd, Vl = viewpoints(ctx, en, reach)
    V = np.concatenate([Vd, Vl], 0) if len(Vl) else Vd
    found, nav_found = {}, {}
    for slot in ("C-moved", "C-relocated", "C-annex"):
        rec = ctx.by_slot[slot]
        tg = _tag_geom(ctx, rec)
        Vs = V
        if ctx.on_sheet(rec) and slot != "C-annex":
            # version 2: on the missing part the viewpoints are the doorways of its room as built (none is drawn)
            Vr = room_doorways(ctx, rec["room"])
            Vs = np.concatenate([V, Vr], 0) if len(Vr) else V
        if ctx.acc.plate_rule is not None:
            # version 2 (findable): the text plate legible from one viewpoint (14 px, 8 px, 60 deg, clear) and at
            # least 10 px wide in a navigation frame from one of them
            from .accept import plate_nav_seen, plate_seen
            if tg is None or not len(Vs):
                found[slot], nav_found[slot] = 0, 0
                continue
            ok = plate_seen(ctx.cfg, ctx.model, ctx.data, tg, Vs, ctx.body_id(rec))
            nav_found[slot] = int(plate_nav_seen(ctx.cfg, ctx.model, ctx.data, tg, Vs, ctx.body_id(rec)).sum())
        else:
            ok = tag_seen(ctx.cfg, ctx.model, ctx.data, tg, Vs, ctx.body_id(rec)) if len(Vs) else np.zeros(0, bool)
        found[slot] = int(ok.sum())
    area, beams = _annex_lidar(ctx, reach)
    F = ctx.L["findable"]
    ok = all(v > 0 for v in found.values()) and area >= float(F["lidar_area_m2"])
    if ctx.acc.plate_rule is not None:
        ok &= all(nav_found.get(s, 0) > 0 for s in found)
    rows = {"findable": _row(ok, {"viewpoints_seeing_tag": found, "annex_opening_area_m2": round(area, 2),
                                  "annex_opening_max_beams": beams, "viewpoints_nav_10px": nav_found},
                             "tag plates of C-moved / C-relocated / C-annex >= %d px from a drawn doorway or corridor "
                             "point or a lidar opening; the annex door spans >= %d beams from >= %.0f m2 of drawn "
                             "floor" % (int(F["tag_px"]), int(F["lidar_beams"]), float(F["lidar_area_m2"])))}
    rows["search_budget"] = _search_v2(ctx, en, reach) if ctx.acc.plate_rule is not None else _search(ctx, en, reach)
    return rows


def _sweep_frames_tiling(ctx, V, room_rect, wall_skip=None):
    """Frames a doorway sweep takes from viewpoint V to cover the walls of room_rect (inspect camera, columns of
    hfov - overlap, rows of vfov - overlap over the instrument band seen at each wall point). Wall points whose tag
    could not be decoded from V (too far, too oblique) are not swept."""
    S = ctx.L["search"]
    F = ctx.L["findable"]
    cam = camera(ctx.cfg, "inspect")
    ov = math.radians(float(S["overlap_deg"]))
    hcol = 2 * math.atan(cam["tan_h"]) - ov
    vrow = 2 * math.atan(cam["tan_v"]) - ov
    z0, z1 = [float(v) for v in S["mount_z"]]
    x0, y0, x1, y1 = room_rect
    pts = []
    for (ax, ay), (bx, by), n in (((x0, y0), (x1, y0), (0, 1)), ((x1, y0), (x1, y1), (-1, 0)),
                                  ((x1, y1), (x0, y1), (0, -1)), ((x0, y1), (x0, y0), (1, 0))):
        L = math.hypot(bx - ax, by - ay)
        for t in np.arange(0.15, L - 0.1, 0.3):
            px, py = ax + (bx - ax) * t / L, ay + (by - ay) * t / L
            pts.append((px, py, n))
    if not pts:
        return 0
    az, elo, ehi = [], [], []
    for px, py, n in pts:
        dx, dy = px - V[0], py - V[1]
        d = math.hypot(dx, dy)
        if d < 0.3:
            continue
        cosang = (-dx * n[0] - dy * n[1]) / d
        if cosang < math.cos(math.radians(float(F["max_off_deg"]))):
            continue
        if cam["f"] * 0.06 * cosang / d < float(F["tag_px"]):
            continue
        az.append(math.atan2(dy, dx))
        elo.append(math.atan2(z0 - V[2], d))
        ehi.append(math.atan2(z1 - V[2], d))
    if not az:
        return 0
    az = np.unwrap(np.array(sorted(az)))
    frames = 0
    a = az.min()
    elo, ehi = np.array(elo), np.array(ehi)
    order = np.argsort(np.array([math.atan2(math.sin(x), math.cos(x)) for x in az]))
    del order
    while a <= az.max() + 1e-9:
        m = (az >= a) & (az < a + hcol)
        if m.any():
            lo = max(float(elo[m].min()), math.radians(float(ctx.L["head"]["tilt_deg"][0])))
            hi = min(float(ehi[m].max()), math.radians(float(ctx.L["head"]["tilt_deg"][1])))
            frames += max(1, int(math.ceil((hi - lo) / vrow - 1e-9)))
        a += hcol
    return frames


def _instrument_targets(ctx):
    """Every instrument of the site (items, decoys, untagged ones): (key, tag geometry or None, face point)."""
    out = []
    for r in ctx.site["ips"]:
        if r["slot"] in ("M", "X"):
            continue
        out.append((r["tag"], _tag_geom(ctx, r), np.asarray(r["pose"]["pos"], float), ctx.body_id(r)))
    for r in ctx.site["decoys"]:
        sid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, "%s/tag/marker" % r["body"])
        tg = None
        if ctx.acc.plate_rule is not None:
            from .accept import plate_compiled
            tg = plate_compiled(ctx.model, ctx.data, r["body"])            # version 2: its text plate
        elif sid >= 0:
            tg = {"marker": ctx.data.site_xpos[sid].copy(), "normal": ctx.data.site_xmat[sid].reshape(3, 3)[:, 2].copy(),
                  "side": 0.06}
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, r["body"])
        out.append((r["tag"], tg, np.asarray(r["pose"]["pos"], float), bid))
    for k, r in enumerate(ctx.site.get("plain_instruments", [])):
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, r["body"])
        out.append(("plain%d" % k, None, np.asarray(r["pos"], float), bid))
    return out


def _search_v2(ctx, en, reach):
    """Version 2's search budget: the same walk as version 1 (each listed C item's recorded room from
    its doorways, the corridor outside it, the neighbouring rooms nearest first; the unrecorded item's annex from its
    door; on the missing part the doorways as built). At each standpoint (one side of one door, three points across
    it) the wide camera's candidates are the yellow plates it shows at least nav_blob_px wide (items and decoys; a
    plain instrument has no plate), plus a fixed allowance of false candidates per viewpoint (each of the three
    points is a navigation frame); one inspection frame
    per candidate, a second when the plate is seen but not legible from there (it stays a candidate for later
    standpoints); a target is found when its plate is legible (14 px, 8 px, 60 deg, clear). All three within the
    frame budget."""
    from .accept import plate_nav_seen, plate_seen
    S = ctx.L["search"]
    R = ctx.acc.plate_rule
    false_n = int(R.get("nav_false_candidates_per_viewpoint", 3))
    site = ctx.site
    h = nominal_head(ctx.cfg, "stand")
    targets = {s: ctx.by_slot[s] for s in ("C-moved", "C-relocated", "C-annex")}
    tagof = {r["tag"]: s for s, r in targets.items()}
    seen = {s: False for s in targets}
    done = set()
    frames = 0
    log = []
    objs = [o for o in _instrument_targets(ctx) if o[1] is not None]
    num2id = {r["number"]: r["id"] for r in site["rooms"]}

    def standpoints(room_id, inside_room=True, drawn_only=True):
        """[(door id, side, (3, 3) points)] of a room's passable doors."""
        out = []
        r = ctx.rooms[room_id]
        x0, y0, x1, y1 = r["rect"]
        if ctx.miss is not None and ctx.miss.room_missing(room_id):
            drawn_only = False
        for d in site["doors"]:
            if (drawn_only and not d.get("on_drawing", True)) or room_id not in d.get("rooms", []) or \
                    d.get("kind") in ("fire",) or d["state"] == "closed":
                continue
            w = ctx.walls[d["wall"]]
            for s in (-1, 1):
                off = (w["t_pos"] if s > 0 else w["t_neg"]) + 0.45
                pts = []
                for a in (-0.25, 0.0, 0.25):
                    c = d["center"] + a * d["width"]
                    x, y = (w["coord"] + s * off, c) if w["axis"] == "x" else (c, w["coord"] + s * off)
                    if inside_room is None or (x0 <= x <= x1 and y0 <= y <= y1) == inside_room:
                        pts.append((x, y, h))
                if pts:
                    out.append((d["id"], s, np.asarray(pts, float)))
        return out

    visited = set()

    def visit(sp, why):
        nonlocal frames
        did, side, P = sp
        if (did, side) in visited:
            return
        visited.add((did, side))
        # the allowance is per viewpoint, i.e. per navigation frame (the plate finder's count
        # is per frame; a standpoint that looks from several points adds them): three points, three frames
        n = false_n * len(P)
        for key, pl, face, bid in objs:
            if key in done:
                continue
            nav = plate_nav_seen(ctx.cfg, ctx.model, ctx.data, pl, P, bid)
            if not nav.any():
                continue
            n += 1
            leg = plate_seen(ctx.cfg, ctx.model, ctx.data, pl, P[nav], bid)
            if leg.any():
                done.add(key)
                if key in tagof:
                    seen[tagof[key]] = True
            else:
                n += 1                      # a second frame; still a candidate later
        frames += n
        log.append((why, n))

    for it in sorted(ctx.prior["round_sheet"]["items"], key=lambda it: it["tag"]):
        slot = tagof.get(it["tag"])
        if slot is None or seen[slot]:
            continue
        loc = it["location"]
        if loc.get("recorded"):
            rid = num2id.get(loc["room"])
            if rid is None:
                continue
            for sp in standpoints(rid, True):
                if not seen[slot]:
                    visit(sp, "room %s" % loc["room"])
            for sp in standpoints(rid, False):
                if not seen[slot]:
                    visit(sp, "corridor/outside %s" % loc["room"])
            if not seen[slot]:
                nb = []
                for w in site["walls"]:
                    if rid in w["rooms"]:
                        o = w["rooms"][0] if w["rooms"][1] == rid else w["rooms"][1]
                        if o in ctx.rooms and ctx.rooms[o]["on_drawing"] and ctx.rooms[o]["type"] != "corridor" and \
                                o not in nb:
                            nb.append(o)
                c0 = loc["pos"]
                nb.sort(key=lambda o: math.hypot(0.5 * (ctx.rooms[o]["rect"][0] + ctx.rooms[o]["rect"][2]) - c0[0],
                                                 0.5 * (ctx.rooms[o]["rect"][1] + ctx.rooms[o]["rect"][3]) - c0[1]))
                for o in nb:
                    for sp in standpoints(o, True):
                        if not seen[slot]:
                            visit(sp, "neighbour %s" % ctx.rooms[o]["number"])
        else:
            for sp in standpoints(site["ladder"]["annex"], None, drawn_only=False):
                if not seen[slot]:
                    visit(sp, "annex door")
    ok = all(seen.values()) and frames <= int(S["frame_budget"])
    return _row(ok, {"frames": frames, "found": seen, "standpoints": len(visited), "views": log[:24],
                     "false_candidates_per_viewpoint": false_n},
                "a doorway search with the wide camera's plate candidates (plus %d false ones per viewpoint), one "
                "inspection frame each and a second where a plate is not legible, finds the three text plates within %d "
                "inspection frames" % (false_n, int(S["frame_budget"])))


def _search(ctx, en, reach):
    """The doorway sweep over the bounded location set ('search budget'), simulated geometrically. For
    each listed C item in round-sheet order: its drawn room (from each drawn doorway), then the corridor walls
    outside the room (from the corridor side of its doors), then its neighbouring drawn rooms nearest first; for the
    item without a recorded location: the annex (from its door, which the lidar shows as an opening while the robot
    walks the outer rooms). At each viewpoint the wide camera finds the instruments in view (line of sight, tag plate
    at least 8 px wide in the 640 px / 90 deg nav image, i.e. within 7.6 m) and the inspection camera takes one
    frame of each instrument not yet looked at; a target is found when its tag is decodable in that frame. Frames of
    all three searches add up; a tag found during another item's search counts."""
    S = ctx.L["search"]
    site = ctx.site
    h = nominal_head(ctx.cfg, "stand")
    targets = {s: ctx.by_slot[s] for s in ("C-moved", "C-relocated", "C-annex")}
    tagof = {r["tag"]: s for s, r in targets.items()}
    seen = {s: False for s in targets}
    looked = set()
    frames = 0
    log = []
    objs = _instrument_targets(ctx)
    nav = camera(ctx.cfg, "nav")
    d_det = nav["f"] * 0.1908 / 8.0
    num2id = {r["number"]: r["id"] for r in site["rooms"]}

    def doorway_points(room_id, inside_room=True, drawn_only=True):
        out = []
        r = ctx.rooms[room_id]
        if ctx.miss is not None and ctx.miss.room_missing(room_id):
            drawn_only = False            # version 2: a room of the missing sheet is searched from its doors as built
        x0, y0, x1, y1 = r["rect"]
        for d in site["doors"]:
            if (drawn_only and not d.get("on_drawing", True)) or room_id not in d.get("rooms", []) or \
                    d.get("kind") in ("fire",) or d["state"] == "closed":
                continue
            w = ctx.walls[d["wall"]]
            for s in (-1, 1):
                off = (w["t_pos"] if s > 0 else w["t_neg"]) + 0.45
                for a in (-0.25, 0.0, 0.25):
                    c = d["center"] + a * d["width"]
                    x, y = (w["coord"] + s * off, c) if w["axis"] == "x" else (c, w["coord"] + s * off)
                    if inside_room is None or (x0 <= x <= x1 and y0 <= y <= y1) == inside_room:
                        out.append((x, y, h))
        return out

    def view(V, why):
        nonlocal frames
        n = 0
        V = np.asarray(V, float)
        for key, tg, face, bid in objs:
            if key in looked:
                continue
            P = np.asarray(tg["marker"] if tg else face, float)
            if np.linalg.norm(P - V) > d_det:
                continue
            if not los(ctx.model, ctx.data, P + 0.01 * (tg["normal"] if tg else 0.0), V[None, :], bid)[0]:
                continue
            looked.add(key)
            n += 1
            if key in tagof and tg is not None:
                if tag_seen(ctx.cfg, ctx.model, ctx.data, tg, V[None, :], bid)[0]:
                    seen[tagof[key]] = True
                else:
                    looked.discard(key)       # seen but not decodable from here: look again from elsewhere
        frames += n
        log.append((why, n))

    for it in sorted(ctx.prior["round_sheet"]["items"], key=lambda it: it["tag"]):
        slot = tagof.get(it["tag"])
        if slot is None or seen[slot]:
            continue
        loc = it["location"]
        if loc.get("recorded"):
            rid = num2id.get(loc["room"])
            if rid is None:
                continue
            for V in doorway_points(rid, True):
                if not seen[slot]:
                    view(V, "room %s" % loc["room"])
            for V in doorway_points(rid, False):
                if not seen[slot]:
                    view(V, "corridor/outside %s" % loc["room"])
            if not seen[slot]:
                nb = []
                for w in site["walls"]:
                    if rid in w["rooms"]:
                        o = w["rooms"][0] if w["rooms"][1] == rid else w["rooms"][1]
                        if o in ctx.rooms and ctx.rooms[o]["on_drawing"] and ctx.rooms[o]["type"] != "corridor" and \
                                o not in nb:
                            nb.append(o)
                c0 = loc["pos"]
                nb.sort(key=lambda o: math.hypot(0.5 * (ctx.rooms[o]["rect"][0] + ctx.rooms[o]["rect"][2]) - c0[0],
                                                 0.5 * (ctx.rooms[o]["rect"][1] + ctx.rooms[o]["rect"][3]) - c0[1]))
                for o in nb:
                    for V in doorway_points(o, True):
                        if not seen[slot]:
                            view(V, "neighbour %s" % ctx.rooms[o]["number"])
        else:
            for V in doorway_points(site["ladder"]["annex"], None, drawn_only=False):
                if not seen[slot]:
                    view(V, "annex door")
    ok = all(seen.values()) and frames <= int(S["frame_budget"])
    return _row(ok, {"frames": frames, "found": seen, "views": log[:24], "detect_range_m": round(d_det, 2)},
                "a doorway sweep (wide camera finds the instruments, one inspect frame each) over the bounded location "
                "set finds the three tag plates within %d inspect frames" % int(S["frame_budget"]))


# ---------------------------------------------------------------- the documents' stated facts
def _fact_rows(ctx):
    site = ctx.site
    errs = []
    nb = {}
    for w in site["walls"]:
        a, b = w["rooms"]
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    num2id = {r["number"]: r["id"] for r in site["rooms"]}
    for it in ctx.prior["round_sheet"]["items"]:
        rec = ctx.recs.get(it["tag"])
        if rec is None or rec["slot"] in ("X",):
            continue
        loc = it["location"]
        true_room = rec["room"]
        if loc.get("recorded"):
            dr = num2id.get(loc["room"])
            if true_room == dr:
                continue
            if true_room in nb.get(dr, set()) and ctx.rooms[true_room]["type"] != "corridor":
                continue
            if ctx.rooms[true_room]["type"] == "corridor":
                # on the corridor wall outside the drawn room: within 0.7 m of the drawn room's rect, outside it
                x0, y0, x1, y1 = ctx.rooms[dr]["rect"]
                px, py = rec["pose"]["pos"][:2]
                dx = max(x0 - px, 0, px - x1)
                dy = max(y0 - py, 0, py - y1)
                if math.hypot(dx, dy) <= 1.0 and (x0 - 1.0 <= px <= x1 + 1.0) and (y0 - 1.0 <= py <= y1 + 1.0):
                    continue
            errs.append("%s: in %s, the round sheet says %s" % (rec["tag"], ctx.rooms[true_room]["number"], loc["room"]))
        else:
            if ctx.rooms[true_room]["type"] != "annex":
                errs.append("%s: no recorded location but not in the annex" % rec["tag"])
    an = [r for r in site["rooms"] if not r["on_drawing"]]
    if len(an) != 1:
        errs.append("%d undrawn rooms" % len(an))
    else:
        a = an[0]
        hosts = sorted({x for d in site["doors"] if a["id"] in d.get("rooms", []) for x in d["rooms"]} -
                       {a["id"], "OUT"})
        if len(hosts) != 1:
            errs.append("annex entered from %s" % hosts)
        else:
            h = hosts[0]
            b = site["building"]["rect"]
            ar = a["rect"]
            hr = ctx.rooms[h]["rect"]
            outside = ar[2] <= b[0] + 1e-6 or ar[0] >= b[2] - 1e-6 or ar[3] <= b[1] + 1e-6 or ar[1] >= b[3] - 1e-6
            on_facade = any(abs(hr[i] - b[i]) < 1e-6 for i in range(4))
            if not (outside and on_facade):
                errs.append("annex %s not outside the building against outer-wall room %s" % (a["id"], h))
            dd = [d for d in site["doors"] if a["id"] in d.get("rooms", [])]
            if len(dd) != 1 or dd[0]["on_drawing"]:
                errs.append("annex door count / drawn")
    n_listed = len([it for it in ctx.prior["round_sheet"]["items"] if it.get("kind") == "instrument"])
    off = len([s for s in INSTR if s in ("C-moved", "C-relocated", "C-annex")])
    frac = off / max(1, n_listed)
    if not (0.25 <= frac <= 0.42):
        errs.append("share of listed instruments away from the drawn place %.2f" % frac)
    return {"facts": _row(not errs, errs[:6], "the facility notes' statements hold: a listed instrument is in its drawn "
                                              "room, a neighbouring room or on the corridor wall outside it; one "
                                              "without a recorded location is in the one undrawn annex")}


# ---------------------------------------------------------------- dose
def _dose_rows(ctx):
    site = ctx.site
    D = ctx.L["dose"]
    f = float(D["unit_factor"])
    rating = float(D["rating_gy"]) / f
    rows = {}
    R = site["route"]
    frac = float(R["dose_total_uSv"]) / rating
    rows["dose_route"] = _row(frac <= float(D["route_fraction_max"]) + 1e-9,
                              {"fraction": round(frac, 3), "uSv": round(R["dose_total_uSv"], 3),
                               "gy": round(R["dose_total_uSv"] * f, 3)},
                              "the reference route uses at most %.0f %% of the dose rating" %
                              (100 * float(D["route_fraction_max"])))
    # the mark
    m = ctx.by_slot["M"]
    p = np.array([m["pose"]["pos"][:2]])
    rate = float(ctx.rate(p)[0])
    lo, hi = [float(v) / f for v in D["mark_band_gyh"]]
    zone = _zone_at(ctx, p[0])
    # version 2: zoned controlled where its sheet survives, or on the missing part
    on_part = ctx.miss is not None and ctx.miss.in_part(float(p[0][0]), float(p[0][1]))
    okm = lo <= rate <= hi and (zone == "controlled" or (on_part and zone is None))
    R_ = ctx.regions[m["tag"]]
    st, cr = ctx.stand_crouch(MODES)
    area = float(((R_["stand"] & st) | (R_["crouch"] & cr)).sum()) * ctx.res ** 2
    rows["M_mark"] = _row(okm and area >= 0.05, {"rate_gyh": round(rate * f, 3), "zone": zone, "area_m2": round(area, 3)},
                          "the survey mark: field inside %.1f-%.1f Gy/h, in a zone drawn controlled, reachable within "
                          "0.25 m" % tuple(D["mark_band_gyh"]))
    # X
    xs = [s for s in site["sources"] if not s["documented"]][0]
    zx = _zone_at(ctx, xs["pos"][:2])
    sup = float(site["ladder"]["dose"]["supervised_hi"])
    need = float(D["x_factor"]) * sup
    best = (0.0, None)
    from scipy.sparse.csgraph import dijkstra
    G = ctx.trav.build_graph(MODES)
    dist, pred = dijkstra(G, directed=True, indices=[ctx.pad], return_predecessors=True)
    dist, pred = dist[0], pred[0]
    for slot in INSTR:
        if not (slot.startswith("B") or slot.startswith("C")):
            continue
        rec = ctx.by_slot[slot]
        n = ctx.trav.node(rec["observation_pose"]["pos"][0], rec["observation_pose"]["pos"][1],
                          rec["observation_pose"]["mode"])
        if not np.isfinite(dist[n]):
            continue
        cells = []
        while n >= 0 and n != ctx.pad and len(cells) < 400000:
            iy, ix, _ = ctx.trav.unnode(n)
            cells.append(ctx.grid.xy(iy, ix))
            n = int(pred[n])
        P = np.array(cells[::-1])
        if len(P) < 2:
            continue
        r = ctx.rate(P)
        zones = np.array([_zone_at(ctx, q) == "supervised" for q in P[::4]]).repeat(4)[:len(P)]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        L = float(seg[(r[1:] >= need) & zones[1:]].sum())
        if L > best[0]:
            best = (L, slot)
    rows["X_zone"] = _row(zx == "supervised" and best[0] >= float(D["x_length_m"]),
                          {"zone": zx, "length_m": round(best[0], 2), "to": best[1],
                           "x_gyh_at_1m": round(float(xs["strength"]) * f, 1)},
                          "X lies in a zone drawn supervised and raises the rate to >= %.0f x that level along >= "
                          "%.0f m of the route to some B or C item" % (float(D["x_factor"]), float(D["x_length_m"])))
    # the evidence place of the source report: reachable, dose distance <= 35 % of the rating
    Rx = ctx.regions["X"]
    dose_cost = _dose_graph(ctx, G)
    dd = dijkstra(dose_cost, directed=True, indices=[ctx.pad])[0]
    N = ctx.grid.nx * ctx.grid.ny
    gen = Rx["generous"]
    nodes = np.concatenate([np.flatnonzero(gen & ctx.trav.enter[m]) + k * N for k, m in enumerate(MODES)])
    best_d = float(np.min(dd[nodes])) if len(nodes) else float("inf")
    if len(nodes):
        k = nodes[int(np.argmin(dd[nodes]))]
        iy, ix, _ = ctx.trav.unnode(k)
        best_d += float(Rx["rate"][iy, ix]) * 3.0 / 3600.0
    lim = float(D["route_fraction_max"]) * rating
    rows["X_evidence"] = _row(np.isfinite(best_d) and best_d <= lim,
                              {"dose_uSv": round(best_d, 3) if np.isfinite(best_d) else None,
                               "fraction": round(best_d / rating, 3) if np.isfinite(best_d) else None},
                              "a place where the robot's dosimeter reads >= %.1f Gy/h is reachable within %.0f %% of the "
                              "rating" % (float(D["evidence_gyh"]), 100 * float(D["route_fraction_max"])))
    return rows


def _zone_at(ctx, p):
    for z in ctx.prior["zoning"]["zones"]:
        xs = [q[0] for q in z["polygon"]]
        ys = [q[1] for q in z["polygon"]]
        if min(xs) - 1e-6 <= p[0] <= max(xs) + 1e-6 and min(ys) - 1e-6 <= p[1] <= max(ys) + 1e-6:
            return z["zone_class"]
    return None


# ---------------------------------------------------------------- elements on the route
def _element_rows(ctx):
    site = ctx.site
    kinds = [str(k) for k in ctx.L["high_step_kinds"]]
    f0, f1 = [float(v) for v in ctx.L["high_step_band"]]
    from .world import draw_interval
    del f0, f1
    bad = []
    cr = site["route"].get("crossed_elements", {})
    for eid, e in cr.items():
        if e["class"] != "needs_high_step":
            continue
        if e["kind"] not in kinds:
            bad.append("%s %s" % (eid, e["kind"]))
            continue
        lo, hi = draw_interval(ctx.cfg, e["kind"], "needs_high_step")      # the drawn range x high_step_band
        v = e["param"].get("height", e["param"].get("diameter"))
        if v is not None and not (lo - 1e-4 <= v <= hi + 1e-4):
            bad.append("%s %s %.3f" % (eid, e["kind"], v))
    return {"route_elements": _row(not bad, {"crossed": {k: (v["kind"], v["class"]) for k, v in cr.items()},
                                             "bad": bad},
                                   "legged crossings on the reference route only of the allowed kinds %s and inside "
                                   "the configured part of the band" % kinds)}


# ---------------------------------------------------------------- access rules
def _access_rows(ctx):
    """No low-friction patch on a required access; no ajar door over a sill that needs a mode (and no needs_*
    element inside a tight gap); every ramp at most access.max_slope_deg; no generic geometry between the rated
    high-step range and a barrier near the reference route."""
    site = ctx.site
    t = ctx.trav
    g = ctx.grid
    rows = {}
    # 1. puddles: making one solid loses no reachable floor (all modes, walking alone) and no observation pose
    bad = []
    full = {k: ctx.reach_cells(k) for k in (MODES, ("walk",))}
    near_r = float(t.th["r_in"]) + 0.10
    ramps = np.zeros(g.shape, np.uint8)
    for e in site["elements"]:
        if e["kind"] == "ramp":
            g.fill_poly(ramps, e["footprint"], 1)
    ramps = ramps.astype(bool)
    for e in [e for e in site["elements"] if e["kind"] == "puddle"]:
        m = np.zeros(g.shape, np.uint8)
        g.fill_poly(m, e["footprint"], 1)
        mask = m.astype(bool)
        if (mask & ramps).any():
            bad.append("%s lies on a ramp" % e["id"])
        near = ndimage.distance_transform_edt(~mask) * g.res <= near_r
        for modes in (MODES, ("walk",)):
            cut = ctx.reach_cells(modes, mask, ("puddle", e["id"]))
            lost = int((full[modes] & ~cut & ~near).sum())
            if lost > 10:
                bad.append("%s cuts %d cells off (%s)" % (e["id"], lost, "/".join(modes)))
        for rec in site["ips"]:
            ob = rec.get("observation_pose")
            if not ob:
                continue
            modes = SLOT_MODES[rec["slot"]]
            cut = ctx.reach_mode(ob["mode"], modes, mask, ("puddle", e["id"], tuple(modes)))
            iy, ix = g.cell(*ob["pos"])
            if not cut[iy, ix] and not near[iy, ix]:
                bad.append("%s blocks the observation pose of %s" % (e["id"], rec["tag"]))
    rows["access_friction"] = _row(not bad, {"puddles": len([e for e in site["elements"] if e["kind"] == "puddle"]),
                                             "pruned": len(site["ladder"].get("pruned_puddles", [])), "bad": bad[:6]},
                                   "no low-friction patch on a required access")
    # 2. no ajar door over a sill that needs a mode; no needs_* element inside a tight gap
    E = ctx.elements
    bad = []
    for d in site["doors"]:
        if d.get("state") == "ajar" and d.get("sill") and E.get(d["sill"], {}).get("class") not in (None, "free"):
            bad.append("door %s ajar over sill %s (%s)" % (d["id"], d["sill"], E[d["sill"]]["class"]))

    def bbox(fp):
        P = np.asarray(fp, float)
        return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
    for e in site["elements"]:
        if e["class"] != "tight":
            continue
        a = bbox(e["footprint"])
        for f in site["elements"]:
            if f["class"] not in ("needs_high_step", "needs_crouch") or f["id"] == e["id"]:
                continue
            b = bbox(f["footprint"])
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                bad.append("tight %s over %s %s" % (e["id"], f["class"], f["id"]))
    rows["access_combinations"] = _row(not bad, bad[:6], "no tight gap combined with an element that needs a mode")
    # 3. slopes
    smax = float(ctx.L.get("access", {}).get("max_slope_deg", 15.0))
    steep = ["%s %.1f deg" % (e["id"], e["param"].get("slope_deg", 0.0)) for e in site["elements"]
             if e["kind"] == "ramp" and float(e["param"].get("slope_deg", 0.0)) > smax + 1e-6]
    rows["access_slopes"] = _row(not steep, steep, "every ramp at most %.0f deg" % smax)
    # 4. generic geometry between the rated high-step range and a barrier near the reference route
    from .world import draw_interval
    th = t.th
    lo = float(th["step_free"])
    hs_top = max(draw_interval(ctx.cfg, k, "needs_high_step")[1] for k in ("sill", "hose", "bund", "floor_pipe"))
    barrier = float(ctx.cfg["capability"]["step"]["barrier_min"])
    cells = np.zeros(g.shape, bool)
    for xy, modes, _T in getattr(ctx, "_route_cells", []) or []:
        for (x, y) in xy:
            iy, ix = g.cell(x, y)
            if 0 <= iy < g.ny and 0 <= ix < g.nx:
                cells[iy, ix] = True
    if not cells.any():
        for leg in site["route"].get("legs", []):
            for sg in leg.get("segments", []):
                for x, y in sg.get("waypoints", []):
                    iy, ix = g.cell(x, y)
                    if 0 <= iy < g.ny and 0 <= ix < g.nx:
                        cells[iy, ix] = True
    near_route = ndimage.distance_transform_edt(~cells) * g.res <= float(th["r_circ"]) if cells.any() else cells
    m, d = ctx.model, ctx.data
    egeoms = {gn for e in site["elements"] for gn in e.get("geoms", [])}
    found = []
    for i in range(m.ngeom):
        if m.geom_group[i] not in (0, 3) or (m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0):
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm in egeoms or nm.startswith(("ip/", "inst/", "marker/")):
            continue
        a = np.asarray(m.geom_aabb[i])
        c = d.geom_xpos[i] + d.geom_xmat[i].reshape(3, 3) @ a[:3]
        ext = np.abs(d.geom_xmat[i].reshape(3, 3)) @ a[3:]
        z0, z1 = c[2] - ext[2], c[2] + ext[2]
        if z0 > 0.03 or not (hs_top + 0.005 < z1 < barrier):
            continue
        iy0, ix0 = g.cell(c[0] - ext[0], c[1] - ext[1])
        iy1, ix1 = g.cell(c[0] + ext[0], c[1] + ext[1])
        sub = near_route[max(0, iy0):max(0, iy1) + 1, max(0, ix0):max(0, ix1) + 1]
        if sub.size and sub.any():
            found.append("%s top %.3f m" % (nm, z1))
    rows["access_bands"] = _row(not found, {"generic_between_bands_near_route": found[:6], "rated_high_step_top_m":
                                            round(hs_top, 4), "barrier_m": barrier, "free_max_m": round(lo, 4)},
                                "no generic geometry between the rated high-step range and a barrier near the route")
    return rows


# ---------------------------------------------------------------- rows added after reviews
RED_MATS = ("pipe_fw", "red_fire", "hose_red", "door_leaf_fire", "door_frame_fire")
RED_BEHIND_DEG = 8.0
PAD_LINK_R = 1.0                 # the documents: the link is up everywhere within 1.0 m of the pad centre


def _red_points(ctx):
    """Sample points of every red geom of the compiled site (fire-water pipes, fire doors, extinguishers, red hoses):
    (points (M, 3), geom ids (M,), geom names)."""
    if getattr(ctx, "_red", None) is None:
        m, d = ctx.model, ctx.data
        from mission.evidence import geom_points
        pts, gids = [], []
        for i in range(m.ngeom):
            mid = int(m.geom_matid[i])
            if mid < 0 or m.geom_group[i] > 3:
                continue
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or "") not in RED_MATS:
                continue
            P = [np.asarray(d.geom_xpos[i], float)[None, :], geom_points(m, d, i)]
            if int(m.geom_type[i]) in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                ax = np.asarray(d.geom_xmat[i], float).reshape(3, 3)[:, 2]
                hl = float(m.geom_size[i][1])
                n = max(2, int(math.ceil(2 * hl / 0.1)) + 1)
                P.append(np.asarray(d.geom_xpos[i], float)[None, :] + np.linspace(-hl, hl, n)[:, None] * ax[None, :])
            P = np.vstack(P)
            pts.append(P)
            gids.append(np.full(len(P), i, np.int32))
        ctx._red = (np.vstack(pts) if pts else np.zeros((0, 3)), np.concatenate(gids) if gids else np.zeros(0, np.int32))
    return ctx._red


def red_behind(ctx, C, aim, rec, deg=RED_BEHIND_DEG):
    """Is red geometry visible from camera C within `deg` of the line of sight to `aim`, beyond it (the lever's own
    station excluded)? Visible: a ray from C to the point meets that red geom first (geom groups 0-3)."""
    Q, G = _red_points(ctx)
    if not len(Q):
        return False
    C = np.asarray(C, float)
    v = np.asarray(aim, float) - C
    dv = float(np.linalg.norm(v))
    v /= max(dv, 1e-9)
    w = Q - C[None, :]
    along = w @ v
    perp = np.linalg.norm(w - along[:, None] * v[None, :], axis=1)
    ang = np.degrees(np.arctan2(perp, np.maximum(along, 1e-9)))
    cand = np.nonzero((along > dv) & (ang < deg))[0]
    if not len(cand):
        return False
    own = "station/%s/" % rec["tag"]
    m, d = ctx.model, ctx.data
    gid = np.zeros(1, np.int32)
    groups = np.array([1, 1, 1, 1, 0, 0], np.uint8)
    for j in cand[np.argsort(ang[cand], kind="stable")][:400]:
        g = int(G[j])
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(own):
            continue
        u = Q[j] - C
        dist = float(np.linalg.norm(u))
        hit = mujoco.mj_ray(m, d, C, u / max(dist, 1e-9), groups, 1, -1, gid)
        if int(gid[0]) == g or hit < 0 or hit >= dist - 0.02:
            return True
    return False


def _fixpass_rows(ctx):
    """Rows added after reviews (each guards a finding)."""
    rows = {}
    site, cfg, g = ctx.site, ctx.cfg, ctx.grid
    a2 = ctx.res ** 2
    from . import linkray as LR
    # -------- the link within 1.0 m of the pad centre, the cell's rule, three postures
    reach = ctx.reach_cells(MODES)
    pad_disc = LR.disc(g, (0.0, 0.0), PAD_LINK_R) & reach
    down = {}
    for posture in ("walk", "crouch_walk", "sit"):
        up = ctx.link3d(posture, base_only=False, region=pad_disc)
        down[posture] = int((pad_disc & ~up).sum())
    rows["link_pad"] = _row(not any(down.values()), {"cells": int(pad_disc.sum()), "link_down": down},
                            "the link is up on every reachable cell within %.1f m of the pad centre (lidar at the "
                            "walking, crouching and sitting heights; the cell's ray rule)" % PAD_LINK_R)
    # -------- slot A: accepted reading places on the airlock side inside the base station's link
    if "A" in ctx.by_slot and ctx.by_slot["A"]["tag"] in ctx.regions:
        rec = ctx.by_slot["A"]
        R = ctx.regions[rec["tag"]]
        st, _ = ctx.stand_crouch(SLOT_MODES["A"])
        mA = R["stand"] & st & ctx.a_side() & ctx.link3d("walk", base_only=True)
        rp = reading_place_cells(cfg, g, R["feat"], mA, rec)
        area = float(rp.sum()) * a2
        rows["A_link_reading"] = _row(area >= float(ctx.L["margin"]["min_area_m2"]),
                                      {"area_m2": round(area, 3), "margin_cells_in_link": int(mA.sum())},
                                      "slot A: at least %.2f m2 of accepted reading places on the airlock side of the "
                                      "corridor sill inside the base station's link (walking)" %
                                      float(ctx.L["margin"]["min_area_m2"]))
    # -------- no tight passage on a required access
    from .ladder import tight_lost
    tights = [e for e in site["elements"] if e["class"] == "tight"]
    ajar = [e for e in tights if e["kind"] == "narrow_gap" and (e.get("param") or {}).get("door")]
    lost, where = tight_lost(ctx.trav, ctx.pad, [ctx.elem_mask([e["id"]]) for e in ajar])
    blocked = []
    if tights:
        union = ctx.elem_mask([e["id"] for e in tights])
        near = ndimage.distance_transform_edt(~union) * ctx.res <= float(ctx.trav.th["r_in"]) + 0.10
        for rec in site["ips"]:
            ob = rec.get("observation_pose")
            if not ob:
                continue
            cut = ctx.reach_mode(ob["mode"], SLOT_MODES[rec["slot"]], union, ("tight", tuple(SLOT_MODES[rec["slot"]])))
            iy, ix = g.cell(*ob["pos"])
            if not cut[iy, ix] and not near[iy, ix]:
                blocked.append(rec["tag"])
    ex = None
    if lost:
        iy, ix = np.nonzero(where)
        ex = [round(float(v), 2) for v in g.xy(iy[0], ix[0])]
    rows["access_tight"] = _row(lost <= 10 and not blocked,
                                {"tight_elements": [e["id"] for e in tights], "ajar_leaves": [e["id"] for e in ajar],
                                 "floor_lost_behind_ajar_leaves": lost, "example": ex,
                                 "observation_poses_behind_tight": blocked,
                                 "opened_ajar": len(site["ladder"].get("opened_ajar", []))},
                                "no tight passage on a required access: no floor only behind a door left ajar, and "
                                "every observation pose (instruments, mark, source) reachable with every tight element "
                                "(narrow gap, ajar leaf) solid")
    # -------- nothing in front of a level gauge's scale plate or a lever valve's tag plate
    bad, tested = {}, 0
    rng = np.random.default_rng(5)
    X, Y = g.centers()
    for rec in site["ips"]:
        if rec["type"] not in ("level", "valve_lever") or rec["tag"] not in ctx.regions:
            continue
        R = ctx.regions[rec["tag"]]
        eg = R["feat"]
        pp = getattr(eg, "plate_pts", None)
        if pp is None or not len(pp):
            bad[rec["tag"]] = "no plate grid"
            continue
        nb = 0
        for mode_key, which in (("stand", "stand"), ("crouch", "crouch")):
            iy, ix = np.nonzero(R[mode_key])
            if not len(iy):
                continue
            pick = rng.choice(len(iy), size=min(150, len(iy)), replace=False)
            B = np.c_[X[iy[pick], ix[pick]], Y[iy[pick], ix[pick]]]
            Cm = camera_from_body(cfg, B, eg.aim, nominal_head(cfg, which), eg.sensor)
            ok = ctx.caster.clear(eg, Cm, pts=pp).all(axis=1)
            nb += int((~ok).sum())
            tested += len(Cm)
        if nb:
            bad[rec["tag"]] = nb
    rows["plates_clear"] = _row(not bad, {"cells_tested": tested, "blocked": bad},
                                "from the margin cells (the accepted reading places) every point of a dense grid over "
                                "a level gauge's scale plate and a lever valve's tag plate is clear")
    # -------- no red object behind a lever valve from its observation pose
    red = []
    for rec in site["ips"]:
        if rec["type"] != "valve_lever" or not rec.get("observation_pose") or rec["tag"] not in ctx.regions:
            continue
        ob = rec["observation_pose"]
        if red_behind(ctx, np.asarray(ob["camera"], float), ctx.regions[rec["tag"]]["feat"].aim, rec):
            red.append(rec["tag"])
    rows["lever_background"] = _row(not red, red, "no red geometry visible within %.0f deg behind a lever valve from "
                                                  "its observation pose" % RED_BEHIND_DEG)
    # -------- frames aimed at a C-moved / C-relocated item's drawn place never accept the true instrument
    rows["drawing_aim_C"] = _drawing_aim_row(ctx)
    # -------- the drawn zoning holds for the documented sources on all reachable floor
    rows["zoning_documented"] = _zoning_row(ctx)
    return rows


# ============================================================================================ version 2 rows
def sheet_nothing_drawn(site, prior, step=0.25):
    """Row sheet_nothing_drawn: the documents hold nothing of a missing room or a cut-off corridor part (no
    room, zone, equipment, marker plate; no wall but those shared with drawn rooms; `continues` on every cut corridor),
    every drawn wall bounds a drawn room or drawn corridor part, and every drawn room's walls are drawn. Works on
    site.json and prior.json alone. Returns a list of errors."""
    from .sheet import Missing
    ms = site.get("missing_sheet")
    if not (isinstance(ms, dict) and ms.get("place")):
        return []
    m = Missing.from_json(ms, site["building"]["rect"])
    errs = []
    rooms = {r["id"]: r for r in site["rooms"]}
    num2id = {r["number"]: r["id"] for r in site["rooms"]}
    areas = m.areas()

    def inside_part(x, y, eps=0.02):
        return any(a[0] + eps < x < a[2] - eps and a[1] + eps < y < a[3] - eps for a in areas)

    def overlap(r):
        return max((max(0.0, min(r[2], a[2]) - max(r[0], a[0])) * max(0.0, min(r[3], a[3]) - max(r[1], a[1]))
                    for a in areas), default=0.0)
    # rooms
    listed = set()
    for r in prior["rooms"]:
        rid = num2id.get(r["number"])
        listed.add(rid)
        if rid in m.rooms:
            errs.append("room %s of the missing sheet is listed" % r["number"])
        if overlap(r["rect"]) > 0.01:
            errs.append("room %s overlaps the missing part" % r["number"])
        c = m.cut_of(rid)
        if c is not None and not r.get("continues"):
            errs.append("cut corridor %s without continues" % r["number"])
        if c is not None and any(abs(a - b) > 1e-3 for a, b in zip(r["rect"], c["drawn_rect"])):
            errs.append("cut corridor %s: rect is not its drawn part" % r["number"])
        if c is None and r.get("continues"):
            errs.append("room %s has continues but is not cut" % r["number"])
    for rid, r in rooms.items():
        if r["on_drawing"] and rid not in m.rooms and rid not in listed:
            errs.append("drawn room %s is not listed" % r["number"])

    # drawn parts (rects) of the rooms: what a drawn wall must bound
    drawn_parts = []
    for r in prior["rooms"]:
        drawn_parts.append(r["rect"])

    def side_ok(x, y):
        return any(a[0] - 1e-6 <= x <= a[2] + 1e-6 and a[1] - 1e-6 <= y <= a[3] + 1e-6 for a in drawn_parts)
    segs = []
    for w in prior["walls"]:
        (ax, ay), (bx, by) = w["a"], w["b"]
        t, off = float(w["thickness"]), float(w.get("offset", 0.0))
        L = math.hypot(bx - ax, by - ay)
        segs.append(((ax, ay), (bx, by)))
        if L < 1e-6:
            continue
        n = max(2, int(math.ceil(L / step)) + 1)
        for k in range(n):
            s = (k + 0.5) / n
            x, y = ax + (bx - ax) * s, ay + (by - ay) * s
            if abs(ax - bx) < 1e-9:     # a wall on x = const
                p1, p2 = (x + off - t / 2 - 0.05, y), (x + off + t / 2 + 0.05, y)
            else:
                p1, p2 = (x, y + off - t / 2 - 0.05), (x, y + off + t / 2 + 0.05)
            if not (side_ok(*p1) or side_ok(*p2)):
                errs.append("wall %s-%s bounds no drawn room at (%.2f, %.2f)" % (w["a"], w["b"], x, y))
                break
    # every drawn room's edges are on drawn walls (except a cut's line, corridor junctions and the airlock's west
    # face, which is the facade line of the airlock vestibule)

    def on_wall(x, y):
        for (ax, ay), (bx, by) in segs:
            if abs(ax - bx) < 1e-9 and abs(x - ax) < 0.02 and min(ay, by) - 0.02 <= y <= max(ay, by) + 0.02:
                return True
            if abs(ay - by) < 1e-9 and abs(y - ay) < 0.02 and min(ax, bx) - 0.02 <= x <= max(ax, bx) + 0.02:
                return True
        return False
    corr = [r["rect"] for r in prior["rooms"] if r["type"] == "corridor"]
    cut_lines = [(c["axis"], c["at"]) for c in m.cuts]
    for r in prior["rooms"]:
        x0, y0, x1, y1 = r["rect"]
        for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            L = math.hypot(bx - ax, by - ay)
            n = max(2, int(math.ceil(L / step)))
            for k in range(n):
                s = (k + 0.5) / n
                x, y = ax + (bx - ax) * s, ay + (by - ay) * s
                if any((ax_ == "x" and abs(ax - bx) < 1e-9 and abs(x - at) < 1e-6) or
                       (ax_ == "y" and abs(ay - by) < 1e-9 and abs(y - at) < 1e-6) for ax_, at in cut_lines):
                    continue
                if r["type"] == "corridor" and sum(1 for c in corr if c[0] - 1e-6 <= x <= c[2] + 1e-6 and
                                                   c[1] - 1e-6 <= y <= c[3] + 1e-6) > 1:
                    continue
                if not on_wall(x, y):
                    errs.append("edge of room %s at (%.2f, %.2f) not on a drawn wall" % (r["number"], x, y))
                    break
    # doors on drawn walls, none inside the missing part
    for d in prior["doors"]:
        x, y = d["pos"]
        if inside_part(x, y):
            errs.append("door at (%.2f, %.2f) inside the missing part" % (x, y))
        elif d["kind"] != "fire_door" and not on_wall(x, y):
            errs.append("door at (%.2f, %.2f) on no drawn wall" % (x, y))
    for e in prior["equipment"]:
        if e.get("shape") == "circle":
            cx, cy = e["center"]
        else:
            xs = [p[0] for p in e["outline"]]
            ys = [p[1] for p in e["outline"]]
            cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        if inside_part(cx, cy):
            errs.append("equipment %s inside the missing part" % e.get("label"))
    for z in prior["zoning"]["zones"]:
        xs = [q[0] for q in z["polygon"]]
        ys = [q[1] for q in z["polygon"]]
        if overlap([min(xs), min(ys), max(xs), max(ys)]) > 0.01:
            errs.append("zone %s overlaps the missing part" % z.get("label"))
    for gm in prior["grid_markers"]:
        x, y = gm["pos"][:2]
        if m.in_cut(x, y, margin=-0.2) or m.in_missing_room(x, y):
            errs.append("grid marker %s on the missing part" % gm.get("id"))
    return errs


def _v2_rows(ctx):
    """Version 2's rows: sheet_items, sheet_rules, sheet_nothing_drawn, sheet_edges, outline_walls,
    approach_flat, variants."""
    rows = {}
    site, cfg = ctx.site, ctx.cfg
    m = ctx.miss
    L = site["ladder"]
    lo, hi = [int(min(ctx.L["v2_params"]["sheet"]["counts"])), int(max(ctx.L["v2_params"]["sheet"]["counts"]))]
    # ---- sheet_items: 2-4 listed instruments with their true place on the missing sheet
    on = [ctx.by_slot[s]["tag"] for s in INSTR if ctx.on_sheet(ctx.by_slot[s])]
    rows["sheet_items"] = _row(m is not None and lo <= len(on) <= hi and sorted(on) == sorted(m.items),
                               {"count": len(on), "tags": on, "slots": [s for s in INSTR if
                                                                       ctx.on_sheet(ctx.by_slot[s])],
                                "place": m.place if m else None},
                               "%d to %d listed instruments have their true place on the missing sheet" % (lo, hi))
    # ---- sheet_rules: the C-sealed room, the relocated item's recorded room, the source's place drawn
    bad = []
    if m is not None:
        if m.room_missing(L["c_sealed_room"]):
            bad.append("C-sealed room on the missing sheet")
        rl = ctx.by_slot["C-relocated"]
        dr = (rl.get("documented") or {}).get("room")
        if dr is None or m.room_missing(dr):
            bad.append("C-relocated's recorded room on the missing sheet")
        xs = [s for s in site["sources"] if not s["documented"]][0]
        if m.in_part(xs["pos"][0], xs["pos"][1]) or m.room_missing(xs["room"]):
            bad.append("the unlisted source on the missing part")
        if m.in_part(0.0, 0.0):
            bad.append("the pad on the missing part")
        if m.room_missing(site["base_station_room"] if "base_station_room" in site else "A1"):
            bad.append("the airlock on the missing sheet")
    rows["sheet_rules"] = _row(m is not None and not bad, bad,
                               "the C-sealed room, the relocated item's recorded room and the unlisted source are drawn "
                               "(its 5 m stretch on drawn zones: row X_zone)")
    # ---- sheet_nothing_drawn
    errs = sheet_nothing_drawn(site, ctx.prior)
    rows["sheet_nothing_drawn"] = _row(m is not None and not errs, errs[:8],
                                       "the documents hold nothing of the missing part; every drawn wall bounds a drawn "
                                       "room; cut corridors carry continues")
    # ---- sheet_edges: no door opening, hatch or fire door within 0.3 m of a corridor cut
    near = []
    if m is not None:
        clear = float(ctx.L["v2_params"]["sheet"]["cut_clear_m"])
        for c in m.cuts:
            for d in site["doors"]:
                if c["leg"] not in d.get("rooms", []):
                    continue
                if d.get("kind") == "fire":
                    a, b = float(d["coord"]) - 0.06, float(d["coord"]) + 0.06
                else:
                    w = ctx.walls[d["wall"]]
                    if (c["axis"] == "x") == (w["axis"] == "y"):
                        a, b = d["center"] - d["width"] / 2, d["center"] + d["width"] / 2
                    else:
                        a, b = w["coord"] - 0.15, w["coord"] + 0.15
                if a - clear + 1e-6 < c["at"] < b + clear - 1e-6:     # exactly `clear` away is clear (as the cut's
                    # choice, site_gen.sheet.compute, takes it; the stored values are rounded)
                    near.append("%s %s at the cut of %s" % (d.get("kind"), d["id"], c["leg"]))
    rows["sheet_edges"] = _row(m is not None and not near, {"cuts": [(c["leg"], c["at"]) for c in (m.cuts if m else [])],
                                                          "near": near},
                               "no door opening, hatch or fire door within 0.3 m of a corridor cut")
    # ---- outline_walls
    rows["outline_walls"] = _row(*_outline_walls(ctx))
    # ---- approach_flat and the sizes of the in-room obstacles
    rows["approach_flat"] = _approach_row(ctx)
    # ---- B-low_back: in the rack variant the reading place lies behind the rack
    rows["B-low_back"] = _b_low_back_row(ctx)
    # ---- variants: the factors are the table's
    from .variants import table_row, assignment_v2
    V2 = L.get("v2") or {}
    exp = table_row(int(V2.get("k", 1)))
    kerb = ctx.elements[L["b_kerb_elements"][0]]
    have = {"bund": "low" if kerb["class"] == "needs_high_step" else "high",
            "b_step": "in_room" if any(o["slot"] == "B-step" for o in L.get("v2_obstacles", [])) else "door_sills",
            "b_low": "rack" if any(o["slot"] == "B-low" for o in L.get("v2_obstacles", [])) else "chase",
            "sheet_place": m.place if m else None}
    want = {k: exp[k] for k in have}
    asg = L.get("assignment") or {}
    rows["variants"] = _row(have == want and asg == assignment_v2(int(V2.get("k", 1)), exp["b_low"]),
                            {"have": have, "table": want, "k": V2.get("k")},
                            "the bund class, the B-step and B-low variants, the missing sheet's place and the type "
                            "assignment are the table's for the candidate number")
    return rows


def _b_low_back_row(ctx):
    """In the rack variant B-low's reachable margin region behind the bridge (body
    centres at least variants.RACK_READ_CLEAR_M = 0.30 m beyond its far edge) is at least ladder.margin.min_area_m2, it
    has standing cells there, no margin cell (either posture) lies nearer the bridge than that, and the stored
    observation pose (the reading pose) stands there too. The chase variant: not applicable (ok)."""
    from .variants import RACK_READ_CLEAR_M
    back = b_low_back_mask(ctx)
    if back is None:
        return _row(True, {"variant": "chase"}, "B-low in the chase: not applicable")
    rec = ctx.by_slot["B-low"]
    R = ctx.regions[rec["tag"]]
    st, cr = ctx.stand_crouch(SLOT_MODES["B-low"])
    ms, mc = R["stand"] & st & back, R["crouch"] & cr & back
    area = float((ms | mc).sum()) * ctx.res ** 2
    out_st, out_cr = int((R["stand"] & ~back).sum()), int((R["crouch"] & ~back).sum())
    pose = rec.get("observation_pose") or {}
    p = pose.get("pos")
    pose_ok = False
    if p is not None:
        iy, ix = ctx.grid.cell(float(p[0]), float(p[1]))
        pose_ok = bool(0 <= iy < ctx.grid.ny and 0 <= ix < ctx.grid.nx and back[iy, ix])
    area_min = float(ctx.L["margin"]["min_area_m2"])
    ok = area >= area_min and bool(ms.any()) and pose_ok and out_st == 0 and out_cr == 0
    return _row(ok, {"area_m2": round(area, 3), "stand_cells": int(ms.sum()), "crouch_cells": int(mc.sum()),
                     "margin_cells_in_front": out_st + out_cr, "pose": [round(float(v), 3) for v in p] if p else None,
                     "pose_mode": pose.get("mode"), "pose_behind": pose_ok, "clear_m": RACK_READ_CLEAR_M},
                "rack variant: B-low's reachable margin region at least %.2f m beyond the bridge's far edge (the whole "
                "body behind it) >= %.2f m2 with standing cells, no margin cell nearer the bridge, the reading pose "
                "there" % (RACK_READ_CLEAR_M, area_min))


def _plate_rows(ctx, n_cells=150):
    """Version 2 (text plates): plate_readable (every sampled margin cell of every listed instrument, standing and
    crouching, gives a frame aimed at its text plate with the plate whole, its grid clear, cap and pitch at least
    1.3 x the legible size, at most 55 deg off the plate: the margin region holds it by construction, this re-checks it
    on the compiled site), plates_clear (the same grid, the plate of every listed instrument) and tags (every plate's
    text unique in the building, inside the grammar, at most 8 characters)."""
    from .accept import plate_compiled, plate_coplanar_cached, plate_readable_cells, plate_sizes, camera_from_body
    rows = {}
    rng = np.random.default_rng(11)
    X, Y = ctx.grid.centers()
    bad, vals, blocked, coplanar = {}, {}, {}, {}
    for slot in INSTR:
        rec = ctx.by_slot[slot]
        R = ctx.regions.get(rec["tag"])
        pl = plate_compiled(ctx.model, ctx.data, rec["body"])
        if pl is None or R is None:
            bad[rec["tag"]] = "no text plate" if pl is None else "no region"
            continue
        eg = R["feat"]
        # no other surface within 2 mm of the plate's face plane over its outline (a flush seam, a joint)
        cop = plate_coplanar_cached(ctx.model, ctx.data, eg, pl)
        if cop:
            coplanar[rec["tag"]] = cop[:4]
            bad[rec["tag"]] = "coplanar: %s" % ", ".join(c["geom"] for c in cop[:3])
            blocked[rec["tag"]] = "coplanar"
        caps, pits, offs, n_bad, n = [], [], [], 0, 0
        for key, which in (("stand", "stand"), ("crouch", "crouch")):
            iy, ix = np.nonzero(R[key])
            if not len(iy):
                continue
            pick = rng.choice(len(iy), size=min(n_cells, len(iy)), replace=False)
            B = np.c_[X[iy[pick], ix[pick]], Y[iy[pick], ix[pick]]]
            hh = nominal_head(ctx.cfg, which)
            ok = plate_readable_cells(ctx.acc, ctx.caster, eg, pl, B, hh)
            Cp = camera_from_body(ctx.cfg, B, pl["c"], hh, "inspect")
            cap, pit, off, _ = plate_sizes(ctx.cfg, pl, Cp)
            caps.append(cap.min())
            pits.append(pit.min())
            offs.append(off.max())
            n_bad += int((~ok).sum())
            n += len(B)
        if n_bad and rec["tag"] not in coplanar:
            bad[rec["tag"]] = n_bad
        vals[rec["tag"]] = {"cells": n, "cap_px_min": round(float(min(caps)), 1) if caps else None,
                            "pitch_px_min": round(float(min(pits)), 1) if pits else None,
                            "off_axis_max": round(float(max(offs)), 1) if offs else None}
        from .accept import plate_grid
        g = ctx.acc.plate_rule.get("clear_grid", [12, 6])
        if n and pl is not None:
            # each posture's cells with its own head height (a crouching cell under a pipe nest or a rack, tested at
            # the standing height, looked through the pipes: 2026-11-16, 11 of 64 sub-seeds of one sweep candidate)
            nb = 0
            for key, which in (("stand", "stand"), ("crouch", "crouch")):
                iy, ix = np.nonzero(R[key])
                if not len(iy):
                    continue
                pick = rng.choice(len(iy), size=min(n_cells, len(iy)), replace=False)
                B = np.c_[X[iy[pick], ix[pick]], Y[iy[pick], ix[pick]]]
                Cp = camera_from_body(ctx.cfg, B, pl["c"], nominal_head(ctx.cfg, which), "inspect")
                clr = ctx.caster.clear(eg, Cp, pts=plate_grid(pl, int(g[0]), int(g[1]))).all(axis=1)
                nb += int((~clr).sum())
            if nb and rec["tag"] not in coplanar:
                blocked[rec["tag"]] = nb
    # the coplanar test for every other plate of the building: no decoy's plate flush with another surface either (version 1's
    # practice buildings had their lamp-panel decoys' plates on a switchgear door seam)
    from .accept import plate_coplanar, subtree
    for d in ctx.site.get("decoys", []):
        pl = plate_compiled(ctx.model, ctx.data, d["body"]) if d.get("body") else None
        if pl is None:
            continue
        cop = plate_coplanar(ctx.model, ctx.data, pl, own_bodies=subtree(ctx.model, ctx.body_id(d)))
        if cop:
            coplanar["decoy %s" % d["tag"]] = cop[:4]
            blocked["decoy %s" % d["tag"]] = "coplanar"
    rows["plate_readable"] = _row(not bad, {"per_instrument": vals, "refused": bad, "coplanar": coplanar},
                                  "from every sampled margin cell a frame aimed at the instrument's text plate is legible "
                                  "with the margin (1.3 x 14 px cap, 1.3 x 8 px pitch, <= 55 deg, whole, grid clear); "
                                  "no other surface within 2 mm of the plate's face plane over its outline (R4)")
    rows["plates_clear"] = _row(not blocked and not bad, {"blocked": blocked, "coplanar": coplanar},
                                "from the margin cells every point of the acceptance file's grid (edge to edge) over "
                                "every listed instrument's text plate is clear, and no plate of the building (listed "
                                "instruments and decoys) has another surface within 2 mm of its face plane over its "
                                "outline (R4)")
    from bricks.lettering import tag_ok
    texts = [r["tag"] for r in ctx.site["ips"] if r.get("slot") in INSTR] + [d["tag"] for d in ctx.site["decoys"]]
    dup = sorted({t for t in texts if texts.count(t) > 1})
    gram = [t for t in texts if not tag_ok(t)]
    from collections import Counter
    near = 0
    for a in texts:
        for b in texts:
            if a < b and len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
                near += 1
    rows["tags"] = _row(not dup and not gram, {"plates": len(texts), "repeated": dup, "outside_grammar": gram,
                                               "pairs_one_character_apart": near,
                                               "lengths": dict(Counter(len(t) for t in texts))},
                        "every plate's text unique in the building; listed and decoy tags inside the grammar, at most 8 "
                        "characters")
    return rows


def _outline_walls(ctx):
    """Outer walls stand on the outline (their inner face 0.125 m inside it) on every sheet; their only openings
    are the two exterior doors (kept shut) and the annex door. Returns (ok, value, detail)."""
    site = ctx.site
    bx0, by0, bx1, by1 = site["building"]["rect"]
    L = site["ladder"]
    annex, host = L["annex"], L["annex_host"]
    bad = []
    n = 0
    for w in site["walls"]:
        rs = w["rooms"]
        party = annex in rs and host in rs
        if "OUT" not in rs and not party:
            continue
        other = [r for r in rs if r not in ("OUT", annex)]
        if not party and (not other or other[0] in ("A1", annex)):
            continue                     # the airlock's and the annex's own walls
        n += 1
        c = float(w["coord"])
        if w["axis"] == "x":
            on_line = abs(c - bx0) < 1e-6 or abs(c - bx1) < 1e-6
            inner = w["t_pos"] if abs(c - bx0) < 1e-6 else w["t_neg"]
        else:
            on_line = abs(c - by0) < 1e-6 or abs(c - by1) < 1e-6
            inner = w["t_pos"] if abs(c - by0) < 1e-6 else w["t_neg"]
        if not on_line:
            bad.append("outer wall %s off the outline" % w["id"])
            continue
        if abs(float(inner) - 0.125) > 1e-6 or w["kind"] != "outer":
            bad.append("outer wall %s: kind %s, inner face %.3f m inside" % (w["id"], w["kind"], float(inner)))
        for did in w.get("openings", []):
            d = ctx.doors.get(did)
            if d is None:
                continue
            if d["kind"] == "stair" and d["state"] == "closed":
                continue
            if did == L["annex_door"]:
                continue
            bad.append("outer wall %s has opening %s (%s, %s)" % (w["id"], did, d["kind"], d["state"]))
    for w in site["walls"]:
        if "A1" in w["rooms"] and "OUT" in w["rooms"]:
            for did in w.get("openings", []):
                d = ctx.doors[did]
                if d["kind"] != "airlock_outer" or d["state"] != "closed":
                    bad.append("airlock wall %s opening %s" % (w["id"], did))
    return (not bad, {"outer_walls": n, "bad": bad[:6]},
            "outer walls on the outline (inner face 0.125 m inside it) on every sheet; their only openings the two "
            "exterior doors, kept shut, and the annex door")


def _approach_row(ctx):
    """Every in-room strip or rack reaches both walls (with it solid for every mode the back part is
    unreachable), has 1.0 m of flat floor before and behind it (no other element, puddle, narrow gap or solid
    geometry there but the walls and door leaves) and a size inside its rated band with the 15 % margin."""
    site, cfg, g = ctx.site, ctx.cfg, ctx.grid
    L = site["ladder"]
    obs = L.get("v2_obstacles", [])
    from .world import draw_interval
    X, Y = g.centers()
    bad = []
    vals = []
    for o in obs:
        e = ctx.elements.get(o.get("element"))
        if e is None:
            bad.append("no element for %s" % o["slot"])
            continue
        z = o["zones"]
        # size and placement
        if o["slot"] == "B-step":
            v = float(e["param"].get("height", e["param"].get("diameter", 0.0)))
            lo, hi = draw_interval(cfg, e["kind"], "needs_high_step")
            if e["class"] != "needs_high_step" or not (lo - 1e-4 <= v <= hi + 1e-4):
                bad.append("strip %s %s %.4f outside %.4f-%.4f" % (e["id"], e["kind"], v, lo, hi))
            n0, n1 = [float(x) for x in ctx.L["v2_params"]["strip"]["near_m"]]
            if not (n0 - 1e-6 <= float(o["near_m"]) <= n1 + 1e-6):
                bad.append("strip near edge %.2f m" % float(o["near_m"]))
        else:
            v = float(e["param"]["clearance"])
            lo, hi = draw_interval(cfg, "low_overhead", "needs_crouch")
            l0, l1 = [float(x) for x in ctx.L["v2_params"]["rack"]["length_m"]]
            if e["class"] != "needs_crouch" or not (lo - 1e-4 <= v <= hi + 1e-4) or \
                    not (l0 - 1e-6 <= float(e["param"]["length"]) <= l1 + 1e-6):
                bad.append("rack %s clearance %.4f / length %.2f" % (e["id"], v, float(e["param"]["length"])))
        vals.append({"slot": o["slot"], "kind": e["kind"], "size_m": round(v, 4), "near_m": o["near_m"]})
        # the approaches: flat floor
        room = ctx.rooms[o["room"]]
        ix0, iy0, ix1, iy1 = room["inner"]
        for nm in ("front", "behind"):
            r = z[nm]
            zone = (X >= r[0]) & (X <= r[2]) & (Y >= r[1]) & (Y <= r[3])
            # away from the walls (0.25 m) along the strip
            zone &= (X >= ix0 + 0.25) & (X <= ix1 - 0.25) & (Y >= iy0 + 0.25) & (Y <= iy1 - 0.25)
            for f in site["elements"]:
                if f["id"] == e["id"]:
                    continue
                P = np.asarray(f["footprint"], float)
                fr = [P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()]
                if fr[0] < r[2] - 0.01 and r[0] < fr[2] - 0.01 and fr[1] < r[3] - 0.01 and r[1] < fr[3] - 0.01:
                    bad.append("%s approach of %s holds element %s (%s)" % (nm, o["slot"], f["id"], f["kind"]))
            del zone
            hit = _geoms_in_box(ctx, [max(r[0], ix0 + 0.25), max(r[1], iy0 + 0.25), min(r[2], ix1 - 0.25),
                                      min(r[3], iy1 - 0.25)], 0.004, 0.5, skip=set(e.get("geoms", [])), room=o["room"])
            if hit:
                bad.append("%s approach of %s not flat: %s" % (nm, o["slot"], hit[:3]))
        # the strip / bridge from wall to wall: with it solid for every mode no cell of the back part is reachable
        cut = ctx.reach_cells(MODES, ctx.elem_mask([e["id"]]), ("v2cut", e["id"]))
        bp = z["back_part"]
        back = (X >= bp[0] + 0.3) & (X <= bp[2] - 0.3) & (Y >= bp[1] + 0.3) & (Y <= bp[3] - 0.3)
        lost = int((cut & back).sum())
        if lost:
            bad.append("the back part of %s is reachable around %s (%d cells)" % (o["room"], e["id"], lost))
    return _row(not bad, {"obstacles": vals, "bad": bad[:6]},
                "every in-room strip or rack spans wall to wall, has 1.0 m of flat floor before and behind it and a "
                "rated size", required=True)


def _geoms_in_box(ctx, rect, z0, z1, skip=(), room=None):
    """Names of the collision geoms (groups 0 and 3) of the compiled site whose world box meets rect x [z0, z1],
    leaving out the walls, the floor, `skip` and the leaves of `room`'s doors (they stand at the door, not on the
    way)."""
    m, d = ctx.model, ctx.data
    leaves = tuple("door/%s/leaf" % dd["id"] for dd in ctx.site["doors"] if room in dd.get("rooms", []))
    out = []
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return out
    for i in range(m.ngeom):
        if m.geom_group[i] not in (0, 3) or (m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0):
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm in skip or nm == "floor" or nm.startswith(("wall/", "floor/", "robot/")) or nm.startswith(leaves):
            continue
        if int(m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        a = np.asarray(m.geom_aabb[i])
        c = d.geom_xpos[i] + d.geom_xmat[i].reshape(3, 3) @ a[:3]
        ext = np.abs(d.geom_xmat[i].reshape(3, 3)) @ a[3:]
        if c[0] + ext[0] <= rect[0] or c[0] - ext[0] >= rect[2] or c[1] + ext[1] <= rect[1] or \
                c[1] - ext[1] >= rect[3] or c[2] + ext[2] <= z0 or c[2] - ext[2] >= z1:
            continue
        out.append(nm)
    return out


def _drawing_aim_row(ctx, max_cells=60000, los=False, hts="range"):
    """From every cell the drawing shows reachable and the robot's camera can reach, within the evidence
    distance of a C-moved / C-relocated item's drawn place, a frame of the item's camera aimed at the drawn place, at
    every camera height of the generous region (7 from 0.145 to 0.375 m): the exact evidence test of the TRUE
    instrument must refuse every one (so a program that photographs the drawn places does not score the item by
    accident). Whether the drawing shows the drawn place in view from the cell does not matter (a review's
    drawing follower, which found 581 such cells on an earlier candidate, has no such filter; los=True restores the
    filter, hts="nominal" the two nominal heights, for comparison)."""
    en, reach = drawn_map(ctx)
    solid = ctx.drawn_solid
    cam_cells = ctx.camera_cells()
    g = ctx.grid
    X, Y = g.centers()
    hits = {}
    tested = 0
    for slot in ("C-moved", "C-relocated"):
        rec = ctx.by_slot[slot]
        loc = None
        for it in ctx.prior["round_sheet"]["items"]:
            if it["tag"] == rec["tag"]:
                loc = it["location"]
        if not loc or not loc.get("pos"):
            continue
        # aimed where a program looks for the face: the recorded position, a rising-stem valve's plate 0.42 m above it
        from .ip_models import face_dz
        p = np.array([loc["pos"][0], loc["pos"][1], float(loc.get("height") or 1.0) + face_dz(rec["type"])])
        eg = ctx.regions[rec["tag"]]["feat"]
        src = reach
        if ctx.miss is not None:
            # version 2: on the missing part every cell the robot's camera can reach counts (nothing drawn
            # there tells a program where not to look)
            src = reach | (ctx.part_mask() & cam_cells)
        sel = src & cam_cells & ((X - p[0]) ** 2 + (Y - p[1]) ** 2 <= (ctx.acc.max_dist + 0.05) ** 2)
        iy, ix = np.nonzero(sel)
        if not len(iy):
            continue
        if len(iy) > max_cells:
            k = np.random.default_rng(3).choice(len(iy), max_cells, replace=False)
            iy, ix = iy[k], ix[k]
        # the drawn place in view on the drawing (the segment to 0.25 m short of it crosses no drawn solid)
        C2 = np.c_[X[iy, ix], Y[iy, ix]]
        d2 = np.linalg.norm(C2 - p[None, :2], axis=1)
        t = np.linspace(0.0, 1.0, 48)[None, :]
        stop = np.clip(1.0 - 0.25 / np.maximum(d2, 1e-9), 0.0, 1.0)[:, None]
        sx = C2[:, 0:1] + (p[0] - C2[:, 0:1]) * t * stop
        sy = C2[:, 1:2] + (p[1] - C2[:, 1:2]) * t * stop
        jx = np.clip(np.floor((sx - g.x0) / g.res).astype(int), 0, g.nx - 1)
        jy = np.clip(np.floor((sy - g.y0) / g.res).astype(int), 0, g.ny - 1)
        seen = ~solid[jy, jx].any(axis=1)
        if los:
            C2 = C2[seen]
        n_ok = 0
        from .accept import look_at, heights as cam_heights
        hl = (nominal_head(ctx.cfg, "stand"), nominal_head(ctx.cfg, "crouch")) if hts == "nominal" else \
            tuple(cam_heights(ctx.cfg, "range"))
        for h in hl:
            C = np.c_[C2, np.full(len(C2), h)]
            if not len(C):
                continue
            T = look_at(C, p)
            tested += len(T)
            ones = np.ones((len(T), len(eg.spts)), bool)
            geo = ctx.acc.evaluate(eg, T, ctx.caster, vis=ones)["ok"]       # every condition but visibility
            k = np.nonzero(geo)[0]
            if len(k):
                n_ok += int(ctx.acc.evaluate(eg, T[k], ctx.caster)["ok"].sum())
        if n_ok:
            hits[rec["tag"]] = n_ok
    return _row(not hits, {"frames_tested": tested, "accepted_true_instrument": hits},
                "a frame aimed at a C-moved / C-relocated item's drawn place, from any drawn-reachable camera cell "
                "within the evidence distance and at any camera height, never gives an accepted frame of the true "
                "instrument")


def _zoning_row(ctx):
    """Every zone the drawing classes supervised or controlled keeps the documented sources' rate inside
    its class on all reachable floor (the old survey is right about what it surveyed)."""
    classes = {c: float(b) for c, b in ctx.cfg["radiation"]["zone_classes"]}
    reach = ctx.reach_cells(MODES)
    X, Y = ctx.grid.centers()
    doc = [s["id"] for s in ctx.site["sources"] if s["documented"]]
    worst, bad = 0.0, []
    for z in ctx.prior["zoning"]["zones"]:
        bound = classes.get(z["zone_class"])
        if bound is None or bound > 1e8:
            continue
        xs = [q[0] for q in z["polygon"]]
        ys = [q[1] for q in z["polygon"]]
        m = reach & (X >= min(xs)) & (X <= max(xs)) & (Y >= min(ys)) & (Y <= max(ys))
        if not m.any():
            continue
        r = float(ctx.rate(np.stack([X[m], Y[m]], 1), only=doc).max() +
                  float(ctx.site["radiation"]["background_uSv_h"]))
        worst = max(worst, r / bound)
        if r > bound + 1e-9:
            bad.append({"zone": z["label"], "class": z["zone_class"], "rate_gyh": round(
                r * float(ctx.L["dose"]["unit_factor"]), 4)})
    return _row(not bad, {"worst_share_of_bound": round(worst, 4), "over": bad[:4]},
                "the drawn zoning holds for the documented sources on every reachable cell")


# ============================================================================================ leak check
def leak_errors(site, prior):
    """prior.json must hold nothing that only the ground truth knows (the first layout's checks plus the ladder's fields)."""
    from .prior import GT_ONLY_KEYS
    errs = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield path + "/" + k, k, v
                yield from walk(v, path + "/" + k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from walk(v, path + "[%d]" % i)
    for path, k, v in walk(prior):
        if k in GT_ONLY_KEYS or k in ("slot", "ladder_slot", "ladder", "acceptance_region", "crossing"):
            errs.append("gt-only key %s" % path)
    txt = json.dumps(prior)
    for word in ("B-step", "B-kerb", "B-low", "C-sealed", "C-moved", "C-relocated", "C-annex"):
        if word in txt:
            errs.append("slot label %s in prior" % word)
    rs = prior["round_sheet"]["items"]
    tags = [it["tag"] for it in rs if it.get("kind") != "unlisted_source"]
    if tags != sorted(tags):
        errs.append("round sheet not in tag order")
    num = {r["id"]: r["number"] for r in site["rooms"]}
    by = {r["tag"]: r for r in site["ips"]}
    drawn_nums = {r["number"] for r in prior["rooms"]}
    if (site.get("ladder") or {}).get("v2"):
        # version 2: the unrecorded instrument's tag is numbered from a room of the original drawing (the
        # missing sheet's rooms included: its number is random among them and tells nothing)
        drawn_nums = {r["number"] for r in site["rooms"] if r["on_drawing"]}
    for it in rs:
        rec = by.get(it["tag"])
        if rec is None:
            if it.get("kind") != "unlisted_source":
                errs.append("round sheet item %s not in the site" % it["tag"])
            continue
        if rec["slot"] in ("M", "X"):
            continue
        digits = "".join(ch for ch in it["tag"].split("-", 1)[1] if ch.isdigit())
        loc = it["location"]
        if loc.get("recorded"):
            if not digits.startswith("".join(ch for ch in loc["room"] if ch.isdigit())):
                errs.append("%s: tag not numbered from its drawn room" % it["tag"])
        else:
            if not any(digits.startswith(n) for n in drawn_nums) or digits.startswith(num[rec["room"]]):
                errs.append("%s: tag of an unrecorded instrument tells its room" % it["tag"])
        for k in ("value", "angle_deg", "opening_pct", "level_pct", "states", "temp_C"):
            if k in json.dumps(it):
                errs.append("%s: truth key %s" % (it["tag"], k))
        if rec["slot"] in ("C-moved", "C-relocated") and loc.get("pos"):
            if math.hypot(loc["pos"][0] - rec["pose"]["pos"][0], loc["pos"][1] - rec["pose"]["pos"][1]) < 1.0:
                errs.append("%s: the true position is given" % it["tag"])
    us = [it for it in rs if it.get("kind") == "unlisted_source"]
    if len(us) != 1 or any(k in us[0] for k in ("location", "pos", "room")):
        errs.append("the unlisted source must be one line without a location")
    if (site.get("ladder") or {}).get("v2"):
        # version 2: no label code, marker id or code table for any instrument in the documents (grid
        # markers keep theirs)
        for sec in ("round_sheet", "work_order"):
            txt_ = json.dumps(prior.get(sec, {}))
            for word in ("label_code", "marker_id", "tag_codes", "36h11"):
                if word in txt_:
                    errs.append("%s holds %s" % (sec, word))
    truth = {(d["kind_log"], d["room_number"]) for d in site["ladder"].get("mod_log_truth", [])}
    for e in prior.get("modification_log", {}).get("entries", []):
        if (e["kind"], e["room"]) not in truth:
            errs.append("modification log entry not true: %s %s" % (e["kind"], e["room"]))
    for z in prior["zoning"]["zones"]:
        if z["zone_class"] not in ("supervised", "controlled", "high_radiation"):
            errs.append("zone class")
    xs = [s for s in site["sources"] if not s["documented"]]
    struct = {k: prior[k] for k in ("rooms", "walls", "doors", "equipment", "building", "grid", "zoning",
                                    "grid_markers") if k in prior}
    other = {k: v for k, v in prior.items() if k not in struct}
    for s in xs:
        for pts, tol, what in ((list(_points(other)), 0.5, "documents"), (list(_points(struct)), 0.02, "drawing")):
            if any(math.hypot(p[0] - s["pos"][0], p[1] - s["pos"][1]) < tol for p in pts):
                errs.append("a %s coordinate locates the unlisted source" % what)
    return errs


def _points(x):
    if isinstance(x, list) and len(x) >= 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                   for v in x[:2]):
        yield (float(x[0]), float(x[1]))
    if isinstance(x, dict):
        for v in x.values():
            yield from _points(v)
    elif isinstance(x, list):
        for v in x:
            yield from _points(v)
