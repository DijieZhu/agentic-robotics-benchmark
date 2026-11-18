"""The grader-side event log of one run (cell side; DESIGN v1.2 sections 3.2-3.3; package P09).

Scoring (subseacell.scoring, P10) reads only the terminal state and this log; nothing the program reports enters it.
Three parts, all functions of the site, the seed and the program's calls (no wall-clock value anywhere):

  series   every 0.1 s (the world's slow tick: process graph, plume, valve history), one row of
           float columns COLUMNS_F: t, the 18 valve hinge angles [deg] (order `valve_names` = BV1..BV6, PV1..PV6,
             TV1..TV6), vehicle / marker / debris / stab positions, vehicle quaternion, linear speeds, leak-segment
             pressure, battery state of charge and voltage
           int columns COLUMNS_I: conduction mask (bit k = valve k conducting, angle < 80 deg), stab state (STAB_STATES)
             and port (index in `port_names`, -1 = none), stab has left its initial port, raw gripper state
             (GRIPPER_RAW), held kind (HELD_KINDS) and name (index in `held_names`, -1 = none), protective stops,
             protection phase (PROTECT_PHASES), firmware mode (FW_MODES) and degraded flag, vehicle in the cage,
             marker touching the roof grating, debris touching the vehicle, frames taken (640, 320), pings
  frames   one record per camera frame: index, t, size, lamp level, tilt, camera pose, and per tag plate within
           RECORD_RANGE_M of the camera the raw quantities of the "imaged" predicate (DESIGN 3.2) and its verdict;
           `imaged` = the plates for which the predicate holds (valve tags BV1.., port names TP-1..)
  events   the world's discrete events in the order they happened: grasp (P05: grasp / seated / slip / release /
           window_miss / stall_no_grasp / eye_miss), stab (P07: join / miss / lock / unlock / free), protect (P04:
           protective_stop / vehicle_disabled / collision_during_stop / out_of_bounds / undervoltage), process
           (conduction changes), world (debris_sleep / debris_wake), plus this module's: contact (begin / end of a
           contact between classes, sampled every 0.1 s), camera (budget refusals), api (call counts are kept, not
           logged; logged: gripper, lights, tilt, nav fix, sonar scans, finish, forbidden calls, bad parameters,
           mode changes and refusals of the motion calls), cell (end of the run)

The "imaged" predicate (DESIGN 3.2, thresholds `score.imaged.*` of spec/task_values.json): a frame images plate k if
the four outer corners of its code (the 160 mm black frame) are inside the image with an edge_margin_px margin, the
slant range camera -> code centre <= slant_max_m, the shortest projected edge of the code's 8 x 8 cells (border
included) >= cell_min_px, the angle between the line of sight and the plate normal <= angle_max_deg, >= unoccluded_min
of the code's 5 x 5 sample points are visible from the camera (rays on the render twin, geom groups 0 and 2: the
scene, the gripper jaws and held objects occlude; fouling (group 4) and the vehicle hull do not), the lamp level >=
light_level_min and the code centre lies within half the lamp's full beam angle of at least one lamp.  Geometry from
the site's true geometry (the twin's kinematics at the frame), never from the image.  Port of proto/g4_optics/predicate.py
(the gate G4 definitions: projection with the camera's own pose and focal length, cell edges of the (M+1)^2 corner
grid, sample points 1.1 mm in front of the code face).

    log = EventLog(world, values)            # attaches to the world (slow tick + frame hook)
    log.api("gripper_close", accepted=True)  # the server's records
    log.export(); log.to_bytes(); log.digest(); log.save(dir); EventLog.load(dir)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Optional

import numpy as np
import mujoco

from .objects import in_cage
from .twin import QUAD_OFFSET_M

SCHEMA = "subseacell-events/1"
RECORD_RANGE_M = 2.0
SAMPLE_FRONT_M = 0.0011                 # occlusion sample points in front of the code face (G4)
CONTACT_END_S = 0.5                     # a contact ends after this long unseen (sampled every 0.1 s)
MAX_CONTACT_EVENTS = 6000
MAX_API_EVENTS = 20000
MAX_SMALL_EVENTS = 400                  # forbidden / bad_params events logged (counted beyond)

STAB_STATES = ("Free", "Joined", "Locked")
GRIPPER_RAW = ("open", "closing", "opening", "closed_empty", "stalled_on_object")
HELD_KINDS = ("none", "paddle", "eye")
PROTECT_PHASES = (None, "stop", "hold", "surfacing", "disabled")
FW_MODES = ("idle", "attitude", "velocity", "station", "wrench")
MOVING = ("vehicle", "jaws", "marker", "stab", "debris")

COLUMNS_F_BASE = ("t",)
COLUMNS_F_TAIL = ("hull_x", "hull_y", "hull_z", "hull_qw", "hull_qx", "hull_qy", "hull_qz", "hull_speed",
                  "marker_x", "marker_y", "marker_z", "marker_speed", "debris_x", "debris_y", "debris_z",
                  "debris_speed", "stab_x", "stab_y", "stab_z", "leak_pressure_bar", "battery_soc", "battery_V")
COLUMNS_I = ("conducting_mask", "stab_state", "stab_port", "stab_left_initial", "gripper_state", "held_kind",
             "held_name", "protect_count", "protect_phase", "fw_mode", "fw_degraded", "in_cage", "marker_on_roof",
             "debris_touch_vehicle", "frames_640", "frames_320", "pings")


def _r(x, n=6):
    return float(round(float(x), n))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def canonical(obj) -> bytes:
    return (json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n").encode()


# ======================================================================================================= the predicate
def imaged_thresholds(values: dict, lamp_half_angle_deg: float) -> dict:
    """The predicate's thresholds from the flat task values (`score.imaged.*`) and the lamps' half beam angle."""
    s = values["score.imaged.samples"]
    return dict(margin_px=float(values["score.imaged.edge_margin_px"]), max_range_m=float(values["score.imaged.slant_max_m"]),
                min_cell_px=float(values["score.imaged.cell_min_px"]), max_angle_deg=float(values["score.imaged.angle_max_deg"]),
                min_lamp=float(values["score.imaged.light_level_min"]), lamp_half_angle_deg=float(lamp_half_angle_deg),
                grid=[int(s["grid"][0]), int(s["grid"][1])], min_visible=int(s["unoccluded_min"]))


def code_layout(values: dict) -> dict:
    """The code square on a tag plate (plate frame: origin at the plate centre, X right, Y up; metres): DESIGN 1.2 /
    the panel standard (424 x 200 mm plate, quiet zone 20 mm, outer frame 160 mm, 20 mm cells incl. a one-cell border).
    Equal to gen/tag_textures.layout() (tests/test_server.py)."""
    W, H = [float(x) for x in values["panel.tag.size_mm"]]
    q = float(values["code.quiet_zone_mm"])
    M = float(values["code.outer_mm"])
    cell = float(values["code.cell_mm"])
    n = int(values["code.data_cells"]) + 2 * int(values["code.border_cells"])
    if abs(n * cell - M) > 1e-9:
        raise ValueError("code cells do not add up to the outer frame")
    to_m = lambda x, y: ((x - W / 2.0) / 1000.0, (H / 2.0 - y) / 1000.0)  # noqa: E731
    x0, y0 = q, q
    return dict(n_cells=n, cell_m=cell / 1000.0, outer_m=M / 1000.0, centre=to_m(x0 + M / 2.0, y0 + M / 2.0),
                top_left=to_m(x0, y0),
                corners=[to_m(x0, y0), to_m(x0 + M, y0), to_m(x0 + M, y0 + M), to_m(x0, y0 + M)])


class ImagedJudge:
    """Per-frame evaluation of the "imaged" predicate on the render twin (see the module doc)."""

    def __init__(self, world, values: dict, front_offset_m: float):
        self.W = world
        tw = world.twin
        self.m = tw.model
        L = code_layout(values)
        self.L = L
        sd = world.sd
        by_key = {("stem", v["key"]): v["tag"] for v in sd["valves"]}
        by_key.update({("port", p["key"]): p["name"] for p in sd["ports"]})
        self.plates = []                     # (label, kind, code centre w, normal w, corners w (4,3), grid w, samples w)
        n = L["n_cells"]
        for name, pl in tw.plates.items():
            of, _, key = name.partition("_")
            label = by_key.get((of, key))
            if label is None:
                continue                             # the cage plate is not a panel position
            c, R = np.asarray(pl["centre_w"], float), np.asarray(pl["R_w"], float)
            zf = float(pl["front_z"]) + float(front_offset_m)
            to_w = lambda xy, z=zf: c + np.asarray(xy, float) @ np.array([R[:, 0], R[:, 1]]) + R[:, 2] * z  # noqa: E731
            corners = np.array([to_w(p) for p in L["corners"]])
            x0, y0 = L["top_left"]
            gx, gy = np.meshgrid(x0 + np.arange(n + 1) * L["cell_m"], y0 - np.arange(n + 1) * L["cell_m"])
            grid = to_w(np.stack([gx.reshape(-1), gy.reshape(-1)], 1))
            self.plates.append(dict(label=label, kind=of, centre=to_w(L["centre"]), normal=R[:, 2].copy(),
                                    corners=corners, grid=grid, grid_n=n + 1, R=R, c=c, zf=zf))
        self.hull = int(self.m.body("hull").id)
        self.groups = np.array([1, 0, 1, 0, 0, 0], np.uint8)
        self._values = values
        self.P = None

    def _samples(self, pl, grid):
        gx, gy = grid
        L = self.L
        cx, cy = L["centre"]
        sx = (np.arange(gx) + 0.5) / gx * L["outer_m"] - L["outer_m"] / 2.0
        sy = (np.arange(gy) + 0.5) / gy * L["outer_m"] - L["outer_m"] / 2.0
        X, Y = np.meshgrid(cx + sx, cy + sy)
        R, c = pl["R"], pl["c"]
        return (c[None, :] + X.reshape(-1, 1) * R[:, 0][None, :] + Y.reshape(-1, 1) * R[:, 1][None, :]
                + (pl["zf"] + SAMPLE_FRONT_M) * R[:, 2][None, :])

    def evaluate(self, cam, lamp_level: float, size) -> tuple:
        """(records {label: quantities}, imaged [labels]) for the current twin state."""
        if self.P is None:
            half = min(float(L.field_angle_deg) for L in cam.lamps) / 2.0
            self.P = imaged_thresholds(self._values, half)
        P = self.P
        d = self.W.twin.data
        p, C = cam.camera_pose(d)
        p = np.asarray(p, float)
        Wd, Hd = int(cam.cfg.width), int(cam.cfg.height)
        f = float(cam.f_render)
        lamps = [(d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3)[:, 0].copy()) for sid in cam.lamp_sids]
        recs, imaged = {}, []

        def proj(X):
            Pc = (np.atleast_2d(X) - p) @ C
            z = -Pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                return Wd / 2.0 + f * Pc[:, 0] / z, Hd / 2.0 - f * Pc[:, 1] / z, z

        for pl in self.plates:
            T = pl["centre"]
            rng = float(np.linalg.norm(p - T))
            if rng > RECORD_RANGE_M:
                continue
            u, v, z = proj(pl["corners"])
            in_front = bool(np.all(z > 0.05))
            margin = float(np.min(np.concatenate([u, Wd - u, v, Hd - v]))) if in_front else -1e9
            los = (p - T) / max(rng, 1e-12)
            ang = math.degrees(math.acos(float(np.clip(los @ pl["normal"], -1.0, 1.0))))
            gu, gv, gz = proj(pl["grid"])
            if np.all(gz > 0.05):
                k = pl["grid_n"]
                U = np.stack([gu, gv], 1).reshape(k, k, 2)
                e_h = np.linalg.norm(U[:, 1:] - U[:, :-1], axis=-1)
                e_v = np.linalg.norm(U[1:, :] - U[:-1, :], axis=-1)
                side = np.minimum(np.minimum(e_h[:-1, :], e_h[1:, :]), np.minimum(e_v[:, :-1], e_v[:, 1:]))
                cell = float(side.min())
            else:
                cell = 0.0
            lamp_deg = 180.0
            for lp, ax in lamps:
                w = T - lp
                lamp_deg = min(lamp_deg, math.degrees(math.acos(float(np.clip(ax @ w / np.linalg.norm(w), -1.0, 1.0)))))
            vis = None
            if in_front and rng <= P["max_range_m"]:
                vis = self._visible(p, pl)
            ok = (in_front and margin >= P["margin_px"] and rng <= P["max_range_m"] and cell >= P["min_cell_px"]
                  and ang <= P["max_angle_deg"] and lamp_level >= P["min_lamp"] and lamp_deg <= P["lamp_half_angle_deg"]
                  and vis is not None and vis >= P["min_visible"])
            recs[pl["label"]] = dict(in_front=in_front, margin_px=_r(margin, 3), range_m=_r(rng, 5),
                                     angle_deg=_r(ang, 3), cell_px=_r(cell, 4), lamp_deg=_r(lamp_deg, 3),
                                     visible=vis, imaged=bool(ok))
            if ok:
                imaged.append(pl["label"])
        return recs, imaged

    def _visible(self, p, pl):
        S = pl.get("_samples")
        if S is None:
            S = pl["_samples"] = self._samples(pl, self.P["grid"])
        vec = S - p[None, :]
        dist = np.linalg.norm(vec, axis=1)
        dirs = np.ascontiguousarray(vec / dist[:, None]).reshape(-1)
        n = S.shape[0]
        gid = np.zeros(n, np.int32)
        out = np.zeros(n, np.float64)
        mujoco.mj_multiRay(self.m, self.W.twin.data, p, dirs, self.groups, 1, self.hull, gid, out, None, n,
                           float(dist.max()) + 0.05)
        vis = (out < 0) | (out >= dist - 0.004)
        return int(np.count_nonzero(vis))


# ======================================================================================================= contacts
class ContactWatch:
    """Contacts between object classes, sampled at the slow tick (every 0.1 s): begin / end events with the peak normal
    force seen while sampled, and the instantaneous flags of the series (marker on the roof grating, debris touching
    the vehicle).  Only pairs with a moving object (MOVING) are tracked."""

    def __init__(self, world):
        self.W = world
        m = world.m
        self.m = m
        hull = m.body("hull").id
        root = int(m.body_rootid[hull])
        valve_bodies = {inf["body"]: inf.get("name", inf["body"]) for inf in world.panel["valves"].values()}
        panel_geoms = {n for names in world.panel["geoms"].values() for n in names}
        roles = {}
        for role, names in getattr(world, "manifold_geoms", {}).items():
            for n in names:
                roles[n] = role
        scenery = set(world.scenery_names()) if hasattr(world, "scenery_names") else set()   # amendment 02
        cls, det = [], []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            bn = m.body(b).name
            gn = m.geom(g).name
            if int(m.body_rootid[b]) == root:
                c = "jaws" if bn in ("jaw_upper", "jaw_lower") else "vehicle"
            elif gn in scenery:
                c = "scenery"
            elif bn in valve_bodies:
                c = "valve"
                det.append((g, valve_bodies[bn]))
            elif bn in ("marker", "stab", "debris", "cage"):
                c = bn
            elif bn.startswith("stabslide_"):
                c = "port"
            elif b == 0:
                r = roles.get(gn)
                if gn in panel_geoms:
                    c = "panel"
                elif r == "roof_proxy":
                    c = "roof"
                elif r == "mudmat":
                    c = "mudmat"
                elif r is not None:
                    c = "structure"
                elif gn.startswith("cage"):
                    c = "cage"
                else:
                    c = "seabed"
            else:
                c = "other"
            cls.append(c)
        names = sorted(set(cls))
        self.names = names
        self.code = np.array([names.index(c) for c in cls], np.int64)
        self.detail = dict(det)
        self.moving = {names.index(c) for c in MOVING if c in names}
        self.i_marker = names.index("marker") if "marker" in names else -1
        self.i_roof = names.index("roof") if "roof" in names else -1
        self.i_debris = names.index("debris") if "debris" in names else -1
        self.i_veh = {names.index(c) for c in ("vehicle", "jaws") if c in names}
        self.active = {}                 # key -> dict(t0, peak, last_seen)
        self.f6 = np.zeros(6)
        self.n_events = 0
        self.dropped = 0
        self.marker_on_roof = False
        self.debris_touch_vehicle = False

    def sample(self, t: float, events: list):
        m, d = self.m, self.W.d
        n = int(d.ncon)
        seen = {}
        mk_roof = deb_veh = False
        if n:
            G = np.asarray(d.contact.geom[:n])
            A, B = self.code[G[:, 0]], self.code[G[:, 1]]
            for i in range(n):
                a, b = int(A[i]), int(B[i])
                if a not in self.moving and b not in self.moving:
                    continue
                if a == b and a in self.moving:
                    continue
                ga, gb = int(G[i, 0]), int(G[i, 1])
                if a > b:
                    a, b, ga, gb = b, a, gb, ga
                key = (a, b, self.detail.get(ga) or self.detail.get(gb))
                mujoco.mj_contactForce(m, d, i, self.f6)
                F = abs(float(self.f6[0]))
                seen[key] = max(seen.get(key, 0.0), F)
                if {a, b} == {self.i_marker, self.i_roof}:
                    mk_roof = True
                if (a == self.i_debris and b in self.i_veh) or (b == self.i_debris and a in self.i_veh):
                    deb_veh = True
        self.marker_on_roof, self.debris_touch_vehicle = mk_roof, deb_veh
        for key, F in seen.items():
            st = self.active.get(key)
            if st is None:
                self.active[key] = dict(t0=t, peak=F, last=t)
                self._emit(events, dict(t=t, src="contact", ev="begin", a=self.names[key[0]], b=self.names[key[1]],
                                        tag=key[2], F_N=_r(F, 3)))
            else:
                st["peak"] = max(st["peak"], F)
                st["last"] = t
        for key in list(self.active):
            st = self.active[key]
            if key not in seen and t - st["last"] >= CONTACT_END_S - 1e-9:
                del self.active[key]
                self._emit(events, dict(t=t, src="contact", ev="end", a=self.names[key[0]], b=self.names[key[1]],
                                        tag=key[2], t_begin=_r(st["t0"], 4), peak_F_N=_r(st["peak"], 3)))

    def _emit(self, events, e):
        if self.n_events >= MAX_CONTACT_EVENTS:
            self.dropped += 1
            return
        if e.get("tag") is None:
            e.pop("tag")
        events.append(e)
        self.n_events += 1

    def open_contacts(self) -> list:
        return [dict(a=self.names[k[0]], b=self.names[k[1]], tag=k[2], t_begin=_r(s["t0"], 4), peak_F_N=_r(s["peak"], 3))
                for k, s in sorted(self.active.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2])))]


# ======================================================================================================= the log
class EventLog:
    def __init__(self, world, values: dict, *, judge: bool = True):
        self.W = world
        self.values = values
        sd = world.sd
        self.valve_names = list(world.valve_names)
        self.port_names = [p["name"] for p in sd["ports"]]
        self.held_names = ["marker", "stab"] + [v["tag"] for v in sd["valves"]] + sorted(getattr(world, "eyes", {}))
        self.columns_f = COLUMNS_F_BASE + tuple(f"valve_{n}_deg" for n in self.valve_names) + COLUMNS_F_TAIL
        self.columns_i = COLUMNS_I
        self.rows_f, self.rows_i = [], []
        self.frames = []
        self.small = {"forbidden": 0, "bad_params": 0}
        self.n_api = 0
        self.api_dropped = 0
        self._last_mode = None
        m = world.m
        self._bid = {n: m.body(n).id for n in ("hull", "marker", "debris", "stab")}
        self.contacts = ContactWatch(world)
        self.judge = ImagedJudge(world, values, QUAD_OFFSET_M) if (judge and getattr(world, "twin", None) is not None) else None
        if self.judge is not None:
            # the predicate's thresholds are written when the log is built (P24a, decision P18-4), from the same lamps the
            # judge would read at the first frame: a run without a single frame carries them too
            cams = getattr(world, "cams", None) or {}
            cam = cams.get((640, 480)) or (next(iter(cams.values())) if cams else None)
            if cam is not None:
                self.judge.P = imaged_thresholds(values, min(float(L.field_angle_deg) for L in cam.lamps) / 2.0)
        self._orig_slow = world._slow_tick
        world._slow_tick = self._slow_tick
        world.frame_hooks.append(self._on_frame)
        self.sample()

    # ------------------------------------------------------------------------------------------ hooks
    def _slow_tick(self):
        self._orig_slow()
        self.sample()

    def sample(self):
        W = self.W
        d, m = W.d, W.m
        t = float(W.sim.time)
        self.contacts.sample(t, W.events)
        ang = W.angle_hist[-1]
        mask = 0
        for k, a in enumerate(ang):
            if a < 80.0:
                mask |= 1 << k
        row = [t] + [float(a) for a in ang]
        hb = self._bid["hull"]
        row += [float(x) for x in d.xpos[hb]] + [float(x) for x in d.xquat[hb]]
        v6 = np.zeros(6)
        for nm in ("hull", "marker", "debris"):
            b = self._bid[nm]
            if nm != "hull":
                row += [float(x) for x in d.xpos[b]]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            row.append(float(math.sqrt(v6[3] * v6[3] + v6[4] * v6[4] + v6[5] * v6[5])))
        row += [float(x) for x in d.xpos[self._bid["stab"]]]
        bat = W.sim.battery
        row += [float(W.process.leak_pressure()), float(bat.status()["soc"]), float(bat.voltage)]
        st = W.stab.public()
        G = W.gripper
        held = G.held
        fw = W.fw
        P = W.protect
        try:
            ph = PROTECT_PHASES.index(P.phase(t))
        except ValueError:
            ph = -1
        irow = [mask, STAB_STATES.index(st["state"]) if st["state"] in STAB_STATES else -1,
                self.port_names.index(st["port"]) if st.get("port") in self.port_names else -1,
                int(bool(st.get("left_initial"))), GRIPPER_RAW.index(G.state) if G.state in GRIPPER_RAW else -1,
                0 if held is None else HELD_KINDS.index(held[0]),
                -1 if held is None else (self.held_names.index(held[1]) if held[1] in self.held_names else -1),
                int(P.count), ph, FW_MODES.index(fw.mode) if fw.mode in FW_MODES else -1, int(bool(fw.degraded)),
                int(in_cage(W.cage, d.xpos[hb])), int(self.contacts.marker_on_roof),
                int(self.contacts.debris_touch_vehicle)]
        nf = W.n_frames
        irow += [int(nf.get((640, 480), 0)), int(nf.get((320, 240), 0)), int(W.n_pings)]
        self.rows_f.append(row)
        self.rows_i.append(irow)

    def _on_frame(self, W, fr, size, cam):
        p, C = cam.camera_pose(W.twin.data)
        rec = dict(i=len(self.frames), t=float(W.sim.time), size=[int(size[0]), int(size[1])], lamp=float(W.lights),
                   tilt_deg=_r(math.degrees(fr.tilt_rad), 4), cam_pos=[_r(x, 6) for x in p],
                   cam_R=[_r(x, 8) for x in np.asarray(C, float).reshape(-1)])
        if self.judge is not None:
            recs, imaged = self.judge.evaluate(cam, float(W.lights), size)
            rec["plates"] = recs
            rec["imaged"] = imaged
        self.frames.append(rec)

    # ------------------------------------------------------------------------------------------ the server's records
    def api(self, ev: str, **kw):
        """A program-side record (src = "api").  forbidden / bad_params are counted and logged up to MAX_SMALL_EVENTS."""
        if ev in self.small:
            self.small[ev] += 1
            if self.small[ev] > MAX_SMALL_EVENTS:
                return
        if self.n_api >= MAX_API_EVENTS:
            self.api_dropped += 1
            return
        e = dict(t=float(self.W.sim.time), src="api", ev=ev)
        e.update(_jsonable(kw))
        self.W.events.append(e)
        self.n_api += 1

    def motion(self, method: str, reply: dict):
        """Motion calls: logged when the firmware mode changes or the call is refused."""
        mode = self.W.fw.mode
        if not reply.get("accepted", False):
            self.api("refused", method=method, reason=reply.get("reason"))
        elif mode != self._last_mode:
            self.api("mode", method=method, mode=mode)
        self._last_mode = mode

    def cell(self, ev: str, **kw):
        e = dict(t=float(self.W.sim.time), src="cell", ev=ev)
        e.update(_jsonable(kw))
        self.W.events.append(e)

    # ------------------------------------------------------------------------------------------ export
    def series(self):
        F = np.asarray(self.rows_f, np.float64).reshape(-1, len(self.columns_f))
        I = np.asarray(self.rows_i, np.int64).reshape(-1, len(self.columns_i))
        return F, I

    def meta(self) -> dict:
        return dict(schema=SCHEMA, site=self.W.sd.get("label"), valve_names=self.valve_names, port_names=self.port_names,
                    held_names=self.held_names, columns_f=list(self.columns_f), columns_i=list(self.columns_i),
                    stab_states=list(STAB_STATES), gripper_raw=list(GRIPPER_RAW), held_kinds=list(HELD_KINDS),
                    protect_phases=[p for p in PROTECT_PHASES], fw_modes=list(FW_MODES),
                    contact_classes=list(self.contacts.names), contacts_dropped=self.contacts.dropped,
                    open_contacts=self.contacts.open_contacts(), api_counts=dict(self.small),
                    api_dropped=self.api_dropped, angles0=dict(self.W.angles0),
                    thresholds=(self.judge.P if self.judge is not None else None))

    def export(self) -> dict:
        F, I = self.series()
        return dict(meta=self.meta(), events=_jsonable(self.W.events), frames=_jsonable(self.frames), series_f=F,
                    series_i=I)

    def to_bytes(self) -> bytes:
        """Canonical bytes of the whole log (series as lists; JSON floats round-trip exactly)."""
        F, I = self.series()
        return canonical(dict(meta=self.meta(), events=self.W.events, frames=self.frames, series_f=F.tolist(),
                              series_i=I.tolist()))

    def digest(self) -> str:
        F, I = self.series()
        h = hashlib.sha256()
        h.update(canonical(dict(meta=self.meta(), events=self.W.events, frames=self.frames)))
        h.update(np.ascontiguousarray(F, "<f8").tobytes())
        h.update(np.ascontiguousarray(I, "<i8").tobytes())
        return h.hexdigest()

    def save(self, folder: str) -> dict:
        """events.json (meta, events, frames) + series.npz; returns the file names and the digest."""
        os.makedirs(folder, exist_ok=True)
        F, I = self.series()
        with open(os.path.join(folder, "events.json"), "wb") as f:
            f.write(canonical(dict(meta=self.meta(), events=self.W.events, frames=self.frames)))
        np.savez_compressed(os.path.join(folder, "series.npz"), series_f=F, series_i=I)
        return dict(files=["events.json", "series.npz"], digest=self.digest())

    @staticmethod
    def load(folder: str) -> dict:
        with open(os.path.join(folder, "events.json"), "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
        z = np.load(os.path.join(folder, "series.npz"))
        doc["series_f"], doc["series_i"] = z["series_f"], z["series_i"]
        return doc


def column(doc: dict, name: str) -> np.ndarray:
    """One column of an exported / loaded log by name."""
    meta = doc["meta"]
    if name in meta["columns_f"]:
        return np.asarray(doc["series_f"])[:, meta["columns_f"].index(name)]
    return np.asarray(doc["series_i"])[:, meta["columns_i"].index(name)]
