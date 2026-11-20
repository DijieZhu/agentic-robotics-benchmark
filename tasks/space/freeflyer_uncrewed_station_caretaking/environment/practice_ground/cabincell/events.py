"""events.py -- the record of one cell run (the record format, caretaking.record/1): the grader's only input besides
the hidden site package.

* ``WorldModel``   the suction faces of the site (8 return inlets of the station's cabin air units, every stowage unit's
                   holding screen SB-<M>-S and pouch pads SB-<M>-P1 (-a) / -P2 (+a)), the loose items (blocker, pouch_n:
                   bodies, box half sizes, grasp bars, own footprint, register reference, starting inlet), the
                   measurement of one instant with predicates.py (sealed area per item and face, first layer only;
                   covered fraction of every inlet; 'in the gripper'; centroid displacement from t = 0).
* ``GeomContactEvents``  every physics step: the robot's collision geoms against the station's static geoms (walls,
                   handrails, cargo bags, return-inlet parts, stowage units, fiducial plates; the dock is not a class),
                   one event per (robot geom, station geom) pair, opened by a contact after >= REARM_S without one
                   (onset normal speed of the robot's contact point, peak normal force over the event, class, names);
                   finger geoms on crew handrails are not counted (finger pads on grasp bars).  The
                   semantics of spacekit 0.4.0's sensors.ContactEvents (per step, onset speed, peak force) at the geom
                   level, which the class needs.
* ``Recorder``     the world state every cell.record_period_s from mission second 0 to the end of the settle (a last
                   sample at 'end' itself), keep-out seconds (robot centre inside a keep-out box, at KEEPOUT_HZ),
                   items touching both the robot and the dock from the first DOCKED on (at KEEPOUT_HZ), and the record
                   (``record()``) with the state and final-qpos hashes.

The geometry is predicates.py's (sealed area by the first layer,
covered fraction, the cage), the faces are the stowage units', the rows follow the record format, the force is
recorded with the speed, the classes are named, the dock check runs through the docked time.

Measurement parameters come from tests/grader_config.json through ``predicates.measure_params`` (only there: the rehearsal service never reads them);
the record period and the settle from the runtime site_spec (budgets.py).
"""
from __future__ import annotations

import hashlib
import json
import math

import mujoco
import numpy as np

RECORD_SCHEMA = "caretaking.record/1"
REARM_S = 0.2                 # a pair out of contact this long opens a new event [G]
KEEPOUT_HZ = 10.0             # keep-out time and the item-dock check run at this rate [G]
MAX_CONTACT_EVENTS = 5000     # events kept in the record (all are counted in the metrics)
HATCH_RIM_M = 0.12            # a structure contact within this of a hatch rim (radially and off its plane): hatch_seal [G]
LAMP_REACH_M = 0.10           # a structure contact within this of a luminaire's centre line: lamp [G]
UNIT_FACES = (("S", "hs", "holding_screen"), ("P1", "pp0", "pouch_pad"), ("P2", "pp1", "pouch_pad"))
FINGER_TOKENS = ("_left_proximal", "_left_distal", "_right_proximal", "_right_distal")


def _name(model, obj, i) -> str:
    return mujoco.mj_id2name(model, obj, int(i)) or ""


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def qpos_sha256(data) -> str:
    return hashlib.sha256(np.ascontiguousarray(data.qpos, dtype="<f8").tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ the site's world
class WorldModel:
    """Faces, items and their measurement (module doc).  ``P``: the predicates module; ``mp``: measure_params."""

    def __init__(self, model, info: dict, site: dict, comps, tf, P, mp: dict):
        self.m, self.info, self.site, self.P, self.mp = model, info, site, P, mp
        self.tf = tf
        # -- faces: return inlets (spacekit Inlets of the station's cabin fans) and the stowage units' faces
        self.faces, self.face_meta = {}, {}
        for mod, net in comps.vent.cabin_fans.items():
            for inl in net.inlets:
                self.faces[inl.name] = dict(c=np.asarray(inl.c, float), n=np.asarray(inl.n, float), t1=np.asarray(inl.t1, float),
                                            half=(float(inl.half[0]), float(inl.half[1])))
                self.face_meta[inl.name] = dict(kind="return_inlet", owner=f"cau_{mod}", module=mod)
        for U in info.get("units", []):
            part, mod = U["part"], U["module"]
            for suffix, which, kind in UNIT_FACES:
                F = part.face(which)
                fid = f"SB-{mod.upper()}-{suffix}"
                self.faces[fid] = dict(c=F.c.copy(), n=F.n.copy(), t1=F.a.copy(), t2=F.b.copy(), half=tuple(F.half))
                self.face_meta[fid] = dict(kind=kind, owner=f"SB-{mod.upper()}", module=mod)
        self.inlet_ids = sorted(f for f, v in self.face_meta.items() if v["kind"] == "return_inlet")
        self._reach = {f: float(math.hypot(*F["half"])) for f, F in self.faces.items()}
        # -- items (cabincell.site.build.items_of: the scene's bodies of the site's blocker and pouches)
        from cabincell.site import build as SB
        entries = {}
        if site.get("blocker"):
            entries["blocker"] = site["blocker"]
        for s in site.get("small_items", []) or []:
            entries[s["id"]] = s
        self.items = {}
        for it in SB.items_of(model, info):
            e = entries.get(it["id"], {})
            size = e.get("size") or [2 * it["half"][0] * len(it["bodies"]), 2 * it["half"][1]]
            mass = np.array([float(model.body_mass[b]) for b in it["bodies"]])
            self.items[it["id"]] = dict(kind=("blocker" if it["id"] == "blocker" else "pouch"), bodies=list(it["bodies"]),
                                        half=tuple(float(h) for h in it["half"]), inlet=it["inlet"],
                                        footprint_m2=float(size[0]) * float(size[1]), register_ref=e.get("register_ref"),
                                        w=mass / mass.sum(), r=float(np.linalg.norm(it["half"])))
        self.bars = {k: [b for b in tf.detector.bars if b.name.startswith(k + ":")] for k in self.items}
        self.ref = {}

    # -- geometry of one instant --------------------------------------------------------------------------------
    def boxes(self, data) -> list:
        out = []
        for k, it in self.items.items():
            for b in it["bodies"]:
                out.append((k, data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy(), it["half"]))
        return out

    def centroid(self, data, k) -> np.ndarray:
        it = self.items[k]
        return np.einsum("i,ij->j", it["w"], data.xipos[it["bodies"]])

    def _near(self, fid, boxes, band) -> list:
        """The boxes that can reach the face (a looser test than predicates.seal_areas' own, so its result is
        unchanged: it only saves the raster of faces nobody is near)."""
        F = self.faces[fid]
        c, n = F["c"], F["n"]
        R = self._reach[fid]
        lo, hi = band
        out = []
        for bx in boxes:
            d = bx[1] - c
            r = float(np.linalg.norm(bx[3]))
            dn = float(d @ n)
            if dn - r > hi + 0.01 or dn + r < lo - 0.01:
                continue
            if float(np.linalg.norm(d - dn * n)) > R + r + 0.01:
                continue
            out.append(bx)
        return out

    def seal_by_face(self, data, band=None) -> dict:
        """{face id: {item: sealed m^2}} (faces without any area omitted)."""
        return self.seal_of_boxes(self.boxes(data), band)

    def seal_of_boxes(self, boxes, band=None) -> dict:
        """``seal_by_face`` for given item boxes [(item, pos, R, half)] (the recorder's measure of any state)."""
        band = band if band is not None else self.mp["seal_band"]
        out = {}
        for fid, F in self.faces.items():
            near = self._near(fid, boxes, band)
            if not near:
                continue
            a = self.P.seal_areas(F, near, band)
            if a:
                out[fid] = a
        return out

    def covered(self, data, seal=None) -> dict:
        """{inlet: covered fraction} of the 8 return inlets (union of the items' first-layer area / the opening)."""
        mp = self.mp
        if seal is None or tuple(mp["cover_band"]) != tuple(mp["seal_band"]):
            seal = self.seal_by_face(data, mp["cover_band"])
        out = {}
        for iid in self.inlet_ids:
            a = sum((seal.get(iid) or {}).values())
            out[iid] = min(1.0, a / float(mp["opening_area_m2"]))
        return out

    def in_gripper(self, data, k) -> bool:
        P, lay = self.P, self.tf.layer
        if not self.bars.get(k):
            return False
        closed = float(lay.gripper.cmd) <= 0.0
        if not closed:
            return False
        pr, Rr = lay.palm_frame()
        pts = []
        for b in self.bars[k]:
            c, a, _ = b.pose(data)
            pts.append(P.to_palm(P.bar_points(c, a, b.length), pr, Rr))
        return P.in_gripper(pts, closed, self.mp["cage_mm"], lay.z_plane)

    def start(self, data) -> None:
        self.ref = {k: self.centroid(data, k) for k in self.items}

    def item_state(self, data, seal) -> dict:
        out = {}
        for k in self.items:
            sm = {f: a[k] for f, a in seal.items() if a.get(k, 0.0) > 0.0}
            on = max(sm, key=lambda f: (sm[f], f)) if sm else None
            out[k] = dict(in_gripper=bool(self.in_gripper(data, k)), seal_m2=sm,
                          disp_m=float(np.linalg.norm(self.centroid(data, k) - self.ref[k])), on_face=on)
        return out

    def items_record(self) -> dict:
        return {k: dict(kind=it["kind"], register_ref=it["register_ref"], footprint_m2=it["footprint_m2"], inlet=it["inlet"])
                for k, it in self.items.items()}

    def faces_record(self) -> dict:
        return {f: dict(v) for f, v in sorted(self.face_meta.items())}


# ------------------------------------------------------------------------------------------------ contacts
class GeomContactEvents:
    """Robot vs station contact events at the geom level, every physics step (module doc)."""

    def __init__(self, model, robot_root: int, feats: dict, t0_fn):
        m = self.m = model
        self.t0_fn = t0_fn
        names = [_name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
        bnames = [_name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom)]
        self.names = names
        coll = (np.asarray(m.geom_contype) | np.asarray(m.geom_conaffinity)) != 0
        root = np.asarray(m.body_rootid)[np.asarray(m.geom_bodyid)]
        weld = np.asarray(m.body_weldid)[np.asarray(m.geom_bodyid)]
        self.rob = coll & (root == int(robot_root))
        dock = np.array([n.startswith(("dock", "c_dock")) or b.startswith(("dock", "c_dock")) for n, b in zip(names, bnames)])
        self.sta = coll & (weld == 0) & ~dock & ~self.rob
        self.finger = np.array([self.rob[g] and n.startswith("c_arm_grip") and any(t in n for t in FINGER_TOKENS)
                                for g, n in enumerate(names)])
        self.rail = np.array([n.startswith("hr_") for n in names])
        cls = []
        for n in names:
            if n.startswith("su_"):
                cls.append("fiducial_plate" if "_fid" in n else "stowage_unit")
            elif n.startswith("ri_"):
                cls.append("inlet_part")
            else:
                cls.append("structure")
        self.cls = cls
        self.hatches = []
        for H in feats.get("hatches", []):
            n = np.asarray(H.get("door_dir_from_axis", [0, 0, 1]), float)
            n = n / max(np.linalg.norm(n), 1e-9)
            self.hatches.append((np.asarray(H["door_centre"], float), n, 0.5 * float(H.get("opening_m", 1.27)),
                                 0.5 * float(max(H.get("door_size_m", [1.56, 1.56])[:2]))))
        self.lamps = []
        for L in feats.get("lights", []):
            a = np.asarray(L["along"], float)
            a = a / max(np.linalg.norm(a), 1e-9)
            self.lamps.append((np.asarray(L.get("fixture_centre", L["pos"]), float), a, 0.5 * float(L.get("length", 0.6))))
        self.last = {}            # (robot geom, station geom) -> last mission time in contact
        self.open = {}            # pair -> event dict
        self.events = []
        self.n_events = 0
        self.by_class = {}
        self._f = np.zeros(6)
        self._v = np.zeros(6)

    def classify(self, g: int, p) -> str:
        c = self.cls[g]
        if c != "structure":
            return c
        p = np.asarray(p, float)
        for C, n, r_in, r_out in self.hatches:
            d = p - C
            off = float(d @ n)
            rad = float(np.linalg.norm(d - off * n))
            if abs(off) <= HATCH_RIM_M and r_in - HATCH_RIM_M <= rad <= r_out + HATCH_RIM_M:
                return "hatch_seal"
        for C, a, h in self.lamps:
            d = p - C
            s = float(np.clip(d @ a, -h, h))
            if float(np.linalg.norm(d - s * a)) <= LAMP_REACH_M:
                return "lamp"
        return c

    def update(self, data) -> None:
        n = int(data.ncon)
        if n == 0:
            return
        gg = data.contact.geom[:n]
        g1, g2 = gg[:, 0], gg[:, 1]
        hit = (self.rob[g1] & self.sta[g2]) | (self.rob[g2] & self.sta[g1])
        if not hit.any():
            return
        m = self.m
        t = float(data.time) - float(self.t0_fn())
        for i in np.flatnonzero(hit):
            a, b = int(g1[i]), int(g2[i])
            r, s = (a, b) if self.rob[a] else (b, a)
            if self.finger[r] and self.rail[s]:
                continue                                           # finger pads on a grasp bar (a crew handrail)
            key = (r, s)
            c = data.contact[int(i)]
            mujoco.mj_contactForce(m, data, int(i), self._f)
            fn = abs(float(self._f[0]))
            prev = self.last.get(key)
            self.last[key] = t
            ev = self.open.get(key)
            if ev is not None and prev is not None and t - prev <= REARM_S + 1e-9:
                if fn > ev["force_N"]:
                    ev["force_N"] = fn
                continue
            p = np.asarray(c.pos, float)
            nrm = np.asarray(c.frame[0:3], float)
            rb = int(m.geom_bodyid[r])
            mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, rb, self._v, 0)
            v = self._v[3:6] + np.cross(self._v[0:3], p - data.xpos[rb])
            cls = self.classify(s, p)
            ev = dict(t=round(max(t, 0.0), 4), speed_m_s=abs(float(v @ nrm)), force_N=fn, other_class=cls,
                      robot_geom=self.names[r], other=self.names[s])
            self.open[key] = ev
            self.n_events += 1
            self.by_class[cls] = self.by_class.get(cls, 0) + 1
            if len(self.events) < MAX_CONTACT_EVENTS:
                self.events.append(ev)

    def record(self) -> list:
        return [dict(e) for e in self.events]


# ------------------------------------------------------------------------------------------------ the recorder
class Recorder:
    """World state and events of one run (module doc).  The cell calls ``start()`` at mission second 0 and
    ``sample_now()`` at the end; ``attach(sim)`` registers the periodic tasks."""

    def __init__(self, cell, world: WorldModel, period_s: float, keepout_boxes: list):
        self.c = cell
        self.w = world
        self.period = float(period_s)
        self.rows = []
        self.on = False
        self.keepout = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in keepout_boxes]
        self.keepout_s = 0.0
        self.item_dock = {k: False for k in world.items}
        self._last_t = None
        self.sample_cpu_s = 0.0
        m = cell.model
        names = [_name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
        bn = [_name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom)]
        root = np.asarray(m.body_rootid)[np.asarray(m.geom_bodyid)]
        self._g_robot = root == int(m.body_rootid[cell.body])
        self._g_dock = np.array([n.startswith(("dock", "c_dock")) or b.startswith(("dock", "c_dock")) for n, b in zip(names, bn)])
        gb = np.asarray(m.geom_bodyid)
        self._g_item = {k: np.isin(gb, it["bodies"]) for k, it in world.items.items()}

    def attach(self, sim) -> None:
        sim.add_periodic(self._periodic, self.period, "cell_recorder")
        sim.add_periodic(self._fast, 1.0 / KEEPOUT_HZ, "cell_keepout_dock")

    def start(self) -> None:
        self.on = True
        self.w.start(self.c.data)
        self._last_t = None
        self.sample_now()

    # -- periodic tasks -----------------------------------------------------------------------------------------
    def _periodic(self, sim) -> None:
        if self.on:
            self.sample_now()

    def _fast(self, sim) -> None:
        if not self.on:
            return
        d = self.c.data
        p = d.xpos[self.c.body]
        if any(bool(np.all(p >= lo) and np.all(p <= hi)) for lo, hi in self.keepout):
            self.keepout_s += 1.0 / KEEPOUT_HZ
        if self.c.t_docked is not None and d.ncon:
            gg = d.contact.geom[:int(d.ncon)]
            g1, g2 = gg[:, 0], gg[:, 1]
            rob = self._g_robot[g1] | self._g_robot[g2]
            dock = self._g_dock[g1] | self._g_dock[g2]
            for k, gi in self._g_item.items():
                if self.item_dock[k]:
                    continue
                touch = gi[g1] | gi[g2]
                if (touch & rob).any() and (touch & dock).any():
                    self.item_dock[k] = True

    def sample_now(self) -> None:
        """One world-state sample at the current instant (skipped when the time did not advance)."""
        import time as _time
        c0 = _time.process_time()
        c, d = self.c, self.c.data
        t = round(float(d.time) - float(c.t0), 6)
        if self._last_t is not None and t <= self._last_t + 1e-9:
            return
        self._last_t = t
        mujoco.mj_kinematics(c.model, d)
        w = self.w
        seal = w.seal_by_face(d)
        cov = w.covered(d, seal)
        items = w.item_state(d, seal)
        row = dict(t=t, inlets={k: round(v, 6) for k, v in cov.items()},
                   items={k: dict(in_gripper=v["in_gripper"], seal_m2={f: round(a, 7) for f, a in sorted(v["seal_m2"].items())},
                                  disp_m=round(v["disp_m"], 6), on_face=v["on_face"]) for k, v in items.items()},
                   fans={k: bool(v) for k, v in sorted(c.svc.fans_running().items())}, docked=bool(c.tf.layer.docked),
                   battery_pct=round(min(100.0, max(0.0, 100.0 * float(c.tf.battery.soc))), 6),
                   robot_pos=[round(float(x), 6) for x in d.xpos[c.body]])
        self.rows.append(row)
        self.sample_cpu_s += _time.process_time() - c0

    # -- the record ---------------------------------------------------------------------------------------------
    def loss_of_control(self) -> dict:
        P, mp = self.w.P, self.w.mp
        out = {}
        for k in self.w.items:
            t = P.loss_of_control(self.rows, k, mp["loss_disp_m"], mp["loss_min_s"])
            if t is not None:
                out[k] = float(t)
        return out

    def state_sha256(self) -> str:
        return sha256_json(self.rows)
