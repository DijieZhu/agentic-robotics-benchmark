"""The single adapter between the site and the bricks package's inspection targets.

Every instrument (inspection point, decoy, or an untagged 'plain' instrument) is built here, at the pose its slot
gives (bricks face-frame convention: +x out of the readable face toward the viewer, +y the viewer's right, +z up;
the origin is the centre of the readable face, or the valve stem axis on the pipe).

BACKEND chooses per target type between the bricks builder and a local stand-in. A stand-in is clearly marked
(magenta accents, 'STAND-IN' on the dial) and has the same pose and names, so swapping in a bricks target is a
one-line change in BACKEND.
"""

from __future__ import annotations

import importlib
import math
import os
import sys

import mujoco
import numpy as np

from .config import SKETCH_DIR
from .util import face_quat, mat_from_quat, quat_from_mat

if SKETCH_DIR not in sys.path:
    sys.path.insert(0, SKETCH_DIR)

# type -> "bricks" | "standin"; "auto" = bricks when its module imports and builds, else the stand-in
BACKEND = {"gauge": "auto", "valve_lever": "auto", "valve_handwheel": "auto", "level": "auto", "lamps": "auto",
           "tag": "auto", "thermal": "standin", "dose_point": "standin", "hotspot": "standin"}
BRICKS_MODULE = {"gauge": "bricks.targets.gauge", "valve_lever": "bricks.targets.valve",
                 "valve_handwheel": "bricks.targets.valve", "level": "bricks.targets.level",
                 "lamps": "bricks.targets.lamps", "tag": "bricks.targets.tag"}


def bricks_module(kind):
    if BACKEND.get(kind) == "standin":
        return None
    name = BRICKS_MODULE.get(kind)
    if not name:
        return None
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001  (None when the module cannot be imported: the stand-in is built instead)
        return None


def face_dz(typ) -> float:
    """Height of an instrument's face centre above its slot's mounting point. The face centre is the reference point
    of the evidence rules (mission/evidence.py face_frame: the instrument body's origin) and the position the site
    record, the round sheet and the hall's calibration sheet give: for every type it is the
    slot's point, except the rising-stem valve, whose bricks body origin is the indicator plate's centre, pipe_drop
    (0.42 m) above the pipe the slot names."""
    if typ != "valve_handwheel":
        return 0.0
    mod = bricks_module("valve_handwheel")
    if mod is None or not hasattr(mod, "GateValveSpec"):
        return 0.0                      # the stand-in is built on the pipe point itself
    return float(getattr(mod.GateValveSpec, "pipe_drop", 0.0))


def face_point(typ, pos) -> list:
    """The face centre (world) of an instrument of type `typ` whose slot point is `pos` (see face_dz)."""
    p = [float(v) for v in pos]
    if len(p) > 2:
        p[2] = round(p[2] + face_dz(typ), 6)
    return p


def _R(q):
    return mat_from_quat(q)


class IPWorks:
    """Collects the instruments of a site and builds them into a spec."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.items: list[dict] = []
        self.markers: list[dict] = []
        self.report: dict = {}

    def add_marker(self, m):
        self.markers.append(m)

    def _v2(self):
        return bool((self.cfg.get("ladder") or {}).get("v2"))

    def _plate_spec(self, tagmod, it):
        """The instrument's tag plate: version 2 a text plate (bricks.targets.tag.text_plate), version 1 the
        marker plate with the label code."""
        if self._v2():
            return tagmod.text_plate(it["tag"])
        return tagmod.TagSpec(text=it["tag"], marker_id=int(it.get("marker_id") or 1))

    def add(self, slot, role, tag, truth, params):
        self.items.append({"slot": slot, "role": role, "tag": tag, "truth": truth, "params": params})

    def add_plate(self, name, text, pos, yaw):
        """A free text plate (the training hall's plate board and posts, version 2): its front face centre at
        `pos`, facing `yaw`."""
        self.plates = getattr(self, "plates", []) + [{"name": name, "text": text, "pos": list(pos), "yaw": float(yaw)}]

    # ------------------------------------------------------------------------------------------------ build
    def build(self, spec, mats, bodies, assets):
        self._assets = assets
        info = {}
        used = {}
        mid = 1
        for it in self.items:
            s = it["slot"]
            kind = s["type"]
            name = ("ip/%s" % it["tag"]) if it["role"] in ("ip", "decoy") else ("inst/%s" % s["id"])
            q = face_quat(s["yaw"], s.get("pitch", 0.0))
            pos = list(s["pos"])
            backend = "standin"
            names = None
            if it["role"] in ("ip", "decoy"):
                # the ladder assigns shuffled label codes (marker ids) itself: an id must not tell the slot; version 2:
                # text plates, no marker id at all
                if self._v2():
                    it["marker_id"] = None
                else:
                    it["marker_id"] = int(it["params"].get("marker_id")) if (it["params"] or {}).get("marker_id") \
                        else mid
                mid += 1
            mod = bricks_module(kind)
            if mod is not None:
                try:
                    names = self._bricks(mod, kind, spec, name, pos, q, it, assets)
                    backend = "bricks"
                except Exception as e:  # noqa: BLE001
                    names = None
                    backend = "standin (bricks failed: %s: %s)" % (type(e).__name__, str(e)[:80])
            if names is None:
                names = self._standin(kind, spec, mats, name, pos, q, it)
            used[kind] = backend
            info[name] = {"type": kind, "role": it["role"], "tag": it["tag"], "backend": backend,
                          "marker_id": it.get("marker_id"), **names}
        for m in self.markers:
            name = "marker/%s" % m["id"]
            q = face_quat(m["yaw"])
            names = None
            mod = bricks_module("tag")
            if mod is not None:
                try:
                    ts = mod.TagSpec(text=m["id"], marker_id=int(m["marker_id"]), marker_mm=110.0)
                    out = mod.add_to_spec(spec, spec.worldbody, _pose(m["pos"], q), ts, assets, name)
                    names = {"body": out.body, "geoms": list(out.geoms), "face_geoms": list(out.face_geoms),
                             "backend": "bricks", "dictionary": ts.dictionary}
                except Exception as e:  # noqa: BLE001
                    names = None
            if names is None:
                b = spec.worldbody.add_body(name=name, pos=list(m["pos"]), quat=list(q))
                mat = mats.get("marker_%s" % m["id"], {"style": "fiducial", "id": m["marker_id"], "text": m["id"]})
                g = b.add_geom(name=name + "/plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.09, 0.105],
                               material=mat, group=1, contype=0, conaffinity=0)
                names = {"body": name, "geoms": [g.name], "face_geoms": [g.name], "backend": "standin",
                         "dictionary": "DICT_5X5_250 (stand-in)"}
            info[name] = {"type": "grid_marker", "role": "marker", "tag": m["id"], "marker_id": m["marker_id"], **names}
            used["grid_marker"] = names["backend"]
        for p in getattr(self, "plates", []):
            tagmod = bricks_module("tag")
            out = tagmod.add_to_spec(spec, spec.worldbody, _pose(p["pos"], face_quat(p["yaw"])),
                                     tagmod.text_plate(p["text"]), assets, p["name"])
            info[p["name"]] = {"type": "text_plate", "role": "plate", "tag": p["text"], "body": out.body,
                               "geoms": list(out.geoms), "face_geoms": list(out.face_geoms), "backend": "bricks"}
        self.report = used
        return info

    def _bricks(self, mod, kind, spec, name, pos, q, it, assets):
        """Translate the slot + truth into the bricks target spec and build it; returns names incl. what
        apply_truth_states() needs to put the target into its true state after compiling."""
        prm = it["params"] or {}
        tr = it["truth"] or {}
        tagmod = bricks_module("tag")
        ts = None
        if it["role"] in ("ip", "decoy") and tagmod is not None:
            ts = self._plate_spec(tagmod, it)
        if kind == "gauge":
            key = _scale_key(mod, prm["lo"], prm["hi"], prm["unit"])
            ts_ = mod.GaugeSpec(name=name, size_mm=float(prm.get("case_mm", 100)), scale=key,
                                value=float(tr.get("value", prm["lo"])), mount="none", tag=ts)
            value = float(tr.get("value", prm["lo"]))
            pose = _pose(pos, q)
        elif kind == "valve_lever":
            ax = np.asarray(prm.get("pipe_axis", [1, 0, 0]), float)
            vert = abs(ax[2]) / max(np.linalg.norm(ax), 1e-9) > 0.7
            ts_ = mod.LeverValveSpec(name=name, angle_deg=float(tr.get("angle_deg", 0.0)),
                                     pipe_od=2 * float(prm.get("pipe_r", 0.04)) - 0.004, pipe_axis="z" if vert else "y",
                                     pipe_len=0.0, tag=ts)
            value = float(tr.get("angle_deg", 0.0))
            pose = _pose(pos, q)
        elif kind == "valve_handwheel":
            ts_ = mod.GateValveSpec(name=name, opening=float(tr.get("opening_pct", 50.0)) / 100.0,
                                    pipe_od=2 * float(prm.get("pipe_r", 0.05)) - 0.004, tag=ts)
            value = float(tr.get("opening_pct", 50.0)) / 100.0
            pose = _pose([pos[0], pos[1], pos[2] + float(ts_.pipe_drop)], q)
        elif kind == "level":
            z0, z1 = prm["span"]
            ts_ = mod.LevelGaugeSpec(name=name, window_m=float(z1 - z0) - 0.12, level=float(tr.get("level_pct", 50)) / 100.0,
                                     liquid="water", tank_gap=0.13, tag=ts)
            value = float(tr.get("level_pct", 50)) / 100.0
            pose = _pose(pos, q)
        elif kind == "lamps":
            cols = tuple(prm.get("colors", []))
            labels = tuple({"green": "RUN", "red": "TRIP", "amber": "ALARM", "white": "POWER"}.get(c, "ON") for c in cols)
            st = tuple(bool(x) for x in tr.get("states", []))
            ts_ = mod.LampPanelSpec(name=name, colors=cols, states=st, labels=labels, tag=ts)
            value = list(st)
            pose = _pose(pos, q)
        else:
            raise KeyError(kind)
        out = mod.add_to_spec(spec, spec.worldbody, pose, ts_, assets)
        names = {"body": out.body, "geoms": list(out.all_geoms()) if hasattr(out, "all_geoms") else list(out.geoms),
                 "face_geoms": list(out.face_geoms), "joints": list(out.joints),
                 "bricks": {"module": mod.__name__, "spec_class": type(ts_).__name__, "spec": _jsonable(ts_),
                            "names": _names_json(out), "value": value}}
        return names

    # ------------------------------------------------------------------------------------------------ stand-ins
    def _standin(self, kind, spec, mats, name, pos, q, it):
        wb = spec.worldbody
        b = wb.add_body(name=name, pos=pos, quat=list(q))
        geoms, face = [], []
        role = it["role"]
        tr = it["truth"] or {}
        prm = it["params"] or {}

        def G(n, typ, p, size, mat, quat=(1, 0, 0, 0), group=1):
            g = b.add_geom(name="%s/%s" % (name, n), type=typ, pos=list(p), size=list(size) + [0] * (3 - len(size)),
                           quat=list(quat), material=mats.get(mat) if not isinstance(mat, tuple) else
                           mats.get(mat[0], mat[1]), group=group, contype=0, conaffinity=0)
            geoms.append(g.name)
            return g

        BOX, CYL = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
        qx = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]       # cylinder axis z -> x
        Rw = _R(q)
        if kind == "gauge":
            r = prm.get("case_mm", 100) / 2000.0
            G("case", CYL, (-0.03, 0, 0), (r + 0.006, 0.03), "steel_dark", qx)
            dial = ("dial/%s" % name.replace("/", "_"), {"style": "dial", "dial": (prm["lo"], prm["hi"],
                                                                                   prm["unit"], tr.get("value", 0))})
            face.append(G("face", BOX, (0.002, 0, 0), (0.001, r, r), dial).name)
            frac = (tr.get("value", prm["lo"]) - prm["lo"]) / max(prm["hi"] - prm["lo"], 1e-9)
            a = math.radians(225.0 - 270.0 * frac)
            L = r * 0.8
            qa = [math.cos(a / 2), math.sin(a / 2), 0, 0]
            face.append(G("needle", BOX, (0.005, 0.5 * L * math.cos(a), 0.5 * L * math.sin(a)), (0.0015, L / 2, 0.0035),
                          "needle", qa).name)
            G("ring", CYL, (0.004, 0, 0), (r + 0.004, 0.0015), "placeholder", qx)
        elif kind == "valve_lever":
            ax = np.asarray(prm.get("pipe_axis", [1, 0, 0]), float)
            ax_l = Rw.T @ ax
            pr = prm.get("pipe_r", 0.04)
            G("body", BOX, (0, 0, 0), (0.07, pr + 0.025, pr + 0.025), "steel_dark",
              quat_from_mat(_frame_from_x(ax_l)))
            ang = math.radians(tr.get("angle_deg", 0.0))
            # stem up on a horizontal pipe, toward the viewer on a vertical one; the lever turns in the plane
            # normal to the stem: 0 deg along the pipe (open), 90 deg across it (closed)
            a_l = ax_l / max(np.linalg.norm(ax_l), 1e-9)
            stem = np.array([0.0, 0.0, 1.0]) if abs(a_l[2]) < 0.7 else np.array([1.0, 0.0, 0.0])
            d0 = a_l - np.dot(a_l, stem) * stem
            d0 /= max(np.linalg.norm(d0), 1e-9)
            perp = np.cross(stem, d0)
            dirv = math.cos(ang) * d0 + math.sin(ang) * perp
            se = stem * (pr + 0.105)
            G("stem", CYL, stem * (pr + 0.06), (0.01, 0.04), "steel", qx if stem[0] > 0.5 else (1, 0, 0, 0))
            Lv = 0.24
            c = se + dirv * Lv / 2
            face.append(G("lever", BOX, c, (Lv / 2, 0.012, 0.008), "actuator_yellow",
                          quat_from_mat(_frame_from_x(dirv))).name)
            G("tip", BOX, se + dirv * (Lv - 0.02), (0.02, 0.016, 0.012), "placeholder",
              quat_from_mat(_frame_from_x(dirv)))
        elif kind == "valve_handwheel":
            ax = np.asarray(prm.get("pipe_axis", [1, 0, 0]), float)
            ax_l = Rw.T @ ax
            pr = prm.get("pipe_r", 0.05)
            G("body", BOX, (0, 0, 0), (0.09, pr + 0.04, pr + 0.04), "steel_dark",
              quat_from_mat(_frame_from_x(ax_l)))
            G("bonnet", CYL, (0, 0, pr + 0.12), (0.035, 0.08), "steel_dark")
            G("yoke", BOX, (0, 0, pr + 0.27), (0.012, 0.05, 0.07), "steel_dark")
            zw = pr + 0.35
            face.append(G("wheel", CYL, (0, 0, zw), (0.13, 0.008), "red_fire").name)
            G("hub", CYL, (0, 0, zw), (0.03, 0.02), "steel_dark")
            op = float(tr.get("opening_pct", 50.0)) / 100.0
            stem_top = zw + 0.03 + 0.14 * op
            face.append(G("stem", CYL, (0, 0, 0.5 * (zw + stem_top)), (0.011, 0.5 * (stem_top - zw)), "steel").name)
            G("mark", BOX, (0.0, 0.0, stem_top + 0.006), (0.016, 0.016, 0.006), "placeholder")
        elif kind == "level":
            z0, z1 = prm["span"]
            zc = pos[2]
            h = z1 - z0
            G("plate", BOX, (-0.03, 0, (z0 + z1) / 2 - zc), (0.004, 0.045, h / 2), "placeholder_face")
            G("border", BOX, (-0.034, 0, (z0 + z1) / 2 - zc), (0.002, 0.052, h / 2 + 0.01), "placeholder")
            face.append(G("glass", CYL, (0, 0, (z0 + z1) / 2 - zc), (0.02, h / 2), "glass").name)
            lv = float(tr.get("level_pct", 50.0)) / 100.0
            hl = max(0.005, lv * h)
            face.append(G("liquid", CYL, (0, 0, z0 + hl / 2 - zc), (0.016, hl / 2), "liquid").name)
            for k in range(11):
                G("tick%d" % k, BOX, (-0.025, 0.03, z0 + k * h / 10 - zc), (0.002, 0.012 if k % 5 else 0.02, 0.002),
                  "drum_band")
        elif kind == "lamps":
            states = tr.get("states", [])
            colors = prm.get("colors", [])
            n = len(states)
            G("plate", BOX, (-0.004, 0, 0), (0.004, 0.04 * n + 0.04, 0.07), "panel_grey")
            G("border", BOX, (-0.006, 0, 0), (0.002, 0.04 * n + 0.05, 0.08), "placeholder")
            for k, (on, col) in enumerate(zip(states, colors)):
                y = (k - (n - 1) / 2) * 0.08
                face.append(G("lamp%d" % k, CYL, (0.008, y, 0.015), (0.022, 0.008),
                              ("lamp_%s_on" % col) if on else "lamp_off", qx).name)
                G("lbl%d" % k, BOX, (0.001, y, -0.035), (0.001, 0.03, 0.01), "cabinet_white")
        elif kind == "dose_point":
            v2 = bool((self.cfg.get("ladder") or {}).get("v2"))
            G("marker", BOX, (0, 0, 0.0025), (0.2, 0.2, 0.001), ("floor/%s" % name.replace("/", "_"),
              {"style": "survey_mark" if v2 else "floor_marker", "text": prm.get("label", "DP")}), quat=[1, 0, 0, 0])
            if v2:
                # version 2: the disc carries the mark's tag in the plant lettering; no separate plate
                return {"body": name, "geoms": geoms, "face_geoms": face, "joints": [], "state": {}}
        # tag plate (IPs and decoys); a ladder temperature point gets a real AprilTag plate beside its hot face
        tagmod = bricks_module("tag")
        if role in ("ip", "decoy") and kind == "thermal" and prm.get("tag_offset") and tagmod is not None:
            ty, tz = prm["tag_offset"]
            ts = self._plate_spec(tagmod, it)
            out = tagmod.add_to_spec(spec, b, _pose((0.004, ty, tz), [1, 0, 0, 0]), ts, self._assets, name + "/tag")
            face += list(out.face_geoms)
            geoms += list(out.geoms)
            return {"body": name, "geoms": geoms, "face_geoms": face, "joints": [], "state": {}}
        if role in ("ip", "decoy") and kind not in ("hotspot",):
            tp = _tag_offset(kind, prm)
            mat = ("tagplate/%s" % it["tag"], {"style": "tag", "text": it["tag"], "size": [256, 128]})
            if kind == "dose_point":
                face.append(G("tag", BOX, (0.0, 0.0, 0.004), (0.06, 0.12, 0.0015), mat,
                              quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4) * 0 + 0, 0]).name)
            else:
                face.append(G("tag", BOX, tp, (0.003, 0.06, 0.03), mat).name)
        return {"body": name, "geoms": geoms, "face_geoms": face, "joints": [], "state": {}}


def _frame_from_x(xdir):
    x = np.asarray(xdir, float)
    x = x / max(np.linalg.norm(x), 1e-9)
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def _tag_offset(kind, prm):
    if kind == "gauge":
        r = prm.get("case_mm", 100) / 2000.0
        return (0.0, 0.0, -r - 0.05)
    if kind in ("valve_lever", "valve_handwheel"):
        return (0.0, 0.0, -prm.get("pipe_r", 0.05) - 0.07)
    if kind == "level":
        return (0.02, 0.0, 0.0)
    if kind == "lamps":
        return (0.0, 0.0, 0.11)
    return (0.0, 0.0, 0.0)


def _pose(pos, q):
    from bricks.common import Pose  # noqa: WPS433
    return Pose(tuple(float(v) for v in pos), tuple(float(v) for v in q))


def _scale_key(mod, lo, hi, unit):
    best, bd = None, 1e18
    for k, v in mod.SCALES.items():
        d = abs(v[0] - lo) + abs(v[1] - hi) + (0 if v[2] == unit else 1000.0)
        if d < bd:
            best, bd = k, d
    return best


def _jsonable(x):
    import dataclasses
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        d = {f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
        d["__class__"] = type(x).__name__
        return d
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _names_json(out):
    d = {k: _jsonable(getattr(out, k)) for k in ("name", "kind", "body", "geoms", "face_geoms", "bodies", "joints",
                                                  "materials")}
    st = dict(out.state)
    st.pop("spec", None)
    d["state"] = _jsonable(st)
    return d


def _from_json(mod, d):
    """Rebuild a bricks spec dataclass (and its nested TagSpec) from _jsonable output (the loader's function)."""
    from .loader.states import _jsonable_to_spec
    return _jsonable_to_spec(mod, d)


def apply_truth_states(model, data, records):
    """Put every bricks instrument into its true state: the loader's function (site_gen.loader.states)."""
    from .loader.states import apply_truth_states as _apply
    return _apply(model, data, records)


def bricks_target_spec(kind, it):
    """Our instrument parameters as a bricks target spec."""
    prm = dict(it["params"] or {})
    tr = dict(it["truth"] or {})
    d = {"name": ("ip/%s" % it["tag"]) if it["role"] in ("ip", "decoy") else "inst/%s" % it["slot"]["id"],
         "tag": it["tag"] if it["role"] in ("ip", "decoy") else None}
    d.update(prm)
    d.update(tr)
    return d
