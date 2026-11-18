"""Load a twin folder (the submission snapshot, or the true twin) through the public gate.

    tw = load_twin(folder, vocabulary)     # Twin: ok, reasons, model (the gate's grader copy), units, gates, cameras
    pm = tw.physics_model()                # the model every simulation runs (excluded units deleted)
    g0 = g0_settle(tw)                     # G0's settle check (twin_gate.settle_check): may exclude units

The submission and the true twin go through exactly the same code, so every quantity derived from the truth is
computed the way the submission's is.

Rules of the contract implemented here and in the public twin_gate:
- a collision geom is any geom with (contype, conaffinity) != (0, 0); everything else is visual and ignored;
- a unit the gate excludes (failed plausibility), or that G0's settle check excludes (it still breaks the speed bound
  after the relaxation), scores 0 in every criterion and is deleted from the model before any
  physics (twin_gate.physics_model); it still counts in the inventory denominators;
- the environment is every collision geom of a body welded to the world outside the units and the gates.

The payload modules (twin_gate, twin_camera, twin_sim, twin_slice) come from $TWIN_PAYLOAD_DIR or tests/setup_payload,
nothing else: without them this module fails loudly.
"""
import importlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODS = {}


def payload_dir():
    """The folder with twin_gate.py, twin_camera.py, twin_sim.py: $TWIN_PAYLOAD_DIR if set, else tests/setup_payload."""
    env = os.environ.get("TWIN_PAYLOAD_DIR", "")
    cands = [env] if env else [os.path.join(os.path.dirname(_HERE), "setup_payload")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "twin_gate.py")):
            return os.path.abspath(c)
    raise RuntimeError("the twin payload modules are missing: no twin_gate.py in %s (set TWIN_PAYLOAD_DIR or ship "
                       "tests/setup_payload)" % (cands[0] if cands[0] else "$TWIN_PAYLOAD_DIR"))


def payload():
    """{'twin_gate', 'twin_camera', 'twin_sim', 'twin_slice'} modules of the payload (twin_slice may be None)."""
    if not _MODS:
        d = payload_dir()
        if d not in sys.path:
            sys.path.insert(0, d)
        for nm in ("twin_camera", "twin_gate", "twin_sim"):
            _MODS[nm] = importlib.import_module(nm)
        try:
            _MODS["twin_slice"] = importlib.import_module("twin_slice")
        except ImportError:
            _MODS["twin_slice"] = None
    return _MODS


def default_vocabulary():
    p = os.path.join(payload_dir(), "class_names.json")
    return p if os.path.isfile(p) else None


def vocabulary(voc=None):
    """The class names as a list (read and validated once; twin_gate.TaskError when missing or malformed)."""
    TG = payload()["twin_gate"]
    v = voc if voc is not None else default_vocabulary()
    if v is None:
        raise TG.TaskError("no class vocabulary (class_names.json) in the payload")
    return TG.load_vocabulary(v)


class Twin:
    """A gated twin at qpos0: units, gates, valid photo cameras, the environment."""

    def __init__(self, folder, gate_result):
        import mujoco
        P = payload()
        self.folder = folder
        self.gate = gate_result
        self.report = gate_result.report
        self.ok = bool(gate_result.ok)
        self.reasons = list(self.report.get("reasons", []))
        self.model = gate_result.model
        self.units = []
        self.gates = []
        self.cameras = {}
        self.n_units = len((gate_result.objects or {}).get("items", []) or [])
        self.data = None
        self._pieces = {}
        if not self.ok or self.model is None:
            return
        m = self.model
        self.data = mujoco.MjData(m)
        mujoco.mj_forward(m, self.data)
        TG = P["twin_gate"]
        TC = P["twin_camera"]
        by_body = TG._geoms_by_body(m)

        def coll_geoms(b):
            bodies = set(TG._subtree(m, b)) | {b} if b >= 0 else set()
            return sorted(g for c in bodies for g in by_body.get(c, []) if TG.is_collision_geom(m, g))
        for rec in self.report["units"]:
            b = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rec["body"]))
            self.units.append({"body": rec["body"], "body_id": b, "label": rec.get("label"),
                               "label_ok": bool(rec.get("label_in_vocabulary")), "status": rec["status"],
                               "geoms": coll_geoms(b)})
        for rec in self.report.get("gates", []):
            b = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rec["body"]))
            j = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, rec["joint"]))
            self.gates.append({"body": rec["body"], "body_id": b, "joint": rec["joint"], "joint_id": j,
                               "geoms": coll_geoms(b), "travel": rec.get("travel")})
        for rec in self.report.get("cameras", []):
            if rec.get("status") != "ok":
                continue
            K = TC.K_matrix(rec["fx"], rec["fy"], rec["cx"], rec["cy"])
            R_cv, p = TC.camera_frame(m, self.data, rec["camera_id"])
            self.cameras[int(rec["photo"])] = {"name": rec["name"], "photo": int(rec["photo"]), "K": K,
                                               "W": int(rec["W"]), "H": int(rec["H"]), "R_cv": R_cv, "p": p}
        excl = [u["body_id"] for u in self.units if u["body_id"] >= 0] + [g["body_id"] for g in self.gates]
        self._static = TG.static_geoms(m, excl)

    def unit(self, body):
        for u in self.units:
            if u["body"] == body:
                return u
        raise KeyError(body)

    def scored_units(self):
        return [u for u in self.units if u["status"] == "ok"]

    def excluded(self):
        return [u["body"] for u in self.units if u["status"] != "ok"]

    def exclude(self, bodies, reasons=None):
        """Exclude units after the gate (G0's settle check): their status becomes 'excluded' in the twin and in
        the gate report (so physics_model, the walks and every criterion leave them out); they still count as
        submitted units."""
        reasons = reasons or {}
        rep = self.report
        rep["excluded"] = sorted(set(rep.get("excluded") or []) | set(bodies))
        recs = {r["body"]: r for r in rep.get("units", [])}
        for b in bodies:
            r = recs.get(b)
            if r is not None:
                r["status"] = "excluded"
                r["reasons"] = list(r.get("reasons") or []) + [reasons.get(b, "excluded by G0's settle check")]
            for u in self.units:
                if u["body"] == b:
                    u["status"] = "excluded"

    def static_geoms(self):
        """Collision geoms of the environment (world-welded bodies, outside the units and the gates), id order."""
        return list(self._static) if self.ok else []

    def physics_model(self):
        """A fresh compiled model for the simulations: the grader copy without the excluded units' bodies."""
        if not self.ok:
            return None
        return payload()["twin_gate"].physics_model(self.gate)

    def pieces(self, geoms):
        """twin_gate.Piece of each geom (world frame, qpos0); planes have none and are skipped. Cached per geom."""
        TG = payload()["twin_gate"]
        out = []
        for g in geoms:
            if g not in self._pieces:
                self._pieces[g] = TG.geom_piece(self.model, self.data, int(g))
            if self._pieces[g] is not None:
                out.append(self._pieces[g])
        return out

    def gate_bodies(self):
        return [g["body"] for g in self.gates]

    def body_point(self, body_id):
        """The world position of a body's centre of mass (for units without a usable solid)."""
        return np.asarray(self.data.xipos[int(body_id)], float).copy()


def load_twin(folder, vocabulary=None, rules=None):
    TG = payload()["twin_gate"]
    voc = vocabulary if vocabulary is not None else default_vocabulary()
    res = TG.gate(folder, vocabulary=voc, rules=rules)
    return Twin(folder, res)


def g0_settle(twin, rules=None):
    """G0's settle check (twin_gate.settle_check) on the twin's physics model, the gates held:
    {'ran', 'fail', 'reasons', 'excluded', energy and speed maxima, ...}. The units it excludes are excluded in the
    twin from here on (Twin.exclude). Not run (and not a failure here) when the gate already failed."""
    if not twin.ok:
        return {"ran": False, "fail": False, "reasons": [], "excluded": []}
    TG = payload()["twin_gate"]
    chk = TG.settle_check(twin.gate, rules=rules, sim=payload()["twin_sim"])
    if chk.get("excluded") and not chk["fail"]:
        twin.exclude(chk["excluded"], chk.get("exclusion_reasons"))
    return chk
