"""Load a twin folder (the submission snapshot, or the ground-truth twin) through the public gate.

    tw = load_twin(folder, vocabulary=None)    # Twin: ok, reasons, model (the gate's grader copy), items, camera
    pm = tw.physics_model()                    # the model every simulation runs (excluded items deleted)
    g0 = g0_settle(tw)                         # G0's settle check: fails only on a numerical blow-up

The submission and the ground truth go through exactly the same code, so every quantity derived from the truth is
computed the way the submission's is.

Rules of the contract implemented here and in the public twin_gate:
- a collision geom is any geom with (contype, conaffinity) != (0, 0); everything else is visual and ignored;
- an item the gate excludes (failed plausibility) scores 0 in every criterion and is deleted from the model before
  any physics (twin_gate.physics_model): it can neither support nor disturb another item in G0's settle check, the
  standard settle, the support truth or the probes; it still counts in the inventory denominators;
- G0's settle check (twin_gate.blowup_check) fails only on a numerical blow-up (non-finite state, mechanical-energy
  gain beyond 0.5 x total item mass x v_tol^2, an item faster than g t + v_tol); falls and slides are left to the
  physics criteria.

The payload modules (twin_gate, twin_camera, twin_sim) come from $TWIN_PAYLOAD_DIR or tests/setup_payload, nothing
else: without them this module fails loudly.
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
    """{'twin_gate', 'twin_camera', 'twin_sim'} modules of the payload."""
    if not _MODS:
        d = payload_dir()
        if d not in sys.path:
            sys.path.insert(0, d)
        for nm in ("twin_camera", "twin_gate", "twin_sim"):
            _MODS[nm] = importlib.import_module(nm)
    return _MODS


def default_vocabulary():
    p = os.path.join(payload_dir(), "product_names.json")
    return p if os.path.isfile(p) else None


def vocabulary(voc=None):
    """The product names as a list (read and validated once; twin_gate.TaskError when missing or malformed)."""
    TG = payload()["twin_gate"]
    v = voc if voc is not None else default_vocabulary()
    if v is None:
        raise TG.TaskError("no product vocabulary (product_names.json) in the payload")
    return TG.load_vocabulary(v)


class Twin:
    """A gated twin: model at qpos0 with roles, items and the photo camera."""

    def __init__(self, folder, gate_result):
        import mujoco
        P = payload()
        self.folder = folder
        self.gate = gate_result
        self.report = gate_result.report
        self.ok = bool(gate_result.ok)
        self.reasons = list(self.report.get("reasons", []))
        self.model = gate_result.model
        self.items = []
        self.n_items = len((gate_result.objects or {}).get("items", []) or [])
        self.drawer = self.report.get("drawer")
        self.K = None
        if not self.ok or self.model is None:
            return
        m = self.model
        self.data = mujoco.MjData(m)
        mujoco.mj_forward(m, self.data)
        TG = P["twin_gate"]
        for rec in self.report["items"]:
            b = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rec["body"]))
            bodies = set(TG._subtree(m, b))
            geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and TG.is_collision_geom(m, g)]
            self.items.append({"body": rec["body"], "body_id": b, "label": rec.get("label"),
                               "label_ok": bool(rec.get("label_in_vocabulary")), "status": rec["status"],
                               "geoms": geoms})
        self.drawer_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.drawer)) if self.drawer else -1
        cam = self.report["camera"]
        self.cam_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam["name"]))
        TC = P["twin_camera"]
        self.K, self.W, self.H = TC.decode(m, self.cam_id)
        self.R_cv, self.p = TC.camera_frame(m, self.data, self.cam_id)
        self.up_cam = self.R_cv.T @ np.array([0.0, 0.0, 1.0])

    def scored_items(self):
        return [it for it in self.items if it["status"] == "ok"]

    def excluded(self):
        return [it["body"] for it in self.items if it["status"] != "ok"]

    def physics_model(self):
        """A fresh compiled model for the simulations: the grader copy without the excluded items' bodies."""
        if not self.ok:
            return None
        return payload()["twin_gate"].physics_model(self.gate)

    def furniture_geoms(self):
        """Collision geoms that are not part of an item: static bodies and the drawer subtree."""
        m = self.model
        TG = payload()["twin_gate"]
        item_geoms = set(g for it in self.items for g in it["geoms"])
        item_bodies = set()
        for it in self.items:
            item_bodies.update(TG._subtree(m, it["body_id"]))
        out = []
        for g in range(m.ngeom):
            if g in item_geoms or int(m.geom_bodyid[g]) in item_bodies:
                continue
            if not TG.is_collision_geom(m, g):
                continue
            out.append(g)
        return out


def load_twin(folder, vocabulary=None, rules=None):
    TG = payload()["twin_gate"]
    voc = vocabulary if vocabulary is not None else default_vocabulary()
    res = TG.gate(folder, vocabulary=voc, rules=rules)
    return Twin(folder, res)


def g0_settle(twin, rules=None):
    """G0's settle check (twin_gate.blowup_check) on the twin's physics model: {'ran', 'fail', 'reasons', energy and
    speed maxima, ...}. Not run (and not a failure here) when the gate already failed."""
    if not twin.ok:
        return {"ran": False, "fail": False, "reasons": []}
    TG = payload()["twin_gate"]
    return TG.blowup_check(twin.physics_model(), drawer=twin.drawer, rules=rules, sim=payload()["twin_sim"])
