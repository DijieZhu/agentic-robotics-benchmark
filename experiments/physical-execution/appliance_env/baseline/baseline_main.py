"""Practice-model baseline (P3): the reference's primitives without the truth.

What it knows is what an agent can have: the practice units' models (units.json and units/ in the agent's
home) and the observations. Per episode it estimates the unit's placement from the first images
(perceive.PoseEstimator, render-and-compare), estimates a knob's angle from fixed-camera images before an
"off" request, and keeps the unit's state itself between observations: a pinched knob turns with the hand,
and the microwave's flag follows RoboCasa's contact rule in its own copy of the scene. Speaks the grading
wire like oracle_main.py.

    bash run.sh  ->  python3 -I baseline_main.py --units /home/user/units.json
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import appliance_env as E  # noqa: E402
import oracle_main as O  # noqa: E402
import perceive as P  # noqa: E402
import reference as R  # noqa: E402


def log(msg):
    print("[baseline] " + msg, file=sys.stderr, flush=True)


class Tracker(object):
    """Keeps the unit's joints in the controller's copy of the scene: while the fingers are closed on a
    knob, the knob turns with the hand about its axis; the microwave's flag follows the contacts."""

    def __init__(self):
        self.attached = None

    def __call__(self, sh, obs):
        env, m, d = sh.env, sh.env.model, sh.env.data
        env.unit.update(m, d, env.gripper_geoms)      # RoboCasa's button rule, in this copy of the scene
        site = d.site_xpos[env.site]
        Rh = d.site_xmat[env.site].reshape(3, 3).copy()
        width = float(obs["gripper"])
        if self.attached is None:
            if width > 0.045:
                return
            for loc in list(env.unit.knobs) + (["temp"] if env.unit.temp_knob is not None else []):
                gid = m.geom("u_knob_%s_main" % loc).id
                if np.linalg.norm(site - d.geom_xpos[gid]) < 0.03:
                    j = m.joint("u_knob_%s_joint" % loc)
                    self.attached = (loc, int(j.qposadr[0]), R.unit(d.xaxis[j.id]), Rh, float(d.qpos[j.qposadr[0]]),
                                     float(j.range[0]), float(j.range[1]))
                    return
            return
        loc, adr, axis, R0, q0, lo, hi = self.attached
        if width > 0.05:                                # released
            self.attached = None
            return
        rv = R.rotvec(Rh @ R0.T)
        d.qpos[adr] = float(np.clip(q0 + float(np.dot(rv, axis)), lo, hi))
        mujoco.mj_forward(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    a = ap.parse_args()
    units = json.load(open(a.units))
    base = os.path.dirname(os.path.abspath(a.units))
    specs = {u: E.unit_spec(c["model"], files=E.read_unit_files(os.path.join(base, c["dir"]))) for u, c in units.items()}
    estimators = {}
    out = sys.stdout

    def write(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    def read():
        line = sys.stdin.readline()
        if not line:
            raise SystemExit(0)
        return json.loads(line)

    out.write("READY\n")
    out.flush()
    msg = read()
    while True:
        kind = msg.get("type")
        if kind == "close":
            write({"ok": True})
            return
        if kind != "act":
            write({"ok": True} if kind == "reset" else {"error": "unknown message"})
            msg = read()
            continue
        obs = msg["obs"]
        uid = obs["unit"]
        if uid not in specs:                         # not a unit this baseline has a model of: hold still
            write({"action": [0, 0, 0, 0, 0, 0, -1, 0]})
            msg = read()
            continue
        if uid not in estimators:
            estimators[uid] = P.PoseEstimator(specs[uid])
        est = estimators[uid]
        jitter, cost = est.estimate(obs)
        log("unit %s: placement estimate %s (cost %.2f)" % (uid, np.round(jitter, 4).tolist(), cost))
        sh = R.Shadow(specs[uid], None, read, write)
        sh.start(msg, jitter)
        sh.hook = Tracker()
        drv = R.Driver(sh)
        drv.MAX_STEPS = 10 ** 6
        known = {}                                   # knob angles this controller has set itself
        try:
            while True:
                rid = int(sh.obs.get("request_id", -1))
                if rid < 0:
                    drv.tick(drv.site_pos(), drv.site_R(), -1.0)
                    continue
                text = sh.obs["request_text"]
                prepare(drv, sh, est, jitter, text, known)
                try:
                    O.run_request(drv, text)
                except R.PrimitiveError as exc:
                    log("request %d failed: %s" % (rid, exc))
                for loc in list(sh.env.unit.knobs) + (["temp"] if sh.env.unit.temp_knob is not None else []):
                    known[loc] = float(sh.env.data.qpos[sh.env.model.joint("u_knob_%s_joint" % loc).qposadr[0]])
                while int(sh.obs.get("request_id", -1)) == rid:
                    drv.tick(drv.site_pos(), drv.site_R(), -1.0)
        except R.EpisodeOver as over:
            msg = over.msg
            continue


def prepare(drv, sh, est, jitter, text, known):
    """Set what the controller believes about the unit before a request: a knob it must turn off is
    estimated from fresh fixed-camera images (unless it set that knob itself); the microwave's flag is
    what the request implies."""
    t = text.lower().rstrip(".")
    env, m, d = sh.env, sh.env.model, sh.env.data
    locs = []
    br = O.burner_request(text)
    if br is not None and br[0] == "off":
        locs = br[1] if br[1] is not None else sorted(env.unit.knobs)
    elif t == "turn off the oven":
        locs = ["temp"]
    fresh = False
    for loc in locs:
        adr = m.joint("u_knob_%s_joint" % loc).qposadr[0]
        if loc in known and abs(known[loc]) > 0.3:
            d.qpos[adr] = known[loc]
        else:
            if not fresh:                            # ask for both fixed images once
                obs = drv.tick(drv.site_pos(), drv.site_R(), -1.0, cams=3)
                if not obs.get("left", {}).get("shape", [0])[0]:
                    drv.tick(drv.site_pos(), drv.site_R(), -1.0)
                fresh = True
            d.qpos[adr] = est.knob_angle(sh.obs, loc, jitter, knobs=known)
            log("knob %s angle estimate %.3f" % (loc, d.qpos[adr]))
    if locs:
        mujoco.mj_forward(m, d)
    if t == "stop the microwave":
        env.unit.turned_on = True
    elif t == "start the microwave":
        env.unit.turned_on = False


if __name__ == "__main__":
    main()
