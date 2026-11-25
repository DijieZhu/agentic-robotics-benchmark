"""Harness cell: the public wire of an episode, and an episode run in a process of its own.

`Cell` is one episode: the simulated cell (harness_env.HarnessEnv), its two cameras (harness_render.Worker, a render
process per episode) and what a controller is told and may command.

  job      at reset, once: the job sheet (harness_env.job_sheet: the printed, nominal places of the connector and the
           fixtures, the cable's radius and length, the mounting tolerances) and the two cameras' geometry.
  obs      every step: t, time, q, dq, gripper_width, tcp {pos, quat}, wrench {force, torque} (the wrist sensor,
           with noise), and the images the last action asked for ({"w", "h", "png"}: base64 PNG, RGB). The first
           observation carries both images. No cable state, no fixture pose, no tension.
  action   [dx, dy, dz, drx, dry, drz, grip, cams] or the same with a ninth number `finish`: a Cartesian step of
           the grip site in world axes from where it is (m, rad as a rotation vector; clipped to MAX_DPOS and
           MAX_DROT), grip -1 open .. 1 closed, cams the images wanted with the next observation (1 overhead,
           2 wrist, 3 both), finish 1 = hand the job over (the step is not applied; the episode ends).

`EpisodeProcess` runs a Cell in a child interpreter and speaks to it over a pipe (JSON lines). A child that ends
by itself (a native fault in the physics or the renderer) raises CellCrashed in the caller: the grader then runs the
episode again from its start in a fresh process; the commissioning service reports a bench fault.
"""

import base64
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAM_BITS = {"overhead": 1, "wrist": 2}
ALL_CAMS = 3
STILL = 1e-9


class CellCrashed(Exception):
    """The episode's process ended by itself."""


def valid_action(action):
    """8 finite numbers, or 9; cams an integer 0-3; finish 0 or 1."""
    if not isinstance(action, list) or len(action) not in (8, 9):
        return False
    for x in action:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    if not (0 <= action[7] <= ALL_CAMS and float(action[7]).is_integer()):
        return False
    return len(action) == 8 or action[8] in (0, 1)


def wants_finish(action):
    return len(action) == 9 and action[8] == 1


def images_asked(action):
    return bin(int(action[7])).count("1")


def encode_image(img):
    import cv2
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img[:, :, ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("png encoder")
    return {"w": int(img.shape[1]), "h": int(img.shape[0]), "png": base64.b64encode(buf.tobytes()).decode("ascii")}


class Cell(object):
    def __init__(self, render=True):
        import harness_env as H
        self.H = H
        self.env = H.HarnessEnv()
        self.render = render
        self.view = None
        self.cams = 0
        self.finished = False

    def reset(self, seed, n_fixtures=None):
        H = self.H
        self.close_view()
        obs = self.env.reset(int(seed), n_fixtures)
        if self.render:
            import harness_render as R
            self.view = R.Worker(int(seed), n_fixtures)
        self.cams = ALL_CAMS
        self.finished = False
        self.job = dict(self.env.job_sheet(), cameras=self.cameras(), max_steps=H.MAX_STEPS, control_dt=H.CONTROL_DT)
        return self.observe(obs)

    def cameras(self):
        """The cameras' geometry: the fixed one in the world, the hand's in the grip site's frame (it moves with
        the hand). A camera looks along its -z axis, +x to the image's right, +y up (MuJoCo's convention)."""
        import mujoco
        H, m, d = self.H, self.env.model, self.env.data
        out = {}
        for name, (mc, w, h) in H.CAMERAS.items():
            cid = m.camera(mc).id
            R = d.cam_xmat[cid].reshape(3, 3)
            p = d.cam_xpos[cid]
            if name == "wrist":
                Rs = d.site_xmat[self.env.site].reshape(3, 3)
                p, R = Rs.T @ (p - d.site_xpos[self.env.site]), Rs.T @ R
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).flatten())
            out[name] = {"frame": "world" if name == "overhead" else "tcp", "pos": [round(float(v), 5) for v in p],
                         "quat": [round(float(v), 6) for v in q], "fovy_deg": float(m.cam_fovy[cid]), "w": w, "h": h}
        return out

    def observe(self, obs):
        out = dict(obs)
        if self.render and self.view is not None:
            for name in self.H.CAMERA_NAMES:
                if self.cams & CAM_BITS[name]:
                    mc, w, h = self.H.CAMERAS[name]
                    out[name] = encode_image(self.view.render(mc, self.env.data.qpos, w, h))
        self.last = out
        return out

    def step(self, action):
        """Apply a valid action. Returns (obs, done)."""
        if wants_finish(action):
            self.finished = True
            self.cams = 0
            return self.observe(self.env.observation()), True
        self.cams = int(action[7])
        obs, done = self.env.step(action[:8])
        return self.observe(obs), bool(done)

    def look(self, cams):
        self.cams = int(cams)
        return self.observe(self.env.observation())

    def probe(self, seed):
        """The probe lock's other copy of the last observation, built as an observation is: the joints read up to
        1 mrad off, the grip site's pose computed from those joints, the wrist sensor's noise drawn again; the
        images (if any) are the same pictures."""
        import mujoco
        H, env = self.H, self.env
        rng = np.random.default_rng(int(seed))
        out = dict(self.last)
        q = np.asarray(out["q"], float) + rng.uniform(-0.001, 0.001, 7)
        d = mujoco.MjData(env.model)
        d.qpos[:] = env.data.qpos
        d.qpos[env.jq] = q
        mujoco.mj_kinematics(env.model, d)
        out["q"] = [round(float(v), 5) for v in q]
        out["tcp"] = {"pos": [round(float(v), 5) for v in d.site_xpos[env.site]],
                      "quat": [round(float(v), 6) for v in H.quat_from_mat(d.site_xmat[env.site].reshape(3, 3))]}
        f = np.asarray(out["wrench"]["force"], float) + rng.normal(0.0, H.WRENCH_NOISE[0], 3)
        t = np.asarray(out["wrench"]["torque"], float) + rng.normal(0.0, H.WRENCH_NOISE[1], 3)
        out["wrench"] = {"force": [round(float(v), 3) for v in f], "torque": [round(float(v), 4) for v in t]}
        return out

    def truth(self):
        """Token runs only: the cell's truth, and `obs`: the real observation's numbers. Both requests of a probe
        pair carry the same truth, so the reference's program runs on the real observation whichever request comes
        first, exactly as it runs in process (tests/review/select_set.py); its replies still differ with what each
        request shows (solution/oracle/oracle_main.py)."""
        t = self.env.truth()
        t["obs"] = {k: self.last[k] for k in ("t", "time", "q", "dq", "gripper_width", "tcp", "wrench")}
        return t

    def finish(self):
        res = self.env.finish()
        res["steps"] = int(self.env.t)
        res["handed_over"] = bool(self.finished)
        res["checks"] = [{"type": c["type"], "ok": bool(c["ok"])} for c in res.get("checks", [])]
        return json.loads(json.dumps(res, default=float))

    def unstable(self):
        return bool(self.env.unstable())

    def close_view(self):
        v, self.view = self.view, None
        if v is not None:
            v.close()

    def close(self):
        self.close_view()


# ------------------------------------------------------------------------------------------------------------
# an episode in a process of its own
# ------------------------------------------------------------------------------------------------------------
class EpisodeProcess(object):
    """A Cell in a child interpreter. Every method returns the child's JSON reply; CellCrashed if the child is gone."""

    def __init__(self, render=True, env=None):
        e = dict(os.environ if env is None else env)
        self.proc = subprocess.Popen([sys.executable, "-I", "-B", os.path.abspath(__file__), "--cell", "1" if render else "0"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE, env=e)
        self.job = None

    def _ask(self, msg):
        p = self.proc
        if p is None:
            raise CellCrashed("the episode's process is closed")
        try:
            p.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            p.stdin.flush()
            line = p.stdout.readline()
        except (OSError, ValueError):
            line = b""
        if not line:
            code = p.poll()
            if code is None:
                try:
                    code = p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    code = p.wait()
            self.proc = None
            raise CellCrashed("the episode's process ended by itself (status %s)" % code)
        rep = json.loads(line.decode("utf-8"))
        if "error" in rep:
            raise RuntimeError(rep["error"])
        return rep["result"]

    def reset(self, seed, n_fixtures=None):
        r = self._ask({"op": "reset", "seed": str(int(seed)), "n": n_fixtures})
        self.job = r["job"]
        return r["obs"]

    def step(self, action):
        r = self._ask({"op": "step", "action": action})
        return r["obs"], r["done"]

    def look(self, cams):
        return self._ask({"op": "look", "cams": int(cams)})

    def probe(self, seed):
        return self._ask({"op": "probe", "seed": str(int(seed))})

    def truth(self):
        return self._ask({"op": "truth"})

    def unstable(self):
        return self._ask({"op": "unstable"})

    def finish(self):
        return self._ask({"op": "finish"})

    def close(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        for f in (p.stdin, p.stdout):
            try:
                f.close()
            except OSError:
                pass
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _cell_main(render):
    out = os.fdopen(os.dup(1), "wb")           # the pipe; whatever anything prints goes to standard error
    os.dup2(2, 1)
    sys.path.insert(0, HERE)
    cell = Cell(render=render)
    try:
        for line in sys.stdin.buffer:
            msg = json.loads(line.decode("utf-8"))
            op = msg["op"]
            try:
                if op == "reset":
                    obs = cell.reset(int(msg["seed"]), msg.get("n"))
                    res = {"obs": obs, "job": cell.job}
                elif op == "step":
                    obs, done = cell.step(msg["action"])
                    res = {"obs": obs, "done": done}
                elif op == "look":
                    res = cell.look(msg["cams"])
                elif op == "probe":
                    res = cell.probe(int(msg["seed"]))
                elif op == "truth":
                    res = cell.truth()
                elif op == "unstable":
                    res = cell.unstable()
                elif op == "finish":
                    res = cell.finish()
                else:
                    raise ValueError("unknown request")
                rep = {"result": res}
            except Exception as exc:  # noqa: BLE001 - told to the caller, which decides whose fault it is
                rep = {"error": "%s: %s" % (type(exc).__name__, exc)}
            out.write((json.dumps(rep) + "\n").encode("utf-8"))
            out.flush()
    finally:
        cell.close()
    return 0


if __name__ == "__main__" and "--cell" in sys.argv[1:]:
    code = _cell_main(sys.argv[sys.argv.index("--cell") + 1] == "1")
    sys.stdout.flush()
    os._exit(int(code or 0))
