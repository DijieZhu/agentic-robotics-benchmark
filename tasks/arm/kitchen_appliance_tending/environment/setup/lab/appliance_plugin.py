"""alelab plugin: the hidden appliance units (the same robot environment as the agent's dev sim,
appliance_env.py, and the same episodes and requests, episode.py).

Instance file (environment/setup/ref/units.json): {unit id: {model, kind, dir}}; each unit's model files
(ref/units/<name>/) are read into memory here, before the worker drops privileges. Lab instances = the
scored units. Every trial starts from a fresh 64-bit seed from OS entropy (placement, initial control
states, request queue, sensor noise). Replies carry the observation only (images, proprioception, wrench,
the open request), never the units' state or any credit; an episode ends when its last request closes.
"""

import os
import secrets

import appliance_env as E
import episode as EP

_U8 = "u8[*,*,3]"
LOCS = sorted(set(EP.LOC_TEXT.values()))
TEXTS = [""] + ["Turn on the %s burner." % s for s in LOCS] + ["Turn off the %s burner." % s for s in LOCS] + [
    "Preheat the oven.", "Turn off the oven.", "Open the oven door.", "Start the microwave.", "Stop the microwave."]
OBS_SCHEMA = {
    "t": "int",
    "unit": "enum:",
    "q": "f64[7]",
    "dq": "f64[7]",
    "ee_pos": "f64[3]",
    "ee_quat": "f64[4]",
    "gripper": "float",
    "wrench": "f64[6]",
    "T_world_wrist": "f64[4,4]",
    "T_world_left": "f64[4,4]",
    "T_world_right": "f64[4,4]",
    "K_left": "f64[3,3]",
    "K_right": "f64[3,3]",
    "K_wrist": "f64[3,3]",
    "left": _U8,
    "right": _U8,
    "wrist": _U8,
    "request_id": "int",
    "request_text": "enum:" + "|".join(TEXTS),
    "request_deadline_t": "int",
}
MESSAGES = {
    "action": "action must be a list of 8 finite numbers (6 motion values, grip, camera mask 0-7)",
}


def _schema(ids):
    s = dict(OBS_SCHEMA)
    s["unit"] = "enum:" + "|".join(ids)
    return {"info": {"control_dt": "float", "request_deadline_steps": "int"}, "reset": s, "step": s}


class _Episode(object):
    def __init__(self, ep, uid):
        self.ep, self.uid = ep, uid

    def obs(self, obs):
        obs = dict(obs)
        obs["unit"] = self.uid
        for cam in ("left", "right", "wrist"):
            if cam not in obs:                      # an image the action did not ask for: empty
                obs[cam] = E.np.zeros((0, 0, 3), E.np.uint8)
        return obs

    def step(self, action):
        obs, done, _info = self.ep.step(action)
        if self.ep.env.unstable():
            done = True
        return self.obs(obs), bool(done)


class ApplianceLab(object):
    def __init__(self, instances, api):
        self.api = api
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref")
        self.specs = {}
        for uid, cfg in sorted(instances.items()):
            if uid.startswith("_"):
                continue
            files = E.read_unit_files(os.path.join(base, cfg["dir"]))
            self.specs[uid] = E.unit_spec(cfg["model"], files=files)
        self.SCHEMA = _schema(sorted(self.specs))
        self.MESSAGES = MESSAGES

    def instance_ids(self):
        return sorted(self.specs)

    def info(self):
        return {"control_dt": E.CONTROL_DT, "request_deadline_steps": EP.DEADLINE}

    def validate_action(self, action):
        # depends only on the request and public constants (never on the instance or hidden state)
        if not isinstance(action, list) or len(action) != 8:
            raise self.api.BadRequest("action")
        out = []
        for v in action:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                raise self.api.BadRequest("action")
            out.append(float(v))
        if not (0 <= out[7] <= 7 and out[7].is_integer()):
            raise self.api.BadRequest("action")
        return out

    def reset(self, instance):
        ep = EP.Episode(E.ApplianceEnv(self.specs[instance], None, render=True), secrets.randbits(64))
        obs = ep.reset()
        e = _Episode(ep, instance)
        return e, e.obs(obs)


def make_plugin(instances, api):
    return ApplianceLab(instances, api)
