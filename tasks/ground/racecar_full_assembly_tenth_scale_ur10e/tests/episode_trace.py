"""Episode trace for the maintainers' replay video (a record only: never a scoring input, never visible to the agent).

For an agent episode the grader attaches an `EpisodeTrace` right after `server.reset(seed)`. It records:
  * the cell's full joint state (`qpos`) every `every_ticks` control ticks (4 ticks = 40 ms of sim time, 25 frames per sim
    second), from the reset to the end of the functional check, with a phase flag (0 = the program runs, 1 = finalize: the
    retract and the functional check);
  * the cell's event log (grasped / released / seated / fastened_off_design / dropped / lost / safety, sim-time stamped);
  * every camera frame the program received, as it received it (the PNG of the reply).
`save()` writes `<root>/<tag>/trace.npz`, `meta.json` and `cams/<index>_<camera>.png` after the episode, so the program's
wall-clock budget never pays for the record; recording during the episode copies one small array per frame.
A maintainers' replay tool rebuilds the same episode model from the hidden kit and the seed (the cell is
deterministic in the seed), checks `model_signature`, replays the frames and renders the video.
A recorder failure is logged and disables the recorder; it never reaches the cell, the program or the verdict.
"""

import base64
import hashlib
import json
import os
import time

import numpy as np

TRACE_VERSION = "arm-assembly-episode-trace/1"


def model_signature(model):
    """sha256 over the model's sizes, body / joint names and default pose: the replay must rebuild the same model."""
    h = hashlib.sha256()
    h.update(json.dumps([int(model.nq), int(model.nv), int(model.nbody), int(model.njnt), int(model.neq)]).encode())
    for i in range(model.nbody):
        h.update(model.body(i).name.encode() + b"\0")
    for i in range(model.njnt):
        h.update(model.joint(i).name.encode() + b"\0")
    h.update(np.asarray(model.qpos0, dtype=np.float64).tobytes())
    return "sha256:" + h.hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


class EpisodeTrace(object):
    def __init__(self, server, every_ticks=4, log=None):
        self.server = server
        self.every = max(1, int(every_ticks))
        self.log = log or (lambda *_a: None)
        self.t, self.q, self.phase = [], [], []
        self.cams = []                       # (sim time, camera, frame index, rgb PNG base64)
        self.phase_now = 0
        self.failed = None
        self.wall0 = time.time()
        self.finalize_sim_t = None
        self._tick, self._cam, self._fin = server.physics_tick, server.get_camera, server.finalize
        server.physics_tick = self._physics_tick           # instance attributes: InfoCellServer.step / handle call self.<name>
        server.get_camera = self._get_camera
        server.finalize = self._finalize
        self._record()

    # ------------------------------------------------------------------ hooks (same signatures as the cell's methods)
    def _physics_tick(self, track=True):
        events = self._tick(track=track)
        if self.failed is None and self.server.tick_count % self.every == 0:
            self._record()
        return events

    def _get_camera(self, name="overview", depth=True):
        reply = self._cam(name, depth)
        if self.failed is None and isinstance(reply, dict) and reply.get("ok"):
            try:
                self.cams.append((float(self.server.data.time), str(reply.get("camera", name)), int(reply.get("frame_index", -1)),
                                  reply.get("rgb_png_b64")))
            except Exception as e:  # noqa: BLE001 - a record only
                self._fail(e)
        return reply

    def _finalize(self, functional_rng, run_functional_check=True):
        if self.failed is None:
            self._record()
            self.finalize_sim_t = float(self.server.data.time)
            self.phase_now = 1
        try:
            return self._fin(functional_rng, run_functional_check=run_functional_check)
        finally:
            if self.failed is None:
                self._record()

    def _record(self):
        try:
            d = self.server.data
            self.t.append(float(d.time))
            self.q.append(np.array(d.qpos, dtype=np.float32))
            self.phase.append(self.phase_now)
        except Exception as e:  # noqa: BLE001 - a record only
            self._fail(e)

    def _fail(self, e):
        self.failed = "%s: %s" % (type(e).__name__, str(e)[:200])
        self.log("[trace] recorder disabled: %s" % self.failed)

    def detach(self):
        """Restore the cell's own methods (the server object is discarded after the episode anyway)."""
        for k in ("physics_tick", "get_camera", "finalize"):
            self.server.__dict__.pop(k, None)

    # ------------------------------------------------------------------ output
    def save(self, root, tag, summary=None, extra=None):
        """Write the trace of this episode under root/tag; returns the folder (None when the recorder failed)."""
        if self.failed is not None or not self.q:
            return None
        out = os.path.join(root, tag)
        os.makedirs(os.path.join(out, "cams"), exist_ok=True)
        np.savez_compressed(os.path.join(out, "trace.npz"), t=np.asarray(self.t, np.float64), qpos=np.stack(self.q),
                            phase=np.asarray(self.phase, np.int8))
        cams = []
        for k, (t, name, idx, b64) in enumerate(self.cams):
            fn = "%03d_%s.png" % (k, name)
            if b64:
                with open(os.path.join(out, "cams", fn), "wb") as f:
                    f.write(base64.b64decode(b64))
            cams.append({"sim_t": t, "camera": name, "frame_index": idx, "file": "cams/" + fn if b64 else None})
        s = self.server
        meta = {"trace_version": TRACE_VERSION, "tag": tag, "every_ticks": self.every, "tick_s": float(s.tick_s),
                "frames": len(self.q), "nq": int(s.model.nq), "model_signature": model_signature(s.model),
                "finalize_sim_t": self.finalize_sim_t, "events": [[float(t), str(e)] for t, e in getattr(s, "events_log", [])],
                "camera_requests": cams, "recorded_wall_s": time.time() - self.wall0}
        if extra:
            meta.update(_jsonable(extra))
        if summary is not None:
            keep = ("who", "success", "all_mates", "sticky_fraction", "safety_ok", "end_reason", "sim_time_s", "episode_wall_s",
                    "camera_frames", "camera_frames_total", "step_calls", "illegal_call_count", "mates", "functional", "damage",
                    "privileged_mode", "episode_draws", "stage_score", "n_achieved", "states", "layout", "coupling_events")
            meta["summary"] = _jsonable({k: summary[k] for k in keep if k in summary})
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        return out
