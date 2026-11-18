"""Episode trace for the maintainers' replay video (a record only: never a scoring input, never visible to the agent).

For an agent episode the grader attaches an `EpisodeTrace` right after `server.reset(seed)`. It records:
  * the cell's full joint state (`qpos`) every `every_ticks` control ticks (4 ticks = 40 ms of sim time, 25 frames per sim
    second), from the reset to the end of the functional check, with a phase flag (0 = the program runs, 1 = finalize: the
    retract and the functional check);
  * the cell's event log (grasped / released / seated / fastened_off_design / dropped / lost / safety, sim-time stamped);
  * the camera frames the program received, as it received them (the PNG of the reply), within the limits below.
`save()` writes `<root>/<tag>/trace.npz`, `meta.json` and `cams/<index>_<camera>.png` after the episode, so the program's
wall-clock budget never pays for the record; recording during the episode copies one small array per frame.
A maintainers' replay tool rebuilds the same episode model from the hidden kit and the seed (the cell is
deterministic in the seed), checks `model_signature`, replays the frames and renders the video.
A recorder failure is logged and disables the recorder; it never reaches the cell, the program or the verdict.

The record's limits (this task's addition to the vendored file: work package P32b, ruling P26-2 of the development
folder's shared/DECISIONS_P2_ASM.md 8.2; the published precedent kept every frame, 600 top frames being about 1.7 GB of
memory and 1.6 GB of disk per episode). The record must never fill the verifier for a legitimate camera-heavy program;
the cell, the program's replies and the verdict are the same with or without them.
  * camera frames: the frames of the program's first FRAMES_ALL_FIRST answered camera requests are kept, then one of
    every FRAMES_EVERY (requests 20, 40, 60, ... counted from 0 in the order the program received them); once a frame to
    be kept would take the kept PNGs past FRAME_BYTES_MAX, it and every later frame are not kept. meta.json lists EVERY
    request (`file` null, `kept` false and `why_not_kept` for a frame not kept) and the counts.
  * qpos records: at most RECORDS_MAX (40 ms apart that is 1600 s of sim time; this cell's sim cap is 1200 s). Past it
    every other record is dropped and the interval doubled (the record times are stored, so a replay follows them).
  * events: at most EVENTS_MAX in meta.json (the first and the last half); the rest are counted.
  * save() writes at most TRACE_BYTES_MAX per episode: a kept frame that would pass it (with room left for meta.json)
    is not written.
  * the agent program's stderr log. verify.py hands the program a regular file, <ARB_RECORD_ROOT>/logs/agent_epNN.log,
    opened for appending (arm/assembly_cell/wire.py). While a trace is attached, `StderrLogCap` keeps every such log in
    that folder to its first STDERR_LOG_MAX_BYTES bytes: it looks every STDERR_LOG_POLL_S s and cuts what is beyond (the
    program appends, so its next write lands at the cut and the first bytes stay as written). When the trace detaches
    (verify.py detaches it after the program and every process of its account are gone) a capped log is cut once more
    to leave room for one line that says so and how many bytes were dropped (at least: what is written between two
    looks is counted at the next), so a log is never longer than the cap once the episode is over. Between two looks
    it can exceed the cap by what the program writes in STDERR_LOG_POLL_S.
"""

import base64
import hashlib
import json
import os
import threading
import time

import numpy as np

TRACE_VERSION = "arm-assembly-episode-trace/1"

# the record's limits (P32b; module constants on purpose: they are not grading knobs and stay out of grader_config.json)
FRAMES_ALL_FIRST = 20                      # every frame of the first 20 camera requests ...
FRAMES_EVERY = 20                          # ... then one of every 20 requests ...
FRAME_BYTES_MAX = 150 * 1000 * 1000        # ... while the kept PNGs stay within 150 MB (a top frame is about 2.7 MB)
RECORDS_MAX = 40000                        # qpos records kept (about 24 MB of float32 for this cell's 149 coordinates)
EVENTS_MAX = 20000                         # cell events listed in meta.json
TRACE_BYTES_MAX = 200 * 1000 * 1000        # all files save() writes for one episode
META_RESERVE_BYTES = 8 * 1000 * 1000       # room kept for meta.json when frames are written
STDERR_LOG_MAX_BYTES = 5 * 1000 * 1000     # the agent program's stderr log of one episode
STDERR_LOG_POLL_S = 0.1
STDERR_LOG_NOTE_RESERVE = 256              # room for the closing line of a capped log
STDERR_LOG_PREFIX, STDERR_LOG_SUFFIX = "agent_ep", ".log"     # verify.py: LOGS/agent_ep%02d.log


def agent_log_dir():
    """the folder verify.py writes the agent program's stderr logs to (verify.py: LOGS = <ARB_RECORD_ROOT>/logs)"""
    return os.path.join(os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify"), "logs")


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


def _png_len(b64):
    """the decoded length of a base64 text (str or bytes) without decoding it"""
    tail = b64[-2:]
    pad = tail.count("=" if isinstance(tail, str) else b"=")
    return len(b64) * 3 // 4 - pad


class StderrLogCap(object):
    """Keeps the agent program's stderr logs (`folder`/agent_ep*.log, regular files) to their first `cap` bytes while it
    runs (a daemon thread looking every `poll_s` s); `close()` stops it and ends every log it capped with one line, the
    log no longer than `cap`. Failures are logged once and stop the watcher; they never reach the cell or the verdict."""

    def __init__(self, folder=None, cap=STDERR_LOG_MAX_BYTES, poll_s=STDERR_LOG_POLL_S, log=None):
        self.folder = folder or agent_log_dir()
        self.cap, self.poll_s = int(cap), float(poll_s)
        self.log = log or (lambda *_a: None)
        self.dropped = {}                  # log file name -> bytes cut off (at least)
        self.failed = None
        self.closed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stderr-log-cap", daemon=True)
        self._thread.start()

    def _logs(self):
        try:
            with os.scandir(self.folder) as it:
                return sorted(e.path for e in it if e.name.startswith(STDERR_LOG_PREFIX) and e.name.endswith(STDERR_LOG_SUFFIX)
                              and e.is_file(follow_symlinks=False))
        except (FileNotFoundError, NotADirectoryError):
            return []

    @staticmethod
    def _cut(path, keep):
        """cut `path` to its first `keep` bytes; the bytes cut off (0 when it was not longer)"""
        if os.lstat(path).st_size <= keep:
            return 0
        fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            size = os.fstat(fd).st_size
            if size <= keep:
                return 0
            os.ftruncate(fd, keep)
            return size - keep
        finally:
            os.close(fd)

    def _pass(self, keep):
        for p in self._logs():
            try:
                n = self._cut(p, keep)
            except FileNotFoundError:          # removed meanwhile: nothing to keep short
                continue
            if n:
                name = os.path.basename(p)
                if name not in self.dropped:
                    self.log("[trace] %s passed %d bytes: kept to its first %d" % (name, self.cap, keep))
                self.dropped[name] = self.dropped.get(name, 0) + n

    def _fail(self, e):
        if self.failed is None:
            self.failed = "%s: %s" % (type(e).__name__, str(e)[:200])
            self.log("[trace] stderr log cap disabled: %s" % self.failed)

    def _run(self):
        while not self._stop.wait(self.poll_s):
            try:
                self._pass(self.cap)
            except Exception as e:  # noqa: BLE001 - a record only
                self._fail(e)
                return

    def close(self):
        """stop watching; a log that passed the cap is cut to leave room for one closing line, which is appended (the log
        stays within the cap); returns {log file name: bytes dropped}"""
        if self.closed:
            return dict(self.dropped)
        self.closed = True
        self._stop.set()
        self._thread.join(timeout=5.0)
        try:
            self._pass(self.cap)
            for name in sorted(self.dropped):
                p = os.path.join(self.folder, name)
                self.dropped[name] += self._cut(p, max(0, self.cap - STDERR_LOG_NOTE_RESERVE))
                note = ("\n[grader] this log is capped at %d bytes: the program wrote at least %d bytes more, which are not "
                        "kept\n" % (self.cap, self.dropped[name])).encode("ascii")
                fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
                try:
                    os.write(fd, note[:STDERR_LOG_NOTE_RESERVE])
                finally:
                    os.close(fd)
                self.log("[trace] %s capped at %d bytes, at least %d bytes dropped" % (name, self.cap, self.dropped[name]))
        except Exception as e:  # noqa: BLE001 - a record only
            self._fail(e)
        return dict(self.dropped)


class EpisodeTrace(object):
    def __init__(self, server, every_ticks=4, log=None):
        self.server = server
        self.every = max(1, int(every_ticks))
        self.every_initial = self.every
        self.log = log or (lambda *_a: None)
        self.t, self.q, self.phase = [], [], []
        self.cams = []                       # (sim time, camera, frame index, rgb PNG base64 or None when not kept)
        self.cam_drop = {}                   # position in cams -> why its frame is not kept ("thinned", "budget")
        self.frames_received = 0
        self.frames_kept = 0
        self.frame_bytes = 0                 # decoded PNG bytes of the kept frames
        self.frames_full = False             # the frame budget is used up: no later frame is kept
        self.records_thinned = 0
        self.phase_now = 0
        self.failed = None
        self.wall0 = time.time()
        self.finalize_sim_t = None
        self.stderr_cap = None
        self.stderr_dropped = {}
        try:
            self.stderr_cap = StderrLogCap(log=self.log)
        except Exception as e:  # noqa: BLE001 - a record only
            self.log("[trace] stderr log cap not started: %r" % (e,))
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
                b64 = reply.get("rgb_png_b64")
                i = self.frames_received
                self.frames_received += 1
                why = None
                if b64:
                    if not (i < FRAMES_ALL_FIRST or (i - FRAMES_ALL_FIRST) % FRAMES_EVERY == 0):
                        why = "thinned"
                    elif self.frames_full or self.frame_bytes + _png_len(b64) > FRAME_BYTES_MAX:
                        why, self.frames_full = "budget", True
                if why is not None:
                    self.cam_drop[len(self.cams)] = why
                    b64 = None
                elif b64:
                    self.frames_kept += 1
                    self.frame_bytes += _png_len(b64)
                self.cams.append((float(self.server.data.time), str(reply.get("camera", name)), int(reply.get("frame_index", -1)),
                                  b64))
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
            if len(self.q) > RECORDS_MAX:
                # every other record dropped (the first kept), the interval doubled
                self.t, self.q, self.phase = self.t[::2], self.q[::2], self.phase[::2]
                self.every *= 2
                self.records_thinned += 1
        except Exception as e:  # noqa: BLE001 - a record only
            self._fail(e)

    def _fail(self, e):
        self.failed = "%s: %s" % (type(e).__name__, str(e)[:200])
        self.log("[trace] recorder disabled: %s" % self.failed)

    def detach(self):
        """Restore the cell's own methods (the server object is discarded after the episode anyway) and close the stderr
        log cap (verify.py detaches after the program and every process of its account are gone)."""
        for k in ("physics_tick", "get_camera", "finalize"):
            self.server.__dict__.pop(k, None)
        if self.stderr_cap is not None:
            try:
                self.stderr_dropped = self.stderr_cap.close()
            except Exception as e:  # noqa: BLE001 - a record only
                self.log("[trace] stderr log cap not closed: %r" % (e,))

    # ------------------------------------------------------------------ output
    def save(self, root, tag, summary=None, extra=None):
        """Write the trace of this episode under root/tag; returns the folder (None when the recorder failed)."""
        if self.failed is not None or not self.q:
            return None
        out = os.path.join(root, tag)
        os.makedirs(os.path.join(out, "cams"), exist_ok=True)
        npz = os.path.join(out, "trace.npz")
        np.savez_compressed(npz, t=np.asarray(self.t, np.float64), qpos=np.stack(self.q), phase=np.asarray(self.phase, np.int8))
        written = os.path.getsize(npz)
        cams = []
        n_size = 0
        for k, (t, name, idx, b64) in enumerate(self.cams):
            fn = "%03d_%s.png" % (k, name)
            why = self.cam_drop.get(k)
            if b64:
                data = base64.b64decode(b64)
                if written + len(data) + META_RESERVE_BYTES > TRACE_BYTES_MAX:
                    why, data = "size", None
                    n_size += 1
                else:
                    with open(os.path.join(out, "cams", fn), "wb") as f:
                        f.write(data)
                    written += len(data)
            else:
                data = None
            ent = {"sim_t": t, "camera": name, "frame_index": idx, "file": "cams/" + fn if data is not None else None}
            if why is not None:
                ent["kept"] = False
                ent["why_not_kept"] = why
            cams.append(ent)
        s = self.server
        events = [[float(t), str(e)] for t, e in getattr(s, "events_log", [])]
        n_events = len(events)
        if n_events > EVENTS_MAX:
            events = events[:EVENTS_MAX // 2] + events[n_events - EVENTS_MAX // 2:]
        dropped = {}
        for why in self.cam_drop.values():
            dropped[why] = dropped.get(why, 0) + 1
        if n_size:
            dropped["size"] = n_size
        meta = {"trace_version": TRACE_VERSION, "tag": tag, "every_ticks": self.every, "tick_s": float(s.tick_s),
                "frames": len(self.q), "nq": int(s.model.nq), "model_signature": model_signature(s.model),
                "finalize_sim_t": self.finalize_sim_t, "events": events, "camera_requests": cams,
                "recorded_wall_s": time.time() - self.wall0,
                "limits": {"frames_all_first": FRAMES_ALL_FIRST, "frames_every": FRAMES_EVERY, "frame_bytes_max": FRAME_BYTES_MAX,
                           "records_max": RECORDS_MAX, "events_max": EVENTS_MAX, "trace_bytes_max": TRACE_BYTES_MAX,
                           "stderr_log_max_bytes": STDERR_LOG_MAX_BYTES, "source": "work package P32b (ruling P26-2)"},
                "camera_frames_received": self.frames_received, "camera_frames_saved": self.frames_kept - n_size,
                "camera_frames_not_kept": dropped, "camera_png_bytes_saved": written - os.path.getsize(npz),
                "every_ticks_initial": self.every_initial, "records_thinned": self.records_thinned,
                "events_total": n_events, "events_omitted": n_events - len(events),
                "stderr_logs_capped": dict(self.stderr_dropped), "bytes_before_meta": written}
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
