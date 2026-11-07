"""The agent's sample camera frames, the render canary and the datasheet's example request (a maintainer's tool; the
grader never runs it).

    # in a container of the task image, as root (the camera bundle is root's alone there):
    python3 -I -B make_sample_frames.py frames --out <task>/environment/assets/payload/given/frames
    python3 -I -B make_sample_frames.py canary --out <task>/tests/camera_canary.json
    python3 -I -B make_sample_frames.py act

`frames`: four dev episodes (labels A-D; their seeds are dev seeds, none in the hidden set or its selection, and the
labels name no seed) flown by the privileged reference through the grader's own episode loop (verify.run_episode,
in process); per episode 10 frames: the start as flown, 3 far-field frames aimed at the target's centre (the craft
turned to look at it, not as flown), 2 mid-range frames, 1 just after the re-key's restart, 2 final-approach frames and
1 at the hold, as flown. Each is the exact frame the grader's camera renders for that pose, noise included (camera.Camera,
the dev noise key, never the set's), saved as a lossless mono8 PNG; truth.json gives each frame's true geometry.
Rendered in the task image, so the samples are what the grader renders there (Mesa lavapipe).

`canary`: the hash verify.py's render canary compares against (a fixed state of dev episode A, rendered in the task
image). `act`: datasheet's example `act` request (a dev seed, scripted duties until t = 37), its frame data cut.
"""

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import oracle_batch  # noqa: E402  (puts tests/ and solution/privileged/ on the path)
import verify  # noqa: E402
import camera  # noqa: E402

DEV_NOISE_KEY = int(hashlib.sha256(b"fault_adaptive_spacecraft_docking dev camera noise key").hexdigest(), 16)
EPISODES = (("A", 0x946aa31115d2e652), ("B", 0x61e358bf304348d6), ("C", 0xf57e8142155a0b8d),
            ("D", 12190882755039057146))           # D: the first draw of random.Random(20261107).getrandbits(64)
ACT_SEED, ACT_T = 1099511640001, 37
CANARY = ("A", 0.62)                               # episode A's frame nearest to 0.62 m from the docking target


def leak_checks(seeds):
    S = json.load(open(os.path.join(TESTS, "hidden", "episodes.json")))
    L = json.load(open(os.path.join(TESTS, "hidden", "selection.json")))
    hidden = {e["seed"] for e in S["episodes"]} | {c["seed"] for c in L["candidates"]}
    text = open(os.path.join(TESTS, "hidden", "episodes.json")).read() + open(
        os.path.join(TESTS, "hidden", "selection.json")).read()
    for s in seeds:
        if s in hidden or str(s) in text or "%x" % s in text or "%016x" % s in text:
            raise SystemExit("seed {} is part of the hidden set or its selection".format(s))
    if DEV_NOISE_KEY == int(S["probe_key"], 16) or oracle_batch.DEV_KEY == S["probe_key"]:
        raise SystemExit("a dev key equals the set's key")


class RecSensors(verify.Sensors):
    """The grader's Sensors, remembering every real observation (t, the frame's record, the star tracker, the truth)."""
    log = []

    def observe(self, env, obs16):
        obs, rec = super(RecSensors, self).observe(env, obs16)
        RecSensors.log.append({"t": env.steps, "rec": rec, "star": obs["star_tracker_yaw"],
                               "truth": (env.x, env.y, env.theta, env.phase, env.port_radius()),
                               "hold": env.hold_steps, "restart": verify.restart_active(env),
                               "rekey": env.rekey_active})
        return obs, rec


def fly(seed):
    """The privileged reference's flight of a dev episode through verify.run_episode; the log of its real requests."""
    verify.Sensors = RecSensors
    RecSensors.log = []
    try:
        out = oracle_batch.run_one(seed)
    finally:
        verify.Sensors = RecSensors.__bases__[0]
    return out, list(RecSensors.log)


def dt_range(rec):
    """The camera's distance to the docking target's flush LED faces (m)."""
    _, x, y, th, gx, gy, ph = rec
    pos = camera.camera_pose(x, y, th)[0]
    dt = (0.8345 * math.cos(ph), 0.8345 * math.sin(ph), -0.115)
    return math.sqrt(sum((pos[i] - dt[i]) ** 2 for i in range(3)))


def pick(log):
    """(kind, attitude, log entry) x 10 for one episode (module docstring)."""
    fresh = [e for e in log if e["rec"] is not None]
    seen, out = set(), []

    def take(kind, att, cands, n):
        cands = [e for e in cands if e["rec"][0] not in seen]
        if len(cands) < n:
            raise SystemExit("not enough {} frames".format(kind))
        idx = [round(i * (len(cands) - 1) / max(n - 1, 1)) for i in range(n)] if n > 1 else [0]
        for i in idx:
            seen.add(cands[i]["rec"][0])
            out.append((kind, att, cands[i]))
    take("start", "flown", [e for e in fresh if e["t"] == 0], 1)
    take("far", "aimed", [e for e in fresh if dt_range(e["rec"]) >= 1.6 and not e["rekey"]], 3)
    take("mid", "flown", [e for e in fresh if 0.9 <= dt_range(e["rec"]) < 1.4], 2)
    first_after = []
    for e in log:
        if e["rekey"] and not e["restart"] and e["rec"] is not None:
            first_after = [e]
            break
    take("after_restart", "flown", first_after, 1)
    after = [e for e in fresh if e["rekey"] and e["hold"] == 0]
    take("final", "flown", [e for e in after if 0.40 <= dt_range(e["rec"]) < 0.9], 2)
    take("hold", "flown", [e for e in fresh if e["hold"] >= 10], 1)
    return out


def frames(a):
    from PIL import Image
    leak_checks([s for _, s in EPISODES] + [ACT_SEED])
    os.makedirs(a.out, exist_ok=True)
    cam = camera.Camera()
    truth = []
    for label, seed in EPISODES:
        result, log = fly(seed)
        if not result["success"]:
            raise SystemExit("episode {} did not dock".format(label))
        for kind, att, e in pick(log):
            stamp, x, y, th, gx, gy, ph = e["rec"]
            rp = math.hypot(gx, gy)
            yaw = math.atan2(-y, -x) + math.pi if att == "aimed" else th   # aimed: body -x through the drum's centre
            yaw = camera.wrap(yaw)
            img = cam.frame(DEV_NOISE_KEY, seed, stamp, x, y, yaw, ph, rp)
            name = "{}_{:03d}_{}.png".format(label, stamp, kind)
            Image.fromarray(img).save(os.path.join(a.out, name), optimize=True)
            pos = camera.camera_pose(x, y, yaw)[0]
            c, s = math.cos(ph), math.sin(ph)
            row = {"file": name, "episode": label, "kind": kind, "attitude": att, "stamp": stamp,
                   "craft_x_m": x, "craft_y_m": y, "craft_yaw_rad": yaw, "target_phase_rad": ph, "port_radius_m": rp,
                   "camera_in_port_frame": {"x_m": c * pos[0] + s * pos[1], "y_m": -s * pos[0] + c * pos[1],
                                            "z_m": pos[2], "boresight_yaw_rad": camera.wrap(yaw + math.pi - ph)},
                   "camera_to_docking_target_m": dt_range((stamp, x, y, yaw, gx, gy, ph))}
            if att == "flown":
                row.update({"delivered_at_t": e["t"], "star_tracker_yaw_at_t": e["star"],
                            "true_yaw_at_t_rad": e["truth"][2]})
            truth.append(row)
            print(json.dumps({k: row[k] for k in ("file", "camera_to_docking_target_m")}))
    cam.close()
    text = json.dumps({"frames": truth}, indent=1)
    for _, seed in EPISODES:
        assert str(seed) not in text and "%x" % seed not in text
    with open(os.path.join(a.out, "truth.json"), "w") as f:
        f.write(text + "\n")
    print("{} frames".format(len(truth)))


def canary(a):
    label, rng = CANARY
    seed = dict(EPISODES)[label]
    _, log = fly(seed)
    e = min((e for e in log if e["rec"] is not None), key=lambda e: abs(dt_range(e["rec"]) - rng))
    stamp, x, y, th, gx, gy, ph = e["rec"]
    state = [x, y, th, ph, math.hypot(gx, gy)]
    cam = camera.Camera()
    sha = [hashlib.sha256(cam.render_rgb(*state).tobytes()).hexdigest() for _ in range(2)]
    cam.close()
    assert sha[0] == sha[1]
    rec = {"state": state, "rgb_sha256": sha[0],
           "about": "verify.py's render canary: the RGB render of this dev state (x, y, yaw, target phase, port radius; "
                    "episode {} of review/make_sample_frames.py, stamp {}) in the task image (Mesa lavapipe {}); "
                    "recorded as camera_canary_match, never scored".format(label, stamp, mesa_version())}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps(rec))


def mesa_version():
    try:
        import subprocess
        out = subprocess.run(["dpkg-query", "-W", "-f", "${Version}", "mesa-vulkan-drivers"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except (OSError, ValueError):
        return "unknown"


def act_request(cam):
    """The datasheet's example request: dev seed ACT_SEED, duties 0.10 + 0.05 ((t + i) mod 3) until t = ACT_T, no
    lock, the dev noise key."""
    env = verify.FaultAdaptiveDocking()
    sensors = verify.Sensors(cam, DEV_NOISE_KEY, ACT_SEED)
    obs16 = env.reset(ACT_SEED)
    for t in range(ACT_T):
        sensors.observe(env, obs16)
        obs16 = env.step([0.10 + 0.05 * ((t + i) % 3) for i in range(6)])[0]
    obs, _ = sensors.observe(env, obs16)
    return {"type": "act", "t": ACT_T, "obs": obs}


def cut(line_obj):
    d = json.loads(json.dumps(line_obj))
    fr = d["obs"].get("frame")
    if fr:
        fr["data"] = fr["data"][:16] + "..." + fr["data"][-16:]
    return json.dumps(d)


def act(a):
    leak_checks([ACT_SEED])
    cam = camera.Camera()
    req = act_request(cam)
    cam.close()
    print(cut(req))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("frames", "canary", "act"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"frames": frames, "canary": canary, "act": act}[a.mode](a)


if __name__ == "__main__":
    main()
