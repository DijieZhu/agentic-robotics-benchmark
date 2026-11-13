"""The privileged reference as a submission: speaks the grading wire on stdin/stdout (P3).

READY, then per episode: `reset` -> ok; `act` queries until the episode ends; `close` -> ok, exit. In a
token run every observation carries `truth` (the unit's placement jitter, its joint values, the
microwave's flag) and `unit` names the unit; the reference builds its own copy of the scene from them
(reference.Shadow) and runs the primitive for each open request. Logs go to stderr.

    bash run.sh  ->  python3 -I oracle_main.py --units units.json --assets <unit models>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="JSON: unit id -> {model, kind}")
    ap.add_argument("--assets", required=True)
    a = ap.parse_args()
    units = json.load(open(a.units))
    out = sys.stdout

    def write(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    def read():
        line = sys.stdin.readline()
        if not line:
            raise SystemExit(0)
        return json.loads(line)

    write_raw = out.write
    write_raw("READY\n")
    out.flush()
    msg = read()
    while True:
        kind = msg.get("type")
        if kind == "close":
            write({"ok": True})
            return
        if kind == "reset":
            write({"ok": True})
            msg = read()
            continue
        if kind != "act":
            write({"error": "unknown message"})
            msg = read()
            continue
        # the first query of an episode
        obs = msg["obs"]
        truth = obs.get("truth")
        if not truth:                       # not a token run: hold still (the reference needs the truth)
            write({"action": [0, 0, 0, 0, 0, 0, -1, 0]})
            msg = read()
            continue
        spec = units[obs["unit"]]
        sh = R.Shadow({"model": spec["model"], "kind": spec["kind"], "scale": 1.0, "door_mass": None},
                      os.path.expanduser(a.assets), read, write)
        sh.start(msg, truth["jitter"])
        drv = R.Driver(sh)
        drv.MAX_STEPS = 10 ** 6
        try:
            while True:
                req = sh.obs.get("request")
                if req is None:
                    drv.tick(drv.site_pos(), drv.site_R(), -1.0)
                    continue
                rid = req["id"]
                text = req["text"]
                try:
                    run_request(drv, text)
                except R.PrimitiveError as exc:
                    log("request %d failed: %s" % (rid, exc))
                while sh.obs.get("request") is not None and sh.obs["request"]["id"] == rid:
                    drv.tick(drv.site_pos(), drv.site_R(), -1.0)
        except R.EpisodeOver as over:
            msg = over.msg
            continue


LOCS = {"rear left": "rear_left", "rear centre": "rear_center", "rear right": "rear_right",
        "front left": "front_left", "front centre": "front_center", "front right": "front_right",
        "centre": "center", "left": "left", "right": "right"}


def run_request(drv, text):
    """Parse the request's text (episode.py's templates) and run the primitive."""
    t = text.lower().rstrip(".")
    if t.startswith("turn on the ") and t.endswith(" burner"):
        return drv.knob(LOCS[t[len("turn on the "):-len(" burner")]], "on")
    if t.startswith("turn off the ") and t.endswith(" burner"):
        return drv.knob(LOCS[t[len("turn off the "):-len(" burner")]], "off")
    if t == "preheat the oven":
        return drv.knob("temp", "on")
    if t == "turn off the oven":
        return drv.knob("temp", "off")
    if t == "open the oven door":
        return drv.door("open")
    if t == "start the microwave":
        return drv.button("start_button")
    if t == "stop the microwave":
        return drv.button("stop_button")
    raise R.PrimitiveError("parse", "unknown request: " + text)


if __name__ == "__main__":
    main()
