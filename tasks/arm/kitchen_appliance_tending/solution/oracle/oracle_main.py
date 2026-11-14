"""The privileged reference as a submission: speaks the grading wire on stdin/stdout (P3).

READY, then per episode: `reset` -> ok; `act` queries until the episode ends; `close` -> ok, exit. In a
token run every observation carries `truth` (the unit's placement jitter, its joint values, the
microwave's flag) and `unit` names the unit; the reference builds its own copy of the scene from them
(reference.Shadow) and runs the primitive for each open request. Logs go to stderr.

    bash run.sh  ->  python3 -I oracle_main.py --units units.json      (units.json: id -> model, kind, dir)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appliance_env as E  # noqa: E402
import reference as R  # noqa: E402


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="JSON: unit id -> {model, kind, dir}")
    ap.add_argument("--assets", default=None, help="a folder of <kind folder>/<model> models, instead of each dir")
    a = ap.parse_args()
    units = json.load(open(a.units))
    base = os.path.dirname(os.path.abspath(a.units))
    specs = {}
    for uid, u in units.items():
        if a.assets:
            specs[uid] = E.unit_spec(u["model"])
        else:
            specs[uid] = E.unit_spec(u["model"], files=E.read_unit_files(os.path.join(base, u["dir"])))
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
        sh = R.Shadow(specs[obs["unit"]], os.path.expanduser(a.assets) if a.assets else None, read, write)
        sh.start(msg, truth["jitter"])
        drv = R.Driver(sh)
        drv.MAX_STEPS = 10 ** 6
        try:
            while True:
                rid = int(sh.obs.get("request_id", -1))
                if rid < 0:
                    drv.tick(drv.site_pos(), drv.site_R(), -1.0)
                    continue
                text = sh.obs["request_text"]
                try:
                    run_request(drv, text)
                except R.PrimitiveError as exc:
                    log("request %d failed: %s" % (rid, exc))
                while int(sh.obs.get("request_id", -1)) == rid:
                    drv.tick(drv.site_pos(), drv.site_R(), -1.0)
        except R.EpisodeOver as over:
            msg = over.msg
            continue


LOCS = {"rear left": "rear_left", "rear centre": "rear_center", "rear right": "rear_right",
        "front left": "front_left", "front centre": "front_center", "front right": "front_right",
        "centre": "center", "left": "left", "right": "right"}


def burner_request(text):
    """A burner request's (want, locations): one or two named burners, or None for 'all the burners'.
    None for any other request."""
    t = text.lower().rstrip(".")
    if t == "turn off all the burners":
        return "off", None
    for want in ("on", "off"):
        head = "turn %s the " % want
        if not t.startswith(head):
            continue
        body = t[len(head):]
        if body.endswith(" burners") and " and " in body:
            a, b = body[:-len(" burners")].split(" and ")
            return want, [LOCS[a], LOCS[b]]
        if body.endswith(" burner"):
            return want, [LOCS[body[:-len(" burner")]]]
    return None


def run_request(drv, text):
    """Parse the request's text (episode.py's templates) and run the primitive (one knob turn per named
    burner; for 'all the burners', every burner that is on)."""
    t = text.lower().rstrip(".")
    if t.startswith("turn the ") and " burner to " in t:
        where, setting = t[len("turn the "):].split(" burner to ")
        return drv.knob(LOCS[where], "set", setting=setting)
    if t.startswith("set the oven to ") and t.endswith(" heat"):
        return drv.knob("temp", "set", setting=t[len("set the oven to "):-len(" heat")])
    br = burner_request(text)
    if br is not None:
        want, locs = br
        unit = drv.env.unit
        if locs is None:
            locs = sorted(unit.knobs)
        out = None
        for loc in locs:
            if want == "off" and not unit.burner_on(drv.d, loc):
                continue
            out = drv.knob(loc, want)
        return out
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
