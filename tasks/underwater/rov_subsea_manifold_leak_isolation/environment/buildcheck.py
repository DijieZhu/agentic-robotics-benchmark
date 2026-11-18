"""Build-time proof of the task image (run once by the Dockerfile, then deleted from the image).

    python3 -B -s buildcheck.py practice-cell        as the practice account: a practice cell is built and renders
    python3 -B -s buildcheck.py agent-client DIR     as the agent account: the client library imports from DIR

`practice-cell` builds the training tank exactly as the practice service builds a run (the facility package under its
sha256 guard, a session, the vehicle cell in practice mode), steps it 0.1 s, takes one 640 x 480 camera frame and one
short sonar sweep, and checks the frame decodes to a non-blank RGB image: the rendering path (OSMesa), the cell, the
simulation kit, the pinned meshes and the task values are all present and readable by the practice account.
"""

import base64
import json
import os
import sys

PRACTICE = os.environ.get("SUBSEACELL_PRACTICE_ROOT", "/opt/subseacell-practice")


def practice_cell():
    for p in (os.path.join(PRACTICE, "kit"), os.path.join(PRACTICE, "cell"), PRACTICE):
        sys.path.insert(0, p)
    import numpy as np
    import facility as PF
    import subseacell.budget as BU
    import subseacell.client as CL
    import subseacell.server as SV
    with open(os.path.join(PRACTICE, "sites", "sites.json")) as f:
        sites = json.load(f)
    e = [x for x in sites["facilities"] if x["name"] == "tank"][0]
    fac = PF.FacilitySite.load(e["package"], os.path.join(PRACTICE, "sites"), e["sha256"])
    ses = fac.session(1, None)
    s = SV.SubseaCellServer(values=BU.load_task_values(os.path.join(PRACTICE, "task_values.json")),
                            config={"budgets": {"sim_time_s": float(e["max_sim_s"])}}, mode="practice",
                            allow_reset=True, privileged=False, render=True, world_factory=PF.world_factory,
                            practice_sites={}, site_guard=PF.server_site_guard)
    s.reset(ses, seed=1, kind="practice", name="tank")
    r = s.handle("step", {"dt": 0.1})
    assert "error" not in r, r
    f = s.handle("camera_frame", {"size": [640, 480]})
    assert "error" not in f and int(f["width"]) == 640 and int(f["height"]) == 480, f
    img = CL.decode_png(base64.b64decode(f["image_png_b64"]))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8 and float(img.std()) > 1.0, (img.shape, float(img.std()))
    sn = s.handle("sonar_scan", {"start_deg": -9.0, "stop_deg": 9.0, "range_m": 10.0})
    assert "error" not in sn and int(sn["n_pings"]) > 0, sn
    s.close()
    print("[subsea] practice cell OK: tank built, 640x480 frame (mean %.1f, std %.1f), %d pings" % (
        float(img.mean()), float(img.std()), int(sn["n_pings"])))


def agent_client(d):
    sys.path.insert(0, d)
    import subseacell.client as c
    assert callable(c.launcher_argv) and callable(c.connect_stdio) and callable(c.connect_practice)
    assert sorted(m for m in sys.modules if m.startswith("subseacell")) == ["subseacell", "subseacell.client"]
    print("[subsea] agent-UID client import OK")


if __name__ == "__main__":
    if sys.argv[1] == "practice-cell":
        practice_cell()
    elif sys.argv[1] == "agent-client":
        agent_client(sys.argv[2])
    else:
        raise SystemExit("unknown check " + sys.argv[1])
