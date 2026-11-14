"""Run the task's lab plugin through the lab server's own reply-schema check (no lab service, no privilege
drop): every reset and step reply of whole idle episodes, run past every deadline, must pass `check_value`.
Catches an episode text or field the lab's schema does not list (2026-11-13: compositional request texts
were missing, so the lab answered "internal lab error").

    MUJOCO_GL=osmesa RESETS=4 python -I lab_check.py <task>/environment/setup/lab u0 u3 u6
"""
import json, os, sys
LAB = sys.argv[1]
sys.path[:0] = [LAB, os.path.join(LAB, "alelab")]
import server as SV
import appliance_plugin as P

class API:
    class BadRequest(Exception):
        pass

inst = json.load(open(os.path.join(LAB, "..", "ref", "units.json")))
lab = P.make_plugin(inst, API)
SV.check_value(lab.SCHEMA["info"], lab.info(), "info")
texts = set()
for uid in sys.argv[2:]:
    for k in range(int(os.environ.get("RESETS", "3"))):
        e, obs = lab.reset(uid)
        SV.check_value(lab.SCHEMA["reset"], obs, "reset")
        texts.add(obs["request_text"])
        done, n = False, 0
        while not done:
            obs, done = e.step(lab.validate_action([0, 0, 0, 0, 0, 0, -1, 0]))
            SV.check_value(lab.SCHEMA["step"], obs, "step")
            texts.add(obs["request_text"])
            n += 1
        print(uid, "reset", k, "steps", n, "ok", flush=True)
print("texts seen:", sorted(texts))
