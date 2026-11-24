#!/usr/bin/env python3
"""Does the grader charge a run its own processor time, on a quiet and on a loaded host alike?

    python3 -I cpu_account_test.py <task tests dir>
Starts a burner the way verify.run_submission starts a run (own session, RIGCAL_RUN mark): the parent burns 10 s of
processor time, a child in the same session burns 10 s, a grandchild that left the session (setsid, parent gone) burns
10 s, then the parent sleeps 5 s. Accounts it with verify._proc_cpu exactly as run_submission does (session, else
mark). Expected: about 30 s charged, whatever the wall time and the host's load."""
import json, os, subprocess, sys, time, types
tests = sys.argv[1]
sys.path.insert(0, tests)
for name in ("robotics_grader", "robotics_grader.account", "robotics_grader.stage"):
    m = types.ModuleType(name)
    for k in ("agent_user", "fail", "ratio_rewards", "read_anchor", "stage_config", "write_verdict", "zero_verdict", "clear_agent_account", "secure_exec_path"):
        setattr(m, k, lambda *a, **k: None)
    sys.modules[name] = m
import verify
BURN = r'''
import os, time
def burn(s):
    t = time.process_time()
    x = 0
    while time.process_time() - t < s:
        x += sum(i * i for i in range(2000))
if os.fork() == 0:
    burn(10.0); os._exit(0)
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        burn(10.0)
    os._exit(0)
burn(10.0); time.sleep(5.0)
'''
p = subprocess.Popen([sys.executable, "-I", "-c", BURN], start_new_session=True, env={"PATH": "/usr/bin:/bin", "RIGCAL_RUN": "0"})
seen = {}; t0 = time.monotonic(); done = None
while True:
    for key, (sid, ticks, marker) in verify._proc_cpu(os.getuid()).items():
        run = 0 if (sid == p.pid or marker == 0) else -1
        old = seen.get(key)
        seen[key] = (run if old is None or old[0] == -1 else old[0], max(ticks, old[1] if old else 0))
    if p.poll() is not None and done is None:
        done = time.monotonic()
    if done is not None and time.monotonic() - done > 12.0 + 0.0:
        live = [k for k, (r, t) in seen.items() if r == 0 and os.path.exists("/proc/%d" % k[0])]
        if not live:
            break
    time.sleep(0.2)
cpu = sum(t for r, t in seen.values() if r == 0) / float(verify.CLK_TCK)
print(json.dumps({"charged_cpu_s": round(cpu, 2), "expected_cpu_s": 30.0, "wall_s": round(time.monotonic() - t0, 1), "processes": sum(1 for r, t in seen.values() if r == 0),
                  "load1": os.getloadavg()[0], "host": os.uname().nodename}))
