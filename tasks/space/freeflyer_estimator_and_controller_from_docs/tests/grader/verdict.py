"""The verdict: written through the engine's own writer inside the verify stage, as an envelope file outside it.
Standard library only.

    write(reward, metrics, diagnostic=None, path=None)
    fail(reason)                     # a TASK ERROR, never a zero

Inside the engine's verify stage (ALE_VERIFY_PYTHON is set) the result goes through the engine's ale_verify.write:
{"reward": r, "metrics": {..., "reward_exact": r}, "diagnostic": ...}; the engine rounds the reward and writes its
own reward file. Outside it (a maintainer's run of verify.py by hand) the envelope {"rewards": {"reward": r},
"metrics": {...}} goes to the given path. The reward lies in [0, 1] and every metric is a finite number, or the run is
a task error.
"""
import json
import math
import os
import sys

ENGINE_RESULT_DIR = "/logs/verifier"


def fail(reason):
    """End the run as a task error, not as a zero."""
    print("[grader] TASK ERROR: {}".format(reason), file=sys.stderr, flush=True)
    raise SystemExit(1)


def write(reward, metrics, diagnostic=None, path=None):
    r = float(reward)
    if not (math.isfinite(r) and 0.0 <= r <= 1.0):
        fail("the reward %r is not a finite number in [0, 1]" % reward)
    m = {}
    for k, v in sorted(metrics.items()):
        x = float(v)
        if not math.isfinite(x):
            fail("the metric %s is not finite" % k)
        m[k] = x
    if "reward_exact" in m:
        fail("reward_exact is the engine's own metric")
    if path is None and os.environ.get("ALE_VERIFY_PYTHON"):
        try:
            from ale_verify import write as engine_write
        except ImportError:
            fail("the engine's verdict writer is not importable in the verify stage")
        result = {"reward": r, "metrics": dict(m, reward_exact=r)}
        if diagnostic:
            result["diagnostic"] = str(diagnostic)
        engine_write(result)
        return result
    if path is None:
        fail("outside the engine's verify stage the verdict needs a path")
    target = os.path.realpath(path)
    if target == ENGINE_RESULT_DIR or target.startswith(ENGINE_RESULT_DIR + os.sep):
        fail("the envelope is never written into the engine's result folder")
    env = {"rewards": {"reward": r}, "metrics": m}
    if diagnostic:
        env["diagnostic"] = str(diagnostic)
    with open(path + ".tmp", "w") as fh:
        json.dump(env, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)
    return env
