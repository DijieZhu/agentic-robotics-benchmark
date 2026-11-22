"""The full-assembly graders that run a submitted program in their cell (robotic_arm_full_assembly_so101_panda,
robotic_arm_full_assembly_openarm_panda, quadruped_full_assembly_12dof_ur10e, tensegrity_strut_assembly_3bar_panda):
nothing the program can see at run time is named by the episode's seed.

tests/verify.py's run_episode runs the program once per hidden episode from a run folder that the grader's runner lays
out under its runs root: the program's working directory, HOME, TMPDIR and PYTHONPATH lie in it and the launcher's argv
names it. The program's stderr is a file that the grader opens and hands to the program's process. The episode's seed
draws the episode and stays with the grader, so neither name may carry it: both are named by the episode's index (the
trace tag agent_epNN).

Two checks per task, stdlib only (tests/verify.py imports the grading stack, which the host venv does not have):
  * the source: in run_episode, no `seed` in the run tag given to rn.prepare nor in the stderr_path given to the
    program's process;
  * the behaviour: run_episode alone, taken from tests/verify.py by name and run against stand-ins for the cell, the
    runner, the wire and the network audit (no simulation, no process), with two different seeds and the same index:
    the run tag, the stderr path, the working directory and the environment hold no digit string of either seed, and
    the run tag and the stderr path are the same for both.
"""
import ast
import os
import sys
import time
import traceback
import types
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
TASKS = ("arm/robotic_arm_full_assembly_so101_panda", "arm/robotic_arm_full_assembly_openarm_panda",
         "quadruped/quadruped_full_assembly_12dof_ur10e", "soft/tensegrity_strut_assembly_3bar_panda")
SEEDS = ((1 << 62) + 1234567891, 987654321987)        # two 63-bit episode seeds, the same episode index
SERVERS = ("FullArmInfoServer", "OpenArmServer", "QuadInfoServer", "StrutInfoServer")


def _verify(task):
    return REPO / "tasks" / task / "tests" / "verify.py"


def _run_episode_node(task):
    src = _verify(task).read_text(encoding="utf-8")
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "run_episode"]
    assert len(fns) == 1, "%s: verify.py defines run_episode %d times" % (task, len(fns))
    return src, fns[0]


class _Server:
    """The cell: built, reset and finalized; nothing is simulated."""

    def __init__(self, *a, **k):
        self.done = True

    def reset(self, seed, **k):
        return {"ok": True}

    def _end(self, reason):
        self.done = True

    def finalize(self, *a, **k):
        return {}


class _Audit:
    """netaudit.SocketSampler / NetCounters: context managers with empty counts."""

    max_seen = samples = total = 0
    delta = {}

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Runner:
    """The verify-side Runner: records the run tag; the folders it hands out lie under the tag, as rn.prepare's do."""

    acct, name, uid, cpu_s = None, None, None, 1.0

    def __init__(self):
        self.tags = []

    def prepare(self, tag, program_dir, private=False):
        self.tags.append(tag)
        root = os.path.join("/opt/arb/runs", tag)
        return dict({k: os.path.join(root, k) for k in ("program", "lib", "home", "tmp")}, root=root, skipped=0)

    def env(self, dirs, extra=None):
        return {"HOME": dirs["home"], "TMPDIR": dirs["tmp"], "PYTHONPATH": dirs["lib"]}

    def snapshot(self, exclude):
        return None

    def cleanup(self, dirs, before):
        pass


class _Proc:
    def terminate(self, grace=5.0):
        pass


def _names_seen(task, seed, trace_tag="agent_ep01"):
    """(run tag, stderr path, cwd, env) that run_episode hands the program for `seed`."""
    _, node = _run_episode_node(task)
    seen = {}
    rn = _Runner()

    def agent_process(argv, cwd=None, user=None, env=None, stderr_path=None, cpu_s=None):
        seen.update(cwd=cwd, env=env, stderr_path=stderr_path, argv=argv)
        return _Proc()

    ns = {"os": os, "sys": sys, "time": time, "traceback": traceback, "log": lambda *a, **k: None,
          "record_dir": lambda p: p, "program_source": lambda: "/home/user/submission",
          "launcher_argv": lambda python, lib, program, *a, **k: [python, "-c", "...", lib, "assemble"],
          "wire": types.SimpleNamespace(AgentProcess=agent_process, serve_episode=lambda *a, **k: "eof"),
          "netaudit": types.SimpleNamespace(SocketSampler=_Audit, NetCounters=_Audit),
          "episode_trace": types.SimpleNamespace(EpisodeTrace=None),
          "GT": "/stage/gt", "PANDA": "/opt/arb/given/panda", "PANDA_DIR": "/opt/arb/given/panda",
          "ROBOT_DIR": "/opt/arb/given/robot", "CACHE": "/opt/arb/cell_cache", "LOGS": "/opt/arb-verify/logs",
          "OUT": "/opt/arb-verify/out"}
    ns.update({name: _Server for name in SERVERS})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_verify(task)), "exec"), ns)
    fn = ns["run_episode"]
    args = {"cfg": {"wall_cap_s": 1.0, "privileged_wall_cap_s": 1.0, "silence_s": 1.0}, "seed": seed, "rn": rn,
            "privileged": False, "kit_info": {}, "plan": {}, "trace_tag": trace_tag}
    params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    fn(**{p: args[p] for p in params})
    assert len(rn.tags) == 1, rn.tags
    return rn.tags[0], seen["stderr_path"], seen["cwd"], seen["env"]


class RunNamesCarryNoSeed(unittest.TestCase):
    def test_source_names_use_no_seed(self):
        for task in TASKS:
            with self.subTest(task=task):
                src, node = _run_episode_node(task)
                calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
                prep = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "prepare"]
                self.assertEqual(len(prep), 1, task)
                logs = [kw.value for c in calls for kw in c.keywords if kw.arg == "stderr_path"]
                self.assertEqual(len(logs), 1, task)
                for expr in [prep[0].args[0]] + logs:
                    names = {x.id for x in ast.walk(expr) if isinstance(x, ast.Name)}
                    self.assertNotIn("seed", names, ast.get_source_segment(src, expr))

    def test_what_the_program_sees_is_named_by_the_index(self):
        for task in TASKS:
            with self.subTest(task=task):
                runs = [_names_seen(task, s) for s in SEEDS]
                for (tag, err, cwd, env), seed in zip(runs, SEEDS):
                    seen = [tag, err, cwd] + sorted(env.values())
                    for text in seen:
                        self.assertNotIn(str(seed), text, (task, text))
                        self.assertNotIn(str(seed & 0xFFFFFFFF), text, (task, text))
                        self.assertNotIn("%x" % seed, text, (task, text))
                    self.assertIn("agent_ep01", tag, (task, tag))
                    self.assertTrue(err.endswith("agent_ep01.log"), (task, err))
                self.assertEqual(runs[0][:2], runs[1][:2], task)


if __name__ == "__main__":
    unittest.main()
