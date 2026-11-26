"""The flight of one hidden session: the hidden world flies the session with the submitted controller behind the
runner (the wire), and the grader collects the world's records and the runner's record into a flown session record
(records.py). The world is the session's whole truth; the runner is the only path to the controller.

    rec = fly_session(paths, entry, program_dir, run_root, address_space_bytes, recovery_s)

paths: the stage's world root (the folder holding the world package), the world's table, its session files, its seed
key and the runner's folder; entry: the hidden set's entry of the session (world_session, tight_mode, list, seed).
Before the world runs, the grader hands the node the session's records of the blocks' ends (read_block_ends_from), so
that the runner ends a block's stretch where the frame cannot show the block's end. The controller's process is always
ended when the session ends (node.close), whatever raised.

reads_at_setup (the controller folder holds the solution's access token, grader/access.py): the node is the runner's
WireNode with one difference: the session paper it sends with the set-up call (at every start of the session) carries
one more key, READS_KEY, holding what the world hands the node's set-up beside the paper (World.reference_reads(): the
session's registered reads) as plain JSON. Without it the node is the runner's WireNode exactly, and the world's second
argument of setup is not passed on.
"""
import os
import sys

import numpy as np


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


_IMPORTED = {}
READS_KEY = "registered_reads"


def load(paths):
    """the world's and the runner's modules, imported once from the stage's own folders (asserted)."""
    if _IMPORTED:
        return _IMPORTED
    for p in (paths["world_root"], os.path.dirname(paths["runner_dir"])):
        if p not in sys.path:
            sys.path.insert(0, p)
    from world import config as wcfg
    wcfg.configure(spec=paths["world_table"], sessions_dir=paths["sessions_dir"], key=paths["seed_key"])
    import world as wpkg
    import runner as rpkg
    if os.path.realpath(os.path.dirname(wpkg.__file__)) != os.path.realpath(os.path.join(paths["world_root"], "world")):
        raise RuntimeError("the world's modules were not imported from the stage's world root")
    if os.path.realpath(os.path.dirname(rpkg.__file__)) != os.path.realpath(paths["runner_dir"]):
        raise RuntimeError("the runner was not imported from the stage's runner folder")
    from world.session import World, Settings
    from runner.node import WireNode

    class ReadsAtSetup(WireNode):
        """the runner's node; its set-up's paper carries the session's registered reads under READS_KEY."""

        def setup(self, paper, *more):
            if not more or not isinstance(more[0], dict):
                raise RuntimeError("the world handed the node's set-up no registered reads")
            return WireNode.setup(self, dict(paper, **{READS_KEY: plain(more[0])}))

    _IMPORTED.update(World=World, Settings=Settings, WireNode=WireNode, ReadsAtSetup=ReadsAtSetup, V=wcfg.V)
    return _IMPORTED


def collect(W, node_log, rep):
    """the flown session record (records.py) from the world after its run and the runner's record."""
    blocks = []
    for b in W.blocks:
        jobs = []
        for j in b["jobs"]:
            jj = {k: j[k] for k in ("id", "kind", "mode", "goal") if k in j}
            for k in ("request_kind", "request_shape"):
                if k in j:
                    jj[k] = j[k]
            jobs.append(jj)
        blocks.append({"name": b["name"], "mode": b["mode"], "start_pose": b["start_pose"], "budget_s": b["budget_s"],
                       "jobs": jobs})
    jrs = []
    for r in W.jobs_out:
        d = {k: v for k, v in r.items() if k != "setpoints"}
        if r.get("kind") == "hold" and r.get("setpoints"):
            s0 = r["setpoints"][0]
            d["hold_target"] = {"position": list(s0[1]), "attitude": list(s0[2])}
        jrs.append(d)
    truth = {k: np.asarray(W.log[k], float) for k in ("t", "p", "q", "v", "w")}
    runner = dict(rep)
    runner["start_outcomes"] = [s.get("outcome") for s in (node_log.get("starts") or [])]
    return {"session": W.name, "tight_mode": W.tight, "list": W.list_no, "seed": W.seed_label,
            "blocks": plain(blocks), "job_records": plain(jrs), "block_records": plain(W.block_records),
            "ended": plain(list(W.ended)) if W.ended else None, "truth": truth, "runner": plain(runner)}


def fly_session(paths, entry, program_dir, run_root, address_space_bytes, recovery_s, reads_at_setup=False):
    mods = load(paths)
    World, Settings, V = mods["World"], mods["Settings"], mods["V"]
    WireNode = mods["ReadsAtSetup"] if reads_at_setup is True else mods["WireNode"]
    rec_table = V("monitor.recovery_s")
    if {k: float(v) for k, v in rec_table.items()} != {k: float(v) for k, v in recovery_s.items()}:
        raise RuntimeError("the world's table and the grader's configuration hold different monitor.recovery_s")
    S = Settings(recovery_s=dict(rec_table))
    W = World(entry["world_session"], entry["tight_mode"], int(entry["list"]), entry["seed"], S)
    node = WireNode(program_dir, run_root, sandbox_required=True, mem_cap_bytes=address_space_bytes,
                    stderr_path=run_root.rstrip("/") + "_stderr.log")
    node.read_block_ends_from(W)
    rep = None
    try:
        W.run(node)
    finally:
        rep = node.close()
    return collect(W, node.log, rep)
