"""The privileged reference program of task 07 (package P15: the core seven items; P16 adds steps_contact.py).

IN THE CELL.  The cell's `reference_run` (server.SubseaCellServer, only on a token-privileged run) loads this file and
calls run(server), which runs steps_contact.ContactProgram (P16: the eleven items; P15's CoreProgram is its base
class) - decision P16-1, P24a.  Everything the program does goes through moves.Cell (the 17 public calls +
get_state_privileged, and the world's per-control-period hook list as the clock of its own servo).  Returns
{"summary", "program"}.
Needs the perception package (DEV/perception: next to this folder's parent, in this folder, or in
$SUBSEACELL_PERCEPTION_DIR) and the manual data (moves.manual_paths).

MAINTAINER SIDE (not used by the cell; local paths):
    PY = the maintainer tools interpreter (-I -B), with the software-rendering environment loaded
    $PY reference.py one SITE.tar.xz OUT.json [--seed N]           # one privileged run, scored (P10's scoring)
    $PY reference.py batch OUT_DIR --cells ABCDEF --k 0 --repeat 2 [--procs 6] [--sites-dir D]
    $PY reference.py audit OUT.json                                # static audit (AST) of the in-cell code
    $PY reference.py collect OUT_DIR BUILD_JSON                     # acceptance numbers -> build/reference_core_runs.json
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_ITEMS = ("M1", "M2", "H", "I1", "I2", "I3", "R")


def _paths():
    """Make moves / steps_core and the perception package importable (in the cell and on the maintainer side)."""
    cand = [os.environ.get("SUBSEACELL_PERCEPTION_DIR"), HERE, os.path.dirname(HERE)]
    for c in cand:
        if c and os.path.isfile(os.path.join(c, "perception", "readers.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            break
    else:
        raise ImportError("the reference needs the perception package")
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def run(server):
    """Entry point of the cell's reference_run: the eleven-item program (steps_contact.ContactProgram, decision P16-1)."""
    _paths()
    import steps_contact as SX
    return SX.run(server)


# ====================================================================================================== maintainer
def _dev_paths():
    DEV = os.path.normpath(os.path.join(HERE, ".."))
    SH = os.path.normpath(os.path.join(DEV, "..", "..", "shared"))
    for p in (os.path.join(DEV, "cell"), SH):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("UWKIT_SPEC_VALUES", os.path.join(SH, "spec_values.json"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return DEV, SH


def _sha(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _canon(obj) -> bytes:
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


WALL_KEYS = ("wall", "wall_s", "prog_cpu_s", "cpu_s")


def _strip_wall(x):
    if isinstance(x, dict):
        return {k: _strip_wall(v) for k, v in x.items() if k not in WALL_KEYS}
    if isinstance(x, list):
        return [_strip_wall(v) for v in x]
    return x


def run_site(site_path, out_json=None, seed=None, log_dir=None):
    """One privileged reference run on a site package, scored with the grader's scoring (P10)."""
    import json
    import time
    import numpy as np
    DEV, SH = _dev_paths()
    from subseacell import server as SV, scoring as SCO, events as EVL, budget as BU
    w0, c0 = time.perf_counter(), time.process_time()
    values = BU.load_task_values(BU.default_values_path())
    S = SV.SubseaCellServer(values=values, privileged=True, render=True, reference_dir=HERE, log=print)
    S.reset(site_path, seed=seed)
    build_s = time.perf_counter() - w0
    reply = S.handle("reference_run", {})
    prog = (S.reference_result or {}).get("program") or {}
    term = SCO.capture_terminal(S.world)
    rec = S.finalize()
    log = S.event_log().export()
    rules = SCO.Rules.from_values(values)
    res = SCO.score_station(site_path, log, term, rules)
    st = SCO.site_truth(site_path, rules)
    # ---------------------------------------------------------------- acceptance quantities
    names = log["meta"]["valve_names"]
    a0 = log["meta"]["angles0"]
    iso = list(st["isolation_set"])
    defl = {}
    for n in names:
        a = EVL.column(log, f"valve_{n}_deg")
        defl[n] = float(np.max(np.abs(a - float(a0[n])))) if len(a) else 0.0
    others = {n: round(v, 4) for n, v in defl.items() if n not in iso}
    max_other = max(others.values()) if others else 0.0
    order = list(prog.get("call_order") or [])
    motion = ("idle", "set_attitude", "set_velocity", "set_hold", "set_wrench")
    first_motion = min([order.index(m) for m in motion if m in order] or [len(order)])
    sensors_first = dict(order_before_first_motion=order[:first_motion],
                         ok=all(k in order[:first_motion] for k in ("state", "camera_frame", "sonar_scan")),
                         sizes=(prog.get("sensors_first") or {}).get("frame640"),
                         sizes320=(prog.get("sensors_first") or {}).get("frame320"))
    ev_frames = log.get("frames") or []
    sizes_first = sorted({tuple(f["size"]) for f in ev_frames if float(f["t"]) <= 1e-9})
    sensors_first["frames_at_t0"] = [list(s) for s in sizes_first]
    # diagnostics: for every valve position never imaged, its closest frame records (the predicate's raw quantities)
    thr = SCO.predicate_thresholds(log["meta"], rules)
    not_imaged = {}
    for tag in [t for t in names if t not in res["details"]["valves_imaged"]]:
        recs = []
        for k, fr in enumerate(ev_frames):
            pr = (fr.get("plates") or {}).get(tag)
            if pr and pr.get("in_front"):
                tab = SCO._cond_table(pr, float(fr["lamp"]), thr)
                recs.append(dict(frame=k, t=fr["t"], lamp=fr["lamp"], tilt=fr.get("tilt_deg"),
                                 failed=[c[0] for c in tab if not c[1]],
                                 raw={c[0]: c[2] for c in tab}))
        recs.sort(key=lambda x: (len(x["failed"]), x["raw"].get("range", 9)))
        not_imaged[tag] = recs[:3]
    protect_events = [{k: e[k] for k in sorted(e) if k not in ("src",)} for e in log.get("events") or ()
                      if e.get("src") == "protect"][:20]
    hard_contacts = [e for e in log.get("events") or () if e.get("src") == "contact" and e.get("ev") == "end"
                     and float(e.get("peak_F_N") or 0.0) > 60.0][:20]
    allowed = set(SV.PUBLIC_METHODS) | {"get_state_privileged", "reference_run"}
    calls = dict(sorted(S.calls.items()))
    items = res["items"]
    core = {k: bool(items[k]) for k in CORE_ITEMS}
    core_points = int(sum(rules.weights_c[k] for k in CORE_ITEMS if items[k]))
    out = dict(
        site=os.path.basename(str(site_path)), site_label=rec.get("site_label"), site_sha256=rec.get("site_sha256"),
        sensor_seed=rec.get("sensor_seed"), leak_branch=st["leak_branch"], isolation_set=iso,
        branch_chosen=prog.get("branch_chosen"), core_items=core, core_points=core_points,
        core_score=None if core_points is None else core_points / 100.0, items=items, score_all=res["score"],
        critical_errors=res["details"]["critical_errors"], misoperated=res["details"]["misoperated"],
        n_imaged=res["details"]["n_imaged"], valves_imaged=res["details"]["valves_imaged"], not_imaged=not_imaged,
        protective_stops=res["details"]["protective_stops"], disabled=res["details"]["disabled"],
        protect_events=protect_events, hard_contacts=hard_contacts,
        max_deflection_outside_iso_deg=round(max_other, 4), deflection_outside_iso_deg=others,
        final_angles={k: res["details"]["final_angles"][k] for k in iso},
        marker=res["details"]["marker"], vehicle=res["details"]["vehicle"],
        sensors_first=sensors_first, calls=calls, calls_ok=set(calls) <= allowed | {"_unknown"} and "_unknown" not in calls,
        illegal_calls=rec.get("illegal_call_count"), internal_errors=rec.get("internal_errors"),
        done_reason=rec.get("done_reason"), cell_fault=rec.get("cell_fault"), sim_time_s=rec.get("sim_time_s"),
        budget=rec.get("budget"), frames=rec.get("frames"), pings=rec.get("pings"),
        events_sha256=rec.get("events_sha256"), fingerprint=rec.get("fingerprint"),
        program_sha256=_sha(_canon(_strip_wall(prog))), reply=reply,
        program=prog, details=res["details"], wall_s=round(time.perf_counter() - w0, 1), build_s=round(build_s, 1),
        cpu_s=round(time.process_time() - c0, 1), cell_work_s=rec.get("cell_work_s"))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        S.event_log().save(log_dir)
    S.close()
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, sort_keys=True, default=str)
    return out


def _site_path(sites_dir, cell, k):
    return os.path.join(sites_dir, f"site_{cell}_{k}.tar.xz")


def cmd_batch(argv):
    import argparse
    import subprocess
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--cells", default="ABCDEF")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--sites-dir", default=os.path.normpath(os.path.join(HERE, "..", "build", "dev_sites")))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = []
    for rep in range(a.repeat):
        for c in a.cells:
            out = os.path.join(a.out_dir, f"run_{c}_{a.k}_r{rep}.json")
            if os.path.isfile(out):
                continue
            jobs.append((c, rep, out))
    running = []
    py = sys.executable
    while jobs or running:
        while jobs and len(running) < a.procs:
            c, rep, out = jobs.pop(0)
            lg = open(out.replace(".json", ".log"), "w")
            p = subprocess.Popen([py, "-I", "-B", os.path.abspath(__file__), "one", _site_path(a.sites_dir, c, a.k), out],
                                 stdout=lg, stderr=subprocess.STDOUT)
            running.append((p, lg, c, rep, time.time()))
            print(f"start {c} r{rep} pid {p.pid}", flush=True)
        time.sleep(5)
        for x in list(running):
            if x[0].poll() is not None:
                x[1].close()
                running.remove(x)
                print(f"done {x[2]} r{x[3]} rc {x[0].returncode} {time.time() - x[4]:.0f} s", flush=True)


def cmd_one(argv):
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-dir", default=None)
    a = ap.parse_args(argv)
    r = run_site(a.site, a.out, seed=a.seed, log_dir=a.log_dir)
    print(json.dumps({k: r[k] for k in ("site", "leak_branch", "branch_chosen", "core_items", "core_score",
                                        "critical_errors", "n_imaged", "protective_stops",
                                        "max_deflection_outside_iso_deg", "sim_time_s", "wall_s")}, default=str))


# ====================================================================================================== static audit
IN_CELL_FILES = ("moves.py", "steps_core.py", "steps_contact.py", "reference.py")
ALLOWED_IMPORTS = {"__future__", "base64", "json", "math", "os", "struct", "zlib", "time", "sys", "io", "traceback",
                   "numpy", "scipy", "perception", "moves", "steps_core", "steps_contact", "PIL"}
FORBIDDEN_IMPORTS = {"subseacell", "uwkit", "mujoco", "cellkit", "world", "server", "scoring", "site", "events", "gc",
                     "inspect", "ctypes", "pickle", "importlib"}
INTROSPECTION = {"__dict__", "__globals__", "__self__", "__class__", "__code__", "__closure__", "__builtins__",
                 "__subclasses__", "__getattribute__", "f_back", "f_globals", "f_locals"}
DECISION_FUNCS = {"steps_core.py": ("CoreProgram.targets", "CoreProgram.wait_window", "PanelBook.layout",
                                    "PanelBook.handle", "PanelBook.position")}
PRIV_NAMES = {"truth", "priv", "start_truth", "last_truth", "scene", "sc", "_handle", "get_state_privileged",
              "PRIVILEGED_CALL"}


def _funcs(tree):
    out = {}
    for node in ast_walk_defs(tree):
        out[node[0]] = node[1]
    return out


def ast_walk_defs(tree, prefix=""):
    import ast
    res = []
    for n in tree.body if hasattr(tree, "body") else []:
        if isinstance(n, ast.ClassDef):
            res += ast_walk_defs(n, prefix + n.name + ".")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            res.append((prefix + n.name, n))
    return res


def static_audit() -> dict:
    """AST audit of the in-cell code (moves.py, steps_core.py, steps_contact.py's in-cell part, reference.run) and of
    the perception package."""
    import ast
    rep = dict(files={}, problems=[])
    PUBLIC = None
    for fn in IN_CELL_FILES:
        path = os.path.join(HERE, fn)
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        info = dict(imports=set(), server_attrs=set(), world_attrs=set(), introspection=[], open_calls=[],
                    handle_literals=set(), dynamic_getattr=[])
        scope = tree
        if fn == "reference.py":
            # in-cell part = module-level imports + _paths() + run(); the maintainer runner imports lazily
            defs = dict(ast_walk_defs(tree))
            nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))] + [defs["_paths"], defs["run"]]
        elif fn == "steps_contact.py":
            # in-cell part = the module less the maintainer functions it lists in MAINT_FUNCS and its __main__ block
            # (the same split as steps_contact.static_audit; P24a, decision P16-1)
            maint = set()
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAINT_FUNCS" for t in n.targets):
                    maint = set(ast.literal_eval(n.value))
            if not maint:
                rep["problems"].append("steps_contact.py: no MAINT_FUNCS list")
            nodes = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in maint)
                     and not isinstance(n, ast.If)]
        else:
            nodes = [tree]
        for root in nodes:
            for n in ast.walk(root):
                if isinstance(n, ast.Import):
                    for a in n.names:
                        info["imports"].add(a.name.split(".")[0])
                elif isinstance(n, ast.ImportFrom):
                    if n.level == 0 and n.module:
                        info["imports"].add(n.module.split(".")[0])
                elif isinstance(n, ast.Attribute):
                    if isinstance(n.value, ast.Name) and n.value.id == "server":
                        info["server_attrs"].add(n.attr)
                    if isinstance(n.value, ast.Attribute) and n.value.attr == "world" and \
                            isinstance(n.value.value, ast.Name) and n.value.value.id == "server":
                        info["world_attrs"].add(n.attr)
                    if n.attr in INTROSPECTION:
                        info["introspection"].append((n.attr, n.lineno))
                elif isinstance(n, ast.Call):
                    f = n.func
                    if isinstance(f, ast.Name) and f.id == "open":
                        info["open_calls"].append(n.lineno)
                    own = (isinstance(f, ast.Name) and f.id == "getattr" and n.args and isinstance(n.args[0], ast.Name)
                           and n.args[0].id == "self")          # the program's own step table (CoreProgram.run_core)
                    if isinstance(f, ast.Name) and f.id in ("getattr", "setattr", "vars", "globals", "locals", "eval",
                                                             "exec", "__import__", "compile") and not own:
                        info["dynamic_getattr"].append((f.id, n.lineno))
                    if isinstance(f, ast.Attribute) and f.attr in ("_handle", "handle") and n.args and \
                            isinstance(n.args[0], ast.Constant):
                        info["handle_literals"].add(n.args[0].value)
                elif isinstance(n, ast.Name) and n.id == "PUBLIC_CALLS" and fn == "moves.py":
                    pass
        if fn == "moves.py":
            mod = ast.parse(src)
            for n in mod.body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC_CALLS" for t in n.targets):
                    PUBLIC = tuple(ast.literal_eval(n.value))
        bad = sorted(info["imports"] - ALLOWED_IMPORTS)
        forb = sorted(info["imports"] & FORBIDDEN_IMPORTS)
        if bad:
            rep["problems"].append(f"{fn}: imports outside the allowed set: {bad}")
        if forb:
            rep["problems"].append(f"{fn}: forbidden imports: {forb}")
        if info["server_attrs"] - {"handle", "world"}:
            rep["problems"].append(f"{fn}: server attributes {sorted(info['server_attrs'])}")
        if info["world_attrs"] - {"hooks"}:
            rep["problems"].append(f"{fn}: world attributes {sorted(info['world_attrs'])}")
        if info["introspection"]:
            rep["problems"].append(f"{fn}: introspection {info['introspection']}")
        if info["dynamic_getattr"]:
            rep["problems"].append(f"{fn}: dynamic attribute / code access {info['dynamic_getattr']}")
        rep["files"][fn] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in info.items()}
    # the server is touched in exactly one class (moves.Cell.__init__)
    mv_src = open(os.path.join(HERE, "moves.py"), encoding="utf-8").read()
    mv = ast.parse(mv_src)
    touch = []
    for name, node in ast_walk_defs(mv):
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "server":
                touch.append((name, n.attr))
    rep["server_touch"] = sorted(set(touch))
    if sorted(set(touch)) != [("Cell.__init__", "handle"), ("Cell.__init__", "world")]:
        rep["problems"].append(f"moves.py: the server is touched outside Cell.__init__: {touch}")
    # the wire: every _handle call passes a checked name (Cell.call: PUBLIC_CALLS guard; Cell.priv: the constant)
    handle_calls = []
    for name, node in ast_walk_defs(mv):
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_handle":
                arg = n.args[0]
                handle_calls.append((name, arg.id if isinstance(arg, ast.Name) else ast.dump(arg)))
    rep["handle_calls"] = handle_calls
    if sorted(handle_calls) != [("Cell.call", "name"), ("Cell.priv", "PRIVILEGED_CALL")]:
        rep["problems"].append(f"moves.py: unexpected _handle call sites {handle_calls}")
    call_src = ast.get_source_segment(mv_src, dict(ast_walk_defs(mv))["Cell.call"])
    if "if name not in PUBLIC_CALLS" not in call_src:
        rep["problems"].append("moves.py: Cell.call does not guard the public call list")
    # privileged reply: only the vehicle / tcp / structure / panel / cage / mudline fields are read
    keys = set()
    for cname in ("Truth.__init__", "Scene.__init__"):
        for n in ast.walk(dict(ast_walk_defs(mv))[cname]):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                keys.add(n.slice.value)
    rep["privileged_keys_read"] = sorted(keys)
    allowed_keys = {"t", "vehicle", "tcp", "structure", "panel", "cage", "mudline_z", "pos_w", "quat_wxyz", "vel_b",
                    "omega_b", "R_w", "origin_w", "deck_top_w"}
    if keys - allowed_keys:
        rep["problems"].append(f"moves.py: privileged keys outside the vehicle / structure poses: {sorted(keys - allowed_keys)}")
    # the hook wrapper never reads its world argument
    for n in ast.walk(dict(ast_walk_defs(mv))["Cell.add_tick"]):
        if isinstance(n, ast.FunctionDef) and n.name == "_hook":
            used = {m.id for m in ast.walk(n) if isinstance(m, ast.Name)}
            rep["hook_world_arg_used"] = "_world" in used
            if "_world" in used:
                rep["problems"].append("moves.py: the hook reads its world argument")
    # decision functions: no privileged source
    sc_src = open(os.path.join(HERE, "steps_core.py"), encoding="utf-8").read()
    sc = ast.parse(sc_src)
    defs = dict(ast_walk_defs(sc))
    dec = {}
    for fname in DECISION_FUNCS["steps_core.py"]:
        names = {m.id for m in ast.walk(defs[fname]) if isinstance(m, ast.Name)} | \
                {m.attr for m in ast.walk(defs[fname]) if isinstance(m, ast.Attribute)}
        hit = sorted(names & PRIV_NAMES)
        dec[fname] = dict(privileged_refs=hit)
        if hit:
            rep["problems"].append(f"steps_core.py: decision function {fname} references {hit}")
    rep["decision_functions"] = dec
    # where each data source is read (table for the record)
    src_table = {}
    for fname, node in sorted(defs.items()):
        names = {m.attr for m in ast.walk(node) if isinstance(m, ast.Attribute)} | \
                {m.id for m in ast.walk(node) if isinstance(m, ast.Name)}
        consts = {m.value for m in ast.walk(node) if isinstance(m, ast.Constant) and isinstance(m.value, str)}
        src_table[fname] = sorted(
            ({"truth"} if names & {"truth", "start_truth", "last_truth"} else set())
            | ({"scene"} if names & {"sc", "scene"} else set())
            | ({"state"} if "state" in names or "state" in consts else set())
            | ({"frames"} if names & {"frame", "shoot", "read_panel"} or "camera_frame" in consts else set())
            | ({"pings"} if names & {"scan", "scan360", "sector", "locate"} or "sonar_scan" in consts else set())
            | ({"mission"} if "mission" in consts else set()))
    rep["sources_by_function"] = src_table
    # perception: numpy / scipy / stdlib only
    pdir = os.path.join(os.path.dirname(HERE), "perception")
    pimp = set()
    for f in sorted(os.listdir(pdir)):
        if f.endswith(".py"):
            t = ast.parse(open(os.path.join(pdir, f), encoding="utf-8").read())
            for n in ast.walk(t):
                if isinstance(n, ast.Import):
                    pimp |= {a.name.split(".")[0] for a in n.names}
                elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                    pimp.add(n.module.split(".")[0])
    rep["perception_imports"] = sorted(pimp)
    if pimp & FORBIDDEN_IMPORTS:
        rep["problems"].append(f"perception imports {sorted(pimp & FORBIDDEN_IMPORTS)}")
    rep["public_calls"] = list(PUBLIC or ())
    rep["ok"] = not rep["problems"]
    return rep


def cmd_audit(argv):
    import json
    rep = static_audit()
    if argv:
        with open(argv[0], "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, sort_keys=True, default=str)
    print(json.dumps(dict(ok=rep["ok"], problems=rep["problems"], server_touch=rep["server_touch"],
                          privileged_keys=rep["privileged_keys_read"]), indent=1, default=str))
    return 0 if rep["ok"] else 1


# ====================================================================================================== collect
def cmd_collect(argv):
    import glob
    import hashlib
    import json
    out_dir, build = argv[0], argv[1]
    runs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "run_*_r*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        b = os.path.basename(p)[:-5].split("_")
        runs.setdefault(b[1], {})[b[3]] = r
    audit = static_audit()
    rows, ok_all = [], True
    for cell in sorted(runs):
        R = runs[cell]
        r0 = R.get("r0")
        if r0 is None:
            continue
        rep = {k: R[k] for k in sorted(R) if k != "r0"}
        same = None
        if rep:
            r1 = rep[sorted(rep)[0]]
            same = dict(events_sha256=r0["events_sha256"] == r1["events_sha256"],
                        qpos_chain=r0["fingerprint"]["qpos_chain_sha256"] == r1["fingerprint"]["qpos_chain_sha256"],
                        valve_hist=r0["fingerprint"]["valve_hist_sha256"] == r1["fingerprint"]["valve_hist_sha256"],
                        program=r0["program_sha256"] == r1["program_sha256"])
        chk = dict(core_070=r0["core_points"] == 70, no_ce=len(r0["critical_errors"]) == 0, imaged_18=r0["n_imaged"] == 18,
                   defl_lt_2=r0["max_deflection_outside_iso_deg"] < 2.0, stops_0=r0["protective_stops"] == 0,
                   sensors_first=bool(r0["sensors_first"]["ok"]), calls_ok=bool(r0["calls_ok"]) and r0["illegal_calls"] == 0,
                   repeat_identical=None if same is None else all(same.values()))
        ok = all(v for v in chk.values() if v is not None) and same is not None
        ok_all = ok_all and ok
        rows.append(dict(cell=cell, site=r0["site_label"], site_sha256=r0["site_sha256"], leak_branch=r0["leak_branch"],
                         branch_chosen=r0["branch_chosen"], isolation_set=r0["isolation_set"],
                         final_angles_iso=r0["final_angles"], core_items=r0["core_items"], core_score=r0["core_score"],
                         critical_errors=r0["critical_errors"], n_imaged=r0["n_imaged"],
                         max_deflection_outside_iso_deg=r0["max_deflection_outside_iso_deg"],
                         protective_stops=r0["protective_stops"], sensors_first=r0["sensors_first"],
                         marker_zone_distance_m=r0["marker"]["zone_distance_m"], vehicle=r0["vehicle"],
                         sim_time_s=r0["sim_time_s"], frames=r0["frames"], pings=r0["pings"], calls=r0["calls"],
                         illegal_calls=r0["illegal_calls"], done_reason=r0["done_reason"], wall_s=r0["wall_s"],
                         cpu_s=r0["cpu_s"], steps={k: v["t"] for k, v in r0["program"]["steps"].items()},
                         valves=r0["program"].get("valves"), plume=r0["program"].get("plume"),
                         current=r0["program"].get("current"), verify=r0["program"].get("verify"),
                         landing=r0["program"].get("landing"), errors=r0["program"].get("errors"),
                         notes=r0["program"].get("notes"), events_sha256=r0["events_sha256"],
                         qpos_chain_sha256=r0["fingerprint"]["qpos_chain_sha256"], repeat=same, checks=chk, ok=ok))
    files = {}
    for fn in IN_CELL_FILES:
        with open(os.path.join(HERE, fn), "rb") as f:
            files[fn] = hashlib.sha256(f.read()).hexdigest()
    res = dict(schema="reference_core_runs/1", package="P15", acceptance_ok=bool(ok_all and audit["ok"] and len(rows) == 6),
               static_audit=dict(ok=audit["ok"], problems=audit["problems"], server_touch=audit["server_touch"],
                                 handle_calls=audit["handle_calls"], privileged_keys_read=audit["privileged_keys_read"],
                                 hook_world_arg_used=audit.get("hook_world_arg_used"),
                                 decision_functions=audit["decision_functions"],
                                 imports={k: v["imports"] for k, v in audit["files"].items()},
                                 perception_imports=audit["perception_imports"],
                                 sources_by_function=audit["sources_by_function"]),
               reference_sha256=files, stations=rows)
    with open(build, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    print(json.dumps(dict(acceptance_ok=res["acceptance_ok"], audit=audit["ok"],
                          stations=[(r["cell"], r["core_score"], r["ok"], r["checks"]) for r in rows]), indent=1, default=str))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "one":
        cmd_one(sys.argv[2:])
    elif cmd == "batch":
        cmd_batch(sys.argv[2:])
    elif cmd == "audit":
        sys.exit(cmd_audit(sys.argv[2:]))
    elif cmd == "collect":
        cmd_collect(sys.argv[2:])
    else:
        print(__doc__)
        sys.exit(2)
