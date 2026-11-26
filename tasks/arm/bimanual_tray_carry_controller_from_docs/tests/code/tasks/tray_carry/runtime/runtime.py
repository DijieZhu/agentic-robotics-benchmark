import math
import os
import subprocess
import sys
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
REF = os.path.join(TASK_DIR, "ref")
for p in (HERE, REF):
    if p not in sys.path:
        sys.path.insert(0, p)

import shift as SH
from openarm_unit import hostcodec as hc

import frames as C
import wire as WI

K = SH.K
CYCLE = K.CYCLE_S
SIDES = K.SIDES
N = K.N_MODULES

ON_TIME, LATE, MALFORMED, ENDED, NOT_CALLED = 0, 1, 2, 3, 4
KIND_NAMES = {ON_TIME: "on_time", LATE: "late", MALFORMED: "malformed", ENDED: "no_controller",
              NOT_CALLED: "held_late"}
FATAL_STATUSES = ("cpu", "timeout", "dead", "protocol")


class SetupFailed(RuntimeError):

    def __init__(self, why, info):
        super().__init__(why)
        self.why, self.info = why, info


def check_modes(res):
    if not isinstance(res, dict):
        raise ValueError("setup returned no object")
    gm = res.get("gripper_mode")
    if not isinstance(gm, dict) or set(gm) != set(SIDES):
        raise ValueError("gripper_mode is an object with the keys 'left' and 'right'")
    for s in SIDES:
        if gm[s] not in K.GRIPPER_MODES:
            raise ValueError(f"gripper_mode.{s} is one of {K.GRIPPER_MODES}")
    return {s: str(gm[s]) for s in SIDES}


def planned_cycles(slots, jobs):
    cy = lambda t: int(round(t / CYCLE))
    out = cy(K.FREE_INTERVAL_S)
    for e in SH.supervisor_schedule(slots, jobs):
        out += e["length"] if e["length"] is not None else cy(0.5 + K.SLOT["decline_next_s"])
    return out


def _quantiles(v):
    if not v:
        return None
    s = sorted(v)
    q = lambda f: s[min(len(s) - 1, int(math.ceil(f * len(s))) - 1)] if f > 0 else s[0]
    return {"n": len(s), "mean": sum(s) / len(s), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": s[-1]}


class WireController:
    jobs = None

    def __init__(self, runner, unit_record, site_public, truth=None, planned=None, step_budget_s=None):
        self.runner = runner
        self.step_budget_s = None if step_budget_s is None else float(step_budget_s)
        self.budget = K.STEP_BUDGET_CPU_S if step_budget_s is None else float(step_budget_s)
        self.unit_record = unit_record
        self.site_public = site_public
        self.truth = truth
        self.planned = planned
        self.kinds = bytearray()
        self.cpu = array("d")
        self.wall = array("d")
        self.decline_cycles = []
        self.hold = 0
        self.pending_decline = None
        self.ended = None
        self.late_cycles = 0
        self.malformed = 0
        self.answered = 0
        self.last = [None] * N
        self.last_cycle = [None] * N
        self.setup_info = None
        self.step0 = None
        self.max_step_cpu = 0.0
        self.max_step_wall = 0.0
        self.t_build = 0.0

    def setup(self, unit, site, slots):
        Bu = unit.bus
        self.bus_of = [int(Bu.modules[i].bus) for i in range(N)]
        self.esc = [int(Bu.esc_id[i]) for i in range(N)]
        self.lim = [Bu.limits(i) for i in range(N)]
        self.reply_key = {(int(Bu.modules[i].bus), int(Bu.reply_id[i])): i for i in range(N)}
        payload = {"unit_record": self.unit_record, "site": self.site_public}
        if self.truth is not None:
            payload["truth"] = self.truth
        b = K.SETUP_BUDGET
        status, rep, cpu, wall = self.runner.call("setup", payload, cpu_s=b["cpu_s"], wall_s=b["wall_s"])
        self.setup_info = {"status": status, "cpu_s": cpu, "wall_s": wall,
                           "error": None if not isinstance(rep, dict) else rep.get("error")}
        if status != "ok":
            raise SetupFailed(f"setup: {status}", self.setup_info)
        try:
            modes = check_modes(rep.get("result"))
        except ValueError as e:
            self.setup_info["error"] = str(e)
            raise SetupFailed(f"setup: {e}", self.setup_info) from None
        self.setup_info["gripper_mode"] = modes
        self.modes = modes
        return {"gripper_mode": dict(modes)}

    def start(self, unit):
        self.step0 = int(unit.bus.step_count)

    def on_trace(self, recs):
        ""

    def _absorb(self, cyc, replies):
        for f in replies or ():
            i = self.reply_key.get((f.bus, f.can_id))
            if i is None:
                continue
            r = hc.unpack_state(f.data, self.lim[i])
            if r is not None:
                self.last[i] = r
                self.last_cycle[i] = cyc - 1

    def _feedback(self, cyc, sig):
        mods = []
        for i in range(N):
            r = self.last[i]
            if r is None:
                r = {"q": 0.0, "dq": 0.0, "tau": 0.0, "t_mos": 0, "t_rotor": 0, "status": 0}
                age = cyc + 1
            else:
                age = cyc - self.last_cycle[i]
            mods.append(C.module_feedback(r, age))
        return C.feedback(cyc, sig.get("tray"), sig.get("slot_cycle"), mods)

    def _encode(self, frames):
        out = []
        for i, (kind, f) in enumerate(frames):
            if kind == "mit":
                out.append(hc.pack_mit(self.bus_of[i], self.esc[i], self.lim[i], f["p"], f["v"], f["kp"], f["kd"],
                                       f["t"]))
            elif kind == "posforce":
                out.append(hc.pack_posforce(self.bus_of[i], self.esc[i], f["p"], f["v_lim"], f["i_lim"]))
        return out

    def _end(self, cyc, why, status=None, error=None):
        self.ended = {"cycle": int(cyc), "why": why, "status": status, "error": error}
        if not self.runner.dead:
            self.runner._kill(f"ended: {why}")

    def _late_stop(self, cyc):
        if self.planned is not None and self.late_cycles > K.LATE_TIME_STOP_SHARE * self.planned:
            raise SH.ReferenceAbort(f"late_time: {self.late_cycles} late cycles over {self.planned} planned at "
                                    f"cycle {cyc}")

    def step(self, cyc, sig, replies):
        tb0 = time.perf_counter()
        self._absorb(cyc, replies)
        pd = self.pending_decline is not None and self.pending_decline == cyc
        if pd:
            self.pending_decline = None
        if self.ended is not None:
            self.kinds.append(ENDED)
            return self._out([], pd, cyc, tb0)
        if self.hold > 0:
            self.hold -= 1
            self.kinds.append(NOT_CALLED)
            return self._out([], pd, cyc, tb0)
        fb = self._feedback(cyc, sig)
        self.t_build += time.perf_counter() - tb0
        sf = K.STEP_FATAL
        status, rep, cpu, wall = self.runner.call("step", {"feedback": fb}, cpu_s=sf["cpu_s"], wall_s=sf["wall_s"])
        tb0 = time.perf_counter()
        self.cpu.append(cpu)
        self.wall.append(wall)
        self.max_step_cpu = max(self.max_step_cpu, cpu)
        self.max_step_wall = max(self.max_step_wall, wall)
        if status in FATAL_STATUSES:
            self._end(cyc, "step " + status, status)
            self.kinds.append(ENDED)
            return self._out([], pd, cyc, tb0)
        reply_error = status == "error" and isinstance(rep, dict) and rep.get("reply_error")
        if status == "error" and not reply_error:
            self._end(cyc, "step raised", status, rep.get("error") if isinstance(rep, dict) else None)
            self.kinds.append(ENDED)
            return self._out([], pd, cyc, tb0)
        self.answered += 1
        b = self.budget
        late = int(cpu // b) if cpu > b else 0
        frames, decline = None, False
        if reply_error:
            kind = MALFORMED
        else:
            try:
                frames, decline = C.parse_step_reply(rep.get("result"), N)
                kind = ON_TIME
            except C.ReplyNotObject as e:
                self._end(cyc, "reply not an object", status, str(e))
                self.kinds.append(ENDED)
                return self._out([], pd, cyc, tb0)
            except C.ReplyMalformed:
                kind = MALFORMED
        if kind == MALFORMED:
            self.malformed += 1
            late = max(late, 1)
            decline = False
        if late >= 1:
            self.kinds.append(MALFORMED if kind == MALFORMED else LATE)
            self.late_cycles += late
            self.hold = late - 1
            if decline:
                self.pending_decline = cyc + late
            self._late_stop(cyc)
            return self._out([], pd, cyc, tb0)
        self.kinds.append(ON_TIME)
        return self._out(self._encode(frames), bool(decline) or pd, cyc, tb0)

    def _out(self, frames, decline, cyc, tb0):
        if decline:
            self.decline_cycles.append(int(cyc))
        self.t_build += time.perf_counter() - tb0
        return frames, bool(decline)

    def summary(self):
        n = len(self.kinds)
        with_ctl = sum(1 for k in self.kinds if k != ENDED)
        counts = {name: self.kinds.count(code) for code, name in KIND_NAMES.items()}
        on_time = counts["on_time"]
        return {"cycles": n, "cycles_with_controller": with_ctl, "counts": counts,
                "late_cycles": self.late_cycles, "malformed_replies": self.malformed,
                "on_time_share": (on_time / with_ctl) if with_ctl else 0.0,
                "steps_answered": self.answered, "ended": self.ended, "setup": self.setup_info,
                "decline_cycles": list(self.decline_cycles), "planned_cycles": self.planned,
                "step_cpu_s": _quantiles(list(self.cpu)), "step_wall_s": _quantiles(list(self.wall)),
                "max_step_cpu_s": self.max_step_cpu, "max_step_wall_s": self.max_step_wall,
                "parent_codec_s": self.t_build, "step_budget": self.step_budget_record()}

    def step_budget_record(self):
        if self.step_budget_s is None:
            return {"rule": "printed", "step_budget_cpu_s": K.STEP_BUDGET_CPU_S}
        return {"rule": "oracle (a controller whose token is valid)", "step_budget_cpu_s": self.step_budget_s}

    def figures(self):
        return {"runtime": self.summary()}

    def cycle_kinds_rle(self):
        out = []
        for c, k in enumerate(self.kinds):
            if out and out[-1][0] == KIND_NAMES[k] and out[-1][1] + out[-1][2] == c:
                out[-1][2] += 1
            else:
                out.append([KIND_NAMES[k], c, 1])
        return out


SWEEP_DIRS = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/dev/mqueue")
SYSV_TABLES = (("shm", "/proc/sysvipc/shm", "shmid"), ("msg", "/proc/sysvipc/msg", "msqid"),
               ("sem", "/proc/sysvipc/sem", "semid"))
IPC_RMID = 0


def _remove_owned_files(root, uid):
    removed = 0
    if not os.path.isdir(root):
        return removed
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        keep = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.islink(p):
                if st.st_uid == uid:
                    try:
                        os.unlink(p)
                        removed += 1
                    except OSError:
                        pass
                continue
            if st.st_uid == uid:
                import shutil as _sh
                _sh.rmtree(p, ignore_errors=True)
                removed += 1
            else:
                keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            p = os.path.join(dirpath, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if st.st_uid == uid:
                try:
                    os.remove(p)
                    removed += 1
                except OSError:
                    pass
    return removed


def _sysv_rows(text, id_col):
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    head = lines[0]
    try:
        cols = [head.index(id_col), head.index("cuid"), head.index("uid")]
    except ValueError:
        return []
    out = []
    for row in lines[1:]:
        if len(row) > max(cols):
            try:
                out.append(tuple(int(row[i]) for i in cols))
            except ValueError:
                pass
    return out


def _owned_sysv(uid):
    out = []
    for kind, path, id_col in SYSV_TABLES:
        try:
            with open(path, encoding="ascii") as fh:
                text = fh.read()
        except OSError:
            continue
        for ident, cuid, ouid in _sysv_rows(text, id_col):
            if int(uid) in (cuid, ouid):
                out.append((kind, int(ident)))
    return out


SYSV_RMID_TIMEOUT_S = 30.0
SYSV_RMID_PROGRAM = (
    "import ctypes, sys\n"
    "IPC_RMID = {}\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "rm = {{'shm': lambda i: libc.shmctl(i, IPC_RMID, None), 'msg': lambda i: libc.msgctl(i, IPC_RMID, None),\n"
    "      'sem': lambda i: libc.semctl(i, 0, IPC_RMID)}}\n"
    "n = 0\n"
    "for a in sys.argv[1:]:\n"
    "    kind, _, ident = a.partition(':')\n"
    "    if kind in rm and ident.isdigit() and rm[kind](int(ident)) == 0:\n"
    "        n += 1\n"
    "print(n)\n").format(IPC_RMID)


def _rmid_child(objects, uid=None, gid=None, timeout_s=SYSV_RMID_TIMEOUT_S):
    objects = list(objects)
    if not objects:
        return 0, None
    try:
        argv = [sys.executable, "-I", "-B", "-c", SYSV_RMID_PROGRAM] + ["{}:{}".format(k, int(i)) for k, i in objects]
        kw = {"user": int(uid), "group": int(gid), "extra_groups": []} if uid is not None else {}
        p = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd="/",
                           timeout=timeout_s, **kw)
    except subprocess.TimeoutExpired:
        return 0, "time-out after {:g} s".format(timeout_s)
    except Exception as e:
        return 0, "{}: {}".format(type(e).__name__, e)
    try:
        n = int((p.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        n = 0
    if p.returncode != 0:
        return n, "exit {}: {}".format(p.returncode, (p.stderr or "").strip()[-300:])
    if n < len(objects):
        return n, "removed {} of {}".format(n, len(objects))
    return n, None


def sweep_runner_residue(run_root, stderr_path, uid, gid=None):
    out = {"swept": False, "files_removed": 0, "sysv_removed": 0, "sysv_error": None, "dirs": []}
    if os.getuid() != 0 or uid is None or int(uid) in (0, os.getuid()):
        return out
    out["swept"] = True
    import shutil as _sh
    if os.path.isdir(run_root):
        _sh.rmtree(run_root, ignore_errors=True)
    if stderr_path and os.path.exists(stderr_path):
        try:
            os.remove(stderr_path)
        except OSError:
            pass
    for d in SWEEP_DIRS:
        if os.path.isdir(d):
            out["dirs"].append(d)
            out["files_removed"] += _remove_owned_files(d, int(uid))
    try:
        if gid is None:
            import pwd
            gid = pwd.getpwuid(int(uid)).pw_gid
        out["sysv_removed"], out["sysv_error"] = _rmid_child(_owned_sysv(int(uid)), int(uid), int(gid))
    except Exception as e:
        out["sysv_error"] = "{}: {}".format(type(e).__name__, e)
    return out


def run_unit(unit, site, slots, jobs, program_dir, run_root, unit_record, site_public, truth=None, recorder=None,
             python=None, runner_user="nobody", step_budget_s=None, **run_shift_kw):
    if run_shift_kw.get("zero_end"):
        raise AssertionError("run_unit: a controller through the wire is never run in run_shift's zero-end mode")
    t0, c0 = time.perf_counter(), time.process_time()
    r = WI.Runner(program_dir, run_root, user=runner_user, python=python)
    ok, info = r.start()
    rec = {"gates": {"started": bool(ok)}, "start": info}
    wc = WireController(r, unit_record, site_public, truth=truth, planned=planned_cycles(slots, jobs),
                        step_budget_s=step_budget_s)
    rec["step_budget"] = wc.step_budget_record()
    res = None
    if ok:
        try:
            res = SH.run_shift(unit, site, slots, wc, jobs=jobs, recorder=recorder, **run_shift_kw)
        except SetupFailed as e:
            rec["setup_failed"] = {"why": e.why, "info": e.info}
    stats = r.close()
    rec["runner_residue"] = sweep_runner_residue(r.run_root, r.stderr_path, getattr(r, "uid", None),
                                                 getattr(r, "gid", None))
    summ = wc.summary()
    rec["gates"].update({"setup_ok": bool(ok and wc.setup_info is not None and wc.setup_info.get("gripper_mode")),
                         "answered": summ["steps_answered"] > 0,
                         "on_time_share": summ["on_time_share"],
                         "on_time_ok": summ["on_time_share"] >= K.ON_TIME_SHARE_MIN,
                         "no_process_left": stats.get("stray_processes", 0) == 0})
    rec["runtime"] = summ
    rec["cycle_kinds"] = wc.cycle_kinds_rle()
    rec["shift"] = res
    rec["shift_end"] = shift_end(res)
    rec["runner"] = stats
    rec["wall_s"] = time.perf_counter() - t0
    rec["cpu_s"] = time.process_time() - c0
    return rec


def shift_end(res):
    if res is None:
        return None
    st = res.get("stopped")
    if st is None:
        return {"kind": "end", "cycle": res.get("cycles")}
    if "event" in st:
        return {"kind": str(st["event"].get("kind")), "cycle": st.get("cycle"), "event": st["event"]}
    for k in ("plant_invalid", "integration"):
        if k in st:
            return {"kind": k, "cycle": st.get("cycle"), "info": st[k]}
    err = st.get("error") or {}
    if err.get("name") == "ReferenceAbort" and str(err.get("message", "")).startswith("late_time"):
        return {"kind": "late_time", "cycle": st.get("cycle"), "message": err.get("message")}
    return {"kind": "stopped", "cycle": st.get("cycle"), "info": st}


DOCGEN = os.path.join(TASK_DIR, "docgen")


def _docgen():
    if DOCGEN not in sys.path:
        sys.path.append(DOCGEN)
    import record as DR
    import celldata as DC
    for m in (DR, DC):
        if os.path.dirname(os.path.abspath(m.__file__)) != DOCGEN:
            raise ImportError(f"{m.__name__} is not the documents' generator's module: {m.__file__}")
    return DR, DC


def setup_data(unit, noise_key, sheet):
    DR, DC = _docgen()
    if not isinstance(noise_key, (bytes, bytearray)):
        raise TypeError("noise_key: bytes (the registry's noise_key(label), or dev_noise_key(label))")
    rec = DR.public_record(DR.delivery_record(unit, bytes(noise_key)))
    return rec, {"cell": DC.public_cell_sheet(), "shift": sheet}


def dev_noise_key(label):
    import hashlib
    return hashlib.sha256(b"openarm05/runtime/development unit/" + str(label).encode()).digest()


def dev_unit_record(unit, unit_id="dev"):
    return setup_data(unit, dev_noise_key(unit_id), None)[0]


def public_site(site, sheet):
    _DR, DC = _docgen()
    return {"cell": DC.public_cell_sheet(), "shift": sheet}
