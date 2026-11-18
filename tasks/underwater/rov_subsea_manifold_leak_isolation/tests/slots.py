"""Package P03 (task 07, decision M12 / N7): the verifier's station-slot scheduler, a prototype that P24's verify.py
uses directly (standard library only, Linux).

A *slot* runs one hidden station: one cell process (single-threaded simulation + rendering) and one program process
(the agent's solve(), on ONE core).  DESIGN section 13: 6 stations, 3 slots, the cell and the program of a slot each
pinned to a core of their own, 2 cores left to the verify parent and the system; with fewer than 6 usable cores the
slot count is floor(cores / 2) (at least 1) -- slower, same results.  Results never depend on the slot, the core or
the order of completion: time passes only inside step(), every station has its own process pair, temporary directory
and random streams (station seed), and results are merged in station order.

    from slots import cpu_info, plan_slots, SlotScheduler, pin_preexec
    plan = plan_slots(n_jobs=6)                         # 3 slots on an 8-core verifier
    sched = SlotScheduler(plan)
    results = sched.map(run_station, stations,         # run_station(job, slot) -> dict, called in a worker thread
                        rerun_if=lambda job, res: res.get("cell_failed"), max_reruns=1)
    # inside run_station (a worker thread: no preexec_fn, the pinning is done by an exec wrapper):
    #   subprocess.Popen(pin_argv(cell_argv, [slot.cell_cpu]))
    #   subprocess.Popen(pin_argv(nonet_exec_argv + prog_argv, [slot.prog_cpu], lock=True))

Core choice (`policy`):
  design        the first 2K CPUs of sched_getaffinity in order; slot k: cell = CPU 2k, program = CPU 2k + 1 (the
                literal DESIGN 13 rule)
  smt_pair      topology-aware (VM / cpuset: the affinity mask is what the quota allows).  With at least 2K physical
                cores, every cell and every program gets a physical core of its own (first hardware thread of cores
                0 .. 2K-1; the DESIGN layout, made SMT-safe).  With fewer (e.g. 8 vCPU = 4 cores x 2 threads): slot k
                = physical core k, cell on its first thread, program on the sibling -- the program mostly waits while
                its own cell steps, so the two rarely run at the same time, and no two cells share a core (the
                literal DESIGN layout would put cells 0 and 2 on one core when siblings are numbered (k, k + 4)).
                Without SMT information this is `design`.
  least_loaded  for a quota-limited container on a larger shared host (affinity has more CPUs than the cgroup's
                cpu.max allows, e.g. ALE's docker provider: `docker run --cpus 8` on a 48-CPU machine): physical cores
                are ranked by their busy fraction over a short /proc/stat sample (ties broken by a hash of the
                hostname), then laid out like smt_pair.  `design` there would put every concurrent verifier on the
                host on CPUs 0-5.
  none          no pinning (the CFS quota alone); the program's one-core rule is then not enforced
  auto          least_loaded when the affinity mask is wider than the CPU quota, else smt_pair

Enforcement of the program's single core: affinity alone is not a boundary -- any process may widen its own mask
within its cpuset.  `pin_preexec(..., lock=True)` therefore also installs a seccomp filter (no capability needed,
PR_SET_NO_NEW_PRIVS) under which sched_setaffinity fails with EPERM for the process and all its descendants; it stacks
with the network filter of nonet_exec.py.  `cpu_overuse(pid)` is the watchdog's check (CPU time > wall x 1.05 means
more than one core was used).

Pinning is done by exec-ing through this file (`pin_argv`: `python3 -I slots.py exec --cpus C [--lock] -- argv...`
sets the affinity, optionally locks it, then execs argv), not by a Popen preexec_fn: the scheduler starts processes
from worker threads, and running Python in a forked child of a multi-threaded parent can deadlock.  The affinity is
set before the target's first instruction, so every thread it ever creates inherits it.

    python3 -I slots.py plan [POLICY] [N_JOBS]        print the plan this machine would get
    python3 -I slots.py exec --cpus 3 [--lock] -- CMD...
"""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import math
import os
import platform
import queue
import socket
import struct
import threading
import time
import traceback
from typing import Callable, Iterable, Optional

# ---------------------------------------------------------------------------------------------------- CPU discovery


@dataclasses.dataclass(frozen=True)
class CpuInfo:
    affinity: tuple            # CPUs this process may run on (cpuset-limited), ascending
    quota: Optional[float]     # cgroup CPU quota in cores (cpu.max / cfs_quota_us), None = unlimited
    n_usable: int              # min(len(affinity), floor(quota)), at least 1
    core_of: dict              # cpu -> physical-core key (from sysfs topology; cpu itself if unknown)
    smt_known: bool            # sysfs topology was readable
    wide_mask: bool            # the affinity mask has more CPUs than the quota allows (shared host)


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cgroup_quota() -> Optional[float]:
    """CPU quota of this process's cgroup in cores (v2 cpu.max, else v1 cfs_quota_us / cfs_period_us)."""
    s = _read("/sys/fs/cgroup/cpu.max")
    if s:
        q, _, per = s.partition(" ")
        if q != "max":
            try:
                return int(q) / int(per or 100000)
            except ValueError:
                pass
        return None
    q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
    per = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    try:
        if q is not None and per is not None and int(q) > 0:
            return int(q) / int(per)
    except ValueError:
        pass
    return None


def _core_key(cpu: int):
    base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
    sib = _read(f"{base}/thread_siblings_list") or _read(f"{base}/core_cpus_list")
    if sib:
        return sib
    pkg, core = _read(f"{base}/physical_package_id"), _read(f"{base}/core_id")
    return f"{pkg}:{core}" if pkg is not None and core is not None else None


def cpu_info() -> CpuInfo:
    aff = tuple(sorted(os.sched_getaffinity(0)))
    q = cgroup_quota()
    n_usable = len(aff) if q is None else max(1, min(len(aff), int(math.floor(q + 1e-9))))
    keys = {c: _core_key(c) for c in aff}
    smt_known = all(k is not None for k in keys.values())
    core_of = {c: (keys[c] if smt_known else str(c)) for c in aff}
    return CpuInfo(affinity=aff, quota=q, n_usable=n_usable, core_of=core_of, smt_known=smt_known,
                   wide_mask=q is not None and len(aff) > math.ceil(q - 1e-9))


def cpu_busy(cpus: Iterable[int], sample_s: float = 0.5) -> dict:
    """Busy fraction of each CPU over `sample_s` from /proc/stat (inside a container this is the host's view)."""
    def snap():
        out = {}
        try:
            with open("/proc/stat") as f:
                for ln in f:
                    if ln.startswith("cpu") and ln[3:4].isdigit():
                        a = ln.split()
                        v = [int(x) for x in a[1:]]
                        out[int(a[0][3:])] = (sum(v), v[3] + (v[4] if len(v) > 4 else 0))
        except OSError:
            pass
        return out
    s0 = snap()
    time.sleep(sample_s)
    s1 = snap()
    busy = {}
    for c in cpus:
        if c in s0 and c in s1 and s1[c][0] > s0[c][0]:
            busy[c] = 1.0 - (s1[c][1] - s0[c][1]) / (s1[c][0] - s0[c][0])
        else:
            busy[c] = 0.0
    return busy


# ---------------------------------------------------------------------------------------------------- slot planning


@dataclasses.dataclass(frozen=True)
class Slot:
    index: int
    cell_cpu: Optional[int]    # None = not pinned
    prog_cpu: Optional[int]


@dataclasses.dataclass(frozen=True)
class SlotPlan:
    slots: tuple
    parent_cpus: tuple         # CPUs left to the verify parent and the system (None-pinned plans: ())
    policy: str                # the policy actually applied
    requested: str
    info: CpuInfo
    note: str = ""

    def as_dict(self) -> dict:
        return dict(n_slots=len(self.slots), policy=self.policy, requested=self.requested, note=self.note,
                    slots=[dataclasses.asdict(s) for s in self.slots], parent_cpus=list(self.parent_cpus),
                    affinity=list(self.info.affinity), quota=self.info.quota, n_usable=self.info.n_usable,
                    smt_known=self.info.smt_known, wide_mask=self.info.wide_mask)


def n_slots_for(n_usable: int, k_max: int = 3) -> int:
    """DESIGN 13: 3 slots on >= 6 usable cores, else floor(cores / 2), at least 1."""
    return max(1, min(int(k_max), n_usable // 2))


def _cores_in_order(cpus, core_of, rank=None):
    """Group CPUs by physical core; cores in ascending order of `rank` (or of their first CPU)."""
    groups = {}
    for c in cpus:
        groups.setdefault(core_of[c], []).append(c)
    cores = [sorted(v) for v in groups.values()]
    if rank is None:
        cores.sort(key=lambda g: g[0])
    else:
        cores.sort(key=lambda g: (rank(g), g[0]))
    return cores


def plan_slots(n_jobs: int, k_max: int = 3, policy: str = "auto", info: Optional[CpuInfo] = None,
               busy: Optional[dict] = None) -> SlotPlan:
    """Plan min(n_jobs, n_slots_for(usable cores, k_max)) slots.  `busy` (cpu -> fraction) overrides the /proc/stat
    sample of least_loaded (tests)."""
    info = info or cpu_info()
    k = max(1, min(int(n_jobs), n_slots_for(info.n_usable, k_max)))
    req = policy
    if policy == "auto":
        policy = "least_loaded" if info.wide_mask else "smt_pair"
    if policy == "none":
        return SlotPlan(tuple(Slot(i, None, None) for i in range(k)), (), "none", req, info)
    aff = list(info.affinity)
    if policy == "design" or (policy == "smt_pair" and not info.smt_known):
        used = aff[:info.n_usable]
        if len(used) >= 2 * k:
            slots = tuple(Slot(i, used[2 * i], used[2 * i + 1]) for i in range(k))
            parent = tuple(used[2 * k:])
        else:                                   # one usable core: cell and program share it
            slots = tuple(Slot(i, used[i % len(used)], used[i % len(used)]) for i in range(k))
            parent = ()
        note = "" if policy == "design" else "no SMT topology: design layout"
        return SlotPlan(slots, parent, "design", req, info, note)
    # smt_pair / least_loaded: lay slots out on physical cores
    if policy == "least_loaded":
        b = busy if busy is not None else cpu_busy(aff)
        salt = int(hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8], 16)

        def rank(g):
            return (round(sum(b.get(c, 0.0) for c in g) / len(g), 2), (g[0] * 2654435761 ^ salt) & 0xFFFFFFFF)
        cores = _cores_in_order(aff, info.core_of, rank)
    elif policy == "smt_pair":
        cores = _cores_in_order(aff, info.core_of)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    if policy == "smt_pair":
        usable, n = [], 0
        for g in cores:
            if n >= info.n_usable:
                break
            usable.append(g[:info.n_usable - n])
            n += len(usable[-1])
        if len(usable) >= 2 * k:                # enough physical cores: one per thread, siblings left to the parent
            slots = [(usable[2 * i][0], usable[2 * i + 1][0]) for i in range(k)]
            used = {c for sl in slots for c in sl}
            spare = sorted(c for g in usable for c in g if c not in used)
            return SlotPlan(tuple(Slot(i, a, p) for i, (a, p) in enumerate(slots)), tuple(spare), "smt_pair", req,
                            info, "spread: one physical core per thread")
    # take whole physical cores until n_usable hardware threads are covered (the quota is in threads)
    chosen, n = [], 0
    for g in cores:
        if n >= info.n_usable:
            break
        g = g[:info.n_usable - n]
        chosen.append(g)
        n += len(g)
    slots, spare = [], []
    paired = [g for g in chosen if len(g) >= 2]
    single = [g[0] for g in chosen if len(g) == 1]
    for g in paired:
        if len(slots) < k:
            slots.append((g[0], g[1]))
            spare += g[2:]
        else:
            spare += g
    # not enough SMT pairs: single-thread cores two at a time (cell, program)
    while len(slots) < k and len(single) >= 2:
        slots.append((single.pop(0), single.pop(0)))
    spare += single
    while len(slots) < k:                       # degenerate: share what is there
        c = (spare or [s[0] for s in slots] or aff)[0]
        slots.append((c, c))
    return SlotPlan(tuple(Slot(i, a, p) for i, (a, p) in enumerate(slots)), tuple(sorted(spare)), policy, req, info)


# ---------------------------------------------------------------------------------------------------- pinning


def _affinity_lock_filter() -> bytes:
    """Classic BPF for seccomp: sched_setaffinity -> EPERM; any non-native syscall ABI (i386 / arm32 compat, x86 x32)
    -> EPERM too, so the call cannot be made through another ABI (the same rule as nonet_exec.py's filter, so the
    lock is complete on its own); everything else allowed."""
    nr = {"x86_64": (0xC000003E, 203, 0x40000000), "aarch64": (0xC00000B7, 122, 0)}.get(platform.machine())
    if nr is None:
        raise OSError("unsupported architecture " + platform.machine())
    audit, nr_aff, x32 = nr
    LD_W_ABS, JEQ_K, JGE_K, RET_K = 0x20, 0x15, 0x35, 0x06
    ALLOW, ERRNO_EPERM = 0x7FFF0000, 0x00050000 | 1

    def ins(code, jt, jf, k):
        return struct.pack("HBBI", code, jt, jf, k)
    prog = [ins(LD_W_ABS, 0, 0, 4), ins(JEQ_K, 1, 0, audit), ins(RET_K, 0, 0, ERRNO_EPERM),
            ins(LD_W_ABS, 0, 0, 0)]
    if x32:
        prog += [ins(JGE_K, 0, 1, x32), ins(RET_K, 0, 0, ERRNO_EPERM)]
    prog += [ins(JEQ_K, 0, 1, nr_aff), ins(RET_K, 0, 0, ERRNO_EPERM), ins(RET_K, 0, 0, ALLOW)]
    return b"".join(prog)


def lock_affinity() -> None:
    """Install the affinity-lock seccomp filter on the calling process (inherited, irremovable)."""
    prog = _affinity_lock_filter()
    buf = ctypes.create_string_buffer(prog, len(prog))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    fprog = SockFprog(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:                     # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    if libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) != 0:   # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP failed")


def pin_preexec(cpus, lock: bool = False, nice: Optional[int] = None) -> Optional[Callable[[], None]]:
    """A Popen preexec_fn that pins the child to `cpus` (None / empty = no pinning) before exec, optionally locks its
    affinity (seccomp) and renices it.  Affinity and the filter survive exec and setuid."""
    cpus = [c for c in (cpus or []) if c is not None]
    if not cpus and not lock and nice is None:
        return None

    def fn():
        if cpus:
            os.sched_setaffinity(0, cpus)
        if nice:
            os.nice(nice)
        if lock:
            lock_affinity()
    return fn


def cpu_overuse(pid: int, wall_s: float, factor: float = 1.05) -> Optional[float]:
    """CPU seconds (user + sys, all threads) of a live process divided by its wall time; > factor means it used more
    than one core on average.  None if the process is gone."""
    s = _read(f"/proc/{pid}/stat")
    if not s:
        return None
    f = s.rsplit(")", 1)[1].split()
    tck = os.sysconf("SC_CLK_TCK")
    cpu = (int(f[11]) + int(f[12])) / tck
    r = cpu / max(wall_s, 1e-9)
    return r if r > factor else 0.0


# ---------------------------------------------------------------------------------------------------- scheduler


@dataclasses.dataclass
class JobRecord:
    index: int
    attempt: int
    slot: int
    cell_cpu: Optional[int]
    prog_cpu: Optional[int]
    t_start: float
    t_end: float = float("nan")
    ok: bool = False
    error: Optional[str] = None


class SlotScheduler:
    """Run jobs on the plan's slots.  mode='queue' (default): jobs start in their fixed order, each as soon as a slot is
    free (never later than the batch layout); mode='batch': batches of len(slots) jobs, the next batch after the whole
    previous one has finished (DESIGN 13's wording).  Either way the result of a job depends only on the job."""

    def __init__(self, plan: SlotPlan, mode: str = "queue", parent_pin: bool = True):
        if mode not in ("queue", "batch"):
            raise ValueError(mode)
        self.plan, self.mode = plan, mode
        self.records: list = []
        self._lock = threading.Lock()
        if parent_pin and plan.parent_cpus:
            try:
                os.sched_setaffinity(0, plan.parent_cpus)       # the parent and every thread it starts later
            except OSError:
                pass

    def _run_one(self, fn, job, index, attempt, slot):
        rec = JobRecord(index, attempt, slot.index, slot.cell_cpu, slot.prog_cpu, time.time())
        try:
            res = fn(job, slot)
            rec.ok = True
        except Exception:                                        # a harness error is the job's result, not a crash
            res = dict(harness_error=traceback.format_exc(limit=8))
            rec.error = res["harness_error"].splitlines()[-1]
        rec.t_end = time.time()
        with self._lock:
            self.records.append(rec)
        return res

    def _pass(self, fn, items, deadline):
        """items: [(index, job)] in order.  Returns {index: result}."""
        out = {}
        free = queue.Queue()
        for s in self.plan.slots:
            free.put(s)
        threads = []
        batch = len(self.plan.slots)

        def work(i, job, slot, attempt):
            try:
                out[i] = self._run_one(fn, job, i, attempt, slot)
            finally:
                free.put(slot)
        for n, (i, job, attempt) in enumerate(items):
            if self.mode == "batch" and n and n % batch == 0:
                for t in threads:
                    t.join()
                threads = []
            slot = free.get()
            if deadline is not None and time.time() > deadline:
                out[i] = dict(harness_error="not started: verifier deadline reached")
                free.put(slot)
                continue
            t = threading.Thread(target=work, args=(i, job, slot, attempt), name=f"slot{slot.index}-job{i}",
                                 daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return out

    def map(self, fn: Callable, jobs: list, rerun_if: Optional[Callable] = None, max_reruns: int = 1,
            deadline: Optional[float] = None, rerun_min_s: float = 0.0) -> list:
        """fn(job, slot) -> result for every job; results in job order.  After all jobs have finished, at most
        `max_reruns` jobs for which rerun_if(job, result) is true are run once more (one at a time, in job order) if
        at least `rerun_min_s` remain before `deadline` (an absolute time.time()); a rerun's result replaces the
        first one and carries `rerun_of_attempt`."""
        res = self._pass(fn, [(i, j, 0) for i, j in enumerate(jobs)], deadline)
        if rerun_if is not None and max_reruns > 0:
            todo = [i for i in range(len(jobs)) if rerun_if(jobs[i], res.get(i))][:max_reruns]
            for i in todo:
                if deadline is not None and deadline - time.time() < rerun_min_s:
                    break
                r2 = self._pass(fn, [(i, jobs[i], 1)], deadline)[i]
                if isinstance(r2, dict):
                    r2 = dict(r2, rerun_of_attempt=0)
                res[i] = r2
        return [res.get(i) for i in range(len(jobs))]

    def timeline(self) -> list:
        return [dataclasses.asdict(r) for r in sorted(self.records, key=lambda r: (r.t_start, r.index))]


def pin_argv(argv, cpus, lock: bool = False, python: Optional[str] = None) -> list:
    """argv wrapped so that it runs pinned to `cpus` (None / empty: unchanged unless lock) and, with lock, unable to
    change its affinity."""
    import sys
    cpus = [int(c) for c in (cpus or []) if c is not None]
    if not cpus and not lock:
        return list(argv)
    w = [python or sys.executable, "-I", "-B", os.path.abspath(__file__), "exec"]
    if cpus:
        w += ["--cpus", ",".join(str(c) for c in cpus)]
    if lock:
        w.append("--lock")
    return w + ["--"] + list(argv)


def _main(argv) -> int:
    import json
    if argv and argv[0] == "exec":
        a = argv[1:]
        cpus, lock = None, False
        while a and a[0] != "--":
            if a[0] == "--cpus":
                cpus = [int(x) for x in a[1].split(",") if x != ""]
                a = a[2:]
            elif a[0] == "--lock":
                lock, a = True, a[1:]
            else:
                raise SystemExit(f"slots exec: unknown option {a[0]}")
        cmd = a[1:]
        if not cmd:
            raise SystemExit("slots exec: no command")
        if cpus:
            os.sched_setaffinity(0, cpus)
        if lock:
            lock_affinity()
        os.execvp(cmd[0], cmd)
    pol = argv[1] if len(argv) > 1 else "auto"
    n = int(argv[2]) if len(argv) > 2 else 6
    print(json.dumps(plan_slots(n, policy=pol).as_dict(), indent=1))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
