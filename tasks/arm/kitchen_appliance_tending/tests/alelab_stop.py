#!/usr/bin/env python3
"""Stop the alelab lab service (and optionally every agent-account process) before grading.

Run as root at the very start of verify (shared verification), BEFORE reading any agent file:

    python3 -I -B alelab_stop.py --agent-user user          # prints one JSON line, exit 0 = confirmed

or from verify.py:

    from alelab_stop import stop_all
    report = stop_all(agent_user="user")                    # raises RuntimeError if not confirmed

It never connects to the lab socket. Steps: SIGTERM the supervisor by its recorded pid (checked to be
a root alelab process), SIGKILL every process of the lab account, SIGKILL every process of the agent
account (SIGSTOP first so nothing can fork away), remove the socket, then re-scan /proc and confirm
that no process of either account is left. After a settle period it reports agent-account processes
that came back (the base image's supervisord respawns its desktop services as the agent account; they
are NOT a lab survivor and do not clear `confirmed`). Python 3.12 standard library only.
"""

import argparse
import json
import os
import pwd
import signal
import stat
import sys
import time

DEFAULT_SOCKET = "/run/ale-lab/lab.sock"


def _status(pid):
    try:
        with open("/proc/%s/status" % pid) as fh:
            txt = fh.read()
    except OSError:
        return None
    out = {}
    for line in txt.splitlines():
        key, _, val = line.partition(":")
        if key in ("Uid", "Gid"):
            out[key] = [int(x) for x in val.split()]
        elif key in ("Name", "State"):
            out[key] = val.strip()
    return out


def _cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            return fh.read(300).replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _pids_of(uids):
    found = []
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        st = _status(d)
        if not st or "Uid" not in st:
            continue
        if st.get("State", "").startswith("Z"):
            continue
        if any(u in uids for u in st["Uid"]):
            found.append(int(d))
    return sorted(found)


def _kill_all(uids, timeout):
    """SIGSTOP + SIGKILL every process with any uid in `uids` until none is left."""
    killed = set()
    deadline = time.monotonic() + timeout
    while True:
        pids = _pids_of(uids)
        if not pids:
            return sorted(killed), True
        if time.monotonic() > deadline:
            return sorted(killed), False
        for sig in (signal.SIGSTOP, signal.SIGKILL):
            for pid in pids:
                try:
                    os.kill(pid, sig)
                    killed.add(pid)
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(0.05)


def stop_all(
    agent_user=None,
    lab_user="alelab",
    socket_path=DEFAULT_SOCKET,
    timeout=15.0,
    strict=True,
    settle_s=3.0,
):
    if os.geteuid() != 0:
        raise RuntimeError("alelab_stop must run as root")
    report = {"supervisor_pid": None, "supervisor_signalled": False}
    pidfile = os.path.join(os.path.dirname(socket_path), "supervisor.pid")
    try:
        st = os.lstat(pidfile)
        if st.st_uid == 0 and not (st.st_mode & 0o022) and stat.S_ISREG(st.st_mode):
            with open(pidfile) as fh:
                pid = int(fh.read(32).strip())
            report["supervisor_pid"] = pid
            info = _status(pid)
            with open("/proc/%d/cmdline" % pid, "rb") as fh:
                cmd = fh.read(4096)
            if info and info.get("Uid", [None])[0] == 0 and b"server.py" in cmd:
                os.kill(pid, signal.SIGTERM)
                report["supervisor_signalled"] = True
                for _ in range(60):
                    if _status(pid) is None or _status(pid).get("State", "").startswith(
                        "Z"
                    ):
                        break
                    time.sleep(0.05)
                else:
                    os.kill(pid, signal.SIGKILL)
    except (OSError, ValueError):
        pass
    try:
        lab_uid = pwd.getpwnam(lab_user).pw_uid
    except KeyError:
        lab_uid = None
    if lab_uid is not None:
        report["lab_killed"], report["lab_confirmed"] = _kill_all({lab_uid}, timeout)
    else:
        report["lab_killed"], report["lab_confirmed"] = [], True
    # the supervisor itself (root) if it ignored SIGTERM
    sup = report["supervisor_pid"]
    if sup and _status(sup) and not _status(sup).get("State", "").startswith("Z"):
        try:
            os.kill(sup, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(0.1)
    report["supervisor_gone"] = (
        not sup or _status(sup) is None or _status(sup).get("State", "").startswith("Z")
    )
    if agent_user:
        agent_uid = pwd.getpwnam(agent_user).pw_uid
        report["agent_killed"], report["agent_confirmed"] = _kill_all(
            {agent_uid}, timeout
        )
        # The base image's supervisord (PID 1) respawns its agent-account desktop services (Xvnc, GNOME
        # session, pulseaudio, cua-driver, ...) within ~1-2 s, and the respawned GNOME session runs the
        # agent's ~/.config/autostart entries again. Report what is back after a settle period; shared
        # verification must treat the agent account as live (see docs/p0/lab.md section 6).
        time.sleep(max(0.0, settle_s))
        back = _pids_of({agent_uid})
        report["agent_respawned"] = len(back)
        report["agent_respawned_cmdlines"] = [_cmdline(p) for p in back[:25]]
    else:
        report["agent_killed"], report["agent_confirmed"] = [], True
    for path in (socket_path, pidfile):
        try:
            os.unlink(path)
        except OSError:
            pass
    report["socket_removed"] = not os.path.lexists(socket_path)
    report["confirmed"] = bool(
        report["lab_confirmed"]
        and report["agent_confirmed"]
        and report["supervisor_gone"]
    )
    report["lab_killed"] = len(report["lab_killed"])
    report["agent_killed"] = len(report["agent_killed"])
    if strict and not report["confirmed"]:
        raise RuntimeError("alelab_stop: processes survived: %s" % json.dumps(report))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent-user", default=None)
    ap.add_argument("--lab-user", default="alelab")
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="seconds to wait before re-scanning for respawns",
    )
    a = ap.parse_args(argv)
    rep = stop_all(
        a.agent_user, a.lab_user, a.socket, a.timeout, strict=False, settle_s=a.settle
    )
    print(json.dumps(rep, sort_keys=True))
    return 0 if rep["confirmed"] else 3


if __name__ == "__main__":
    sys.exit(main())
