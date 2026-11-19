"""Clearing the agent's account before agent code runs in the verify stage.

**Why.** A grader that runs agent code (a closed_loop solver, an artifact it has to deserialize)
confines it: the kit drops it to the agent's account, and a task may add a seccomp filter (no
network), resource limits, or a pause between requests. Every one of these binds the process the
grader starts and its descendants. None binds a process that something *else* starts as the same
account, and the sandbox image keeps starting such processes. The ALE desktop base image runs
supervisord as PID 1, and supervisord runs the desktop session (Xvnc, websockify, PulseAudio,
GNOME) as the agent's account with `autorestart=true`. That gives agent code two ways out of
whatever its launcher imposed:

* **a respawn.** It kills a session program (same UID, so it may); supervisord starts the program
  again, and the new session runs what the account controls: its `~/.config/autostart` entries,
  its configuration files, a start script in a directory the account owns. Measured on
  `container-ubuntu22-base` 0.1.1 with a controller under a no-network seccomp filter: the code
  the new session ran had no filter and could open an internet socket.
* **a live session.** Its D-Bus services, its X server, its terminal server and the engine's
  desktop driver act on requests from the same account, and none of them descends from the
  launcher.

Freezing the session (SIGSTOP) does not hold: the agent's code may SIGCONT or kill what the grader
froze. Reclaiming root's PATH directories (`stage.secure_exec_path`) closes only the start script.

**What.** `clear_agent_account()` makes one statement true before the first agent process starts:
*the only processes of the agent's account are the ones the grader starts.*

1. Every supervisord program gets `autostart=false` and `autorestart=false` in supervisord's
   configuration files, and supervisord is told to reload (SIGHUP). It stops its programs, reads
   the configuration again and starts nothing. Nothing can start a program afterwards, because the
   image's supervisord has no control socket. Root's own programs are disabled with the agent's:
   in the base image the only one is the system D-Bus, which exists for the desktop, and a reload
   that started root's programs again would run them from whatever the sandbox holds by then.
2. Every process of the account that the caller did not start is killed, again and again, until
   none is left and none has come back for a settle period after the reload.

The verify stage needs no desktop: the engine reads the result files through its own guest
service, which does not run under supervisord. A grader that draws on the sandbox's X display or
needs a program the image runs under supervisord would lose it; none does (renderers here are
offscreen), and `deprivileged(..., clear_account=False)` or `Solver(..., clear_account=False)`
keeps the session for one that must, with what that leaves open.

It runs once per sandbox: the report is kept in this process and in a root-only marker file, so a
solver restart, a second `deprivileged` call or a helper process of the grader returns the first
report and kills nothing. It acts only as root inside the engine's verify stage
(`ALE_VERIFY_PYTHON` is set); anywhere else (the selftest, a practice grader, a maintainer's
shell) it does nothing unless `force=True`.

It is best-effort where it cannot know better. An init system other than supervisord is left
alone; the account's processes are still killed, and if they keep coming back the last ones are
left stopped and the report says `cleared: False`. `deprivileged` and `Solver` log that and go
on, as before this module existed. A grader whose rules lean on the confinement (a no-network
promise, a pause between requests) should call `clear_agent_account()` itself and `fail` when the
report is not `cleared`.

Stdlib only, like the rest of the kit.
"""

import configparser
import glob
import json
import os
import pwd
import re
import signal
import stat
import sys
import threading
import time

from .stage import agent_user

__all__ = ["account_processes", "clear_agent_account", "disable_programs", "enabled_programs"]

#: Where the report of the one clearing of this sandbox is kept for other processes of the grader.
_MARKER = "/run/robotics_grader/agent_account_cleared.json"

#: A quiet period this long, with no process of the account, is asked for when a supervisord
#: reload cannot be observed (no pid file), or when the account's processes had a parent this
#: module does not know: longer than supervisord's longest pause before it retries a start.
_UNCONFIRMED_SETTLE_S = 5.0

_SECTION = re.compile(r"^\s*\[([^\]]+)\]")
_SWITCH = re.compile(r"^\s*(autostart|autorestart)\s*[=:]", re.IGNORECASE)
_PROGRAM_SECTIONS = ("program:", "fcgi-program:", "eventlistener:")
_OFF = ("false", "off", "no", "0")

_reports = {}
_lock = threading.Lock()


def _log(message):
    print("[grader] " + message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- processes
def _status(pid):
    """`(uids, parent pid, state letter)` of a process, or None when it is gone."""
    try:
        with open("/proc/{}/status".format(pid)) as handle:
            text = handle.read()
    except (OSError, IOError):
        return None
    uids, parent, state = (), 0, ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        try:
            if key == "Uid":
                uids = tuple(int(part) for part in value.split())
            elif key == "PPid":
                parent = int(value.strip() or 0)
            elif key == "State":
                state = value.strip()[:1]
        except ValueError:
            return None
    return uids, parent, state


def _process_table():
    table = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            row = _status(name)
            if row is not None:
                table[int(name)] = row
    return table


def _descends(pid, ancestor, table):
    for _ in range(256):
        if pid == ancestor:
            return True
        row = table.get(pid)
        if row is None or row[1] <= 0:
            return False
        pid = row[1]
    return False


def account_processes(uid, keep=None, table=None):
    """PIDs of the live processes that run as `uid` (its real, effective, saved or filesystem UID)
    and do not descend from the process `keep` (this process by default): what the caller did not
    start. Zombies are not counted; they run nothing."""
    keep = os.getpid() if keep is None else keep
    table = _process_table() if table is None else table
    found = []
    for pid, (uids, _parent, state) in table.items():
        if uid in uids and state not in ("Z", "X") and not _descends(pid, keep, table):
            found.append(pid)
    return sorted(found)


def _describe(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as handle:
            text = handle.read(200).replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (OSError, IOError):
        text = ""
    return "{} {}".format(pid, text or "(gone)")


# --------------------------------------------------------------------------- supervisord
def _argv(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as handle:
            return [part.decode("utf-8", "replace") for part in handle.read().split(b"\0") if part]
    except (OSError, IOError):
        return []


def _supervisors(table=None):
    """`[(pid, configuration file or None)]` of every supervisord that runs as root."""
    table = _process_table() if table is None else table
    found = []
    for pid, (uids, _parent, state) in sorted(table.items()):
        if len(uids) < 2 or uids[1] != 0 or state in ("Z", "X"):
            continue
        argv = _argv(pid)
        for index, arg in enumerate(argv[:2]):       # `supervisord ...` or `python3 supervisord ...`
            if os.path.basename(arg) == "supervisord":
                found.append((pid, _supervisor_config(pid, argv[index + 1:])))
                break
    return found


def _supervisor_config(pid, args):
    named = None
    for index, arg in enumerate(args):
        if arg in ("-c", "--configuration") and index + 1 < len(args):
            named = args[index + 1]
        elif arg.startswith("--configuration="):
            named = arg.split("=", 1)[1]
        elif arg.startswith("-c") and not arg.startswith("--") and len(arg) > 2:
            named = arg[2:]
    try:
        cwd = os.readlink("/proc/{}/cwd".format(pid))
    except OSError:
        cwd = "/"
    # Without -c supervisord looks in its working directory, then in /etc.
    candidates = [named] if named else ["supervisord.conf", "etc/supervisord.conf",
                                        "/etc/supervisord.conf", "/etc/supervisor/supervisord.conf"]
    for candidate in candidates:
        path = os.path.normpath(os.path.join(cwd, candidate))
        if os.path.isfile(path):
            return path
    return None


def _read_config(path):
    # The parser settings supervisord itself uses: `;` starts an inline comment, a section may repeat.
    parser = configparser.RawConfigParser(inline_comment_prefixes=(";",), strict=False)
    with open(path) as handle:
        parser.read_file(handle)
    return parser


def _config_files(main, cwd="/"):
    """`(files, pid file)`: the configuration file, the files its `[include]` names, and where
    this supervisord writes its pid file (it writes a new one after every reload)."""
    files, pidfile = [main], None
    here = os.path.dirname(os.path.abspath(main))
    try:
        parser = _read_config(main)
    except (OSError, IOError, configparser.Error):
        return files, pidfile
    if parser.has_option("include", "files"):
        for pattern in parser.get("include", "files").split():
            pattern = os.path.join(here, pattern.replace("%(here)s", here))
            for name in sorted(glob.glob(pattern)):
                if os.path.isfile(name) and name not in files:
                    files.append(name)
    named = parser.get("supervisord", "pidfile") if parser.has_option("supervisord", "pidfile") else "supervisord.pid"
    pidfile = os.path.normpath(os.path.join(cwd, named.replace("%(here)s", here)))
    return files, pidfile


def _off(parser, section, option):
    return parser.has_option(section, option) and parser.get(section, option).strip().lower() in _OFF


def enabled_programs(path):
    """The sections of one supervisord configuration file that describe a program supervisord
    would still start or restart."""
    parser = _read_config(path)
    return [section for section in parser.sections()
            if section.startswith(_PROGRAM_SECTIONS)
            and not (_off(parser, section, "autostart") and _off(parser, section, "autorestart"))]


def disable_programs(path):
    """Rewrite one supervisord configuration file so that no program is started or restarted:
    every program section gets `autostart=false` and `autorestart=false`, everything else stays
    byte for byte. Returns the sections changed (empty: the file is left untouched). The new text
    is read back with supervisord's own parser settings before it replaces the file; a file that
    does not read back as intended is left as it was and raises ValueError."""
    targets = enabled_programs(path)
    if not targets:
        return []
    with open(path) as handle:
        lines = handle.readlines()
    out, current = [], None
    for line in lines:
        header = _SECTION.match(line)
        if header:
            current = header.group(1).strip()
            out.append(line if line.endswith("\n") else line + "\n")
            if current in targets:
                out += ["autostart=false\n", "autorestart=false\n"]
            continue
        if current in targets and _SWITCH.match(line):
            continue
        out.append(line)
    draft = path + ".robotics_grader"
    with open(draft, "w") as handle:
        handle.writelines(out)
    try:
        if enabled_programs(draft):
            raise ValueError("{} does not read back with its programs disabled".format(path))
        os.chmod(draft, stat.S_IMODE(os.stat(path).st_mode) & ~0o022)
        if os.geteuid() == 0:
            os.chown(draft, 0, 0)
        os.replace(draft, path)
    finally:
        if os.path.lexists(draft):
            os.remove(draft)
    return targets


def _stamp(path):
    """What changes when supervisord writes its pid file again; None while the file is absent."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_ino, info.st_mtime_ns


# --------------------------------------------------------------------------- the clearing
def _clear(uid, timeout=30.0, settle=1.0, strays=None, supervisors=None):
    """The work of `clear_agent_account`, with its inputs open to a test: `strays()` lists the
    processes to kill, `supervisors()` the `(pid, configuration file)` pairs to reload."""
    started = time.monotonic()
    me = os.getpid()
    strays = strays or (lambda: account_processes(uid, keep=me))
    report = {"cleared": False, "skipped": None, "supervisors": 0, "programs_disabled": [],
              "reloads_confirmed": 0, "killed": 0, "left": [], "seconds": 0.0}
    watched, unconfirmed = [], False
    for pid, config in (supervisors() if supervisors else _supervisors()):
        report["supervisors"] += 1
        if not config:
            _log("supervisord {} has no configuration file this kit can find".format(pid))
            unconfirmed = True
            continue
        try:
            cwd = os.readlink("/proc/{}/cwd".format(pid))
        except OSError:
            cwd = "/"
        files, pidfile = _config_files(config, cwd)
        changed = []
        for path in files:
            try:
                changed += disable_programs(path)
            except (OSError, IOError, ValueError, configparser.Error) as exc:
                _log("could not disable the programs in {}: {}".format(path, exc))
                unconfirmed = True
        if not changed:
            continue
        report["programs_disabled"] += changed
        before = _stamp(pidfile) if pidfile else None
        try:
            os.kill(pid, signal.SIGHUP)
        except OSError as exc:
            _log("could not tell supervisord {} to reload: {}".format(pid, exc))
            unconfirmed = True
            continue
        if before is None:
            unconfirmed = True          # no pid file to watch: only the quiet period can tell
        else:
            watched.append((pidfile, before))

    killed, quiet_from = set(), None
    deadline = started + float(timeout)
    while True:
        last = strays()
        now = time.monotonic()
        if last:
            # Stop first, so that nothing forks away between the two passes, then kill.
            for sig in (signal.SIGSTOP, signal.SIGKILL):
                for pid in last:
                    try:
                        os.kill(pid, sig)
                    except OSError:
                        pass
            killed.update(last)
            quiet_from = None
            if not watched:
                unconfirmed = True      # no reload to wait for: whatever starts them is unknown here
        reloading = [path for path, before in watched if _stamp(path) in (None, before)]
        if reloading:
            quiet_from = None           # the quiet period counts from the end of the last reload
        elif not last and quiet_from is None:
            quiet_from = now
        need = max(float(settle), _UNCONFIRMED_SETTLE_S) if unconfirmed else float(settle)
        if quiet_from is not None and now - quiet_from >= need:
            report["cleared"] = True
            break
        if now >= deadline:
            break
        time.sleep(0.05)
    report["reloads_confirmed"] = len([1 for path, before in watched if _stamp(path) not in (None, before)])
    report["killed"] = len(killed)
    if not report["cleared"]:
        # What is left comes back as fast as it is killed. A stopped process does not exit, so
        # whatever restarts it has nothing to restart: stop the generation that is there (the
        # last kill may still be bringing one up, so wait for one and see it stay stopped).
        steady = 0
        for _ in range(100):
            present = strays()
            running = [pid for pid in present if (_status(pid) or ((), 0, "T"))[2] not in ("T", "t")]
            for pid in running:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except OSError:
                    pass
            steady = steady + 1 if present and not running else 0
            if steady >= 3:
                break
            time.sleep(0.05)
        report["left"] = [_describe(pid) for pid in strays()][:20]
    report["seconds"] = round(time.monotonic() - started, 2)
    return report


def _read_marker(uid):
    try:
        info = os.lstat(_MARKER)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            return None
        with open(_MARKER) as handle:
            data = json.load(handle)
    except (OSError, IOError, ValueError):
        return None
    row = data.get(str(uid)) if isinstance(data, dict) else None
    return row if isinstance(row, dict) and "cleared" in row else None


def _write_marker(uid, report):
    folder = os.path.dirname(_MARKER)
    try:
        os.makedirs(folder, exist_ok=True)
        os.chown(folder, 0, 0)
        os.chmod(folder, 0o700)
        draft = _MARKER + ".{}".format(os.getpid())
        with open(draft, "w") as handle:
            json.dump({str(uid): report}, handle)
        os.chmod(draft, 0o600)
        os.replace(draft, _MARKER)
    except (OSError, IOError):
        pass                            # the report still lives in this process


def clear_agent_account(user=None, force=False, timeout=30.0, settle=1.0):
    """Before agent code runs in the verify stage: disable every supervisord program, reload
    supervisord, and kill every process of the agent's account that the caller did not start,
    until none is left and none comes back (see the module text).

    Returns a report: `cleared` (the statement holds), `skipped` (why nothing was done: not root,
    not the verify stage, no such account), `supervisors`, `programs_disabled` (their section
    names), `reloads_confirmed`, `killed`, `left` (what was still coming back, left stopped) and
    `seconds`. The first call of a sandbox does the work; later calls, from this process or
    another one of the grader, return its report.

    Call it before starting anything as the agent's account: a process of the account that the
    calling process did not start is killed, whoever started it.
    """
    report = {"cleared": False, "skipped": None, "supervisors": 0, "programs_disabled": [],
              "reloads_confirmed": 0, "killed": 0, "left": [], "seconds": 0.0}
    name = user or agent_user()
    if os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return dict(report, skipped="not root")
    if not (force or os.environ.get("ALE_VERIFY_PYTHON")):
        return dict(report, skipped="not the engine's verify stage")
    try:
        uid = pwd.getpwnam(name).pw_uid
    except KeyError:
        return dict(report, skipped="no account {!r}".format(name))
    if uid == 0:
        return dict(report, skipped="the agent's account is root")
    with _lock:
        if uid in _reports:
            return _reports[uid]
        report = _read_marker(uid)
        if report is None:
            report = _clear(uid, timeout=timeout, settle=settle)
            _write_marker(uid, report)
            if report["cleared"]:
                _log("agent account {} cleared in {:.1f} s: {} process(es) killed, supervisord "
                     "programs disabled: {}".format(name, report["seconds"], report["killed"],
                                                    ", ".join(report["programs_disabled"]) or "none"))
            else:
                _log("WARNING: agent account {} is NOT cleared after {:.1f} s: {} process(es) keep "
                     "coming back and were left stopped: {}".format(
                         name, report["seconds"], len(report["left"]), "; ".join(report["left"]) or "?"))
        _reports[uid] = report
        return report
