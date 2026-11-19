"""The kit clears the agent's account before it starts agent code (`shared/robotics_grader/account.py`).

What a run in a sandbox does (root, supervisord as PID 1, the desktop session as the agent's account) is recorded in
`PROGRESS.md`; a test cannot be root and must not kill anything of the account it runs as. So these tests pin the
parts in pieces:

- the rewrite of supervisord's configuration files, on the base image's own files and on awkward ones;
- the loop that reloads a supervisor and kills until nothing comes back, against a real stand-in supervisor (a child
  process of the test that restarts its programs when they die and reads its configuration again on SIGHUP), with the
  set of processes it may kill narrowed to that stand-in's own children;
- what it refuses to do (outside root, outside the engine's verify stage);
- that `Solver` and `deprivileged` call it once, before the first agent process, and not on a restart.
"""

from __future__ import annotations

import configparser
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from robotics_grader import account, solver as solver_mod, stage  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")

# container-ubuntu22-base 0.1.1, /etc/supervisor (images/base/container-ubuntu22/Dockerfile of the engine).
MAIN = """\
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[include]
files = /etc/supervisor/conf.d/*.conf
"""
XVNC = """\
[program:xvnc]
command=/usr/bin/Xvnc :1 -depth 24 -geometry %(ENV_VNC_RESOLUTION)s -SecurityTypes None -localhost no -alwaysshared
autostart=true
autorestart=true
user=user
environment=HOME="/home/user",USER="user"
priority=10
stdout_logfile=/var/log/supervisor/xvnc.log
stderr_logfile=/var/log/supervisor/xvnc.err
"""
SYSTEM_DBUS = """\
[program:system-dbus]
command=/usr/local/bin/system-dbus.sh
autostart=true
autorestart=true
user=root
priority=15
"""
GNOME_READY = """\
[program:gnome-ready]
command=/usr/local/bin/wait-for-gnome-ready.sh
autostart=true
autorestart=false
startsecs=0
user=user
environment=HOME="/home/user",USER="user",DISPLAY=":1"
priority=31
"""
AWKWARD = """\
; a comment line with [program:not_a_section] in it
[program:spaced]
command = /bin/true ; an inline comment
AutoStart = true ; mixed case, spaces
autorestart: unexpected
user = 1000

[group:desk]
programs=spaced,quiet

[program:quiet]
command=/bin/true
autostart=false
autorestart=false

[eventlistener:watch]
command=/bin/true
events=TICK_5
"""


def _parser(path):
    parser = configparser.RawConfigParser(inline_comment_prefixes=(";",), strict=False)   # as supervisord reads it
    parser.read(path)
    return parser


def _tree(tmp_path, files):
    conf = tmp_path / "conf.d"
    conf.mkdir()
    main = tmp_path / "supervisord.conf"
    main.write_text(MAIN.replace("/etc/supervisor/conf.d", str(conf)).replace("/var/run/supervisord.pid",
                                                                              str(tmp_path / "supervisord.pid")))
    for name, text in files.items():
        (conf / name).write_text(text)
    return main, conf


# --------------------------------------------------------------------------- the configuration rewrite
def test_every_program_of_the_base_image_is_switched_off_and_nothing_else_changes(tmp_path):
    main, conf = _tree(tmp_path, {"10-xvnc.conf": XVNC, "15-system-dbus.conf": SYSTEM_DBUS,
                                  "31-gnome-ready.conf": GNOME_READY})
    files, pidfile = account._config_files(str(main), cwd="/")
    assert [Path(f).name for f in files] == ["supervisord.conf", "10-xvnc.conf", "15-system-dbus.conf",
                                             "31-gnome-ready.conf"]
    assert pidfile == str(tmp_path / "supervisord.pid")
    changed = [section for path in files for section in account.disable_programs(path)]
    assert changed == ["program:xvnc", "program:system-dbus", "program:gnome-ready"]   # root's own program too
    assert main.read_text().startswith("[supervisord]\nnodaemon=true")                # no program in it: untouched
    for name, section in (("10-xvnc.conf", "program:xvnc"), ("15-system-dbus.conf", "program:system-dbus"),
                          ("31-gnome-ready.conf", "program:gnome-ready")):
        parser = _parser(conf / name)
        assert parser.get(section, "autostart") == "false" and parser.get(section, "autorestart") == "false"
    xvnc = _parser(conf / "10-xvnc.conf")
    assert xvnc.get("program:xvnc", "command").endswith("%(ENV_VNC_RESOLUTION)s -SecurityTypes None -localhost no -alwaysshared")
    assert xvnc.get("program:xvnc", "environment") == 'HOME="/home/user",USER="user"'
    before = (conf / "10-xvnc.conf").read_text()
    kept = [line for line in XVNC.splitlines() if not line.startswith(("autostart", "autorestart"))]
    assert [line for line in before.splitlines() if not line.startswith(("autostart", "autorestart"))] == kept
    # a second pass finds nothing to do and leaves the bytes alone
    assert [s for path in files for s in account.disable_programs(path)] == []
    assert (conf / "10-xvnc.conf").read_text() == before
    assert not list(conf.glob("*.robotics_grader"))


def test_awkward_files_are_rewritten_as_supervisord_reads_them(tmp_path):
    path = tmp_path / "awkward.conf"
    path.write_text(AWKWARD)
    os.chmod(path, 0o664)
    assert account.enabled_programs(str(path)) == ["program:spaced", "eventlistener:watch"]
    assert account.disable_programs(str(path)) == ["program:spaced", "eventlistener:watch"]
    parser = _parser(path)
    for section in ("program:spaced", "program:quiet", "eventlistener:watch"):
        assert parser.get(section, "autostart") == "false" and parser.get(section, "autorestart") == "false"
    assert parser.get("program:spaced", "command") == "/bin/true" and parser.get("program:spaced", "user") == "1000"
    assert parser.get("group:desk", "programs") == "spaced,quiet"
    text = path.read_text()
    assert "; a comment line with [program:not_a_section] in it" in text
    assert text.count("autostart") == 3 and "AutoStart" not in text and "unexpected" not in text
    assert (os.stat(path).st_mode & 0o777) == 0o644          # no group or other write left on a file root reads
    assert account.enabled_programs(str(path)) == []


def test_a_file_that_would_not_read_back_disabled_is_left_alone(tmp_path, monkeypatch):
    path = tmp_path / "x.conf"
    path.write_text(XVNC)
    real = account.enabled_programs
    monkeypatch.setattr(account, "enabled_programs", lambda p: ["program:xvnc"] if p.endswith(".robotics_grader") else real(p))
    with pytest.raises(ValueError):
        account.disable_programs(str(path))
    assert path.read_text() == XVNC and not list(tmp_path.glob("*.robotics_grader"))


def test_the_configuration_file_is_found_from_the_command_line(tmp_path):
    main, _conf = _tree(tmp_path, {})
    me = os.getpid()
    assert account._supervisor_config(me, ["-c", str(main)]) == str(main)
    assert account._supervisor_config(me, ["-n", "--configuration=" + str(main)]) == str(main)
    assert account._supervisor_config(me, ["-c" + str(main)]) == str(main)
    assert account._supervisor_config(me, ["-c", str(tmp_path / "missing.conf")]) is None


# --------------------------------------------------------------------------- the loop, against a stand-in supervisor
STAND_IN = textwrap.dedent('''
    """A stand-in for supervisord: starts the programs of its configuration file, starts one again when it dies
    (autorestart), and on SIGHUP stops them all, reads the file again and writes a new pid file. With --deaf it
    ignores SIGHUP, like an init system the kit does not know."""
    import configparser, os, signal, subprocess, sys, time
    conf, pidfile, deaf = sys.argv[1], sys.argv[2], "--deaf" in sys.argv
    reload = []
    signal.signal(signal.SIGHUP, (lambda *a: None) if deaf else (lambda *a: reload.append(1)))

    def programs():
        parser = configparser.RawConfigParser(inline_comment_prefixes=(";",), strict=False)
        parser.read(conf)
        return {s: parser for s in parser.sections() if s.startswith("program:")}

    def on(parser, section, key, default):
        return (parser.get(section, key) if parser.has_option(section, key) else default).strip().lower() == "true"

    def write_pid():
        if os.path.exists(pidfile):
            os.remove(pidfile)
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))

    table, children = programs(), {}
    write_pid()
    for name, parser in table.items():
        if on(parser, name, "autostart", "true"):
            children[name] = subprocess.Popen(["sleep", "1000"])
    while True:
        time.sleep(0.02)
        if reload:
            reload.clear()
            for child in children.values():
                child.kill(); child.wait()
            time.sleep(0.3)                           # a reload takes a moment
            table, children = programs(), {}
            write_pid()
            for name, parser in table.items():
                if on(parser, name, "autostart", "true"):
                    children[name] = subprocess.Popen(["sleep", "1000"])
        for name, child in list(children.items()):
            if child.poll() is not None:
                if on(table[name], name, "autorestart", "true"):
                    children[name] = subprocess.Popen(["sleep", "1000"])
                else:
                    del children[name]
''')
PROGRAMS = "[program:desk]\ncommand=sleep 1000\nautostart=true\nautorestart=true\nuser=user\n\n" \
           "[program:sound]\ncommand=sleep 1000\nautorestart=true\nuser=user\n"


def _children(pid):
    found = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            row = account._status(name)
            if row is not None and row[1] == pid and row[2] not in ("Z", "X"):
                found.append(int(name))
    return sorted(found)


def _wait(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def stand_in(tmp_path):
    started = []

    def start(*flags):
        script, conf, pidfile = tmp_path / "stand_in.py", tmp_path / "programs.conf", tmp_path / "stand_in.pid"
        script.write_text(STAND_IN)
        conf.write_text(PROGRAMS)
        proc = subprocess.Popen([sys.executable, str(script), str(conf), str(pidfile), *flags])
        started.append(proc)
        assert _wait(lambda: len(_children(proc.pid)) == 2 and pidfile.exists()), "the stand-in did not start its programs"
        return proc, conf, pidfile

    yield start
    for proc in started:
        for pid in _children(proc.pid):
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
        proc.kill()
        proc.wait()


def test_a_supervisor_that_restarts_its_programs_is_reloaded_and_stays_quiet(stand_in):
    proc, conf, pidfile = stand_in()
    first = _children(proc.pid)
    os.kill(first[0], signal.SIGKILL)                      # what agent code can do: it comes straight back
    assert _wait(lambda: len(_children(proc.pid)) == 2 and _children(proc.pid) != first)

    report = account._clear(os.getuid(), timeout=20.0, settle=0.5,
                            strays=lambda: _children(proc.pid),
                            supervisors=lambda: [(proc.pid, str(conf))])
    assert report["cleared"] is True and report["left"] == []
    assert report["supervisors"] == 1 and report["programs_disabled"] == ["program:desk", "program:sound"]
    assert report["reloads_confirmed"] == 0     # this stand-in's configuration names no pid file: the quiet period decides
    time.sleep(1.0)
    assert _children(proc.pid) == [] and proc.poll() is None      # nothing came back; the supervisor itself lives
    assert account.enabled_programs(str(conf)) == []
    assert pidfile.exists()


def test_the_reload_is_confirmed_by_the_pid_file_when_the_configuration_names_one(stand_in, tmp_path):
    proc, conf, pidfile = stand_in()
    main = tmp_path / "main.conf"
    main.write_text("[supervisord]\npidfile={}\n\n[include]\nfiles = {}\n".format(pidfile, conf.name))
    started = time.monotonic()
    report = account._clear(os.getuid(), timeout=20.0, settle=0.5,
                            strays=lambda: _children(proc.pid),
                            supervisors=lambda: [(proc.pid, str(main))])
    assert report["cleared"] is True and report["reloads_confirmed"] == 1
    assert report["programs_disabled"] == ["program:desk", "program:sound"]
    assert time.monotonic() - started < account._UNCONFIRMED_SETTLE_S     # a confirmed reload needs only the short settle
    assert _children(proc.pid) == []


def test_an_unknown_respawner_is_reported_and_its_last_generation_left_stopped(stand_in):
    proc, conf, _pidfile = stand_in("--deaf")
    report = account._clear(os.getuid(), timeout=2.0, settle=0.5,
                            strays=lambda: _children(proc.pid),
                            supervisors=lambda: [])           # nothing the kit recognises as a supervisor
    assert report["cleared"] is False and report["supervisors"] == 0 and report["programs_disabled"] == []
    assert report["left"] and report["killed"] >= 2
    left = _children(proc.pid)
    assert left and all(account._status(pid)[2] in ("T", "t") for pid in left)   # stopped: nothing restarts them
    assert account.enabled_programs(str(conf)) == ["program:desk", "program:sound"]  # an unknown init is left alone


def test_only_what_the_caller_did_not_start_counts_as_the_accounts_processes():
    child = subprocess.Popen(["sleep", "30"])
    try:
        uid = os.getuid()
        mine = account.account_processes(uid)
        assert child.pid not in mine and os.getpid() not in mine           # this process and what it started
        assert os.getppid() in mine                                         # its parent is somebody else's
        seen_from_child = account.account_processes(uid, keep=child.pid)
        assert os.getpid() in seen_from_child and child.pid not in seen_from_child
    finally:
        child.kill()
        child.wait()


# --------------------------------------------------------------------------- what it refuses, and who calls it
def test_it_does_nothing_outside_root_and_outside_the_verify_stage(monkeypatch):
    monkeypatch.delenv("ALE_VERIFY_PYTHON", raising=False)
    monkeypatch.setattr(account, "_clear", lambda *a, **k: pytest.fail("the clearing ran"))
    if os.geteuid() != 0:
        assert account.clear_agent_account()["skipped"] == "not root"
        assert account.clear_agent_account(force=True)["skipped"] == "not root"
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    report = account.clear_agent_account()
    assert report["skipped"] == "not the engine's verify stage" and report["cleared"] is False
    monkeypatch.setenv("ALE_VERIFY_PYTHON", "/opt/venv/bin/python3")
    assert account.clear_agent_account(user="no-such-account-xyz")["skipped"] == "no account 'no-such-account-xyz'"
    assert account.clear_agent_account(user="root")["skipped"] == "the agent's account is root"


class _Local(solver_mod.Solver):
    def _spawn(self):
        self.spawned = getattr(self, "spawned", 0) + 1
        return subprocess.Popen([sys.executable, "-u", "-c",
                                 "import sys\nprint('READY', flush=True)\n"
                                 "for line in sys.stdin: print('{\"ok\": true}', flush=True)"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1,
                                start_new_session=True)


def test_solver_clears_the_account_once_before_its_first_start_and_not_on_a_restart(tmp_path, monkeypatch):
    (tmp_path / "run.sh").write_text("")
    calls = []
    monkeypatch.setattr(solver_mod, "clear_agent_account", lambda: calls.append("clear"))
    monkeypatch.setattr(_Local, "_start", lambda self: (calls.append("start"), solver_mod.Solver._start(self))[1])
    solver = _Local(step_timeout=5.0, submission=str(tmp_path))
    assert calls == ["clear", "start"]
    solver._broken = True
    solver.reset()
    assert calls == ["clear", "start", "start"] and solver.restarts == 1
    solver.close()

    calls.clear()
    kept = _Local(step_timeout=5.0, submission=str(tmp_path), clear_account=False)
    assert calls == ["start"] and kept.clear_account is False
    kept.close()


def test_a_missing_submission_is_reported_before_anything_is_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_mod, "clear_agent_account", lambda: pytest.fail("cleared for an untouched sandbox"))
    with pytest.raises(solver_mod.SolverMissing):
        solver_mod.Solver(submission=str(tmp_path))


def test_deprivileged_clears_the_account_unless_told_not_to(monkeypatch):
    calls, seen = [], {}
    monkeypatch.setattr(account, "clear_agent_account", lambda: calls.append("clear"))
    monkeypatch.setattr(stage.subprocess, "Popen", lambda argv, **kwargs: seen.update(argv=argv, kwargs=kwargs))
    stage.deprivileged(["bash", "run.sh"], cwd="/tmp")
    assert calls == ["clear"] and seen["argv"][-2:] == ["bash", "run.sh"] and seen["kwargs"] == {"cwd": "/tmp"}
    stage.deprivileged(["bash", "run.sh"], clear_account=False)
    assert calls == ["clear"] and seen["kwargs"] == {}      # the switch is the kit's, not an argument of Popen
