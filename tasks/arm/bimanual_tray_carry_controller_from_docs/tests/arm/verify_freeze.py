"""Keep the agent account frozen for the whole verify stage (PLAN.md 21.16; tier-wide since 2026-10-02).

The task images run supervisord as PID 1, and it starts the desktop services (Xvnc, websockify, GNOME, PulseAudio, ...) as
the agent account with autorestart. A verify-start lockdown that only KILLS the agent account's processes is therefore
undone within seconds: supervisord respawns the desktop, and the respawned GNOME session runs the account's autostart
entries (measured 2026-10-02 on container-ubuntu22-base: an entry written in ~/.config/autostart before the kill ran
again within 42 s of `pkill -KILL -u user`). A stopped process cannot exit, so supervisord does not respawn it: the freezer
SIGSTOPs every process of the agent account, polling /proc, until the verdict is written.

Usage in a task's verify.py (stdlib only; vendor this file into verify/):

    freezer = verify_freeze.AgentFreezer.for_user("user").start()   # right after the existing kill / lockdown
    ...                                                                # grading
    stats["agent_processes_frozen"] = float(len(freezer.frozen))
    write_verdict(...)
    freezer.stop()

`start()` is a no-op unless the caller is root and the account exists (so host-side tests and practice graders run the
same code). The agent account must not be root.
"""

import os
import signal
import threading


def uid_processes(uid):
    """PIDs of every live process owned by `uid` (Linux /proc)."""
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/status".format(name)) as fh:
                for line in fh:
                    if line.startswith("Uid:"):
                        if int(line.split()[1]) == int(uid):
                            pids.append(int(name))
                        break
        except (OSError, ValueError):
            continue
    return pids


class AgentFreezer(object):
    """SIGSTOP every process of `uid` every `period_s` seconds until stop()."""

    def __init__(self, uid, period_s=0.1, lister=uid_processes, killer=os.kill):
        self.uid = uid
        self.period_s = float(period_s)
        self.frozen = set()
        self._lister, self._killer = lister, killer
        self._stop = None
        self._t = None

    @classmethod
    def for_user(cls, user="user", **kw):
        import pwd
        try:
            uid = pwd.getpwnam(user).pw_uid
        except KeyError:
            uid = None
        return cls(uid if uid not in (None, 0) else None, **kw)

    def freeze_now(self):
        """One pass: SIGSTOP every current process of the account; returns how many were signalled."""
        n = 0
        if self.uid is None:
            return 0
        for pid in self._lister(self.uid):
            if pid == os.getpid():
                continue
            try:
                self._killer(pid, signal.SIGSTOP)
                self.frozen.add(pid)
                n += 1
            except OSError:
                pass
        return n

    def _run(self):
        while not self._stop.is_set():
            try:
                self.freeze_now()
            except Exception:  # noqa: BLE001 - the freezer must never take the grader down
                pass
            self._stop.wait(self.period_s)

    def start(self, require_root=True):
        if self.uid is None or (require_root and os.getuid() != 0):
            return self
        try:
            self.freeze_now()
        except Exception:  # noqa: BLE001 - the freezer must never take the grader down
            pass
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        if self._t is not None:
            self._stop.set()
            self._t.join(timeout=5.0)
            self._t = None
