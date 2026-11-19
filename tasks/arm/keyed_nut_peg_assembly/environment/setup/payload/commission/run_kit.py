"""Commissioning run: runs YOUR controller over the grading wire in one commissioning episode on a development kit.

    /opt/venv/bin/python3 /home/user/commission/run_kit.py --kit K [--submission /home/user/submission]
        [--final-images DIR] [--record FILE]

It starts your controller (bash run.sh in a fresh copy of --submission) and drives it the way the grader does, with
the commissioning service's cell in the place of the grader's: READY within 120 s, each reply within 30 s, repeated
step indices with perturbed observations (only the reply to the real observation is applied), replies longer than
64 KiB abort the episode, only AF_UNIX sockets (the same seccomp filter as at grading time), and the per-kit
controller budget (600 s: start-up and reply time plus 0.45 s per left or right image and 0.35 s per wrist image
asked for). The episode is one of your commissioning window's (data sheet, section 10): it counts from the moment
the controller is ready and the kit is laid out.

Like the cell itself, it reports no verdict: the steps run and how the episode ended (all steps run, a safety
stop: a ruined nut or an unstable simulation, an abort of the wire with its reason, or the controller budget used
up), your controller's budget use and image count, and the wall time. It does not judge the assembly (no completion
verdict, no count of assembled nuts, no ground truth); how a kit went, you judge from what your controller saw and
recorded, or from the cameras' images of the final state (--final-images DIR saves the left, right and wrist images
at the end as PNG files). --record FILE writes every observation's readings (no images) and every applied action as
JSON lines. Since nothing here judges completion, the episode runs all its steps unless a safety stop, an abort or
the budget ends it (the grader ends an episode as soon as its kit is complete).
"""

import argparse
import json
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import CommissioningError, Episode, decode_image, status  # noqa: E402

READY_TIMEOUT = 120.0
STEP_TIMEOUT = 30.0
MAX_LINE = 65536
CONTROLLER_BUDGET = 600.0                                   # s per kit, as when graded
IMAGE_CHARGE = {"left": 0.45, "right": 0.45, "wrist": 0.35}  # s per image asked for, as when graded
CAMERA_NAMES = ("left", "right", "wrist")

_FILTER_SRC = r'''
import ctypes, os, platform, struct, sys
ARCH = {"x86_64": (0xC000003E, 41, 53, 425, 0x40000000), "aarch64": (0xC00000B7, 198, 199, 425, 0)}
def ins(c, jt, jf, k): return struct.pack("HBBI", c, jt, jf, k)
audit, ns, nsp, nu, x32 = ARCH[platform.machine()]
deny = 0x00050000 | 1; allow = 0x7FFF0000
p = [ins(0x20,0,0,4), ins(0x15,1,0,audit), ins(0x06,0,0,deny), ins(0x20,0,0,0)]
if x32: p += [ins(0x35,0,1,x32), ins(0x06,0,0,deny)]
p += [ins(0x15,0,1,nu), ins(0x06,0,0,deny), ins(0x15,2,0,ns), ins(0x15,1,0,nsp), ins(0x06,0,0,allow),
      ins(0x20,0,0,16), ins(0x15,0,1,1), ins(0x06,0,0,allow), ins(0x06,0,0,deny)]
prog = b"".join(p); buf = ctypes.create_string_buffer(prog, len(prog))
class F(ctypes.Structure): _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
f = F(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p)); libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(f), 0, 0) != 0:
    sys.stderr.write("[commission] could not install the network filter\n")
env = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/home/user", "LANG": "C.UTF-8"}
os.execvpe("bash", ["bash", "run.sh"], env)
'''


class WireError(Exception):
    pass


class Wire(object):
    """The controller's side of the grading wire (READY / reset / act / close), with the grader's deadlines."""

    def __init__(self, submission):
        self.proc = subprocess.Popen([sys.executable, "-I", "-c", _FILTER_SRC], cwd=submission,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True,
                                     bufsize=1, start_new_session=True)
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()
        deadline = time.monotonic() + READY_TIMEOUT          # one deadline for the whole handshake
        for _ in range(200):
            if self._read(READY_TIMEOUT, "READY", deadline).strip() == "READY":
                return
        raise WireError("solver never printed READY")

    @staticmethod
    def _pump(proc, lines):
        while True:
            try:
                line = proc.stdout.readline(MAX_LINE + 1)
            except (ValueError, OSError):
                break
            if not line or (len(line) > MAX_LINE and not line.endswith("\n")):
                break
            lines.put(line)
        lines.put(None)

    def _read(self, timeout, what, deadline):
        try:
            line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise WireError("solver exceeded its {:.0f} s {} deadline".format(timeout, what))
        if line is None:
            raise WireError("solver closed its stdout (or wrote a line over 64 KiB)")
        return line

    def request(self, msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise WireError(str(exc))
        deadline = time.monotonic() + STEP_TIMEOUT           # one deadline per request: blank lines do not extend it
        while True:
            line = self._read(STEP_TIMEOUT, "step", deadline).strip()
            if not line:
                continue
            try:
                reply = json.loads(line)
            except ValueError:
                raise WireError("non-JSON output on the protocol stdout: {!r}".format(line[:80]))
            if not isinstance(reply, dict):
                raise WireError("reply is not a JSON object")
            return reply

    def act(self, t, obs):
        reply = self.request({"type": "act", "t": t, "obs": obs})
        if "action" not in reply:
            raise WireError("solver error: {}".format(reply.get("error", "no action")))
        return reply["action"]

    def stop(self, polite=True):
        if polite:
            try:
                self.request({"type": "close"})
            except Exception:  # noqa: BLE001
                pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass


def check_action(action):
    """The grader's test of a reply's action: a list of 8 finite numbers, the last an integer camera mask 0-7.
    Returns the cameras it asks for."""
    if not isinstance(action, (list, tuple)) or len(action) != 8:
        raise WireError("action must be a list of 8 numbers")
    for v in action:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) > 1e3:
            raise WireError("action must be a list of 8 numbers")
    cams = action[7]
    if cams != int(cams) or not 0 <= cams <= 7:
        raise WireError("action must be a list of 8 numbers")
    return [n for i, n in enumerate(CAMERA_NAMES) if int(cams) & (1 << i)]


def differ(a, b):
    try:
        aa, bb = [float(v) for v in a], [float(v) for v in b]
    except (TypeError, ValueError):
        return False
    return len(aa) != len(bb) or max([abs(x - y) for x, y in zip(aa, bb)] or [0.0]) > 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=int, required=True, help="the development kit (1 .. the number of kits)")
    ap.add_argument("--submission", default="/home/user/submission")
    ap.add_argument("--final-images", default=None, help="save the three cameras' images of the final state here")
    ap.add_argument("--record", default=None, help="write the observations' readings and the actions here (JSON lines)")
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.submission, "run.sh")):
        print(json.dumps({"error": "no run.sh in {}".format(a.submission)}))
        return 2
    rng = random.Random()
    tmp = tempfile.mkdtemp(prefix="commission-")
    work = os.path.join(tmp, "submission")
    shutil.copytree(a.submission, work, symlinks=True)       # like the grader: a fresh copy of the submission
    wire = ep = rec = None
    spent, images, steps, pairs, differed = 0.0, 0, 0, 0, 0
    ended, closing = None, {}
    t0 = time.monotonic()
    try:
        started = time.monotonic()
        wire = Wire(work)
        wire.request({"type": "reset"})
        spent += time.monotonic() - started
        ep = Episode(a.kit)                                  # the episode counts from here
        rec = open(a.record, "w") if a.record else None
        obs = ep.obs
        t = 0
        while not ep.done and obs is not None:
            other = dict(obs, **ep.probe_obs) if ep.probe_obs else None
            asked = time.monotonic()
            if other is not None:
                if rng.random() < 0.5:
                    b = wire.act(t, other)
                    act = wire.act(t, obs)
                else:
                    act = wire.act(t, obs)
                    b = wire.act(t, other)
                pairs += 1
                differed += int(differ(act, b))
            else:
                act = wire.act(t, obs)
            spent += time.monotonic() - asked
            cams = check_action(act)
            spent += sum(IMAGE_CHARGE[c] for c in cams)
            if spent > CONTROLLER_BUDGET:
                ended = "controller budget ({:.0f} s) used up at step {}".format(CONTROLLER_BUDGET, t)
                break
            images += len(cams)
            if rec is not None:
                rec.write(json.dumps({"t": t, "obs": {k: v for k, v in obs.items() if k not in CAMERA_NAMES},
                                      "action": act}) + "\n")
            obs = ep.step(act)
            t += 1
            steps = t
    except WireError as exc:
        ended = "aborted at step {}: {}".format(steps, str(exc)[:200])
    except CommissioningError as exc:
        if ep is None or exc.code != "closed":
            ended = "the commissioning service refused: {}".format(exc)
    finally:
        if rec is not None:
            rec.close()
        if wire is not None:
            wire.stop(polite=ended is None or not ended.startswith("aborted"))
        if ep is not None:
            try:
                closing = ep.close(images=bool(a.final_images))
            except (CommissioningError, OSError):
                closing = {}
        shutil.rmtree(tmp, ignore_errors=True)
    if a.final_images and closing.get("final_images"):
        from PIL import Image
        os.makedirs(a.final_images, exist_ok=True)
        for cam, img in closing["final_images"].items():
            Image.fromarray(decode_image(img)).save(os.path.join(a.final_images, "kit{:02d}_ep{:02d}_{}.png".format(
                a.kit, ep.episode, cam)))
    try:
        window = status()
    except (OSError, ValueError):
        window = None
    print(json.dumps({
        "kit": a.kit,
        "episode": ep.episode if ep is not None else None,
        "order": ep.first["obs"]["order"] if ep is not None else None,
        "steps": closing.get("steps", steps),
        "ended": ended or closing.get("ended") or (ep.ended if ep is not None else None),
        "controller_s": round(spent, 1), "controller_budget_s": CONTROLLER_BUDGET,
        "images_requested": images,
        "repeated_t_pairs": pairs, "repeated_t_pairs_that_differed": differed,
        "wall_s": round(time.monotonic() - t0, 1),
        "window": window,
        "note": "a commissioning run: it reports how the episode ended, never whether the kit was assembled",
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
