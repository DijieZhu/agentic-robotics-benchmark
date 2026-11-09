#!/usr/bin/env python3
"""Render a MuJoCo scene through one of its cameras with NVIDIA OVRTX (RTX path tracing) on this machine's GPU.

    python3 render_twin.py scene.xml --out render.png [--camera photo_cam] [--spp 64] [--blend photo.png]
                           [--timeout 1800] [--size WxH] [--groups 0,1,2] [--dome-lux 60] [--exposure-ev 0]
                           [--denoise] [--tonemap srgb|ovrtx] [--keep DIR] [--json]
    python3 render_twin.py --warmup          # render a small built-in scene once, so the shader cache is filled

What it does. The scene is compiled with MuJoCo and its initial state (qpos0) is computed; every geom MuJoCo's viewer
would draw (groups 0, 1 and 2 unless --groups says otherwise; alpha above zero) is written, with its materials and
textures, into a self-contained OpenUSD stage, together with the scene's lights in physical units, a uniform dome
light and the camera as an exact pinhole: the camera's own resolution, fx, fy, cx and cy as MuJoCo computes them
(ovrtx_render/camera.py states the rule). A second process (the OVRTX Python, $ARB_OVRTX_PYTHON) path-traces the
stage and the image is written as an 8-bit sRGB PNG with the camera's resolution. With --blend, a 50 % mix of the
render and the given photo is written next to it (<out>_blend.png).

Light and exposure. Spot and point lights emit `intensity` candela along their axis, directional lights give
`intensity` lux; a light without `intensity` gets 1000 cd (spot, point) or 400 lux (directional) per unit of its
largest diffuse component. The dome light adds --dome-lux lux on an upward-facing surface (0 turns it off; it is
also the background). The camera is exposed like ISO 100, f/2, 1/25 s; --exposure-ev adds stops. MuJoCo's headlight
and ambient terms have no physical counterpart and are not rendered. --tonemap srgb (default) encodes the linear
exposed image as sRGB and clips it; --tonemap ovrtx uses OVRTX's own tone-mapped output.

Time. With an empty OVRTX cache, the first render compiles shaders and materials for about 10 to 25 minutes (on
the CPU, while the stage loads; longer when the machine is busy) and needs up to about 13 GB of memory; later renders of a 1920 x 1080 image
take 15 to 25 s. Start a first render in the background. Progress lines are printed meanwhile. One render runs at a
time on the machine: a lock file, held until the OVRTX process has exited; a second render waits for it. --timeout
bounds the whole render, waiting included.

Exit codes: 0 rendered; 2 the scene cannot be rendered (it does not compile, the camera is missing or has no
resolution); 3 OVRTX cannot start here (no NVIDIA GPU, a driver without what OVRTX needs, or a Python without the
helper's modules); 4 the render failed; 5 --timeout ran out (waiting for another render, or rendering); 130, 143
stopped by a signal. ovrtx_render/README.md has the details. Renders and caches are best kept outside the
submission folder.
"""
import argparse
import ctypes
import fcntl
import json
import os
import pwd
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXIT_SCENE, EXIT_START, EXIT_RENDER, EXIT_TIMEOUT = 2, 3, 4, 5
DEFAULT_OVRTX_PYTHON = "/opt/ovrtx-venv/bin/python"
TASK_PYTHON = os.environ.get("RENDER_TWIN_PYTHON", "/opt/venv/bin/python3")   # the Python with the modules below
NEEDED_MODULES = ("numpy", "mujoco", "pxr", "PIL")
LOCK_PATH = os.environ.get("RENDER_TWIN_LOCK", os.path.join(tempfile.gettempdir(), "render_twin.lock"))
WORK_ROOT = os.path.join(tempfile.gettempdir(), "render_twin_work-%d" % os.getuid())
TERM_GRACE_S = 5.0          # after SIGTERM, how long the OVRTX process gets before everything left is killed
# after it reported a saved image, how long it gets to shut the renderer down (normally 1-2 s)
TEARDOWN_GRACE_S = float(os.environ.get("RENDER_TWIN_TEARDOWN_GRACE_S") or 60.0)
GPU_HINT = ("render_twin needs an NVIDIA RTX GPU whose driver exposes Vulkan graphics to this machine "
            "(OVRTX path tracing)")
# Not passed to the OVRTX process. The Python variables belong to the task's Python. The display and desktop-session
# variables: OVRTX renders off-screen, and when it can reach an X display it compiles a different set of shaders, so a
# render whose caller has a display and one whose caller has none do not share the shader cache (measured: a render
# without DISPLAY after a warm-up with one compiled again for 15 min). Without them every render is headless and
# finds the same cache, whatever the caller's environment.
JOB_ENV_DROP = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "MUJOCO_GL", "PYOPENGL_PLATFORM",
                "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR",
                "XDG_SESSION_ID", "XDG_SESSION_TYPE", "XDG_SESSION_CLASS", "XDG_SESSION_DESKTOP", "XDG_CURRENT_DESKTOP")

# Linux prctl options, called through ctypes (best effort: without them the process tree is still cleaned up on
# every exit path this process lives through)
PR_SET_PDEATHSIG, PR_SET_CHILD_SUBREAPER = 1, 36
# True when render_twin.py is the program (main() sets it): it then starts no child but the OVRTX process, so every
# other child it has is a leftover of that process. Imported as a module, only the OVRTX process's own tree is touched.
OWN_ALL_CHILDREN = False
try:
    _PRCTL = ctypes.CDLL(None, use_errno=True).prctl
    _PRCTL.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    _PRCTL.restype = ctypes.c_int
except (OSError, AttributeError):
    _PRCTL = None


def prctl(option, value):
    return _PRCTL is not None and _PRCTL(option, value, 0, 0, 0) == 0


def say(msg):
    print("[render_twin] " + msg, flush=True)


def fail(code, msg):
    say("ERROR: " + msg)
    sys.exit(code)


def parse_size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("--size must look like 1920x1080")


def srgb_encode(lin):
    import numpy as np
    x = np.clip(lin, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def gpu_present():
    return any(os.path.exists(p) for p in ("/dev/nvidiactl", "/dev/nvidia0"))


def ensure_task_python():
    """Make sure this Python has the helper's modules. If it has not and the task's Python exists, run this script
    again with that Python (same arguments); otherwise stop with exit code 3."""
    missing = []
    for name in NEEDED_MODULES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return
    if (os.path.isfile(TASK_PYTHON) and os.access(TASK_PYTHON, os.X_OK) and not os.environ.get("RENDER_TWIN_REEXEC")
            and os.path.abspath(sys.executable) != os.path.abspath(TASK_PYTHON)):
        say("note: this Python (%s) lacks %s; running render_twin with %s instead"
            % (sys.executable, ", ".join(missing), TASK_PYTHON))
        env = dict(os.environ, RENDER_TWIN_REEXEC="1")
        os.execve(TASK_PYTHON, [TASK_PYTHON, os.path.abspath(__file__)] + sys.argv[1:], env)
    fail(EXIT_START, "this Python (%s) lacks %s, which render_twin needs; run it with the task's Python (%s)"
         % (sys.executable, ", ".join(missing), TASK_PYTHON))


class Lock:
    """An exclusive flock on a file every account can open (flock works on a read-only descriptor).

    The descriptor is handed to the OVRTX process (render_twin passes it on), so the lock stays held for as long as
    that process, or anything it started, is alive, even when this process is killed first. It is therefore never
    unlocked explicitly: closing the last descriptor releases it."""

    def __init__(self, path):
        self.path = path
        self.fd = None

    @staticmethod
    def _open(path):
        try:
            return os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            pass
        old = os.umask(0)
        try:
            return os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
        except PermissionError:
            # another account created it just now: with fs.protected_regular, O_CREAT on another account's file in a
            # sticky folder such as /tmp is refused even where a plain open is allowed
            return os.open(path, os.O_RDONLY)
        finally:
            os.umask(old)

    def acquire(self, deadline):
        try:
            self.fd = self._open(self.path)
        except OSError as e:
            alt = os.path.join(tempfile.gettempdir(), "render_twin-%d.lock" % os.getuid())
            say("note: the render lock %s cannot be opened (%s); using %s, which does not wait for other accounts' "
                "renders" % (self.path, e.strerror or e, alt))
            self.path = alt
            try:
                self.fd = self._open(alt)
            except OSError as e2:
                fail(EXIT_RENDER, "no render lock file can be opened (%s: %s)" % (alt, e2.strerror or e2))
        noted = False
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if not noted:
                    say("another render is running; waiting for it to finish (lock %s)" % self.path)
                    noted = True
                if time.time() > deadline:
                    return False
                time.sleep(0.25)

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# ---- the OVRTX process and its process tree ---------------------------------------------------------------------
def _proc_table():
    """{pid: (ppid, state)} of every process this one can see in /proc."""
    table = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return table
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % name, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        fields = data[data.rfind(b")") + 2:].split()       # the command name may contain spaces and brackets
        if len(fields) >= 2:
            table[int(name)] = (int(fields[1]), fields[0].decode("ascii", "replace"))
    return table


def _descendants(root, table):
    kids = {}
    for pid, (ppid, _) in table.items():
        kids.setdefault(ppid, []).append(pid)
    out, todo = [], [root]
    while todo:
        for child in kids.get(todo.pop(), ()):
            out.append(child)
            todo.append(child)
    return out


def _signal(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _renderer_preexec(parent_pid):
    """Runs in the forked child just before it becomes the OVRTX Python: it gets SIGKILL when this process dies,
    however that happens, and it ends at once if this process is already gone."""
    sigkill = int(signal.SIGKILL)

    def preexec():
        prctl(PR_SET_PDEATHSIG, sigkill)
        if os.getppid() != parent_pid:
            os._exit(EXIT_RENDER)
    return preexec


def stop_renderer(proc):
    """SIGTERM to the OVRTX process, TERM_GRACE_S to exit, then SIGKILL to it and everything it started."""
    if proc.poll() is not None:
        return
    _signal([proc.pid], signal.SIGTERM)
    try:
        proc.wait(timeout=TERM_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    # not reaped yet, so its pid and its children's parent links still belong to it
    _signal(_descendants(proc.pid, _proc_table()) + [proc.pid], signal.SIGKILL)
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        say("note: the OVRTX process %d did not end within 10 s of SIGKILL (the lock stays held until it does)"
            % proc.pid)


def reap_leftovers(exclude=(), budget_s=10.0):
    """Kill and reap what is left among this process's children: as a child subreaper it adopts the orphans of the
    OVRTX process (helpers it started, and their exited, unreaped entries) instead of the machine's init. Only for
    render_twin.py as the program (OWN_ALL_CHILDREN), which has no other children. Returns (killed, reaped)."""
    me = os.getpid()
    killed, reaped = set(), 0
    end = time.time() + budget_s
    while True:
        table = _proc_table()
        kids = [p for p, (pp, _) in table.items() if pp == me and p not in exclude]
        if not kids:
            return len(killed), reaped
        alive = [p for p in kids if table[p][1] not in ("Z", "X")]
        if alive:
            _signal([d for p in alive for d in _descendants(p, table)] + alive, signal.SIGKILL)
            killed.update(alive)
        for p in kids:
            try:
                got, _ = os.waitpid(p, os.WNOHANG)
                reaped += 1 if got else 0
            except ChildProcessError:
                pass
        if time.time() > end:
            say("note: %d leftover process(es) of the renderer could not be reaped" % len(kids))
            return len(killed), reaped
        time.sleep(0.05)


def run_job(ovrtx_python, job_path, work, deadline, lock_fd=None):
    """Run the OVRTX process and stream its progress lines. It stays in this process's process group (a signal to
    the caller's group reaches it), dies with this process (PR_SET_PDEATHSIG), and holds the render lock with an
    inherited descriptor. Returns a dict: code (exit code, negative for a signal), rec (its result record), phase
    (the last phase it announced), timed_out, teardown_killed, leftovers_killed, leftovers_reaped."""
    env = {k: v for k, v in os.environ.items() if k not in JOB_ENV_DROP}
    try:
        account = pwd.getpwuid(os.getuid()).pw_name
        env.setdefault("USER", account)
        env.setdefault("LOGNAME", account)
    except KeyError:
        pass
    env["PYTHONNOUSERSITE"] = "1"
    env["RENDER_TWIN_PARENT_PID"] = str(os.getpid())
    pass_fds = ()
    if lock_fd is not None:
        os.set_inheritable(lock_fd, True)
        pass_fds = (lock_fd,)
        env["RENDER_TWIN_LOCK_FD"] = str(lock_fd)
    out = {"code": None, "rec": None, "phase": None, "timed_out": False, "teardown_killed": False,
           "leftovers_killed": 0, "leftovers_reaped": 0}
    err = open(os.path.join(work, "ovrtx_stderr.txt"), "wb")
    proc = subprocess.Popen([ovrtx_python, "-I", "-B", os.path.join(HERE, "ovrtx_render", "ovrtx_job.py"), job_path],
                            stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL, env=env,
                            pass_fds=pass_fds, preexec_fn=_renderer_preexec(os.getpid()))
    fd = proc.stdout.fileno()
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    state = {"buf": b"", "open": True}

    def handle(raw):
        line = raw.decode("utf-8", "replace").rstrip("\r")
        if line.startswith("RENDER_TWIN_JOB "):
            try:
                out["rec"] = json.loads(line[len("RENDER_TWIN_JOB "):])
            except ValueError:
                print(line, flush=True)
        elif line.startswith("RENDER_TWIN_PHASE "):
            out["phase"] = line[len("RENDER_TWIN_PHASE "):].strip()
        elif line:
            print(line, flush=True)

    def pump(timeout):
        """Read what the OVRTX process wrote within `timeout` s: 'data', 'idle' or 'closed'. Raw reads, so that
        select() sees exactly what is pending."""
        if not state["open"]:
            return "closed"
        if not sel.select(timeout=timeout):
            return "idle"
        chunk = os.read(fd, 65536)
        if not chunk:
            state["open"] = False
            sel.unregister(fd)
            if state["buf"]:
                handle(state["buf"])
                state["buf"] = b""
            return "closed"
        lines = (state["buf"] + chunk).split(b"\n")
        state["buf"] = lines.pop()
        for raw in lines:
            handle(raw)
        return "data"

    def drain():
        while pump(0.05) == "data":
            pass

    done_at = None
    try:
        while True:
            now = time.time()
            if now >= deadline:
                out["timed_out"] = True
                break
            if done_at is None and (out["rec"] or {}).get("ok"):
                done_at = now                     # the image is saved; the renderer is shutting down
            if done_at is not None and now - done_at > TEARDOWN_GRACE_S:
                out["teardown_killed"] = True
                break
            if pump(min(1.0, deadline - now)) == "closed":
                try:
                    proc.wait(timeout=min(1.0, max(0.0, deadline - time.time())))
                except subprocess.TimeoutExpired:
                    pass
            if proc.poll() is not None:
                break
        if proc.poll() is None:
            stop_renderer(proc)
        drain()
        out["code"] = proc.returncode
    finally:
        if proc.poll() is None:                   # left by an exception or a signal (SIGTERM, Ctrl-C)
            stop_renderer(proc)
        if OWN_ALL_CHILDREN:
            out["leftovers_killed"], out["leftovers_reaped"] = reap_leftovers(
                exclude=() if proc.returncode is not None else (proc.pid,))
        sel.close()
        proc.stdout.close()
        err.close()
    return out


def compile_scene(args):
    import mujoco
    try:
        if args.warmup:
            from ovrtx_render import warmup
            m = mujoco.MjModel.from_xml_string(warmup.SCENE_XML)
        else:
            m = mujoco.MjModel.from_xml_path(os.path.abspath(args.scene))
    except Exception as e:  # noqa: BLE001 - MuJoCo raises ValueError with the compiler's message
        fail(EXIT_SCENE, "the scene does not compile: %s" % e)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def clean_stale_work_dirs():
    """Remove the work folders of earlier renders whose render_twin process is gone (it was killed)."""
    try:
        names = os.listdir(WORK_ROOT)
    except OSError:
        return
    for name in names:
        pid = name.split("-", 1)[0]
        if not pid.isdigit():
            continue
        try:
            os.kill(int(pid), 0)
            continue                              # still running (or the pid was reused): leave it
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        shutil.rmtree(os.path.join(WORK_ROOT, name), ignore_errors=True)


def describe_exit(code):
    if code is None:
        return "did not end"
    if code < 0:
        try:
            return "was killed by %s" % signal.Signals(-code).name
        except ValueError:
            return "was killed by signal %d" % -code
    return "exited with code %d" % code


def stderr_tail(work):
    try:
        with open(os.path.join(work, "ovrtx_stderr.txt"), errors="replace") as fh:
            return fh.read()[-2000:]
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("scene", nargs="?", help="the MJCF file to render")
    ap.add_argument("--out", help="the PNG to write (default render.png; warm-up: none)")
    ap.add_argument("--camera", default="photo_cam", help="the camera to render through (default photo_cam)")
    ap.add_argument("--spp", type=int, default=64, help="path-traced samples per pixel (default 64)")
    ap.add_argument("--blend", metavar="PHOTO", help="also write <out>_blend.png, a 50 %% mix with this photo")
    ap.add_argument("--warmup", action="store_true", help="render the built-in warm-up scene (fills the shader cache)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds for the whole render (default 1800)")
    ap.add_argument("--size", type=parse_size, help="render at WxH instead of the camera's resolution")
    ap.add_argument("--groups", default="0,1,2", help="geom groups to draw (default 0,1,2)")
    ap.add_argument("--dome-lux", type=float, default=60.0, help="dome light illuminance in lux (default 60; 0 = off)")
    ap.add_argument("--exposure-ev", type=float, default=0.0, help="camera exposure offset in stops (default 0)")
    ap.add_argument("--denoise", action="store_true", help="run OVRTX's denoiser (where the driver provides it)")
    ap.add_argument("--tonemap", choices=("srgb", "ovrtx"), default="srgb", help="how to make the 8-bit image")
    ap.add_argument("--keep", metavar="DIR", help="keep the USD stage and raw outputs in DIR")
    ap.add_argument("--json", action="store_true", help="print the summary as one JSON line at the end")
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + args.timeout
    if not args.warmup and not args.scene:
        ap.error("give a scene file, or --warmup")
    out = args.out or (None if args.warmup else "render.png")
    if args.spp < 1:
        ap.error("--spp must be at least 1")
    try:
        groups = tuple(int(g) for g in args.groups.split(",") if g.strip() != "")
    except ValueError:
        ap.error("--groups must be a comma-separated list of integers")
    ensure_task_python()
    # orphans of the OVRTX process come to this process (which kills and reaps them) instead of to init
    global OWN_ALL_CHILDREN
    OWN_ALL_CHILDREN = True
    prctl(PR_SET_CHILD_SUBREAPER, 1)

    ovrtx_python = os.environ.get("ARB_OVRTX_PYTHON", DEFAULT_OVRTX_PYTHON)
    if not gpu_present():
        fail(EXIT_START, "no NVIDIA GPU device in this machine (/dev/nvidiactl is missing). " + GPU_HINT)
    if not os.path.isfile(ovrtx_python):
        fail(EXIT_START, "the OVRTX Python %s does not exist (set ARB_OVRTX_PYTHON). %s" % (ovrtx_python, GPU_HINT))

    m, d = compile_scene(args)
    from ovrtx_render import camera as CAM
    from ovrtx_render import export_usd
    if args.keep:
        work = os.path.abspath(args.keep)
        os.makedirs(work, exist_ok=True)
    else:
        os.makedirs(WORK_ROOT, mode=0o700, exist_ok=True)
        clean_stale_work_dirs()
        work = tempfile.mkdtemp(prefix="%d-" % os.getpid(), dir=WORK_ROOT)
    try:
        opts = {"spp": 16 if args.warmup and args.spp == 64 else args.spp, "dome_lux": args.dome_lux,
                "exposure_ev": args.exposure_ev, "denoise": bool(args.denoise)}
        t1 = time.time()
        try:
            info = export_usd.export_scene(m, d, work, camera=args.camera, size=args.size, groups=groups, opts=opts)
        except CAM.CameraError as e:
            fail(EXIT_SCENE, str(e))
        t_export = time.time() - t1
        for w in info["warnings"]:
            say("note: " + w)
        cam = info["camera"]
        say("exported %d geoms (%d triangles, %d materials, %d textures, %d lights) in %.1f s; camera %s %dx%d "
            "fx %.2f fy %.2f cx %.2f cy %.2f" % (info["counts"]["geoms"], info["counts"]["triangles"],
                                                  info["counts"]["materials"], info["counts"]["textures"],
                                                  info["counts"]["lights"], t_export, cam["name"], cam["W"], cam["H"],
                                                  cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
        job = {"stage": info["stage"], "product": "/Render/Product",
               "vars": {"ldr": "/Render/Vars/LdrColor", "hdr": "/Render/Vars/HdrColor"},
               "outputs": {"ldr": os.path.join(work, "ldr.npy"), "hdr": os.path.join(work, "hdr.npy")},
               "log": os.path.join(work, "ovrtx.log"), "max_steps": 8}
        job_path = os.path.join(work, "job.json")
        with open(job_path, "w") as fh:
            json.dump(job, fh, indent=1)
        lock = Lock(LOCK_PATH)
        t_wait = time.time()
        if not lock.acquire(deadline):
            lock.release()
            fail(EXIT_TIMEOUT, "--timeout %.0f s ran out after %.0f s of waiting for another render to finish (lock "
                 "%s)" % (args.timeout, time.time() - t_wait, lock.path))
        t_locked = time.time()
        t_wait = t_locked - t_wait
        try:
            say("rendering %dx%d at %d spp with OVRTX (with an empty OVRTX cache the first render compiles shaders "
                "and materials for 10-25 min)" % (cam["W"], cam["H"], opts["spp"]))
            t2 = time.time()
            res = run_job(ovrtx_python, job_path, work, deadline, lock.fd)
            t_ovrtx = time.time() - t2
        finally:
            lock.release()                    # the OVRTX process tree is gone by now (or holds its own copy)
        code, rec, phase = res["code"], res["rec"], res["phase"]
        notes = []
        if res["leftovers_killed"]:
            notes.append("killed %d leftover process(es) of the renderer" % res["leftovers_killed"])
        if rec is not None and rec.get("ok"):
            if res["teardown_killed"]:
                notes.append("the renderer did not shut down within %.0f s of saving the image and was killed"
                             % TEARDOWN_GRACE_S)
            elif code != 0:
                notes.append("the renderer %s while shutting down, after saving the image" % describe_exit(code))
        elif res["timed_out"]:
            fail(EXIT_TIMEOUT, "--timeout %.0f s ran out while rendering (phase %s); the OVRTX process was stopped"
                 % (args.timeout, (rec or {}).get("phase") or phase or "start"))
        else:
            detail = (rec or {}).get("error") or "the OVRTX process %s" % describe_exit(code)
            logt = (rec or {}).get("log", "")
            where = (rec or {}).get("phase") or phase
            if code == EXIT_START or where in (None, "import", "start"):
                fail(EXIT_START, "OVRTX could not start (phase %s): %s%s\n%s%s"
                     % (where or "before start", detail, ("\n" + logt) if logt else "", GPU_HINT,
                        ("\n" + stderr_tail(work)) if rec is None else ""))
            tail = stderr_tail(work)
            fail(EXIT_RENDER, "the render failed (phase %s): %s%s%s" % (where, detail, ("\n" + logt) if logt else "",
                                                                       ("\n" + tail) if tail else ""))
        for n in notes:
            say("note: " + n)
        import numpy as np
        from PIL import Image
        if args.tonemap == "ovrtx":
            rgb = np.load(job["outputs"]["ldr"])[..., :3].astype(np.uint8)
        else:
            hdr = np.load(job["outputs"]["hdr"]).astype(np.float32)[..., :3]
            rgb = np.round(srgb_encode(np.nan_to_num(hdr, nan=0.0, posinf=1.0, neginf=0.0)) * 255.0).astype(np.uint8)
        summary = {"ok": True, "out": None, "camera": cam, "spp": opts["spp"], "converged": rec.get("converged"),
                   "steps": rec.get("steps"), "t_export_s": round(t_export, 2), "t_lock_wait_s": round(t_wait, 2),
                   "lock_acquired_unix": round(t_locked, 3),
                   "t_ovrtx_s": round(t_ovrtx, 2), "t_ovrtx_render_s": rec.get("t_render_s"),
                   "t_ovrtx_start_s": rec.get("t_start_s"), "t_ovrtx_load_s": rec.get("t_load_s"),
                   "ovrtx_exit": code, "notes": notes,
                   "mean_rgb": [round(float(v), 1) for v in rgb.reshape(-1, 3).mean(axis=0)],
                   "clipped_fraction": round(float((rgb.max(axis=2) >= 255).mean()), 4),
                   "counts": info["counts"], "warnings": info["warnings"]}
        if out:
            out_path = os.path.abspath(out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            Image.fromarray(rgb).save(out_path)
            summary["out"] = out_path
            if args.blend:
                photo = Image.open(args.blend).convert("RGB")
                if photo.size != (rgb.shape[1], rgb.shape[0]):
                    say("note: the photo is %dx%d, the render %dx%d; the photo is resized for the blend"
                        % (photo.size[0], photo.size[1], rgb.shape[1], rgb.shape[0]))
                    photo = photo.resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR)
                mix = (0.5 * np.asarray(photo, np.float32) + 0.5 * rgb.astype(np.float32)).round().astype(np.uint8)
                stem, ext = os.path.splitext(out_path)
                blend_path = stem + "_blend" + (ext or ".png")
                Image.fromarray(mix).save(blend_path)
                summary["blend"] = blend_path
        summary["t_total_s"] = round(time.time() - t0, 2)
        say("done in %.1f s (OVRTX %.1f s, render %.1f s%s): %s; mean sRGB %s, %.1f %% of pixels clipped"
            % (summary["t_total_s"], t_ovrtx, rec.get("t_render_s") or 0.0,
               "" if rec.get("converged") else ", NOT converged", summary["out"] or "warm-up, no image kept",
               summary["mean_rgb"], 100.0 * summary["clipped_fraction"]))
        if args.json:
            print(json.dumps(summary), flush=True)
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    def _stop(signum, frame):
        # one-shot: a second signal must not cut short the clean-up the first one starts (SIGKILL still works)
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)
    for _s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(_s, _stop)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("interrupted")
        sys.exit(130)
