#!/usr/bin/env python3
"""Second process of render_twin.py: path-trace one USD stage with NVIDIA OVRTX and save the raw outputs.

    <ovrtx python> -I -B ovrtx_job.py <job.json>

Runs in the OVRTX Python (ovrtx, ovstage, numpy) and imports nothing else: no MuJoCo, and no GL context of any
kind, in this process. The job file names the stage, its RenderProduct and RenderVars, the output .npy files and
the log file. The renderer is attached to an ovstage stage (ovrtx's recommended mode), the product is stepped until
the path tracer reports convergence (one step normally suffices), and the colour outputs are copied to the CPU.

Stdout carries progress lines for the user, 'RENDER_TWIN_PHASE <name>' before each phase (import, start, load,
render, teardown) and one 'RENDER_TWIN_JOB {json}' result record. The record is printed as soon as the outputs are
saved (or the job has failed), before the renderer is shut down, so a hang in the shut-down cannot lose a finished
render. With an empty OVRTX cache the 'load' phase compiles shaders and materials for many minutes.

When started by render_twin.py ($RENDER_TWIN_PARENT_PID set), it ends itself (exit 4) as soon as that process is
gone, in addition to the SIGKILL render_twin.py arranges for that case. Exit codes: 0 rendered; 3 OVRTX could not
start (no usable NVIDIA GPU, driver or Vulkan); 4 the render failed.
"""
import json
import os
import sys
import threading
import time

EXIT_START, EXIT_RENDER = 3, 4


def emit(rec):
    print("RENDER_TWIN_JOB " + json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}), flush=True)


def phase(rec, name):
    rec["phase"] = name
    print("RENDER_TWIN_PHASE " + name, flush=True)


def log_tail(path, n=40):
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    keep = [ln.rstrip() for ln in lines if "[Error]" in ln or "error" in ln.lower() or "failed" in ln.lower()]
    return "\n".join((keep or [ln.rstrip() for ln in lines])[-n:])


def watch(rec, t0, parent, every=30.0):
    """A background thread: once a second, end this process if render_twin (the parent) is gone; every `every`
    seconds, print a progress line (the main thread may be blocked in OVRTX for many minutes)."""
    def run():
        last = time.time()
        while not rec.get("_done"):
            time.sleep(1.0)
            if parent and os.getppid() != parent:
                os._exit(EXIT_RENDER)
            if rec.get("_done") or time.time() - last < every:
                continue
            last = time.time()
            try:
                print("[render_twin] still working (%s, %.0f s so far; with an empty OVRTX cache the first render "
                      "compiles shaders and materials for 10-25 min, in the load phase)"
                      % (rec.get("phase"), time.time() - t0), flush=True)
            except OSError:                       # nobody reads the progress any more: the parent is gone
                os._exit(EXIT_RENDER)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def teardown(renderer, stage):
    try:
        if renderer is not None and stage is not None:
            renderer.detach_ovstage()
    except Exception:  # noqa: BLE001
        pass
    try:
        if stage is not None:
            stage.destroy()
    except Exception:  # noqa: BLE001
        pass
    try:
        if renderer is not None:
            renderer.destroy()
    except Exception:  # noqa: BLE001
        pass


def reap_children():
    """Collect the exit status of any child process OVRTX left behind (only once the renderer is destroyed)."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def main():
    job = json.load(open(sys.argv[1]))
    t0 = time.time()
    rec = {"ok": False, "phase": None}
    try:
        parent = int(os.environ.get("RENDER_TWIN_PARENT_PID") or 0)
    except ValueError:
        parent = 0
    watch(rec, t0, parent)
    phase(rec, "import")
    try:
        import numpy as np
        import ovrtx
        import ovstage
    except Exception as e:  # noqa: BLE001 - any import failure means OVRTX is unusable here
        rec.update(error="cannot import ovrtx / ovstage: %s" % e, _done=True)
        emit(rec)
        return EXIT_START
    rec["ovrtx_version"] = getattr(ovrtx, "__version__", "?")
    log_path = os.path.abspath(job["log"])
    cfg = ovrtx.RendererConfig(
        log_file_path=log_path, log_level="warn",
        enable_crash_reporter=False,                       # no crash handlers, no dump uploads
        active_cuda_gpus="0",                              # the first CUDA-visible GPU (CUDA_VISIBLE_DEVICES selects it)
        texture_streaming_mode=ovrtx.TextureStreamingMode.DISABLE,   # full-resolution textures in the first frame
        selection_outline_enabled=False,
        motion_bvh=ovrtx.MotionBvh.DISABLE)
    renderer = stage = None
    phase(rec, "start")
    try:
        renderer = ovrtx.Renderer(config=cfg)
    except Exception as e:  # noqa: BLE001
        rec.update(error=str(e).splitlines()[0] if str(e) else repr(e), log=log_tail(log_path), _done=True)
        emit(rec)
        return EXIT_START
    rec["t_start_s"] = round(time.time() - t0, 2)
    code = EXIT_RENDER
    try:
        phase(rec, "load")
        stage = ovstage.Stage("render_twin")
        renderer.attach_ovstage(stage)
        ordinal = 1
        ovstage.population.open_usd(stage, os.path.abspath(job["stage"]), ordinal=ordinal)
        stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
        t1 = time.time()
        rec["t_load_s"] = round(t1 - t0 - rec["t_start_s"], 2)
        phase(rec, "render")
        prod = job["product"]
        frame = None
        steps = 0
        for steps in range(1, int(job.get("max_steps", 8)) + 1):
            op = renderer.step_async({prod}, 1.0 / 60.0, ordinal=ordinal)
            pending = None
            while pending is None:
                pending = op.wait(timeout_ns=5_000_000_000)
            out = pending.fetch()
            frame = out[prod].frames[-1]
            if frame.converged:
                break
        rec["steps"] = steps
        rec["converged"] = bool(frame.converged)
        rec["progression"] = int(frame.progression)
        for key, var in job["vars"].items():
            mapped = frame.render_vars[var].map(device=ovrtx.Device.CPU)
            try:
                arr = np.from_dlpack(mapped).copy()
            finally:
                mapped.unmap()
            np.save(job["outputs"][key], arr)
            rec.setdefault("shapes", {})[key] = list(arr.shape)
        del frame, out
        rec["t_render_s"] = round(time.time() - t1, 2)
        rec["ok"] = True
        rec["phase"] = "done"
        code = 0
    except Exception as e:  # noqa: BLE001
        rec.update(error="%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""), log=log_tail(log_path))
    rec["t_total_s"] = round(time.time() - t0, 2)
    emit(rec)                                     # before the shut-down, which must not be able to lose the image
    phase(rec, "teardown")
    teardown(renderer, stage)
    reap_children()
    rec["_done"] = True
    return code


if __name__ == "__main__":
    sys.exit(main())
