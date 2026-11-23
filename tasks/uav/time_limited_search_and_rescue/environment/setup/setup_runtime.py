"""Trusted setup: write the session's time stamp and start the commissioning service.

Only the client is made public. The service's code and ledger stay in a root-only folder; the graded
sites are not in this sandbox at all (they live in the verifier's tests/ stage).
"""
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    stage = Path(__file__).resolve().parent
    stage.chmod(0o700)
    private = Path("/var/lib/visual_robotics")
    if private.exists():
        raise RuntimeError("refusing to reset an existing session")
    private.mkdir(mode=0o700)
    shutil.copytree(stage/"runtime", private/"runtime")
    public = Path("/opt/visual_robotics/client")
    public.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(stage/"payload", public, dirs_exist_ok=True)
    # Public clients are immutable; the agent can copy them but cannot change MCP execution.
    for path in public.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    public.chmod(0o755)
    home = Path("/home/user/robot")
    shutil.copytree(public, home, dirs_exist_ok=True)
    for path in [home, *home.rglob("*")]:
        os.chown(path, 1000, 1000)
    script = Path("/home/user/session_time.sh")
    shutil.copyfile(stage/"payload"/"session_time.sh", script)
    os.chown(script, 0, 0)
    script.chmod(0o755)
    submission = Path("/home/user/submission")
    submission.mkdir(exist_ok=True)
    os.chown(submission, 1000, 1000)
    # The session's time stamp, in a folder that is root's: the agent's account can read it and
    # cannot change or replace it.
    session = Path("/run/session")
    session.mkdir(mode=0o755, exist_ok=True)
    os.chown(session, 0, 0)
    session.chmod(0o755)
    stamp = session/"start"
    stamp.write_text(str(int(time.time()))+"\n")
    os.chown(stamp, 0, 0)
    stamp.chmod(0o444)
    with (private/"service.log").open("wb") as log:
        process = subprocess.Popen(
            ["/opt/venv/bin/python3", "-E", "-s", str(private/"runtime"/"service.py"), "--mode", "commissioning",
             "--directory", str(private/"commissioning")],
            stdout=log, stderr=log, start_new_session=True, cwd=private/"runtime",
            env={"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8",
                 "MUJOCO_GL": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "2",
                 "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                 "OPENBLAS_CORETYPE": "HASWELL",
                 "NPY_DISABLE_CPU_FEATURES": "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR"},
        )
    for _ in range(600):
        if process.poll() is not None:
            raise RuntimeError((private/"service.log").read_text())
        if Path("/run/visual_robotics/world.sock").exists():
            return
        time.sleep(.1)
    process.terminate()
    raise RuntimeError("aircraft service did not become ready")


if __name__ == "__main__":
    main()
