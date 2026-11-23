"""Trusted setup: start the commissioning service; only the client, not the world, is made public."""
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
        raise RuntimeError("refusing to reset an existing episode")
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
    # Session time: a start stamp the agent's account can read and cannot change, and a script that reads it.
    session = Path("/run/session")
    session.mkdir(mode=0o755, exist_ok=True)
    os.chown(session, 0, 0)
    session.chmod(0o755)
    stamp = session/"start"
    if not stamp.exists():
        stamp.write_text(str(int(time.time()))+"\n")
    os.chown(stamp, 0, 0)
    stamp.chmod(0o444)
    script = Path("/home/user/session_time.sh")
    shutil.copyfile(public/"session_time.sh", script)
    os.chown(script, 0, 0)
    script.chmod(0o755)
    with (private/"service.log").open("wb") as log:
        process = subprocess.Popen(
            ["/opt/venv/bin/python3", "-E", "-s", str(private/"runtime"/"service.py"), "--mode", "commissioning",
             "--directory", str(private/"commissioning")],
            stdout=log, stderr=log, start_new_session=True, cwd=private/"runtime",
            env={"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8",
                 "MUJOCO_GL": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "2",
                 "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"},
        )
    for _ in range(600):
        if process.poll() is not None:
            raise RuntimeError((private/"service.log").read_text())
        if Path("/run/visual_robotics/world.sock").exists():
            return
        time.sleep(.1)
    process.terminate()
    raise RuntimeError("robot service did not become ready")


if __name__ == "__main__":
    main()
