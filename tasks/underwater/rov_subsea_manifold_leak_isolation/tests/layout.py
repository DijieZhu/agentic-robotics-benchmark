"""Where the grader's own copies live in the verify stage folder (tests/), and the environment the cell runs in.

Every path is relative to the stage folder (this file's folder); tools/assemble_task.py in the development tree writes
exactly this layout, and both verify.py (the parent) and station.py (one station's cell) read it from here, so the
two cannot disagree.  Standard library only (verify.py imports it before anything else).

    cellkit/          the process kit (byte copy shared with the other program-under-test tasks)
    robotics_grader/  the benchmark's grading kit (byte copy)
    subseacell/       the vehicle cell: server, firmware, world, sensors, scoring (the practice ground's copy is the
                      same bytes, less scoring.py)
    kit/              the simulation kit (uwkit/, byte copy) and the reference vehicle's two data files next to it
    meshes/           the vehicle's pinned meshes: coll/ (physics) and lo/ (the render twin's look)
    data/             task_values.json (every number of the task the cell and the scoring read)
    reference/        the privileged reference program (only a token-privileged run loads it) and its perception
    hidden/           the fixed hidden set: stations.json (pinned packages), sites/*.tar.xz, selection.json,
                      oracle_token.sha256
    setup_payload/    the givens (byte copy of environment/setup/payload/); tests/test.sh builds /opt/arb from it
"""

import os
import stat

STAGE = os.path.dirname(os.path.abspath(__file__))

KIT_DIR = os.path.join(STAGE, "kit")
MESH_DIR = os.path.join(STAGE, "meshes")
VALUES_PATH = os.path.join(STAGE, "data", "task_values.json")
REFERENCE_DIR = os.path.join(STAGE, "reference")
HIDDEN_DIR = os.path.join(STAGE, "hidden")
CELL_DIR = os.path.join(STAGE, "subseacell")
STATION_SCRIPT = os.path.join(STAGE, "station.py")


def cell_env(threads=1):
    """The variables the cell (and the reference program inside it) needs: the kit's data files, the pinned meshes,
    the task values, single-threaded numerics and OSMesa rendering with one rasteriser thread."""
    t = str(int(threads))
    return {
        "UWKIT_SPEC_VALUES": os.path.join(KIT_DIR, "spec_values.json"),
        "UWKIT_MANIFEST": os.path.join(KIT_DIR, "vehicle_parts_manifest.json"),
        "UWKIT_MESHLIB": MESH_DIR,
        "SUBSEACELL_VEHICLE_MESHES": MESH_DIR,
        "SUBSEACELL_VISUAL_MESHES": MESH_DIR,
        "SUBSEACELL_TASK_VALUES": VALUES_PATH,
        "SUBSEACELL_SPEC_VALUES": os.path.join(KIT_DIR, "spec_values.json"),
        "OMP_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t, "MKL_NUM_THREADS": t, "LP_NUM_THREADS": "1",
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"), "PYOPENGL_PLATFORM": "osmesa",
    }


# the program's environment (cellkit.runner.child_env) less the rendering variables: their names carry the simulator's
# name and a submitted program can read os.environ; the program never renders (P24a, request of P23)
PROGRAM_ENV_DROP = ("MUJOCO_GL", "PYOPENGL_PLATFORM")
# the world-writable temporary folders that every account shares; closed to the runner accounts while programs run
SHARED_TMP_DIRS = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock")
SHARED_TMP_MODE = 0o1770


def program_env(env):
    """A copy of a program's environment without PROGRAM_ENV_DROP."""
    return {k: v for k, v in env.items() if k not in PROGRAM_ENV_DROP}


def close_shared_tmp(dirs=SHARED_TMP_DIRS, log=None):
    """While the programs of the stations run (P24a, decision P24-5): the shared world-writable folders lose every
    permission of "other" (mode 1770, owner root), so a runner account can neither create a file there nor list or
    read one - nothing can pass between two stations that run at the same time.  Each program's own TMPDIR and HOME
    are its station's private 0700 folders (cellkit.runner.prepare_run_dir), removed when the station ends.  Root only;
    returns [(path, mode)] for reopen_shared_tmp (empty when not root)."""
    saved = []
    if os.getuid() != 0:
        return saved
    for d in dirs:
        try:
            st = os.lstat(d)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode):
            continue                                  # a symbolic link (e.g. /var/lock): its target is in the list
        mode = stat.S_IMODE(st.st_mode)
        try:
            os.chmod(d, SHARED_TMP_MODE)
            saved.append((d, mode))
        except OSError as e:
            if log:
                log("[grader] could not close %s to the runner accounts: %s" % (d, e))
    return saved


def reopen_shared_tmp(saved):
    for d, mode in saved or ():
        try:
            os.chmod(d, mode)
        except OSError:
            pass


def apply_cell_env(sys_path):
    """Set the cell's variables in this process and put the kit and the stage on `sys_path` (call before any cell
    module is imported: the cell reads some of these at import time)."""
    os.environ.update(cell_env())
    for p in (KIT_DIR, STAGE):
        if p in sys_path:
            sys_path.remove(p)
        sys_path.insert(0, p)
