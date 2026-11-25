"""The robot models the cell's model loads.

New file of this task. The UR10e's six joint caps (five cap meshes) carry the arm maker's logo as relief and the wide
gripper's base carries its maker's name as relief; in this task every copy of those meshes the cell simulates is
LEVELLED at its generator. The generator (the levelling script, a byte-for-byte --check) writes the
levelled models folder below: universal_robots_ur10e/ and wide_gripper/ as the vendored models with the six meshes
levelled (the rules of the pictures), every other file a byte copy, and LEVELLED.json (every file's bytes and sha256).

  models_dir(check=True)   the folder of record (MODELS_DIR, or $ARB_LEVELLED_ROBOT_MODELS when a task image places
                           it elsewhere: the given robot folder generated from the same files); check=True verifies
                           every file against its LEVELLED.json (sha256) and refuses a folder that is not levelled
  is_levelled(path)        True when `path` holds a LEVELLED.json whose files are all present and equal
The episode builder (episode.py) builds every episode on models_dir(); scene.build_cell itself takes any folder (the
before / after proofs and the stub tests use the unlevelled vendored models).
"""
import hashlib
import json
import os

__all__ = ["MODELS_DIR", "models_dir", "is_levelled", "levelled_report", "RobotModelsError"]

MODELS_DIR = "<bound at verify start by tests/cell_roots.py>"
MANIFEST = "LEVELLED.json"
_OK = {}


class RobotModelsError(RuntimeError):
    pass


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def levelled_report(path):
    with open(os.path.join(path, MANIFEST)) as f:
        return json.load(f)


def is_levelled(path):
    path = os.path.abspath(path)
    if path in _OK:
        return _OK[path]
    ok = False
    mp = os.path.join(path, MANIFEST)
    if os.path.isfile(mp):
        man = levelled_report(path)
        ok = bool(man.get("levelled")) and all(
            os.path.isfile(os.path.join(path, r["path"])) and _sha(os.path.join(path, r["path"])) == r["sha256"]
            for r in man["files"])
    _OK[path] = ok
    return ok


def models_dir(check=True):
    p = os.environ.get("ARB_LEVELLED_ROBOT_MODELS") or MODELS_DIR
    if check and not is_levelled(p):
        raise RobotModelsError("%s is not the levelled robot models folder of record (no %s or a file differs)" % (p, MANIFEST))
    return p
