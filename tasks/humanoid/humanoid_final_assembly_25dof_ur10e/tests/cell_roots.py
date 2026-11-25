"""The data roots of the cell's code in this stage folder (stdlib only).

The cell's package (arm/assembly_humanoid/) is used as it is; its module-level defaults are no paths of this
stage. bind() points every one of them at the files of this verify stage before any episode is built: the
cell's hidden kit unpacked into gt/ (the kit index, the collision model, the mating graph, the fixtures' design and
meshes, the frozen numbers) and the given robot models of the root-owned copy of the givens. It changes no value of
the cell, and it refuses a root that is missing. check() lists what is bound."""
import os

ROOTS = {
    "kit_dir": ("kit",),
    "collision": ("collision",),
    "graph": ("graph", "mating_graph.json"),
    "fixtures": ("fixtures", "fixtures.json"),
    "fixture_meshes": ("fixtures", "meshes"),
    "frozen": ("frozen", "frozen_numbers.json"),
    "twin_map": ("render", "twin_map.json"),
}
_BOUND = {}


class RootError(RuntimeError):
    pass


def roots(gt_dir, given_dir):
    out = {k: os.path.join(gt_dir, *v) for k, v in ROOTS.items()}
    out["robot_models"] = os.path.join(given_dir, "robot")
    for k, p in sorted(out.items()):
        if not os.path.exists(p):
            raise RootError("the cell's %s is missing (%s)" % (k, os.path.relpath(p, os.path.dirname(gt_dir))))
    return out


def bind(gt_dir, given_dir):
    """point the cell's defaults at this stage's files; returns the roots"""
    for k in ("HUMANOID_COLLISION_PIECES", "HUMANOID_KIT_DIR", "HUMANOID_FIXTURE_MESHES"):
        os.environ.pop(k, None)
    r = roots(gt_dir, given_dir)
    from arm.assembly_humanoid import episode as EP
    from arm.assembly_humanoid import fixtures_h as FH
    from arm.assembly_humanoid import kit_adapter as KA
    from arm.assembly_humanoid import robot_models as RM
    KA.KIT_DIR = r["kit_dir"]
    KA.COLLISION_SOURCE_OF_RECORD = r["collision"]
    KA.PIECES_SOURCE = r["collision"]
    KA.FROZEN_PATH = r["frozen"]
    KA._PIECES.clear()
    KA._MODELS.clear()
    FH.DESIGN_PATH = r["fixtures"]
    FH.MESH_DIR = r["fixture_meshes"]
    EP.GRAPH_PATH = r["graph"]
    RM.MODELS_DIR = r["robot_models"]
    _BOUND.clear()
    _BOUND.update(r)
    return r


def check():
    """{name: path} of the bound roots (empty before bind())"""
    from arm.assembly_humanoid import episode as EP
    from arm.assembly_humanoid import fixtures_h as FH
    from arm.assembly_humanoid import kit_adapter as KA
    seen = {"kit_dir": KA.KIT_DIR, "collision": KA.PIECES_SOURCE, "frozen": KA.FROZEN_PATH, "fixtures": FH.DESIGN_PATH,
            "fixture_meshes": FH.MESH_DIR, "graph": EP.GRAPH_PATH}
    bad = [k for k, v in seen.items() if _BOUND.get(k) != v]
    if bad:
        raise RootError("not bound: %s" % ", ".join(sorted(bad)))
    return dict(_BOUND)
