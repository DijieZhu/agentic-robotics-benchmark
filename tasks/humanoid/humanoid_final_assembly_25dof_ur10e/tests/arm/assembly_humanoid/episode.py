"""The humanoid assembly cell's EPISODE BUILDER.

New file of this task. It takes the place of the precedent servers' `reset(seed)` (arm/assembly_car/server.py
`CarInfoServer.reset`, arm/assembly_rov/server.py `RovInfoServer.reset`: layout draw -> the cell -> the routed tick)
and composes the cell's parts, each through its OWN defaults and nothing else:

  seed (an integer; at verify one per episode from OS entropy: os_seed())
    -> the layout's draws: the fixtures' draw  fixtures_h.draw_layout(numpy default_rng(seed), cfg)  (the stream the
       fixture, reach and camera proofs used for their seeds 0..299) and the camera mount's draw
       arm.assembly_info.layout_info.draw_cameras(seed, <the cameras block of record>)  (the precedents' camera draw
       and the camera coverage measurement's draws: the same function, the same configuration, so seed s gives its camera s);
       nominal=True: the nominal layout and the nominal cameras (the privileged reference's layout)
    -> layout["top_camera_T"] = (world_T_cam drawn, world_T_cam nominal) (OpenCV convention for both): the overhead
       camera's housing follows its PHYSICAL mount draw and its hanger is rebuilt up to the beam
       (fixtures_h)
    -> the cell's data through the kit adapter's own defaults: kit_adapter.cell_data(...) with no kit folder and no
       collision source (KIT_DIR = the kit of record; PIECES_SOURCE = COLLISION_SOURCE_OF_RECORD, the collision model of
       record; self-collision on with the model's excluded pairs); a HUMANOID_* variable that
       would select another kit or model is REFUSED (EpisodeError), not followed
    -> the fixtures of the drawn layout (fixtures_h.fixture_records, the design of record pinned by the configuration's
       fixtures.design_sha256), the base on its stand, every module resting in its fixture at its delivered state
    -> one inactive weld per mate (coupling_h.weld_records of the hidden graph's interfaces)
    -> the overhead camera as a world camera at its drawn mount (the wrist camera comes with the robot; its vertical
       field is set before the compile and asserted after by scene.build_cell, cameras.mounts.wrist.fovy_deg)
    -> scene.build_cell (the UR10e and the wide gripper from the LEVELLED robot models, robot_models.py; the safety
       monitor; the settle) -> the coupling's hooks (coupling_h.Coupling(...).attach(): its tick after the monitor,
       the monitor's locks() / on_scrap())
    -> server.HumanoidCellServer(cell, cfg, seed, privileged, renderer, episode=...) (every tick routed through it).

Privileged mode = the oracle token (os the precedents: verify.py token_state: the program's .oracle_token whose sha256
equals hidden/oracle_token.sha256); `token_state(token_path, want_path)` here is that rule; without the token every
privileged call answers {"error": "forbidden"} and is counted.

API: os_seed(); camera_layout_cfg(cfg); draw_cameras(seed, cfg, nominal=False); draw_episode(seed, cfg, des,
nominal=False); token_state(token_path, want_path); load_inputs(cfg=None); episode_data(draw, inputs);
build_episode(seed=None, nominal=False, cfg=None, privileged=False, renderer=None, robot_models_dir=None, log=None,
settle=True, inputs=None) -> Episode; Episode.server / .cell / .coupling / .draw / .report.
"""
import copy
import hashlib
import json
import math
import os
import resource
import secrets
import time

import numpy as np

from . import config as CF
from . import coupling_h as CH
from . import fixtures_h as FH
from . import kit_adapter as KA
from . import scene as SC

__all__ = ["EpisodeError", "GRAPH_PATH", "os_seed", "camera_layout_cfg", "draw_cameras", "draw_episode", "token_state",
           "load_inputs", "episode_data", "build_episode", "Episode", "top_camera_record", "ENV_REFUSED"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
GRAPH_PATH = os.path.join(DEV, "graph", "mating_graph.json")
# the environment variables that would make the adapter or the fixtures read another kit, collision model or mesh folder
# than the ones of record: the episode builder refuses to run under them (the task's rule: no wrapper, no variable)
ENV_REFUSED = ("HUMANOID_COLLISION_PIECES", "HUMANOID_KIT_DIR", "HUMANOID_FIXTURE_MESHES")
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])        # OpenCV camera axes (+z forward, +y down) -> MuJoCo's (-z forward, +y up)


class EpisodeError(RuntimeError):
    pass


def os_seed():
    """an episode seed from OS entropy (63 bits: fresh layouts per verify)"""
    return int(secrets.randbits(63))


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------- cameras
def camera_layout_cfg(cfg):
    """the arm.assembly_info.layout_info camera configuration of this cell (the cameras block of record):
    the overhead camera's nominal mount (eye, target, up; its hfov from fovy and size), the wrist camera's mount on the
    hand (the robot's, gripper_wide DEFAULT_CFG wrist_camera), the camera draw"""
    from ..assembly_info import layout_info as LI
    from ..assembly_rov import gripper_wide as GW
    C = cfg["cameras"]
    top = C["mounts"]["top"]
    w, h = [int(v) for v in C["size"]["top"]]
    hfov_top = 2.0 * math.degrees(math.atan(math.tan(math.radians(float(top["fovy_deg"])) / 2.0) * w / h))
    wc = GW.DEFAULT_CFG["wrist_camera"]
    lens = np.asarray(wc["lens_in_flange_m"], float)
    aim = np.asarray(wc["aim_in_flange_m"], float)
    tilt = math.degrees(math.atan2(abs(lens[1] - aim[1]), aim[2] - lens[2]))
    out = {"cameras": {"top": {"size": [w, h], "hfov_deg": hfov_top, "depth": bool(C["depth"]["top"]),
                               "eye": [float(v) for v in top["pos"]], "target": [float(v) for v in top["lookat"]],
                               "up": [float(v) for v in top["up"]]},
                       "wrist": {"size": [int(v) for v in C["size"]["wrist"]], "hfov_deg": float(C["mounts"]["wrist"]["hfov_deg"]),
                                 "depth": bool(C["depth"]["wrist"]), "offset_m": float(abs(lens[1])), "z_m": float(lens[2]),
                                 "tilt_deg": float(tilt)}},
           "camera_draw": copy.deepcopy(C["camera_draw"])}
    for n in LI.CAMERA_NAMES:
        out["cameras"].setdefault(n, {"size": [640, 480], "hfov_deg": 80.0})   # the classes layout_info knows (no mount)
    return out


def draw_cameras(seed, cfg, nominal=False):
    """{"top": {"intrinsics", "world_T_cam", "nominal_world_T_cam"} (OpenCV), "wrist": {"intrinsics", "hand_T_cam"}}:
    the precedents' draw (layout_info.draw_cameras) for this seed, or the nominal cameras"""
    from ..assembly_info import layout_info as LI
    cc = camera_layout_cfg(cfg)
    t = cc["cameras"]["top"]
    Tn = LI.look_at_T(t["eye"], t["target"], t["up"])
    if nominal:
        w, h = t["size"]
        f0 = 0.5 * w / math.tan(math.radians(t["hfov_deg"]) / 2.0)
        wr = cc["cameras"]["wrist"]
        ww, wh = wr["size"]
        fw = 0.5 * ww / math.tan(math.radians(wr["hfov_deg"]) / 2.0)
        zero = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
        return {"top": {"intrinsics": dict(width=w, height=h, fx=f0, fy=f0, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0, **zero),
                        "world_T_cam": Tn.tolist(), "nominal_world_T_cam": Tn.tolist(), "depth": t["depth"]},
                "wrist": {"intrinsics": dict(width=ww, height=wh, fx=fw, fy=fw, cx=(ww - 1) / 2.0, cy=(wh - 1) / 2.0, **zero),
                          "hand_T_cam": None, "depth": wr["depth"], "nominal": True}}
    d = LI.draw_cameras(int(seed), cc, strict=False)
    out = {"top": dict(d["top"]), "wrist": dict(d["wrist"])}
    out["top"]["nominal_world_T_cam"] = Tn.tolist()
    return out


def top_camera_record(cam, name="top"):
    """the data contract's CAMERA record (MuJoCo convention: -z forward, +y up) of a drawn OpenCV world_T_cam; the model
    camera's fovy from fy (the drawn lens' other terms are the render layer's)"""
    T = np.asarray(cam["world_T_cam"], float)
    R = T[:3, :3] @ CV_TO_MJ
    K = cam["intrinsics"]
    fovy = 2.0 * math.degrees(math.atan(0.5 * float(K["height"]) / float(K["fy"])))
    return {"name": name, "pos": [float(v) for v in T[:3, 3]], "quat": [float(v) for v in SC.quat_wxyz(R)], "fovy_deg": fovy}


# ------------------------------------------------------------------------------------------------------- draws
def draw_episode(seed, cfg, des, nominal=False):
    """the episode's draws: {"seed", "nominal", "layout" (fixtures_h's draw + top_camera_T), "cameras"}"""
    if nominal:
        lay = FH.nominal_layout()
    else:
        lay = FH.draw_layout(np.random.default_rng(int(seed)), cfg)
    cams = draw_cameras(seed, cfg, nominal=nominal)
    lay = dict(lay)
    lay["top_camera_T"] = (cams["top"]["world_T_cam"], cams["top"]["nominal_world_T_cam"])
    return {"seed": None if seed is None else int(seed), "nominal": bool(nominal), "layout": lay, "cameras": cams}


def layout_digest(draw):
    """sha256 of the episode's draws (the grader can compare it; never shown to the agent)"""
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return round(float(x), 12)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x
    return hashlib.sha256(json.dumps(clean({"layout": draw["layout"], "cameras": draw["cameras"]}), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------- the token
def token_state(token_path, want_path):
    """(privileged, mismatch): the program's oracle token file against the hidden sha256 file (the precedents'
    verify.token_state rule: a missing file or a symbolic link is no token; a wrong token is ignored and counted)"""
    if not token_path or not want_path or not os.path.isfile(token_path) or os.path.islink(token_path) or not os.path.isfile(want_path):
        return False, False
    try:
        tok = open(token_path, "rb").read().strip()
        want = open(want_path).read().strip()
    except OSError:
        return False, False
    got = hashlib.sha256(tok).hexdigest()
    return got == want, got != want


# ------------------------------------------------------------------------------------------------------- inputs
def _refuse_env():
    bad = [k for k in ENV_REFUSED if k in os.environ]           # set at all (an empty value would make the adapter read '')
    if bad:
        raise EpisodeError("the episode builder runs on the kit, collision model and fixture meshes of record only; "
                           "unset %s" % ", ".join(bad))


def load_inputs(cfg=None):
    """(cfg, fixture design, kit, frozen numbers, graph, interfaces) of record: the configuration (cell_config.json),
    the fixture design it pins, the kit and the frozen numbers through the adapter's defaults, the hidden graph"""
    _refuse_env()
    cfg = CF.load_config() if cfg is None else CF.check_config(copy.deepcopy(cfg))
    if KA.PIECES_SOURCE != KA.COLLISION_SOURCE_OF_RECORD:
        raise EpisodeError("kit_adapter.PIECES_SOURCE is not COLLISION_SOURCE_OF_RECORD")
    des = FH.load_design(cfg=cfg)
    kit = KA.load_kit()
    fr = KA.load_frozen()
    with open(GRAPH_PATH) as f:
        graph = json.load(f)
    ifs = CH.interfaces_from_graph(graph, cfg=cfg)
    return {"cfg": cfg, "des": des, "kit": kit, "frozen": fr, "graph": graph, "interfaces": ifs}


def episode_data(draw, inputs, visual=False):
    """(data, arrays) of the drawn episode (module doc); visual=True adds the kit's drawn meshes (not needed by the
    physics; the given render layer draws its own instanced version)"""
    cfg, des, kit, fr, ifs = inputs["cfg"], inputs["des"], inputs["kit"], inputs["frozen"], inputs["interfaces"]
    lay = draw["layout"]
    pl = FH.module_rest_placements(des, kit, fr, lay)
    pl["base"] = FH.base_placement(des, lay)
    fx = FH.fixture_records(des, lay, visual=visual)
    cams = [top_camera_record(draw["cameras"]["top"])]
    data, arrays = KA.cell_data(cfg=cfg, frozen=fr, placements=pl, fixtures=fx, welds=CH.weld_records(ifs), cameras=cams,
                                visual=visual, rest_bodies=FH.REST_BODIES)
    data["provenance"]["episode"] = {"seed": draw["seed"], "nominal": draw["nominal"], "builder": "episode.py"}
    return data, arrays


# ------------------------------------------------------------------------------------------------------- the episode
class Episode(object):
    """one built episode: .draw, .cell, .coupling, .server, .inputs, .report (counts, times, memory)"""

    def __init__(self, draw, cell, coupling, server, inputs, report):
        self.draw, self.cell, self.coupling, self.server, self.inputs, self.report = draw, cell, coupling, server, inputs, report

    def privileged_record(self):
        """what only the oracle token may see: the draws and their digest"""
        return {"seed": self.draw["seed"], "nominal": self.draw["nominal"], "layout_digest": self.report["layout_digest"],
                "layout": _jsonable(self.draw["layout"]), "cameras": _jsonable(self.draw["cameras"])}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _rss_kb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1])
    except OSError:
        pass
    return None


def build_episode(seed=None, nominal=False, cfg=None, privileged=False, renderer=None, robot_models_dir=None, log=None,
                  settle=True, inputs=None, token_mismatch=False):
    """build one episode (module doc). seed None: os_seed(). robot_models_dir None: the levelled robot models of record
    (robot_models.models_dir(): the maker's relief levelled); any other folder must be given here
    explicitly. Returns an Episode whose server routes every tick."""
    from . import robot_models as RM
    from .server import HumanoidCellServer
    t0 = time.time()
    rss0 = _rss_kb()
    if seed is None:
        seed = os_seed()
    inputs = inputs or load_inputs(cfg)
    cfg = inputs["cfg"]
    draw = draw_episode(seed, cfg, inputs["des"], nominal=nominal)
    t1 = time.time()
    data, arrays = episode_data(draw, inputs)
    t2 = time.time()
    mdir = robot_models_dir or RM.models_dir(check=True)
    cell = SC.build_cell(data, cfg, arrays=arrays, robot=True, robot_models_dir=mdir, monitor=True, log=log, settle=False)
    t3 = time.time()
    cell.episode_draw = draw                     # the render layer's seam reads the drawn layout and lenses (render_iface)
    cp = CH.Coupling(cell, inputs["interfaces"], cfg).attach()
    if settle:
        cell.settle(float(cfg["physics"]["settle_s"]))
    # the cell's pace levers from the configuration, after the settle (the settle runs as of record):
    # the first physics step of a tick reuses the tick's mj_forward when nothing changed (performance.skip_first_step1)
    # and the contact clusters (contact_clusters; clusters_h.py), whose tick-end hook runs after the coupling's
    from . import clusters_h as CLU
    cell.skip_first_step1 = bool((cfg.get("performance") or {}).get("skip_first_step1", False))
    if CLU.clusters_config(cfg)["enabled"]:
        CLU.ContactClusters(cell, cfg, log=log).attach()
    t4 = time.time()
    report = {"seed": int(seed), "nominal": bool(nominal), "layout_digest": layout_digest(draw),
              "robot_models_dir": mdir, "levelled_robot_models": RM.is_levelled(mdir),
              "build_report": {k: v for k, v in cell.build_report.items() if k != "range_rows"},
              "times_s": {"inputs_and_draws": round(t1 - t0, 3), "data": round(t2 - t1, 3), "compile_and_checks": round(t3 - t2, 3),
                          "coupling_and_settle": round(t4 - t3, 3), "total": round(t4 - t0, 3)},
              "rss_kb": {"before": rss0, "after": _rss_kb(), "max": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
              "model_buffer_bytes": int(cell.m.nbuffer), "data_buffer_bytes": int(cell.d.nbuffer) if hasattr(cell.d, "nbuffer") else None,
              "collision_source": KA.PIECES_SOURCE, "kit_dir": KA.KIT_DIR, "fixtures_design_sha256": cfg["fixtures"]["design_sha256"],
              "pace_levers": {"skip_first_step1": bool(cell.skip_first_step1),
                              "contact_clusters": None if cell.clusters is None else
                              {"reducible": int(cell.clusters.nr), "reduced_at_start": int(cell.clusters.reduced.sum()),
                               "slots": len(cell.clusters.slot_name)}}}
    ep = Episode(draw, cell, cp, None, inputs, report)
    ep.server = HumanoidCellServer(cell, cfg=cfg, seed=int(seed), privileged=bool(privileged), renderer=renderer, log=log, episode=ep)
    ep.server.token_mismatch = bool(token_mismatch)
    return ep
