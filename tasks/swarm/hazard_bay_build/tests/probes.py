"""Measure a hazard-bay scene in Isaac Sim: what is where, what it weighs, how the valve and the hazards behave, how the
bucket hangs and swings on its ropes, how the agent's controller flies the hoist, and what the fleet is built from.

    python3 probes.py <scene> <out.json> <group> [<seed>]
    groups: canary canary_render layout settle steam hoist control fleet

The grader passes the seed in the environment (`ARB_SEED`; the command line is readable by every account), a maintainer
may pass it as the fourth argument. The scene probes run at the fixed hidden set's seeds (0, the seed truth.json was
measured with); the seed draws their random amounts (the steam test's points, the hoist's robot and directions, the
fleet's arm configurations). The control group runs one mission of the set: its day is the `missions` entry of the
probe's own config file (`ARB_GRADER_CFG`) that `ARB_MISSION` names (a maintainer may instead name a file holding one
day in `ARB_DAY_FILE`), and its seed draws the sensors' noise.

The control group's wire carries no ground truth (control.md, What you observe): each robot's own state as its
onboard systems estimate it (`wire_obs.py`: the legacy observation of `ControlSim.observe()` filtered, the vehicle's
odometry replaced, never recomputed) and a reference to the latest frame set its declared sensors delivered
(`sensor_rig.py`: the grader's cameras in the real bay, dressed by `mission_stage.dress`, rendered every 6 steps,
written to slot files in the frames folder root made for the mission, `ARB_FRAMES_DIR`). The canary_render group
renders the dressed real bay alone (no robots) with one camera of each kind, a lidar head and the frame clock: the
renderer compiles there, before any of the agent's scene is involved.

Everything is read from PhysX or from the stage as authored, never from anything the scene reports about itself except
the state attributes the interface declares (`arb:steam_active`, `arb:fire_active`, `arb:disabled`), and everything is
pushed from outside (forces and torques on bodies, robots moved kinematically). The grader runs this on the submission
as an unprivileged account after the static scan passed (no code in the scene), one fresh process per group, and
compares the result with the same measurement of the reference build (truth.json) and with the brief's numbers.
"""

import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
from isaacsim import SimulationApp

os.umask(0o077)  # (whatever this process writes, here or in /tmp, is the probe account's alone)
SCENE, OUT, GROUP = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = int(os.environ.get("ARB_SEED") or (sys.argv[4] if len(sys.argv) > 4 else 0))
DT = 1.0 / 120.0
_cfg = {"headless": True}
if os.environ.get("ARB_KIT_DIR"):  # the grader's own Kit folders (cache, data, logs), not the shared ones
    _k = os.environ["ARB_KIT_DIR"]
    _cfg["extra_args"] = [f"--/app/folder/cache={_k}/cache", f"--/app/folder/data={_k}/data", f"--/app/folder/logs={_k}/logs"]
app = SimulationApp(_cfg)
import omni.kit.app  # noqa: E402

for _ext in ("isaacsim.core.nodes", "isaacsim.sensors.physics"):
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(_ext, True)


def enable_replicator():
    """Replicator for the groups that render (canary_render, the control group with sensors), enabled before their stage
    opens (its graph node types then have the stage's open updates to register in)."""
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    for _ in range(2):
        app.update()
import omni.physics.tensors as pt  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

RESULT = {"scene": SCENE, "seed": SEED, "groups": {}}
PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
FAC, FLEET = "/World/Facility", "/World/Fleet"
UAVS = ("UAV_1", "UAV_2", "UAV_3")
ROBOTS = UAVS + ("UGV",)
RNG = random.Random(SEED)
# the brief's fixed points (the grader's own, the same numbers the agent gets): where the hazard tests happen
APPROACH = (2.60, -0.40, 2.50)  # the water drop's approach hover
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import control_sim  # noqa: E402

CFG = json.load(open(os.environ["ARB_GRADER_CFG"])) if os.environ.get("ARB_GRADER_CFG") else {}


# --------------------------------------------------------------------------------------------- math
def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot(q, v):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v, dtype=float)


def angle_deg(q0, q1):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0, q1))))))


def gfq(qd):
    return np.array([qd.GetReal(), *qd.GetImaginary()])


def r3(v):
    return [round(float(x), 4) for x in v]


def fin(x):
    return float(x) if x is not None and np.isfinite(x) else None


def tf_pq(tf):
    """PhysX transform row [x y z qx qy qz qw] -> position, quaternion (w x y z)."""
    return np.array(tf[:3], dtype=float), np.array([tf[6], tf[3], tf[4], tf[5]], dtype=float)


def pq_tf(p, q):
    return np.array([p[0], p[1], p[2], q[1], q[2], q[3], q[0]], dtype=np.float32)


# --------------------------------------------------------------------------------------------- stage
class Scene:
    def __init__(self, path=None):
        ctx = omni.usd.get_context()
        ctx.open_stage(path or SCENE)
        for _ in range(3):
            app.update()
        self.ctx, self.stage = ctx, ctx.get_stage()
        self.cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES, useExtentsHint=False)
        self.sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
        self.t = 0.0

    def start(self):
        self.sim.reset()
        self.view = pt.create_simulation_view("numpy", self.ctx.get_stage_id())

    def step(self, n=1):
        for _ in range(n):
            self.sim.step(render=False)
            self.t += DT

    def prim(self, path):
        return self.stage.GetPrimAtPath(path)

    def usd_pose(self, path):
        m = UsdGeom.Xformable(self.prim(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(m.ExtractTranslation()), gfq(m.ExtractRotationQuat())

    def world_box(self, path, any_visibility=False):
        cache = self.cache
        if any_visibility:  # (an invisible volume counts: the plume's hazard box may be hidden geometry)
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES, useExtentsHint=False, ignoreVisibility=True)
        r = cache.ComputeWorldBound(self.prim(path)).ComputeAlignedRange()
        if r.IsEmpty():
            return None, None
        return np.array(r.GetMin()), np.array(r.GetMax())

    def own_box(self, path):
        r = self.cache.ComputeUntransformedBound(self.prim(path)).ComputeAlignedRange()
        if r.IsEmpty():
            return None, None
        return np.array(r.GetMin()), np.array(r.GetMax())

    def flag(self, path, name):
        a = self.prim(path).GetAttribute(name)
        v = a.Get() if a and a.IsValid() else None
        return None if v is None else bool(v)

    def art_root(self, root):
        p = self.prim(root)
        if not p.IsValid():
            return None
        arts = [q for q in Usd.PrimRange(p) if q.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not arts:
            return None
        a = arts[0]
        if a.HasAPI(UsdPhysics.RigidBodyAPI):
            return a.GetPath().pathString
        bodies = [q for q in Usd.PrimRange(a) if q.HasAPI(UsdPhysics.RigidBodyAPI)]  # (the root on an Xform: its first body)
        return bodies[0].GetPath().pathString if bodies else a.GetPath().pathString

    def joints(self):
        return [p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Joint) and p.IsActive()]


def rigid_bodies(stage, root):
    p = stage.GetPrimAtPath(root)
    if not p.IsValid():
        return []
    return [q.GetPath().pathString for q in Usd.PrimRange(p) if q.HasAPI(UsdPhysics.RigidBodyAPI)]


def box_facts(s, path, xy_only=False, any_visibility=False):
    lo, hi = s.world_box(path, any_visibility)
    if lo is None:
        return None
    if xy_only:
        return dict(center=r3((lo + hi)[:2] / 2), extent=r3((hi - lo)[:2]), top=round(float(hi[2]), 4), bottom=round(float(lo[2]), 4))
    return dict(center=r3((lo + hi) / 2), extent=r3(hi - lo), top=round(float(hi[2]), 4), bottom=round(float(lo[2]), 4))


def box_facts_except(s, path, skip):
    """The world box of a prim's children except those named in `skip` (a robot without its rope on the floor)."""
    lo, hi = None, None
    for c in s.prim(path).GetChildren():
        if c.GetName() in skip:
            continue
        clo, chi = s.world_box(c.GetPath().pathString)
        if clo is None:
            continue
        lo = clo if lo is None else np.minimum(lo, clo)
        hi = chi if hi is None else np.maximum(hi, chi)
    if lo is None:
        return None
    return dict(center=r3((lo + hi) / 2), extent=r3(hi - lo), top=round(float(hi[2]), 4), bottom=round(float(lo[2]), 4))


def group(name):
    def deco(fn):
        def run():
            t0 = time.time()
            out = {}
            try:
                fn(out)
            except Exception:  # noqa: BLE001
                out["error"] = traceback.format_exc()[-1500:]
            out["wall_s"] = round(time.time() - t0, 1)
            RESULT["groups"][name] = out
            with open(OUT, "w") as f:
                json.dump(RESULT, f)
        run.group_name = name
        return run
    return deco


class Robot:
    """A fleet robot as PhysX sees it: its articulation view, moved from outside by resetting its root every step."""

    def __init__(self, s, name):
        self.s, self.name = s, name
        self.root = s.art_root(f"{FLEET}/{name}")
        self.view = s.view.create_articulation_view(self.root) if self.root else None
        tf = self.view.get_root_transforms()[0]
        self.p0, self.q0 = tf_pq(tf)
        self.idx = np.array([0], dtype=np.int32)

    def hold(self, p, q=None):
        q = self.q0 if q is None else q
        self.view.set_root_transforms(pq_tf(p, q)[None], self.idx)
        self.view.set_root_velocities(np.zeros((1, 6), dtype=np.float32), self.idx)

    def pos(self):
        return tf_pq(self.view.get_root_transforms()[0])[0]

    def disabled(self):
        return self.s.flag(f"{FLEET}/{self.name}", "arb:disabled")


def cameras_of(s, robot_root):
    """The robot's declared sensors: cameras with arb:sensor_model (world pose at rest, intrinsics), other sensor prims."""
    out = []
    for p in Usd.PrimRange(s.prim(robot_root)):
        a = p.GetAttribute("arb:sensor_model")
        if not a or not a.IsValid() or not a.Get():
            continue
        entry = dict(path=p.GetPath().pathString, model=str(a.Get()), camera=bool(p.IsA(UsdGeom.Camera)))
        if p.IsA(UsdGeom.Camera):
            c = UsdGeom.Camera(p)
            cp, cq = s.usd_pose(p.GetPath().pathString)
            focal, hap, vap = c.GetFocalLengthAttr().Get(), c.GetHorizontalApertureAttr().Get(), c.GetVerticalApertureAttr().Get()
            look = rot(cq, (0.0, 0.0, -1.0))  # USD cameras look along their -z
            up = rot(cq, (0.0, 1.0, 0.0))
            entry.update(pos=r3(cp), look=r3(look), up=r3(up), focal=fin(focal), hap=fin(hap), vap=fin(vap),
                         hfov_deg=round(math.degrees(2 * math.atan(hap / 2 / focal)), 2) if focal and hap and focal > 0 else None,
                         vfov_deg=round(math.degrees(2 * math.atan(vap / 2 / focal)), 2) if focal and vap and focal > 0 else None)
        out.append(entry)
    return out


def declared(s, robot_root, name):
    """The one value of arb:<name> declared somewhere under the robot (None if absent or ambiguous)."""
    vals = set()
    for p in Usd.PrimRange(s.prim(robot_root)):
        a = p.GetAttribute(f"arb:{name}")
        if a and a.IsValid() and a.Get():
            vals.add(str(a.Get()))
    return sorted(vals)[0] if len(vals) == 1 else None


# --------------------------------------------------------------------------------------------- groups
@group("canary")
def canary(out):
    """The grader's own trivial scene: physics runs, on the GPU, with the sensor extension loaded."""
    import carb
    s = Scene()
    s.start()
    v = s.view.create_rigid_body_view("/World/Cube")
    z0 = float(v.get_transforms()[0][2])
    s.step(60)
    z1 = float(v.get_transforms()[0][2])
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda = False
    out.update(z0=round(z0, 4), z1=round(z1, 4), fell=bool(z1 < z0 - 0.3), device=str(getattr(s.sim, "device", "")), cuda_available=cuda,
               cuda_device=carb.settings.get_settings().get("/physics/cudaDevice"),
               sensors_ext=bool(omni.kit.app.get_app().get_extension_manager().is_extension_enabled("isaacsim.sensors.physics")),
               kit_dir=os.environ.get("ARB_KIT_DIR"))


@group("canary_render")
def canary_render(out):
    """The renderer on the grader's own stage of the real bay (`SCENE` is its layer), dressed as for the mission, with
    no robots: one camera of each catalog kind at its catalog size from the config's fixed poses (the parking bay at the
    dial, the approach hover at the pan), a LIDAR-3D head and the frame clock. Checked: full, non-black, non-uniform
    RGB; the burning pan hot in thermal; depth and ranges finite where something is drawn; the clock decoding. Records
    the first render's time (on an empty cache: the compile) and the device memory."""
    import mission_stage
    import sensor_rig
    from pxr import Sdf
    enable_replicator()
    scfg = CFG["sensors"]
    cat = CFG["catalog"]["sensors"]
    path = os.path.splitext(OUT)[0] + "_stage.usda"
    if os.path.exists(path):
        os.remove(path)
    layer = Sdf.Layer.CreateNew(path)
    layer.subLayerPaths.append(SCENE)
    layer.defaultPrim = "World"
    layer.Save()
    t0 = time.time()
    out["dress"] = mission_stage.dress(path, {}, scfg["look"])
    s = Scene(path)
    out["look_physics"] = look_physics(s)[:8]
    s.start()
    out["open_s"] = round(time.time() - t0, 2)
    plan_, poses = {"sensors": [], "not_streamed": []}, {}
    for k, c in enumerate(scfg["canary"]["sensors"]):
        sm = sensor_rig.stream_of(c["model"], cat[c["model"]])
        q = sensor_rig.look_at(c["eye"], c["target"]) if sm["kind"] != "lidar" else np.array([1.0, 0.0, 0.0, 0.0])
        body = f"/canary/{k}"
        poses[body] = (np.array(c["eye"], dtype=float), q)
        plan_["sensors"].append(dict(index=k, id=f"s{k:02d}", stem=f"s{k:02d}", robot="canary", prim=body, stage_prim=body, model=c["model"],
                                     kind=sm["kind"], body=body, stage_body=body, pos_in_body=[0.0, 0.0, 0.0], quat_in_body=[1.0, 0.0, 0.0, 0.0],
                                     stream=sm))
    fdir = os.path.splitext(OUT)[0] + "_frames"
    os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    needle = needle_of(s, out["dress"]["needle"])
    rig = sensor_rig.SensorRig(s.sim, s.stage, plan_, scfg, fdir, SEED, bodies=lambda b: poses[b], needle=needle,
                               pressure=lambda: 8.0, log=lambda *a: print("[canary_render]", *a, flush=True))
    rig.open_slots(sensor_rig.manifest(plan_, scfg))
    t0 = time.time()
    warm = rig.warm_up()
    out["warm_up"] = rig.warm
    out["first_render_s"] = round(rig.warm["first_render_ms"] / 1000.0, 1)
    for k in range(0, 2 * rig.every + 1):  # (two captures: the second delivers the first)
        rig.step(k, k * 2 * DT, True)
    out["frames"] = rig.summary()
    checks = {}
    for i, sd in enumerate(plan_["sensors"]):
        fr = rig.slots[i][0].payload
        kind = sd["kind"]
        if kind == "rgb":
            v = np.asarray(fr, dtype=np.float32)
            checks[sd["id"]] = dict(kind=kind, ok=bool(v.max() > 0 and v.std() > 2.0), max=float(v.max()), std=round(float(v.std()), 2))
        elif kind == "thermal":
            hot = int(np.sum(fr > 700.0))
            checks[sd["id"]] = dict(kind=kind, ok=hot > 0, hot_px=hot, max_k=round(float(fr.max()), 2))
        else:
            good = float(np.mean(fr > 0))
            checks[sd["id"]] = dict(kind=kind, ok=bool(good > 0.1 and np.all(np.isfinite(fr))), valid_share=round(good, 3))
    out["checks"] = checks
    out["clock_ok"] = bool(warm and rig.stats["lag_anomalies"] == 0)
    out["ok"] = bool(out["clock_ok"] and all(c["ok"] for c in checks.values()) and not out["look_physics"])
    out["gpu"] = gpu_memory()
    rig.close()


def look_physics(s):
    """The prims under the look and the grader's sensors that carry a physics schema in the open stage (none: the dress
    removed the hall's; a stage that still has one is not opened for the mission)."""
    import mission_stage
    bad = []
    for top in (mission_stage.LOOK, mission_stage.SENSORS):
        tp = s.prim(top)
        for p in (Usd.PrimRange(tp, Usd.TraverseInstanceProxies()) if tp.IsValid() else []):
            if mission_stage._physics_names(p) or mission_stage._physics_typed(p):
                bad.append(p.GetPath().pathString)
    return bad


def gpu_memory():
    """The device's free and total memory (torch), and the renderer's own account of its use."""
    rec = {}
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        rec.update(free_gb=round(free / 2 ** 30, 2), total_gb=round(total / 2 ** 30, 2))
    except Exception as exc:  # noqa: BLE001
        rec["torch_error"] = str(exc)[:200]
    try:
        import omni.hydra.engine.stats as es
        rec["renderer"] = [{k: v for k, v in d.items() if isinstance(v, (int, float, str))} for d in es.get_device_info()]
    except Exception as exc:  # noqa: BLE001
        rec["renderer_error"] = str(exc)[:200]
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    rec[line.split(":")[0].lower() + "_mb"] = round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return rec


def needle_of(s, nd):
    """The gauge needle's head op (mission_stage.dress) in the open stage, with its parent's frame."""
    prim = s.prim(nd["path"])
    op = next((o for o in UsdGeom.Xformable(prim).GetOrderedXformOps() if o.GetName() == nd["op"]), None)
    if op is None:
        return None
    parent = UsdGeom.XformCache().GetParentToWorldTransform(prim)
    return dict(op=op, rest_deg=float(nd["rest_deg"]), axis_point=nd["axis_point"], parent=parent, parent_inv=parent.GetInverse())


@group("layout")
def layout(out):
    """Static things, as authored: the bay's furniture and fixtures, the fleet at rest, the sensors."""
    s = Scene()
    for name in ("Table_A", "Table_B"):
        out[name] = box_facts(s, f"{FAC}/Tables/{name}", xy_only=True)
    out["crate"] = box_facts(s, f"{FAC}/Crate")
    out["pads"] = {n: box_facts(s, f"{FAC}/Pads/{n}", xy_only=True) for n in ("Pad_1", "Pad_2", "Pad_3", "UGV_Bay", "Valve_Square")}
    out["valve_handle"] = box_facts(s, f"{FAC}/SteamLine/Handle")
    out["plume"] = box_facts(s, f"{FAC}/SteamLine/Plume", any_visibility=True)
    dial = box_facts(s, f"{FAC}/SteamLine/Gauge/Dial")
    if dial:
        ext = sorted(dial["extent"])
        dial["diameter"] = round(ext[-1], 4)
    out["gauge_dial"] = dial
    out["fire_pan"] = box_facts(s, f"{FAC}/FirePan")
    out["tank"] = box_facts(s, f"{FAC}/Tank")
    out["bucket"] = box_facts(s, f"{FAC}/Bucket/Body")
    out["robots"] = {}
    for name in ROBOTS:
        root = f"{FLEET}/{name}"
        if not s.prim(root).IsValid():
            out["robots"][name] = None
            continue
        out["robots"][name] = dict(box=box_facts_except(s, root, ("Rope",)), art_root=s.art_root(root), sensors=cameras_of(s, root),
                                   airframe=declared(s, root, "airframe_model"), arm=declared(s, root, "arm_model"),
                                   base=declared(s, root, "base_model"))
    out["flags"] = dict(steam=s.flag(f"{FAC}/SteamLine", "arb:steam_active"), fire=s.flag(f"{FAC}/FirePan", "arb:fire_active"),
                        disabled={n: s.flag(f"{FLEET}/{n}", "arb:disabled") for n in ROBOTS})


@group("settle")
def settle(out):
    """Start the simulation, let it run 5 s: does everything stay put (from its first step on); where do the movable
    things rest; masses; the robots' structure."""
    s = Scene()
    bodies = []
    for root in (f"{FAC}/Items", f"{FAC}/Bucket", f"{FAC}/SteamLine", FLEET):
        bodies += rigid_bodies(s.stage, root)
    s.start()
    views = {}
    for b in bodies:
        try:
            views[b] = s.view.create_rigid_body_view(b)
        except Exception:  # noqa: BLE001
            pass
    s.step(1)  # (joint states apply on the first step: what counts is what moves after it)
    first = {b: v.get_transforms()[0][:3].copy() for b, v in views.items()}
    s.step(int(5.0 / DT))
    moved = {}
    for b, v in views.items():
        tf = v.get_transforms()[0]
        moved[b] = float(np.linalg.norm(tf[:3] - first[b])) if np.all(np.isfinite(tf)) else 99.0
    out["moved"] = {k: round(v, 4) for k, v in moved.items() if "/Rope/" not in k}
    out["rope_moved"] = {k: round(v, 4) for k, v in moved.items() if "/Rope/" in k}
    items = {}
    ip = s.prim(f"{FAC}/Items")
    for prim in (ip.GetChildren() if ip.IsValid() else []):
        path = prim.GetPath().pathString
        if path not in views:
            continue
        p, q = tf_pq(views[path].get_transforms()[0])
        lo, hi = s.own_box(path)
        c_own = (lo + hi) / 2 if lo is not None else np.zeros(3)
        size = sorted((hi - lo).tolist()) if lo is not None else None
        items[prim.GetName()] = dict(center=r3(p + rot(q, c_own)), size=r3(size) if size else None,
                                     mass=fin(views[path].get_masses()[0][0]), moved=round(moved[path], 4))
    out["items"] = items
    bk = f"{FAC}/Bucket/Body"
    out["bucket_mass"] = fin(views[bk].get_masses()[0][0]) if bk in views else None
    robots = {}
    for name in ROBOTS:
        root = s.art_root(f"{FLEET}/{name}")
        if not root:
            continue
        try:
            av = s.view.create_articulation_view(root)
        except Exception as exc:  # noqa: BLE001
            robots[name] = dict(error=str(exc)[:200])
            continue
        meta = av.shared_metatype
        p, q = tf_pq(av.get_root_transforms()[0])
        lims = av.get_dof_limits()[0]
        robots[name] = dict(root=root, mass=fin(np.sum(av.get_masses()[0])), links=int(av.max_links), dofs=int(av.max_dofs),
                            dof_types=[int(t) for t in meta.dof_types], dof_names=list(meta.dof_names),
                            link_names=list(meta.link_names), fixed_base=bool(meta.fixed_base), position=r3(p), quat=r3(q),
                            n_limited=int(sum(1 for lo, hi in lims if np.isfinite(lo) and np.isfinite(hi) and hi - lo < 1000)),
                            link_masses=[fin(m) for m in av.get_masses()[0]])
    out["robots"] = robots
    out["flags"] = dict(steam=s.flag(f"{FAC}/SteamLine", "arb:steam_active"), fire=s.flag(f"{FAC}/FirePan", "arb:fire_active"),
                        disabled={n: s.flag(f"{FLEET}/{n}", "arb:disabled") for n in ROBOTS})


# --------------------------------------------------------------------------------------------- the valve, the steam
class ValveRig:
    """The valve's handle (a lever or a wheel on one revolute joint), turned from outside by a torque about its joint
    axis; its angle counted without wrapping. interface.md leaves to the scene how the valve is put together: the
    handle is either a link of an articulation below the steam line (its angle: the joint's position) or a plain rigid
    body on its joint (its angle: the body's own rotation about the joint's axis since the start)."""

    def __init__(self, s):
        self.s = s
        handle = f"{FAC}/SteamLine/Handle"
        self.root = s.art_root(f"{FAC}/SteamLine")
        self.art, self.body = None, None
        self.axis = np.array([0.0, 1.0, 0.0])
        self.dof = None
        if self.root:
            self.art = s.view.create_articulation_view(self.root)
            names = list(self.art.shared_metatype.link_names)
            paths = [p for p in self.art.link_paths[0]] if hasattr(self.art, "link_paths") else None
            self.i_handle = paths.index(handle) if paths else names.index("Handle")
            self.nl = self.art.max_links
            dof_names = list(self.art.shared_metatype.dof_names)
        else:
            self.body = s.view.create_rigid_body_view(handle)
            dof_names = []
        for j in s.joints():
            jj = UsdPhysics.Joint(j)
            b1 = [t.pathString for t in jj.GetBody1Rel().GetTargets()]
            b0 = [t.pathString for t in jj.GetBody0Rel().GetTargets()]
            if j.IsA(UsdPhysics.RevoluteJoint) and (handle in b1 or handle in b0):
                axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[j.GetAttribute("physics:axis").Get() or "X"]
                side = "0" if handle in b1 else "1"
                lrot = j.GetAttribute(f"physics:localRot{side}").Get()
                lrot = gfq(lrot) if lrot is not None else np.array([1.0, 0, 0, 0])
                other = (b0 if handle in b1 else b1)
                oq = s.usd_pose(other[0])[1] if other else np.array([1.0, 0, 0, 0])
                self.axis = rot(oq, rot(lrot, axis))
                if j.GetName() in dof_names:
                    self.dof = dof_names.index(j.GetName())
                break
        if self.body is not None:
            self.q0 = tf_pq(self.body.get_transforms()[0])[1]
        self.last = self._position() if self.body is not None or self.dof is not None else 0.0
        self.acc = 0.0

    def _position(self):
        """The joint's angle now (radians, modulo 2 pi): the articulation's joint position, or the handle body's rotation
        since the start about the joint's axis (the twist of that rotation about the axis)."""
        if self.body is None:
            return float(self.art.get_dof_positions()[0][self.dof])
        r = qmul(tf_pq(self.body.get_transforms()[0])[1], qinv(self.q0))
        return 2.0 * math.atan2(float(np.dot(r[1:], self.axis)), float(r[0]))

    def angle(self):
        """Degrees turned since the start (unwrapped: a joint without limits reports its angle modulo 2 pi)."""
        if self.body is None and self.dof is None:
            return 0.0
        q = self._position()
        d = (q - self.last + math.pi) % (2 * math.pi) - math.pi
        self.acc += d
        self.last = q
        return math.degrees(self.acc)

    def torque(self, nm):
        if self.body is not None:
            pos = np.array(self.body.get_transforms()[0][:3], dtype=np.float32).reshape(1, 3)
            self.body.apply_forces_and_torques_at_position(np.zeros((1, 3), dtype=np.float32),
                                                           (nm * self.axis).astype(np.float32).reshape(1, 3), pos,
                                                           np.array([0], dtype=np.int32), True)
            return
        f = np.zeros((1, self.nl, 3), dtype=np.float32)
        tq = np.zeros((1, self.nl, 3), dtype=np.float32)
        pos = np.zeros((1, self.nl, 3), dtype=np.float32)
        tq[0, self.i_handle] = nm * self.axis
        pos[0, self.i_handle] = self.art.get_link_transforms()[0, self.i_handle][:3]
        self.art.apply_forces_and_torques_at_position(f, tq, pos, np.array([0], dtype=np.int32), True)


@group("steam")
def steam(out):
    """The plume disables a flying robot inside it; a quarter turn of the valve's handle stops the steam a few seconds later; the
    plume is then harmless; opening the valve brings the steam back. The burning pan disables the ground vehicle."""
    s = Scene()
    plume_lo, plume_hi = s.world_box(f"{FAC}/SteamLine/Plume", any_visibility=True)
    pan = box_facts(s, f"{FAC}/FirePan")
    # the robots are moved one at a time here: the ropes are taken off for this test (a robot carried into the
    # plume must not drag the bucket and the others along)
    unrig(s)
    s.start()
    uav = {n: Robot(s, n) for n in UAVS}
    ugv = Robot(s, "UGV")
    valve = ValveRig(s)
    s.step(int(1.0 / DT))
    out["initial"] = dict(steam=s.flag(f"{FAC}/SteamLine", "arb:steam_active"), fire=s.flag(f"{FAC}/FirePan", "arb:fire_active"),
                          disabled={n: uav[n].disabled() for n in UAVS}, ugv_disabled=ugv.disabled())
    # UAV_1 into the plume (a random point well inside), UAV_2 just outside it
    f = np.array([RNG.uniform(0.2, 0.8) for _ in range(3)])
    p_in = plume_lo + f * (plume_hi - plume_lo)
    side = RNG.choice([0, 1])
    p_out = p_in.copy()
    p_out[side] = (plume_lo[side] - 0.5) if RNG.random() < 0.5 else (plume_hi[side] + 0.5)
    out["p_in"], out["p_out"] = r3(p_in), r3(p_out)
    dis1 = []
    for i in range(int(1.5 / DT)):
        uav["UAV_1"].hold(p_in)
        uav["UAV_2"].hold(p_out)
        s.step()
        dis1.append(bool(uav["UAV_1"].disabled()))
    out["uav1_disabled_1s"] = dis1[int(1.0 / DT)]
    out["uav1_disabled_1p5s"] = dis1[-1]
    out["uav2_disabled_outside"] = bool(uav["UAV_2"].disabled())
    out["uav3_disabled_untouched"] = bool(uav["UAV_3"].disabled())
    out["uav1_pos"] = r3(uav["UAV_1"].pos())
    # the ground vehicle near the burning pan: disabled
    p_ugv = np.array([pan["center"][0] + 1.4, pan["center"][1], ugv.p0[2]])
    for i in range(int(2.5 / DT)):
        uav["UAV_1"].hold(p_in)
        uav["UAV_2"].hold(p_out)
        ugv.hold(p_ugv)
        s.step()
        if i == int(2.0 / DT):
            out["ugv_disabled_2s_near_fire"] = bool(ugv.disabled())
    out["ugv_disabled_near_fire"] = bool(ugv.disabled())
    # close the valve: a steady torque on the handle, a quarter turn to its stop (whichever way it goes); when does
    # the steam stop?
    sign, t_closed, t_off, tr = 1.0, None, None, []
    for i in range(int(30.0 / DT)):
        a = valve.angle()
        tr.append(a)
        if i == int(2.0 / DT) and abs(a) < 10.0:
            sign = -1.0  # (it does not go this way: the other)
        if abs(a) < 150.0:
            valve.torque(sign * 6.0)
        uav["UAV_1"].hold(p_in)
        uav["UAV_2"].hold(p_out)
        s.step()
        if t_closed is None and abs(a) >= 85.0:
            t_closed = i * DT
        if t_off is None and s.flag(f"{FAC}/SteamLine", "arb:steam_active") is False:
            t_off = i * DT
        if t_closed is not None and i * DT > t_closed + 8.0:
            break
    out["handle_turned_deg"] = round(abs(tr[-1]), 2)
    out["handle_direction"] = sign
    out["t_closed"] = round(t_closed, 2) if t_closed is not None else None
    out["t_steam_off"] = round(t_off, 2) if t_off is not None else None
    out["steam_off_delay_s"] = round(t_off - t_closed, 2) if t_closed is not None and t_off is not None else None
    out["steam_after_close"] = s.flag(f"{FAC}/SteamLine", "arb:steam_active")
    # the plume is harmless now: UAV_3 into it
    for i in range(int(2.0 / DT)):
        uav["UAV_3"].hold(p_in + np.array([0.0, 0.0, 0.05]))
        uav["UAV_1"].hold(p_in + np.array([0.3, 0.0, 0.0]))
        s.step()
    out["uav3_disabled_after_close"] = bool(uav["UAV_3"].disabled())
    out["uav1_still_disabled"] = bool(uav["UAV_1"].disabled())
    # open it again: the steam returns
    t_on = None
    for i in range(int(20.0 / DT)):
        a = valve.angle()
        if abs(a) > 5.0:
            valve.torque(-sign * 6.0)
        uav["UAV_3"].hold(p_in + np.array([0.0, 0.0, 0.05]))
        s.step()
        if abs(a) <= 5.0 and s.flag(f"{FAC}/SteamLine", "arb:steam_active") is True:
            t_on = i * DT
            break
    out["handle_reopened_deg"] = round(abs(valve.angle()), 2)
    out["steam_back_on"] = s.flag(f"{FAC}/SteamLine", "arb:steam_active")
    s.step(int(1.5 / DT))
    out["uav3_disabled_after_reopen"] = bool(uav["UAV_3"].disabled())


# --------------------------------------------------------------------------------------------- the ropes, the hoist
BUCKET = f"{FAC}/Bucket/Body"


def joint_ends(j):
    jj = UsdPhysics.Joint(j)
    return [t.pathString for t in jj.GetBody0Rel().GetTargets()], [t.pathString for t in jj.GetBody1Rel().GetTargets()]


def unrig(s):
    """Take the ropes off: the chains and every joint on the bucket (the robots then move independently)."""
    for n in UAVS:
        p = s.prim(f"{FLEET}/{n}/Rope")
        if p.IsValid():
            p.SetActive(False)
    for j in s.joints():
        b0, b1 = joint_ends(j)
        if BUCKET in b0 + b1:
            j.SetActive(False)


class Rope:
    """One robot's rope as authored: its links (rigid bodies under <robot>/Rope), the hook joint on a body of the
    robot, the ring joint on the bucket; after start(): its link view."""

    def __init__(self, s, name):
        self.s, self.name = s, name
        self.root = f"{FLEET}/{name}"
        self.links = rigid_bodies(s.stage, f"{self.root}/Rope")
        linkset = set(self.links)
        self.hook_body, self.hook_local, self.ring_local, self.n_joints = None, None, None, 0
        for j in s.joints():
            b0, b1 = joint_ends(j)
            ends = b0 + b1
            if not any(e in linkset for e in ends):
                continue
            self.n_joints += 1
            if BUCKET in ends:
                side = "0" if BUCKET in b0 else "1"
                self.ring_local = np.array(j.GetAttribute(f"physics:localPos{side}").Get() or (0, 0, 0), dtype=float)
            others = [e for e in ends if e not in linkset and e != BUCKET and e.startswith(self.root + "/")]
            if others:
                side = "0" if others[0] in b0 else "1"
                self.hook_body = others[0]
                self.hook_local = np.array(j.GetAttribute(f"physics:localPos{side}").Get() or (0, 0, 0), dtype=float)
        self.view, self.mass, self.robot, self.hook_idx = None, None, None, None

    def facts(self):
        return dict(links=len(self.links), joints=self.n_joints, hook_body=self.hook_body,
                    hook_local=r3(self.hook_local) if self.hook_local is not None else None,
                    ring_local=r3(self.ring_local) if self.ring_local is not None else None)

    def start(self, robot):
        self.view = self.s.view.create_rigid_body_view(self.links) if self.links else None
        self.mass = fin(np.sum(self.view.get_masses())) if self.view is not None else None
        self.robot = robot
        names = list(robot.view.shared_metatype.link_names)
        leaf = self.hook_body.split("/")[-1] if self.hook_body else None
        self.hook_idx = names.index(leaf) if leaf in names else None

    def hook(self):
        tf = self.robot.view.get_link_transforms()[0][self.hook_idx] if self.hook_idx is not None else self.robot.view.get_root_transforms()[0]
        p, q = tf_pq(tf)
        return p + rot(q, self.hook_local if self.hook_local is not None else np.zeros(3))

    def ring(self, bv):
        p, q = tf_pq(bv.get_transforms()[0])
        return p + rot(q, self.ring_local if self.ring_local is not None else np.zeros(3))

    def shape(self, bv):
        """The hook-ring chord, the links' largest distance from it (sag, or straightness), the lowest link."""
        h, r = self.hook(), self.ring(bv)
        L = np.array(self.view.get_transforms()[:, :3], dtype=float) if self.view is not None else np.zeros((0, 3))
        chord = r - h
        d = float(np.linalg.norm(chord))
        u = chord / (d + 1e-9)
        rel = L - h
        along = rel @ u
        perp = rel - np.outer(along, u)
        dev = float(np.max(np.linalg.norm(perp, axis=1))) if len(L) else None
        return dict(chord=round(d, 4), dev=round(dev, 4) if dev is not None else None, zmin=round(float(L[:, 2].min()), 4) if len(L) else None)


@group("hoist")
def hoist(out):
    """The bucket on its three ropes, the robots moved from outside (held): slack ropes sag, taut ropes are straight
    and as long as the rope, the bucket comes up level and hangs where the geometry puts it, swings like a pendulum,
    tips when one rope is lowered, puts the fire out when it pours over the pan, survives a dead stop, comes back
    down upright."""
    s = Scene()
    pan = box_facts(s, f"{FAC}/FirePan")
    ropes = {n: Rope(s, n) for n in UAVS}
    out["ropes"] = {n: r.facts() for n, r in ropes.items()}
    rings = [r.ring_local for r in ropes.values() if r.ring_local is not None]
    if len(rings) >= 2:
        angs = sorted(math.degrees(math.atan2(b[1], b[0])) for b in rings)
        gaps = [(angs[(k + 1) % len(angs)] - angs[k]) % 360 for k in range(len(angs))]
        out["ring_min_gap_deg"] = round(min(gaps), 1)
    if any(not r.links or r.hook_local is None or r.ring_local is None for r in ropes.values()):
        out["error"] = "a rope is not a chain of bodies hooked to its robot and joined to the bucket"
        return
    b_lo, b_hi = s.own_box(BUCKET)
    s.start()
    uav = {n: Robot(s, n) for n in UAVS}
    ugv = Robot(s, "UGV")
    bv = s.view.create_rigid_body_view(BUCKET)
    for n in UAVS:
        ropes[n].start(uav[n])
    out["rope_mass"] = {n: ropes[n].mass for n in UAVS}
    L_ROPE = CFG.get("spec", {}).get("rope_length_m", 1.5)
    stats = dict(max_speed=0.0, max_chord=0.0, nan=False, phase_speed=0.0)

    def fire():
        return s.flag(f"{FAC}/FirePan", "arb:fire_active")

    def bucket_state():
        p, q = tf_pq(bv.get_transforms()[0])
        up = rot(q, (0, 0, 1))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        corners = [p + rot(q, (x, y, z)) for x in (b_lo[0], b_hi[0]) for y in (b_lo[1], b_hi[1]) for z in (b_lo[2], b_hi[2])]
        v = bv.get_velocities()[0][:3]
        return dict(pos=r3(p), tilt_deg=round(tilt, 2), bottom_z=round(float(min(c[2] for c in corners)), 4), speed=round(float(np.linalg.norm(v)), 3))

    def hooks():
        return {n: ropes[n].hook() for n in UAVS}

    def offset():
        p = tf_pq(bv.get_transforms()[0])[0]
        return (p - np.mean(list(hooks().values()), axis=0))[:2]

    def track():
        p = tf_pq(bv.get_transforms()[0])[0]
        if not np.all(np.isfinite(p)):
            stats["nan"] = True
            return
        sp = float(np.linalg.norm(bv.get_velocities()[0][:3]))
        stats["max_speed"] = max(stats["max_speed"], sp)
        stats["phase_speed"] = max(stats["phase_speed"], sp)
        for n in UAVS:
            stats["max_chord"] = max(stats["max_chord"], float(np.linalg.norm(ropes[n].ring(bv) - ropes[n].hook())))

    def shapes():
        return {n: ropes[n].shape(bv) for n in UAVS}

    def calm(targets, seconds=2.0, free=2.0):
        """The rig steadies the load by hand: the bucket's and the links' velocities zeroed every step (a quasi-static
        descent to the hanging equilibrium), then a short free settle. What is measured afterwards is the rig's own
        static geometry, not the swing left by the previous move."""
        z6 = np.zeros((1, 6), dtype=np.float32)
        for i in range(int(seconds / DT)):
            for n, g in targets.items():
                uav[n].hold(g)
            bv.set_velocities(z6, np.array([0], dtype=np.int32))
            for n in UAVS:
                v = ropes[n].view
                if v is not None:
                    v.set_velocities(np.zeros((v.count, 6), dtype=np.float32), np.arange(v.count, dtype=np.int32))
            s.step()
            if i % 4 == 0:
                track()
        fly(targets, free)

    def fly(targets, seconds, rec=None, extra=None, smooth=True):
        """The held robots ramped to `targets` over `seconds` (smoothstep: no velocity jumps, unless `smooth` is off).
        `rec` rows: time, the bucket's offset from the hooks' centroid (x, y), its tilt, the fire, its distance to the pan."""
        starts = {n: uav[n].pos().copy() for n in targets}
        ns = int(seconds / DT)
        for i in range(ns):
            f = (i + 1) / ns
            if smooth:
                f = f * f * (3.0 - 2.0 * f)
            for n, g in targets.items():
                uav[n].hold(starts[n] + f * (np.asarray(g) - starts[n]))
            if extra:
                extra()
            s.step()
            if i % 4 == 0:
                track()
            if rec is not None:
                o = offset()
                bs = bucket_state()
                rec.append((s.t, float(o[0]), float(o[1]), bs["tilt_deg"], 1.0 if fire() else 0.0,
                            math.hypot(bs["pos"][0] - pan["center"][0], bs["pos"][1] - pan["center"][1])))

    s.step(int(2.0 / DT))
    out["initial"] = dict(bucket=bucket_state(), ropes=shapes(), fire=fire())
    bp0 = np.array(out["initial"]["bucket"]["pos"])
    hook_off = {n: ropes[n].hook() - uav[n].pos() for n in UAVS}  # (level robots: the hook rides at a fixed offset)
    # --- slack: each hook 1.0 m up and 0.99 m out from its ring along the ring's bearing: a 1.2 m chord of a longer rope
    sag = {}
    for n in UAVS:
        ring = ropes[n].ring(bv)
        d = (ring - bp0)[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        sag[n] = np.array([ring[0] + 0.99 * d[0], ring[1] + 0.99 * d[1], 1.0]) - hook_off[n]
    fly(sag, 3.0)
    fly(sag, 3.0)
    out["slack"] = dict(bucket=bucket_state(), ropes=shapes(), bucket_moved=round(float(np.linalg.norm(np.array(bucket_state()["pos"]) - bp0)), 4))
    # --- the lift: each robot's hook 0.5 m out from the bucket along the robot's own bearing, the hooks at 2.52 m:
    # first over the formation points with the ropes still slack, then straight up
    goals = {}
    for n in UAVS:
        d = uav[n].p0[:2] - bp0[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        goals[n] = np.array([bp0[0] + 0.5 * d[0], bp0[1] + 0.5 * d[1], 2.6]) - np.array([hook_off[n][0], hook_off[n][1], 0.0])
    fly({n: np.array([goals[n][0], goals[n][1], 1.0 - hook_off[n][2]]) for n in UAVS}, 3.0)
    fly(goals, 5.0)
    calm(goals, 2.0, 3.0)
    out["lifted"] = dict(bucket=bucket_state(), ropes=shapes(), offset=r3(offset()), fire=fire())
    # --- the hang: one robot (drawn) 0.3 m further out: where does the bucket settle, how does it tilt?
    pick = RNG.choice(UAVS)
    out["pick"] = pick
    dpick = (goals[pick][:2] - bp0[:2]) / (np.linalg.norm(goals[pick][:2] - bp0[:2]) + 1e-9)
    shifted = dict(goals)
    shifted[pick] = goals[pick] + np.array([0.3 * dpick[0], 0.3 * dpick[1], 0.0])
    fly(shifted, 3.0)
    calm(shifted, 2.0, 3.0)
    off0 = np.array(out["lifted"]["offset"])
    out["hang"] = dict(bucket=bucket_state(), offset=r3(offset()), shift_m=round(float(np.linalg.norm(offset() - off0)), 4), ropes=shapes())
    fly(goals, 3.0)
    calm(goals, 2.0, 2.0)
    # --- the swing: a 0.4 m step along a drawn direction in 1.5 s, then 10 s free
    ang = RNG.uniform(0, 2 * math.pi)
    sd = np.array([math.cos(ang), math.sin(ang)])
    stepped = {n: goals[n] + np.array([0.4 * sd[0], 0.4 * sd[1], 0.0]) for n in UAVS}
    rec = []
    fly(stepped, 1.5, rec)
    fly(stepped, 10.0, rec)
    r = np.array(rec)
    x = (r[:, 1:3] @ sd) - np.mean(r[-int(2.0 / DT):, 1:3] @ sd)
    zc = [r[i, 0] for i in range(1, len(x)) if x[i - 1] < 0 <= x[i]]
    periods = np.diff(zc).tolist() if len(zc) > 1 else []
    peaks = [abs(x[i]) for i in range(1, len(x) - 1) if abs(x[i]) > abs(x[i - 1]) and abs(x[i]) >= abs(x[i + 1]) and abs(x[i]) > 0.01]
    n3 = int(3.0 / DT)
    free = r[int(1.5 / DT):, 1:3] - np.mean(r[-int(2.0 / DT):, 1:3], axis=0)  # (after the step; both axes: the modes exchange energy)
    rms_first, rms_last = float(np.sqrt(np.mean(np.sum(free[:n3] ** 2, axis=1)))), float(np.sqrt(np.mean(np.sum(free[-n3:] ** 2, axis=1))))
    out["swing"] = dict(period_s=round(float(np.median(periods)), 3) if periods else None, periods=[round(p, 3) for p in periods][:8],
                        peaks=[round(p, 3) for p in peaks][:6], decay=round(rms_last / rms_first, 3) if rms_first > 0.005 else None,
                        rms_first=round(rms_first, 4), rms_last=round(rms_last, 4), max_tilt=round(float(np.max(r[:, 3])), 2), bucket=bucket_state())
    calm(stepped, 2.0, 1.0)
    # --- the tip: the drawn robot 0.8 m down
    low = dict(stepped)
    low[pick] = stepped[pick] - np.array([0.0, 0.0, 0.8])
    fly(low, 3.0)
    calm(low, 2.0, 3.0)
    out["tip"] = dict(bucket=bucket_state(), ropes=shapes())
    fly(stepped, 3.0)
    calm(stepped, 2.0, 1.0)
    # --- over the pan, slowly (the fire burns on: nothing is poured), then the drawn robot 1.0 m down: the pour, the fire out
    over = {n: np.array([goals[n][0] - bp0[0] + pan["center"][0], goals[n][1] - bp0[1] + pan["center"][1], goals[n][2]]) for n in UAVS}
    carry = []
    fly(over, 12.0, carry)
    calm(over, 2.0, 2.0)
    # (a bucket that swung past 15 degrees within the pan zone as it arrived has poured by the brief's rule: the fire
    # going out then is the scene doing right, not wrong)
    tipped_in_zone = any(row[3] > 15.0 and row[5] <= 0.6 for row in carry)
    out["over_pan"] = dict(bucket=bucket_state(), fire=fire(), pouring=s.flag(BUCKET, "arb:pouring"), tipped_in_zone=bool(tipped_in_zone),
                           carry_tilt_max=round(max(row[3] for row in carry), 2) if carry else None)
    pour = dict(over)
    pour[pick] = over[pick] - np.array([0.0, 0.0, 0.8])
    trace = []
    fly(pour, 3.0, trace)
    fly(pour, 5.0, trace)
    t_tilt = next((row[0] for row in trace if row[3] > 15.0), None)
    t_out = next((row[0] for row in trace if row[4] < 0.5), None)
    out["pour"] = dict(bucket=bucket_state(), fire=fire(), pouring=s.flag(BUCKET, "arb:pouring"), t_tilt=round(t_tilt, 3) if t_tilt else None,
                       t_fire_out=round(t_out, 3) if t_out else None,
                       fire_out_delay_s=round(t_out - t_tilt, 3) if t_tilt is not None and t_out is not None else None)
    # the pan is harmless now: the vehicle beside it stays enabled
    p_ugv = np.array([pan["center"][0] + 1.4, pan["center"][1], ugv.p0[2]])
    fly(pour, 2.5, extra=lambda: ugv.hold(p_ugv))
    out["ugv_disabled_after_fire_out"] = bool(ugv.disabled())
    out["fire_end"] = fire()
    # --- the dead stop: level again, then 1 m in 0.5 s and a stop
    fly(over, 3.0)
    calm(over, 2.0, 2.0)
    ang2 = RNG.uniform(0, 2 * math.pi)
    fast = {n: over[n] + np.array([math.cos(ang2), math.sin(ang2), 0.0]) for n in UAVS}
    stats["phase_speed"] = 0.0
    rec2 = []
    fly(fast, 0.5, rec2, smooth=False)
    fly(fast, 3.0, rec2)
    r2 = np.array(rec2)
    out["abrupt"] = dict(max_speed=round(stats["phase_speed"], 3), max_tilt=round(float(np.max(r2[:, 3])), 2), bucket=bucket_state(), ropes=shapes())
    # --- set down: back over the start, 1.3 m down: the bucket upright on the floor, the ropes slack
    fly(goals, 8.0)
    calm(goals, 2.0, 1.0)
    down = {n: goals[n] - np.array([0.0, 0.0, 1.3]) for n in UAVS}
    fly(down, 4.0)
    fly(down, 3.0)
    out["set_down"] = dict(bucket=bucket_state(), ropes=shapes())
    out["stats"] = dict(max_speed=round(stats["max_speed"], 3), max_chord=round(stats["max_chord"], 4), stretch=round(stats["max_chord"] - L_ROPE, 4), nan=stats["nan"])


# --------------------------------------------------------------------------------------------- the bay mission (the agent's controller)
def privileged_state(sim):
    """The true state at the instant of the observation (the token-privileged reference's channel; read only, no
    random draw: a privileged run is the same simulation as any other), the wind included: the force acting now on
    each flying robot and on the bucket, and the day's gusts on the mission's clock (`ControlSim.wind_state`)."""
    tfs = sim.iv.get_transforms() if sim.iv is not None else []
    bp, bq, bv, _, tilt = sim.bucket_state()
    parts = {}
    for i, n in enumerate(sim.item_names):
        c, lo, hi = sim.item_box_world(n, tfs)
        pp, qq = control_sim.tf_pq(tfs[i])
        b = sim.items[n]["box"]
        parts[sim.wire[n]] = dict(center=r3(c), lo=r3(lo), hi=r3(hi), pos=r3(pp), quat=[round(float(x), 5) for x in qq],
                                  own_lo=r3(b[0]) if b is not None else None, own_hi=r3(b[1]) if b is not None else None,
                                  held=bool(sim.ugv is not None and sim.ugv["held"] == n))
    uav = {}
    for n in UAVS:
        p, q, v = sim.uav_state(n)
        uav[n] = dict(pos=r3(p), vel=r3(v), lost=n in sim.lost)
    out = dict(uav=uav, hooks={n: r3(h) for n, h in sim.hooks().items()},
               bucket=dict(pos=r3(bp), quat=[round(float(x), 5) for x in bq], vel=r3(bv), tilt_deg=round(float(tilt), 3), water_l=round(float(sim.water), 3)),
               parts=parts, pan=r3(sim.pan), fire=bool(sim.fire), steam=bool(sim.steam), pressure=round(float(sim.pressure()), 3),
               bay=dict(crate=sim.bay.get("crate"), plume=sim.bay.get("plume"), tables=sim.cfg.get("table_tops")))
    if sim.ugv is not None:
        p, q, yaw = sim.ugv_pose()
        v6 = sim.ugv["gv"].get_root_velocities()[0]
        out["ugv"] = dict(pos=r3(p), quat=[round(float(x), 5) for x in q], heading_deg=round(math.degrees(control_sim.wrap(yaw)), 3),
                          vel=r3(v6[:3]), yaw_rate=round(math.degrees(float(v6[5])), 3), lost="UGV" in sim.lost)
    lever = dict(deg=round(float(sim.lever.angle()), 3))
    g = sim.handle_grip()
    if g is not None:
        lever["grip"] = r3(g)
    if sim.lever.art is not None:
        names = list(sim.lever.art.shared_metatype.link_names)
        if "Handle" in names:
            hp, hq = control_sim.tf_pq(sim.lever.art.get_link_transforms()[0][names.index("Handle")])
            lever.update(hub=r3(hp), hub_quat=[round(float(x), 5) for x in hq])
    out["lever"] = lever
    out["wind"] = sim.wind_state()
    return out


@group("control")
def control(out):
    """The agent's controller drives the fleet (the flying robots dynamic under its thrust commands, the vehicle under
    its joint targets and grip) through the bay mission in the real bay: the grader's own stage of the bay as it really
    is (`bay_layer`), with the agent's four robots mounted in it (`mission_stage`), on one day of the fixed hidden set
    (the config's `missions[ARB_MISSION]`, or the day in `ARB_DAY_FILE`; the sensors' noise from the seed). The grader
    relays the wire through two pipes (`ARB_WIRE_DIR`): this process never runs the agent's code. It keeps the bay's
    rules itself and measures the outcomes against the real bay's facts (`bay` in the grader's config)."""
    import mission_stage
    import sensor_rig
    import wire_obs
    wire = os.environ.get("ARB_WIRE_DIR")
    if not wire:
        out["error"] = "no wire to a controller"
        return
    scfg = CFG.get("sensors") or {}
    sensors_on = not CFG.get("sensors_off")  # (off: a maintainer's development run only, review/run_mission.py --sensors off)
    frames_dir = os.environ.get("ARB_FRAMES_DIR")
    if sensors_on and not (frames_dir and os.path.isdir(frames_dir)):
        out["error"] = "no frames folder for the sensors"
        return
    if sensors_on:
        enable_replicator()  # (before any stage is open: no physics runs in these updates)
    if os.environ.get("ARB_DAY_FILE"):  # (a maintainer's day: one day, or a mission entry holding one)
        with open(os.environ["ARB_DAY_FILE"]) as f:
            day = json.load(f)
        day = day.get("day", day) if isinstance(day, dict) else day
    elif os.environ.get("ARB_MISSION") is not None and CFG.get("missions"):
        day = CFG["missions"][int(os.environ["ARB_MISSION"])]
    else:
        out["error"] = "no day for the mission (neither ARB_MISSION with the config's missions nor ARB_DAY_FILE)"
        return
    day = control_sim.normalise_day(day)
    facts = CFG["bay_facts"]
    stage_path = os.path.splitext(OUT)[0] + "_mission.usda"
    out["setup_stage"] = "build"
    t0 = time.time()
    try:
        info = mission_stage.build(SCENE, CFG["bay_layer"], facts, stage_path)
    except mission_stage.MissionStageError as exc:
        out["abort"] = f"the fleet does not fit the real bay: {exc}"[:400]
        out["end"] = "aborted"
        return
    out["real_bay"] = info
    timing = dict(build_s=round(time.time() - t0, 2))
    plan_ = None
    if sensors_on:  # the stage as the sensors see it, and which of the fleet's sensors stream (pxr, before Kit opens it)
        t0 = time.time()
        out["dress"] = mission_stage.dress(stage_path, info, scfg["look"])
        plan_ = sensor_rig.plan(Usd.Stage.Open(stage_path), info["fleet_root"], CFG["catalog"]["sensors"], scfg, unmount=unmount)
        timing["dress_s"] = round(time.time() - t0, 2)
    out["setup_stage"] = "open"
    t0 = time.time()
    s = Scene(stage_path)
    if sensors_on:
        bad = look_physics(s)
        if bad:
            raise RuntimeError(f"physics under the look in the open stage: {bad[:4]}")
    control_sim.prepare(s.stage, hidden=day, fleet_root=info["fleet_root"], rope_root=info["rope_root"])
    s.start()
    timing["open_s"] = round(time.time() - t0, 2)
    out["setup_stage"] = "sim"
    t0 = time.time()
    cfg = dict(CFG.get("control_sim") or {}, fleet_root=info["fleet_root"], rope_root=info["rope_root"],
               item_alias=CFG.get("item_alias") or None, table_tops=facts["table_tops"], pan_nominal=facts["pan"]["center"])
    sim = control_sim.ControlSim(s, CFG["catalog"], cfg, CFG["bay"], day, seed=SEED, log=lambda *a: print("[control]", *a, flush=True))
    out["mission"] = sim.mission()
    timing["init_s"] = round(time.time() - t0, 2)
    rig, mission_extra, export = None, {}, None
    pv_at, own_at = {}, {}  # (a maintainer's frame export: the true state and the robots' own numbers at each capture step)
    if sensors_on:
        out["setup_stage"] = "sensors"
        t0 = time.time()
        bodies = BodyPoses(s, sim)
        fx = CFG.get("frames_out")  # (review/run_mission.py --frames-out: never set by the grader)
        if fx:
            export = sensor_rig.Export(fx["dir"], fx.get("plan"), event_times=lambda: event_times(sim), truth_at=lambda n: pv_at.get(n),
                                       own_at=lambda n: own_at.get(n))
        rig = sensor_rig.SensorRig(s.sim, s.stage, plan_, scfg, frames_dir, SEED, bodies=bodies, needle=needle_of(s, out["dress"]["needle"]),
                                   pressure=sim.pressure, log=lambda *a: print("[sensors]", *a, flush=True), export=export)
        rig.open_slots(sensor_rig.manifest(plan_, scfg))
        mission_extra = {"sensors": {"dir": frames_dir, "manifest": "manifest.json", "every": rig.every, "delay": rig.delay,
                                     "slots": int(scfg["slots"])}}
        timing["sensors_s"] = round(time.time() - t0, 2)
        out["setup_stage"] = "warmup"
        if not rig.warm_up():
            raise RuntimeError(f"the sensors did not warm up: {rig.warm}")
        timing["warmup_s"] = rig.warm["s"]
        out["streamed"] = [dict(id=x["id"], robot=x["robot"], prim=x["prim"], model=x["model"]) for x in plan_["sensors"]]
        out["not_streamed"] = plan_["not_streamed"]
    uav_dofs = {n: list(sim.av[n].shared_metatype.dof_names) for n in UAVS}
    wo = wire_obs.WireObs(SEED, sim.dt, ugv_start=out["mission"]["ugv"]["start"][:2] if sim.ugv else None, uav_dofs=uav_dofs,
                          mission_extra=mission_extra, cfg=scfg.get("odometry"))
    out["setup_stage"] = "ready"
    t0 = time.time()
    privileged = bool(CFG.get("privileged"))
    m = None
    try:
        with open(os.path.join(wire, "obs.fifo"), "w") as fo, open(os.path.join(wire, "act.fifo")) as fi:
            def policy(t, obs):
                frames = None
                if rig is not None:
                    try:
                        frames = rig.step(t, obs["time"], bool(sim.fire))
                    except Exception:  # noqa: BLE001 - (a render failure mid-run ends the run, as any error after the first step)
                        out["render_error"] = traceback.format_exc()[-1200:]
                        raise
                ugv_truth = None
                if sim.ugv is not None:
                    p, q, yaw = sim.ugv_pose()
                    v6 = sim.ugv["gv"].get_root_velocities()[0]
                    ugv_truth = ((float(p[0]), float(p[1])), float(yaw), (float(v6[0]), float(v6[1])))
                uj = {n: (sim.av[n].get_dof_positions()[0], sim.av[n].get_dof_velocities()[0]) for n in UAVS if uav_dofs[n]}
                vacuum = bool(sim.ugv is not None and sim.ugv["held"] is not None)
                w = wo.build(obs, ugv_truth, uj, vacuum, frames)
                if export is not None and t % rig.every == 0:
                    own_at[t // rig.every] = dict(time=w["time"], uav=w["uav"], ugv=w.get("ugv"))
                    for old in [c for c in own_at if c < t // rig.every - 64]:
                        del own_at[old]
                if privileged:
                    w["privileged"] = privileged_state(sim)
                    if export is not None and t % rig.every == 0:
                        pv_at[t // rig.every] = w["privileged"]
                        for old in [c for c in pv_at if c < t // rig.every - 64]:
                            del pv_at[old]
                fo.write(json.dumps({"t": t, "obs": w}) + "\n")
                fo.flush()
                line = fi.readline()
                if not line:
                    out["abort"] = "the wire closed"
                    return None
                reply = json.loads(line)
                if "action" not in reply:
                    out["abort"] = reply.get("abort", "no action")
                    return None
                return reply["action"]

            m = sim.run(policy)
            try:
                fo.write(json.dumps({"end": True}) + "\n")
                fo.flush()
            except (OSError, ValueError):
                pass
    except OSError:  # (closing the wire after the run ended: the relay may have closed its ends already, after an abort)
        if m is None:
            raise
    out.update(m)
    out["hidden"] = sim.hidden_record()
    timing["run_s"] = round(time.time() - t0, 2)
    timing["ms_per_step"] = round(1000.0 * timing["run_s"] / max(1, m.get("steps") or 1), 2)
    out["timing"] = timing
    out["setup_stage"] = "run"
    if rig is not None:
        out["frames"] = dict(rig.summary(), gpu=gpu_memory())
        if export is not None:
            out["frames"]["exported"] = export.n
        rig.close()


def unmount(path):
    """A mission-stage path as the agent's scene names it (`/Agent/...` -> `/World/...`)."""
    p = str(path)
    return "/World" + p[len("/Agent"):] if p == "/Agent" or p.startswith("/Agent/") else p


class BodyPoses:
    """The true world pose of a robot's body from PhysX's tensor views (read only): an articulation link found by its
    prim path (names repeat: a Jackal with a Gen3 has two `base_link`s), else a rigid-body view of its own."""

    def __init__(self, s, sim):
        self.s, self.sim, self.where, self.own = s, sim, {}, {}
        self.arts = dict(sim.av)
        if sim.ugv is not None:
            self.arts["UGV"] = sim.ugv["gv"]

    def _find(self, path):
        for name, av in self.arts.items():
            paths = None
            try:
                paths = list(av.link_paths[0])
            except Exception:  # noqa: BLE001 - (no path list: names, where they are unique)
                paths = None
            if paths is not None:
                if path in paths:
                    return ("art", av, paths.index(path))
                continue
            names = list(av.shared_metatype.link_names)
            leaf = path.split("/")[-1]
            if names.count(leaf) == 1 and path.startswith(self.sim.fleet + "/" + name + "/"):
                return ("art", av, names.index(leaf))
        v = self.s.view.create_rigid_body_view(path)
        return ("rb", v, 0)

    def __call__(self, path):
        w = self.where.get(path)
        if w is None:
            w = self.where[path] = self._find(path)
        kind, v, i = w
        tf = v.get_link_transforms()[0][i] if kind == "art" else v.get_transforms()[i]
        p, q = tf_pq(tf)
        return p, q / np.linalg.norm(q)


def event_times(sim):
    """The moments a maintainer's frame export plans around (the grader's own record): lift-off, the pour's start,
    the fire out, the lever closed; each part's grips and releases."""
    m = sim.metrics
    out = {k: [m[f"{k}_t"]] for k in ("liftoff", "pour_start", "fire_out", "lever_closed") if m.get(f"{k}_t") is not None}
    held = sim.ugv["held"] if sim.ugv is not None else None
    last = getattr(sim, "_arb_held_last", None)
    ev = getattr(sim, "_arb_held_events", {"grip": [], "release": []})
    if held != last:
        ev["grip" if held is not None else "release"].append(round(sim.t, 3))
        sim._arb_held_last, sim._arb_held_events = held, ev
    out.update(ev)
    return out


# --------------------------------------------------------------------------------------------- the fleet's design
@group("fleet")
def fleet(out):
    """What the fleet is built from and whether it can do the job: masses against the catalog's limits, the sensors'
    intrinsics against the catalog, what they see from the brief's fixed points, the ground vehicle's reach."""
    s = Scene()
    dial = box_facts(s, f"{FAC}/SteamLine/Gauge/Dial")
    if dial:
        dial["diameter"] = round(sorted(dial["extent"])[-1], 4)
    pan = box_facts(s, f"{FAC}/FirePan")
    handle = box_facts(s, f"{FAC}/SteamLine/Handle")
    square = box_facts(s, f"{FAC}/Pads/Valve_Square", xy_only=True)
    robots = {}
    for name in ROBOTS:
        root = f"{FLEET}/{name}"
        if not s.prim(root).IsValid():
            continue
        robots[name] = dict(art_root=s.art_root(root), sensors=cameras_of(s, root), airframe=declared(s, root, "airframe_model"),
                            arm=declared(s, root, "arm_model"), base=declared(s, root, "base_model"))
    s.start()
    views = {}
    for name in robots:
        try:
            views[name] = s.view.create_articulation_view(robots[name]["art_root"])
        except Exception as exc:  # noqa: BLE001
            robots[name]["error"] = str(exc)[:200]
    s.step(1)
    for name, av in views.items():
        p, q = tf_pq(av.get_root_transforms()[0])
        robots[name].update(mass=fin(np.sum(av.get_masses()[0])), root_pos=r3(p), root_quat=r3(q), dofs=int(av.max_dofs))
        # each sensor's pose relative to the root body (from the stage as authored: both at rest)
        rp, rq = s.usd_pose(robots[name]["art_root"])
        for c in robots[name]["sensors"]:
            if c.get("camera") and c.get("pos"):
                c["pos_in_root"] = r3(rot(qinv(rq), np.array(c["pos"]) - rp))
                c["look_in_root"] = r3(rot(qinv(rq), np.array(c["look"])))

    def view_of(cam, root_p, root_q, target, size):
        """Angular position of a target (centre, size) in a camera placed by its root body's pose: in view? px across?"""
        cp = root_p + rot(root_q, cam["pos_in_root"])
        look = rot(root_q, cam["look_in_root"])
        up = rot(root_q, rot(qinv(np.array(robots_rq[cam["_robot"]])), np.array(cam["up"])))
        right = np.cross(look, up)
        d = np.asarray(target) - cp
        dist = float(np.linalg.norm(d))
        dn = d / (dist + 1e-9)
        ang_h = math.degrees(math.atan2(float(np.dot(dn, right)), float(np.dot(dn, look))))
        ang_v = math.degrees(math.atan2(float(np.dot(dn, up)), float(np.dot(dn, look))))
        in_view = (float(np.dot(dn, look)) > 0 and abs(ang_h) + math.degrees(math.atan(size / 2 / dist)) <= (cam["hfov_deg"] or 0) / 2
                   and abs(ang_v) + math.degrees(math.atan(size / 2 / dist)) <= (cam["vfov_deg"] or 0) / 2)
        ang_size = math.degrees(2 * math.atan(size / 2 / dist))
        return dict(dist=round(dist, 3), ang_h=round(ang_h, 2), ang_v=round(ang_v, 2), in_view=bool(in_view), ang_size_deg=round(ang_size, 3))

    robots_rq = {name: s.usd_pose(robots[name]["art_root"])[1] for name in robots}
    for name in robots:
        for c in robots[name]["sensors"]:
            c["_robot"] = name
    # the thermal view of the pan from the approach hover, with the robot's rest orientation (heading +y, the pads' arrow)
    for name in UAVS:
        if name not in robots or name not in views:
            continue
        rq = np.array(robots[name]["root_quat"])
        rp = np.array(APPROACH)
        for c in robots[name]["sensors"]:
            if c.get("camera") and c.get("pos_in_root") and pan:
                c["pan_from_approach"] = view_of(c, rp, rq, pan["center"], max(pan["extent"][:2]))
    # the ground vehicle's cameras: the gauge dial from the parking bay, as parked
    if "UGV" in robots and "UGV" in views and dial:
        rp, rq = np.array(robots["UGV"]["root_pos"]), np.array(robots["UGV"]["root_quat"])
        for c in robots["UGV"]["sensors"]:
            if c.get("camera") and c.get("pos_in_root"):
                c["dial_from_bay"] = view_of(c, rp, rq, dial["center"], dial["diameter"])
                # the dial faces -y: a camera in front of it, not behind
                c["dial_from_bay"]["in_front"] = bool(c["pos"][1] < dial["center"][1])
    for name in robots:
        for c in robots[name]["sensors"]:
            c.pop("_robot", None)
    # reach: the vehicle placed at the valve square (as parked: same heading), the arm's joints sampled within their
    # limits; the nearest any link's origin comes to the handle's grip (the far end of the handle from its hub)
    if "UGV" in views and handle and square:
        av = views["UGV"]
        rq = np.array(robots["UGV"]["root_quat"])
        rp = np.array([square["center"][0], square["center"][1], robots["UGV"]["root_pos"][2]])
        hub = s.usd_pose(f"{FAC}/SteamLine/Handle")[0]
        hc = np.array(handle["center"])
        grip = hub + 2.0 * (hc - hub) * 0.85  # (the handle's centre is half-way along it: its end is twice as far, less the grip)
        lims = av.get_dof_limits()[0]
        n = av.max_dofs
        best = 1e9
        rng = np.random.default_rng(SEED)
        for _ in range(60):  # 60 batches of 400 configurations
            qs = np.zeros((400, n), dtype=np.float32)
            for k in range(n):
                lo, hi = lims[k]
                lo, hi = (lo if np.isfinite(lo) and lo > -50 else -math.pi), (hi if np.isfinite(hi) and hi < 50 else math.pi)
                qs[:, k] = rng.uniform(lo, hi, 400)
            for q in qs:
                av.set_dof_positions(q[None], np.array([0], dtype=np.int32))
                av.set_root_transforms(pq_tf(rp, rq)[None], np.array([0], dtype=np.int32))
                s.view.update_articulations_kinematic()
                links = av.get_link_transforms()[0][:, :3]
                m = float(np.linalg.norm(links - grip, axis=1).min())
                if m < best:
                    best = m
        out["reach"] = dict(min_dist_to_grip=round(best, 4), grip=r3(grip), hub=r3(hub), samples=24000)
    out["robots"] = robots


if __name__ == "__main__":
    {g.group_name: g for g in (canary, canary_render, layout, settle, steam, hoist, control, fleet)}[GROUP]()
    RESULT["done"] = True
    with open(OUT, "w") as f:
        json.dump(RESULT, f)
    print("PROBES DONE", flush=True)
    os._exit(0)
