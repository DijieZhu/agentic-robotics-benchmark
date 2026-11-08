"""The bay mission's sensors (control.md, The sensors): what the fleet's declared cameras and lidars deliver, rendered
by the grader in the real bay, and how the frames reach the program.

    plan = sensor_rig.plan(stage, fleet_root, catalog, cfg)            # pxr: which sensors stream, on which body
    rig = sensor_rig.SensorRig(sim_ctx, stage, view, plan, cfg, frames_dir, noise_seed, bodies=...)   # Kit
    rig.warm_up(); ...; frames = rig.step(k, time, fire)              # every control step, in the probe's closure

- **Which sensors stream** (`plan`): per robot, in prim order, the first `cap.cameras` cameras that count (a
  `UsdGeom.Camera` with `arb:sensor_model` naming a catalog camera, its intrinsics the catalog's within 2 %) and the
  first `cap.lidars` lidar (a prim naming a catalog lidar); never inside an instance (`scan.stream_selection`, the
  agent's own check makes the same choice). Each rides on its nearest rigid-body ancestor inside its robot, or on the
  articulation's root body; its pose in that body's frame is the composed scene's at rest.
- **The grader's cameras** (`/World/Sensors/...`, no physics schema): one per streamed camera, at the catalog's
  intrinsics (pinhole, square pixels, clipping 0.01-100 m), and one equirectangular range camera per lidar head;
  posed at each capture from the body's true pose (PhysX tensor views, read only) composed with the mount. Nothing of
  the agent's camera prims reaches the renderer.
- **The schedule**: every `every` control steps all sensors are captured together: one Replicator orchestrator step
  (`rep.orchestrator.step(rt_subframes=1, delta_time=0)`, physics held off: `/app/player/playSimulations` false around
  it), after which every product's read shows that capture (measured in P1: a plain `SimulationContext.render()` read
  showed the render 2 to 7 renders back, varying; the orchestrator's was fresh in every read). The frame clock (a
  grader-only depth product looking at a quad moved to 1 + 0.25 (n mod 64) m) proves it: frames are stamped with the
  capture the clock decodes, a mismatch is counted (`lag_anomalies`). The processed frame set is held for one capture
  period and written to its slots at the next capture step: delivered `delay` steps after its capture.
- **Processing**: thermal = the instance segmentation mapped to the stated temperatures (the fire pan hot while it
  burns, then cooling) + NETD noise; depth = z-depth in range + noise; lidar = the range image sampled at the beams +
  noise; RGB = the renderer's sRGB, repacked RGBA -> RGB. Every frame's noise from its own stream
  (`wire_obs.frame_rng(noise_seed, sensor index, frame set index)`).
- **Delivery**: slot files in the probe-owned frames folder (two per sensor, a 64-byte seqlock header, `Slot`), a
  manifest written once before the wire opens (`manifest`).

The writer (`Slot`, `write_excl`, `manifest`) and the planning are pxr / NumPy only, so the maintainers' format check
(`review/check_given_copies.py`) runs them without Kit.
"""

import hashlib
import json
import math
import mmap
import os
import struct
import time

import numpy as np

try:  # (OpenCV's SIMD repack of an RGBA frame to RGB: a tenth of NumPy's strided copy)
    import cv2
except ImportError:
    cv2 = None

HEADER = struct.Struct("<8sQQQdIIIIQ")  # magic, seq, n, t, time, width, height, channels, dtype, payload bytes
HEADER_BYTES = 64
MAGIC = b"BAYFRM1\x00"
DTYPES = {"uint8": (1, np.uint8), "float32": (2, np.float32)}
NONE_N = (1 << 64) - 1  # (a slot before its first frame)
FORMAT = "bay frames 1"
UAVS = ("UAV_1", "UAV_2", "UAV_3")
ROBOTS = UAVS + ("UGV",)
SCOPE = "/World/Sensors"
assert HEADER.size == HEADER_BYTES


# --------------------------------------------------------------------------------------------- the slots (no Kit)
def write_excl(path, data):
    """A new file the probe alone creates (never one that exists, never through a link), world-readable."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o644)
        os.write(fd, data)
    finally:
        os.close(fd)


class Slot:
    """One slot file of one sensor: a 64-byte header (seqlock) and the payload, row-major, little-endian. The writer
    sets seq odd, writes the payload and the header's frame fields, then sets seq even; a reader that works between
    requests checks that seq is equal and even before and after its copy."""

    def __init__(self, path, width, height, channels, dtype):
        self.code, self.np = DTYPES[dtype]
        self.shape = (height, width, channels) if channels > 1 else (height, width)
        self.w, self.h, self.c = int(width), int(height), int(channels)
        self.nbytes = int(width) * int(height) * int(channels) * np.dtype(self.np).itemsize
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o644)
            os.ftruncate(fd, HEADER_BYTES + self.nbytes)
            self.mm = mmap.mmap(fd, HEADER_BYTES + self.nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self.payload = np.frombuffer(self.mm, dtype=self.np, count=self.nbytes // np.dtype(self.np).itemsize,
                                     offset=HEADER_BYTES).reshape(self.shape)
        self.seq = 0
        HEADER.pack_into(self.mm, 0, MAGIC, 0, NONE_N, 0, 0.0, self.w, self.h, self.c, self.code, self.nbytes)

    def write(self, n, t, tm, arr):
        """Frame n (captured at step t, time tm) into this slot. `arr` is the frame (an RGBA image is cut to RGB)."""
        self.seq += 1
        struct.pack_into("<Q", self.mm, 8, self.seq)
        if self.c == 3 and arr.ndim == 3 and arr.shape[2] == 4:
            if cv2 is not None and arr.dtype == np.uint8 and arr.flags.c_contiguous:
                cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB, dst=self.payload)
            else:
                np.copyto(self.payload, arr[..., :3], casting="unsafe")
        else:
            np.copyto(self.payload, arr.reshape(self.shape), casting="unsafe")
        struct.pack_into("<QQd", self.mm, 16, int(n), int(t), float(tm))
        self.seq += 1
        struct.pack_into("<Q", self.mm, 8, self.seq)

    def close(self):
        try:
            del self.payload
            self.mm.close()
        except (BufferError, ValueError):
            pass


# --------------------------------------------------------------------------------------------- models (no Kit)
def stream_of(model, entry):
    """What a catalog model streams (`entry["stream"]`, the catalog's own numbers): kind, frame size, format, unit,
    noise, range, and for a lidar its beams."""
    s = dict(entry.get("stream") or {})
    kind = entry["type"]
    out = dict(model=model, kind=kind, dtype=s.get("format", "uint8" if kind == "rgb" else "float32"),
               channels=int(s.get("channels", 3 if kind == "rgb" else 1)), unit=s.get("unit"), noise=s.get("noise"), range_m=s.get("range_m"))
    if kind == "lidar":
        out.update(width=int(s["columns"]), height=int(s["rows"]), elevation_deg=[float(v) for v in s["elevation_deg"]])
    else:
        out.update(width=int(entry["res"][0]), height=int(entry["res"][1]), focal=float(entry["focal"]), hap=float(entry["hap"]))
    return out


def intrinsics(sm):
    """Pinhole, no distortion: fx = fy = focal / horizontal aperture x W; the principal point at the image centre
    ((W - 1) / 2, (H - 1) / 2), (0, 0) the centre of the top-left pixel."""
    fx = sm["focal"] / sm["hap"] * sm["width"]
    return dict(fx=fx, fy=fx, cx=(sm["width"] - 1) / 2.0, cy=(sm["height"] - 1) / 2.0)


def beams(sm):
    """A lidar's beam grid: elevations of row 0 .. the last row (linear), azimuths k * 360 / columns counter-clockwise
    about the prim's +Z from its +X."""
    rows, cols = sm["height"], sm["width"]
    e0, e1 = sm["elevation_deg"]
    elev = np.array([e0 + (e1 - e0) * c / (rows - 1) for c in range(rows)]) if rows > 1 else np.array([e0])
    return dict(rows=rows, columns=cols, elevation_first_deg=e0, elevation_last_deg=e1, azimuth_step_deg=360.0 / cols), elev


def thermal_temperature(fire, t, t_out, th):
    """The fire pan's temperature at a capture: hot while the fire burns, then cooling toward `cool_to_k` with
    `tau_s` from the first observation in which the fire is out."""
    if fire or t_out is None:
        return float(th["burning_k"])
    return float(th["cool_to_k"] + (th["burning_k"] - th["cool_to_k"]) * math.exp(-max(0.0, t - t_out) / th["tau_s"]))


def needle_angle(p_bar, gauge):
    """The gauge needle's angle from vertical, clockwise as seen from the bay [deg]."""
    return float(gauge["zero_deg"] + gauge["deg_per_bar"] * p_bar)


def depth_noise(z, noise, rng):
    """N(0, abs + rel z) or N(0, quad z^2) per valid pixel (the manifest's stated noise)."""
    sig = np.zeros_like(z)
    if noise.get("sigma_m") is not None:
        sig = sig + float(noise["sigma_m"])
    if noise.get("rel") is not None:
        sig = sig + float(noise["rel"]) * z
    if noise.get("quad_m") is not None:
        sig = sig + float(noise["quad_m"]) * z * z
    return z + sig * rng.standard_normal(z.shape, dtype=np.float32)


# --------------------------------------------------------------------------------------------- planning (pxr)
def _gfq(qd):
    return np.array([qd.GetReal(), *qd.GetImaginary()])


def _pose(prim):
    from pxr import Usd, UsdGeom
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    m = m.RemoveScaleShear()
    return np.array(m.ExtractTranslation()), _gfq(m.ExtractRotationQuat())


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotm(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_quat(r):
    """A rotation matrix -> quaternion (w, x, y, z)."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def art_root_body(stage, root):
    """The articulation's root body (control.md, Positions): the prim with ArticulationRootAPI, or the first rigid
    body below it."""
    from pxr import Usd, UsdPhysics
    arts = [q for q in Usd.PrimRange(stage.GetPrimAtPath(root)) if q.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not arts:
        return None
    a = arts[0]
    if a.HasAPI(UsdPhysics.RigidBodyAPI):
        return a.GetPath().pathString
    bodies = [q for q in Usd.PrimRange(a) if q.HasAPI(UsdPhysics.RigidBodyAPI)]
    return bodies[0].GetPath().pathString if bodies else a.GetPath().pathString


def body_of(stage, prim_path, robot_root):
    """The rigid body a sensor rides on: its nearest ancestor with RigidBodyAPI inside its robot (itself included),
    else the articulation's root body."""
    from pxr import UsdPhysics
    p = stage.GetPrimAtPath(prim_path)
    while p and p.IsValid() and p.GetPath().pathString.startswith(robot_root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p.GetPath().pathString
        if p.GetPath().pathString == robot_root:
            break
        p = p.GetParent()
    return art_root_body(stage, robot_root)


def plan(stage, fleet_root, catalog, cfg, unmount=lambda p: p):
    """Which sensors stream and how they are mounted (pxr only; the composed mission stage at rest). Returns
    dict(sensors=[...], not_streamed=[...]) in manifest order: robot by robot (UAV_1, UAV_2, UAV_3, UGV), each in prim
    order. `unmount` turns a mission-stage path into the agent's own (`/Agent/...` -> `/World/...`)."""
    import scan
    cap = cfg.get("cap") or {}
    out, skipped = [], []
    for robot in ROBOTS:
        root = f"{fleet_root}/{robot}"
        if not stage.GetPrimAtPath(root).IsValid():
            continue
        streamed, others = scan.stream_selection(stage, root, catalog, int(cap.get("cameras", 4)), int(cap.get("lidars", 1)))
        for path, model in streamed:
            body = body_of(stage, path, root)
            bp, bq = _pose(stage.GetPrimAtPath(body))
            sp, sq = _pose(stage.GetPrimAtPath(path))
            pos = rotm(bq).T @ (sp - bp)
            quat = qmul(qinv(bq), sq)
            quat = quat / np.linalg.norm(quat)
            sm = stream_of(model, catalog[model])
            k = len(out)
            out.append(dict(index=k, id=f"s{k:02d}", stem=f"s{k:02d}", robot=robot, prim=unmount(path), stage_prim=path, model=model,
                            kind=sm["kind"], body=unmount(body), stage_body=body, robot_root=root, pos_in_body=pos.tolist(),
                            quat_in_body=quat.tolist(), stream=sm))
        for path, model, why in others:
            skipped.append(dict(robot=robot, prim=unmount(path), model=model, why=why))
    return dict(sensors=out, not_streamed=skipped)


def manifest(plan_, cfg, sensors_dir_name="."):
    """The frames folder's manifest (control.md, The frames folder): everything a program needs to read and place each
    streamed sensor's frames, nothing about the bay."""
    sens = []
    for s in plan_["sensors"]:
        sm = s["stream"]
        e = dict(id=s["id"], file=s["stem"], robot=s["robot"], prim=s["prim"], model=s["model"], kind=s["kind"], width=sm["width"],
                 height=sm["height"], channels=sm["channels"], dtype=sm["dtype"], unit=sm["unit"])
        if s["kind"] == "lidar":
            e["beams"] = beams(sm)[0]
        else:
            e["intrinsics"] = intrinsics(sm)
        e.update(range_m=sm["range_m"], noise=sm["noise"], body=s["body"],
                 pose_in_body=dict(pos=[round(float(v), 6) for v in s["pos_in_body"]], quat=[round(float(v), 7) for v in s["quat_in_body"]]))
        sens.append(e)
    return {"format": FORMAT, "every": int(cfg["every"]), "delay": int(cfg["delay"]), "slots": int(cfg["slots"]),
            "header_bytes": HEADER_BYTES, "slot_files": "<file>.<slot>", "sensors": sens, "not_streamed": plan_["not_streamed"]}


# --------------------------------------------------------------------------------------------- the renderer (Kit)
def pin_settings(stage, rp_path, settings):
    """The render product's own RTX settings (Kit 110 keeps them per product; carb settings set before the stage
    opened are reset by it): each one the product has is set; returns what was set and what the product lacks."""
    prim = stage.GetPrimAtPath(rp_path)
    done, absent = {}, []
    for name, val in settings.items():
        a = prim.GetAttribute(name) if prim and prim.IsValid() else None
        if a is not None and a.IsValid():
            a.Set(val)
            done[name] = val
        else:
            absent.append(name)
    return done, absent


def product_rtx_attrs(stage, rp_path):
    prim = stage.GetPrimAtPath(rp_path)
    return {a.GetName(): str(a.Get()) for a in (prim.GetAttributes() if prim and prim.IsValid() else []) if a.GetName().startswith("omni:rtx")}


class Products:
    """The grader's cameras and their render products: camera prims under `/World/Sensors` (no physics schema), one
    render product each, the annotators the kind needs. `spec` per camera: name, kind ("rgb", "thermal", "depth",
    "range" (a lidar head, equirectangular), "clock"), width, height, focal / hap (pinhole), clip, settings."""

    ANNOT = {"rgb": "rgb", "thermal": "instance_id_segmentation", "depth": "distance_to_image_plane", "range": "distance_to_camera",
             "clock": "distance_to_image_plane"}

    def __init__(self, stage, specs, settings, log=print, update=None):
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom
        self.rep, self.stage, self.log = rep, stage, log
        self.update = update  # (an app update that holds physics off: Replicator's node types may register late)
        self.retries = 0
        if not stage.GetPrimAtPath(SCOPE).IsValid():
            UsdGeom.Scope.Define(stage, SCOPE)
        self.items = []
        self.pinned, self.absent = {}, set()
        for sp in specs:
            path = f"{SCOPE}/{sp['name']}"
            cam = UsdGeom.Camera.Define(stage, path)
            if sp["kind"] == "range":  # the lidar head: a 360 x 180 degree equirectangular range camera
                cam.CreateFocalLengthAttr(10.0)
                cam.CreateHorizontalApertureAttr(20.0)
                cam.CreateVerticalApertureAttr(10.0)
                cam.GetPrim().CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set("fisheyeSpherical")
            else:
                cam.CreateFocalLengthAttr(float(sp["focal"]))
                cam.CreateHorizontalApertureAttr(float(sp["hap"]))
                cam.CreateVerticalApertureAttr(float(sp["hap"]) * sp["height"] / sp["width"])  # (square pixels)
            cam.CreateClippingRangeAttr(Gf.Vec2f(*[float(v) for v in sp.get("clip", (0.01, 100.0))]))
            cam.CreateFStopAttr(0.0)  # (no depth of field)
            cam.CreateProjectionAttr("perspective")
            op = cam.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
            rp = rep.create.render_product(path, (int(sp["width"]), int(sp["height"])), name=f"arb_{sp['name']}")
            rpp = rp.path if hasattr(rp, "path") else str(rp)
            kw = {"init_params": {"colorize": False}} if sp["kind"] == "thermal" else {}
            an = self._annotator(self.ANNOT[sp["kind"]], kw)
            an.attach([rp])
            done, absent = pin_settings(stage, rpp, sp.get("settings") or settings)
            self.pinned[sp["kind"]] = done
            self.absent |= set(absent)
            self.items.append(dict(spec=sp, path=path, op=op, rp=rp, rp_path=rpp, an=an))
        self.first_attrs = product_rtx_attrs(stage, self.items[0]["rp_path"]) if self.items else {}

    def _annotator(self, name, kw, tries=20):
        """An annotator; when its node type is not registered yet, the app updated once with physics held off, and again
        (Replicator registers its graph nodes with the app's updates; in P2 a Kit cache another account had written kept
        them from registering at all: verify makes Kit's folders the probe account's alone)."""
        for k in range(tries):
            try:
                return self.rep.AnnotatorRegistry.get_annotator(name, **kw)
            except Exception:  # noqa: BLE001
                if k == tries - 1 or self.update is None:
                    raise
                self.retries += 1
                self.update()
        raise RuntimeError(name)

    def set_pose(self, i, p, q):
        """Camera i to world position p, orientation q (w, x, y, z)."""
        from pxr import Gf
        m = Gf.Matrix4d(1.0)
        m.SetRotateOnly(Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))
        m.SetTranslateOnly(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        self.items[i]["op"].Set(m)

    def read(self):
        """Every annotator's data on the host: [(array, info)] in product order."""
        out = []
        for it in self.items:
            d = it["an"].get_data()
            info = None
            if isinstance(d, dict):
                info = d.get("info")
                d = d.get("data")
            out.append((d, info))
        return out


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """The orientation (w, x, y, z) of a USD camera at `eye` looking at `target` (-Z forward, +Y up)."""
    f = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, dtype=float))
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return mat_quat(np.stack([r, u, -f], axis=1))


# (the lidar head's camera frame in the lidar prim's frame: camera -Z = lidar +X, camera +Y = lidar +Z)
HEAD_FIX = mat_quat(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def range_grid(sm, cfg_lidar):
    """The lidar head's render size and where each beam samples it: (W, H, rows index, cols index, yaw offset)."""
    rows, cols = sm["height"], sm["width"]
    s = int(cfg_lidar.get("px_per_column", 1))
    W = cols * s
    e = beams(sm)[1]
    if rows > 1:
        m = int(cfg_lidar.get("rows_per_band", 1))  # (H = 2 (rows - 1) m, m odd: every beam's elevation on a row centre)
        span = abs(sm["elevation_deg"][0] - sm["elevation_deg"][1])
        H = int(round(180.0 / span * (rows - 1))) * m
        ri = np.round(H / 2.0 - e * H / 180.0 - 0.5).astype(int)
    else:
        H = int(cfg_lidar.get("rows_2d", 63))
        ri = np.array([(H - 1) // 2])
    ci = np.array([int((W // 2 - s * k) % W) for k in range(cols)])
    yaw = math.pi / W  # (half a pixel: the beam's azimuth on a pixel centre)
    return W, H, ri, ci, yaw


class SensorRig:
    """The mission's sensors in the probe: the grader's cameras posed from the bodies' true poses, one render per
    capture, the frames processed and written to their slots, the frame clock checked. `bodies(path) -> (p, q)`
    gives a body's true world pose (PhysX tensor views)."""

    def __init__(self, sim_ctx, stage, plan_, cfg, frames_dir, noise_seed, bodies, needle=None, pressure=None, log=print,
                 export=None):
        from pxr import Gf, UsdGeom
        import wire_obs
        self.wire_obs = wire_obs
        self.ctx, self.stage, self.plan, self.cfg, self.log = sim_ctx, stage, plan_, cfg, log
        self.seed = int(noise_seed)
        self.bodies = bodies
        self.every, self.delay = int(cfg["every"]), int(cfg["delay"])
        self.th = cfg["thermal"]
        self.gauge = cfg["gauge"]
        self.pressure = pressure
        self.dt = None
        specs = []
        self.sens = plan_["sensors"]
        for s in self.sens:
            sm = s["stream"]
            if s["kind"] == "lidar":
                W, H, ri, ci, yaw = range_grid(sm, cfg["lidar"])
                s["grid"] = (ri, ci)
                s["yaw"] = yaw
                # (a lidar draws nothing nearer than its minimum range: its own housing round the prim's origin
                # included; the renderer's distance_to_camera runs from that near plane, added back in `process`)
                s["near"] = float(sm["range_m"][0])
                specs.append(dict(name=s["id"], kind="range", width=W, height=H, clip=(s["near"], cfg["clip_m"][1]),
                                  settings=cfg.get("range_settings")))
            else:
                specs.append(dict(name=s["id"], kind=s["kind"], width=sm["width"], height=sm["height"], focal=sm["focal"], hap=sm["hap"],
                                  clip=cfg["clip_m"]))
        ck = cfg["clock"]
        specs.append(dict(name="Clock", kind="clock", width=4, height=4, focal=50.0, hap=5.0, clip=ck["clip_m"], settings=cfg.get("range_settings")))
        self._clock_scene(ck)
        t0 = time.time()
        self.P = Products(stage, specs, cfg["render"], log=log, update=sim_ctx.render)
        self.create_s = round(time.time() - t0, 2)
        self.P.set_pose(len(specs) - 1, self.clock_eye, self.clock_q)
        self.needle = needle  # (the needle's head op and its rest angle: mission_stage.dress)
        self.slots = None
        self.frames_dir = frames_dir
        self.rec = {}  # capture index -> {t, time, pan_k}
        self.t_fire_out = None
        self.delivered = None  # the latest frame set delivered: {n, t, time, slot}
        self.last_m = -1
        self.stats = dict(render_ms=[], read_ms=[], cpu_ms=[], write_ms=[], pose_ms=[], frame_sets=0, lag_anomalies=0, lag_seen={},
                          stale=0, bytes=0)
        self.digest = {s["id"]: hashlib.blake2b(digest_size=8) for s in self.sens}
        self.export = export
        self.bench = dict(cfg.get("bench") or {})
        self.Gf, self.UsdGeom = Gf, UsdGeom
        import carb
        import omni.replicator.core as rep
        self.settings, self.rep = carb.settings.get_settings(), rep
        self.read_mode = cfg.get("read", "orchestrator")
        self.near = float(cfg["clip_m"][0])
        self.ready = {}  # (processed frame sets by capture, waiting for their delivery step)

    # -- the frame clock ------------------------------------------------------------------------------------------
    def _clock_scene(self, ck):
        from pxr import Gf, UsdGeom
        st = self.stage
        eye = np.array(ck["at"], dtype=float)
        box = UsdGeom.Cube.Define(st, f"{SCOPE}/ClockBox")
        box.CreateSizeAttr(1.0)
        box.CreateDoubleSidedAttr(True)
        box.AddTranslateOp().Set(Gf.Vec3d(*eye))
        box.AddScaleOp().Set(Gf.Vec3f(*([float(ck["box_m"])] * 3)))
        quad = UsdGeom.Cube.Define(st, f"{SCOPE}/ClockQuad")
        quad.CreateSizeAttr(1.0)
        self.quad_op = quad.AddTranslateOp()
        quad.AddScaleOp().Set(Gf.Vec3f(0.01, float(ck["quad_m"]), float(ck["quad_m"])))
        self.clock_eye = eye
        self.clock_q = look_at(eye, eye + np.array([1.0, 0.0, 0.0]))
        self.clock_ck = ck

    def _clock_set(self, n):
        d = self.clock_ck["d0_m"] + self.clock_ck["step_m"] * (n % self.clock_ck["modulo"])
        self.quad_op.Set(self.Gf.Vec3d(float(self.clock_eye[0] + d + 0.005), float(self.clock_eye[1]), float(self.clock_eye[2])))

    def _clock_decode(self, arr):
        if arr is None or not getattr(arr, "size", 0):
            return None
        v = np.asarray(arr, dtype=np.float64).reshape(-1)
        v = v[np.isfinite(v)]
        if not v.size:
            return None
        ck = self.clock_ck
        return int(round((float(np.median(v)) - ck["d0_m"]) / ck["step_m"])) % ck["modulo"]

    # -- the slots ------------------------------------------------------------------------------------------------
    def open_slots(self, man):
        """The slot files and the manifest in the frames folder (before the wire opens)."""
        self.slots = []
        for s in self.sens:
            sm = s["stream"]
            self.slots.append([Slot(os.path.join(self.frames_dir, f"{s['stem']}.{j}"), sm["width"], sm["height"], sm["channels"], sm["dtype"])
                               for j in range(int(self.cfg["slots"]))])
        write_excl(os.path.join(self.frames_dir, "manifest.json"), json.dumps(man, indent=1).encode())
        if self.export is not None:  # (the export keeps the manifest beside its frames)
            with open(os.path.join(self.export.dir, "manifest.json"), "w") as f:
                json.dump(man, f, indent=1)

    # -- posing ---------------------------------------------------------------------------------------------------
    def pose_all(self, n):
        from pxr import Sdf
        with Sdf.ChangeBlock():
            for i, s in enumerate(self.sens):
                bp, bq = self.bodies(s["stage_body"])
                p = bp + rotm(bq) @ np.array(s["pos_in_body"])
                q = qmul(bq, np.array(s["quat_in_body"]))
                if s["kind"] == "lidar":
                    c, sn = math.cos(s["yaw"] / 2), math.sin(s["yaw"] / 2)
                    q = qmul(qmul(q, np.array([c, 0.0, 0.0, sn])), HEAD_FIX)
                self.P.set_pose(i, p, q)
            if self.needle is not None and self.pressure is not None:
                self.needle_set(self.pressure())
            self._clock_set(n)

    def needle_set(self, p_bar):
        """The needle turned rigidly about the dial's axis (+y through its centre) to the pressure's angle."""
        nd = self.needle
        delta = math.radians(needle_angle(p_bar, self.gauge) - nd["rest_deg"])
        c = np.array(nd["axis_point"], dtype=float)
        Gf = self.Gf
        r = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), math.degrees(delta)))
        w = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*(-c))) * r * Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*c))
        nd["op"].Set(nd["parent"] * w * nd["parent_inv"])

    # -- the frames -----------------------------------------------------------------------------------------------
    def process(self, i, s, arr, info, m):
        """One sensor's read for capture m -> its delivered frame (the stated model and noise)."""
        sm = s["stream"]
        rng = self.wire_obs.frame_rng(self.seed, s["index"], m)
        if s["kind"] == "rgb":
            return arr
        if s["kind"] == "thermal":
            ids = np.asarray(arr).reshape(sm["height"], sm["width"]).astype(np.int64)
            lab = (info or {}).get("idToLabels") or {}
            hot_prefix = self.th["hot_prim"]
            top = int(ids.max()) if ids.size else 0
            lut = np.full(max(top, max((int(k) for k in lab), default=0)) + 1, float(self.th["ambient_k"]), dtype=np.float32)
            pan_k = self.rec.get(m, {}).get("pan_k", float(self.th["burning_k"]))
            for k_, v in lab.items():
                if isinstance(v, str) and (v == hot_prefix or v.startswith(hot_prefix + "/")):
                    lut[int(k_)] = pan_k
            t = lut[ids]
            return t + float(sm["noise"]["sigma_k"]) * rng.standard_normal(t.shape, dtype=np.float32)
        if s["kind"] == "depth":
            z = np.asarray(arr, dtype=np.float32).reshape(sm["height"], sm["width"])
            lo, hi = sm["range_m"]
            ok = np.isfinite(z) & (z >= lo) & (z <= hi)
            z = np.where(ok, z, 0.0).astype(np.float32)
            zn = depth_noise(z, sm["noise"], rng)
            return np.where(ok, zn, 0.0).astype(np.float32)
        if s["kind"] == "lidar":
            ri, ci = s["grid"]
            img = np.asarray(arr, dtype=np.float32)
            img = img.reshape(img.shape[0], img.shape[1]) if img.ndim == 3 else img
            r = img[np.ix_(ri, ci)] + s["near"]  # (the renderer's distance_to_camera runs from the near clipping plane)
            lo, hi = sm["range_m"]
            ok = np.isfinite(r) & (r >= lo) & (r <= hi)
            rn = r + float(sm["noise"]["sigma_m"]) * rng.standard_normal(r.shape, dtype=np.float32)
            return np.where(ok, rn, 0.0).astype(np.float32)
        raise ValueError(s["kind"])

    def render(self):
        """One capture: the orchestrator's step (every product's read then shows it), or a plain app update."""
        t0 = time.perf_counter()
        if self.read_mode == "orchestrator":
            self.settings.set_bool("/app/player/playSimulations", False)  # (as SimulationContext.render does: no physics)
            try:
                self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            finally:
                self.settings.set_bool("/app/player/playSimulations", True)
        else:
            self.ctx.render()
        return 1000.0 * (time.perf_counter() - t0)

    def warm_up(self, max_updates=400):
        """Before the wire opens: pose everything at t = 0, render until two consecutive reads are full, the clock
        shows that render or the one before and some RGB frame is not black (the renderer warm; a camera of the fleet's that
        sits inside geometry may stay black: that is its view). The first render on a cold cache compiles here, on the
        set-up's time."""
        t0 = time.time()
        good, k, first, why = 0, 0, None, None
        for k in range(1, max_updates + 1):
            n = 32 + (k % 32)  # (the warm-up's own clock values, away from the run's first ones)
            self.pose_all(n)
            ms = self.render()
            first = ms if first is None else first
            reads = self.P.read()
            dec = self._clock_decode(reads[-1][0])
            want = n
            ok = dec is not None and (n - dec) % self.clock_ck["modulo"] in (0, 1)  # (fresh or one render behind)
            rgb_lit = rgb_any = False
            for (arr, _), s in zip(reads[:-1], self.sens):
                sm = s["stream"]
                if arr is None or not getattr(arr, "size", 0):
                    ok = False
                elif s["kind"] == "rgb":
                    rgb_any = True
                    if np.asarray(arr).ndim != 3 or arr.shape[0] != sm["height"] or arr.shape[1] != sm["width"]:
                        ok = False
                    elif int(np.asarray(arr)[..., :3].max()) > 0:
                        rgb_lit = True
            if rgb_any and not rgb_lit:
                ok = False
            why = dict(clock=dec, want=None if want is None else want % self.clock_ck["modulo"], rgb_lit=rgb_lit,
                       empty=[s["id"] for (arr, _), s in zip(reads[:-1], self.sens) if arr is None or not getattr(arr, "size", 0)])
            good = good + 1 if ok else 0
            if good >= 2:
                break
        self.warm = dict(updates=k, first_render_ms=round(first or 0.0, 1), s=round(time.time() - t0, 2), ok=good >= 2,
                         last=why if good < 2 else None)
        return self.warm["ok"]

    def observe_step(self, time_s, fire):
        """Every step (the thermal model's clock): the first observation in which the fire is out."""
        if not fire and self.t_fire_out is None:
            self.t_fire_out = float(time_s)

    def step(self, k, time_s, fire):
        """The probe's frame step at control step k (call at every step; it acts at k = 0 mod every). Returns the
        latest frame set delivered ({n, t, time, slot}) or None.

        At capture step 6n the read shows the capture the frame clock decodes, m: n (fresh: the orchestrator's read for
        the reference fleet in P2) or n - 1 (one render behind: the orchestrator's read for a fleet streaming 21
        products in P1). It is processed as capture m unless m was processed already, and the frame set due now, n - 1,
        is delivered from what was processed: this step's read (m = n - 1) or the previous step's (m = n then). A frame
        set that is not there when it is due (the clock older, or not decoded) is an anomaly: the newest one that is
        newer than the last delivered goes out instead, stamped with its own capture (`lag_anomalies`)."""
        self.observe_step(time_s, fire)
        if k % self.every:
            return self.delivered
        n = k // self.every
        self.rec[n] = dict(t=int(k), time=float(time_s), pan_k=thermal_temperature(fire, time_s, self.t_fire_out, self.th))
        t0 = time.perf_counter()
        self.pose_all(n)
        t1 = time.perf_counter()
        r_ms = self.render()
        if n == 0:  # (capture 0 rendered twice: a read one render behind then shows it, not the warm-up)
            r_ms += self.render()
        t2 = time.perf_counter()
        reads = self.P.read()
        t3 = time.perf_counter()
        dec = self._clock_decode(reads[-1][0])
        st = self.stats
        st["pose_ms"].append(1000.0 * (t1 - t0))
        st["render_ms"].append(r_ms)
        st["read_ms"].append(1000.0 * (t3 - t2))
        mod = self.clock_ck["modulo"]
        m = None
        if dec is not None:
            cands = [c for c in self.rec if c % mod == dec and c <= n]
            m = max(cands) if cands else None
        st["lag_seen"][str(None if m is None else n - m)] = st["lag_seen"].get(str(None if m is None else n - m), 0) + 1
        if m is not None and m > self.last_m:  # (a capture not processed yet)
            cap = self.rec.get(m) or dict(t=m * self.every, time=None)
            frames = [self.process(i, s, arr, info, m) for i, ((arr, info), s) in enumerate(zip(reads[:-1], self.sens))]
            self.ready[m] = (cap, frames)
            self.last_m = m
        t4 = time.perf_counter()
        st["cpu_ms"].append(1000.0 * (t4 - t3))
        if self.bench:
            self._bench(n, reads)
        due = n - 1
        if due < 0:
            return self.delivered
        last = self.delivered["n"] if self.delivered else -1
        have = [c for c in self.ready if last < c <= due]
        if due not in self.ready:
            st["lag_anomalies"] += 1
            if len(st.setdefault("anomalies", [])) < 20:
                st["anomalies"].append([int(n), dec, m])
        if not have:
            st["stale"] += 1
            return self.delivered
        c = max(have)
        cap, frames = self.ready[c]
        slot = c % int(self.cfg["slots"])
        nbytes = 0
        for i, (s, fr) in enumerate(zip(self.sens, frames)):
            self.slots[i][slot].write(c, cap["t"], cap["time"] if cap["time"] is not None else 0.0, fr)
            nbytes += self.slots[i][slot].nbytes
        t5 = time.perf_counter()
        for i, s in enumerate(self.sens):  # (a stride-sampled sum of each delivered frame, rolled into one digest per sensor)
            flat = self.slots[i][slot].payload.reshape(-1)
            stride = max(1, flat.size // 4096)
            self.digest[s["id"]].update(struct.pack("<Qd", c, float(flat[::stride].astype(np.float64).sum())))
        st["write_ms"].append(1000.0 * (t5 - t4))
        st["frame_sets"] += 1
        st["bytes"] = nbytes
        self.delivered = {"n": int(c), "t": int(cap["t"]), "time": cap["time"], "slot": int(slot)}
        if self.export is not None:
            self.export.offer(c, cap, self.sens, frames)
        for old in [x for x in self.ready if x <= c]:
            del self.ready[old]
        for old in [x for x in self.rec if x < c - 8]:
            del self.rec[old]
        return self.delivered

    # -- a maintainer's bench (`sensors.bench` in the probe's config; the grader never sets it) ----------------------
    def _bench(self, n, reads):
        """lag_check_every N: is the read's lag the same in every product? At captures N k - 1 and N k the same state
        is rendered twice more after the ordinary read; the last of three renders of one state reads fresh for any lag
        up to 2 (C). Each depth-like product's ordinary read at N k (A) is then the fresh one (A = C_Nk), one capture
        behind (A = C_Nk-1) or neither, against the clock's own lag. lidar_check_at [captures]: each lidar's noise-free
        ranges against PhysX ray casts that hit a static collider of the real bay."""
        b = self.bench
        lc = b.setdefault("_lag", dict(points=0, clock_lag={}, product_lag={}, mixed=0, c_fresh=0))
        every = int(b.get("lag_check_every") or 0)
        mod = self.clock_ck["modulo"]

        def fresh_read():
            self.render()
            self.render()
            rc = self.P.read()
            lc["c_fresh"] += int(self._clock_decode(rc[-1][0]) == n % mod)
            return [np.array(a) if a is not None else None for a, _ in rc]
        if every and n >= 2 and (n + 1) % every == 0:
            b["_prev"] = (n, fresh_read())
        elif every and n >= 2 and n % every == 0 and b.get("_prev") and b["_prev"][0] == n - 1:
            a_reads = [np.array(a) if a is not None else None for a, _ in reads]
            dec = self._clock_decode(reads[-1][0])
            c_now = fresh_read()
            c_prev = b["_prev"][1]
            lc["points"] += 1
            clag = None if dec is None else (n - dec) % mod
            lc["clock_lag"][str(clag)] = lc["clock_lag"].get(str(clag), 0) + 1
            for a, cn, cp, s in zip(a_reads[:-1], c_now[:-1], c_prev[:-1], self.sens):
                if s["kind"] == "rgb" or a is None:
                    continue
                if cn is not None and cp is not None and np.array_equal(cn, cp):
                    lag = "static"
                elif cn is not None and np.array_equal(a, cn):
                    lag = "0"
                elif cp is not None and np.array_equal(a, cp):
                    lag = "1"
                else:
                    lag = "other"
                key = f"{s['kind']}:{lag}"
                lc["product_lag"][key] = lc["product_lag"].get(key, 0) + 1
                if lag in ("0", "1") and clag is not None and lag != str(clag):
                    lc["mixed"] += 1
            b["_prev"] = None
        if self.delivered and self.delivered["n"] in set(b.get("lidar_check_at") or []):
            b.setdefault("_lidar", []).append(self._lidar_check(self.delivered["n"], reads))

    def _lidar_check(self, m, reads):
        from omni.physx import get_physx_scene_query_interface
        sq = get_physx_scene_query_interface()
        out = []
        for (arr, _), s in zip(reads[:-1], self.sens):
            if s["kind"] != "lidar":
                continue
            sm = s["stream"]
            ri, ci = s["grid"]
            img = np.asarray(arr, dtype=np.float32)
            img = img.reshape(img.shape[0], img.shape[1]) if img.ndim == 3 else img
            r = img[np.ix_(ri, ci)] + s["near"]
            bp, bq = self.bodies(s["stage_body"])
            o = bp + rotm(bq) @ np.array(s["pos_in_body"])
            R = rotm(qmul(bq, np.array(s["quat_in_body"])))
            e = np.radians(beams(sm)[1])
            a = 2 * np.pi * np.arange(sm["width"]) / sm["width"]
            errs, n_hit, n_both, n_static = [], 0, 0, 0
            for c in range(0, sm["height"], max(1, sm["height"] // 8)):
                for k in range(0, sm["width"], 8):
                    d = R @ np.array([math.cos(e[c]) * math.cos(a[k]), math.cos(e[c]) * math.sin(a[k]), math.sin(e[c])])
                    h = sq.raycast_closest(tuple(float(v) for v in o), tuple(float(v) for v in d), float(sm["range_m"][1]))
                    if not h or not h.get("hit"):
                        continue
                    n_hit += 1
                    col = str(h.get("collision") or h.get("rigidBody") or "")
                    if not col.startswith("/World/Facility/") or col.startswith("/World/Facility/Items"):
                        continue
                    n_static += 1
                    if np.isfinite(r[c, k]) and r[c, k] > 0:
                        n_both += 1
                        errs.append(float(r[c, k]) - float(h["distance"]))
            ae = np.abs(np.array(errs)) if errs else np.zeros(1)
            out.append(dict(sensor=s["id"], capture=m, rays_hit=n_hit, static=n_static, both=n_both,
                            median_m=round(float(np.median(ae)), 4), p95_m=round(float(np.percentile(ae, 95)), 4), max_m=round(float(ae.max()), 4),
                            within_2cm=round(float(np.mean(ae <= 0.02)), 4), mean_signed_m=round(float(np.mean(errs)) if errs else 0.0, 4)))
        return out

    def summary(self):
        st = self.stats

        def mp(v):
            return dict(mean=round(float(np.mean(v)), 2), p90=round(float(np.percentile(v, 90)), 2), max=round(float(np.max(v)), 1)) if v else None
        return dict(sensors=len(self.sens), products=len(self.P.items), frame_sets=st["frame_sets"], render_ms=mp(st["render_ms"]),
                    read_ms=mp(st["read_ms"]), cpu_ms=mp(st["cpu_ms"]), write_ms=mp(st["write_ms"]), pose_ms=mp(st["pose_ms"]),
                    bytes_per_set=st["bytes"], lag_anomalies=st["lag_anomalies"], lag_seen=st["lag_seen"], anomalies=st.get("anomalies", []),
                    stale=st["stale"], read_mode=self.read_mode,
                    digests={k: v.hexdigest() for k, v in self.digest.items()}, warm_up=getattr(self, "warm", None), create_s=self.create_s,
                    pinned=self.P.pinned, absent=sorted(self.P.absent), product_rtx=self.P.first_attrs, annotator_retries=self.P.retries,
                    bench={k: v for k, v in self.bench.items() if k in ("_lag", "_lidar")} or None)

    def close(self):
        for row in self.slots or []:
            for sl in row:
                sl.close()


class Export:
    """A maintainer's frame export (`review/run_mission.py --frames-out`; the grader never sets it): chosen frame
    sets copied from the delivered frames into the probe's own folder, each with the true state at its capture step.
    `plan`: every (every Nth frame set), windows_s ([[t0, t1], ...]: every frame set in them), events ({name: [before_s,
    after_s]} around the moments `event_times()` reports), between ([event a, event b, N]: every Nth frame set from a's
    first moment to b's, open while b has not come), sensors (ids, "all", or a selector {robot: [...], model: [...],
    prim_contains: [...]}, each key optional). `groups`: a list of such plans, each with its own sensors; a frame set is
    kept with the union of the sensors its groups want."""

    def __init__(self, out_dir, plan_, event_times=None, truth_at=None, own_at=None):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.plan = plan_ or {}
        self.groups = list(self.plan.get("groups") or [self.plan])
        self.event_times = event_times or (lambda: {})
        self.truth_at = truth_at or (lambda m: None)
        self.own_at = own_at or (lambda m: None)  # (the robots' own numbers at the capture step: what a program has)
        self.buf = []  # (m, time, frames) of the last frame sets, for the windows before an event
        self.saved = set()
        self.index = open(os.path.join(out_dir, "index.jsonl"), "a")
        self.n = 0

    def _sel(self, g, s):
        ids = g.get("sensors", self.plan.get("sensors"))
        if not ids or ids == "all":
            return True
        if isinstance(ids, dict):
            return ((not ids.get("robot") or s["robot"] in ids["robot"]) and (not ids.get("model") or s["model"] in ids["model"])
                    and (not ids.get("prim_contains") or any(x in s["prim"] for x in ids["prim_contains"])))
        return s["id"] in ids

    def _want_group(self, g, m, t, ev):
        if g.get("every") and m % int(g["every"]) == 0:
            return True
        if any(a <= t <= b for a, b in g.get("windows_s") or []):
            return True
        for name, (before, after) in (g.get("events") or {}).items():
            for te in ev.get(name) or []:
                if te - before <= t <= te + after:
                    return True
        if g.get("between"):
            a, b, k = g["between"]
            ta, tb = (ev.get(a) or [None])[0], (ev.get(b) or [None])[0]
            if ta is not None and t >= ta and (tb is None or t <= tb) and m % int(k) == 0:
                return True
        return False

    @staticmethod
    def _copy(f):
        a = np.asarray(f)
        return np.ascontiguousarray(a[..., :3]) if a.ndim == 3 and a.shape[2] == 4 and a.dtype == np.uint8 else np.array(a)  # (RGB as delivered)

    def offer(self, m, cap, sens, frames):
        t = cap["time"] if cap["time"] is not None else 0.0
        self.buf.append((m, t, cap, [(s, self._copy(f)) for s, f in zip(sens, frames) if any(self._sel(g, s) for g in self.groups)]))
        horizon = max([0.0] + [float(b) for g in self.groups for b, _ in (g.get("events") or {}).values()])
        while self.buf and self.buf[0][1] < t - horizon - 0.2:
            self.buf.pop(0)
        ev = self.event_times()
        for bm, bt, bcap, bfr in list(self.buf):
            if bm in self.saved:
                continue
            want = [g for g in self.groups if self._want_group(g, bm, bt, ev)]
            if not want:
                continue
            keep = [(s["id"], f) for s, f in bfr if any(self._sel(g, s) for g in want)]
            for sid, f in keep:
                np.save(os.path.join(self.dir, f"{bm:06d}_{sid}.npy"), f)
            self.index.write(json.dumps(dict(n=bm, t=bcap["t"], time=bt, sensors=[sid for sid, _ in keep], own=self.own_at(bm),
                                             truth=self.truth_at(bm))) + "\n")
            self.index.flush()
            self.saved.add(bm)
            self.n += 1
