"""RenderBackend: the one interface every renderer implements; RenderSession: its live-frame (policy-rate) form.

A backend implements ONLY `render_raw` (ideal pinhole images of the raw cameras) plus `version()` / `capabilities()`,
and, if it can serve live cameras, `open(scene) -> RenderSession` with `capabilities().sessions = True`.
`render()` and `RenderSession.render()` are final in spirit: they plan the raw cameras (postprocess.plan_view), call
the backend's raw renderer, run the shared post-processing (postprocess.finish_view, incl. the depth sensor) and write
the provenance record, so post-processing is structurally the same after every backend and a new backend cannot
forget it.

What `render_raw(request, raw_cameras)` (and `RenderSession.render_raw_views(raw_cameras, outputs)`) must return
(RawRender / a list of RawView), per raw camera, in the order given:

    rgb    uint8 (H, W, 3), display-referred sRGB, exactly raw_camera.width x raw_camera.height, OpenCV pixel grid
           (row 0 = top), ideal pinhole with raw_camera's fx fy cx cy and world_T_cam; no distortion, noise, blur,
           exposure or white balance (postprocess applies them), unless capabilities().native_lens_distortion, in which
           case the raw cameras keep their k1..k3 / p1 p2 and the backend distorts itself
    depth  float32 (H, W) metres along the optical axis, 0.0 where nothing was hit   (only if "depth" was requested)
    seg    int32 (H, W) ids of RawRender.labels, -1 background                        (only if "segmentation" requested)

and `labels` {id: {"geom": str, "body": str, "scale_ref": str | None}}, `resolved_scene_digest` (a digest of the scene
as the backend resolved it, i.e. after SceneSpec.lod), `lod` (the LOD record, None for lod "full"), `environment`
(versions / devices that can change pixels) and free-form `facts`.  The backend never sees "sensor_depth": the layer
asks it for "depth" instead (raw_outputs) and derives the sensor stream itself.
Determinism: the same request on the same software stack and device must give the same bytes.

Session contract (RA4; a live-camera task calls require_capabilities(backend, sessions=True, outputs=[...]) first):

    session = backend.open(scene)          compile / load the scene once (SceneSpec incl. lod, shadows, overrides)
    session.set_state(StateSpec | qpos=... | joints={...})          or
    session.sync_from(model, data)         copy the grader's physics configuration by joint name (hinge / slide / ball /
                                           free), then forward kinematics of the render model incl. the poses of
                                           cameras and lights riding on bodies (MuJoCo: mj_kinematics + mj_comPos + mj_camlight)
    session.render(cameras, photo=PhotoSpec(), seed=None, outputs=("rgb",), keep_raw=False, encode_png=True)
                                           -> RenderResult, the same pipeline and bytes as backend.render(RenderRequest(
                                           scene, cameras, photo, seed, outputs)) at the same state.  seed = the frame's
                                           SECRET seed, derive_seed(<OS-entropy episode seed>, frame): the noise fields are
                                           a public function of (seed, code).  Omitted: 0, with a RuntimeWarning if the photo
                                           is stochastic; sensor_depth from a stochastic DepthSensorSpec refuses seed 0.
                                           RenderResult.agent_view() = the fields that may reach an agent.
    session.render_raw_views(raw_cameras, outputs) -> [RawView]      the backend part (what subclasses implement)
    session.resolved_digest(), session.lod_record(), session.close()  (+ context manager)
"""
from __future__ import annotations

import abc
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import postprocess
from ..provenance import library_environment, request_digest
from ..spec import RENDER_LAYER_VERSION, CameraSpec, PhotoSpec, Provenance, RenderRequest, RenderResult, SceneSpec, _Spec, sha256_hex


@dataclass
class BackendCapabilities(_Spec):
    name: str
    implemented: bool
    outputs: List[str] = field(default_factory=lambda: ["rgb"])
    native_lens_distortion: bool = False
    deterministic: str = "bitwise on the same software stack and device"
    sessions: bool = False                 # persistent scene + GL context for policy-rate rendering (RenderBackend.open)
    max_image_px: int = 0                  # 0 = no declared limit
    notes: str = ""
    # ---- added after 1.0.0
    one_pass_rgbd: bool = False            # rgb + depth of a camera from ONE render pass (RA2)
    lod_levels: List[str] = field(default_factory=lambda: ["full"])     # SceneSpec.lod values the backend honours (RA3)

    _OMIT_IF_DEFAULT = frozenset({"one_pass_rgbd", "lod_levels"})

    def provides(self, output: str) -> bool:
        """sensor_depth is derived by the layer from depth, so a backend provides it iff it provides depth."""
        return output in self.outputs or (output == "sensor_depth" and "depth" in self.outputs)


@dataclass
class RawView:
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    seg: Optional[np.ndarray] = None


@dataclass
class RawRender:
    views: List[RawView]
    labels: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    resolved_scene_digest: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    lod: Optional[Dict[str, Any]] = None     # the backend's LOD record (Provenance.lod); None = full meshes


def raw_outputs(outputs: Sequence[str]) -> List[str]:
    """The outputs a backend is asked for: sensor_depth -> depth (the layer applies the sensor model)."""
    out = [o for o in outputs if o != "sensor_depth"]
    if "sensor_depth" in outputs and "depth" not in out:
        out.append("depth")
    return out


class CapabilityError(RuntimeError):
    pass


def require_capabilities(backend: "RenderBackend", sessions: bool = False, outputs: Sequence[str] = (), lod: Optional[str] = None,
                         one_pass_rgbd: bool = False) -> "BackendCapabilities":
    """Fail early and clearly when a backend cannot serve a task (live-camera tasks: sessions=True)."""
    caps = backend.capabilities()
    missing = []
    if not caps.implemented:
        missing.append("implemented")
    if sessions and not caps.sessions:
        missing.append("sessions")
    missing += ["output %s" % o for o in outputs if not caps.provides(o)]
    if lod is not None and lod not in caps.lod_levels:
        missing.append("lod %s" % lod)
    if one_pass_rgbd and not caps.one_pass_rgbd:
        missing.append("one_pass_rgbd")
    if missing:
        raise CapabilityError("render backend %s (%s) lacks %s; a live-camera task needs a backend with sessions "
                              "(render/backends/base.py, 'Session contract')" % (backend.name, caps.notes or "no notes", ", ".join(missing)))
    return caps


class RenderBackend(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def version(self) -> str:
        """Backend id + the versions that define its pixels, e.g. 'mujoco-backend/1.0.0 mujoco-3.14.0 gl-egl'."""

    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        ...

    @abc.abstractmethod
    def render_raw(self, request: RenderRequest, raw_cameras: List[CameraSpec]) -> RawRender:
        """Ideal pinhole images of `raw_cameras` of request.scene (module docstring for the exact contract)."""

    def open(self, scene: SceneSpec) -> "RenderSession":
        """A live-frame session on `scene` (module docstring, 'Session contract').  Backends with
        capabilities().sessions = False raise NotImplementedError (the stubs document their session path)."""
        raise NotImplementedError("render backend %r has no sessions (capabilities().sessions = False); live-camera tasks "
                                  "need RenderBackend.open (render/backends/base.py)" % self.name)

    def environment(self, session: Optional["RenderSession"] = None) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ the shared pipeline
    def plan(self, request: RenderRequest) -> List[postprocess.ViewPlan]:
        caps = self.capabilities()
        return [postprocess.plan_view(c, request.photo, caps.native_lens_distortion) for c in request.cameras]

    def _check(self, request: RenderRequest) -> None:
        if request.backend != self.name:
            raise ValueError("request names backend %r but this is %r: a photo set is tied to its backend (a different "
                             "backend is a new task version, render/README.md)" % (request.backend, self.name))
        caps = self.capabilities()
        missing = [o for o in request.outputs if not caps.provides(o)]
        if missing:
            raise ValueError("backend %s cannot produce %s" % (self.name, missing))
        if request.scene.lod not in caps.lod_levels:
            raise ValueError("backend %s cannot render SceneSpec.lod %r (supports %s)" % (self.name, request.scene.lod, caps.lod_levels))

    def render(self, request: RenderRequest, keep_raw: bool = False) -> RenderResult:
        self._check(request)
        t0 = time.perf_counter()
        plans = self.plan(request)
        raw = self.render_raw(raw_request(request), [p.raw_camera for p in plans])
        t1 = time.perf_counter()
        return self.finish(request, plans, raw, keep_raw=keep_raw, t_raw=t1 - t0)

    def finish(self, request: RenderRequest, plans, raw: RawRender, keep_raw=False, t_raw=None, encode_png=True,
               scene_content=None) -> RenderResult:
        t1 = time.perf_counter()
        if len(raw.views) != len(plans):
            raise ValueError("backend %s returned %d views for %d cameras" % (self.name, len(raw.views), len(plans)))
        want_sd = "sensor_depth" in request.outputs
        fin = [postprocess.finish_view(v.rgb, p, request.photo, request.seed, i, v.depth, v.seg, encode_png=encode_png,
                                       sensor_depth=want_sd)
               for i, (v, p) in enumerate(zip(raw.views, plans))]
        t2 = time.perf_counter()
        env = library_environment()
        env.update(raw.environment)
        prov = Provenance(backend=self.name, backend_version=self.version(), render_layer_version=RENDER_LAYER_VERSION,
                          request_digest=request_digest(request, scene_content), image_sha256=[sha256_hex(f.encoded) for f in fin],
                          resolved_scene_digest=raw.resolved_scene_digest, environment=env,
                          postprocess_version=postprocess.POSTPROCESS_VERSION, lod=raw.lod,
                          sensor_depth_sha256=[sha256_hex(f.sensor_depth_encoded) for f in fin] if want_sd else None,
                          sensor_depth_model=postprocess.DEPTH_SENSOR_VERSION if want_sd else None)
        want_d = "depth" in request.outputs
        want_s = "segmentation" in request.outputs
        facts = dict(raw.facts)
        if t_raw is not None:
            facts["t_backend_s"] = round(t_raw, 4)
        facts["t_postprocess_s"] = round(t2 - t1, 4)
        return RenderResult(rgb=[f.rgb for f in fin], encoded=[f.encoded for f in fin], image_format=request.photo.image_format,
                            provenance=prov, depth=[f.depth for f in fin] if want_d else None,
                            segmentation=[f.segmentation for f in fin] if want_s else None, segmentation_labels=raw.labels,
                            raw_rgb=[v.rgb for v in raw.views] if keep_raw else None, facts=facts,
                            sensor_depth=[f.sensor_depth for f in fin] if want_sd else None,
                            sensor_depth_encoded=[f.sensor_depth_encoded for f in fin] if want_sd else None)


def raw_request(request: RenderRequest) -> RenderRequest:
    """The request as the backend's render_raw sees it: outputs mapped by raw_outputs (sensor_depth -> depth)."""
    outs = raw_outputs(request.outputs)
    if outs == list(request.outputs):
        return request
    return RenderRequest(scene=request.scene, cameras=request.cameras, photo=request.photo, seed=request.seed,
                         backend=request.backend, outputs=outs, label=request.label)


class RenderSession(abc.ABC):
    """A persistent scene of one backend for live frames (module docstring, 'Session contract').  Subclasses implement
    set_state / sync_from / render_raw_views / close (+ resolved_digest, lod_record); render() is shared."""

    backend: RenderBackend
    scene: SceneSpec
    labels: Dict[int, Dict[str, Any]]
    scene_content: Optional[Dict[str, str]] = None      # scene file digests, fixed while the session is open

    @abc.abstractmethod
    def set_state(self, state=None, qpos=None, joints=None) -> None:
        ...

    @abc.abstractmethod
    def sync_from(self, model, data) -> None:
        ...

    @abc.abstractmethod
    def render_raw_views(self, cameras: Sequence[CameraSpec], outputs=("rgb",)) -> List[RawView]:
        """Raw views of the RAW cameras at the current state; outputs may contain sensor_depth (treated as depth)."""

    @abc.abstractmethod
    def resolved_digest(self) -> str:
        ...

    def lod_record(self) -> Optional[Dict[str, Any]]:
        return None

    def facts(self) -> Dict[str, Any]:
        return {"session": True}

    def render(self, cameras: Sequence[CameraSpec], photo: Optional[PhotoSpec] = None, seed: Optional[int] = None,
               outputs=("rgb",), keep_raw=False, encode_png=True) -> RenderResult:
        """The full pipeline on the session's current state (no recompilation): policy-rate rendering.  encode_png=False
        skips writing PNG bytes (RenderResult.encoded / sensor_depth_encoded hold b"", image_sha256 hashes that) when
        only arrays are needed.  `seed`: the frame's secret seed (module docstring); None = 0 (the 1.0.0 default), with a
        RuntimeWarning when the photo is stochastic (every frame would carry the same, publicly reproducible noise)."""
        photo = photo or PhotoSpec()
        if seed is None:
            if "sensor_depth" in outputs and photo.depth_sensor is not None and photo.depth_sensor.is_stochastic:
                raise ValueError("sensor_depth from a stochastic DepthSensorSpec needs the frame's secret seed (seed 0 = a public "
                                 "noise field): pass seed=derive_seed(<OS-entropy episode seed>, frame)")
            if photo.is_stochastic:
                warnings.warn("RenderSession.render without a seed: every frame gets the noise field of seed 0, reproducible "
                              "from the render code; pass seed=derive_seed(<OS-entropy episode seed>, frame)", RuntimeWarning,
                              stacklevel=2)
            seed = 0
        req = RenderRequest(scene=self.scene, cameras=list(cameras), photo=photo, seed=seed,
                            backend=self.backend.name, outputs=list(outputs))
        self.backend._check(req)
        t0 = time.perf_counter()
        plans = self.backend.plan(req)
        views = self.render_raw_views([p.raw_camera for p in plans], raw_outputs(req.outputs))
        raw = RawRender(views=views, labels=self.labels, resolved_scene_digest=self.resolved_digest(),
                        environment=self.backend.environment(self), facts=self.facts(), lod=self.lod_record())
        return self.backend.finish(req, plans, raw, keep_raw=keep_raw, t_raw=time.perf_counter() - t0, encode_png=encode_png,
                                   scene_content=self.scene_content)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
