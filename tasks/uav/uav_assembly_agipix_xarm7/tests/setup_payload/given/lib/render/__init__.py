"""arb render layer: renderer-agnostic agent-visible images (render/README.md).

    from render import spec, scale_refs, photo_set, postprocess, qc, lod
    from render.backends import get_backend        # "mujoco" (implemented), "isaac" / "neural" (documented stubs)

Rule: no task calls the MuJoCo renderer directly for agent-visible images; they go through a RenderRequest and a
backend (or a backend's RenderSession for live frames), and a photo set regenerated with another backend (or a backend
version that changes pixels) is a new task version recorded in the provenance.
"""
from .spec import (AGENT_VISIBLE_OUTPUTS, GRADER_ONLY_OUTPUTS, LOD_LEVELS, OUTPUTS, RENDER_LAYER_VERSION, BackgroundSpec, BlurSpec,
                   CameraSpec, ColourJitterSpec, DepthSensorSpec, GeomOverride, HeadlightSpec, LightSpec, MaterialOverride, NoiseSpec,
                   PhotoSpec, Provenance, RenderRequest, RenderResult, ScaleRef, SceneSpec, StateSpec, TextureSpec, canonical_json,
                   derive_seed, new_seed)

__all__ = ["AGENT_VISIBLE_OUTPUTS", "GRADER_ONLY_OUTPUTS", "LOD_LEVELS", "OUTPUTS", "RENDER_LAYER_VERSION", "BackgroundSpec",
           "BlurSpec", "CameraSpec", "ColourJitterSpec", "DepthSensorSpec", "GeomOverride", "HeadlightSpec", "LightSpec",
           "MaterialOverride", "NoiseSpec", "PhotoSpec", "Provenance", "RenderRequest", "RenderResult", "ScaleRef", "SceneSpec",
           "StateSpec", "TextureSpec", "canonical_json", "derive_seed", "new_seed"]
