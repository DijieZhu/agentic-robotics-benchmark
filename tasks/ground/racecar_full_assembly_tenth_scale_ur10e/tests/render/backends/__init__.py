"""Backend registry.  get_backend("mujoco" | "isaac" | "neural", **kw) -> RenderBackend; register_backend(name, factory).

Backends are imported lazily so importing the render layer never pulls in mujoco / a ray-traced renderer / a neural runtime.
"""
from __future__ import annotations

from typing import Callable, Dict

from .base import (BackendCapabilities, CapabilityError, RawRender, RawView, RenderBackend, RenderSession, raw_outputs,
                   require_capabilities)

_FACTORIES: Dict[str, Callable[..., RenderBackend]] = {}


def register_backend(name: str, factory: Callable[..., RenderBackend]) -> None:
    _FACTORIES[name] = factory


def _mujoco(**kw):
    from .mujoco_backend import MujocoBackend
    return MujocoBackend(**kw)


def _isaac(**kw):
    from .isaac_backend import RTSimBackend
    return RTSimBackend(**kw)


def _neural(**kw):
    from .neural_backend import NeuralPassBackend
    return NeuralPassBackend(**kw)


register_backend("mujoco", _mujoco)
register_backend("isaac", _isaac)
register_backend("neural", _neural)


def get_backend(name: str, **kw) -> RenderBackend:
    if name not in _FACTORIES:
        raise KeyError("unknown render backend %r (registered: %s)" % (name, sorted(_FACTORIES)))
    return _FACTORIES[name](**kw)


def available_backends() -> Dict[str, bool]:
    """{name: implemented}."""
    out = {}
    for n in sorted(_FACTORIES):
        try:
            out[n] = bool(get_backend(n).capabilities().implemented)
        except Exception:  # noqa: BLE001
            out[n] = False
    return out


__all__ = ["BackendCapabilities", "CapabilityError", "RawRender", "RawView", "RenderBackend", "RenderSession", "available_backends",
           "get_backend", "raw_outputs", "register_backend", "require_capabilities"]
