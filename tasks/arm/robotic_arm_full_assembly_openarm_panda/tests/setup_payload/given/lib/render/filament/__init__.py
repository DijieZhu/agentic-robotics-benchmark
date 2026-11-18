"""Filament photo backend for the renderer-agnostic photo layer (`render`); see backend.py.

    import render_filament; render_filament.register()      # get_backend("filament") / PhotoSetSpec(backend="filament")
"""
from .backend import BACKEND_VERSION, FilamentBackend, Look, apply_look, compile_with_look, register  # noqa: F401
