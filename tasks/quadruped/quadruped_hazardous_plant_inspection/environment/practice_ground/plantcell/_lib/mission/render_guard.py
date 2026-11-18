"""Guarded offscreen rendering for the cell's sensors.

What went wrong (an early verify): in the 6th and 7th cell built
in one verify process, the thermal camera's segmentation render returned a shaded colour image instead of segment ids
(`IndexError: index 13841584 = 0xD33470 is out of bounds for axis 0 with size 2380` inside mujoco's Renderer). That
cell (a test stub over the source tree's world) never closed its renderers. The mechanism, reproduced by
`tests/review/cell/test_render_cells.py --lifecycle late-release`: mujoco.Renderer.close() (also run by its __del__)
frees its OpenGL context FIRST and its MjrContext (framebuffers, render buffers, textures, display lists) AFTER, so the
GL deletions run in whatever context is current at that moment. When an old cell's renderer is finalised while a
live cell's renderer is current, the live renderer loses its offscreen framebuffer and textures (the object names are
the same numbers in every context); its next render draws nowhere and the read-back returns a stale colour image. A
finaliser can also run between a live renderer's make_current() and its read-back (the cyclic garbage collector runs
on any allocation), which leaves no context current for the render.

The guard, used for every renderer of a mission world (world.MissionWorld.pool):
  * SafeRenderer.close() makes the renderer's OWN context current, frees its MjrContext there, then frees the
    context: closing (or finalising) a renderer never touches another context;
  * every render and every renderer construction runs with the cyclic garbage collector paused, so no finaliser can
    run between make_current() and the read-back;
  * every segmentation render is validated against the scene it rendered: each pixel's raw id must be 0 (background)
    or the segment id of a geom in that scene, and every geom-type id must be a geom of the model. On a failure the
    renderer rebuilds its own GL and MjrContext and renders the same scene once more; if that fails too it raises
    RenderFailure (the server answers `sensor_unavailable`, counts a lost frame, and the grader runs the mission again).

A valid segmentation is bit-identical to mujoco.Renderer.render()'s (the same flags, the same read-back, the same
mapping and the same vertical flip; tests/review/cell/test_render_cells.py --parity), so the frames stay deterministic.
The extra cost is the validation of the raw ids (a lookup per pixel) in place of mujoco's own mapping.
"""

from __future__ import annotations

import contextlib
import gc

import mujoco
import numpy as np

from . import paths

paths.setup()

from bricks.sensors import camera as bcam  # noqa: E402

try:                                         # the classic renderer's context class (osmesa / egl / glfw by MUJOCO_GL)
    from mujoco.rendering.classic import gl_context as _glc
except ImportError:                          # pragma: no cover - older bindings
    _glc = None

__all__ = ["RenderFailure", "SafeRenderer", "GuardedRenderPool", "gc_paused", "validate_segmentation"]

_SEG = mujoco.mjtRndFlag.mjRND_SEGMENT
_IDC = mujoco.mjtRndFlag.mjRND_IDCOLOR
_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
MAX_INCIDENTS = 20


class RenderFailure(RuntimeError):
    """A render the cell could not get right (twice): the sensor call answers `sensor_unavailable`."""


@contextlib.contextmanager
def gc_paused():
    """Pause the cyclic garbage collector (finalisers of dead renderers) for the duration of a GL critical section."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def validate_segmentation(raw, scene, ngeom_model):
    """(ok, mapped, why): raw is the (H, W) int64 id image as read back (before the flip); `mapped` is the (H, W, 2)
    int32 (object id, object type) image of mujoco.Renderer.render() (background -1, -1)."""
    n = int(scene.ngeom)
    lut = np.full((n + 1, 2), -1, np.int32)
    allowed = np.zeros(n + 1, bool)
    allowed[0] = True
    for g in scene.geoms[:n]:
        s = int(g.segid)
        if s != -1:
            lut[s + 1, 0] = int(g.objid)
            lut[s + 1, 1] = int(g.objtype)
            allowed[s + 1] = True
    hi = int(raw.max()) if raw.size else 0
    if hi > n or int(raw.min()) < 0:
        return False, None, "raw id %d outside 0..%d (the scene's %d geoms)" % (hi, n, n)
    bad = ~allowed[raw]
    if bad.any():
        return False, None, "%d pixels carry ids of no geom in the scene (e.g. %d)" % (int(bad.sum()),
                                                                                    int(raw[bad][0]))
    mapped = lut[raw]
    geom = mapped[..., 1] == _GEOM
    if geom.any():
        ids = mapped[..., 0][geom]
        if int(ids.min()) < 0 or int(ids.max()) >= int(ngeom_model):
            return False, None, "geom ids outside the model's 0..%d" % (int(ngeom_model) - 1)
    return True, mapped, ""


class SafeRenderer(mujoco.Renderer):
    """mujoco.Renderer with a close() that frees its GL objects in its own context, renders with the cyclic garbage
    collector paused, and validated segmentation renders (one rebuild-and-retry, then RenderFailure)."""

    incidents = None                         # set by the pool: the shared list of what went wrong (the mission record)

    def close(self) -> None:
        gl = getattr(self, "_gl_context", None)
        mj = getattr(self, "_mjr_context", None)
        self._mjr_context = None
        self._gl_context = None
        try:                                 # (also runs from __del__, possibly late at interpreter exit)
            was = gc.isenabled()
            gc.disable()
        except Exception:  # noqa: BLE001
            was = False
        try:
            if gl is not None and mj is not None:
                try:
                    gl.make_current()        # free the GL objects in THIS renderer's context, never another's
                except Exception:  # noqa: BLE001 - a dead context: free the MjrContext's memory anyway
                    pass
            if mj is not None:
                mj.free()
            if gl is not None:
                gl.free()                    # unbinds it when current: no context is current afterwards
        finally:
            if was:
                gc.enable()

    def _note(self, what, **kw):
        if self.incidents is not None and len(self.incidents) < MAX_INCIDENTS:
            row = {"what": what, "size": [int(self._width), int(self._height)]}
            row.update(kw)
            self.incidents.append(row)

    def _rebuild(self):
        """A fresh GL context and MjrContext of the same size (as __init__ makes them); the scene is kept."""
        self.close()
        with gc_paused():
            ctx_cls = getattr(_glc, "GLContext", None) if _glc is not None else None
            self._gl_context = ctx_cls(self._width, self._height) if ctx_cls is not None else None
            if self._gl_context:
                self._gl_context.make_current()
            self._mjr_context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context)
            self._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR

    def _segment_raw(self):
        """One segmentation render of the current scene; the raw id image as read back (not flipped)."""
        sc = self._scene
        flags0 = sc.flags.copy()
        sc.flags[_SEG] = True
        sc.flags[_IDC] = True
        # pre-filled with 0xFFFFFF, never a valid id: a read-back that writes nothing (a broken or deleted framebuffer)
        # fails the validation instead of passing as an all-background frame or as stale memory
        buf = np.full((self._height, self._width, 3), 255, np.uint8)
        try:
            with gc_paused():
                if self._gl_context:
                    self._gl_context.make_current()
                mujoco.mjr_render(self._rect, sc, self._mjr_context)
                mujoco.mjr_readPixels(buf, None, self._rect, self._mjr_context)
        finally:
            np.copyto(sc.flags, flags0)
        b = buf.astype(np.int64)
        return b[..., 0] + (b[..., 1] << 8) + (b[..., 2] << 16)

    def render(self, *, out=None):
        if self._mjr_context is None:
            raise RuntimeError("render cannot be called after close.")
        if not self._segmentation_rendering:
            with gc_paused():
                return super().render(out=out)
        why = first_why = ""
        for attempt in (0, 1):
            try:
                raw = self._segment_raw()
                ok, mapped, why = validate_segmentation(raw, self._scene, self._model.ngeom)
            except Exception as e:  # noqa: BLE001 - a GL / MuJoCo error is a failed render like a bad read-back
                ok, mapped, why = False, None, "%s: %s" % (type(e).__name__, str(e)[:200])
            if ok:
                if attempt:
                    self._note("segmentation_rerendered", first_failure=first_why)
                if self._gl_context:
                    mapped = np.ascontiguousarray(mapped[::-1])
                if out is not None:
                    out[...] = mapped
                    return out
                return mapped
            if attempt == 0:
                first_why = why
                self._rebuild()
        self._note("segmentation_failed", first_failure=first_why, second_failure=why)
        raise RenderFailure("segmentation render failed twice: %s; %s" % (first_why, why))


class GuardedRenderPool(bcam.RenderPool):
    """bricks' RenderPool (one renderer per image size) with SafeRenderer renderers, a safe close and a record of the
    incidents (`incidents`: re-rendered or failed segmentation renders)."""

    def __init__(self, model):
        super().__init__(model)
        self.incidents = []

    def get(self, width: int, height: int):
        key = (int(width), int(height))
        r = self._r.get(key)
        if r is None:
            with gc_paused():
                r = SafeRenderer(self.model, height=key[1], width=key[0])
            r.incidents = self.incidents
            self._r[key] = r
        return r

    def close(self):
        rs, self._r = list(self._r.values()), {}
        for r in rs:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
