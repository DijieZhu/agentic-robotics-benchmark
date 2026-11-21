"""Optional camera images through mujoco.Renderer (the spacekit core never renders).

Use a CPU OSMesa backend (MUJOCO_GL=osmesa) where EGL is unavailable; one Renderer per process.
Image noise is added here (MuJoCo's sensor noise attribute is metadata only).
"""
from __future__ import annotations

import mujoco
import numpy as np

__all__ = ["CameraRenderer"]


class CameraRenderer:
    """Wrapper around one mujoco.Renderer: rgb(), gray(), depth() of a named camera (or MjvCamera).

    rng + noise_sigma add Gaussian pixel noise (8-bit levels) to colour images.  Use as a context manager
    or call close().
    """

    def __init__(self, model, width: int = 640, height: int = 480, rng: np.random.Generator | None = None,
                 noise_sigma: float = 0.0):
        self.r = mujoco.Renderer(model, height, width)
        self.rng, self.sigma = rng, noise_sigma

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        """Release the GL context."""
        self.r.close()

    def rgb(self, data, camera, scene_option=None) -> np.ndarray:
        """(H, W, 3) uint8 colour image."""
        self.r.disable_depth_rendering()
        self.r.update_scene(data, camera=camera, scene_option=scene_option)
        img = self.r.render()
        if self.rng is not None and self.sigma > 0:
            img = np.clip(img.astype(float) + self.rng.normal(0.0, self.sigma, img.shape), 0, 255).astype(np.uint8)
        return img

    def gray(self, data, camera) -> np.ndarray:
        """(H, W) uint8 luminance image."""
        img = self.rgb(data, camera).astype(float)
        return np.clip(img @ np.array([0.299, 0.587, 0.114]), 0, 255).astype(np.uint8)

    def depth(self, data, camera) -> np.ndarray:
        """(H, W) metric depth image [m]."""
        self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=camera)
        d = self.r.render().copy()
        self.r.disable_depth_rendering()
        return d
