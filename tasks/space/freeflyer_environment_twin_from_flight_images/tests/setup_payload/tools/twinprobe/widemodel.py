"""The forward wide camera's model: the maker's one-coefficient wide-angle ("field of view") model.

Camera frame: +z along the optical axis (forward), +x image right, +y image down; origin at the camera centre.
Pixel coordinates (u, v) = (column, row) in pixels, integer values at pixel centres: (0, 0) is the centre of the
top-left pixel, (width - 1, height - 1) the centre of the bottom-right one, and the image covers
-0.5 <= u <= width - 0.5, -0.5 <= v <= height - 0.5. This is the convention of the maker's own code, where a
detected image point is taken in these coordinates and the principal point (cx, cy) lives in them.

Projection of a point X = (X, Y, Z), Z > 0:
    x_u = X / Z, y_u = Y / Z, r_u = sqrt(x_u^2 + y_u^2)
    r_d = atan(2 r_u tan(w / 2)) / w
    u = fx x_u (r_d / r_u) + cx,  v = fy y_u (r_d / r_u) + cy
with r_d / r_u taken at its limit 2 tan(w / 2) / w on the axis (r_u = 0). The inverse (a pixel's ray) is
r_u = tan(r_d w) / (2 tan(w / 2)) with r_d the radius of ((u - cx) / fx, (v - cy) / fy); it exists while
r_d w < pi / 2 (every pixel of this camera is inside that domain).
"""
import math

import numpy as np

from . import need


class WideModel:
    def __init__(self, fx, fy, cx, cy, w, width, height):
        self.fx, self.fy, self.cx, self.cy, self.w = float(fx), float(fy), float(cx), float(cy), float(w)
        self.width, self.height = int(width), int(height)
        self.k = 2.0 * math.tan(self.w / 2.0)

    @classmethod
    def from_config(cls, cfg):
        W, H = need(cfg, "wide_camera.image_size")
        return cls(need(cfg, "wide_camera.fx"), need(cfg, "wide_camera.fy"), need(cfg, "wide_camera.cx"),
                   need(cfg, "wide_camera.cy"), need(cfg, "wide_camera.fov_coefficient_w"), W, H)

    def project(self, X):
        """Pixel coordinates (n, 2) of camera-frame points (n, 3) and a mask of the points in front (Z > 0); the
        coordinates of points not in front are NaN."""
        X = np.asarray(X, float).reshape(-1, 3)
        Z = X[:, 2]
        front = Z > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            xu = np.where(front, X[:, 0] / Z, np.nan)
            yu = np.where(front, X[:, 1] / Z, np.nan)
            ru = np.hypot(xu, yu)
            conv = np.where(ru > 0, np.arctan(ru * self.k) / (self.w * ru), self.k / self.w)
        uv = np.stack([self.fx * xu * conv + self.cx, self.fy * yu * conv + self.cy], axis=1)
        return uv, front

    def ray(self, uv):
        """Unit ray (n, 3) in the camera frame of pixel coordinates (n, 2); NaN outside the model's domain."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        rd = np.hypot(xd, yd)
        ok = rd * self.w < math.pi / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            conv = np.where(rd > 0, np.tan(np.where(ok, rd, 0.0) * self.w) / (self.k * rd), self.w / self.k)
        d = np.stack([xd * conv, yd * conv, np.ones_like(xd)], axis=1)
        d = d / np.linalg.norm(d, axis=1)[:, None]
        d[~ok] = np.nan
        return d

    def in_image(self, uv, margin=0.0):
        """Pixel coordinates inside the image, at least `margin` pixels from its edges."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        m = float(margin)
        return ((uv[:, 0] >= -0.5 + m) & (uv[:, 0] <= self.width - 0.5 - m) &
                (uv[:, 1] >= -0.5 + m) & (uv[:, 1] <= self.height - 0.5 - m))

    def pixel_index(self, uv):
        """(column, row) integer index of the pixel that contains each image point."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        return np.floor(uv + 0.5).astype(np.int64)
