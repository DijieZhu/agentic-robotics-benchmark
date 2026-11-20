"""AgX display transform (B. Wrensch's minimal AgX: the Rec.709 inset matrix, log2 encoding over [-12.47, 4.03] EV, the
6th-order sigmoid fit, the outset matrix, 2.2 decode; the 'punchy' look), the transform Blender's AgX approximates.

Used after a Filament render made with tone mapping "linear" at a reduced exposure (backend.Look.post_tonemap): the
8-bit sRGB output decodes to scene-linear / exposure, so photos from Filament and Cycles share one display transform.
"""
import numpy as np

_INSET = np.array([[0.842479062253094, 0.0423282422610123, 0.0423756549057051],
                   [0.0784335999999992, 0.878468636469772, 0.0784336],
                   [0.0792237451477643, 0.0791661274605434, 0.879142973793104]])     # rows = GLSL columns
_OUTSET = np.array([[1.19687900512017, -0.0528968517574562, -0.0529716355144438],
                    [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
                    [-0.0990297440797205, -0.0989611768448433, 1.15107367264116]])
_MIN_EV, _MAX_EV = -12.47393, 4.026069
_LW = np.array([0.2126, 0.7152, 0.0722])
LOOKS = {"none": (1.0, 1.0, 1.0), "punchy": (1.0, 1.35, 1.4), "golden": (1.0, 0.8, 0.8)}


def srgb_decode(u8):
    x = u8.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb_encode(lin):
    x = np.clip(lin, 0.0, 1.0)
    return (np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)


def agx(scene_linear, look="punchy"):
    """Scene-linear Rec.709 (N..., 3) -> display-linear (N..., 3)."""
    v = np.maximum(scene_linear, 1e-10) @ _INSET
    v = np.clip(np.log2(np.maximum(v, 1e-10)), _MIN_EV, _MAX_EV)
    x = (v - _MIN_EV) / (_MAX_EV - _MIN_EV)
    x2 = x * x
    x4 = x2 * x2
    y = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
    slope, power, sat = LOOKS[look]
    y = np.power(np.clip(y * slope, 0.0, None), power)
    luma = (y * _LW).sum(-1, keepdims=True)
    y = luma + sat * (y - luma)
    y = y @ _OUTSET
    return np.power(np.clip(y, 0.0, None), 2.2)


def regrade(rgb_u8, gain, look="punchy"):
    """An 8-bit sRGB render of a linear-tone-mapped frame -> AgX display sRGB; `gain` scales it to the scene-linear
    exposure AgX expects (the exposure of the Filament frame, the calibration and the photo's EV)."""
    return srgb_encode(agx(srgb_decode(rgb_u8) * float(gain), look))
