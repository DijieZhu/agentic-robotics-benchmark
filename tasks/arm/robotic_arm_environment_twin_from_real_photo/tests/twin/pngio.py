"""Minimal PNG reader for the grader's reference images (greyscale, 8 or 16 bit, not interlaced): zlib + numpy."""
import struct
import zlib

import numpy as np

SIG = b"\x89PNG\r\n\x1a\n"


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path):
    """The image as a uint8 or uint16 array (H, W)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != SIG:
        raise ValueError("%s is not a PNG" % path)
    pos, idat, hdr = 8, [], None
    while pos + 8 <= len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if typ == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat.append(body)
        elif typ == b"IEND":
            break
    if hdr is None:
        raise ValueError("%s has no IHDR" % path)
    W, H, bd, ct, _comp, _filt, inter = hdr
    if ct != 0 or inter != 0 or bd not in (8, 16):
        raise ValueError("%s: only non-interlaced 8/16-bit greyscale PNGs are supported" % path)
    bpp = bd // 8
    stride = W * bpp
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    if raw.size != H * (stride + 1):
        raise ValueError("%s: unexpected data size" % path)
    rows = raw.reshape(H, stride + 1)
    out = np.zeros((H, stride), np.uint8)
    prev = np.zeros(stride, np.int64)
    for r in range(H):
        f = int(rows[r, 0])
        x = rows[r, 1:].astype(np.int64)
        if f == 0:
            cur = x
        elif f == 1:
            cur = np.cumsum(x.reshape(W, bpp), axis=0).reshape(-1) & 0xFF
        elif f == 2:
            cur = (x + prev) & 0xFF
        elif f in (3, 4):
            xl, pl = x.tolist(), prev.tolist()
            cl = [0] * stride
            for i in range(stride):
                a = cl[i - bpp] if i >= bpp else 0
                b = pl[i]
                if f == 3:
                    cl[i] = (xl[i] + ((a + b) >> 1)) & 0xFF
                else:
                    c = pl[i - bpp] if i >= bpp else 0
                    cl[i] = (xl[i] + _paeth(a, b, c)) & 0xFF
            cur = np.asarray(cl, np.int64)
        else:
            raise ValueError("%s: bad filter type %d" % (path, f))
        out[r] = cur
        prev = cur
    if bd == 16:
        return out.reshape(-1).view(">u2").reshape(H, W).astype(np.uint16)
    return out.reshape(H, W)
