"""Random numbers of the shared unit. PUBLIC: imports nothing of the package.

The only source of random numbers anywhere in openarm_unit:

    digest(key, stream, counter) = SHA-256( key (32 bytes) || len(stream) (1 byte) || stream (ASCII) ||
                                            counter (8 bytes, big endian) )

Each digest yields four 64-bit big-endian words; of each the top 53 bits scaled by 2^-53 give one uniform in [0, 1).
Uniform number n of a stream is word n % 4 of the digest at counter n // 4 (four uniforms per digest).
A "normal-like" draw is the sum of four uniforms minus 2, times the literal 1.7320508075688772 (unit variance); normal-
like draw n of a stream uses the four uniforms of digest n, summed left to right: ((u0 + u1) + u2) + u3.

No transcendental function, no BLAS, no threads, no wall clock, no hash-seed dependence. The test vectors below come
from an independent derivation in integer arithmetic and fractions (no numpy), not from this file; `check_vectors`
recomputes them with this file.
"""
import hashlib

import numpy as np

KEY_BYTES = 32
NORMAL_SCALE = 1.7320508075688772          # the literal
_TWO_M53 = 1.0 / 9007199254740992.0         # 2^-53, exact in float64


class RngError(ValueError):
    """A key or stream name that this module does not allow."""


def _check(key, stream):
    if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_BYTES:
        raise RngError("the key must be exactly 32 bytes")
    if not isinstance(stream, str):
        raise RngError("the stream name must be a str")
    try:
        name = stream.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RngError("the stream name must be ASCII") from exc
    if len(name) > 255:
        raise RngError("the stream name must be at most 255 bytes (its length is one byte)")
    return bytes(key), name


def digest(key, stream, counter):
    """The 32-byte digest of (key, stream, counter)."""
    k, name = _check(key, stream)
    if not (0 <= int(counter) < (1 << 64)):
        raise RngError("the counter must fit in 8 bytes")
    return hashlib.sha256(k + bytes([len(name)]) + name + int(counter).to_bytes(8, "big")).digest()


def _words_to_uniforms(d):
    w = np.frombuffer(d, dtype=">u8").astype(np.uint64)
    return (w >> np.uint64(11)).astype(np.float64) * _TWO_M53      # exact: < 2^53 converts without rounding


def uniforms(key, stream, start, count):
    """Uniforms number start .. start + count - 1 of a stream, as a float64 array."""
    if count < 0 or start < 0:
        raise RngError("start and count must be non-negative")
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    first, last = start // 4, (start + count - 1) // 4
    k, name = _check(key, stream)
    prefix = k + bytes([len(name)]) + name
    buf = b"".join(hashlib.sha256(prefix + c.to_bytes(8, "big")).digest() for c in range(first, last + 1))
    u = _words_to_uniforms(buf)
    off = start - 4 * first
    return u[off:off + count].copy()


def uniform(key, stream, n):
    """Uniform number n of a stream (a Python float)."""
    return float(uniforms(key, stream, n, 1)[0])


def normals(key, stream, start, count):
    """Normal-like draws number start .. start + count - 1 of a stream (each from the four uniforms of one digest)."""
    u = uniforms(key, stream, 4 * start, 4 * count).reshape(count, 4)
    s = ((u[:, 0] + u[:, 1]) + u[:, 2]) + u[:, 3]
    return (s - 2.0) * NORMAL_SCALE


_HEADS = {}                                 # (key, stream) -> SHA-256 state after its prefix (a memo, values unchanged)
_HEADS_MAX = 256


def _head(key, stream):
    """The SHA-256 state after key || len(stream) || stream (checked as digest() checks them); h.copy() continued
    with the counter's 8 bytes gives digest(key, stream, counter)."""
    k = (bytes(key) if isinstance(key, (bytes, bytearray)) else key, stream)
    h = _HEADS.get(k) if isinstance(stream, str) and isinstance(k[0], bytes) else None
    if h is None:
        kb, name = _check(key, stream)
        h = hashlib.sha256(kb + bytes([len(name)]) + name)
        if len(_HEADS) >= _HEADS_MAX:
            _HEADS.clear()
        _HEADS[(kb, stream)] = h
    return h


def normals_streams(key, streams, start, count):
    """Normal-like draws number start .. start + count - 1 of each stream of `streams` under one key, as a (len(streams)
    x count) array: row r equals normals(key, streams[r], start, count), value for value (the same digests, words and
    element-wise operations, taken for all the streams in one pass)."""
    if count < 0 or start < 0:
        raise RngError("start and count must be non-negative")
    heads = [_head(key, stream) for stream in streams]
    if count == 0 or not heads:
        return np.zeros((len(heads), count), dtype=np.float64)
    counters = [c.to_bytes(8, "big") for c in range(start, start + count)]
    out = []
    put = out.append
    for h0 in heads:                    # the SHA-256 state after key || len || stream, continued with each counter
        for c in counters:
            h = h0.copy()
            h.update(c)
            put(h.digest())
    u = _words_to_uniforms(b"".join(out)).reshape(len(heads), count, 4)
    s = ((u[:, :, 0] + u[:, :, 1]) + u[:, :, 2]) + u[:, :, 3]
    return (s - 2.0) * NORMAL_SCALE


def normal(key, stream, n):
    """Normal-like draw number n of a stream (a Python float)."""
    return float(normals(key, stream, n, 1)[0])


def uniform_in(key, stream, n, lo, hi):
    """lo + (hi - lo) u with u = uniform number n of the stream (an affine map: + and x only)."""
    return lo + (hi - lo) * uniform(key, stream, n)


def law_stream(quantity, element, attempt=0, unit_index=0):
    """The stream name '<quantity>/u<unit>/e<element>/a<attempt>' of a quantity id and three non-negative indices;
    nothing in this package depends on the form."""
    if not quantity or "/" in quantity:
        raise RngError("a quantity id is a non-empty name without '/'")
    for v in (element, attempt, unit_index):
        if int(v) != v or v < 0:
            raise RngError("indices are non-negative integers")
    return f"{quantity}/u{int(unit_index)}/e{int(element)}/a{int(attempt)}"


# Test vectors (at least eight). (key hex, stream, kind, index, value as float.hex()). Computed by an
# independent derivation (integers and fractions, no numpy), not by this file.
TEST_VECTORS = (
    ("00" * 32, "", "uniform", 0, "0x1.3c2e6d887a322p-1"),
    ("00" * 32, "", "uniform", 3, "0x1.29375ad4f2816p-1"),
    ("00" * 32, "noise/left/q", "uniform", 0, "0x1.6054b03472718p-2"),
    ("00" * 32, "noise/left/q", "uniform", 5, "0x1.164ccf319174ap-2"),
    ("01" * 32, "P1/u0/e3/a0", "uniform", 0, "0x1.834cb2ab8b1c0p-1"),
    ("ff" * 32, "C0/u0/e15/a0", "uniform", 1000003, "0x1.9e39db304e51fp-1"),
    ("00" * 32, "noise/left/q", "normal", 0, "0x1.0b44cab04c820p-3"),
    ("0123456789abcdef" * 4, "noise/right/torque", "normal", 7, "-0x1.86739323d2682p+0"),
    ("0123456789abcdef" * 4, "HS1/u2/e13/a1", "normal", 123456, "0x1.bbe087b26b9c0p-1"),
)


def check_vectors():
    """Recompute TEST_VECTORS; return a list of mismatches (empty when every vector holds)."""
    bad = []
    for key_hex, stream, kind, n, want in TEST_VECTORS:
        key = bytes.fromhex(key_hex)
        got = uniform(key, stream, n) if kind == "uniform" else normal(key, stream, n)
        if got.hex() != want:
            bad.append((key_hex[:8], stream, kind, n, want, got.hex()))
    return bad
