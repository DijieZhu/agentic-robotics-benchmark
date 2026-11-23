"""Code 128 (ISO/IEC 15417) encoder and a module-level decoder used by the kit's own checks.

encode(text) -> list of module widths, starting with a bar (alternating bar / space), start, data, check and stop
symbols included, quiet zones excluded. The encoder uses code set B for non-digits and switches to code set C for
runs of four or more digits (an even run at the end of the text, or six or more inside it).
"""
from __future__ import annotations

# bar / space widths of the 107 symbols (value 0..105, 106 = stop)
_PATTERNS = (
    "212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312", "132212", "221213",
    "221312", "231212", "112232", "122132", "122231", "113222", "123122", "123221", "223211", "221132",
    "221231", "213212", "223112", "312131", "311222", "321122", "321221", "312212", "322112", "322211",
    "212123", "212321", "232121", "111323", "131123", "131321", "112313", "132113", "132311", "211313",
    "231113", "231311", "112133", "112331", "132131", "113123", "113321", "133121", "313121", "211331",
    "231131", "213113", "213311", "213131", "311123", "311321", "331121", "312113", "312311", "332111",
    "314111", "221411", "431111", "111224", "111422", "121124", "121421", "141122", "141221", "112214",
    "112412", "122114", "122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111",
    "111242", "121142", "121241", "114212", "124112", "124211", "411212", "421112", "421211", "212141",
    "214121", "412121", "111143", "111341", "131141", "114113", "114311", "411113", "411311", "113141",
    "114131", "311141", "411131", "211412", "211214", "211232", "2331112",
)
START_B, START_C, CODE_C, CODE_B, STOP = 104, 105, 99, 100, 106

assert len(_PATTERNS) == 107
assert all(sum(map(int, p)) == 11 for p in _PATTERNS[:106]) and sum(map(int, _PATTERNS[106])) == 13
assert len(set(_PATTERNS)) == 107
# every symbol's bar modules add up to an even number (the code's parity rule)
assert all(sum(int(p[i]) for i in range(0, len(p), 2)) % 2 == 0 for p in _PATTERNS)


def symbol_values(text: str) -> list[int]:
    """Symbol values for ``text`` (printable ASCII), start and check symbol included, stop excluded."""
    if not text or any(not (32 <= ord(c) < 127) for c in text):
        raise ValueError("Code 128 text must be non-empty printable ASCII")
    vals: list[int] = []
    i, n = 0, len(text)

    def digit_run(k):
        j = k
        while j < n and text[j].isdigit():
            j += 1
        return j - k

    run0 = digit_run(0)
    mode = "C" if (run0 >= 4 and run0 % 2 == 0) or run0 == n and run0 >= 2 and run0 % 2 == 0 else "B"
    vals.append(START_C if mode == "C" else START_B)
    while i < n:
        if mode == "B":
            run = digit_run(i)
            at_end = i + run == n
            if (at_end and run >= 4 and run % 2 == 0) or (not at_end and run >= 6):
                vals.append(CODE_C)
                mode = "C"
                continue
            vals.append(ord(text[i]) - 32)
            i += 1
        else:
            if i + 1 < n and text[i].isdigit() and text[i + 1].isdigit():
                vals.append(int(text[i:i + 2]))
                i += 2
            else:
                vals.append(CODE_B)
                mode = "B"
    check = vals[0] + sum(k * v for k, v in enumerate(vals[1:], start=1))
    vals.append(check % 103)
    return vals


def encode(text: str) -> list[int]:
    """Module widths (bar first), quiet zones excluded."""
    widths: list[int] = []
    for v in symbol_values(text) + [STOP]:
        widths.extend(int(c) for c in _PATTERNS[v])
    return widths


def modules(text: str, quiet: int = 10) -> list[int]:
    """One entry per module: 1 = bar, 0 = space, with ``quiet`` space modules each side."""
    out = [0] * quiet
    bar = 1
    for w in encode(text):
        out.extend([bar] * w)
        bar ^= 1
    out.extend([0] * quiet)
    return out


_LOOKUP = {p: v for v, p in enumerate(_PATTERNS)}


def decode_widths(widths: list[int]) -> str:
    """Inverse of ``encode`` for exact module widths (bar first): checks start, check symbol and stop."""
    if len(widths) < 6 + 6 + 7:
        raise ValueError("too short")
    syms, k = [], 0
    while k < len(widths):
        p = "".join(str(w) for w in widths[k:k + 6])
        if len(widths) - k == 7:
            p = "".join(str(w) for w in widths[k:k + 7])
        if p not in _LOOKUP:
            raise ValueError("unknown symbol %s at %d" % (p, k))
        syms.append(_LOOKUP[p])
        k += len(p)
    if syms[-1] != STOP:
        raise ValueError("no stop symbol")
    body, check = syms[:-2], syms[-2]
    if (body[0] + sum(i * v for i, v in enumerate(body[1:], start=1))) % 103 != check:
        raise ValueError("check symbol mismatch")
    mode = {START_B: "B", START_C: "C"}.get(body[0])
    if mode is None:
        raise ValueError("unsupported start symbol")
    out = []
    for v in body[1:]:
        if mode == "B":
            if v == CODE_C:
                mode = "C"
            else:
                out.append(chr(v + 32))
        else:
            if v == CODE_B:
                mode = "B"
            else:
                out.append("%02d" % v)
    return "".join(out)


def decode_scanline(profile, threshold=None) -> str:
    """Decode one scanline (1-D array of grey values, dark bars low) of a Code 128 symbol of unknown module size.

    Run lengths are measured between threshold crossings (sub-pixel by linear interpolation) and divided by the
    module size the start symbol gives (its 11 modules). Raises ValueError when the line does not decode."""
    import numpy as np
    y = np.asarray(profile, float)
    if threshold is None:
        threshold = 0.5 * (np.percentile(y, 5) + np.percentile(y, 95))
    s = y - threshold
    xs = []
    for i in range(len(s) - 1):
        if (s[i] >= 0) != (s[i + 1] >= 0):
            xs.append(i + s[i] / (s[i] - s[i + 1]))
    if len(xs) < 20:
        raise ValueError("too few edges")
    xs = np.asarray(xs)
    # first edge = white -> dark (start of the first bar)
    if s[0] < 0:
        raise ValueError("line must start on the quiet zone")
    runs = np.diff(xs)
    # module size from the start symbol (6 elements = 11 modules)
    mod = runs[:6].sum() / 11.0
    widths = [int(round(r / mod)) for r in runs]
    # refine the module size from the whole symbol: total modules = 11 * nsym + 2
    tot = sum(widths)
    nsym = (tot - 2) // 11
    if nsym > 2:
        mod = runs.sum() / (11 * nsym + 2)
        widths = [max(1, min(4, int(round(r / mod)))) for r in runs]
    return decode_widths(widths)
