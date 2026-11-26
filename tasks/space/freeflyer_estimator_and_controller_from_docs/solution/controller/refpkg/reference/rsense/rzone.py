# Generated file (the task's solution): do not edit; regenerate it from the maintained source.
import os
import struct
import numpy as np
FAR_M = 10.0
THIN_M = 0.004
NEAR_M = 1.0
SLACK_M = 0.25

def box_dist(P, lo, hi):
    d = np.maximum(np.maximum(lo[None, :, :] - P[:, None, :], P[:, None, :] - hi[None, :, :]), 0.0)
    return np.sqrt((d * d).sum(-1))

class ExactZones:

    def __init__(self, zones):
        kin = [z for z in zones if z['type'] == 'keepin']
        kout = [z for z in zones if z['type'] == 'keepout']
        if not kin:
            raise ValueError('zero keep-in zones')
        self.other_types = sorted({str(z['type']) for z in zones if z['type'] not in ('keepin', 'keepout')})
        lo = np.array([np.minimum(z['min'], z['max']) for z in kin], float)
        hi = np.array([np.maximum(z['min'], z['max']) for z in kin], float)
        self.kin_lo, self.kin_hi = (lo, hi)
        br = []
        for a in range(3):
            v = np.unique(np.concatenate([lo[:, a], hi[:, a]]))
            br.append(np.concatenate([[v[0] - FAR_M], v, [v[-1] + FAR_M]]))
        self.br = br
        cx = [0.5 * (b[:-1] + b[1:]) for b in br]
        G = np.stack(np.meshgrid(*cx, indexing='ij'), -1)
        inside = np.zeros(G.shape[:3], bool)
        for i in range(len(lo)):
            inside |= np.all((G > lo[i]) & (G < hi[i]), axis=-1)
        self.inside = inside
        nb = np.zeros_like(inside)
        nb[1:, :, :] |= inside[:-1, :, :]
        nb[:-1, :, :] |= inside[1:, :, :]
        nb[:, 1:, :] |= inside[:, :-1, :]
        nb[:, :-1, :] |= inside[:, 1:, :]
        nb[:, :, 1:] |= inside[:, :, :-1]
        nb[:, :, :-1] |= inside[:, :, 1:]
        idx = np.argwhere(nb & ~inside)
        self.out_lo = np.stack([br[a][idx[:, a]] for a in range(3)], 1)
        self.out_hi = np.stack([br[a][idx[:, a] + 1] for a in range(3)], 1)
        gaps = 0
        for a in range(3):
            w = np.diff(br[a])
            for k in range(1, len(w) - 1):
                if w[k] < THIN_M:
                    s0 = [slice(None)] * 3
                    s1 = list(s0)
                    s2 = list(s0)
                    s0[a], s1[a], s2[a] = (k - 1, k, k + 1)
                    gaps += int(np.sum(inside[tuple(s0)] & ~inside[tuple(s1)] & inside[tuple(s2)]))
        self.thin_gap_cells = gaps
        self._near = None
        if kout:
            self.ko_lo = np.array([np.minimum(z['min'], z['max']) for z in kout], float)
            self.ko_hi = np.array([np.maximum(z['min'], z['max']) for z in kout], float)
        else:
            self.ko_lo = np.zeros((0, 3))
            self.ko_hi = np.zeros((0, 3))

    def describe(self):
        return {'keep_in_boxes': int(len(self.kin_lo)), 'keep_out_boxes': int(len(self.ko_lo)), 'boundary_outside_cells': int(len(self.out_lo)), 'thin_gap_cells_below_m': [self.thin_gap_cells, THIN_M], 'other_zone_types_ignored': self.other_types}

    def in_union(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        r = np.zeros(len(P), bool)
        for i in range(len(self.kin_lo)):
            r |= np.all((P >= self.kin_lo[i]) & (P <= self.kin_hi[i]), axis=1)
        return r

    def in_keep_out(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        r = np.zeros(len(P), bool)
        for i in range(len(self.ko_lo)):
            r |= np.all((P >= self.ko_lo[i]) & (P <= self.ko_hi[i]), axis=1)
        return r

    def d_in(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        out = np.zeros(len(P))
        ins = self.in_union(P)
        if ins.any():
            Q = P[ins]
            res = np.empty(len(Q))
            for s in range(0, len(Q), 256):
                res[s:s + 256] = box_dist(Q[s:s + 256], self.out_lo, self.out_hi).min(1)
            out[ins] = res
        return out

    def d_out(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        if not len(self.ko_lo):
            return np.full(len(P), np.inf)
        return box_dist(P, self.ko_lo, self.ko_hi).min(1)

    def clearance(self, P):
        return np.minimum(self.d_in(P), self.d_out(P))

    def point(self, p):
        p = np.asarray(p, float).reshape(3)
        if not np.any(np.all((p >= self.kin_lo) & (p <= self.kin_hi), axis=1)):
            di = 0.0
        else:
            nr = self._near
            if nr is None or np.max(np.abs(p - nr[0])) > SLACK_M:
                h = NEAR_M + SLACK_M
                m = np.all(self.out_hi >= p - h, axis=1) & np.all(self.out_lo <= p + h, axis=1)
                nr = self._near = (p.copy(), self.out_lo[m], self.out_hi[m])
            if len(nr[1]):
                d = np.maximum(np.maximum(nr[1] - p, p - nr[2]), 0.0)
                di = min(float(np.sqrt((d * d).sum(1)).min()), NEAR_M)
            else:
                di = NEAR_M
        if len(self.ko_lo):
            d = np.maximum(np.maximum(self.ko_lo - p, p - self.ko_hi), 0.0)
            do = min(float(np.sqrt((d * d).sum(1)).min()), NEAR_M)
        else:
            do = NEAR_M
        return (di, do)

    def breach(self, p, r):
        di, do = self.point(p)
        assert r < NEAR_M
        if di >= r and do >= r:
            return None
        return 'keep_in' if di <= do else 'keep_out'

    def contact(self, p):
        p = np.asarray(p, float).reshape(3)
        if not np.any(np.all((p >= self.kin_lo) & (p <= self.kin_hi), axis=1)):
            return 'outside'
        if len(self.ko_lo) and np.any(np.all((p >= self.ko_lo) & (p <= self.ko_hi), axis=1)):
            return 'structure'
        return None
TYPES = {0: 'keepout', 1: 'keepin', 2: 'clutter'}

def parse_zones(buf: bytes) -> dict:
    off = 0
    sec, nsec = struct.unpack_from('<II', buf, off)
    off += 8
    n, = struct.unpack_from('<I', buf, off)
    off += 4
    zones = []
    for _ in range(n):
        ln, = struct.unpack_from('<I', buf, off)
        off += 4
        name = buf[off:off + ln].decode('ascii')
        off += ln
        idx, typ = struct.unpack_from('<iB', buf, off)
        off += 5
        v = struct.unpack_from('<6d', buf, off)
        off += 48
        lo = [min(v[i], v[i + 3]) for i in range(3)]
        hi = [max(v[i], v[i + 3]) for i in range(3)]
        zones.append({'name': name, 'index': idx, 'type': TYPES.get(typ, str(typ)), 'min': lo, 'max': hi, 'size': [hi[i] - lo[i] for i in range(3)]})
    if off != len(buf):
        raise ValueError('trailing bytes: parsed %d of %d' % (off, len(buf)))
    return {'timestamp_sec': sec, 'timestamp_nsec': nsec, 'zones': zones}
ROOT_ENV = 'RS_PAYLOAD_ROOT'

def beside_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))

def roots():
    r = []
    v = os.environ.get(ROOT_ENV)
    if v:
        r.append(v)
    r.append(beside_root())
    return r

def relative_parts(zone_file):
    if not isinstance(zone_file, str) or not zone_file:
        return None
    s = zone_file.replace('\\', '/')
    if s.startswith('/') or (len(s) > 1 and s[1] == ':'):
        return None
    parts = s.split('/')
    if any((p in ('', '.', '..') for p in parts)):
        return None
    return parts

def load_for_paper(paper):
    zf = paper.get('zone_file') if isinstance(paper, dict) else None
    rec = {'paper_zone_file': zf, 'status': 'none', 'path': None, 'roots_tried': [], 'size': None, 'reason': None}
    if not zf:
        return (None, rec)
    try:
        parts = relative_parts(zf)
        if parts is None:
            rec.update(status='unavailable', reason="the paper's zone_file is not a plain relative path")
            return (None, rec)
        path = None
        for r in roots():
            rec['roots_tried'].append(r)
            p = os.path.join(r, *parts)
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            rec.update(status='unavailable', reason='no payload root holds the file')
            return (None, rec)
        rec['path'] = path
        with open(path, 'rb') as fh:
            raw = fh.read()
        rec['size'] = len(raw)
        ez = ExactZones(parse_zones(raw)['zones'])
        rec.update(status='loaded', keep_in_boxes=int(len(ez.kin_lo)), keep_out_boxes=int(len(ez.ko_lo)))
        return (ez, rec)
    except Exception as e:
        rec.update(status='unavailable', reason='%s: %s' % (type(e).__name__, e))
        return (None, rec)
