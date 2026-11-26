import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C

import numpy as np

SH, QT = C.SH, C.QT
EXTRA_VEC = {"T_w": 16, "status": 16, "stop_margin": 14, "T_del": 16}
CHUNK = 1 << 15


def grader_trace_fields(cell=None):
    base = tuple(SH.trace_fields(cell))
    return base + tuple(f for f in ("T_del",) if f not in base)


class GradeTrace(QT.TraceStore):
    VEC = dict(QT.TraceStore.VEC, **EXTRA_VEC)
    SCALAR = tuple(QT.TraceStore.SCALAR) + ("F_tray_N",)

    def __init__(self):
        self._chunks = {k: [] for k in self.keys()}
        self._cur = None
        self._n_cur = 0
        self._n = 0
        self._a = None
        self._rows = None
        self.missing = set()

    @classmethod
    def keys(cls):
        return ("step", "S_tray_N") + tuple(cls.VEC) + tuple(cls.SCALAR)

    def _new_chunk(self):
        c = {"step": np.empty(CHUNK, np.int64), "S_tray_N": np.empty(CHUNK)}
        for k, d in self.VEC.items():
            c[k] = np.empty((CHUNK, d))
        for k in self.SCALAR:
            c[k] = np.empty(CHUNK)
        self._cur = c
        self._n_cur = 0

    def _flush(self):
        if self._cur is not None and self._n_cur:
            for k, v in self._cur.items():
                self._chunks[k].append(v[:self._n_cur].copy())
        self._cur = None
        self._n_cur = 0

    def add(self, rec):
        if self._cur is None or self._n_cur == CHUNK:
            self._flush()
            self._new_chunk()
        c, j = self._cur, self._n_cur
        c["step"][j] = int(rec["step"])
        c["S_tray_N"][j] = SH.Monitor.shelf_normal(rec)
        for k, d in self.VEC.items():
            v = rec.get(k)
            if v is None:
                self.missing.add(k)
                c[k][j] = np.nan
            else:
                c[k][j] = np.asarray(v, dtype=np.float64).reshape(d)
        for k in QT.TraceStore.SCALAR:
            c[k][j] = float(rec[k])
        Fd = rec.get("F.normal_N") or {}
        c["F_tray_N"][j] = float(sum(v for kk, v in Fd.items() if "tray" in kk))
        self._n_cur += 1
        self._n += 1
        self._a = None

    def __len__(self):
        return self._n if self._chunks is not None else len(self._a["step"])

    def arrays(self):
        if self._a is None:
            self._flush()
            a = {}
            for k in self.keys():
                ch = self._chunks[k]
                if ch:
                    a[k] = np.concatenate(ch, axis=0)
                else:
                    a[k] = np.empty((0,) + ((self.VEC[k],) if k in self.VEC else ()),
                                    dtype=np.int64 if k == "step" else np.float64)
            self._chunks = {k: [a[k]] for k in self.keys()}
            self._a = a
        return self._a

    def save(self, path):
        a = self.arrays()
        np.savez_compressed(path, **{k.replace(".", "__"): v for k, v in a.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        t = cls()
        a = {k.replace("__", "."): z[k] for k in z.files}
        t._a = a
        t._chunks = {k: [a[k]] for k in a}
        t._n = len(a["step"])
        return t

    def nbytes(self):
        return int(sum(v.nbytes for v in self.arrays().values()))
