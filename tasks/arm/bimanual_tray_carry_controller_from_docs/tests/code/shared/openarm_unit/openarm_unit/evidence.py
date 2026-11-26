import hashlib
import os
from contextlib import contextmanager

from . import registers as rg
from .frames import BUSES, Frame
from .state import public_params

PARAM_ID = 0x7FF
POSFORCE_BASE = 0x300
OP_READ, OP_WRITE, OP_SAVE = 0x33, 0x55, 0xAA
STEP_NS = 1_000_000

RID_INFO = {
    0: ("Under-Voltage Threshold", "RW", "float", "(10.0, fmax]"),
    1: ("KT Value (Torque Const)", "RW", "float", "[0.0, fmax]"),
    2: ("Over-Temp Threshold", "RW", "float", "[80.0, 200)"),
    3: ("Over-Current Threshold", "RW", "float", "(0.0, 1.0)"),
    4: ("Acceleration", "RW", "float", "(0.0, fmax)"),
    5: ("Deceleration", "RW", "float", "[-fmax, 0.0)"),
    6: ("Max Speed", "RW", "float", "(0.0, fmax]"),
    7: ("Master ID", "RW", "uint32", "[0, 0x7FF]"),
    8: ("Motor (ESC) ID", "RW", "uint32", "[0, 0x7FF]"),
    9: ("CAN Timeout", "RW", "uint32", "[0, 2^32-1]"),
    10: ("Control Mode", "RW", "uint32", "[0, 4]"),
    11: ("Damping Ratio", "RO", "float", "/"),
    12: ("Inertia", "RO", "float", "/"),
    13: ("Hardware Version", "RO", "uint32", "/"),
    14: ("Software Version", "RO", "uint32", "/"),
    15: ("Serial Number", "RO", "uint32", "/"),
    16: ("Number of Pole Pairs", "RO", "uint32", "/"),
    17: ("Stator Resistance (Rs)", "RO", "float", "/"),
    18: ("Stator Inductance (Ls)", "RO", "float", "/"),
    19: ("Rotor Flux", "RO", "float", "/"),
    20: ("Gear Ratio", "RO", "float", "/"),
    21: ("Position Limit (PMAX)", "RW", "float", "(0.0, fmax]"),
    22: ("Velocity Limit (VMAX)", "RW", "float", "(0.0, fmax]"),
    23: ("Torque Limit (TMAX)", "RW", "float", "(0.0, fmax]"),
    24: ("Current Loop Bandwidth", "RW", "float", "[100.0, 1.0e4]"),
    25: ("Speed Loop KP", "RW", "float", "[0.0, fmax]"),
    26: ("Speed Loop KI", "RW", "float", "[0.0, fmax]"),
    27: ("Position Loop KP", "RW", "float", "[0.0, fmax]"),
    28: ("Position Loop KI", "RW", "float", "[0.0, fmax]"),
    29: ("Over-Voltage Threshold", "RW", "float", "TBD"),
    30: ("Gear Torque Efficiency", "RW", "float", "(0.0, 1.0]"),
    31: ("Speed Loop Damping", "RW", "float", "[1.0, 30.0]"),
    32: ("Velocity Loop Bandwidth", "RW", "float", "(0.0, 500.0)"),
    33: ("Current Loop C1", "RW", "float", "[100.0, 1.0e4]"),
    34: ("Velocity Loop C1", "RW", "float", "(0.0, 1.0e4]"),
    35: ("CAN Baudrate", "RW", "uint32", "[0, 9]"),
    36: ("Sub Version", "RO", "uint32", "/"),
    50: ("I-Phase U Offset", "RO", "float", "/"),
    51: ("I-Phase V Offset", "RO", "float", "/"),
    52: ("Gain K1", "RO", "float", "/"),
    53: ("Gain K2", "RO", "float", "/"),
    54: ("Mechanical Offset", "RO", "float", "/"),
    55: ("Motor Direction", "RO", "float", "/"),
    80: ("Motor Position", "RO", "float", "/"),
    81: ("Output Shaft Position", "RO", "float", "/"),
}
CTRL_MODE_NAMES = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "POS_FORCE"}
CAN_BR_NAMES = {0: "125K", 1: "200K", 2: "250K", 3: "500K", 4: "1M", 5: "2M", 6: "2.5M", 7: "3.2M", 8: "4M",
                9: "5M", 10: "8M", 11: "10M"}
_FIXED0 = (7, 8, 9, 13, 14, 15, 16)
_SCI6 = (12, 11, 17, 18, 19)
LIVE = (rg.P_M, rg.XOUT)

INTERFACE = {0: "can1", 1: "can0"}
SIDE_BUS = {"left": 0, "right": 1}


class EvidenceError(RuntimeError):
    ""


def _modules():
    return public_params()["conventions"]["modules"]["value"]


def module_index(side, joint):
    for m in _modules():
        if m["side"] == side and int(m["joint"]) == int(joint):
            return int(m["index"])
    raise EvidenceError(f"no module {side} J{joint}")


def _wire_value(rid, value):
    return rg.unpack_value(rid, rg.pack_value(rid, value))


def format_value(rid, val):
    if rid == rg.CTRL_MODE:
        code = int(val)
        return f"{code} ({CTRL_MODE_NAMES.get(code, 'Unknown')})"
    if rid == rg.CAN_BR:
        code = int(val)
        return f"{code} ({CAN_BR_NAMES.get(code, 'Unknown')})"
    if rid in _FIXED0:
        return f"{float(val):.0f}"
    if rid in _SCI6:
        return f"{float(val):.6e}"
    return f"{float(val):.4f}"


def register_rows(unit, module, include_live=False):
    regs = unit.bus.regs
    out = []
    for rid in sorted(regs.module(module)):
        if rid in LIVE:
            if not include_live:
                continue
            th = float(unit.bus.theta()[module])
            val = th * float(unit.bus.gr[module]) if rid == rg.P_M else th
        else:
            val = regs.get(module, rid)
            if val is None:
                continue
        out.append((rid, _wire_value(rid, val)))
    return out


def register_dump_text(unit, module, include_live=False, power_up=True, header=True):
    m = _modules()[module]
    bus = int(m["bus"])
    sid = int(unit.bus.esc_id[module])
    rid_ = int(unit.bus.reply_id[module])
    lines = []
    if header:
        lines.append(f">>> Connecting to {INTERFACE[bus]} (FD Mode)...")
        lines.append(">>> Querying all registers for 1 motors. Please wait...")
    lines.append("")
    lines.append("=" * 50)
    lines.append(f" MOTOR ID: 0x{sid:x} (Response ID: 0x{rid_:x})")
    lines.append("=" * 50)
    lines.append(f"{'Parameter':<30}{'R/W':<5}{'Type':<10}{'Range':<20}Value")
    lines.append("-" * 80)
    for rid, val in register_rows(unit, module, include_live):
        desc, rw, typ, rng_ = RID_INFO.get(rid, (f"Register {rid}", "??", "??", "??"))
        lines.append(f"{desc:<30}{rw:<5}{typ:<10}{rng_:<20}{format_value(rid, val)}")
    if power_up:
        lines.append("")
        lines.append("Power-up serial output:")
        lines.append(f"Imax: {float(unit.bus.imax[module]):.6f}")
    return "\n".join(lines) + "\n"


def dump_file_name(module):
    m = _modules()[module]
    return f"{m['side']}_j{int(m['joint'])}.txt"


def register_dump(unit, out_dir, include_live=False, power_up=True):
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for i in range(len(_modules())):
        txt = register_dump_text(unit, i, include_live=include_live, power_up=power_up).encode("ascii")
        name = dump_file_name(i)
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(txt)
        out[name] = hashlib.sha256(txt).hexdigest()
    return out


def parse_register_dump(text):
    out = {"motor_id": None, "response_id": None, "rows": {}, "imax": None}
    in_table = False
    for line in text.splitlines():
        if line.startswith(" MOTOR ID: "):
            a, b = line.split("(Response ID: ")
            out["motor_id"] = int(a.split("0x")[1].strip(), 16)
            out["response_id"] = int(b.rstrip(")").split("0x")[1], 16)
        elif line.startswith("-" * 80):
            in_table = True
        elif in_table and line.strip() == "":
            in_table = False
        elif in_table:
            out["rows"][line[:30].rstrip()] = line[65:].strip()
        elif line.startswith("Imax: "):
            out["imax"] = float(line.split(":", 1)[1])
    return out


def module_record(unit):
    p = public_params()
    fw = p["registers"]["firmware"]["value"]
    out = []
    for i, m in enumerate(_modules()):
        out.append({"index": i, "label": m["label"], "side": m["side"], "joint": int(m["joint"]),
                    "product": m["product"], "edition": unit.bus.editions[i], "Imax_A": float(unit.bus.imax[i]),
                    "firmware": fw})
    return out


def _expected_reply(frame, route):
    if not isinstance(frame, Frame) or len(frame.data) != 8:
        return None
    d = frame.data
    if frame.can_id == PARAM_ID:
        rid = route.get((frame.bus, d[0] | (d[1] << 8)))
        op = d[2]
        return None if rid is None else (frame.bus, rid, op if op in (OP_READ, OP_WRITE, OP_SAVE) else None)
    cid = frame.can_id - POSFORCE_BASE if frame.can_id >= POSFORCE_BASE else frame.can_id
    rid = route.get((frame.bus, cid))
    return None if rid is None else (frame.bus, rid, None)


class BusCapture:

    def __init__(self, unit, epoch_ns=0):
        if int(epoch_ns) != epoch_ns:
            raise EvidenceError("epoch_ns is an integer number of nanoseconds")
        self.unit = unit
        self.epoch_ns = int(epoch_ns)
        self.records = []
        self.replies_seen = 0
        self.receptions = 0
        self._bus = None
        self._c0 = None
        self._unpaired = 0

    def attach(self):
        if self._bus is not None:
            raise EvidenceError("the capture is already attached")
        B = self.unit.bus
        if "receive" in B.__dict__ or "_drop" in B.__dict__:
            raise EvidenceError("another capture is attached to this unit")
        self._bus = B
        self._c0 = (list(B.frames_in), dict(B.drops))
        orig_receive = type(B).receive.__get__(B)
        orig_drop = type(B)._drop.__get__(B)
        cap = self
        pending = {}

        def _drop(bus, reason, frame):
            pending[id(frame)] = reason
            return orig_drop(bus, reason, frame)

        def receive(frames, steps=4, admit="all"):
            step = int(B.step_count)
            route = {(B.modules[i].bus, int(B.esc_id[i])): int(B.reply_id[i]) for i in range(B.n)}
            if isinstance(frames, (list, tuple)):
                frames_list = [Frame(f.bus, f.can_id, f.data) if type(f) is Frame else f for f in frames]
                frames = frames_list if isinstance(frames, list) else tuple(frames_list)
            else:
                frames_list = list(frames) if frames is not None else []
            pending.clear()
            replies = orig_receive(frames, steps, admit)
            cap._record(step, frames_list, replies, dict(pending), route)
            pending.clear()
            return replies

        B.receive = receive
        B._drop = _drop
        return self

    def detach(self):
        if self._bus is not None:
            for name in ("receive", "_drop"):
                self._bus.__dict__.pop(name, None)
            self._bus = None
        return self

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    def _record(self, step, frames, replies, dropped, route):
        self.receptions += 1
        self.replies_seen += len(replies)
        rq = list(replies)
        k = 0
        for f in frames:
            bus = f.bus if isinstance(f, Frame) else -1
            cid = f.can_id if isinstance(f, Frame) else -1
            data = bytes(f.data) if isinstance(f, Frame) else b""
            reason = dropped.get(id(f))
            self.records.append((step, "TX", bus, cid, data, reason))
            if reason is not None:
                continue
            exp = _expected_reply(f, route)
            if exp is None or k >= len(rq):
                continue
            r = rq[k]
            if (r.bus, r.can_id) != exp[:2]:
                continue
            if exp[2] is not None and (len(r.data) < 3 or r.data[2] != exp[2]):
                continue
            self.records.append((step, "RX", r.bus, r.can_id, bytes(r.data), None))
            k += 1
        for r in rq[k:]:
            self._unpaired += 1
            self.records.append((step, "RX", r.bus, r.can_id, bytes(r.data), None))

    def line(self, rec):
        step, d, bus, cid, data, reason = rec
        iface = INTERFACE.get(bus, f"bus{bus}")
        hexs = " ".join(f"{b:02X}" for b in data)
        s = f"{self.epoch_ns + step * STEP_NS:d} {d} {iface} {cid & 0x7FF:03X} [{len(data)}] {hexs}"
        if cid < 0 or cid > 0x7FF:
            s = f"{self.epoch_ns + step * STEP_NS:d} {d} {iface} {cid:X} [{len(data)}] {hexs}"
        if reason is not None:
            s += f" dropped:{reason}"
        return s

    def lines(self, bus=None):
        if isinstance(bus, str):
            bus = SIDE_BUS[bus]
        if bus is not None and bus not in BUSES:
            raise EvidenceError("bus is 0, 1, 'left', 'right' or None")
        return [self.line(r) for r in self.records if bus is None or r[2] == bus]

    def write(self, path, bus=None):
        txt = ("\n".join(self.lines(bus)) + "\n").encode("ascii") if self.records else b""
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            f.write(txt)
        return hashlib.sha256(txt).hexdigest()

    def counts(self):
        tx = sum(1 for r in self.records if r[1] == "TX")
        rx = sum(1 for r in self.records if r[1] == "RX")
        by_reason = {}
        for r in self.records:
            if r[5] is not None:
                by_reason[r[5]] = by_reason.get(r[5], 0) + 1
        return {"tx": tx, "rx": rx, "dropped": by_reason, "receptions": self.receptions,
                "replies_returned": self.replies_seen, "unpaired_replies": self._unpaired}

    def check(self):
        B = self._bus if self._bus is not None else self.unit.bus
        f0, d0 = self._c0
        d_in = sum(B.frames_in) - sum(f0)
        drops = {}
        for (b, r), c in B.drops.items():
            dc = c - d0.get((b, r), 0)
            if dc:
                drops[r] = drops.get(r, 0) + dc
        before_budget = sum(drops.get(r, 0) for r in ("malformed", "not_admitted", "budget"))
        c = self.counts()
        want = {"tx": d_in + before_budget, "dropped": drops, "rx": self.replies_seen}
        got = {"tx": c["tx"], "dropped": c["dropped"], "rx": c["rx"]}
        if want != got or c["unpaired_replies"]:
            raise EvidenceError(f"the capture lost or misplaced frames: bus counters {want}, capture {got}, "
                                f"unpaired replies {c['unpaired_replies']}")
        return {"bus_counters": want, "capture": got}


@contextmanager
def bus_capture(unit, path=None, bus=None, epoch_ns=0):
    cap = BusCapture(unit, epoch_ns=epoch_ns).attach()
    try:
        yield cap
    except BaseException:
        cap.detach()
        raise
    cap.check()
    cap.detach()
    if path is not None:
        cap.write(path, bus)


def parse_capture_line(line):
    parts = line.split()
    t, d, iface, cid = int(parts[0]), parts[1], parts[2], int(parts[3], 16)
    n = int(parts[4].strip("[]"))
    data = bytes(int(x, 16) for x in parts[5:5 + n])
    reason = None
    if len(parts) > 5 + n and parts[5 + n].startswith("dropped:"):
        reason = parts[5 + n].split(":", 1)[1]
    return t, d, iface, cid, data, reason
