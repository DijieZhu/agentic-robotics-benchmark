"""The fictional carrier and the per-parcel waybill identity (tracking number, routing code, address lines).

Everything here is invented: the carrier word, its tracking-number format, the hub codes, the town and street names.
No person's name appears on a waybill: sender and recipient lines name a dock, desk or unit. The names were checked
(best effort) against public encyclopaedia and map data when the kit was made; that check's record is not part of the kit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CARRIER = {
    "wordmark": "KESTOVAR",
    "tagline": "PARCEL NETWORK",
    "tracking_prefix": "KV",
    "tracking_format": "KV + 9 digits + 1 check digit (Luhn); printed 'KV 1234 5678 90'",
}

# invented town names (checked: no Wikipedia page or exact-phrase hit, no OpenStreetMap feature of that name)
TOWNS = (
    "OSKELBY", "VELBRIGG", "QUENMOUTH", "KELVORAN", "UMMERFELL", "PELLOGATE", "ZANDERVALE", "MIRROWDEN",
    "FENDLECROSS", "CORVANBY", "WYSTRANDLE", "GORVANMERE", "OSTMERROW", "CALDERQUIN", "HOLLENVAY", "SKARNFORD",
    "ESKELDOWN", "PENTISCOMBE", "TOLVENHAM", "QUILLMARSH", "DUNSEVROW", "ORLESMOUTH", "KIRVENDALE", "BELVORROW",
    "STENNABY", "GADDLESBURY", "LUNSQUAY", "NESTERBY", "PELLINGDOWN", "ROSTREVIN", "SELVANHURST", "TREVALLOCH",
    "WENDLEQUAY", "ZELLINGFORD", "CRANDLEWICK", "BRINDLEQUAY", "HALVORSEND", "KETTLEMIRE", "AMBERVOSS", "TAMBERLOW",
)
STREETS = (
    "VELMARK", "QUINDLE", "LANTERNFIELD", "IRONWILLOW", "THORNQUILL", "ASHVENNY", "DUSKMERE", "HOLLINBRAE",
    "FENWHISTLE", "HALDERWICK", "DRAVENHURST", "BRASTHOLM", "MORRAVIK", "HETHERMOOR", "VOSSMERE",
)
STREET_TYPES = ("ROW", "QUAY", "LANE", "WAY", "YARD", "PARADE", "COURT", "ROAD")
RECIPIENT_UNITS = ("GOODS IN · BAY {n}", "RECEIVING DOCK {n}", "PARCEL LOCKER {n}", "STORE {n} · BACK OFFICE",
                   "UNIT {n} · RECEPTION", "STOCKROOM {n}", "COLLECTION POINT {n}", "WORKSHOP {n}")
SENDER_UNITS = ("DISPATCH UNIT {n}", "RETURNS DESK {n}", "WAREHOUSE {l}{n}", "FULFILMENT HALL {n}",
                "OUTBOUND DOCK {n}", "PACKING LINE {n}")
HUBS = ("NV", "TK", "OS", "HV", "BR", "QM", "KL", "DV", "ZE", "RW", "FX", "PL")
SERVICES = ("P1", "P2", "EX", "ST")


def luhn_digit(digits: str) -> int:
    s = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return (10 - s % 10) % 10


@dataclass(frozen=True)
class WaybillIdentity:
    tracking: str          # encoded string, e.g. KV4821773059
    routing: str           # large routing code, e.g. NV27-4C
    service: str           # service box, e.g. P2
    sender: tuple          # three lines
    recipient: tuple       # four lines
    weight: str            # e.g. 1.84 KG
    date: str              # e.g. 14.03
    ref: str               # small reference, e.g. A7-4471

    @property
    def tracking_print(self) -> str:
        t = self.tracking
        return "%s %s %s %s" % (t[:2], t[2:6], t[6:10], t[10:])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sender"], d["recipient"] = list(self.sender), list(self.recipient)
        d["tracking_print"] = self.tracking_print
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WaybillIdentity":
        keys = ("tracking", "routing", "service", "sender", "recipient", "weight", "date", "ref")
        return cls(**{k: (tuple(d[k]) if k in ("sender", "recipient") else d[k]) for k in keys})


def _one(rng: np.random.Generator, mass=None) -> WaybillIdentity:
    body = "".join(str(int(v)) for v in rng.integers(0, 10, 9))
    tracking = CARRIER["tracking_prefix"] + body + str(luhn_digit(body))
    hub = HUBS[int(rng.integers(len(HUBS)))]
    routing = "%s%02d-%d%s" % (hub, int(rng.integers(10, 100)), int(rng.integers(1, 10)),
                               "ABCDEFGHJKLMNPRSTUVWXYZ"[int(rng.integers(23))])
    service = SERVICES[int(rng.integers(len(SERVICES)))]

    def street():
        return "%d %s %s" % (int(rng.integers(2, 240)), STREETS[int(rng.integers(len(STREETS)))],
                             STREET_TYPES[int(rng.integers(len(STREET_TYPES)))])
    t_from, t_to = rng.choice(len(TOWNS), 2, replace=False)
    su = SENDER_UNITS[int(rng.integers(len(SENDER_UNITS)))].format(
        n=int(rng.integers(1, 40)), l="ABCDEFGH"[int(rng.integers(8))])
    sender = (su, street(), "%s  D%02d" % (TOWNS[int(t_from)], int(rng.integers(1, 60))))
    ru = RECIPIENT_UNITS[int(rng.integers(len(RECIPIENT_UNITS)))].format(n=int(rng.integers(1, 60)))
    recipient = (ru, street(), TOWNS[int(t_to)], "DISTRICT %02d / %03d" % (int(rng.integers(1, 60)),
                                                                          int(rng.integers(100, 1000))))
    w = float(mass) if mass is not None else float(rng.uniform(0.2, 9.5))
    weight = "%.2f KG" % w
    date = "%02d.%02d" % (int(rng.integers(1, 29)), int(rng.integers(1, 13)))
    ref = "%s%d-%04d" % ("ABCDEFGHJK"[int(rng.integers(10))], int(rng.integers(1, 10)), int(rng.integers(0, 10000)))
    return WaybillIdentity(tracking, routing, service, sender, recipient, weight, date, ref)


def identity_for_seed(seed: int, mass=None) -> WaybillIdentity:
    """One identity from a seed (deterministic). ``mass``: print this weight (kg) instead of a random one."""
    return _one(np.random.default_rng([0x5EED, int(seed) & 0xFFFFFFFF]), mass)


def identities(n: int, shift_seed: int, masses=None) -> list[WaybillIdentity]:
    """``n`` identities that differ pairwise in tracking number, routing code, sender and recipient lines
    (redraws on any collision). ``masses``: optional per-parcel printed weights (kg)."""
    rng = np.random.default_rng([0x5417, int(shift_seed) & 0xFFFFFFFF])
    out, seen = [], {"tracking": set(), "routing": set(), "sender": set(), "recipient": set()}
    while len(out) < n:
        m = None if masses is None else masses[len(out)]
        ident = _one(rng, m)
        keys = {"tracking": ident.tracking, "routing": ident.routing, "sender": ident.sender,
                "recipient": ident.recipient}
        if any(keys[k] in seen[k] for k in keys):
            continue
        for k in keys:
            seen[k].add(keys[k])
        out.append(ident)
    return out
