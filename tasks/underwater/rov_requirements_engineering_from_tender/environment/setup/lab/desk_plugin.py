"""Enquiry and witness desk plugin of the lab service (lab/alelab/server.py, used unmodified).

Loaded by the lab server in its worker, as root, before the worker drops to the lab account:
make_plugin(instances, api) receives the parsed desk table (`ref/desk_table.json`, format
uw-desk-table/1) and keeps everything it needs in memory, because the worker cannot read the setup
folder after the privilege drop.

Twelve instances: the Owner's four staff members and the eight bidders. One request kind,
`execute(instance, plan)`, with two plan forms:

    {"ask": <topic>}                                   a question to one person or bidder
    {"witness": <test>, <parameter>: <value>, ...}     a witnessed factory test at a bidder

and the witness tests and their public parameter grids:

    {"witness": "weigh"}
    {"witness": "bollard_pull", "bus_v": 12|13|14|15|16, "limiter": "on"|"off"}
    {"witness": "battery_discharge", "chamber_c": 10|15|23, "load_w": 100|440}
    {"witness": "tow", "speed_mps": 0.4|0.6|0.8}

Every answer is a pure function of the (instance, canonical request) cell of the frozen table: no
state, the same bytes every time. A reply carries UTF-8 text and an optional PNG page as `u8[*]`
arrays (the server has no string type) and the measured values as `f64[*]` (empty for a question).
A numeric parameter is accepted as an integer or a float equal to a grid value (14 and 14.0 are the
same request); booleans and strings are not numbers.

Requests are validated against public constants only (validate_plan sees only the plan, never the
instance), so a rejection (free) never depends on the table. Consequence: a plan that is valid for
one kind of instance but sent to the other kind (an owner topic to a bidder, a bidder topic or a
witness test to the Owner's staff) passes validation and is CHARGED; it is answered with the fixed
redirect text of that instance kind (table section `redirects`).

The plugin takes no options: any key in the configuration's `plugin_options` refuses the start.

At start the whole table is checked, and any failure refuses the start (a table that differs from
the registered one never serves):
  * the top-level keys; the instance, topic and witness lists, grids and output names in their
    public order;
  * every attachment: valid base64, its sha256, a PNG made of IHDR / PLTE / IDAT / IEND chunks only,
    with correct chunk CRCs, 8-bit grey or 8-bit RGB; every attachment used by some reply; all
    attachments together at most ATTACHMENT_MAX_BYTES;
  * exactly one reply per cell (248 cells) and per redirect; its text non-empty UTF-8 in NFC with
    `\\n` line ends, no trailing spaces, at most TEXT_MAX_BYTES bytes; its values finite floats, as
    many as the test's public outputs (none for a question or a redirect);
  * every reply digest against the registry, and the table digest.
"""

import base64
import hashlib
import json
import math
import struct
import unicodedata
import zlib

import numpy as np

OWNER_STAFF = ("operations_engineer", "civil_engineer", "rov_pilot", "contracts_officer")
BIDDERS = tuple("bidder_T%02d" % i for i in range(1, 9))
INSTANCES = OWNER_STAFF + BIDDERS
OWNER_TOPICS = (
    "operations_during_inspection",
    "inspection_scope",
    "launch_and_recovery",
    "water_quality",
    "previous_campaign",
    "schedule_and_season",
    "data_provenance",
    "site_drawings",
    "acceptance_and_delivery",
    "anything_else",
)
BIDDER_TOPICS = (
    "trim_test_conditions",
    "thrust_test_conditions",
    "pressure_test_conditions",
    "battery_test_conditions",
    "as_built_configuration",
    "options_and_deviations",
)
TOPICS = OWNER_TOPICS + BIDDER_TOPICS
# test -> ((parameter, grid values in public order), ...); the grid literal is the canonical value
WITNESS_GRID = (
    ("weigh", ()),
    ("bollard_pull", (("bus_v", (12, 13, 14, 15, 16)), ("limiter", ("on", "off")))),
    ("battery_discharge", (("chamber_c", (10, 15, 23)), ("load_w", (100, 440)))),
    ("tow", (("speed_mps", (0.4, 0.6, 0.8)),)),
)
WITNESS_OUTPUTS = (
    ("weigh", ("air_mass_kg", "pool_weight_n", "pool_sg", "pool_temp_c")),
    ("bollard_pull", ("forward_n", "reverse_n", "supply_current_a")),
    ("battery_discharge", ("energy_wh", "internal_resistance_mohm")),
    ("tow", ("drag_n",)),
)
TESTS = tuple(t for t, _ in WITNESS_GRID)
GRID = {t: dict(params) for t, params in WITNESS_GRID}
OUTPUTS = dict(WITNESS_OUTPUTS)
STRING_PARAMETERS = frozenset({"limiter"})
TABLE_SCHEMA = "uw-desk-table/1"
TABLE_KEYS = frozenset(
    {
        "schema",
        "owner_staff",
        "bidders",
        "owner_topics",
        "bidder_topics",
        "witness_grid",
        "witness_outputs",
        "cells",
        "redirects",
        "attachments",
        "registry",
        "table_digest",
    }
)
REDIRECT_KINDS = ("owner_staff", "bidders")
TEXT_MAX_BYTES = 2000
ATTACHMENT_MAX_BYTES = 700_000  # the PNG bytes of all attachments together (so each at most that)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_ALLOWED_CHUNKS = (b"IHDR", b"PLTE", b"IDAT", b"IEND")
PNG_COLOUR_TYPES = (0, 2)  # 8-bit grey, 8-bit RGB
OPTIONS = frozenset()  # the plugin has no options

MESSAGES = {
    "bad_plan_shape": 'a desk plan is {"ask": <topic>} or {"witness": <test>, <parameters>}',
    "unknown_topic": "the topic is not one of the public desk topics",
    "unknown_test": "the witness test is not one of the public witness tests",
    "bad_parameters": "the witness parameters are not exactly the public ones for that test",
    "off_grid": "a witness parameter value is not one of its public values",
}


def _fmt(value):
    """The canonical text of a grid value inside a request key."""
    return repr(value) if isinstance(value, float) else str(value)


def request_key(plan):
    """The canonical request key of a validated plan ('ask:<topic>' or 'witness:<test>[:p=v...]')."""
    if "ask" in plan:
        return "ask:" + plan["ask"]
    test = plan["witness"]
    return ":".join(["witness", test] + ["%s=%s" % (p, _fmt(plan[p])) for p, _ in WITNESS_GRID[TESTS.index(test)][1]])


def _all_requests():
    keys = ["ask:" + t for t in TOPICS]
    for test, params in WITNESS_GRID:
        combos = [{}]
        for name, values in params:
            combos = [dict(c, **{name: v}) for c in combos for v in values]
        keys += [request_key(dict(c, witness=test)) for c in combos]
    return tuple(keys)


REQUESTS = _all_requests()  # 16 topics + 20 witness points = 36


def cell_keys():
    """The 248 cells in public order: owner staff x owner topics, bidders x (bidder topics, witness points)."""
    owner = ["%s|ask:%s" % (p, t) for p in OWNER_STAFF for t in OWNER_TOPICS]
    bidder_requests = ["ask:" + t for t in BIDDER_TOPICS] + [r for r in REQUESTS if r.startswith("witness:")]
    return tuple(owner + ["%s|%s" % (b, r) for b in BIDDERS for r in bidder_requests])


CELLS = cell_keys()


def _public_grid_json():
    return {t: {p: list(v) for p, v in params} for t, params in WITNESS_GRID}


def _public_outputs_json():
    return {t: list(o) for t, o in WITNESS_OUTPUTS}


SCHEMA = {
    "info": {
        "owner_staff": {p: "int" for p in OWNER_STAFF},
        "bidders": {b: "int" for b in BIDDERS},
        "owner_topics": {t: "int" for t in OWNER_TOPICS},
        "bidder_topics": {t: "int" for t in BIDDER_TOPICS},
        "witness": {
            "weigh": {"outputs": {o: "int" for o in OUTPUTS["weigh"]}},
            "bollard_pull": {
                "bus_v": "f64[5]",
                "limiter": {"on": "int", "off": "int"},
                "outputs": {o: "int" for o in OUTPUTS["bollard_pull"]},
            },
            "battery_discharge": {
                "chamber_c": "f64[3]",
                "load_w": "f64[2]",
                "outputs": {o: "int" for o in OUTPUTS["battery_discharge"]},
            },
            "tow": {"speed_mps": "f64[3]", "outputs": {o: "int" for o in OUTPUTS["tow"]}},
        },
        "text_max_bytes": "int",
        "attachment_max_bytes": "int",
    },
    "execute": {
        "instance": "enum:" + "|".join(INSTANCES),
        "request": "enum:" + "|".join(REQUESTS),
        "text_utf8": "u8[*]",
        "attachment_png": "u8[*]",
        "values": "f64[*]",
    },
}

# extra probes for the server's start-up validation table (it adds its generic probes)
VALIDATION_PROBES = {
    "plan": [{"ask": t} for t in TOPICS]
    + [{"witness": "weigh"}]
    + [{"witness": "bollard_pull", "bus_v": v, "limiter": lim} for v in (12, 16) for lim in ("on", "off")]
    + [{"witness": "battery_discharge", "chamber_c": c, "load_w": w} for c in (10, 23) for w in (100, 440)]
    + [{"witness": "tow", "speed_mps": s} for s in (0.4, 0.8)]
    + [
        {"ask": "INSPECTION_SCOPE"},
        {"ask": "inspection_scope "},
        {"ask": ""},
        {"ask": 1},
        {"ask": None},
        {"ask": ["inspection_scope"]},
        {"topic": "inspection_scope"},
        {"ask": "inspection_scope", "to": "rov_pilot"},
        {"ask": "inspection_scope", "witness": "weigh"},
        {"witness": "bollard_pull", "bus_v": 11, "limiter": "on"},
        {"witness": "bollard_pull", "bus_v": 14, "limiter": "auto"},
        {"witness": "bollard_pull", "bus_v": True, "limiter": "on"},
        {"witness": "bollard_pull", "bus_v": "14", "limiter": "on"},
        {"witness": "bollard_pull", "bus_v": 14},
        {"witness": "battery_discharge", "chamber_c": 5, "load_w": 100},
        {"witness": "battery_discharge", "chamber_c": 10, "load_w": 441},
        {"witness": "tow", "speed_mps": 0.5},
        {"witness": "tow"},
        {"witness": "weigh", "bus_v": 14},
        {"witness": "pressure_test"},
        {"witness": None},
        "inspection_scope",
        ["weigh"],
    ]
}


class TableError(ValueError):
    """The desk table breaks the contract; the plugin refuses to start."""


def _need(cond, message):
    if not cond:
        raise TableError(message)


def canonical_digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def f64_hex(values):
    return [struct.pack(">d", float(v)).hex() for v in values]


def reply_digest(cell, text, png, values):
    """The registered digest of one reply: cell key, text and PNG sha256, and the values' IEEE-754 bits."""
    return canonical_digest(
        {
            "cell": cell,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "attachment_sha256": hashlib.sha256(png).hexdigest(),
            "values_f64": f64_hex(values),
        }
    )


def table_digest(table):
    return canonical_digest({k: v for k, v in table.items() if k not in ("registry", "table_digest")})


def check_png(data):
    """Check one attachment's PNG structure; return its chunk kinds in order."""
    _need(data.startswith(PNG_SIGNATURE), "attachment is not a PNG")
    pos, kinds, ihdr = len(PNG_SIGNATURE), [], None
    while pos < len(data):
        _need(pos + 12 <= len(data), "truncated PNG chunk")
        length = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        _need(end <= len(data), "truncated PNG chunk")
        body = data[pos + 8 : pos + 8 + length]
        crc = int.from_bytes(data[pos + 8 + length : end], "big")
        _need(zlib.crc32(kind + body) == crc, "PNG chunk CRC mismatch")
        _need(kind in PNG_ALLOWED_CHUNKS, "attachment carries ancillary PNG chunks")
        if kind == b"IHDR":
            _need(not kinds and length == 13, "malformed PNG header")
            ihdr = body
        kinds.append(kind)
        pos = end
    _need(
        ihdr is not None and kinds[-1] == b"IEND" and kinds.count(b"IEND") == 1 and b"IDAT" in kinds,
        "malformed PNG chunk sequence",
    )
    width, height = int.from_bytes(ihdr[0:4], "big"), int.from_bytes(ihdr[4:8], "big")
    depth, colour = ihdr[8], ihdr[9]
    _need(width > 0 and height > 0, "malformed PNG header")
    _need(depth == 8 and colour in PNG_COLOUR_TYPES, "attachment is not 8-bit grey or RGB")
    return kinds


def check_text(text):
    """Check one reply text against the text rules; return its UTF-8 bytes."""
    _need(isinstance(text, str) and text, "reply text must be a non-empty string")
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate (JSON allows "\ud800")
        raise TableError("reply text is not valid UTF-8") from None
    _need(len(raw) <= TEXT_MAX_BYTES, "reply text exceeds %d bytes" % TEXT_MAX_BYTES)
    _need(unicodedata.normalize("NFC", text) == text, "reply text is not NFC")
    _need("\r" not in text, "reply text has a CR")
    _need(all(line == line.rstrip(" \t") for line in text.split("\n")), "reply text has trailing spaces")
    return raw


def check_values(values, n):
    _need(isinstance(values, list) and len(values) == n, "values: expected %d numbers" % n)
    for v in values:
        _need(isinstance(v, float) and math.isfinite(v), "values must be finite floats")
    return tuple(values)


def expected_values(cell):
    """How many values a cell's reply carries (0 for a question)."""
    request = cell.split("|", 1)[1]
    if request.startswith("ask:"):
        return 0
    return len(OUTPUTS[request.split(":")[1]])


def load_table(table):
    """Check a desk table completely; return ({cell: (text, png, values)}, {kind: text bytes})."""
    _need(isinstance(table, dict) and table.get("schema") == TABLE_SCHEMA, "not a %s table" % TABLE_SCHEMA)
    _need(set(table) == TABLE_KEYS, "table keys differ from the contract")
    _need(
        table["owner_staff"] == list(OWNER_STAFF)
        and table["bidders"] == list(BIDDERS)
        and table["owner_topics"] == list(OWNER_TOPICS)
        and table["bidder_topics"] == list(BIDDER_TOPICS),
        "instances or topics differ from the public lists",
    )
    _need(
        json.dumps(table["witness_grid"], sort_keys=True) == json.dumps(_public_grid_json(), sort_keys=True)
        and json.dumps(table["witness_outputs"], sort_keys=True) == json.dumps(_public_outputs_json(), sort_keys=True),
        "witness grid or outputs differ from the public lists",
    )
    attachments = table["attachments"]
    _need(isinstance(attachments, dict), "attachments must be an object")
    pngs, total = {}, 0
    for att_id, att in attachments.items():
        _need(
            isinstance(att, dict) and set(att) == {"png_b64", "sha256"},
            "attachment %s: keys differ from the contract" % att_id,
        )
        try:
            data = base64.b64decode(att["png_b64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise TableError("attachment %s: not base64" % att_id) from exc
        _need(hashlib.sha256(data).hexdigest() == att["sha256"], "attachment %s: sha256 mismatch" % att_id)
        try:
            check_png(data)
        except TableError as exc:
            raise TableError("attachment %s: %s" % (att_id, exc)) from None
        total += len(data)
        pngs[att_id] = data
    _need(total <= ATTACHMENT_MAX_BYTES, "attachments total %d bytes, more than %d" % (total, ATTACHMENT_MAX_BYTES))
    cells_in, redirects, registry = table["cells"], table["redirects"], table["registry"]
    _need(isinstance(cells_in, dict) and set(cells_in) == set(CELLS), "the table does not have exactly the 248 cells")
    _need(isinstance(redirects, dict) and set(redirects) == set(REDIRECT_KINDS), "redirects differ from the contract")
    reg_keys = set(CELLS) | {"redirect." + k for k in REDIRECT_KINDS}
    _need(isinstance(registry, dict) and set(registry) == reg_keys, "the registry does not list exactly the cells and redirects")
    used, out = set(), {}
    for cell in CELLS:
        rec = cells_in[cell]
        _need(
            isinstance(rec, dict) and set(rec) == {"text", "attachment", "values"},
            "cell %s: keys differ from the contract" % cell,
        )
        try:
            raw = check_text(rec["text"])
            values = check_values(rec["values"], expected_values(cell))
        except TableError as exc:
            raise TableError("cell %s: %s" % (cell, exc)) from None
        att = rec["attachment"]
        _need(att is None or (isinstance(att, str) and att in pngs), "cell %s: unknown attachment" % cell)
        png = pngs[att] if att is not None else b""
        used.add(att)
        _need(
            reply_digest(cell, rec["text"], png, values) == registry[cell],
            "cell %s: reply digest differs from the registry" % cell,
        )
        out[cell] = (raw, png, values)
    red = {}
    for kind in REDIRECT_KINDS:
        rec = redirects[kind]
        _need(isinstance(rec, dict) and set(rec) == {"text"}, "redirect %s: keys differ from the contract" % kind)
        try:
            raw = check_text(rec["text"])
        except TableError as exc:
            raise TableError("redirect %s: %s" % (kind, exc)) from None
        _need(
            reply_digest("redirect." + kind, rec["text"], b"", ()) == registry["redirect." + kind],
            "redirect %s: reply digest differs from the registry" % kind,
        )
        red[kind] = raw
    _need(not set(pngs) - used, "unused attachments in the table")
    _need(table_digest(table) == table["table_digest"], "table digest mismatch")
    return out, red


def _frozen(data):
    arr = np.frombuffer(bytes(data), dtype=np.uint8).copy()
    arr.setflags(write=False)
    return arr


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class DeskPlugin:
    SCHEMA = SCHEMA
    MESSAGES = MESSAGES
    VALIDATION_PROBES = VALIDATION_PROBES

    def __init__(self, instances, api):
        unknown = sorted(set(api.options) - OPTIONS)
        if unknown:
            raise ValueError("unknown plugin options: %s" % ", ".join(map(str, unknown)))
        self._bad_request = api.BadRequest
        cells, redirects = load_table(instances)
        self._replies = {
            cell: (_frozen(raw), _frozen(png), tuple(values)) for cell, (raw, png, values) in cells.items()
        }
        self._redirects = {kind: _frozen(raw) for kind, raw in redirects.items()}
        self._empty = _frozen(b"")

    def instance_ids(self):
        return list(INSTANCES)

    def info(self):
        return {
            "owner_staff": {p: i for i, p in enumerate(OWNER_STAFF)},
            "bidders": {b: i for i, b in enumerate(BIDDERS)},
            "owner_topics": {t: i for i, t in enumerate(OWNER_TOPICS)},
            "bidder_topics": {t: i for i, t in enumerate(BIDDER_TOPICS)},
            "witness": {
                "weigh": {"outputs": {o: i for i, o in enumerate(OUTPUTS["weigh"])}},
                "bollard_pull": {
                    "bus_v": [float(v) for v in GRID["bollard_pull"]["bus_v"]],
                    "limiter": {v: i for i, v in enumerate(GRID["bollard_pull"]["limiter"])},
                    "outputs": {o: i for i, o in enumerate(OUTPUTS["bollard_pull"])},
                },
                "battery_discharge": {
                    "chamber_c": [float(v) for v in GRID["battery_discharge"]["chamber_c"]],
                    "load_w": [float(v) for v in GRID["battery_discharge"]["load_w"]],
                    "outputs": {o: i for i, o in enumerate(OUTPUTS["battery_discharge"])},
                },
                "tow": {
                    "speed_mps": [float(v) for v in GRID["tow"]["speed_mps"]],
                    "outputs": {o: i for i, o in enumerate(OUTPUTS["tow"])},
                },
            },
            "text_max_bytes": TEXT_MAX_BYTES,
            "attachment_max_bytes": ATTACHMENT_MAX_BYTES,
        }

    def validate_plan(self, plan):
        bad = self._bad_request
        if not isinstance(plan, dict):
            raise bad("bad_plan_shape")
        if "ask" in plan:
            if set(plan) != {"ask"}:
                raise bad("bad_plan_shape")
            topic = plan["ask"]
            if not isinstance(topic, str) or topic not in TOPICS:
                raise bad("unknown_topic")
            return {"ask": topic}
        if "witness" in plan:
            test = plan["witness"]
            if not isinstance(test, str) or test not in TESTS:
                raise bad("unknown_test")
            grid = GRID[test]
            if set(plan) != {"witness"} | set(grid):
                raise bad("bad_parameters")
            out = {"witness": test}
            for name, values in grid.items():
                value = plan[name]
                if name in STRING_PARAMETERS:
                    ok = isinstance(value, str) and value in values
                    canonical = value
                else:
                    ok = _number(value) and value in values
                    canonical = values[values.index(value)] if ok else None
                if not ok:
                    raise bad("off_grid")
                out[name] = canonical
            return out
        raise bad("bad_plan_shape")

    def execute(self, instance, plan):
        request = request_key(plan)
        cell = "%s|%s" % (instance, request)
        found = self._replies.get(cell)
        if found is None:  # a valid plan for the other kind of instance: the fixed redirect (charged)
            kind = "owner_staff" if instance in OWNER_STAFF else "bidders"
            text, png, values = self._redirects[kind], self._empty, ()
        else:
            text, png, values = found
        return {
            "instance": instance,
            "request": request,
            "text_utf8": text,
            "attachment_png": png,
            "values": list(values),
        }


def make_plugin(instances, api):
    return DeskPlugin(instances, api)
