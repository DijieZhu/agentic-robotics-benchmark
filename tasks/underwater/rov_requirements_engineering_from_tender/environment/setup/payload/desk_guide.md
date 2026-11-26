# The enquiry and witness desk

You can put questions to the Owner's staff and to the eight bidders, and you can have a bidder run a factory test on
its tendered unit while you witness it. All of it goes through one service, the desk, with `labclient.py`. This page
lists who answers, the topics, the witnessed tests with their parameters and readings, what a request costs, and how
to use the client.

## 1. Who answers

| side | ids |
|---|---|
| the Owner's staff | `operations_engineer`, `civil_engineer`, `rov_pilot`, `contracts_officer` |
| the bidders | `bidder_T01`, `bidder_T02`, `bidder_T03`, `bidder_T04`, `bidder_T05`, `bidder_T06`, `bidder_T07`, `bidder_T08`: the bidder of bid `T-01`, ..., `T-08` |

Each of them speaks for their own area, in their own words and at their own precision: a reply may give a number,
say it in words, or point to a page instead of quoting it.

## 2. Questions

A question is one id and one topic: the plan `{"ask": "<topic>"}`. The topics are two fixed lists, one for each side.
There is no free-text question and no follow-up: a reply depends on who is asked and on the topic, and on nothing
else.

Topics for the Owner's staff:

| topic | what it covers |
|---|---|
| `operations_during_inspection` | how the plant is operated while an inspection is under way |
| `inspection_scope` | what is to be inspected |
| `launch_and_recovery` | how the vehicle is launched and recovered |
| `water_quality` | the quality of the water in the reservoir |
| `previous_campaign` | the previous inspection campaign |
| `schedule_and_season` | when the inspection is done |
| `data_provenance` | where the site records come from |
| `site_drawings` | the drawings of the intake tower |
| `acceptance_and_delivery` | acceptance and delivery of the vehicle |
| `anything_else` | anything the person wishes to add |

Topics for a bidder:

| topic | what it covers |
|---|---|
| `trim_test_conditions` | the conditions of the bidder's trim and weighing test |
| `thrust_test_conditions` | the conditions of the bidder's bollard-pull test |
| `pressure_test_conditions` | the conditions of the bidder's pressure test |
| `battery_test_conditions` | the conditions of the bidder's battery discharge test |
| `as_built_configuration` | the tendered unit as it is built |
| `options_and_deviations` | the priced options and the declared deviations |

Every id has a reply on every topic of its side, and the desk never refuses a valid question. A member of staff who
is asked about something outside their area says so and names who to ask. A reply may refer you to a document
instead of answering. Someone who has nothing to add on a topic says that. Such a reply costs a request like any
other.

A reply to a question is a short text, sometimes with one page attached as a PNG image.

## 3. Witnessed tests

A witnessed test is run in the bidder's own facility, on the bidder's tendered unit: the unit as tendered, with none
of the priced options fitted, ballasted as the works left it. The test returns its readings and nothing else. It
gives no verdict, and nothing about the Owner's site ever appears in a reply. The plan is
`{"witness": "<test>", <parameters>}`: every parameter of the test is given, and each takes one of its listed
values.

| test | plan | parameters |
|---|---|---|
| `weigh` | `{"witness": "weigh"}` | none |
| `bollard_pull` | `{"witness": "bollard_pull", "bus_v": V, "limiter": L}` | `bus_v`: voltage of the bench supply in V, one of 12, 13, 14, 15, 16; `limiter`: `"on"` or `"off"` (below) |
| `battery_discharge` | `{"witness": "battery_discharge", "chamber_c": C, "load_w": W}` | `chamber_c`: temperature of the climatic chamber in degC, one of 10, 15, 23; `load_w`: constant-power load in W, one of 100, 440 |
| `tow` | `{"witness": "tow", "speed_mps": S}` | `speed_mps`: speed of the tow carriage in m/s, one of 0.4, 0.6, 0.8 |

A numeric parameter may be written as an integer or as a decimal of the same value: `14` and `14.0` are the same
request. A string (`"14"`) and a boolean are not numbers.

**The facility.** A bidder runs a witnessed test inside its own facility, as section 1 of its delivery statement
declares it. A bidder does not run a test with a parameter value of the list that lies outside the range which it
declares there. The desk accepts such a request and charges it like any other. The reply is one fixed text of the
works, which says that the test was not run, with no readings (`values` is empty) and no attachment.

**`limiter`.** `"on"` is the unit as delivered: the bench supply feeds the battery interface, and the firmware's limit
on the battery current, which is the factory current preset, is active. `"off"`: the four horizontal thrusters run
at full throttle with no current budget. The word has the same meaning in the facility quantities
`bollard_thrust_n` and `bollard_bus_current_a` of TS-01 Part D. The bidders' factory bollard reports (TR-2), this
witnessed test and those two facility quantities are one calculation, `bench_bollard_thrust_n` and
`bollard_bus_current_a` of `tender/evaluation_basis.py`.

The readings come back in `values`, as numbers, in this order:

| test | reading | unit | read to | what it is |
|---|---|---|---|---|
| `weigh` | `air_mass_kg` | kg | 0.001 | mass of the tendered unit in air |
| | `pool_weight_n` | N | 0.01 | weight of the unit in the water of the bidder's trim tank; positive when the unit is heavier than the water it displaces (the facility quantity `tank_net_buoyancy_n` with the opposite sign) |
| | `pool_sg` | 1 | 0.0001 | specific gravity of the tank water: its density divided by 1000 kg/m3 |
| | `pool_temp_c` | degC | 0.1 | temperature of the tank water |
| `bollard_pull` | `forward_n` | N | 0.1 | bollard thrust ahead: `bench_bollard_thrust_n("surge_ahead", bus_v, limiter, unit)` |
| | `reverse_n` | N | 0.1 | bollard thrust astern: `bench_bollard_thrust_n("surge_astern", bus_v, limiter, unit)` |
| | `supply_current_a` | A | 0.1 | current of the bench supply during the astern pull: `bollard_bus_current_a("surge_astern", bus_v, limiter, unit)` |
| `battery_discharge` | `energy_wh` | Wh | 0.1 | energy the tendered battery pack delivers in the chamber at `chamber_c`, from full charge, at the constant power `load_w`, down to a terminal voltage of 12.0 V: `battery_delivered_energy_wh(pack, load_w, chamber_c, 12.0)` |
| | `internal_resistance_mohm` | mOhm | 0.1 | steady-state DC resistance of the pack at `chamber_c`, after 60 s at load: `pack_dcir_mohm(pack, chamber_c)` |
| `tow` | `drag_n` | N | 0.01 | surge drag of the unit, towed at `speed_mps` without its tether: `tow_drag_n(unit, speed_mps)` |

The functions are those of `tender/evaluation_basis.py`; `unit` and `pack` stand for the bidder's tendered unit and
its battery pack as they really are. A reading is the value of the function, read at the resolution of the
instrument (the column "read to").

## 4. What a request costs

- You have **14 requests** in total: questions and witnessed tests together, to both sides together.
- A request is charged when the desk accepts it, before the reply is sent. Nothing is refunded: not after a dropped
  connection, a timeout or a crash of your program.
- The replies are fixed. The same request to the same id returns the same reply and costs one more request.
- **A request to the wrong side is charged.** A bidder topic or a witnessed test addressed to a member of the
  Owner's staff, and an Owner topic addressed to a bidder, are well-formed requests. Each costs one request and is
  answered with a fixed text that points to the other side, without readings and without an attachment. The two
  texts are quoted at the end of this section.
- **A witnessed test that a bidder does not run is charged.** A parameter value of the list that lies outside the
  facility which the bidder declares (section 3) makes a well-formed request. It costs one request and is answered
  with the works' fixed text, without readings and without an attachment.
- Free: `hello`, `budget`, `close`, and every request the desk rejects (an unknown id, an unknown topic or test, a
  missing or extra parameter, a parameter value that is not on its list, a malformed request, too many
  connections).
- When the budget is used up, a request is answered with the error `budget_exhausted`.
- The desk keeps nothing for you. Save every text, every attached page and every reading when it arrives.
- The desk writes every request it accepts into a log of its own, under the canonical key of the request: the
  `request` string of section 5. You cannot read that log.
- The desk exists only during your session. Which requests you made is not graded.

The reply of a member of the Owner's staff to a bidder topic or a witnessed test:

> That is a matter for the bidders. Questions about a bid, and witnessed tests, are addressed to the bidder concerned.

The reply of a bidder to an Owner topic:

> That is a matter for the Owner. Questions about the site and the tender are addressed to the Owner's staff.

## 5. The client

`labclient.py` is in your home directory and needs only the Python standard library. It finds the desk's Unix
socket through the environment variable `ALE_LAB_SOCKET`, which is set in your sandbox:
`/run/ale-desk/lab.sock`. Should a shell of yours lack the variable, set it to that path, or pass the path:
`LabClient("/run/ale-desk/lab.sock")`.

From the shell:

```
/opt/venv/bin/python3 labclient.py hello        # ids, topics, tests with their parameter values and readings, limits (free)
/opt/venv/bin/python3 labclient.py budget       # requests left (free)
echo '{"ask": "<topic>"}' | /opt/venv/bin/python3 labclient.py execute <id> -                             # one question
echo '{"witness": "<test>", "<parameter>": <value>}' | /opt/venv/bin/python3 labclient.py execute <bidder id> -   # one witnessed test
```

Each command prints the desk's reply as one JSON object. For a request it is

```json
{
  "ok": true,
  "op": "execute",
  "instance": "<id>",
  "outcome": {
    "instance": "<id>",
    "request": "<the request as the desk read it>",
    "text_utf8": {"dtype": "uint8", "shape": ["<n>"], "b64": "<base64 of the reply text, UTF-8>"},
    "attachment_png": {"dtype": "uint8", "shape": ["<m>"], "b64": "<base64 of a PNG file; empty when m is 0>"},
    "values": ["<the readings of a witnessed test, as numbers; an empty list for a question>"]
  },
  "remaining": "<requests left>"
}
```

where `<n>`, `<m>` and the requests left are integers. `request` is the canonical key of the request: `ask:<topic>`
for a question and `witness:<test>` followed by `:<parameter>=<value>` for each parameter of a witnessed test, in
the order of the table of section 3, each value as that table lists it, the word of `limiter` without its
quotation marks. The text is at most 2000 bytes. A reply without an attachment has the `shape`
`[0]`; an attached page is one PNG image. The reply to a witnessed test carries its readings in `values` and a
short text; when the bidder does not run the test (section 3), `values` is empty.

From Python:

```python
import base64
from labclient import LabClient, LabError

def request(who, plan):
    """One request: (reply text, PNG bytes, readings, requests left); the bytes and the readings are empty when
    there are none. Raises LabError when the desk answers with an error."""
    with LabClient() as desk:
        reply = desk.request({"op": "execute", "instance": who, "plan": plan})
    outcome = reply["outcome"]
    text = base64.b64decode(outcome["text_utf8"]["b64"]).decode("utf-8")
    png = base64.b64decode(outcome["attachment_png"]["b64"])
    return text, png, outcome["values"], reply["remaining"]

def request_and_save(who, plan):
    text, png, values, left = request(who, plan)
    stem = "desk_%s_%s" % (who, "_".join(str(v) for v in plan.values()))
    with open(stem + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n" + repr(values) + "\n")
    if png:
        with open(stem + ".png", "wb") as fh:
            fh.write(png)
    return text, values, left
```

Inside `with LabClient() as desk:`, `desk.hello()` returns the ids (`instances`) and, under `info`, the two sides
(`owner_staff`, `bidders`), their topics (`owner_topics`, `bidder_topics`) and the witnessed tests (`witness`: for
each test the values of its parameters and, under `outputs`, the position of each reading in `values`);
`desk.budget()` returns the number of requests left. The `with` block closes the connection when it ends. At most
2 connections can be open at the same time, and the desk closes a connection on which
nothing was sent for 900 s; a request on a new connection costs the same as on an old one. An
error reply raises `LabError`; its `code` is one of:

| code | meaning | charged |
|---|---|---|
| `bad_instance` | the id is not one of the twelve | no |
| `bad_plan` | the plan is not a question or a witnessed test as sections 2 and 3 define them; the message says what is wrong | no |
| `bad_request`, `bad_json`, `too_complex`, `line_too_long`, `unknown_op` | the request is malformed | no |
| `unsupported` | the desk does not have this operation | no |
| `busy` | too many connections are open at the same time | no |
| `rate_limited` | too many requests in one second | no |
| `budget_exhausted` | no requests are left | no |
| `internal` | the desk failed while it answered | yes, when the request had been accepted |

The client is a general one. Its `reset` and `step` do not exist at this desk, and where its own texts say "trial"
or "instance", read "request" and "id".
