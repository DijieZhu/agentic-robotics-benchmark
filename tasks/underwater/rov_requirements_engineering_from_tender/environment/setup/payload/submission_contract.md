# Submission contract

You deliver three files in `/home/user/submission/`. This page is their contract: what each file holds, every rule
its structure must meet, and how each scored item of `instruction.md` is judged.

| file | what it is | section |
|---|---|---|
| `evaluation.json` | your verdict on each of the eight bids, your predictions for the bids you accept, and the award | 2 |
| `spec_review.json` | your label for every clause of TS-01 Part B | 3 |
| `acceptance_schedule.json` | the factory acceptance schedule for the configuration you award | 4 |

Each file is a regular file (not a link), UTF-8 without a byte-order mark, of at most 128 kB (131072 bytes), and
strict JSON (section 7). The JSON schemas in `schemas/` are the machine-readable form of the shapes;
`schema_check.py` checks them and the rules a schema cannot express (section 8). A file that breaks a rule of this
page zeroes the whole result.

## 1. The words this page uses

Everything the grader computes, it computes with `tender/evaluation_basis.py`, the same file you have, from the site
as it really is, the units as they really are and the makers' page values as corrected by their errata. The
grader's world holds no physics that is not in that file.

- A **configuration** is a bid together with a subset of its two priced options: none, either one, or both. A bid
  has four configurations. The **price** of a configuration is the base price of the bid plus the prices of the
  options in the subset, as the bid's price sheet gives them.
- The **tendered unit** of a configuration is the unit the bidder tenders, in the state in which it is tendered,
  with the options of the subset fitted.
- The **margin** of a configuration on a need is what `evaluation_basis.rows` returns for its tendered unit on the
  real site. Zero or more meets the need.
- **N-01 has two named rules.** In the evaluation under TS-01 Part C, N-01 is judged by `n01_trim_reachable`: the
  best trim that the bid's own ballast kit can reach on site, by taking out removable pieces that are fitted or by
  adding the loose spares that are supplied. In the acceptance under TS-01 Part D, N-01 is judged by
  `n01_as_delivered`: the unit as it is ballasted and sealed at the works, with no trimming on site and no spares.
  Sections 2, 3 and 5.1 to 5.4 use the first rule; sections 4 and 5.5 use the second.
- Each need has a **tolerance**. It reflects the precision with which the documents and the replies of the desk
  allow a margin to be established. The tolerances are not published. A configuration or a unit **clearly fails** a
  need when its margin is below zero by more than the tolerance, and **clearly meets** it when its margin is above
  zero by more than the tolerance. In between it does neither.
- A configuration is **fit** when it clearly fails no need.
- A bid is **acceptable** when at least one of its configurations is fit. Its **cheapest fit configuration** is the
  fit configuration with the lowest price. A bid none of whose configurations is fit is **to be rejected**.
- For a bid to be rejected, a need is a **valid ground** when every configuration of the bid clearly fails it. A
  need that is not a valid ground is **borderline** for that bid when no configuration of the bid clearly meets it.

## 2. `evaluation.json`

```json
{
  "bids": {
    "<a bid id>": {
      "verdict": "accept",
      "options": ["<an option id of that bid>"],
      "site": {
        "as_tendered_net_buoyancy_n": -6.8,
        "depth_rating_margin_m": 41.27,
        "station_thrust_ratio": 2.36,
        "dive_energy_reserve_frac": 0.88,
        "tether_reach_margin_m": 118.5,
        "pitch_authority_ratio": 6.24,
        "wet_temperature_margin_c": 9.6,
        "launch_mass_margin_kg": 2.46
      }
    },
    "<another bid id>": {
      "verdict": "reject",
      "grounds": ["<a need id>"]
    }
  },
  "award": "<the id of a bid you accept>"
}
```

The ids between angle brackets are placeholders, and every number is made up to show the format. Read nothing about
the site or about any bid from them.

- The file has exactly the two keys `bids` and `award`.
- `bids` has exactly the eight bid ids `T-01` to `T-08`. Each value has one of two shapes, with exactly these keys:
  - `{"verdict": "accept", "options": [...], "site": {...}}`. `options` is the subset of the bid's two option ids
    that you accept it with: a list of zero, one or two ids of that bid's price sheet, without repeats, in any
    order. `site` has exactly the eight keys shown above, each a finite number.
  - `{"verdict": "reject", "grounds": [...]}`. `grounds` is a list of one or two different need ids, `N-01` to
    `N-12`.
- `award` is the id of a bid that you accept, or `null` (the JSON value, not a string) when you award to no one.
  The **awarded configuration** is that bid with the `options` you gave it.

What the verdicts mean:

- **accept** says that the bid is acceptable and that `options` is the option subset of its cheapest fit
  configuration.
- **reject** says that the bid is to be rejected, and `grounds` names why: needs that are valid grounds for it.
- `site` is your prediction, for the configuration you accept, of the eight values that
  `evaluation_basis.site_values` returns for its tendered unit on the real site. `as_tendered_net_buoyancy_n` is
  the net buoyancy of the tendered unit in the site water as it is ballasted, without trimming, in N, positive when
  it floats; `station_thrust_ratio` is available thrust over drag; `pitch_authority_ratio` is the margin of N-06
  plus one; the other five are the margins of N-02, N-04, N-05, N-07 and N-08 (`tender/evaluation_basis.md`,
  section 6).

## 3. `spec_review.json`

```json
{
  "clauses": {
    "B-01": "<label>",
    "B-02": "<label>"
  }
}
```

The file has exactly the key `clauses`. `clauses` has every clause id of TS-01 Part B exactly once and no other
key. Each value is one of `"keep"`, `"lax"`, `"strict"`, `"lax_and_strict"`.

Part B names, for each clause, the need it supports, or none. A bid's compliance matrix states, for each clause,
Comply or Deviate; the **statement of a configuration** on a clause is that of the matrix, with the changes the
matrix gives for the options of the subset. A clause is judged against the real site:

- It is **too lax** when some configuration, of any bid, states Comply on it and clearly fails the need the clause
  supports: complying with the clause does not secure the need. A clause that supports no need is never too lax.
- It is **too strict** when the cheapest fit configuration of some acceptable bid does not state Comply on it: the
  clause would turn away a configuration that is fit.

| label | too lax | too strict |
|---|---|---|
| `keep` | no | no |
| `lax` | yes | no |
| `strict` | no | yes |
| `lax_and_strict` | yes | yes |

You give the label only. The review does not ask for a corrected number or for a reason.

## 4. `acceptance_schedule.json`

```json
{
  "for_bid": "<the bid you award>",
  "for_options": ["<an option id you gave it>"],
  "clauses": [
    {"id": "AC-01", "quantity": "tether_diameter_mm", "comparator": "<=", "bound": 8.25, "unit": "mm",
     "conditions": {}},
    {"id": "AC-02", "quantity": "proof_pressure_bar", "comparator": ">=", "bound": 6.35, "unit": "bar",
     "conditions": {"hold_min": 112.5}},
    {"id": "AC-03", "quantity": "tow_drag_n", "comparator": "<=", "bound": 23.75, "unit": "N",
     "conditions": {"speed_mps": 0.53}}
  ]
}
```

Again the ids between angle brackets are placeholders and every number is made up. The three clauses show the
format of a clause without conditions and of clauses with one; they are not a suggestion of what a schedule should
hold. Read nothing about the site, the bids or a sensible schedule from them.

### 4.1 Shape

- The file has exactly the three keys `for_bid`, `for_options` and `clauses`.
- `for_bid` equals `award` of `evaluation.json`. `for_options` holds the same option ids as the `options` of that
  bid in `evaluation.json`, in any order. When `award` is `null`, `for_bid` is `null`, `for_options` is `[]` and
  `clauses` is `[]`.
- `clauses` is a list of 0 to 24 clauses. A clause has exactly these six keys:

| key | value |
|---|---|
| `id` | a string matching `^AC-[0-9]{2}$`, different for every clause of the file. Never scored |
| `quantity` | a facility quantity of the dictionary (section 4.2) |
| `comparator` | `">="` or `"<="`, and one that the dictionary allows for the quantity |
| `bound` | a finite number, in the unit of the quantity |
| `unit` | exactly the unit string the dictionary gives for the quantity |
| `conditions` | an object with exactly the condition names of the quantity (`{}` when it has none), each value as section 4.3 defines it |

- The same quantity with the same comparator and equal conditions may appear only once (12 and 12.0 are equal). A
  window is two clauses: the same quantity and conditions with one `">="` clause and one `"<="` clause, for a
  quantity that allows both.

### 4.2 The dictionary of facility quantities

A clause is one line of the acceptance at the supplier's works: *measured or computed for the delivered unit in the
supplier's facility, under the conditions the clause writes out, the quantity satisfies the bound.* The dictionary
is closed: 15 quantities in six groups, one group for each kind of test the works can run. No quantity names a part,
a maker or any other identity: a clause can only bound what a unit measures or does. No free text binds the
supplier. Conditions are written out in full and nothing has a default. What the schedule does not demand, the
supplier does not owe.

| group | quantity | unit | conditions | comparators | what it is |
|---|---|---|---|---|---|
| weighing | `air_mass_kg` | `kg` | (none) | `>=`, `<=` | mass of the unit in air |
| weighing | `tank_net_buoyancy_n` | `N` | (none) | `>=`, `<=` | net buoyancy of the unit in the water of the supplier's own trim tank, positive when it floats |
| weighing | `displaced_volume_l` | `L` | (none) | `>=`, `<=` | displaced volume of the unit |
| propulsion | `bollard_thrust_n` | `N` | `axis`, `supply_v`, `limiter` | `>=` | static thrust along the axis in the bollard pool, with the bench supply at the battery interface |
| propulsion | `bollard_bus_current_a` | `A` | `axis`, `supply_v`, `limiter` | `>=` | current of the bench supply in that test |
| energy | `battery_delivered_energy_wh` | `Wh` | `load_w`, `chamber_c`, `cutoff_v` | `>=` | energy the pack delivers on the cycler in the climatic chamber, from full charge, at constant power, down to the cut-off voltage |
| energy | `pack_dcir_mohm` | `mOhm` | `chamber_c` | `<=` | steady-state DC resistance of the pack in the climatic chamber, after 60 s at load |
| energy | `hotel_power_w` | `W` | `lights_pct` | `<=` | hotel electrical load at the given light setting |
| pressure | `proof_pressure_bar` | `bar` | `hold_min` | `>=` | proof pressure that the weakest pressure part holds in the pressure vessel |
| hull and tether | `tow_drag_n` | `N` | `speed_mps` | `<=` | surge drag of the unit in the tow tank, without its tether |
| hull and tether | `tether_length_m` | `m` | (none) | `>=` | length of the tether |
| hull and tether | `tether_diameter_mm` | `mm` | (none) | `<=` | diameter of the tether |
| sensing and work | `camera_hfov_deg` | `deg` | `medium` | `>=` | horizontal field of view of the forward camera through its port |
| sensing and work | `sonar_min_range_m` | `m` | (none) | `<=` | minimum range of the imaging sonar |
| sensing and work | `gripper_opening_mm` | `mm` | (none) | `>=` | opening of the gripper jaw |

The unit strings are plain ASCII: `kg`, `N`, `L`, `A`, `Wh`, `mOhm`, `W`, `bar`, `m`, `mm`, `deg`. The comparators
are fixed by the dictionary: a quantity with one comparator can be bounded on that side only, and a clause with the
other comparator is a structural error. `tender/evaluation_basis.md` (section 8) gives the formula of every
quantity; `evaluation_basis.facility_value(quantity, conditions, vehicle)` computes it.

### 4.3 Condition values

| condition | value |
|---|---|
| `axis` | `"surge_ahead"` or `"surge_astern"` |
| `supply_v` | voltage of the bench supply at the battery interface, in V; a finite number |
| `limiter` | `"on"`: the unit as delivered, with the firmware's limit on the battery current, which is the factory current preset, active. `"off"`: the four horizontal thrusters at full throttle, with no current budget |
| `load_w` | constant-power load of the cycler, in W; a finite number |
| `chamber_c` | temperature of the climatic chamber, in degC; a finite number |
| `cutoff_v` | terminal voltage at which the discharge ends, in V; a finite number |
| `lights_pct` | light setting, in percent of full output; a finite number |
| `hold_min` | hold time of the proof pressure, in minutes; a finite number |
| `speed_mps` | speed of the tow carriage, in m/s; a finite number |
| `medium` | what is in front of the camera port: `"air"` or `"tank_water"` |

A number is a JSON number; a string such as `"12"` and a boolean are not numbers. The structure check does not look
at whether the supplier's facility can run the test that a clause writes: that is judged with the schedule (section
5.5). Each bidder's delivery statement gives the envelope of its facility in section 1; `supply_v`, `chamber_c`,
`load_w`, `hold_min` and `speed_mps` have to lie inside it.

## 5. How each item is judged

`instruction.md` gives the weights. Each item is a number from 0 to 1.

### 5.1 `verdicts`

The item is the product of two shares.

- **Accept side**: of the acceptable bids, the share that you accept with exactly the options of their cheapest fit
  configuration. An acceptable bid that you reject, or accept with another option subset, counts nothing.
- **Reject side**: of the bids to be rejected, the share that you reject on sound grounds. Your grounds are sound
  when, the borderline needs among them left aside, at least one need remains and every need that remains is a
  valid ground. **Every ground you cite must hold**: one cited need that is neither a valid ground nor borderline
  spoils the rejection, whatever else you cite. A bid to be rejected that you accept counts nothing.

### 5.2 `award`

The award is **self-consistent** when the awarded configuration is fit and no configuration that you accept has a
lower price: **the award must go to the cheapest of the configurations you accept.** An award that is not
self-consistent, and no award, score 0 here and in `acceptance_schedule`.

For a self-consistent award the item is a price factor times a grounds factor.

- **Price factor**: 1 when the awarded configuration is the cheapest fit configuration of all bids. A dearer one
  earns less, in a straight line down to 0 at a price 8 % above that cheapest price.
- **Grounds factor**: 1 when every bid to be rejected whose base price is below the price of the awarded
  configuration is rejected by you on sound grounds (section 5.1), and 0 when one of them is not. It is 1 when
  there is no such bid.

### 5.3 `site_prediction`

For each acceptable bid that you accept with exactly the options of its cheapest fit configuration, each of the
eight `site` values is compared with the value `evaluation_basis.site_values` gives for the tendered unit of that
configuration on the real site. A value earns full credit when its error is within the tolerance of that value, no
credit from five times that tolerance on, and in between it falls in a straight line. These tolerances are not
published either: they start from the tolerances of section 1 and are wider where the documents fix a value less
precisely. The bid earns the mean over its eight values. Every other acceptable bid earns 0.

The item is the mean over the acceptable bids, multiplied by the reject side of section 5.1. It is 0 when
`verdicts` is 0.

### 5.4 `spec_review`

With the true labels by the definitions of section 3: the number of clauses whose true label is not `keep` and
which you labelled correctly, minus the number of clauses to which you gave a label other than `keep` that is not
the true label, divided by the number of clauses whose true label is not `keep`; not below 0. It is 0 when
`verdicts` is 0.

### 5.5 `acceptance_schedule`

Under TS-01 Part D the supplier may deliver any unit of the delivery band it has declared (delivery statement,
section 2) that passes every clause of your schedule, and the delivered state is the launch state. The grader
therefore takes every unit of the delivery band of the awarded configuration and computes for it

- the value of the facility quantity of each of your clauses, under the conditions of the clause, at the awarded
  bidder's works (`evaluation_basis.facility_value`); the unit **passes your schedule** when every clause holds for
  it, value >= bound for `">="` and value <= bound for `"<="`;
- its margins on the real site as delivered, with N-01 by `n01_as_delivered`.

The schedule **cannot be contracted** when no unit of the band passes it, or when a condition of one of its clauses
lies outside the facility envelope of the awarded bidder (section 4.3). Then the item is 0. Otherwise it is
protection times economy:

- **Protection**: of the needs that at least one unit of the band clearly fails, the share that your schedule
  protects. A need is protected when no unit that passes your schedule clearly fails it. Protection is 1 when no
  unit of the band clearly fails any need.
- **Economy**: of the clearly good units of the band, the share that passes your schedule. A unit is clearly good
  when it is above zero on every need by at least twice the tolerance.

The item is judged over all the units that pass your schedule, and not only over the one the supplier would choose
first. It is 0 unless the award is self-consistent (section 5.2). An award to a fit configuration that is not the
cheapest fit one of all bids is judged on the delivery band of the configuration you awarded.

### 5.6 The result

The result is the weighted sum of the five items, and 0 when a file breaks a rule of this page or the submission
breaks the integrity rule of `instruction.md`.

## 6. Every limit in one table

| limit | value |
|---|---|
| size of each file | 128 kB (131072 bytes) |
| bids in `evaluation.json` | exactly `T-01` to `T-08` |
| `options`, `for_options` | 0 to 2 option ids of the bid, no repeats |
| `grounds` | 1 or 2 different need ids, `N-01` to `N-12` |
| `site` | exactly the eight keys of section 2, finite numbers |
| `spec_review.json` | exactly the clause ids of TS-01 Part B |
| clauses of the schedule | 0 to 24 |
| clause id | `^AC-[0-9]{2}$`, unique in the file |
| option id | `^O-[A-Z0-9]{1,6}$`, of the bid's price sheet |

## 7. Writing the JSON

The three files are read strictly. Each of these zeroes the result:

- a file that is not UTF-8, or that starts with a byte-order mark;
- a top level that is not an object, a key that this page does not define, or a missing key;
- a duplicate key in any object;
- `NaN`, `Infinity` or `-Infinity`, a number that is not finite once read, or an integer literal of thousands of
  digits;
- a string where a number is required, or a boolean where a number is required.

Whole numbers may be written with or without a decimal point: `12` and `12.0` are the same number.

## 8. Checking the structure

```
/opt/venv/bin/python3 schema_check.py
```

checks that the three files exist in `/home/user/submission/`, are regular files within the size limit, parse
strictly, match their schemas and follow the rules of sections 2 to 4 that a schema cannot express (the options are
options of the bid, `award` is a bid you accept, `for_bid` and `for_options` agree with `evaluation.json`, no
clause is repeated). It knows nothing of the site, of the units or of any delivery band. It looks at structure only
and says nothing about any score.
