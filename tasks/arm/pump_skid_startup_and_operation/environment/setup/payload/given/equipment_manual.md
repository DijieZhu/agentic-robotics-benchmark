# Equipment manual: the transfer and dosing skid

How every device on the skid is built, operated and read. The skid is a series design: the process (`pid.png`), the
series tags and the conventions here hold on every skid of the series, while each site builds and numbers its own skid
within the choices this manual lists (section 6). The instrument catalog (`~/given/catalog/`, `catalog.md`) has the
datasheet drawings of every dial face and scale plate the series fits and two sample tag plates.

## 1. Frame and conventions

* The skid's front plane is y = 0 (world frame of the cell, metres, z up, the floor at z = 0, x along the skid from 0
  to 5 m). Every handle, button, dial, scale and tag plate faces -y, the side of the robot's rail.
* **Turning sense.** Every operable hinge turns about the axis (0, -1, 0), the axis pointing at the operator. A
  positive turn is **counter-clockwise as the operator (looking along +y) sees it**. Valves open counter-clockwise.
* **Tag plates.** Every valve and instrument carries a stainless plate (about 68 x 26 mm; 60 x 23 mm on the small
  valves; black text in the font of Pillow's `ImageFont.load_default`) with its **site tag** (`HV-2317`, `PI-2342`,
  ...): above or below a handwheel, beside or below a lever, below a dial, a scale or a needle valve. The plates are
  legible from close up (the wrist camera), not from across the cell. The pumps, tanks, drums and the strainer carry
  name plates with their site tags; the analyser's tag is printed under its display.
* **Site tags.** A site numbers its equipment in its area: `HV-<area><nn>` for the hand valves, `P-`, `TK-`, `F-`, `PI-`,
  `PDI-`, `FI-`, `LI-` (level), `AI-` for the rest, the numbers in no particular order; the station is `LCS-<area>`. The
  site's equipment list (`describe()["site"]["equipment"]`) maps every series tag to its site tag.

## 2. Valves

Each valve's service is fixed by the series; how it is built is the site's choice among these:

| series tag | service | built as | travel closed -> open | turning effort |
|---|---|---|---|---|
| V-101, V-103 | P-101 suction / discharge isolation | gate valve with a handwheel (160-200 mm), rising stem; **or** ball valve with a lever (160-200 mm) | 2, 2.5 or 3 turns; or a quarter turn | 0.35-0.7 N m; 1.0-1.6 N m |
| V-105 | product flow control | globe valve with a handwheel (160-200 mm), rising stem | 3 or 4 turns | 0.35-0.7 N m |
| V-107 | bypass of V-105 (on a loop under V-105) | globe valve with a small handwheel (120 or 140 mm) | 1.5 or 2 turns | 0.35-0.7 N m |
| V-106 | T-101 make-up (bypass of T-101's float valve) | globe valve with a small handwheel (120 or 140 mm) | 1.5 or 2 turns | 0.35-0.7 N m |
| V-104 | minimum-flow return, on its vertical drop | ball valve with a lever (120-150 mm); **or** gate valve with a small handwheel (120 or 140 mm) | a quarter turn; or 1.5 or 2 turns | 1.0-1.6 N m; 0.35-0.7 N m |
| V-102, V-112, V-302 | vents (pump casing, strainer chamber B, P-301 head), on vertical stubs | small ball valve with a lever (80 mm); **or** needle valve with a knob (32 mm) | a quarter turn; or 1.5 or 2 turns | 0.6-1.0 N m; 0.08-0.15 N m |
| V-111 | strainer changeover | changeover cock with a lever (130 mm) | half a turn (A -> B) | 1.2-1.8 N m |
| V-303 | dosing suction selector (drum A / drum B) | three-way cock with a small lever (80 mm) | half a turn (A -> B) | 0.6-1.0 N m |
| V-301 | dosing rate | needle valve with a knob (38 mm) | 2 or 3 turns | 0.08-0.15 N m |

* **Turning effort** is the torque about the valve's stem that turns it slowly (the packing's friction; the site's
  valve has one value in the range). Turning faster adds a viscous part: 0.4 N m per rad/s on the handwheels and
  levers, 0.05 N m per rad/s on the needle valves. A valve turned at up to 35 degrees a second therefore takes at most
  its effort plus 0.25 N m, never more than about 2.1 N m on this skid. At each end of its travel a valve meets a hard
  stop: pushed on, the torque it takes rises steeply while it no longer turns.
* **Handwheels**: 3 to 5 spokes (13 mm round bar) between a hub (44 mm) and a rim (18 mm round bar); the wheel turns in
  a plane 0.11-0.15 m in front of the pipe. The stem rises out of the hub as the valve opens: flush with the hub when
  closed, 34 mm out when fully open (linear in the valve's travel). The flow through a gate valve rises steeply over
  the first quarter of its travel; a globe valve passes a flow roughly proportional to its opening.
* **Levers**: a flat steel bar (22 mm wide; the small ones 20 mm) with a coloured grip sleeve over its outer 45 %. A
  ball valve is **open when its lever lies along its pipe** and closed when the lever is across it. On the horizontal
  suction and discharge lines (V-101, V-103) the closed lever points up and turns counter-clockwise to lie along the
  pipe; on vertical pipes (V-104, the vents) the closed lever is horizontal and the open one vertical.
* **V-111** (changeover cock between the strainer's chambers A and B): its **lever points at the chamber in service**,
  along the pipe line; changing from A to B is half a turn counter-clockwise (the lever sweeps through the vertical).
  Between the end positions both chambers pass flow.
* **V-303** (the dosing pump's suction selector): its lever points at the drum lined up, at the plate "A" or "B"
  beside the end positions; A -> B is half a turn counter-clockwise. Below 90 degrees drum A is lined up, above it drum B.
* **Needle valves** (V-301; the vents when so built): a black knob with a white index mark on its face;
  counter-clockwise opens.
* **Vents** (V-102, V-112, V-302) pass air and liquid only once they are open past about 40 % of their travel (V-302:
  past half of it); a vent cracked open less does nothing you can see. Open a vent fully to vent.
* Every valve has a hard stop at each end of its travel. A valve stays where it was left. The minimum-flow return V-104
  is left open at a shutdown (`sop.md` section 4).

## 3. The local control station

A grey box on a stand in front of the supply tank T-101, titled `LCS-<area> LOCAL CONTROL STATION`. Its front face:

* two groups of controls, one per pump, each under a label with the pump's site tag; the site chooses which group is
  on the left:
  * P-101's group: the selector **HS-101** (a black wing knob: OFF = pointing up, HAND = 45 degrees counter-clockwise,
    AUTO = 45 degrees clockwise, with a detent at each; the legend HAND OFF AUTO above it), the green **START**
    button, the red **STOP** button; the selector turns against 0.05 N m of friction and a detent spring that pulls it
    towards the nearest position (1.5 N m per rad, at most 0.45 N m);
  * P-301's group: its **RUN** lamp (green), green **START**, red **STOP**;
* the middle column, lamps (lit = bright): P-101's **RUN** (green) and **FAULT** (red), the **COMMON ALARM** (amber);
* the bottom row: the blue **RESET / ACK** button under P-101's group and the **EMERGENCY STOP** (a red mushroom head
  on a yellow disc) under P-301's group.

Every button and lamp has its label plate below it (the lamps' labels carry the pump's site tag: `P-2342 RUN`). Push
buttons (31 mm caps standing 18 mm proud of the face; the mushroom 52 mm, 30 mm proud) travel 8 mm into the panel
against a spring (4.4 N preload, about 9 N at 4 mm) and register a press past 4 mm. The horn sounds while any alarm is
unacknowledged (`get_state()["horn"]`).

## 4. Instruments

* **Dial gauges** PI-101 (pump suction, compound), PI-103 (pump discharge), PDI-111 (strainer differential): a black
  case (100 mm; PDI-111 90 mm) with a white face and a red needle. Every dial sweeps 270 degrees clockwise, its low end at
  the 7:30 position and its high end at the 4:30 position. The face shows the scale's
  numbers, the unit under the centre and a title (SUCTION / DISCHARGE / FILTER DP) above it. The dial types:

  | gauge | possible scales |
  |---|---|
  | PI-101 | -1 ... 3 bar, or -100 ... 300 kPa |
  | PI-103 | 0 ... 6 bar, 0 ... 10 bar, 0 ... 100 psi, or 0 ... 600 kPa |
  | PDI-111 | 0 ... 1 bar, or 0 ... 100 kPa |

  (the catalog has each face as printed.) A cavitating or air-bound pump makes the pressure needles flutter.
* **FI-105**, variable-area flow meter: a clear vertical tube (360 mm) with a red float, in a loop of the discharge line;
  read the float's top edge on the scale plate beside the tube. Scales: 0.5 ... 5 m3/h or 10 ... 80 L/min.
* **LG-101, LG-201**, magnetic level indicators: a column of orange flaps rises with the liquid on a 0 ... 100 % scale
  plate (1.1 m tall) beside it.
* **AI-201**, residual chlorine analyser: a small red 7-segment display (16 mm digits) showing `d.dd` mg/L (Cl2); legible
  from close up only.
* **VS-102, VS-112**, sight glasses on the vent lines of the pump casing and of strainer chamber B: clear with air in
  the line, blue once liquid stands in it.
* **Dosing drums** T-301 (drum A, duty) and T-302 (drum B, standby): translucent 20 L drums behind the dosing pump; the
  yellow hypochlorite shows through the wall (full = 0.52 m of liquid). Each carries its name plate and a large letter
  "A" or "B".

## 5. Where things are

The line runs along the skid: T-101 and the control station at one end, then V-106 on T-101's make-up line, LG-101,
the strainer (chambers A and B, V-111 between them, V-112 and VS-112 on chamber B, PDI-111 above), V-101 and PI-101 on
the suction line, P-101 (VS-102, V-102 and PI-103 beside it), V-103 on the discharge, V-104 on the minimum-flow drop,
the flow meter's loop and V-105 with its bypass loop and V-107 below it (in the site's order), the dosing package (AI-201
above, V-301 on the dosing line, P-301 with V-302 on its head, V-303 between the drums), LG-201 and T-201 at the other
end. The suction line runs 0.78-0.86 m above the floor, the discharge line 1.04-1.12 m, the make-up line 1.10-1.18 m;
every handle lies between about 0.7 and 1.25 m above the floor.

## 6. Site options

Each skid of the series is built to order, within this manual:

* the numbering: the site's area number and every device's site tag (section 1);
* the arrangement: the line runs left to right (T-101 and the station at the left end) or mirrored; the flow meter sits
  before or after V-105; the station's two pump groups either way round; each device's position along the skid varies
  by a few centimetres from one skid to the next, and the pipe heights within the ranges of section 5;
* each valve's build within section 2 (a handwheel or a lever where the table allows; the handwheels' diameter, spokes
  and angular position, the turns from closed to open, the turning efforts, the lever lengths, the colours) and the
  dial and flow-meter scales fitted (section 4).

How the skid runs on a given day belongs to the plant, not to the manual: the pump's performance, the tanks' levels, the
residual chlorine, the consumer's draw, the strainer's fouling, the metering pump's capacity and the hypochlorite's
strength (every delivery differs) are marked nowhere on the skid. Read them off the instruments and their trends.
