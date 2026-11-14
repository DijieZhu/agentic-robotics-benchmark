# SOP-TD-01  Transfer and chlorine dosing skid: start-up, operation, shutdown

The skid transfers water from the supply tank T-101 to the product tank T-201 with the centrifugal pump P-101 and doses
sodium hypochlorite into T-201 with the metering pump P-301, which draws from dosing drum A (T-301, duty) or B (T-302,
standby). The consumer downstream draws from T-201. See `pid.png` for the process, `equipment_manual.md` for how each
device is built, operated and read.

**Tags.** This procedure, the P&ID and the equipment manual use the series tags (V-101, P-101, ...), the same on every
skid of the series. Each site numbers its equipment its own way: the tag plates and the control station's labels show
the **site tags**, and the site's **equipment list** (`describe()["site"]["equipment"]`) gives, for every series tag, the
site tag, the service and the start-up position. The log sheet takes the site tags.

All times are the cell's simulated clock (`sim_time`, seconds from the start of the shift).

| time | what |
|---|---|
| 0 s | shift start: the skid is in the shutdown state the previous shift left (not necessarily the one section 1 asks for) |
| 0 - 600 s | start-up (section 1): everything in section 1 done before the operating window opens |
| 600 s | the operating window opens; the consumer starts drawing from T-201 |
| 600 - 1500 s | operation (section 2) and the abnormal-situation procedures (section 3) |
| 1500 s | shift end: shut the skid down (section 4) |
| 1800 s | the episode ends (or earlier, when your program calls `finish`) |

## 1. Start-up

1.1 **Line-up.** Before P-101 is started, every valve must be in its line-up position. Check each one: the previous
shift may have left some of them elsewhere. Check the make-up valve V-106 first: left open, it fills T-101 to its
overflow within minutes (in 1.5-10 minutes, typically 2-6, while P-101 stands), since the float valve shuts and the
make-up keeps coming.

| tag | service | line-up position |
|---|---|---|
| V-101 | P-101 suction isolation | fully open |
| V-104 | minimum-flow return (to T-101) | fully open |
| V-103 | P-101 discharge isolation | closed |
| V-105 | product flow control | closed |
| V-107 | bypass of V-105 | closed |
| V-106 | T-101 make-up (bypass of T-101's float valve) | closed |
| V-111 | strainer changeover | chamber A in service |
| V-102 | P-101 casing vent | closed |
| V-112 | strainer chamber B vent | closed |
| V-302 | P-301 head vent | closed |
| V-303 | dosing suction selector | drum A |

1.2 **Prime P-101.** The pump casing was drained during maintenance. With V-101 open, open V-102 fully: air escapes until
liquid stands in the sight glass VS-102 (about 12 s with V-102 fully open). Then close V-102. A pump started with air
in its casing is air-bound: it delivers little and its gauges' needles flutter; stop it, vent it and start it again.

1.3 **Start P-101.** Turn the selector HS-101 to HAND and press START (P-101). The RUN lamp lights; PI-103 rises to
the pump's shutoff pressure while the flow goes through the minimum-flow line only.

1.4 **Open V-103 fully** within 240 s of the start.

1.5 **Fill T-201.** Open V-105 to set the transfer (FI-105). Bring LG-201 into its band (55-75 %) by 600 s: the consumer
starts drawing at 600 s, at a rate no instrument shows (the level's trend shows it). The flow control valve is V-105;
its bypass V-107 stays closed.

1.6 **Start the dosing.** Press START (P-301) (P-301 runs only while P-101 runs) and set the needle valve V-301: more
open doses more. P-301 must be dosing when the window opens.

## 2. Operation (600 - 1500 s)

2.1 Hold **LG-201 between 55 and 75 %** (aim at 65 %): at a steady level the transfer equals the consumer's draw;
re-set V-105 when the level trends away. Never let T-201 reach 95 % (it overflows).

2.2 Hold the residual chlorine **AI-201 between 1.8 and 2.6 mg/L** (aim at 2.2): re-set V-301. AI-201 lags the tank by
about 20 s, and T-201 mixes over minutes: change V-301 in small steps and wait.

2.3 **Log sheet.** The sheet has two rounds: the first from 600 s to 1050 s, the second from 1050 s to the shift's end
at 1500 s. In each round, with P-101 running, record each instrument on the site's log sheet
(`describe()["site"]["log_sheet"]`: the site tags of PI-101, PI-103, PDI-111, FI-105, LG-201 and AI-201) once with
`log_reading(site_tag, value, unit)`, reading the value and the unit off the instrument itself. The first entry of each
tag in a round, made while P-101 runs, is the one on the sheet; it is checked against the instrument's true value at
that moment (tolerance: 1 % of the instrument's span; LG-201 1 %; AI-201 0.05 mg/L). Units: bar, psi, kPa, m3/h,
L/min, %, mg/L.

2.4 **Alarms.** The control station has one **COMMON ALARM** lamp: it lights while any of these lasts, and the horn
sounds (`get_state()["horn"]`) until RESET / ACK is pressed:
* the strainer's differential pressure PDI-111 above 0.6 bar;
* T-201's level LG-201 above 85 %;
* T-101's level LG-101 below 30 %.

The lamp does not say which: find the cause on the instruments (section 3). P-101's own trip lights its **FAULT**
lamp and sounds the horn too. Not every problem raises an alarm: watch the trends.

2.5 **Rounds.** Nothing on the panel shows the chlorine, the flows or the pressures: read the instruments where they
are, often enough to catch a trend (once the dosing stops, AI-201 falls by about 0.2-0.5 mg/L a minute, the faster
the higher the transfer).

## 3. Abnormal situations

**What to check first:**

| you see | check | then |
|---|---|---|
| COMMON ALARM, PI-101 falling, needles fluttering | PDI-111 high? | AOP-1 |
| COMMON ALARM, PI-101 falling | LG-101 low (T-101 emptying)? | AOP-5 |
| COMMON ALARM, LG-201 high | | reduce the transfer (V-105) at once |
| LG-201 drifting with no alarm | | AOP-2 |
| AI-201 falling while P-301 runs | drum A's level at the bottom (with V-303 at A)? | yes: AOP-4; no: AOP-3 |
| P-101 FAULT lamp, horn, no flow | | AOP-6 |

**AOP-1  Strainer differential pressure high** (PDI-111 above 0.6 bar). Within 240 s of the alarm, change over from the
chamber in service (the one V-111's lever points at; A on a normal shift) to the other one:
1. if you change to chamber B, which was left drained after cleaning, fill and vent it first: open V-112 fully until the
   sight glass VS-112 shows liquid (about 12 s), then close V-112 (a chamber taken out of service stays full of water and
   needs no venting);
2. turn the changeover V-111 to the other chamber (half a turn: its lever ends pointing at that chamber);
3. press RESET / ACK.

Switching to an unvented chamber pushes its air into the pump (air-bound). A debris slug lands in whichever chamber is
in service when it comes: a chamber changed over to beforehand is clogged the same way, and only a change-over away
from the clogged chamber helps. A clogged strainer starves the pump: it
cavitates (the suction gauge PI-101 falls, the pressure needles flutter, the flow drops); 300 s of cavitation in all
damages the pump.

**AOP-2  Demand change.** The consumer's draw may change during the window. When LG-201 trends away from 65 %, re-set V-105.
360 s after the change, LG-201 should be in its band and the transfer matched to the new draw: within 0.35 m3/h of it,
or up to 1 m3/h above (below) it while the level is being brought back up (down) to 65 %.

**AOP-3  Metering pump air-locked** (P-301 runs, drum A or B has liquid, but AI-201 keeps falling). Within 300 s:
1. press STOP (P-301);
2. open V-302 fully and keep it open at least 8 s;
3. close V-302;
4. press START (P-301).

Never open V-302 while P-301 runs: the hypochlorite leaves through the vent and nothing reaches T-201. The air lock is
in the pump's head: do not change drums for it. Venting does not help while the drum lined up is empty (AOP-4).

**AOP-4  Duty drum empty.** Drum A (T-301) is translucent: its level shows through its wall. When drum A runs dry, P-301
draws air and air-locks. Within 300 s of drum A running dry:
1. turn V-303 from A to B (half a turn: its lever ends pointing at the "B" plate);
2. if P-301 had drawn air, vent its head (AOP-3, steps 1-4).

Change over only once drum A is nearly empty (below 5 % of the drum) or dry: changing over while it still holds more
wastes the standby drum. Changing over in that last 5 %, before drum A runs dry, avoids the air lock and counts the same.

**AOP-5  Supply tank level low** (LG-101 below 30 %; T-101's float valve has stuck shut and the plant's supply no longer
refills T-101). Open the make-up valve V-106 (it bypasses the float valve) and hold T-101 between 40 and 80 % with it:
open it about as far as the transfer needs and re-set it as LG-101 trends. Act before T-101 runs down to about 8 %:
P-101 then trips on dry running. Never open V-106 while the float valve works: the float valve then shuts, and once the
make-up passes more than the transfer T-101 overfills (at 95 % it overflows to the drain: the shift loses its line-up
credit, section 6); a make-up valve that is already open when the float valve sticks is not this procedure and earns
nothing. When P-101 stops at the shift's end, close V-106.

**AOP-6  P-101 tripped** (P-101 FAULT lamp, the horn; a dip of the plant's power supply). P-301 stops with it. Within
180 s of the trip:
1. press RESET / ACK (the fault clears, the horn stops);
2. close V-103 (P-101 restarts against a closed discharge);
3. with HS-101 at HAND, press START (P-101);
4. open V-103 fully and re-set V-105 if the level needs it;
5. press START (P-301) (dosing again within 300 s of the trip).

## 4. Shutdown (after 1500 s)

In this order:
1. press STOP (P-301);
2. close V-103 fully;
3. press STOP (P-101);
4. close V-101 fully (and V-106 if it was opened in AOP-5);
5. turn HS-101 to OFF.

The minimum-flow return V-104 stays open: it is the pump's flow path at the next start.

## 5. Never

Each of these ends the shift badly (the episode scores 0):
* running P-101 with no flow path (V-103 and V-104 both shut) for 30 s (deadheading: seal damage);
* 300 s of cavitation in all;
* T-201 reaching 95 % (it overflows);
* pressing the EMERGENCY STOP (no situation on this shift calls for it);
* the robot damaging the skid, or being taken out of service. The robot cell stops the arm on a hard contact, as a
  collaborative robot does: more than 40 N for 30 ms on anything but a handle or a button, or more than 150 N on a
  handle or a button, is a protective stop (the arm halts for 5 s, `cell_manual.md` section 4), which costs time but
  not the shift. Contact above 400 N for 30 ms (it can bend a stub or crack a sight glass) or a sixth protective stop
  in the shift takes the robot out of service for the rest of the shift.

T-101 reaching 95 % overflows to the drain: water is wasted, nothing is damaged, and the shift loses its line-up
credit (section 6).

Running P-101 with V-101 shut (or T-101 nearly empty) trips the motor protection after 5 s (FAULT lamp): remove the
cause, press RESET / ACK, start again. The selector at OFF or AUTO stops P-101 (AUTO is not available on this skid).

## 6. How the shift is scored

| part | weight | what counts |
|---|---|---|
| start-up | 0.20 | at the first start of P-101: the line-up of 1.1 (0.05; lost if T-101 overflows during the shift) and a primed casing (0.04); a clean start: no trip and no deadheading in its first 120 s, still running then (0.04); V-103 fully open within 240 s of the start and before 600 s (0.04); P-301 dosing at 600 s (0.03) |
| operation | 0.25 | the fraction of the window with LG-201 in 55-75 % (0.125) and with AI-201 in 1.8-2.6 mg/L (0.125) |
| events | 0.25 | split equally between the window's events, the abnormal situations of section 3 that happen on the shift. AOP-2: in band and matched (or recovering towards 65 %) 360 s after the change. AOP-1: changed over from the clogged chamber to the other one within 240 s of the alarm (0.5 of the event), the chamber changed to free of air when it came into service (0.3), RESET after (0.2) (a slug that never raises the alarm in the window is not judged; changing over on the trend, after the slug came and before the alarm, counts as on time). AOP-3: dosing again within 300 s, from the drum that was lined up when the pump air-locked. AOP-4: V-303 turned to drum B once drum A was below 5 % (or dry), and dosing from drum B within 300 s of drum A running dry. AOP-5: V-106 opened after the float valve stuck (it was closed then) and before P-101 trips on dry running, and no such trip until the shift ends (not judged if T-101 never reaches its low alarm with V-106 closed; a V-106 already open when the float valve stuck earns nothing). AOP-6: P-101 running again within 180 s (0.5), restarted with V-103 closed (0.25), dosing again within 300 s (0.25) |
| shutdown | 0.15 | after 1500 s: P-301 stopped before P-101 (0.03), V-103 closed before P-101 stops (0.04), P-101 stopped (0.03), V-101 closed at the end (0.03), HS-101 at OFF at the end (0.02) |
| log sheet | 0.15 | 0.0125 for each of the six instruments of 2.3 recorded correctly in each of the two rounds |

A protective stop of the robot (section 5) costs no credit by itself, only the robot's time; the robot taken out of
service (contact above 400 N for 30 ms, or a sixth protective stop in the shift) ends the shift badly like the other
cases of section 5. T-101's overflow costs the line-up credit only.
