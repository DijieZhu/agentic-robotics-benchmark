# Site rules for robotic entries

Rules of the plant for a robot that enters an auxiliary building, and what follows from them for its program.

## 1. Dose

The robot's sensor head (cameras, lidar) is rated for a limited radiation dose. The entry is over when it has
received that dose: the robot stops where it stands and the mission ends. What the team had received until then
stays received; what was still waiting on board is lost, and so is the robot's return.

<!-- BEGIN GENERATED dose -->
| dose rate where the robot stands (Gy/h) | dosimeter (counts per second) | time until the rating of 2.0 Gy is reached | share of the rating used in 10 s |
|---|---|---|---|
| 0.1 | 4.5 | 20 h | 0.01 % |
| 1 | 45 | 2 h | 0.14 % |
| 5 | 225 | 24 min | 0.69 % |
| 20 (1 m from the weakest unlisted source) | 900 | 6 min | 2.78 % |
| 60 (1 m from the strongest) | 2700 | 2 min | 8.33 % |
| 240 (0.5 m from the strongest) | 10800 | 30 s | 33.33 % |
<!-- END GENERATED dose -->

- The dosimeter at the body centre shows the dose rate where the robot is, and its own running sum of the dose
  (`get_state()["dosimeter"]`: `dose_gy`, and `dose_fraction`, its share of the rating). The sum is made from counts,
  so it follows the dose the head really received only within the counting statistics: do not plan to the last
  percent.
- The drawing's zoning says what dose rate to expect where (facility notes, section 4), as far as it is up to date.
  The listed sources are the drums of the waste store and process equipment that holds activity; their rooms are
  drawn as controlled or high-radiation zones.
- Where a sheet of the drawing is missing, no zone is drawn. A listed source can be in a room of the missing sheet:
  the work order names the room (`work_order.documented_sources`), and nothing else warns of it. Plan there with the
  dosimeter, not with the drawing: a robot that walks into a waste store because nothing was drawn loses its mission
  on the dose rating.
- The unlisted source is not in the zoning. Finds of this kind in the plant's buildings measured between 20 and 60
  Gy/h at 1 m. The dose rate falls with the square of the distance from a source and is much lower behind a wall, a
  shield block or a piece of equipment. Such a source sits 0.45 to 0.85 m above the floor, and the distance that
  counts is the one to the source itself, not the one along the floor.
- The unlisted source can lie in a room, or in a corridor that the round has to use: then it is on or right beside
  the robot's way, every passage costs dose (the table's last rows), and a robot that lingers near it ends its
  mission on the dose rating.
- To report the unlisted source the robot must have been near the point it reports (round procedure, section 5); it
  need not have stood next to the source.

## 2. The radio link

<!-- BEGIN GENERATED link -->
| item | value |
|---|---|
| base station (airlock) | reaches 12 m; its aerial is 1.9 m above the floor, at the place the link plan gives (`link`) |
| repeater, when it works | reaches 9 m; it hangs 2 m above the floor, at the place the link plan gives |
| condition | a free straight line from the top of the robot's back (the lidar) to the node's aerial, within the node's reach: walls, door leaves, shield blocks and equipment block it |
| checked | every 0.5 s of simulated time, and once more at the instant the mission ends |
<!-- END GENERATED link -->

- The shield walls stop the radio. The link is up only where the robot's back has a free line to a node's aerial.
- **The base station** hangs on a side wall of the airlock, where the link plan shows it. In every building the link
  is up everywhere within 1.0 m of the pad's centre and on nearly all of the airlock's floor, and through the open
  inner door the base station reaches a strip of the corridor in front of the airlock, a few metres long.
- **The repeaters** hang in the corridors, where the link plan shows them, also on a stretch of corridor that a
  missing sheet of the drawing would have shown. One of the two no longer relays (facility notes, section 3); the
  other covers the corridor stretch it can see and, through open doors, a little of the rooms off it. A repeater in
  service shows a green lamp.
- A report made where the link is down is kept on board and goes out when the link is next up. Whether the link is up
  is in `get_state()["link"]`; that a queued report has gone out is told by the event `report_sent`.
- Reports that are still on board when the mission ends are lost. There is no other way to hand results over: the
  robot's memory is not read out after an entry.
- **A round can end with nothing delivered.** A reading made out of the link's reach is worth nothing until the
  robot stands in the link again. A robot that has read every instrument and then does not get back to a place with
  a link, because it is caught, overturned, out of dose or out of time, has delivered nothing.

## 3. The pad and the return

The entry starts on the deployment pad in the airlock, where the team sets the robot down, and the work order ends
with the instruction to return to the pad and stop upright.

<!-- BEGIN GENERATED end -->
- **A mission ends** when the program ends it (`finish()`, or `solve` returning), after 1350 s of simulated time, when the dose received at the dosimeter's place reaches the rating of 2.0 Gy, when roll or pitch has been beyond 70° for 2 s, or when the program breaks one of its limits (`robot_api.md`, section 9).
- **The return counts** when at that moment the robot is upright (roll and pitch below 30°, whether it stands or sits) and its body centre is within 1.0 m (horizontally) of the pad's centre. Otherwise the mission's score is multiplied by 0.85; it always is when the mission ended because the robot was tilted over or the dose rating was reached.
<!-- END GENERATED end -->

A robot that has sat down on the pad has returned; a robot that lies on its side has not.

## 4. Conduct in the building

- Doors stay as they are: the robot cannot open, close or push one.
- A tank room's bund has one crossing for this robot where its kerb is too high to step over: the ramp. It is the
  way in and the only way out (facility notes, section 3; robot manual, section 4.4).
- A robot that overturns stays where it is; it cannot right itself, and the mission ends.
- Nothing in the building may be relied on to be where the drawing has it, except what the facility notes name;
  and what a missing sheet would have shown is not known at all.
