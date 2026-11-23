# Bought parts inside the units

The units arrive from their stations complete: nothing in this list is fitted by the cell. It names what the units are
made of besides their printed parts, under the parts' generic names and the part numbers of the lab's parts list. Masses
are those of this build's model (the cell simulates each unit with the total mass that `kit.json` gives; they were not
weighed). The columns give the pieces in one unit of each kind; the build has three finger modules and one of every
other kind.

| part | part number | mass of one (g) | palm base | finger module | thumb module | electronics stack | upper shell | dorsal cover | power hub | press-fit cap | in the whole build |
|---|---|---|---|---|---|---|---|---|---|---|---|
| servo actuator | VKR-4LBA | 27 | - | 3 | 4 | - | - | - | - | - | 13 |
| miniature ball bearing | HNL-EX77 | 0.149 | - | 6 | - | - | - | - | - | - | 18 |
| dowel pin | HNL-NQ37 | 0.108 | - | 5 | - | - | - | - | - | - | 15 |
| fingertip tactile module | ZCP-F561 | 2.12 | - | 1 | - | - | - | - | - | - | 3 |
| thumb tactile module | ZCP-DG1Z | 7.71 | - | - | 1 | - | - | - | - | - | 1 |
| bus distribution board | BWD-RS8C | 4.76 | - | - | - | 1 | - | - | - | - | 1 |
| tactile interface board | ZCP-384F | 3.57 | - | - | - | 1 | - | - | - | - | 1 |
| power and bus hub board | JQT-NLP9 | 25 | - | - | - | - | - | - | 1 | - | 1 |
| M2 x 4 pan head screw | standard hardware | - | 2 | 22 | 28 | - | 2 | - | - | - | 98 |
| M2 x 6 socket head screw | standard hardware | - | 16 | 3 | 6 | - | 9 | 5 | 3 | - | 48 |
| M1.7 x 4 cheese head screw | standard hardware | - | - | 4 | 12 | - | 6 | - | - | - | 30 |
| M1.4 x 4 socket head screw | standard hardware | - | - | 4 | 3 | - | - | - | - | - | 15 |
| M2 x 6 flat head screw | standard hardware | - | - | 3 | 4 | - | - | - | - | - | 13 |
| M1.6 x 16 socket head screw | standard hardware | - | - | - | - | 4 | - | - | - | - | 4 |

* **Servo actuator**: the one actuator type of the hand, one per driven joint. In this cell it is never powered: each
  servo joint turns against 0.02 N m of friction (`kit.json` gives the joint data as simulated). Its case, output boss
  and rear bearing ring collide as one solid; its own case screws and connectors are drawn only.
* **Tactile modules**: the pad at the tip of each finger and of the thumb. They collide as rigid pieces.
* **Boards**: the bus distribution board and the tactile interface board are stacked with a printed spacer frame on four
  long screws: together they are the electronics stack. The power and bus hub board is the board of the power hub, held
  by three printed locks.
* **Screws, pins, bearings**: standard hardware, drawn on the unit their head bears on or that holds them, from the
  start; none of them collides in the cell (`cell_manual.md`, "Collision model"). Cables are not part of this build.
