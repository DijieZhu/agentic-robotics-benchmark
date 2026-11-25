# Site briefing: what the team knows before the robot goes down

This is the rescue team's briefing for the search of the collapsed racking hall. It holds for every site the robot is
sent into, the scored sites and the practice sites alike. What is specific to one site comes from `station.mission()`
when the episode starts.

## 1. The situation

The hall's racking and parts of its upper floor have come down. The team works through holes in the roof. Through one
hole it lowers a mast with the search camera; through the same hole it lowers the robot on a rope, and later hooks it
again to pull it out. A **site** is the patch of the hall's floor under one such hole: a rectangle that the team has
surveyed from above, enclosed by a perimeter wall of rubble and standing structure that the robot cannot cross.

The team wants an air reading from three places on the site, the **search points**. Two of them lie on open floor.
The third lies under a collapsed slab, in a void where a person could be lying; that reading is the one that matters
most, and its weight in the score says so.

## 2. What the team has surveyed

- **The site frame.** x points east, y north, z up; the origin is the south-west corner of the floor; the floor is
  z = 0. The camera's mast stands at the south side of the site, and the north side is the hall's back wall.
- **The site rectangle**, the **set-down pad's centre** and the **three search points**: coordinates in the site
  frame, in `mission()`. They are right. Nothing marks the pad or a search point on the floor: no paint, no plate.
- **The site's walls.** The perimeter wall stands outside the site rectangle: its inner faces are the four edges of
  the rectangle, and they are surveyed like the rectangle itself.
- **The search camera's calibration**: its intrinsics and its pose in the site frame, in `mission()`; the accuracy is
  in `camera_datasheet.md`.
- **The retrieval area**: the circle around the pad's centre whose radius `mission()` gives. A robot standing inside
  it can be hooked.

<!-- BEGIN GENERATED setdown -->
At the start of an episode the robot stands in its rest shape on one of its six rollable faces, its centre within 0.10 m of the pad's centre, at any heading. Which face is down and which way it points differ from episode to episode.
<!-- END GENERATED setdown -->

## 3. What the ruin is made of

From the south, where the pad is, to the back wall in the north:

- **Open floor around the pad.** Search point 1 lies on it.
- **A ridge of rubble** across the whole width of the site: blocks with openings between and under them. Search point
  2 lies on the open floor behind it.
- **A collapsed slab** that covers the whole north part of the site, from its front edge to the back wall and from the
  west wall to the east wall, resting on rubble and on what is left of the racking. Under it lies a **crawl space**;
  search point 3 lies in it. The crawl space is entered from the south, through openings under the slab's front edge.

**Openings.** The rubble is passed through openings of two kinds: open gaps between blocks, and covered openings
where a fallen panel rests on supports. Covered openings can be short passages, low slots, or mouths that lead under
the slab. The openings of a site differ in width and in clear height. Some are too narrow or too low for the robot.
The team has no list of which.

**Blocked openings.** An opening that leads under a slab can be blocked by fallen material a short way inside. From
outside it looks like an opening that is free, and the search camera cannot see the difference. This is so for the
openings under the slab's front edge: there are three, alike in width, height and cover; one of them leads into the
crawl space and the other two are blocked.

**The crawl space.** The racking stood in rows parallel to the hall's walls, and what the collapse has left open under
the slab follows them: straight passages, parallel to the site's walls, that meet at right angles. They are wider than
the openings under the slab's front edge and as high. A passage can end blind. Fallen material can stand in a passage:
a block against one of its two sides, from the floor to the slab. Where a block stands, the passage is narrowed for a
short stretch, and what stays open beside the block is as wide as the openings under the slab's front edge. In one of
the site's two back corners the crawl space widens to a chamber: the hall's back wall and the site's east or west wall
are two of its walls. No block stands in the chamber. Search point 3 lies in that chamber; the floor between the point
and these two walls is clear, and nothing stands within 0.6 m of the point. From the opening that leads in, the way to
the chamber is several metres long and turns more than once.

**Crates.** A few crates may stand on the open floor; some are red or blue.

**Heaps.** Heaps stand on the open floor, each of one of two kinds. A **rubble heap** is bricks and lumps of broken
concrete; a **mound of fill** is dumped earth. A heap is between 0.14 and 0.30 m high at its top and at most 0.60 m
long in x and in y, and its outline is irregular. A mound of fill has sloping sides; a rubble heap can have upright
faces. The robot cannot get over a heap: it has to go around it. What a roll against a heap does to the robot is in
the handbook, section 8.

**Where heaps stand.** Every heap stands against a wall, against the ridge or against the slab's front edge; none
stands in the crawl space. Between a heap and anything else that stands on the floor, the floor is either less than
0.15 m wide, far too narrow for the robot, or at least 0.60 m wide. No heap stands within 0.45 m of the pad's centre
or within 0.40 m of search point 1 or 2. No heap stands within 0.40 m of the floor point 0.25 m in front of or behind
the middle of an opening of the ridge. In front of each opening under the slab the floor is free of heaps over the
opening's width and 0.70 m southwards. No heap lies on a patch of dust; one can stand right at a patch's edge.

**The floor.** Level concrete everywhere, without steps, kerbs or slopes; its friction differs somewhat from site to
site (the handbook gives the range). On it lie:

- **patches of loose ground**, dust and gravel, flush with the concrete; in the camera's picture a patch is an area of
  another colour. The robot moves differently on them than on concrete (handbook, section 8);
- **plastic chips**: flat pieces a few centimetres across, discs and squares, in red, green and blue, the colours of
  the robot's end caps, scattered over the floor by the collapse. They do not hinder the robot;
- **thin litter**: loose pages, flattened cardboard, film, strapping, spilled powder, grit, and the earth and grit
  strewn around the foot of a heap. It is flat and does not affect the robot.

**The light.** Daylight through the holes in the roof. The sun can reach the floor in patches, with hard shadows
beside them, and there is dust in the air; direction and strength differ from site to site and do not change during
an episode.

## 4. What the camera sees and what it does not

- The camera looks down from the mast at the south side, tilted towards the north. Its picture covers the site's floor
  from the south wall to about 4 m north of it, from the west wall to the east wall; the site rectangle reaches
  further north, and that part lies under the slab, outside the picture.
- The pad and search points 1 and 2 lie in the open, in the camera's view.
- The camera sees every heap: its top and, seen from straight above, at least two thirds of the area the heap covers.
- The camera sees the top of the rubble, the front part of the slab's top, and the south side of what stands. It does
  not see behind high rubble where that hides the floor, and it does not see under a cover: of a covered opening it
  sees the south face and at most a narrow strip of floor inside.
- The crawl space and search point 3 are out of the camera's view altogether. While the robot is under the slab, the
  frames do not show it.

## 5. What the team does not know

- the layout of a site in numbers: where the blocks and the heaps stand, how wide and how high each opening is;
- which openings the robot can pass;
- which of the three openings under the slab leads into the crawl space;
- how the passages of the crawl space run: where they turn, where they end blind, how long and how wide they are,
  where a block narrows them and on which side it leaves them open, and where the chamber's other two walls stand;
- how the robot moves on a site's dust and gravel, and the friction of its concrete.

## 6. What holds on every site

- The three search points can be reached one after the other from the pad, and the pad again from the third, by a
  robot that uses only the handbook's primitives.
- Nothing on a site moves except the robot. Nothing the robot touches harms it.
- The radio link holds everywhere, under the slab too.
- The search camera does not move and its calibration holds for the whole episode.
