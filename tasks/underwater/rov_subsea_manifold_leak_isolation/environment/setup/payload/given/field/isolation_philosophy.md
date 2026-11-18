# Isolation philosophy, operating limits and assessment of the intervention

This document states what the intervention has to achieve at a leaking manifold, what must never happen while doing
it, under which conditions contact work at the panel is allowed, and how the result of a dive is assessed. Every
condition by which a dive is assessed is written here in full; nothing is assessed that is not written here.

## 1. The process

Each of the six branches of a manifold connects one well to the two headers:

```
well -- BV-i -- leak segment S-i (test port TP-i, flanges F1 and F2) --+-- PV-i -- header P
                                                                       +-- TV-i -- header T
```

- BV-i is the branch valve of branch i; PV-i and TV-i are its two header valves. All three are on the intervention
  panel (panel standard), together with the test port TP-i of the segment between them.
- The wells and both headers are pressure sources. The leak segment of a branch stays at line pressure as long as it
  is connected to any source through an open valve.
- Once a leak segment is cut off from every source it bleeds down through its own leak. This is gradual: from some
  tens of seconds to a few minutes pass before the pressure is low enough for the hot stab, and the stream of bubbles
  weakens as the pressure falls. A segment that is still connected to a source does not change, whatever else is done.
- After the loss of the control umbilical the valves are where they happened to be. The hand-over state table in the
  work order lists the last known position of every valve; between 0 and 2 of its entries can be out of date. The
  handles on the panel show the positions as found.

## 2. What to isolate

One branch of the manifold leaks at a flange of its leak segment. All six wells are live, so any branch can be the
one, and closing all of them is not an option: a healthy well that is shut in at sea-bed temperature forms a hydrate
plug, and bringing it back takes weeks.

- The **isolation set** is made of those of the three valves BV, PV and TV of the leaking branch that are open as
  found. It has one, two or three members.
- The intervention closes exactly the isolation set. Then the leak segment is cut off from its well and from both
  headers, and nothing else in the process has changed.
- A valve is **closed** when its handle is at 80 deg or more from the open stop.
- A valve is **open** when its handle is at 10 deg or less. Anywhere below 80 deg the valve still passes.

## 3. What must never happen

- A **critical error** is any change of the passing state of a valve that is **not** in the isolation set, against its
  state as found, that lasts 0.2 s or longer: an open valve turned to 80 deg or beyond, or a closed valve brought
  below 80 deg. It counts at any time of the dive, also when the valve is turned back afterwards, and also when the
  change is still there at the end of the dive. The state of the process is evaluated every 0.1 s, and each evaluation
  stands until the next.
- This covers shutting in a healthy well, connecting the two headers of a branch, and opening another path into the
  leak segment. The reason is the one above: a healthy well that has been shut in, even for a short time and even if
  it is opened again, is a production incident. Closing a valve to see whether the bubbles stop and reopening it when
  they do not is such an incident.
- A **misoperation** is a valve outside the isolation set that ends the dive more than 10 deg away from its position
  as found although its passing state never changed.
- The same rules apply to what a collision does to a handle.

## 4. Operating limits

- **Contact work at the panel is not done while the current at the panel is above 0.30 m/s.** Contact work is the last
  part of an approach to a handle, a port or a lifting eye in front of the panel, gripping, turning a valve, taking
  and placing the hot stab, and taking a lifting eye.
- The dive is timed so that there is a workable window within the mission time. **The window is not necessarily open
  when the vehicle leaves the cage**: at some sites the current at the panel is above the limit at first and falls
  below it later. The work order's tide forecast and the vehicle's own measurements tell when.
- Transit, the search for the leak and the inspection of the panel are not limited by the current.
- Outside the window nothing is promised: contact work may not be possible at all.

## 5. The dive

- The vehicle starts in the deployment cage with the leak marker in its jaws, and the dive ends when the program ends
  it or a limit of the work order is reached (vehicle interface document). What has been achieved by then counts.
- The order of the work is free. Physics imposes the only constraints: the marker is in the jaws until it is put down;
  a hot stab cannot be latched into a test port that is under pressure; a dropped object that stands in front of a
  port has to be moved before that port can be reached; contact work waits for the window of section 4.
- The program reports nothing. Only the state of the site at the end of the dive and what happened on the way are
  assessed.

### 5.1 Deployment cage

The cage is a steel box, open on one side (the entrance), set down on the sea bed near the manifold. The work order
gives its position, the heading of its entrance and the height of its deck.

- Inside: 1.40 m deep (from the entrance), 1.40 m wide, 1.20 m high above the deck.
- The deck is 0.15 m above the sea bed and carries rubber landing pads; under the place where the vehicle starts there
  is a pocket for the marker that hangs below the jaws.
- A tag plate of the panel standard is inside on the back wall, facing the entrance, its centre 0.40 m above the deck;
  its code word is 26.
- Two sonar reflector poles stand 0.10 m outside the entrance, one on each side, 1.2 m apart; each is 100 mm in
  diameter and 2.0 m high.
- The sea bed within 4 m of the centre of the cage and in front of its entrance is clear: no boulder, no line and no
  mattress lies there.
- The vehicle starts seated on the deck, heading out of the entrance, with the origin of its body frame on the centre
  line of the cage, 0.12 m behind the cage's position in the work order (towards the back wall). It counts as inside
  the cage when the origin of its body frame is inside this box, above the deck.
- A vehicle that hovers in a current is never quite still. To end the dive at rest in the cage, sit the vehicle on the
  deck (vehicle manual).

## 6. How a dive is assessed

A dive at one site is assessed by 11 items. Each item is either earned in full or not at all; the points of the earned
items add up to the result for the site, at most 1.00.

| Item | Points | Part of the work | In short (the conditions are in section 6.2) |
|---|---|---|---|
| M1 | 0.03 | marking | the leak marker lies at rest on the roof grating |
| M2 | 0.12 | marking | ... over the marker zone of the leaking branch |
| H | 0.12 | survey and no harm | panel survey complete, no critical error, vehicle not disabled |
| I1 | 0.08 | isolation | at least one valve of the isolation set closed |
| I2 | 0.22 | isolation | every valve of the isolation set closed |
| I3 | 0.08 | isolation | ... and no misoperation |
| D1 | 0.07 | dropped object | dropped object moved clear of the panel by a lifting eye |
| D2 | 0.05 | dropped object | ... and at rest inside the laydown area |
| S1 | 0.04 | hot stab | hot stab taken out and latched in the leaking branch's test port or a parking port |
| S2 | 0.14 | hot stab | hot stab latched in the leaking branch's test port |
| R | 0.05 | return | survey complete, vehicle at rest in the cage |

### 6.1 Terms

- An object is **at rest** when its speed stays below 0.01 m/s over the last 5 s of the dive, the end of the dive
  included. A dive shorter than that has no object at rest.
- The vehicle is **at rest** when its speed stays below 0.02 m/s over the same last 5 s.
- The vehicle is **disabled** by its 6th protective stop (vehicle manual).
- The **panel survey** is complete when at least 15 of the 18 valve positions have been imaged.
- A valve position is **imaged** when one camera frame taken by the program shows its tag plate so that all of the
  following hold for that frame; they are judged from where the camera really was, not from what the program reads in
  the frame:
  - the four outer corners of the plate's square code are inside the image, at least 20 px from its edges;
  - the camera is at most 1.3 m from the centre of the code (slant range);
  - the shortest edge of any of the code's 8 by 8 cells, as projected into the image, is at least 4 px long;
  - the line of sight is within 60 deg of the normal of the plate;
  - the lamps are at level 0.3 or higher and the centre of the code is inside the beam of at least one lamp, that is
    within 67.5 deg of that lamp's axis;
  - of 5 by 5 points spread evenly over the code, at least 20 are in free view from the camera. The jaws, anything
    held in them, the guard rail and a dropped object block the view; marine growth on the plate does not.

Frames of either size count. The conditions on range and cell size together decide how far away a frame of a given
size still counts.

### 6.2 The items

Marking:

- **M1, 0.03.** At the end of the dive the leak marker is at rest, lying on the roof grating of the manifold, and has
  not been in the jaws during the last 5 s.
- **M2, 0.12.** M1 holds and the marker's centre of mass is over the marker zone of the leaking branch (panel
  standard, section 7).

Survey and no harm:

- **H, 0.12.** The panel survey is complete, there was no critical error, and the vehicle was not disabled.

Isolation:

- **I1, 0.08.** At the end of the dive at least one valve of the isolation set is closed.
- **I2, 0.22.** At the end of the dive every valve of the isolation set is closed.
- **I3, 0.08.** I2 holds and there is no misoperation.

Dropped object:

- **D1, 0.07.** At the end of the dive the point of the dropped object nearest to the panel face, its lifting eyes
  included, is at least 1.2 m from the face, and at some time in the dive the jaws held one of its lifting eyes. An
  object that was pushed away by the hull, or that moved by itself, does not earn the item.
- **D2, 0.05.** D1 holds, the object is at rest, and its centre of mass is inside the laydown area (panel standard,
  section 7).

Hot stab:

- **S1, 0.04.** The hot stab has left the port it was parked in, and at the end of the dive it is latched in the test
  port of the leaking branch or in either parking port.
- **S2, 0.14.** At the end of the dive the hot stab is latched in the test port of the leaking branch.

Return:

- **R, 0.05.** The panel survey is complete, and during the last 5 s of the dive the vehicle is inside the cage and at
  rest.

### 6.3 Critical error, totals

- A critical error at a site removes these items at that site, whatever else was achieved: I1, I2, I3, S1, S2 and H.
  The other items are not affected, and nothing is carried to another site.
- The result for a site is the sum of the points of the items earned there. There are no negative points, and no
  single failure cancels the rest: a disabled vehicle, a dive that runs out of time or an automatic ascent only end
  the dive.
- The program is run once at each of six sites. The overall result is the mean of the site results, and it is zero for
  a submission that is not admissible (task instruction).