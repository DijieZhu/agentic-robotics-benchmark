# DA-6 desktop arm - robot documents

Documentation of the DA-6 desktop arm (six BUS-SERVO-12V-A bus servos: base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist roll, jaw).

| file | content |
|---|---|
| [kinematics_sheet.md](kinematics_sheet.md) | base frame and zero pose, modified-DH table (Craig), tool transforms (TCP, needle tip), joints (types, axes, positive and drive directions, servo ids), design-angle limits, TCP and pads, pad gap against the jaw angle, clamped tools (part seat, needle holder, needle tip), link CAD frames, unit tolerance block |
| [figures/zero_pose.png](figures/zero_pose.png) | Figure 1: front and side views at the zero pose with the base, DH and TCP frames |
| [drawings/general_arrangement_front.png](drawings/general_arrangement_front.png) | DA6-GA-101, general arrangement, front view (zero pose), scale 1:2 |
| [drawings/general_arrangement_side.png](drawings/general_arrangement_side.png) | DA6-GA-102, general arrangement, side view (zero pose) with the axis heights and distances, scale 1:2 |
| [drawings/general_arrangement_top.png](drawings/general_arrangement_top.png) | DA6-GA-103, general arrangement, top view (zero pose), base footprint, scale 1:1 |
| [drawings/gripper.png](drawings/gripper.png) | DA6-GR-201, gripper and jaw: TCP location and frame, pad faces and pad gap, table T1 (pad gap against the jaw angle), part seat, jaw hinge, envelope, scale 1:1 |
| [drawings/needle.png](drawings/needle.png) | DA6-TL-301, dispensing needle in its holder block clamped in the gripper: holder, needle, tip location, diameter, mass and centre of mass, scale 1:1 with a 5:1 detail |
| [cad/README.md](cad/README.md) | link CAD files (STL per link, link CAD frame, metres) and how the collision envelope differs from them |
| [cad/collision/README.md](cad/collision/README.md) | the collision envelope, on which contact and clearance are judged: convex pieces per link (STL per link, link CAD frame, metres; index `cad/collision/pieces.json`), what is checked against what, the clamped tools' shapes |
| [mass_properties.md](mass_properties.md) | mass, centre of mass and inertia per link, tolerance |

Drawings are to scale when printed on A3; their dimensions are rounded to 0.01 mm, the sheet gives the values to 0.0001 mm. Units: mm and degrees in the documents, metres in the STL files.

