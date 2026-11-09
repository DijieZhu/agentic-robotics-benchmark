# Site brief: the console table with the toy groceries

## The customer and the job

A home-robotics team is preparing a mobile manipulator (one arm, a parallel gripper) for a tidy-up job at a client's
home: clear the toy groceries off a console table, empty its open drawer, and put everything back on demand. The
team's planner and grasp tests never run first on the real robot. They run in simulation, on a digital twin of the
site as it was found, and only plans that work in the twin are sent to the robot.

The site cannot be visited again. A survey camera took one colour photo of the console before anything was touched
(`photo.png`; what is known about the camera is in `photo_info.json`). That photo is all there is: no depth, no
measurements, no second view. You build the twin from it.

## What the twin is used for, and what that requires

- **The robot picks items one at a time.** Every loose grocery toy in the photo is an item: on the top, in the open
  drawer, on a shelf, standing, lying, or resting on another item. Each item is its own rigid body that can move
  freely. Two items fused into one body would be lifted together in simulation, so the plan that unstacks them would
  be wrong on the real table.
- **The drawer slides.** The robot will pull the drawer out and push it back. The drawer is one body on one straight
  slide, and the items in it must ride along with it as they would in reality.
- **The twin holds the photographed state.** The planner starts from the state in the photo. When the simulation
  starts, every item that rests in the photo must stay where it is: stacks keep standing, items that lean on each
  other keep leaning, nothing sinks into the furniture or flies off. A twin that rearranges itself in the first
  seconds plans for a scene that does not exist.
- **What touches what.** Lifting an item must disturb, or carry along, exactly the items it would disturb or carry
  in reality, so each item has to sit on (or lean on) what it really sits on, in the right place, at the right size.
- **Where things are.** The robot localises with its own camera against the twin, so the twin's geometry seen
  through the photo's camera must line up with the photo: the items, the console's visible surfaces, the drawer,
  and the wall and floor that the photo shows.

## What is an item and what is environment

- Items: the toy grocery products. Their product names are in `product_names.json`; label each item with its
  product. The list names the products of the whole toy set; it does not say which of them are in the photo, or how
  many of each.
- Environment: the console (its carcass, top, shelves and panels), the drawer, the wall, the window and the floor
  that the photo shows. It is static, except the drawer.
- Anything else in the photo is environment or can be left out.

## What the team does not need

Textures and colours are welcome (they help you compare a render with the photo) but play no part in acceptance.
Exact masses, friction coefficients and contact softness are not needed: the team's simulator applies its own
standard values (`twin_contract.md`), as long as each item's mass is physically plausible for its size.

The full technical contract (files, frames, the camera, the rules for bodies and the drawer, the standard
simulation, the probes and how the twin is assessed) is in `twin_contract.md`.
