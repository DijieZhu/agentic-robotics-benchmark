# NOTICE - sources of the appliance models

The appliance models in units/ (the practice units) and the lab's units are RoboCasa v1.0 fixture models from the Hugging Face
dataset `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` (revision `420a04af`), licence
**CC-BY-4.0** (https://creativecommons.org/licenses/by/4.0/), as RoboCasa's `download_kitchen_assets.py` ships them (`fixtures_lightwheel.zip`, sha256
`62c1c877...bd6c3e`). This notice is the attribution the licence asks for.

- **Practice units:** p0 `stovetops/Stovetop018`, p1 `microwaves/Microwave075`, p2 `ovens/Oven039`.
- **Modifications:** none to the files. At load time appliance_env.py makes visual meshes massless (as robosuite
  does for its objects), hides collision geometry from the cameras and sets every burner and temperature knob's
  friction loss to 0.20 N m and armature to 0.002 kg m^2 (RoboCasa: 0.75-1.0 N m and 1.0, tuned for its own
  controller; these are a real cooker knob's values). It also adds visual state indicators (a microwave's display bar, an oven's
  lamp; RoboCasa shows neither) and applies a unit's declared mechanism (push to turn: knob friction and
  damping released by an axial push; reversed: knob joint axes flipped; hold to start: a held start button).
  RoboCasa's appliance predicates are re-implemented, not imported.
- **The robot** is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
