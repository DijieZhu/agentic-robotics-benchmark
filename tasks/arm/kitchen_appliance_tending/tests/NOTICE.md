# NOTICE - sources of this task's assets (verify / maintainers only; never shown to the agent)

The appliance models (tests/units/, environment/setup/ref/units/, solution/oracle/units/ for the hidden units;
environment/setup/payload/units/ for the practice units) are RoboCasa v1.0 fixture models from the Hugging Face
dataset `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` (revision `420a04af`), licence
**CC-BY-4.0**, as RoboCasa's `download_kitchen_assets.py` ships them (`fixtures_lightwheel.zip`, sha256
`62c1c877...bd6c3e`). This notice is the attribution the licence asks for.

- **Hidden units:** u0 `stovetops/Stovetop011`, u1 `microwaves/Microwave081`, u2 `ovens/Oven036`,
  u3 `stovetops/Stovetop002`, u4 `microwaves/Microwave087`, u5 `ovens/Oven053`, u6 `stovetops/Stovetop025`,
  u7 `microwaves/Microwave071`, u8 `ovens/Oven034`.
- **Practice units:** p0 `stovetops/Stovetop018`, p1 `microwaves/Microwave075`, p2 `ovens/Oven039`.
- **Selection:** tests/hidden/selection.json (the rule, the os.urandom draw, the recheck results).
- **Modifications:** none to the files. At load time appliance_env.py makes visual meshes massless (as robosuite
  does for its objects), hides collision geometry from the cameras and sets every burner and temperature knob's
  friction loss to 0.20 N m and armature to 0.002 kg m^2 (RoboCasa: 0.75-1.0 N m and 1.0; owner decision
  2026-11-12, design section 10). It also adds visual state indicators (a microwave's display bar, an oven's
  lamp; RoboCasa shows neither) and applies a unit's declared mechanism (push to turn: knob friction and
  damping released by an axial push; reversed: knob joint axes flipped; hold to start: a held start button).
  RoboCasa's appliance predicates are re-implemented, not imported.
- **The robot** is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
