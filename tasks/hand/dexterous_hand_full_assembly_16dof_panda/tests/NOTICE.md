# Sources and credits of dexterous_hand_full_assembly_16dof_panda

This note says where the hand of this task, its cell and its other material come from. It is a maintainer file: the
agent never sees this folder, and every file the agent gets calls the hand "the hand", its units by the generic names
of the kit and its bought parts by generic names and invented part numbers. Recorded on 2026-11-21 from the sources
themselves; nothing here is a legal opinion.

## The hand

The hand is the **MIDAS Hand**, right hand, an open-source four-digit robot hand by Alvin Zhu, Mingzhang Zhu, Beom Jun
Kim, Quanyou Wang, Jose Victor S. H. Ramos and Dennis Hong (University of California, Los Angeles), described in
"MIDAS Hand: Modular low-Impedance Direct-drive Anthropomorphic Sensing Hand" (arXiv:2607.14487v1, submitted
2026-07-16) and published at https://midas-hand.com/. It is used exactly as its designers released it. There is no
hidden variant; nothing of the hand is modelled again, scaled, simplified or completed for this task. The nine units
the robot sets are the designers' own bolt-on modules: three identical finger modules, the thumb module, the
electronics stack, the upper shell, the dorsal cover, the power hub and the press-fit cap, onto the palm's bottom
shell. The palm pad and the six screws under it belong to a later station and are left out of the cell.

## Sources, with what each source says about its licence

| Source | Taken from | Version | Licence as stated by the source |
|---|---|---|---|
| The release CAD of the whole hand, `Right_Hand_Assembly.step` (19,364,138 bytes) | the download link of the project's CAD page (Google Drive file `1_KcX3wh2XLXMfi233odvl-5opAglAAh3`) | the designers' STEP export of 2026-05-28, sha256 `228f76b548ed71e754d8ab69e247e42ecf66caff9e8bbacc8462e63b9106c5a7`; the same page links their public Onshape document `77f88cdb5968e438112c92f5` at version `c98f05d28c65fc3cba8c7cf6` | CERN-OHL-P-2.0 |
| The bill of materials | the sheet linked from the project's parts page (CSV export, sheet `764940597`) | sha256 `1c6b1d874a695ed7ad01444c8774e27543a896f7121f7268bc20aba0e949834a` | CERN-OHL-P-2.0 |
| The assembly guide | the project's assembly page and the document it links ("MIDAS Assembly Instruction", 25 pages) | as published on 2026-11-20 | CERN-OHL-P-2.0 (assembly design materials) |
| The project's website | https://midas-hand.com/ and https://github.com/midas-hand-org/midas-hand-org.github.io | commit `45a0e4484f0bffec14177a9ca12591b331ae5c92` (2026-07-20) | website text and media CC BY 4.0; hardware materials CERN-OHL-P-2.0; software MIT |
| The simulation model (URDF, MJCF, meshes) | https://github.com/midas-hand-org/midas_hand_mujoco | commit `6c7d17f4a8cb5414e81653aa2795acc9610e1730` (2026-05-29) | MIT, Copyright (c) 2026 MIDAS Hand Contributors |
| The hand's control software (read for its homing routine) | https://github.com/midas-hand-org/midas_hand_api | commit `6a08263fc607bf219a6b23a9afa5795e3863255a` | MIT, Copyright (c) 2026 MIDAS Hand Contributors |
| The paper | https://arxiv.org/abs/2607.14487 | v1 | CC BY 4.0 |
| The servo maker's CAD model and drawing of the hand's servo, ROBOTIS XM335: `XM335.stp`, `XM335.pdf` | https://www.robotis.com/service/download.php?no=2236 and `no=2235` | sha256 `df2e22d7110e8de1db42a71e409d6cc3984195fcdbb0144b79e0415441b407e7` and `4beaa90293f6ca656951e91b336faa422ef271a72259562f7508836a3762e9aa` | the maker's own CAD download |
| The servo maker's product page (ROBOTIS e-manual, XM335-T323) | https://emanual.robotis.com/docs/en/dxl/x/xm335-t323/ | the page of 2026-11-20 | (c) ROBOTIS |
| The arm maker's ROS driver, read for the arm's default collision reflex (the basis of the robot-contact limit) | https://github.com/frankaemika/franka_ros, `franka_control/config/franka_control_node.yaml` | fetched 2026-11-20 | Apache-2.0 |

The public Onshape document was changed after this export, on 2026-10-02; that later state is not used. The
simulation model was read on the maintainer side as a cross-check of the hinge axes and the four-bar closure (the
shared asset store's posable model copies its body frames and closes the loops as it does); the paper gives the
joints' friction (measured by the designers at five joint types) and the tactile modules' measuring range (the basis
of the part-load limit); the control software's homing routine with the servo maker's torque and current figures is
the basis of the joint-stop limit; the maker's product page gives the servo's mass.

## What this task makes of the sources

| Material | Where | How it derives from the sources |
|---|---|---|
| The cell's units | the hidden kit `gt.tar.xz` (fetched by `verify.py`, pinned in `grader_config.json`): `kit.json`, `mating_graph.json`, `joints.json`, `meshes/visual/`, `meshes/collision/` | The 325 leaf solids of the release STEP grouped into the palm base on the fixture and the nine units (13 solids of the palm pad station left out). Visual meshes: the solids tessellated by the program's shared asset store, and again at 0.05 mm / 20 deg for the cameras' render set; each of the 13 servos is drawn from its maker's CAD model at the designers' placement (three case parts, eight case screws, two connectors, output horn, bearing ring; no idler disc, which the designers' hand does not have). Collision: convex pieces cut from the true solids (CoACD per part, the servo as four slabs of the designers' one-solid servo, carved along every insertion path and joint sweep); fasteners, pins and small wire fixtures are drawn only. Design poses, insertion paths and tool access volumes are measured on the release STEP. No face of any solid was changed. |
| The joints of the digit modules | `joints.json` of the kit; the given module MJCFs | The hinge axes, stops and the four-bar of each finger from the hand's mechanism record (sweeps of the release solids until they meet, cross-checked against the simulation model's axes); friction 0.02 N m from the paper; damping and armature as the cell's estimates (stated in the cell manual). The joints are unpowered in the cell. |
| The given files | the payload `given/parts/` (one STL per unit kind; the two digit modules also as per-link STLs with an MJCF each), `given/kit.json`, `given/purchased/purchased_parts.md`, `given/cell/` | Exported from the kit's visual meshes in each unit's delivered frame (millimetres), neutral headers; the module MJCFs carry the joints as the cell simulates them; the documents are written for this task. |
| The picture manual (25 photographs) | the payload `given/manual/` | Path-traced Blender Cycles renders of the same triangles, in the hand's approved materials, on a cutting mat with a checker card and a steel rule made for the hand tasks; the units laid out, an exploded view, each mate before and after, the finished hand and close-ups. The set in this task folder is the full render. |

Masses. A printed part's mass is its solid volume times the density of its material (PLA 1.24 g/cm3, TPU 1.21 g/cm3);
the bought parts take the hand tasks' one mass table: the servo 27.0 g (its maker's figure), the power hub's board
25.0 g (one component's maker figure plus the rest by volume), the other bought parts by volume times a nominal
density (estimates). The agent's documents give each mass as the mass of this build.

Marks. No name, logo or maker's mark is modelled as geometry on any solid, given file or kit mesh of this task (the
geometric text check of the task's own files, run on every given mesh and every mesh the cameras draw, found none),
so nothing was removed from any solid. Marks that name nobody stay as modelled: the two power symbols on the hub
board's rocker switch, a blank label recess on the servo, the recesses of screw heads.

Modification notice (CERN-OHL-P-2.0, section 3.3): the files of this task that derive from the hardware design files
(the hidden kit's meshes and data, the given STL files and module models, the picture manual) were made from
`Right_Hand_Assembly.step` in November 2026 by grouping its solids into units, changing their frames, tessellating
them, cutting convex collision pieces from them, converting their format (mesh, MJCF, image) and measuring them. The
geometry of no part was changed.

## Invented names

The unit and part names of the agent's files and the part numbers of its bought parts list are generic or invented
for the benchmark and stand for no product. The hand, its designers, their school, the makers of its bought parts and
every real part number are absent from the instruction, the payload, the file headers, the photographs and the given
meshes.

## Not shipped

Neither the release STEP, the print files, the simulation model, the bill of materials, the assembly guide, the paper
nor the maker's servo files are in the task, its image or its payload.

## The cell, the room and the tools

The Franka Panda is the MuJoCo Menagerie `franka_emika_panda` model (Apache-2.0, commit c96a32d, byte-identical to the
model of `tensegrity_strut_assembly_3bar_panda`), with its stock flat finger pads; its licence file ships with it in
`given/panda/`. The cell's structure and its grader modules (`arm/assembly_cell`, `assembly_six`, `assembly_info`,
`render`, `render_filament`, `room`, `robotics_grader`, `common`) are vendored from the benchmark's arm assembly tasks
(`robotic_arm_full_assembly_so101_panda` and `tensegrity_strut_assembly_3bar_panda`); `verify.py`, the kit's download
and the reach check follow `tensegrity_strut_assembly_3bar_panda`. The physics is MuJoCo (Apache-2.0). The agent
cameras' RGB is rendered by MuJoCo's Filament module on CPU in a baked room; depth by MuJoCo on the same geometry.

The room is a hardware team's lab bay laid out by code from the modular room pieces of the **Office and Police Station
Pack (Modular)**, version 2.0.0, by Finward Studios (Unity Asset Store; one of the benchmark maintainers' art packs,
licensed for redistribution), exported without the Unity Editor: walls, windows, doors, ceiling, floor, ductwork,
pendant lamps, shelving, desks, chairs, whiteboards and most of the dressing. The pack's readme states: "One or more
textures on this 3D-model have been created with images from Textures.com." The rest of the dressing comes from three
more of the maintainers' Unity Asset Store packs: a wall clock, an electrical panel, hand tools and a mug from House
Props Furniture Vol 2 (1.2.0); a fire extinguisher and a call point from Factory Interior Factory Props Vol 1; a utility
cart from Warehouse Props Vol 1 (1.0). The camera portal, the T-slot benches, an open-frame printer, the totes, an ESD
mat, wire coils, cable trays, hanger rods and the set lamps are made by code. Lettering and print found on the pack
textures (screens, signage, labels) were painted out of the task's copies; the room's lightmaps are baked with Blender
Cycles. The bench top is a maple butcher block generated for the hand tasks. The fixture, the rest deck, the thumb's pad, the trays and the datum boards
are made by code for this cell.

Tools: the collision pieces were made with CoACD (MIT) and manifold3d (Apache-2.0); the mesh handling uses trimesh
(MIT); the manual is rendered with Blender (GPL-2.0-or-later; only the rendered images ship).

## Licence texts

### CERN Open Hardware Licence Version 2 - Permissive (CERN-OHL-P-2.0)

Covers the hand's hardware design files: the release STEP, the bill of materials and the assembly guide. The text below
is the licence as CERN publishes it (the copy recorded with the sources, sha256
`6e471d647db50527d0c75ee5f1a55a74012ab177c73282bef0bdc7f6562b7ded`).

```
CERN Open Hardware Licence Version 2 - Permissive


Preamble

CERN has developed this licence to promote collaboration among
hardware designers and to provide a legal tool which supports the
freedom to use, study, modify, share and distribute hardware designs
and products based on those designs. Version 2 of the CERN Open
Hardware Licence comes in three variants: this licence, CERN-OHL-P
(permissive); and two reciprocal licences: CERN-OHL-W (weakly
reciprocal) and CERN-OHL-S (strongly reciprocal).

The CERN-OHL-P is copyright CERN 2020. Anyone is welcome to use it, in
unmodified form only.

Use of this Licence does not imply any endorsement by CERN of any
Licensor or their designs nor does it imply any involvement by CERN in
their development.


1 Definitions

  1.1 'Licence' means this CERN-OHL-P.

  1.2 'Source' means information such as design materials or digital
      code which can be applied to Make or test a Product or to
      prepare a Product for use, Conveyance or sale, regardless of its
      medium or how it is expressed. It may include Notices.

  1.3 'Covered Source' means Source that is explicitly made available
      under this Licence.

  1.4 'Product' means any device, component, work or physical object,
      whether in finished or intermediate form, arising from the use,
      application or processing of Covered Source.

  1.5 'Make' means to create or configure something, whether by
      manufacture, assembly, compiling, loading or applying Covered
      Source or another Product or otherwise.

  1.6 'Notice' means copyright, acknowledgement and trademark notices,
      references to the location of any Notices, modification notices
      (subsection 3.3(b)) and all notices that refer to this Licence
      and to the disclaimer of warranties that are included in the
      Covered Source.

  1.7 'Licensee' or 'You' means any person exercising rights under
      this Licence.

  1.8 'Licensor' means a person who creates Source or modifies Covered
      Source and subsequently Conveys the resulting Covered Source
      under the terms and conditions of this Licence. A person may be
      a Licensee and a Licensor at the same time.

  1.9 'Convey' means to communicate to the public or distribute.


2 Applicability

  2.1 This Licence governs the use, copying, modification, Conveying
      of Covered Source and Products, and the Making of Products. By
      exercising any right granted under this Licence, You irrevocably
      accept these terms and conditions.

  2.2 This Licence is granted by the Licensor directly to You, and
      shall apply worldwide and without limitation in time.

  2.3 You shall not attempt to restrict by contract or otherwise the
      rights granted under this Licence to other Licensees.

  2.4 This Licence is not intended to restrict fair use, fair dealing,
      or any other similar right.


3 Copying, Modifying and Conveying Covered Source

  3.1 You may copy and Convey verbatim copies of Covered Source, in
      any medium, provided You retain all Notices.

  3.2 You may modify Covered Source, other than Notices.

      You may only delete Notices if they are no longer applicable to
      the corresponding Covered Source as modified by You and You may
      add additional Notices applicable to Your modifications.

  3.3 You may Convey modified Covered Source (with the effect that You
      shall also become a Licensor) provided that You:

       a) retain Notices as required in subsection 3.2; and

       b) add a Notice to the modified Covered Source stating that You
          have modified it, with the date and brief description of how
          You have modified it.

  3.4 You may Convey Covered Source or modified Covered Source under
      licence terms which differ from the terms of this Licence
      provided that You:

       a) comply at all times with subsection 3.3; and

       b) provide a copy of this Licence to anyone to whom You
          Convey Covered Source or modified Covered Source.


4 Making and Conveying Products

You may Make Products, and/or Convey them, provided that You ensure
that the recipient of the Product has access to any Notices applicable
to the Product.


5 DISCLAIMER AND LIABILITY

  5.1 DISCLAIMER OF WARRANTY -- The Covered Source and any Products
      are provided 'as is' and any express or implied warranties,
      including, but not limited to, implied warranties of
      merchantability, of satisfactory quality, non-infringement of
      third party rights, and fitness for a particular purpose or use
      are disclaimed in respect of any Source or Product to the
      maximum extent permitted by law. The Licensor makes no
      representation that any Source or Product does not or will not
      infringe any patent, copyright, trade secret or other
      proprietary right. The entire risk as to the use, quality, and
      performance of any Source or Product shall be with You and not
      the Licensor. This disclaimer of warranty is an essential part
      of this Licence and a condition for the grant of any rights
      granted under this Licence.

  5.2 EXCLUSION AND LIMITATION OF LIABILITY -- The Licensor shall, to
      the maximum extent permitted by law, have no liability for
      direct, indirect, special, incidental, consequential, exemplary,
      punitive or other damages of any character including, without
      limitation, procurement of substitute goods or services, loss of
      use, data or profits, or business interruption, however caused
      and on any theory of contract, warranty, tort (including
      negligence), product liability or otherwise, arising in any way
      in relation to the Covered Source, modified Covered Source
      and/or the Making or Conveyance of a Product, even if advised of
      the possibility of such damages, and You shall hold the
      Licensor(s) free and harmless from any liability, costs,
      damages, fees and expenses, including claims by third parties,
      in relation to such use.


6 Patents

  6.1 Subject to the terms and conditions of this Licence, each
      Licensor hereby grants to You a perpetual, worldwide,
      non-exclusive, no-charge, royalty-free, irrevocable (except as
      stated in this section 6, or where terminated by the Licensor
      for cause) patent licence to Make, have Made, use, offer to
      sell, sell, import, and otherwise transfer the Covered Source
      and Products, where such licence applies only to those patent
      claims licensable by such Licensor that are necessarily
      infringed by exercising rights under the Covered Source as
      Conveyed by that Licensor.

  6.2 If You institute patent litigation against any entity (including
      a cross-claim or counterclaim in a lawsuit) alleging that the
      Covered Source or a Product constitutes direct or contributory
      patent infringement, or You seek any declaration that a patent
      licensed to You under this Licence is invalid or unenforceable
      then any rights granted to You under this Licence shall
      terminate as of the date such process is initiated.


7 General

  7.1 If any provisions of this Licence are or subsequently become
      invalid or unenforceable for any reason, the remaining
      provisions shall remain effective.

  7.2 You shall not use any of the name (including acronyms and
      abbreviations), image, or logo by which the Licensor or CERN is
      known, except where needed to comply with section 3, or where
      the use is otherwise allowed by law. Any such permitted use
      shall be factual and shall not be made so as to suggest any kind
      of endorsement or implication of involvement by the Licensor or
      its personnel.

  7.3 CERN may publish updated versions and variants of this Licence
      which it considers to be in the spirit of this version, but may
      differ in detail to address new problems or concerns. New
      versions will be published with a unique version number and a
      variant identifier specifying the variant. If the Licensor has
      specified that a given variant applies to the Covered Source
      without specifying a version, You may treat that Covered Source
      as being released under any version of the CERN-OHL with that
      variant. If no variant is specified, the Covered Source shall be
      treated as being released under CERN-OHL-S. The Licensor may
      also specify that the Covered Source is subject to a specific
      version of the CERN-OHL or any later version in which case You
      may apply this or any later version of CERN-OHL with the same
      variant identifier published by CERN.

  7.4 This Licence shall not be enforceable except by a Licensor
      acting as such, and third party beneficiary rights are
      specifically excluded.
```

### MIT License, Copyright (c) 2026 MIDAS Hand Contributors

Covers the hand's software repositories (the simulation model, whose body frames and loop closure the shared asset store's posable model follows, and the control software, read for its homing routine).

```
MIT License

Copyright (c) 2026 MIDAS Hand Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The designers' website text and media and their paper are under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). No code, text, figure or image of the designers is in the task.
