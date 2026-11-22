# Sources and credits of dexterous_hand_cad_design_from_photos

This note says where the hand of this task and its other material come from. The agent never sees this folder: every
file the agent gets calls the hand "the hand" and its bought parts by generic names and invented part numbers.
Recorded on 2026-11-20 from the sources themselves; nothing here is a legal opinion.

## The hand

The hand is the **MIDAS Hand**, right hand, an open-source four-digit robot hand by Alvin Zhu, Mingzhang Zhu, Beom Jun
Kim, Quanyou Wang, Jose Victor S. H. Ramos and Dennis Hong (University of California, Los Angeles), described in
"MIDAS Hand: Modular low-Impedance Direct-drive Anthropomorphic Sensing Hand" (arXiv:2607.14487v1, submitted
2026-07-16) and published at https://midas-hand.com/. It is used exactly as its designers released it. There is no
hidden variant; nothing of the hand is modelled again, scaled, simplified or completed for this task.

## Sources, with what each source says about its licence

| Source | Taken from | Version | Licence as stated by the source |
|---|---|---|---|
| The release CAD of the whole hand, `Right_Hand_Assembly.step` (19,364,138 bytes) | the download link of the project's CAD page (Google Drive file `1_KcX3wh2XLXMfi233odvl-5opAglAAh3`) | the designers' STEP export of 2026-05-28, sha256 `228f76b548ed71e754d8ab69e247e42ecf66caff9e8bbacc8462e63b9106c5a7`; the same page links their public Onshape document `77f88cdb5968e438112c92f5` at version `c98f05d28c65fc3cba8c7cf6` | CERN-OHL-P-2.0 |
| The print files, `MIDAS_ready_to_print_bambu.zip` | the same page (Google Drive file `14vvAxNZlIkaA5PynW53nXllJ382IE7gY`) | sha256 `411aed8b772b813df424753f64b1307907a51a832ca1651013ada8619281bff7` | CERN-OHL-P-2.0 |
| The bill of materials | the sheet linked from the project's parts page (CSV export, sheet `764940597`) | sha256 `1c6b1d874a695ed7ad01444c8774e27543a896f7121f7268bc20aba0e949834a` | CERN-OHL-P-2.0 |
| The assembly guide | the project's assembly page and the document it links ("MIDAS Assembly Instruction", 25 pages) | as published on 2026-11-20 | CERN-OHL-P-2.0 (assembly design materials) |
| The project's website | https://midas-hand.com/ and https://github.com/midas-hand-org/midas-hand-org.github.io | commit `45a0e4484f0bffec14177a9ca12591b331ae5c92` (2026-07-20); the live pages are byte-identical to it | website text and media CC BY 4.0; hardware materials CERN-OHL-P-2.0; software MIT |
| The simulation model (URDF, MJCF, meshes) | https://github.com/midas-hand-org/midas_hand_mujoco | commit `6c7d17f4a8cb5414e81653aa2795acc9610e1730` (2026-05-29) | MIT, Copyright (c) 2026 MIDAS Hand Contributors |
| The paper | https://arxiv.org/abs/2607.14487 | v1 | CC BY 4.0 |
| The servo maker's CAD model and drawing of the hand's servo, ROBOTIS XM335: `XM335.stp`, `XM335.pdf` | https://www.robotis.com/service/download.php?no=2236 and `no=2235` | sha256 `df2e22d7110e8de1db42a71e409d6cc3984195fcdbb0144b79e0415441b407e7` and `4beaa90293f6ca656951e91b336faa422ef271a72259562f7508836a3762e9aa` | the maker's own CAD download, recorded as reference material |
| The servo maker's product page (ROBOTIS e-manual, XM335-T323) | https://emanual.robotis.com/docs/en/dxl/x/xm335-t323/ | the page of 2026-11-20 | (c) ROBOTIS |

The public Onshape document was changed after this export, on 2026-10-02; that later state is not used. The simulation
model and the paper were read on the maintainer side only, as a cross-check of the hinge axes and of the way the last
two hinges of a finger move together, and the maker's product page for the servo's mass.

## What this task makes of the sources

| Material | Where | How it derives from the sources |
|---|---|---|
| The true solids of the 52 made parts (27 shapes) | `reference/parts/K01.brep` to `K27.brep`, byte-identical in `../solution/reference/parts/` | Each shape is the solid of its part family in the release STEP, taken out entity for entity and written as an exact B-rep in the part's own frame. No face is changed. Three shapes carry a triangle mesh beside the B-rep (`K02.mesh.npz`, `K15.mesh.npz`, `K23.mesh.npz`), made from the same B-rep for the volume measures. |
| The reference of the grading | `reference/reference.json`, `reference/pairs.json`, `reference/samples.npz`, `reference/purchased_pins.json` | Measured on the release STEP: the pose of every made part and given piece, the round holes of the made parts, the hinge axes, the closed chains of the three fingers, the fingertip points, the contacts and overlaps the hand has itself, surface samples. The hinge limits are the ranges through which the release solids turn until they meet (collision sweeps), not the range table of the paper. |
| The given parts (8 types, 53 pieces) | the payload: `purchased/<part number>/<part number>.step` | The designers' own solids of the bought parts as they stand in the release STEP (servo, bearing, dowel pin, two tactile modules, three circuit boards), each moved into a documented frame, with neutral STEP headers and product names. The servo's given solid is the designers' one-solid model, which follows the maker's outer shape with the rear bearing kept and without the idler disc. |
| The data pages, drawings and the parts list | the payload: `purchased/<part number>/`, `parts_list.md` | Written and drawn for this task from the given solids and from the release STEP's part counts; none is a copy of a maker's document. The fastener counts are those of the release STEP. |
| The photographs (363) | the payload: `photos/` | Path-traced renders of the release geometry: the hand complete, in other poses, taken apart stage by stage in the order of the designers' assembly guide read backwards, and part by part on a cutting mat. The printed parts carry the layer relief of the designers' print settings (0.16 mm layers, 0.20 mm on the large palm parts). In the photographs each of the 13 servos is drawn from its maker's CAD model at the designers' placement (three case parts, eight case screws, two connectors, output horn, bearing ring; no idler disc, which the designers' hand does not have). The 220 screws of the release STEP are in the photographs and in neither the given files nor the reference. No cable is shown: the release STEP has none. |

Masses. The made parts' masses are solid volume times the density of the material (PLA 1.24 g/cm3, one part in TPU
1.21 g/cm3; the real prints are lighter, being printed with two walls and 15 % infill). No measured mass of a bought
part exists in the designers' files. The agent's documents state each given part's mass as the mass of this build; the
basis is: the servo 27.0 g, its maker's published figure; the power and bus hub board 25.0 g, one component's maker
figure plus the rest of the solid by volume times a nominal density; the other six types by volume times a nominal
density (estimates).

Marks. No name, logo or maker's mark is modelled as geometry on any made or given solid, so nothing was removed from
any solid. Marks that name nobody stay as modelled: the power symbols on the rocker switch of the hub board's solid, a
blank label recess on the servo, the pin-1 marks (triangles and strokes) beside the connectors of the two tactile
modules and of the tactile interface board, the recesses of screw heads. The cap of the palm cover carries no
lettering in the release STEP (a figure of the paper shows the hand's and the lab's names there).

Modification notice (CERN-OHL-P-2.0, section 3.3): the files of this task that derive from the hardware design files
(the B-reps and meshes of `reference/parts/` and `../solution/reference/parts/`, the reference data, the given STEP
files, the photographs) were made from `Right_Hand_Assembly.step` in November 2026 by taking each part family out of
the assembly, changing its frame, converting its format (B-rep, mesh, image) and measuring it. The geometry of no part
was changed.

## Invented names

The part numbers in the agent's files (prefixes `VKR-`, `HNL-`, `ZCP-`, `BWD-`, `JQT-`) and the generic names of the
given parts are invented for the benchmark and stand for no product. The tags `P01` to `P52` are ordered by size
and say nothing about function. The hand, its designers, their school, the makers of its bought parts and every real
part number are absent from the instruction, the payload, the STEP headers, the photographs and the given solids.

## Not shipped

Neither the release STEP, the print files, the simulation model, the bill of materials, the assembly guide, the paper
nor the maker's servo files are in the task, its image or its payload. `public_blocklist.json` lists 59 public source
files by digest: the release STEP, the published archives with their members, the robot and simulation descriptions, the
simulation meshes, the two web models of the project's site, the STEP files of one circuit board and of the maker's
servos, and one data sheet and two drawings of the maker. A submission folder that holds one of them scores 0.

## Background of the photographs

The photographs are rendered with Blender Cycles on a bench in the lab room of the project's licensed art assets (the
room bundle `room_lab_v4r3`), as in the sibling CAD-from-photos tasks. The world light is the HDRI `schadowplatz_4k`,
public CC0 material. The bench top is a procedural maple butcher block made for the hand tasks. The cutting mat, the two
kinds of checker card, the block behind the standing card, the steel rule and the tape tags are models made for this
task and carry no maker's name; the tags are lettered in DejaVu Sans Mono. None of these assets is shipped: only the
rendered photographs are.

## Third-party software

- FreeCAD 1.1.4 (LGPL-2.0-or-later, https://github.com/FreeCAD/FreeCAD) is installed unmodified in the task image from
  its release AppImage, pinned by sha256 in `../environment/Dockerfile`.
- The FreeCAD MCP server neka-nat/freecad-mcp at commit `78bf0230a03324b56c9a0b7f55150fb5be2a0d42` (MIT,
  https://github.com/neka-nat/freecad-mcp) is vendored unmodified in `../environment/vendor/freecad-mcp/` with its
  `LICENSE`.
- The grader's Python packages are those of the benchmark's shared base image (`../environment/requirements.txt`).

## Licence text

### CERN Open Hardware Licence Version 2 - Permissive (CERN-OHL-P-2.0)

Covers the hand's hardware design files: the release STEP, the print files, the bill of materials and the assembly
guide. The text below is the licence as CERN publishes it (the copy recorded with the sources, sha256
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

The designers' software is under the MIT License (Copyright (c) 2026 MIDAS Hand Contributors), their website text and
media and their paper under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). No code, text, figure or image of
them is in the task.
