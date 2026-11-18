# NOTICE (grader-side; never staged into the agent's home)

## The robot

The robot of this task is the open-source three-strut, cable-driven tensegrity rolling robot of the Yale Faboratory
(Rebecca Kramer-Bottiglio, Yale University) and the PRACSYS lab (Kostas Bekris, Rutgers University), the fourth
version of their three-bar platform ("Tribar"): W. R. Johnson III, P. Meng, N. Chen, L. Cimatti, A. Vercoutere,
M. Aanjaneya, R. Kramer-Bottiglio, K. E. Bekris, "An Open-Source, Reproducible Tensegrity Robot that can Navigate
Among Obstacles", IEEE Robotics and Automation Letters 11(6):7054-7061, 2026, DOI 10.1109/LRA.2026.3681109
(arXiv:2511.05798). The strain sensors and their cut outlines come from the earlier paper on the platform, W. R.
Johnson III, X. Huang, S. Lu, K. Wang, J. W. Booth, K. Bekris, R. Kramer-Bottiglio, "Impact-resistant, autonomous
robots inspired by tensegrity architecture" (arXiv:2501.15078; DOI 10.1038/s42256-026-01280-2) and its dataset.

Nothing of the robot is modelled, redrawn or completed for this task, and there is no hidden variant: the photos are
renders of the designers' published whole-strut mesh, the given files are the designers' own files (cropped, never
rewritten), and every hidden truth is read from the designers' files or from data the vendors publish themselves.
The cables, the strain sensors and the leads have no geometry in the sources and are in no photo.

## Sources, licences, and what this task does with each

"Given" = the agent receives it (in the image's payload or the photo set); "hidden" = only the grader side reads it
(`tests/`, `solution/`, the maintainers' build). sha256 of the file as taken from its source. The licence column
states what the source itself says, as recorded on 2026-11-17; nothing here is a legal opinion.

| # | source (the designers') | where it was taken from | sha256 | what the source says about its licence | use in this task |
|---|---|---|---|---|---|
| 1 | whole-strut mesh `Will_strut_scaled_centered.ply` | https://github.com/shiyoung77/tensegrity_perception , branch `ros` (commit 726ed401) | `8dc4caff1efa26468660f30d5e2d91f7d927d6b287480ecd3b06b351f88a4614` | the repository has no licence file; the `package.xml` of its `ros` branch says `MIT` | given only as renders: the 320 photos. Hidden: the key dimensions, counts and fastener sizes are measured on it. No mesh file is shipped |
| 2 | tracking configuration `data_cfg.json` | the same repository and branch | `5e6db514515d42610f6ed6865a7403d39478e08c62cd7ffd64f896172584a393` | as row 1 | given byte for byte as `software/host/tracking_config.json` |
| 3 | controller firmware `onboard/onboard.ino/onboard.ino.ino` | https://github.com/ChemStud24/tensegrity , branch `rss25` (commit 4b79620, blob 665ace8c3cfe) | `b19b59eb75c6cdbbc8cabdd78a21de4c9f8f66fe62679e892010c93cea9f5ed2` | the repository has **no licence file**; its `package.xml` says `<license>TODO</license>`. A Zenodo deposit of the repository's `master` branch (record 20347587, `tensegrity-NMI.zip`, sha256 `14175a2bb142491caef8fa4322f4c925b96b3ab687a55823da855efda3dafc8a`, folder `ChemStud24-tensegrity-e4711c8/`) is recorded by the maintainers as marked CC BY 4.0; the zip itself holds no licence file. In that zip the firmware equals this file but for one trailing newline | given as `software/controller/onboard.ino` with three edits (below) |
| 4 | host control script `src/run_tensegrity.py` | the same repository, branch `rss25` (blob 061b86c6b18f) | `14a53af9e3c2e432b4df8b250caec9858d5aa03a15f3a81011bf0c9e63ae2ed7` | as row 3; the script in the Zenodo deposit (the `master` version) **differs** from this `rss25` version | given as `software/host/run_tensegrity.py` with one deletion (below) |
| 5 | calibration files `calibration/calibration.json`, `calibration/new_calibration.json` | the same repository, `master` (commit e4711c8; blobs 60f3ed2678e9, 1226fe008380) | `f4e8703d8a06007986f70f6b65284c2ffbf273de45db17d4a5cd70f647372f71`, `5ca8e4ab7cd217bc4d35a0f85235678412fd5512693f61b3a7ad0dff57fc8e5a` | as row 3; both files are byte-identical in the Zenodo deposit | given byte for byte under `software/host/calibration/` |
| 6 | main board files `electronics/v3 Tensegrity Motherboard.brd`, `.sch` (EAGLE) | the same repository, `master` (commit e4711c8) | `9b82a9fbc8f01d505b7e2dcada6e47cdb6cbc13acdec08d7c172d485822b156c`, `d7ca816ab816ad368820fb9625beff33ab618fd56985fd5f3b381a00d539457d` | the board file carries, on its documentation layer, a frame element with the texts "Released under the Creative Commons Attribution Share-Alike 4.0 License", "https://creativecommons.org/licenses/by-sa/4.0/" and "Designed by:" (with the initials WRJ on the board); the repository's own statement is as row 3; both files are byte-identical in the Zenodo deposit | given only as the adaptation `board/fab_data.json` (below). Hidden: the nets, designators and part names are the ledger's sources |
| 7 | strain-sensor cut outlines, `sensors.DXF` of the short and of the long sensors | figshare dataset "Impact-resistant, autonomous robots inspired by tensegrity architecture", DOI 10.6084/m9.figshare.32273583 (v1), file `Sensor_Fabrication.zip` | `e8cbb80050fecdc968319c99dd3b11d7b9e0ce867cbea6616d4d67ec242149bb` (short), `a3033959c3db3ccb34f64b0e45380c3469bfaca5317717b38ec10a0f5414d58a` (long) | **CC BY 4.0** (the dataset's licence field) | given as `sensors/short_sensor_cut.dxf`, `sensors/long_sensor_cut.dxf` with three edits each (below) |
| 8 | bill-of-materials workbook `BOM.xlsx` | the project's website https://tensegrity.yale.edu/ (media file 62) | `3dea5cc931d8758edf7f7f91b350ff498b703577aa277af90cf918284448c613` | neither the site nor the file carries a licence or copyright statement | hidden only: the BOM rows and quantities of `gt/ledger.json`, the pack sizes, unit prices and discount tiers of `gt/price_table.json`. Not given, in whole or in part |
| 9 | assembly guide, version 3 (`Guide_to_assembly_a_tensegrity_robot_V3.pdf`) | the same website (media file 75) | `912c44ccae2bd04e7d734ca0d4771612b2ed40e37c9bd8a30f138cbe300de49e` | as row 8 | hidden cross-check only (the colour rule of the net, where the sensors plug in); short statements of it are quoted in the ledger's `contradictions` and `cross_checks`. Not given |
| 10 | the two papers named above | IEEE Xplore, arXiv | - | the journal version is (c) IEEE; the arXiv versions carry arXiv's non-exclusive distribution licence | attribution and cross-check only; no text or figure of them is in the task |
| 11 | the designers' CAD package ("tribar-2" on GrabCAD), as downloaded by the benchmark's maintainers | https://grabcad.com/library/tribar-2 | (package not redistributed) | GrabCAD's site terms: a user's own, non-commercial, internal use; no redistribution to third parties | **no file of the package is in this task** (payload, photo set, `tests/`, `solution/`). The maintainers read its four purchase invoices as a cross-check of part numbers and prices: nine rows of `gt/ledger.json` carry an `invoice_check` (seller, part number, unit price), one of them as a recorded contradiction that widened an accepted set of part numbers. No name, address, order number or quantity of an invoice is recorded anywhere |

### What was changed in the given files (modification statements)

Date of the modifications: 2026-11-17. Every edit is listed below and, with its position, in the maintainers' cut manifest (ten edits) and
their map of the board's designators; apart from them the given files equal their sources byte for byte.

* **`board/fab_data.json`, adapted from the board file (row 6).** A deterministic, item-by-item export of the board
  file's geometry and connectivity into a JSON document: the board outline and its two cut-outs; for each of the 23
  placed parts its position, rotation and side and its pads (position, drill and pad diameter and shape, or the
  size and copper layer of a surface-mount pad) and mounting holes; the connectivity of the 36 nets (which pads share
  copper). Left out: every part number, value, package, library and device name and description, every symbol pin
  name, every net name and designator (replaced by the neutral numbers `N001`..`N036` and `X01`..`X23`; pad numbers
  kept), every text on the board (the designer's initials, "Designed by:", the licence frame), the documentation
  frame element itself (it has no pads) and two signals that reach one pad only. No coordinate, pad, hole or
  connection was altered, added or removed. `board/fab_data.md` (the description of the format) is written for this
  task. If the board file's Creative Commons Attribution-ShareAlike 4.0 statement governs it, `fab_data.json` is an
  adaptation under the same licence, with this paragraph as the indication of changes and row 6 and the paper above
  as the attribution.
* **`software/controller/onboard.ino` (row 3).** Two string constants (the network name and the network password of
  the lab's hotspot) replaced by `"<network-name>"` and `"<network-password>"`; one given name in a comment replaced
  by "the designer". Nothing else; the commented-out code is kept.
* **`software/host/run_tensegrity.py` (row 4).** One nine-line docstring-style description block (original lines
  308-316: which strut carries which sensors, motors and IMU) deleted; the abstract syntax tree of the rest is
  unchanged.
* **`sensors/short_sensor_cut.dxf`, `sensors/long_sensor_cut.dxf` (row 7, CC BY 4.0).** In each file two text
  entities (the two lines of a CAD program's educational-version watermark) deleted and the "last saved by" account
  name of the drawing properties blanked; every geometric entity (326 and 272) is unchanged.
* **The photo set (row 1).** 320 JPEG renders (2016 x 1512, Blender 5.2.2 Cycles) of the mesh's 494 edge-connected
  pieces, grouped and posed as rigid bodies only (the complete strut, eight teardown stages, the made parts laid out
  on a cutting mat); no triangle is changed, no cable, lead or sensor is drawn. 287 of them are the photo set of the
  sibling task `tensegrity_cad_design_from_photos`, byte for byte; 33 were rendered for this task with the same scene.
  The files carry no metadata beyond a plain JFIF header.

## Vendor data

The eight vendor family pages of the payload (`vendor/VD-01` .. `VD-08`) carry facts only (outlines, pin counts, pin
tables, ratings, series tables, standard sizes), under the vendors' real names and part numbers, in this task's own
layout: no price, no vendor text, drawing or image is copied. Every number is in the maintainers' vendor
fact pack with the public page or document it was read from (147 captures, no login, each with its URL, time and
sha256): Arduino, Pololu, Adafruit, SparkFun, Pimoroni, Seeed Studio, JST,
Sullins, Texas Instruments, Allegro MicroSystems, Microchip, CNC Kitchen, Albany County Fasteners and Aspen
Fasteners, and three chip makers' data sheets hosted by those sellers (TDK InvenSense, NXP / Freescale, Toshiba). One
further page was captured and no number taken from it (McMaster-Carr: it needs JavaScript). Where a vendor publishes
no value the page says so.

Hidden-side vendor files, none of them shipped: the gearmotor's product page (Pololu item 5192, capture sha256
`bd881913ff9ef94ab073eba9ede3b0d619efa1ab1c770ef65d211b87253d171b`); the controller board's CAD package
`ABX00027-cad-files.zip` (Arduino, sha256 `56477d319fa577bf70df52f949b6b4ea7fa4e128cdc9f1a0ba256a319f50fb16`; its
`License.txt`: Creative Commons Attribution-ShareAlike 4.0 International); and, for the maintainers' trace
self-check only, the vendors' own board files and pin table from their public repositories:
`Adafruit_MPR121Q_QT.brd` (github.com/adafruit/Adafruit-MPR121-PCB, sha256
`e75b285fefa6d5598eafa155771f5674764fb269482d9f3071dd61d4bbf072cd`; its `license.txt`: Creative Commons
Attribution-ShareAlike 3.0 Unported), `SparkFun_Motor_Driver-TB6612FNG_v11.brd`
(github.com/sparkfun/Motor_Driver-Dual_TB6612FNG, sha256
`0c71c32c7531e2c468b7752a91a1cac0903d8cac6a23562e9ce99de4ec855f6e`; its `LICENSE.md`: hardware under Creative Commons
Share-alike 4.0 International) and `variants/nano_33_iot/variant.h` (github.com/arduino/ArduinoCore-samd, sha256
`1170607f2bcd554c06c5c0305a3a2140652b11057927b186b442f5b9ab4042b3`; file header: GNU Lesser General Public License
2.1 or later).

## Publication

The photo set and the payload are published as described above (the adapted board data under CC BY-SA 4.0 with the
statements above; the cut outlines under CC BY 4.0; the cropped firmware and host script and the unchanged tracking
configuration and calibration files from repositories without a licence file; renders of a mesh from a repository
without a licence file): cropped and de-named, with the attribution of this file repeated in the pull request and on
the card of the asset dataset (places the agent never sees).

## Scene, tools and vendored code

The bench scene is the sibling CAD task's: the lab room `room_lab_v4r3`, built from the project's licensed Unity
Asset Store pack (Archviz Laboratory Pack; items with printed text or of recognisable size removed), an oak bench top
(Poly Haven `oak_veneer_01`, CC0), and scale references generated by code (cutting mat, checker cards, steel rules,
paper tags; the 300 mm rule was added for this task). The grader vendors the benchmark's shared verify kit
`robotics_grader/` (byte-identical to `shared/robotics_grader/`); the stage-lock helpers inside `verify.py` are
copied from the published sister task `bimanual_requirements_replica_from_photos`, whose `environment/` (Dockerfile,
requirements, `fetch_assets.py`, `setup.sh`) this task's image follows with two added instructions. The agent's tools
in the image are open-source Python packages installed from PyPI (numpy, scipy, OpenCV, scikit-image, matplotlib,
Pillow, PyYAML, jsonschema, trimesh, ezdxf).

By the benchmark's rule for this task the givens, the instruction, the image and the public card never name the robot,
its designers, their schools, labs, papers, design names or repositories; bought parts carry their real vendors and
part numbers. This file carries the attribution.
