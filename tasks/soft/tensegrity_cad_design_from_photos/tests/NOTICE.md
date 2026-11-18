# NOTICE (grader side; never staged into the agent's sandbox)

## The robot

The robot of this task is the open-source three-strut, cable-driven tensegrity rolling robot of the Yale Faboratory
(Rebecca Kramer-Bottiglio, Yale University) and the PRACSYS lab (Kostas Bekris, Rutgers University), the fourth
version of their three-bar platform: W. R. Johnson III, P. Meng, N. Chen, L. Cimatti, A. Vercoutere, M. Aanjaneya,
R. Kramer-Bottiglio, K. E. Bekris, "An Open-Source, Reproducible Tensegrity Robot that can Navigate Among
Obstacles", IEEE Robotics and Automation Letters (IEEE Xplore document 11474858; arXiv:2511.05798). Project site:
https://tensegrity.yale.edu/ .

The task grades one strut of that robot. The hidden reference is a subset of the authors' own published file, kept
triangle for triangle: no part of the robot was modelled, redrawn, rebuilt or completed for this task, nothing was
remeshed, reduced, smoothed or filled, and there is no hidden variant. The cables, the strain sensors and the wires
have no geometry in the sources and are in no file and no photograph of the task.

## Geometry sources

sha256 of each file as taken from its source. The licence column states what the source itself says, as recorded on
2026-11-17; nothing here is a legal opinion.

| # | Source | Taken from | File, bytes, sha256 | What the source says about its licence | Use in this task |
|---|---|---|---|---|---|
| 1 | the authors' whole-strut mesh | https://github.com/shiyoung77/tensegrity_perception , branch `ros`, commit `726ed401` | `mesh/Will_strut_scaled_centered.ply`, 21 243 165 bytes, `8dc4caff1efa26468660f30d5e2d91f7d927d6b287480ecd3b06b351f88a4614` (1 118 331 triangles, metres, 494 edge-connected pieces) | the repository has no licence file; the `package.xml` of its `ros` branch says `<license>MIT</license>` | the 8 reference designs (`reference/parts/`, `solution/reference/parts/`); the rod's STEP (row 3); the 287 photographs, which are renders of it; every measured table of `reference/` |
| 2 | the gearmotor's vendor model | Pololu, "Micro Metal Gearmotor 3D (STEP) Models", https://www.pololu.com/file/0J2042/ (zip: 59 981 034 bytes, `eeafd5682d7a2c7deed7c7c7e24719eeb36761706aca4a1c5c815c07c36cbb55`) | `With Encoder, Back Connector/mmgm-cb-enc-back-con.step`, 14 142 476 bytes, `d397d4608260a8866cf02c3b5f5bf0ec090256e23446920bcd3f8d6fc4bf7dfa` | the zip's `README.txt` is a key to the file names and carries no licence text; the vendor's site terms apply | given to the agent as `purchased/POLOLU-5192/POLOLU-5192.step` (18 773 944 bytes, `db5532e79fb01e7838d55692b3fc35844e406970fc0c6f87ac53e1f8d23234a5`): the vendor's solid moved rigidly into the part's own frame (origin on the output-shaft axis in the gearbox front face, +z along the shaft) and written again with Open CASCADE under a neutral header; nothing of its shape changed. Pinned in `reference/purchased_pins.json`; graded by where it is placed |
| 3 | the rod | piece `c022` of the mesh of row 1 (144 triangles; the piece file's sha256 `f4c3d4ad1288e2ffa4115b0429d87fef4c42cf14e105d36af97789f95dbef049`) | given to the agent as `purchased/MCMASTER-8974K19/MCMASTER-8974K19.step`, 313 079 bytes, `44dca7360bc1bc8fb3b4fb5aa8fb05abc9cf4b2f8e1af3619d78621a2c832853` | as row 1 | the piece converted triangle for triangle into a faceted solid (one planar face per triangle: a 36-sided prism, 304.80 mm long); no analytic cylinder was written and no vendor file is used. The part number (McMaster-Carr 8974K19, 1/8 in x 12 in aluminium rod) is the one of the authors' bill of materials; its mass is the solid's volume at 2.70 g/cm3 |
| 4 | the gearmotor's data | Pololu, https://www.pololu.com/product/5192/specs (page captured 2026-11-17 04:31:32 UTC, `0c69a5222ad984038ab422231252172a07843ab949d944e2fa7c825dfd0a955b`) | - | a vendor's public product page | the product name and the mass (11 g) on the data sheet page and in `reference/purchased.json` |
| 5 | the authors' bill of materials and assembly guide (version 3) | https://tensegrity.yale.edu/ | `BOM.xlsx`, 27 220 bytes, `3dea5cc931d8758edf7f7f91b350ff498b703577aa277af90cf918284448c613`; `Guide_to_assembly_a_tensegrity_robot_V3.pdf`, 2 214 212 bytes, `912c44ccae2bd04e7d734ca0d4771612b2ed40e37c9bd8a30f138cbe300de49e` | neither the site nor the files carry a licence or copyright statement | hidden side only: which pieces of the mesh are made parts, purchased components or out-of-scope hardware, the vendors and part numbers of the purchased components, the materials of the made parts. No text or figure of either is in the task |
| 6 | the authors' CAD package ("tribar-2" on GrabCAD), as downloaded by the benchmark's maintainers | https://grabcad.com/library/tribar-2 | `tribar-2.snapshot.2.zip`, 525 380 949 bytes, `ddc207b6f1674d286a6f916a5d4663e62b7db18698f1747b06cd903fd2d407e8` (package not redistributed) | GrabCAD's site terms: a user's own, non-commercial, internal use; no redistribution to third parties | **no file of the package is in this task** (payload, photographs, `tests/`, `solution/`). It served as a cross-check only: its six final print STL files are the same geometry as the corresponding pieces of the mesh of row 1 (five triangle for triangle, vertices within 2.4e-5 mm; the electronics cover the same design with one fillet faceted differently, at most 0.011 mm apart); its whole-strut STL is the mesh of row 1 (a translation and a change of unit, at most 2.19e-5 mm apart); its copy of the gearmotor STEP has the sha256 of row 2. `public_blocklist.json` lists the sha256 and names of its 22 STL files |
| 7 | print files of an earlier version of the platform | figshare dataset "Impact-resistant, autonomous robots inspired by tensegrity architecture", DOI 10.6084/m9.figshare.32273583 | `3D printing.zip`, 402 907 bytes, `ad600d6f7f421dde481bf3ccc91ee13c9499a18cdc19b76ad756c1644d8ce4f1` | CC BY 4.0 (the dataset's licence field) | only the sha256 and names of its six STL files, in `public_blocklist.json` |

## The reference, file by file

The mesh of row 1 was converted from metres to millimetres, expressed in the strut frame (rod axis = +z, rod midpoint
= origin) and split into its 494 edge-connected pieces without any repair (no hole was filled).
Thirteen pieces are the made parts of one strut; they are 8 designs, five of them present twice as rigid copies
(residual at most 3.7e-5 mm, no mirror image). Each reference file is a byte copy of one piece's file:

| File (`reference/parts/`) | Tags | Mesh piece | Triangles | Bytes | sha256 |
|---|---|---|---:|---:|---|
| `F-EA.ply` | P01 | c001 | 3 428 | 65 229 | `86141554087331400d6db16ff67d4fabe488842e1e498f13324ae6cd8a607cf0` |
| `F-EB.ply` | P02 | c000 | 1 706 | 32 486 | `d2f42be190b98721c39f5c3c41f31d885f7501331f3cf5cf3094ec650ae1f40d` |
| `F-PB.ply` | P03 | c004 | 788 | 15 067 | `6b0cec7d82f98aa36e82de49efc2105e8a99b110bf9e63448e8ae48df62efbdc` |
| `F-MA.ply` | P04, P05 | c002 | 2 070 | 39 403 | `a61ff89372e543807dd6a1e1ed6a5e1e95c33ec4cb462c389018848dab470abc` |
| `F-MB.ply` | P06, P07 | c014 | 1 422 | 27 186 | `0ec26b554e83d3ed6359fef178308e2587430d80e63230d341340220972aec41` |
| `F-WI.ply` | P08, P09 | c032 | 6 282 | 119 527 | `8b3449d597088c8f69e8450f503703903485270b63cf56995339ec0399ef4543` |
| `F-EC.ply` | P10, P11 | c006 | 6 340 | 120 701 | `23e27e8a640f530abd928e93832eddf9b8c9f0bf48d9f4da4a7669763d4be099` |
| `F-CO.ply` | P12, P13 | c007 | 6 568 | 125 033 | `84c49339b739378b65df09fea1210a455f7a7cb5f7b89cdd365469959798d8b5` |

Checked when the files were taken: every triangle of the 13 instances (51 286) matches a triangle of the source one
to one, the largest coordinate difference being 7.6e-6 mm (single-precision storage); every design is a closed
manifold as published. `solution/reference/parts/` holds the same 8 files, byte for byte; the oracle turns each into
a faceted solid, one planar face per triangle.

The other files of `reference/` are measurements on these meshes and nothing else: `families.json` (poses, symmetry
tests, sizes), `features.json` (hole steps, flat faces, functional primitives and values, contacts and overlaps of
the original assembly), `seen.json` and its bitmaps (which reference samples, hole mouths and flat faces the
photographs show). `purchased.json` holds the poses of the purchased components in the original: the vendor's
gearmotor solid (row 2) registered onto the mesh's two gearmotor pieces (c013, c011; 99th-percentile surface distance
0.015 mm on both, which also shows that the mesh's gearmotor is that vendor model), and the rod by construction.

## What the agent is given

* **287 photographs** (`photos/photo_001.jpg` to `photo_287.jpg`; 2016 x 1512 JPEG, 126 379 043 bytes together;
  list hash in `README.md` section 6). Renders (Blender 5.2.2, Cycles) of the mesh of row 1: its 494 pieces grouped
  and posed as rigid bodies only (the assembled strut, eight stages of taking it apart, every made part laid out on a
  cutting mat); no triangle is changed and no cable, wire or sensor is drawn. The files carry a plain JFIF header and
  no other metadata.
* **`purchased/POLOLU-5192/`** and **`purchased/MCMASTER-8974K19/`**: the STEP files of rows 2 and 3, and for each a
  data sheet page (`*_datasheet.md`) and a dimensioned drawing (`*_drawing.png`) written and drawn for this task. They
  carry the vendor's name, part number and product name, the gearmotor's mass as the vendor states it, and
  dimensions measured on the given STEP file; no drawing or image of a vendor is copied.

By design, the instruction, the payload, the image and the public card name neither the robot nor its
designers, their schools, labs, papers, design names or repositories; the purchased components carry their real
vendors and part numbers. This file carries the attribution.

## The scene of the photographs

Not graded. The room is the benchmark's lab room bundle `room_lab_v4r3`, built from the Archviz Laboratory
Pack 2.0 (Unity Asset Store), with the items that carry printed text or have a recognisable size (glassware,
instruments) left out. The bench top is Poly Haven's `oak_veneer_01` (CC0), 2k maps: `oak_veneer_01_diff_2k.jpg`
`0718c2e1a209afe6138ccb41d70e1a18cd825d1e73db33ad7cf80806fa9ce5a4`, `oak_veneer_01_arm_2k.jpg`
`c1a3453f517fe566ca01f431c72b0ab29120ffce33922f88f3b1100a720b17ce`, `oak_veneer_01_nor_gl_2k.jpg`
`e74a9d3ddcd86473a21815a28464f4dac4ceb8f86e7f170226dede8ed70d8eb9`. The scale references (cutting mat, checker cards,
a standing card with its stop, a 150 mm steel rule, paper tags) are generated by code.

## Vendored and adapted code

* `environment/vendor/freecad-mcp/`: FreeCAD MCP, https://github.com/neka-nat/freecad-mcp , commit
  `78bf0230a03324b56c9a0b7f55150fb5be2a0d42`, MIT (its `LICENSE` is kept); copied unmodified, without its assets,
  documents, examples, scripts and tests (`VENDORED.md` in that folder).
* `tests/robotics_grader/`: the benchmark's shared verify kit, byte-identical to `shared/robotics_grader/`.
* `tests/arm/` (`verify_freeze.py`, `assembly_six/lockdown.py`, `assembly_cell/runner.py`): the benchmark's
  verify-start lockdown helpers, byte-identical to the copies in `uav_cad_design_from_photos`.
* `tests/lockdown_runtime.py`, `tests/verify.py`, `tests/fcstd_extract.py`, `tests/strutgrade/recognise.py` and
  `solution/build_oracle.py` are adapted from `uav_cad_design_from_photos` (and through it from
  `robotic_arm_cad_design_from_photos` and `bimanual_cad_design_from_photos`); `tests/test.sh` is that task's file,
  byte for byte. The rest of `tests/strutgrade/` and `tests/grade.py` were written for this task.
* `environment/`: the Dockerfile has the instructions of `uav_cad_design_from_photos` (comments changed);
  `requirements.txt`, `fetch_assets.py` and `setup/setup.sh` are that task's files, byte for byte. The image installs
  FreeCAD 1.1.4 from the project's release AppImage, pinned by sha256 in the Dockerfile, and the Python packages of
  `requirements.txt` from PyPI (among them numpy, scipy, trimesh, manifold3d and cadquery-ocp, which the grader
  uses).

## Distribution

The repository holds the 8 reference designs (pieces of the mesh of row 1) under `tests/reference/parts/` and
`solution/reference/parts/` and the tables measured on them. The benchmark's asset dataset holds the photographs
(renders of that mesh), the rod's STEP (a piece of that mesh) and the gearmotor STEP (the vendor's model, written
again), fetched at the revision pinned in `environment/assets.sha256`.
