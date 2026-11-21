# Lost-item register: entries, classes of items, catalogue

Before the station was left uncrewed, the last crew entered every item it could not account for in the lost-item
register. Each site has its own register; a sortie's program gets it from `mission()["register"]`. This document
describes the entries and what they are worth, the classes of items with their grasp features, the catalogue
photographs, and shows a sample.

## 1. The register of a site

A register is a list of 9 to 12 entries. An entry is a dict:

| key | content |
|---|---|
| `ref` | the entry's reference, `LI-<nn>`, numbered from `LI-01` through the register; `report` takes it as `item_ref` |
| `class` | the class of the item: `soft stowage sack`, `equipment cover`, `crew towel` or `tool pouch`, or `other` for an item of none of these classes (section 3) |
| `colour` | the colour of the item's body, a plain word such as `orange` |
| `size_m` | `[length, width, thickness]` of the item lying flat, in metres |
| `mass_kg` | its mass in kilograms |
| `features` | what the item can be grasped by: `carry strap + handle loop`, `handle` or `none` |
| `location` | where the crew saw the item last: a module and a place in it (`equipment rack face`, `hatch area`, `near a crew handrail` or `cabin centre`) |

## 2. What the register tells and what it does not

The register is the crew's report from the day it left, not the state of the station now:

- Of the registered items, 3 to 4 are in the 3 modules of the sortie. The others are not there.
- `location` is where an item was seen last. A loose item does not stay where it was seen: it drifts with the cabin
  air until something holds it (`station_handbook.md`).
- An item that blocks a return inlet is in the register: it is a sheet-like item with a carry strap and a handle loop.
  At least 3 entries of a register describe such items, and no 2 of them have the same class and the same colour.
- What lies loose in the modules besides is small: pouches. At most 1 of them is missing from the register.

## 3. The classes of items

| class | kind | length × width (× thickness), lying flat | mass | grasp features | registered colours | a stowage unit takes it on |
|---|---|---|---|---|---|---|
| `soft stowage sack` | sheet-like | 0.29 to 0.30 × 0.27 to 0.29 m, 4 mm thick | 0.10 to 0.13 kg | carry strap + handle loop | orange, beige, grey | the holding screen |
| `equipment cover` | sheet-like | 0.28 to 0.29 × 0.26 to 0.28 m, 4 mm thick | 0.08 to 0.10 kg | carry strap + handle loop | teal, grey, green | the holding screen |
| `crew towel` | sheet-like | 0.29 to 0.30 × 0.28 to 0.30 m, 4 mm thick | 0.12 to 0.14 kg | carry strap + handle loop | purple, yellow, blue | the holding screen |
| `tool pouch` | pouch | 0.18 to 0.22 × 0.11 to 0.14 × 0.02 to 0.03 m | 0.06 to 0.15 kg | handle | blue, green, yellow, grey | a pouch pad |

The size and mass columns give the range of the class; an entry gives the size and the mass of its own item. An item's
body has its registered colour all over. A colour does not always name a class: `grey` is registered for more than one
class, and a register can hold entries of the same colour that differ in class.

### 3.1 Sheet-like items

A sheet-like item is made of 5 stiff panels side by side, joined along their long edges by soft folds: it bends along
its 4 fold lines and stays straight along them. Its `length` (the first number of `size_m`) runs across the folds, its
`width` along them.

2 grasp bars stand on one face of the item: the **carry strap** over the middle of the middle panel, which is the
middle of the item, and the **handle loop** over the middle of an end panel. The bars lie on the item's centre line
across the folds, about 0.12 m apart, each with its axis along that line.

A grasp bar has the section of a crew handrail, 19 × 35 mm, and stands on posts that leave 44 mm between the bar and
the item's face. The bars of a sheet-like item are up to 44 mm long.

### 3.2 Pouches

A pouch is a stiff, flat box. Its **handle** is a grasp bar of the same section on posts of the same height, over the
middle of the pouch's largest face, 70 mm long, with its axis along the pouch's length.

## 4. The catalogue photographs

`~/given/catalogue/` holds 16 studio photographs of articles of the classes, each at the nominal size of its class, in
front of a neutral backdrop, and `index.json`, which lists them. Between them the photographs of a class show every
registered colour of the class and its grasp features:

| class | colour | view | file |
|---|---|---|---|
| `soft stowage sack` | orange | front | `soft_stowage_sack_orange_front.jpg` |
| `soft stowage sack` | beige | oblique | `soft_stowage_sack_beige_oblique.jpg` |
| `soft stowage sack` | grey | grasp bars | `soft_stowage_sack_grey_grasp_bars.jpg` |
| `soft stowage sack` | orange | curled | `soft_stowage_sack_orange_curled.jpg` |
| `equipment cover` | teal | front | `equipment_cover_teal_front.jpg` |
| `equipment cover` | grey | oblique | `equipment_cover_grey_oblique.jpg` |
| `equipment cover` | green | grasp bars | `equipment_cover_green_grasp_bars.jpg` |
| `equipment cover` | teal | curled | `equipment_cover_teal_curled.jpg` |
| `crew towel` | purple | front | `crew_towel_purple_front.jpg` |
| `crew towel` | yellow | oblique | `crew_towel_yellow_oblique.jpg` |
| `crew towel` | blue | grasp bars | `crew_towel_blue_grasp_bars.jpg` |
| `crew towel` | purple | curled | `crew_towel_purple_curled.jpg` |
| `tool pouch` | blue | top | `tool_pouch_blue_top.jpg` |
| `tool pouch` | green | oblique | `tool_pouch_green_oblique.jpg` |
| `tool pouch` | yellow | handle | `tool_pouch_yellow_handle.jpg` |
| `tool pouch` | grey | side | `tool_pouch_grey_side.jpg` |

The views `front` and `oblique` show an article lying flat, `grasp bars` and `handle` its bars from close by, `curled`
a sheet-like item bent along its folds, `top` and `side` a pouch from above and from the side.

## 5. A sample register

A register of the form `mission()["register"]` returns. It belongs to no site. It has 10 entries, 4 of them sheet-like
items with a carry strap:

| `ref` | `class` | `colour` | `size_m` | `mass_kg` | `features` | `location` |
|---|---|---|---|---|---|---|
| `LI-01` | other | blue | [0.12, 0.08, 0.05] | 0.173 | none | H2, cabin centre |
| `LI-02` | other | green | [0.12, 0.08, 0.05] | 0.232 | none | H2, equipment rack face |
| `LI-03` | soft stowage sack | grey | [0.298, 0.281, 0.004] | 0.107 | carry strap + handle loop | H2, cabin centre |
| `LI-04` | tool pouch | grey | [0.206, 0.12, 0.023] | 0.129 | handle | L1, equipment rack face |
| `LI-05` | soft stowage sack | orange | [0.294, 0.278, 0.004] | 0.118 | carry strap + handle loop | L2, near a crew handrail |
| `LI-06` | crew towel | yellow | [0.295, 0.298, 0.004] | 0.127 | carry strap + handle loop | H2, equipment rack face |
| `LI-07` | equipment cover | teal | [0.282, 0.265, 0.004] | 0.099 | carry strap + handle loop | L1, cabin centre |
| `LI-08` | tool pouch | grey | [0.188, 0.13, 0.022] | 0.095 | handle | L2, hatch area |
| `LI-09` | tool pouch | green | [0.21, 0.128, 0.022] | 0.082 | handle | H2, equipment rack face |
| `LI-10` | tool pouch | green | [0.196, 0.121, 0.028] | 0.096 | handle | L1, hatch area |

Its first entry as the call returns it:

    {"ref": "LI-01", "class": "other", "colour": "blue", "size_m": [0.12, 0.08, 0.05], "mass_kg": 0.173, "features": "none", "location": "H2, cabin centre"}
