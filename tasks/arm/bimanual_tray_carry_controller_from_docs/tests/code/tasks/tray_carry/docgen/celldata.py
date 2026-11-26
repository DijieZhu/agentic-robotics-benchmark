import _genenv

import json

FORMAT = "openarm05/CellSheet/1"


def cell_sheet(T):
    n, v = T.number, T.value
    C, L, P = "profile_contract:", "profile_cell:", "profile_stage05:"
    return {
        "format": FORMAT,
        "frame": "the pedestal's: origin on the column axis in the mounting plane, z up, x to the pair's front, y "
                 "towards the left arm; positions in m, angles in deg, quaternions (w, x, y, z)",
        "gravity_m_s2": [0.0, 0.0, -n(L + "gravity")],
        "bench": {"top_z_m": n(C + "BENCH_TOP_Z_M"), "top_tolerance_m": n(C + "BENCH_TOL_M"),
                  "x_m": list(v(L + "bench_outline")["x"]), "half_width_y_m": v(L + "bench_outline")["half_y"]},
        "F": {"what": "the seating fixture: a plate, its top flush with the bench top, with four cups that locate the "
                      "tray's feet",
              "reference_point_drawn_m": list(v(C + "F_REFERENCE_POINT")),
              "frame_drawn": [list(v(L + "F_DRAWN")[0]), list(v(L + "F_DRAWN")[1])],
              "frame_note": "F's frame: on the bench top under the reference point of a tray standing centred in its "
                            "cups, axes as the cell's",
              "survey_tolerance": {"per_axis_m": n(C + "F_SURVEY_TOL.m"), "about_each_axis_deg": n(C + "F_SURVEY_TOL.deg")},
              "cups": {"centres_in_F_m": [list(p) for p in v(L + "FOOT_XY")],
                       "inner_size_m": [2.0 * (n(L + "FOOT_HALF.0") + n(L + "CUP_CLEAR")),
                                        2.0 * (n(L + "FOOT_HALF.1") + n(L + "CUP_CLEAR"))],
                       "wall_height_m": 2.0 * n(L + "CUP_WALL.1"), "wall_thickness_m": 2.0 * n(L + "CUP_WALL.0"),
                       "lead_in": False},
              "presented": "the tray is presented standing in its cups"},
        "S": {"what": "the rack shelf: a plate",
              "x_m": [n(C + "S_REFERENCE_POINT.0") - n(L + "S_HALF.0"), n(C + "S_REFERENCE_POINT.0") + n(L + "S_HALF.0")],
              "half_width_y_m": n(L + "S_HALF.1"), "top_z_drawn_m": n(C + "S_TOP_Z_M"),
              "seat_reference_point_drawn_m": list(v(C + "S_REFERENCE_POINT")),
              "frame_drawn": [list(v(L + "S_DRAWN")[0]), list(v(L + "S_DRAWN")[1])],
              "frame_note": "S's frame: at the centre of S's seat mark on its top face, axes as the cell's",
              "survey_tolerance": {"per_axis_m": n(C + "S_SURVEY_TOL.m"), "about_each_axis_deg": n(C + "S_SURVEY_TOL.deg")}},
        "clear_volume": {"outside_boxes_grown_by_m": n(C + "CLEAR_VOLUME.grow_m"), "above_z_m": n(C + "CLEAR_VOLUME.z_min_m"),
                         "boxes": "the boxes that contain a tray standing in F and a tray standing on S"},
        "ready_pose": {"tool_points_above_grip_points_m": n("profile_vendor:ready_above_m"), "grippers": "open",
                       "of": "a tray standing in F"},
        "trays": {"band_kg_about": list(v(P + "tray_mass_and_centre")["classes_kg"]["band"]),
                  "centre_within_along_handle_line_m": n(C + "CENTRE_REGION.along_m"),
                  "centre_within_fore_aft_m": n(C + "CENTRE_REGION.fore_aft_m"),
                  "centre_below_grip_line_m": list(v(C + "CENTRE_REGION.below_m")),
                  "heavier_trays_up_to_about_kg": v(P + "tray_mass_and_centre")["classes_kg"]["decline"][1],
                  "heavier_trays": "sometimes offered"},
        "ambient_C": list(v(P + "S3_ambient")),
    }


_PUBLIC = []


def public_cell_sheet():
    if not _PUBLIC:
        import tables
        T = tables.build(with_obs=False, extra={"vendor": tables.vendor_facts_table()})
        _PUBLIC.append(dumps(cell_sheet(T)))
    return json.loads(_PUBLIC[0])


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"
