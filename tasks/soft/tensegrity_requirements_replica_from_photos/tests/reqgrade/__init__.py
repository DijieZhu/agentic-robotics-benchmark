"""Hidden grader modules of tensegrity_requirements_replica_from_photos (static: no submitted code runs).

    schema_check  gates G0 and G2; a byte copy of the public environment/setup/payload/schema_check.py
    gates         runs that copy on the submission snapshot with the schema copies in schemas/
    load          reads the deliverables (through schema_check) and the truth (gt/, the contract copies in
                  schemas/contract/) and the settings of grader_config.json
    bands         band functions and unit conversion
    c1, c2, c3    criteria net_and_geometry_replication, requirements_sheet, derived_quantities
    match, cost, c4, c5   the BOM criteria

No module except schema_check and the public schemas is ever copied into the agent's sandbox.
"""
import sys

sys.dont_write_bytecode = True
