# Unit documents

| file | content |
|---|---|
| [calibration_record.md](calibration_record.md) | the DA-6 unit calibration record (format calibration-record/1): how it is taken, its fields, the conversions between ticks and design angles, the EEPROM angle limits, the home pose through the record |
| [example_calibration_record.json](example_calibration_record.json) | the calibration record of the demonstration unit MU-DEMO1 |

The records of the units your controller drives (three units at acceptance, one run on each) are not in this folder:
the cell computer passes the record of the unit at hand to the controller at runtime (`setup`, `unit["calibration"]`;
[../controller/controller_interface.md](../controller/controller_interface.md), section 4). MU-DEMO1 is another unit;
its record shows the format and serves the worked examples.
