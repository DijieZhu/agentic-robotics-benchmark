# Controller documents

| file | content |
|---|---|
| [controller_interface.md](controller_interface.md) | the motion controller interface specification: the deliverable, the run environment, the calls with their payloads and budgets, the 10 ms cycle, path_start and done, failures and restarts, the acceptance sequence |
| [example_controller.py](example_controller.py) | a minimal example of the interface: it reads the calibration record, converts ticks and design angles and holds the arm still; it is not a motion controller |
| [check_controller.py](check_controller.py) | the interface self-test: runs a controller against a toy plant (not the DA-6) with the payloads, the reply checks and the CPU accounting of the cell computer, and reports protocol errors, exceptions, budget overruns and the fraction of cycles answered in time |

A first run, from the directory that holds `submission/` (`DOCS` stands for the folder of these documents):

```sh
mkdir -p submission/controller
cp DOCS/controller/example_controller.py submission/controller/controller.py
python3 DOCS/controller/check_controller.py submission/controller/controller.py
```

The self-test checks the interface and the timing only; it says nothing about how a controller would do on the DA-6.
