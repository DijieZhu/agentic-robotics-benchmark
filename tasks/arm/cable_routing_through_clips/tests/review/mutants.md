# Control programs on the graded set

Variants of the reference (`solution/oracle/fitter.py`, `variant(name)`; a file `variant` beside
`oracle_main.py` selects one), each run once over the wire through `tests/verify.py` on the fixed hidden set, with
the oracle token (they read the truth as the reference does; what each one leaves out is its point). 14 of the 20
boards have a post. Measured 2026-11-24.

| program | mean_route_credit | complete | connector damaged | prefix credit | how it loses |
|---|---|---|---|---|---|
| reference | 1.00 | 20 | 0 | 1.00 | |
| stops after the first fixture | 0.00 | 0 | 0 | 0.31 | the credit is the whole route's |
| stops after two fixtures | 0.10 | 2 | 0 | 0.63 | the same; on two boards the last fixture was made by the cable as it lay |
| does not watch the tension | 0.20 | 4 | 12 | 0.28 | overloads the connector |
| takes every post on the wrong side | 0.25 | 5 | 0 | 0.66 | no board with a post is complete; five of the six boards without one are (there it is the reference; the sixth differs by the old wire entry, see below) |
| trusts the printed places | 0.20 | 4 | 0 | 0.36 | presses beside the clips, passes posts wrongly; 8 of 20 episodes ended by a hand-over, the rest ran out of steps |

The five controls were run with the first wire entry of the oracle, which advanced the program on whichever request
of a probe pair came first; with it the reference itself completed 19 of 20. The entry now advances the program on
the real observation (carried in the token truth), and the reference takes on every board exactly the steps it
took in process when the set was chosen. The controls were not run again: a difference of that size does not
change what they show.

Removed before the set was drawn: "clip turned by 90 degrees" (tests/review/PREREG.md: it changes nothing the
reference does; a clip's axis is neither scored nor needed).
