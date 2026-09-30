# The rhombicosidodecahedron is not Rupert — computer-assisted proof

Code accompanying the paper *The rhombicosidodecahedron is not Rupert*.

## Requirements
Python >= 3.10 with `numpy` and `scipy` (LP weights via `scipy.optimize.linprog`/HiGHS are only *witnesses*;
every inequality is re-verified in interval arithmetic).

## Structure
| file | role |
|---|---|
| `ia.py` | outward-rounded interval arithmetic; rigorous sin/cos (no libm assumption) |
| `q5.py`, `exact_check.py`, `exact_verify.py` | exact arithmetic in Q(sqrt5); structural identities |
| `rid_exact.py`, `exact_pts.py`, `dec_util.py` | 60-digit data of the vertices, singular views, half-twists |
| `rig_global.py` | Steininger–Yurkevich global theorem (Thm 17) in interval arithmetic |
| `rig_sylocal.py` | Steininger–Yurkevich local theorem (Thm 36) |
| `rig_local2.py` | fixed-direction coincidence theorem (Thm 4.2) |
| `rig_local5.py` | multi-contact gradient certificate (Thm 4.3) |
| `rig_sector.py`, `rig_sector_run.py`, `rig_queue5.py` | sector certificates at the singular points A, B, C, D |
| `rig_tube.py`, `rig_tube_run.py` | arc-tube certificates at C (two references) and D |
| `rig_ball.py` | membership of a box in a certified singular ball |
| `rig_prover.py`, `rig_main.py` | rigorous cover of all 211,680 cells |
| `verify_sector_claims.py` | exact verification of all structural claims of the sector certificates |
| `indep_global.py`, `indep_sector.py` | independent re-verification in exact rational arithmetic (samples) |

## Reproducing (run everything from inside `code/`)
```
python rig_tube_run.py C ref1 1e-3 0.4 20000 0.5     # ~1 h each
python rig_tube_run.py C ref2 1e-3 0.4 20000 0.5
python rig_tube_run.py D ref1 1e-3 0.4 20000 0.5
python verify_sector_claims.py
python rig_queue5.py ABCD <nproc>                     # sectors
python rig_main.py <nproc>                            # all cells
```
Every run writes JSON/JSONL result files to `rig/`; a run is successful when it reports 0 failures.
All programs are resumable.
