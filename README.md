# The rhombicosidodecahedron is not Rupert — computer-assisted proof

Code and result files accompanying the paper *The rhombicosidodecahedron is not Rupert* (Gihyo Jung), `paper/main.pdf`.

## Requirements
Python >= 3.10 with `numpy`, `scipy` and `matplotlib` (the float oracle uses `matplotlib.path`). Linear programs
(`scipy.optimize.linprog`/HiGHS) only produce *witnesses*; every certificate is re-verified in interval arithmetic, and
all structural identities in exact arithmetic in Q(sqrt5).

## Structure (`code/`)
| file | role (theorem numbers as in the paper) |
|---|---|
| `ia.py` | outward-rounded interval arithmetic; rigorous sin/cos (Section 6.1) |
| `q5.py`, `exact_check.py`, `exact_verify.py` | exact arithmetic in Q(sqrt5); structural identities (Section 6.3) |
| `rid_exact.py`, `exact_pts.py`, `dec_util.py` | decimal data (error < 1e-55) of the vertices, singular views, half-twists |
| `rig_global.py` | Steininger–Yurkevich global theorem (SY25, Thm 17) in interval arithmetic |
| `rig_sylocal.py` | Steininger–Yurkevich local theorem (SY25, Thm 36); congruence by symmetry or exact Gram equality |
| `rig_local2.py` | fixed-direction coincidence theorem (Theorem 4.1) |
| `rig_local5.py` | multi-contact gradient certificate (Theorem 4.2) |
| `rig_sector.py`, `rig_sector_run.py`, `rig_queue5.py` | sector certificates at A, B, C, D (Theorem 4.6, Lemma 4.8) |
| `rig_tube.py`, `rig_tube_run.py` | arc-tube certificates at C (two references) and D (Theorem 4.9) |
| `rig_ball.py` | membership of a box in a certified singular ball |
| `rig_prover.py`, `rig_main.py` | cover of all 211,680 cells |
| `verify_sector_claims.py` | exact verification of the structural claims of the sector certificates |
| `indep_global.py`, `indep_sector.py`, `indep_tube.py` | second implementation in exact rational arithmetic (samples, Section 6.2) |
| `sample_leaves.py`, `indep_leaves.py`, `run_laptop.py` | third implementation: re-checks actual leaves of the cell cover closed by SY-local, Theorem 4.1, Theorem 4.2 and ball membership (Section 6.2) |
| `check_results.py` | checks the result files in `results/` against the expected inventory (cells, 196 sector files with 54 start regions, three tube runs with their box and claim counts); exits with an error if anything is missing. `python check_results.py DIR` checks another directory with the same layout |
| `verify_gwn.py` | exact check of the norm bound used in the sector certificates for all 36,974 contact rows (Appendix C) |
| `verify_decimal_precision.py`, `dump_cert_rows.py` | repeats the decimal preparation with 160 digits and compares the stored data and all derived certificate decisions (Appendix C) |
| `check_decimal_range.py` | checks the magnitude range of all enclosed decimal values (Appendix C) |
| `check_branches_exact.py` | exact check that the second references meet the first ones at A and C (Lemma A.2) |
| `validate_contact.py`, `validate_tube.py`, `sanity_sing.py` | tests of the implementation (not part of the proof) |

## Reproducing (run from inside `code/`)
```
python rig_tube_run.py C ref1 1e-3 0.4 20000 0.5     # about 1 h each
python rig_tube_run.py C ref2 1e-3 0.4 20000 0.5
python rig_tube_run.py D ref1 1e-3 0.4 20000 0.5
python verify_sector_claims.py
python rig_queue5.py ABCD <nproc>                     # sectors, with refinement rounds
python rig_main.py <nproc>                            # all cells
python check_results.py                               # prints ALL OK
```
All programs are resumable and write JSON/JSONL result files to `code/rig/` (a resumed `rig_main.py` re-runs every cell
without a successful record, and writes `MAIN_DONE` only when all 211,680 cells are ok). Note that `check_results.py`
checks the published files in `results/` by default; to check new runs, collect them in the same layout and pass that
directory.

## Results (`results/`)
| part | size | result |
|---|---|---|
| cell cover, all 211,680 cells | 1.23e7 leaves | 0 failures |
| sectors A (two references), B, C, D | 3.61e6 boxes | 0 failures |
| arc tubes C (two references), D | 2 x 6963 + 6662 boxes | 0 failures |
| exact structural claims in Q(sqrt5) (sectors / tubes) | 606 / 3148 | 0 violations |
| second implementation, exact rational (sector / tube boxes, global leaves) | 3500 / 500 / see log | all confirmed |
| third implementation, actual leaves (SY-local / Thm 4.1 / Thm 4.2 / ball) | 630 / 640 / 925 / 378 | all confirmed |

- `cells.jsonl.gz` — one record per cell (`k` = cell index, `ok`, time, numbers of leaves by certificate: `G` global,
  `S` SY-local, `L2` Theorem 4.1, `L5` Theorem 4.2, `B` ball). Cells whose cover uses Theorem 4.1 were re-run after
  adding an explicit margin for the length of floating-point unit vectors (see `results/recheck_L2/`).
- `sectors.zip` — all sector runs (`sectors/final`: files without failures; `sectors/intermediate`: runs whose failed
  boxes were re-run in chunks). Box counts include piece-1 boxes lying outside the unit disc, which are empty.
- `tubes/` — tube results and logs (box and claim counts); `exact/` — structural-claim log;
  `independent/` — logs of the second implementation, with `CODE_MD5.txt` (md5 of the programs used).

All runs use the final versions of the rigorous checks: the tubes were re-run with the published `rig_tube.py`
(separation conditions decided in decimal arithmetic; `tubes/tube_decimal_vs_float_contacts.log` shows that the contact
sets agree with the earlier float decisions), and the 1314 cells whose cover uses Theorem 4.1 were re-run after the
|n| margin was added (`recheck_L2/`, identical leaf counts). During the sector runs only the witness search and the
maximal subdivision depth were extended, which does not affect the verification of a box. The second implementation
(`independent/`) was run with the programs listed in `independent/CODE_MD5.txt`; `CODE_MD5_FINAL.txt` lists the
published programs.

## License
MIT (see `LICENSE`).
