# Code history and hashes

- `CODE_MD5_FINAL.txt`: md5 of every program **as used for the runs reported in the paper** (cells, sectors, tubes).
- `independent/CODE_MD5.txt`: md5 of all programs at the time the second implementation (Section 6.2: sector, tube and
  global samples) was run. It differs from `CODE_MD5_FINAL.txt` in `rig_local2.py`, `rig_sylocal.py`, `rig_tube.py` and
  `rig_tube_run.py`, which were changed afterwards (the |n| margin in Theorem 4.1, the Gram-matrix congruence test, and
  the decimal separation decisions in the tubes) and then re-run for the final results; see the README. The third
  implementation (`indep_leaves.py`, `results/independent_leaves/`) was run with the final programs.
- `CODE_MD5_V5.txt`: md5 of the programs in the published repository (current state).

## Changes after the runs (2026-10-04, after an external review)

Only two programs listed in `CODE_MD5_FINAL.txt` were changed; neither change affects any certificate:

| file | change |
|---|---|
| `check_results.py` | checks the published files against the expected inventory (cells, 196 sector files with 54 start regions, three tube runs with their box and claim counts) and exits with an error if anything is missing; accepts another results directory as argument |
| `rig_main.py` | on resume, cells without a successful record are re-run (previously every recorded cell was skipped); `MAIN_DONE` is written only when all 211,680 cells are ok. The cell decomposition (`cellbox`) and the prover call are unchanged |

The original versions are those with the hashes in `CODE_MD5_FINAL.txt` (`check_results.py` d3bcb207..., `rig_main.py`
887690ea...); they are identical to the current files except for the lines described above.

New programs (checks only, not used by the certificates): `verify_gwn.py`, `verify_decimal_precision.py`,
`dump_cert_rows.py`, `certify_exact_data.py`, `sample_leaves.py`, `sample_global_leaves.py`, `indep_leaves.py`,
`run_laptop.py`.
