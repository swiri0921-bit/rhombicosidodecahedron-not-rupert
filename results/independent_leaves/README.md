# Third implementation: re-check of actual leaves (Section 6.2)

`code/sample_leaves.py TYPE NCELLS PERCELL SEED OUT` re-runs the rigorous prover on randomly chosen cells whose recorded
cover contains leaves of type TYPE (S = SY-local, L2 = Theorem 4.1, L5 = Theorem 4.2, B = ball membership), checks that
the regenerated cover has the recorded leaf counts, and writes a random sample of the leaves with the witness proposed
by the floating-point oracle. `code/indep_leaves.py OUT` re-verifies them (exact rational interval arithmetic,
vertex data rebuilt from the definition, group and congruences proved in Q(sqrt5); no code shared with the main programs).

| type | leaves | cells | result | files |
|---|---|---|---|---|
| S (SY25 Thm 36) | 630 | 59 | 630/630 | `S_leaves.jsonl`, `verify_S.log` |
| L2 (Theorem 4.1) | 640 | 65 | 640/640 | `L2_leaves.jsonl`, `verify_L2_final_*.log` |
| L5 (Theorem 4.2) | 925 | 58 | 925/925 | `laptop/L5_*` |
| B (ball membership) | 378 | 3 | 378/378 | `laptop/B_*` |
| G (Global Theorem, deepest leaves) | 1332 | 40 | 1332/1332 | `verify_G_actual.log`, `verify_G_actual_deep.log` (`code/sample_global_leaves*.py`, checked with `indep_global.py`) |

Notes
- `verify_L2_first_5dirs.log`: a first run with five contact directions per vertex of the outer shadow did not confirm
  two leaves of cell 175746 (the lower bound c0 of Theorem 4.1 depends on the choice of directions). The final version
  uses eleven directions and confirms all 640 leaves.
- The L5 and B runs (`laptop/`) used the version of `indep_leaves.py` before this change; the two versions differ only in
  the choice of contact directions inside `check_L2`.
- The lower bound for c0 in `check_L2` uses float dot products with an explicit rounding margin; everything else is
  exact rational interval arithmetic.
