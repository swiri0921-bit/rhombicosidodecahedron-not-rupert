# Independent check of xi-mode sector boxes with tau_l = 0

`code/indep_sector_tau0.py` is `code/indep_sector.py` restricted to piece 2 in xi-mode with tau_l = 0
(boxes tau in [0, 2^-dep]). The main program certifies each random box; the exact-rational second
implementation re-verifies the certificate actually used, including the extra hypothesis of the
sector theorem for tau_l = 0 (alpha_k + r0|d_ik|f^2/2 <= 0 for all active partners, so e_c = 0).

    python indep_sector_tau0.py A ref1 xi 0.0015 2000 51   -> 2000 / 2000 confirmed
    python indep_sector_tau0.py A ref2 xi 0.0015 2000 52   -> 2000 / 2000 confirmed
    python indep_sector_tau0.py B ref1 xi 0.0015 2000 53   -> 2000 / 2000 confirmed

Negative control: passing the reflected cap centre -omega_c to the verifier confirms 0 / 30.
