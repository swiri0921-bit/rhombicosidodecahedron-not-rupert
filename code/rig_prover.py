"""Rigorous recursive prover: a box is closed only by an interval-verified theorem."""
import numpy as np, sys, time
sys.path.insert(0, '.')
from rid_setup import rid_z5
from disprover import global_theorem
import rig_global, rig_sylocal, rig_local2, rig_local5
PF = rid_z5()
def classify(box):
    g = global_theorem(PF, box)
    if g is not None and rig_global.check(box, g[1], g[2]): return 'G'
    try:
        if rig_sylocal.check(box): return 'S'
    except Exception: pass
    hw = (box[:, 1] - box[:, 0]).max() / 2
    if hw < 0.02:
        try:
            if rig_local2.check(box): return 'L2'
        except Exception: pass
    if hw <= 1e-2:
        try:
            if rig_local5.check(box): return 'L5'
        except Exception: pass
    return None
def prove(box, depth, maxdepth, st, deadline, fails, ball=None):
    st['n'] = st.get('n', 0) + 1
    if deadline and st['n'] % 32 == 0 and time.time() > deadline: st['to'] = True; return False
    if ball is not None and ball(box): st['B'] = st.get('B', 0) + 1; return True
    c = classify(box)
    if c: st[c] = st.get(c, 0) + 1; return True
    if depth >= maxdepth: fails.append(box.tolist()); return False
    lo = box[:, 0]; hi = box[:, 1]; mid = (lo + hi) / 2; ok = True
    w = hi - lo; dims = [d for d in range(5) if w[d] >= 0.6 * w.max()]
    import itertools
    for bits in itertools.product([0, 1], repeat=len(dims)):
        b = box.copy()
        for d, bt in zip(dims, bits): b[d] = [mid[d], hi[d]] if bt else [lo[d], mid[d]]
        if not prove(b, depth + 1, maxdepth, st, deadline, fails, ball): ok = False
        if st.get('to'): return False
    return ok
