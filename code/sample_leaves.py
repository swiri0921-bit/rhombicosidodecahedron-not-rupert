"""Re-run the rigorous prover on randomly chosen cells and record the ACTUAL leaves of the cover
(box, certificate type, floating-point witness).  Witnesses are only proposals; they are re-verified by indep_leaves.py.
Usage: python sample_leaves.py TYPE NCELLS PERCELL SEED OUT   (TYPE in S, L2, L5, B)"""
import sys, json, gzip, random, time, numpy as np
sys.path.insert(0, '.')
import rig_prover, rig_sylocal, rig_local5
from rig_main import cellbox
from rig_ball import in_ball

typ, ncell, percell, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
recs = [json.loads(l) for l in gzip.open('../results/cells.jsonl.gz')]
pool = [r for r in recs if r.get(typ)]
if typ == 'B': pool = sorted(pool, key=lambda r: r['s'])[:6]
elif typ == 'L5': pool = [r for r in pool if r['s'] < 90]
rng = random.Random(seed); cells = rng.sample(pool, min(ncell, len(pool)))
if len(sys.argv) > 6: cells = [pool[int(sys.argv[6])]]

last = {}
_orig_or = rig_sylocal.local_oracle
def _or(*a, **k):
    r = _orig_or(*a, **k); last['S'] = r; return r
rig_sylocal.local_oracle = _or
_orig_fc = rig_local5.float_cert
def _fc(*a, **k):
    r = _orig_fc(*a, **k); last['L5'] = r; return r
rig_local5.float_cert = _fc
_orig_cls = rig_prover.classify
found = []
def _cls(box):
    c = _orig_cls(box)
    if c == typ:
        w = None
        if c == 'S':
            p1, p2, p3, q1, q2, q3, sp, sq, r = last['S']
            w = dict(ps=[int(p1), int(p2), int(p3)], qs=[int(q1), int(q2), int(q3)], sp=int(sp), sq=int(sq), r=float(r))
        elif c == 'L5':
            w = [dict(i=int(ct[0]), n=[float(ct[1][0]), float(ct[1][1])], K=[int(k) for k in ct[2]], lam=float(lam))
                 for ct, lam in last['L5']]
        found.append(dict(box=box.tolist(), w=w))
    return c
rig_prover.classify = _cls
def ball(b):
    ok = in_ball(b) is not None
    if ok and typ == 'B': found.append(dict(box=b.tolist(), w=None))
    return ok

f = open(out, 'w'); t0 = time.time()
for r in cells:
    found.clear(); st = {}; fails = []
    ok = rig_prover.prove(cellbox(r['k']), 0, 22, st, time.time() + 3600, fails, ball=ball)
    same = all(st.get(k, 0) == r.get(k, 0) for k in ('G', 'S', 'L2', 'L5', 'B'))
    pick = rng.sample(found, min(percell, len(found)))
    for p in pick: f.write(json.dumps(dict(k=r['k'], typ=typ, **p)) + '\n')
    f.flush()
    print(r['k'], ok, 'same_as_record' if same else f'DIFF {st} vs {r}', len(found), round(time.time() - t0), flush=True)
print('DONE', flush=True)
