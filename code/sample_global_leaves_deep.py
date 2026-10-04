"""Re-run the prover on cells with deep covers and record ACTUAL Global-Theorem leaves (SY25 Thm 17) with their
witness (direction w, inner vertex s), preferring the deepest (smallest) leaves; they are then re-verified with the
exact rational implementation indep_global.check.  Usage: python sample_global_leaves_deep.py NCELLS PERCELL SEED  (cells with ball or Theorem 4.2 leaves first)"""
import sys, json, gzip, random, time, numpy as np
sys.path.insert(0, '.')
import rig_prover
from rig_main import cellbox
from rig_ball import in_ball
import indep_global
ncell, percell, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
recs = [json.loads(l) for l in gzip.open('../results/cells.jsonl.gz')]
deep = sorted([r for r in recs if (r.get("B") or r.get("L5")) and r["s"] < 120], key=lambda r: (0 if r.get("B") else 1, r["s"]))[:ncell // 2]
rng = random.Random(seed); rest = rng.sample([r for r in recs if r['s'] < 90 and r not in deep], ncell - len(deep))
last = {}; orig = rig_prover.global_theorem
def gt(*a, **k):
    r = orig(*a, **k); last['g'] = r; return r
rig_prover.global_theorem = gt
found = []; ocls = rig_prover.classify
def cls(box):
    c = ocls(box)
    if c == 'G': found.append((box.copy(), last['g'][1], last['g'][2]))
    return c
rig_prover.classify = cls
tot = ok = 0; t0 = time.time(); maxdepth = 0
for r in deep + rest:
    found.clear(); st = {}; fails = []
    rig_prover.prove(cellbox(r['k']), 0, 22, st, time.time() + 3600, fails, ball=lambda b: in_ball(b) is not None)
    same = all(st.get(k, 0) == r.get(k, 0) for k in ('G', 'S', 'L2', 'L5', 'B'))
    found.sort(key=lambda x: (x[0][:, 1] - x[0][:, 0]).max())
    pick = found[:percell] + rng.sample(found[percell:], min(percell // 3, max(0, len(found) - percell)))
    full = (cellbox(r['k'])[:, 1] - cellbox(r['k'])[:, 0]).max()
    for box, w, s in pick:
        d = int(round(np.log2(full / (box[:, 1] - box[:, 0]).max()))); maxdepth = max(maxdepth, d)
        res = indep_global.check(box, w, s); tot += 1; ok += bool(res)
        if not res: print('NOT CONFIRMED', r['k'], box.tolist(), flush=True)
    print(r['k'], 'same_as_record' if same else 'DIFF', 'G leaves', len(found), 'checked', len(pick), 'total', ok, '/', tot,
          'max depth so far', maxdepth, round(time.time() - t0), 's', flush=True)
print(f'actual Global-Theorem leaves: {ok}/{tot} confirmed (deepest relative depth {maxdepth})', flush=True)
