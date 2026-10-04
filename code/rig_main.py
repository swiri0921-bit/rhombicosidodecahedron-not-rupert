"""Complete rigorous cell pass (one program): every cell of the 12x7x12x7x30 cover is proved by the rigorous prover
(interval-verified Global / SY-local / local2 / local5 certificates, or membership in a certified singular ball).
Resumable: results in rig/main_<w>.jsonl.  Usage: python rig_main.py NPROC"""
import numpy as np, sys, os, json, time
from multiprocessing import Pool
R = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, R); RIG = os.path.join(R, 'rig')
def cellbox(k):
    from ia import up
    CAP = up(0.6524); TOP = up(2 * np.pi / 5); AH = up(np.pi / 2)
    full = np.array([[0, TOP], [0, CAP], [0, TOP], [0, CAP], [-AH, AH]]); splits = [12, 7, 12, 7, 30]
    edges = [np.array([full[d, 0] + (full[d, 1] - full[d, 0]) * i / splits[d] for i in range(splits[d] + 1)]) for d in range(5)]
    for d in range(5): edges[d][-1] = full[d, 1]; edges[d][0] = full[d, 0]
    idx = np.unravel_index(k, splits); return np.array([[edges[d][idx[d]], edges[d][idx[d] + 1]] for d in range(5)])
def work(args):
    w, W = args
    from rig_prover import prove
    from rig_ball import in_ball
    out = os.path.join(RIG, f'main_{w}.jsonl'); done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                d = json.loads(l)
                if d.get('ok'): done.add(d['k'])          # failed or timed-out cells are re-run on resume
            except Exception: pass
    f = open(out, 'a')
    order = np.random.default_rng(777).permutation(211680)
    for k in order[w::W]:
        k = int(k)
        if k in done: continue
        st = {}; fails = []; t = time.time()
        ok = prove(cellbox(k), 0, 22, st, time.time() + 7200, fails, ball=lambda b: in_ball(b) is not None)
        rec = dict(k=k, ok=ok, s=round(time.time() - t, 2), **st, nf=len(fails))
        if fails: rec['fails'] = fails[:5]
        f.write(json.dumps(rec) + '\n'); f.flush()
if __name__ == '__main__':
    NP = int(sys.argv[1]); OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 0; TOT = int(sys.argv[3]) if len(sys.argv) > 3 else NP
    os.makedirs(RIG, exist_ok=True)     # workers OFF..OFF+NP-1 of TOT (the cell list is shared between machines)
    with Pool(NP) as p: p.map(work, [(w, TOT) for w in range(OFF, OFF + NP)], chunksize=1)
    okk = set()
    for fn in os.listdir(RIG):
        if fn.startswith('main_') and fn.endswith('.jsonl'):
            for l in open(os.path.join(RIG, fn)):
                try:
                    d = json.loads(l)
                    if d.get('ok'): okk.add(d['k'])
                except Exception: pass
    if len(okk) == 211680: open(os.path.join(RIG, 'MAIN_DONE'), 'w').write('ok'); print('MAIN DONE: all 211680 cells ok')
    else: print('NOT DONE:', 211680 - len(okk), 'cells without a successful record (run again to retry them)')
