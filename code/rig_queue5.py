"""Queue v5 (resumable): python rig_queue5.py WHICHS NPROC.  Sector jobs for the given singular points, then refinement
rounds that re-run the failed boxes of each result file in chunks (INITFILE), keeping their depth. Every job is skipped
when its output exists, so the queue can be restarted at any time.  Writes rig/QUEUE5_DONE_<WHICHS>."""
import subprocess, os, sys, json, glob
from multiprocessing import Pool
R = os.path.dirname(os.path.abspath(__file__)); RIG = os.path.join(R, 'rig'); TL = float(os.environ.get('QTLIM', '7200'))
def run(job):
    which, ref, mode, piece, face, r0, tl, tag, initfile = job
    out = os.path.join(RIG, f'sec_{which}_{ref}_{mode}_{piece}_{face}{tag}.json')
    if os.path.exists(out) or os.path.exists(os.path.join(RIG, 'src', os.path.basename(out))): return
    env = dict(os.environ); env['TAG'] = tag; env['MAXDP'] = '22'
    if initfile: env['INITFILE'] = initfile
    with open(os.path.join(RIG, f'seclog_{which}.log'), 'a') as lg:
        subprocess.run([sys.executable, 'rig_sector_run.py', which, ref, mode, str(piece), str(face), str(r0), str(tl)], cwd=R, env=env, stdout=lg, stderr=subprocess.STDOUT)
BASE = [('A','ref1','xi',(1,2),1.5e-3), ('A','ref2','xi',(2,),1.5e-3), ('B','ref1','xi',(1,2),1.5e-3), ('C','ref1','s',(1,2),1e-3), ('D','ref1','s',(1,2),1e-3)]
def base_jobs(W):
    return [(w, ref, mode, p, f, r0, TL, '', None) for w, ref, mode, ps, r0 in BASE if w in W for p in ps for f in range(6)]
def round_jobs(W, rnd):
    J = []; os.makedirs(os.path.join(RIG, 'src'), exist_ok=True); os.makedirs(os.path.join(RIG, 'chunks'), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(RIG, 'sec_*.json'))):
        base = os.path.basename(f)[4:-5]; parts = base.split('_')
        if parts[0] not in W: continue
        Rj = json.load(open(f))
        if not Rj['fail']: continue
        which, ref, mode, piece, face = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4][0]); tag0 = parts[4][1:]
        fl = Rj['fail']; n = max(1, min(40, len(fl) // 500 + 1)); r0 = 1.5e-3 if which in 'AB' else 1e-3
        for j in range(n):
            cf = os.path.join(RIG, 'chunks', f'{base}c{j}.json')
            json.dump(fl[j::n], open(cf, 'w'))
            J.append((which, ref, mode, piece, face, r0, TL, f'{tag0}c{j}', cf))
        os.replace(f, os.path.join(RIG, 'src', os.path.basename(f)))
    return J
if __name__ == '__main__':
    W = sys.argv[1]; NP = int(sys.argv[2]); os.makedirs(RIG, exist_ok=True)
    with Pool(NP) as p: p.map(run, base_jobs(W), chunksize=1)
    # pending chunk jobs from an interrupted round
    pend = []
    for cf in sorted(glob.glob(os.path.join(RIG, 'chunks', '*.json'))):
        b = os.path.basename(cf)[:-5]; parts = b.split('_'); which = parts[0]
        if which not in W: continue
        tagc = parts[4][1:]
        pend.append((which, parts[1], parts[2], int(parts[3]), int(parts[4][0]), 1.5e-3 if which in 'AB' else 1e-3, TL, tagc, cf))
    with Pool(NP) as p: p.map(run, pend, chunksize=1)
    for rnd in range(8):
        J = round_jobs(W, rnd)
        if not J: break
        with Pool(NP) as p: p.map(run, J, chunksize=1)
    open(os.path.join(RIG, f'QUEUE5_DONE_{W}'), 'w').write('ok')
    print('ALL DONE', W)
