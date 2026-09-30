"""Independent re-check of Theorem 4.2 (L5) and ball (B) leaves on the laptop.  Run from the code folder:
    python run_laptop.py
Starts 6 processes (3 x L5, 3 x B).  Each re-runs the prover on its cells, records the actual leaves,
and re-verifies them with indep_leaves.py.  Results: ..\\indep_out\\*.log   (look for 'confirmed')."""
import subprocess, sys, os, time
out = os.path.join('..', 'indep_out'); os.makedirs(out, exist_ok=True)
jobs = [('L5', 20, 25, 21, None), ('L5', 20, 25, 22, None), ('L5', 20, 25, 23, None),
        ('B', 1, 300, 31, 0), ('B', 1, 300, 32, 1), ('B', 1, 300, 33, 2)]
procs = []
for typ, nc, pc, seed, idx in jobs:
    name = f'{typ}_{seed}'; js = os.path.join(out, name + '.jsonl')
    cmd = (f'"{sys.executable}" sample_leaves.py {typ} {nc} {pc} {seed} "{js}"' + (f' {idx}' if idx is not None else '') +
           f' > "{os.path.join(out, name + "_sample.log")}" 2>&1 && "{sys.executable}" indep_leaves.py "{js}" > "{os.path.join(out, name + "_verify.log")}" 2>&1')
    procs.append((name, subprocess.Popen(cmd, shell=True)))
    print('started', name, flush=True)
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(60); print(f'{int(time.time() - t0) // 60} min, running: {[n for n, p in procs if p.poll() is None]}', flush=True)
print('\nALL DONE. Summary:')
for name, _ in procs:
    try: print(open(os.path.join(out, name + '_verify.log')).read().strip().splitlines()[-1])
    except Exception as e: print(name, 'no result', e)
input('Press Enter to close')
