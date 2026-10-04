"""Check of the decimal preparation (Appendix C).  Repeats the whole preparation in a temporary copy of this folder
with 160-digit (instead of 60/70-digit) decimal arithmetic and 1e-165 series cut-offs, and compares
 (1) all 232 stored exact-data values (exact_pts.pkl: views, half-twists, h2, vertices, rotation) and
 (2) every decision derived from them in the sector and tube certificates (contact rows, active sets, exact partners,
     separation decisions, D-values, number of exact claims), via dump_cert_rows.py.
Usage: python verify_decimal_precision.py      (about 10 minutes)"""
import os, sys, shutil, tempfile, subprocess, pickle, json
from decimal import Decimal as D, getcontext
HERE = os.path.dirname(os.path.abspath(__file__)); TMP = tempfile.mkdtemp(); HI = os.path.join(TMP, 'code')
shutil.copytree(HERE, HI, ignore=shutil.ignore_patterns('rig', '__pycache__'))
def sub(fn, a, b):
    p = os.path.join(HI, fn); s = open(p).read(); assert a in s, (fn, a); open(p, 'w').write(s.replace(a, b))
sub('rid_exact.py', 'getcontext().prec = 60', 'getcontext().prec = 160')
sub('dec_util.py', 'getcontext().prec = 70', 'getcontext().prec = 170')
sub('dec_util.py', 'D(10) ** -75', 'D(10) ** -165')
subprocess.run([sys.executable, 'exact_pts.py'], cwd=HI, check=True)
getcontext().prec = 200
def flat(x):
    if isinstance(x, dict):
        for k in sorted(x): yield from flat(x[k])
    elif isinstance(x, (list, tuple)):
        for y in x: yield from flat(y)
    else: yield x
A = pickle.load(open(os.path.join(HERE, 'exact_pts.pkl'), 'rb')); B = pickle.load(open(os.path.join(HI, 'exact_pts.pkl'), 'rb'))
n = 0; mx = D(0)
for k in A:
    a, b = list(flat(A[k])), list(flat(B[k])); assert len(a) == len(b); n += len(a)
    mx = max(mx, max(abs(D(x) - D(y)) for x, y in zip(a, b)))
print('stored values', n, 'max |d(60 digits) - d(160 digits)| = %.3e' % mx)
for d, out in ((HERE, os.path.join(TMP, 'rows_std.json')), (HI, os.path.join(TMP, 'rows_hi.json'))):
    subprocess.run([sys.executable, os.path.join(HERE, 'dump_cert_rows.py'), out], cwd=d, check=True)
R1 = json.load(open(os.path.join(TMP, 'rows_std.json'))); R2 = json.load(open(os.path.join(TMP, 'rows_hi.json')))
diff = [k for k in R1 if R1[k] != R2[k]]
for k in R1: print(' ', k, 'identical' if k not in diff else 'DIFFERENT')
ok = mx < D('1e-55') and not diff
print('OK' if ok else 'PROBLEM'); shutil.rmtree(TMP); sys.exit(0 if ok else 1)
