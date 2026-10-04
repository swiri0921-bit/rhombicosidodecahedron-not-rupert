"""Rigorous certification of the decimal input data (Appendix C).

For every decimal number d that the programs enclose as input data, this script computes an interval [lo, hi] with
rational end points that contains the exact value e (an element of Q(sqrt5) combined with square roots of elements of
Q(sqrt5), see below), and checks max(|d - lo|, |d - hi|) < 1e-55, which implies |d - e| < 1e-55.
The exact values are computed from their definitions only:
  - body vertices V (exact, Q(sqrt5)^3), five-fold axis w5 = (0, phi, 1), |w5| = sqrt(phi^2 + 1);
  - the rotation of Section 2: R = [[1,0,0],[0,c,-s],[0,s,c]], c = 1/|w5|, s = phi/|w5| (it maps w5/|w5| to e3);
  - vertices in the frame: R V / |V|;   singular views: R X / |X| (X in Proposition 3.1), in (sin, cos) form;
  - group elements in the frame: R g R^T with g the exact body symmetry of each vertex permutation;
  - half-twists: Rot(a, +-36 deg) g with a a five-fold axis (R applied to a cyclic permutation of (0, +-phi, +-1),
    normalised), cos 36 = phi/2, sin 36 = sqrt(10 - 2 sqrt5)/4; the stored matrix must match one such product.
Square roots are enclosed with integer square roots; all arithmetic is in rational intervals rounded outward to 2^-400.
Checked data: exact_pts.pkl (views A-D, half-twists T_C, T_D, h2 at C, vertices, rotation) and the 60 group matrices GE
of exact_pts.py (used by rig_local2.py and rig_ball.py).  Usage: python certify_exact_data.py"""
import sys, math, pickle, json, itertools
from fractions import Fraction as F
from decimal import Decimal as D, getcontext
import numpy as np
sys.path.insert(0, '.')
getcontext().prec = 120
BITS = 400; M = 1 << BITS
def rd(x): return F(math.floor(x * M), M)
def ru(x): return F(-math.floor(-x * M), M)
class Iv:
    __slots__ = ('lo', 'hi')
    def __init__(s, lo, hi=None): s.lo = F(lo); s.hi = F(lo) if hi is None else F(hi); assert s.lo <= s.hi
    @staticmethod
    def c(x): return x if isinstance(x, Iv) else Iv(x)
    def __add__(a, b): b = Iv.c(b); return Iv(a.lo + b.lo, a.hi + b.hi)
    __radd__ = __add__
    def __sub__(a, b): b = Iv.c(b); return Iv(a.lo - b.hi, a.hi - b.lo)
    def __rsub__(a, b): return Iv.c(b) - a
    def __neg__(a): return Iv(-a.hi, -a.lo)
    def __mul__(a, b):
        b = Iv.c(b); p = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi); return Iv(rd(min(p)), ru(max(p)))
    __rmul__ = __mul__
    def inv(a):
        assert a.lo > 0 or a.hi < 0; return Iv(rd(1 / a.hi), ru(1 / a.lo))
    def __truediv__(a, b): return a * Iv.c(b).inv()
    def sqrt(a):
        assert a.lo >= 0
        S = 1 << (2 * BITS)
        lo = F(math.isqrt(math.floor(a.lo * S)), 1 << BITS)
        hi = F(math.isqrt(-math.floor(-a.hi * S)) + 1, 1 << BITS)
        return Iv(lo, hi)
    def width(a): return a.hi - a.lo
S5 = Iv(5).sqrt()
from q5 import Q5, PHI
def iv(q): return q.a + q.b * S5 if isinstance(q, Q5) else Iv(q)
from exact_check import V as VB, XB, mat_from_perm
from rid_setup import rid_z5
from local2 import rot_group
PF = rid_z5()
def mm(A, B): return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3)] for i in range(3)]
def mv(A, v): return [A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2] for i in range(3)]
def T(A): return [[A[j][i] for j in range(3)] for i in range(3)]
def norm(v): return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
phi = iv(PHI); na = (phi * phi + 1).sqrt(); c = Iv(1) / na; s = phi / na
R = [[Iv(1), Iv(0), Iv(0)], [Iv(0), c, -s], [Iv(0), s, c]]
worst = [F(0), '']
def check(d, e, what):
    d = F(D(d)) if not isinstance(d, F) else d
    err = max(abs(d - e.lo), abs(d - e.hi))
    if err > worst[0]: worst[0], worst[1] = err, what
    return err < F(1, 10 ** 55)
EX = pickle.load(open('./exact_pts.pkl', 'rb'))
bad = 0; n = 0
# rotation
for i in range(3):
    for j in range(3): n += 1; bad += not check(EX['RM'][i][j], R[i][j], 'RM')
# vertices
for k in range(60):
    w = mv(R, [iv(x) for x in VB[k]]); r = norm([iv(x) for x in VB[k]])
    for t in range(3): n += 1; bad += not check(EX['VZ'][k][t], w[t] / r, f'VZ[{k}]')
# views
for name in 'ABCD':
    x = mv(R, [iv(q) for q in XB[name]]); nx = norm(x); x = [y / nx for y in x]
    if x[2].hi < 0: x = [-y for y in x]
    assert x[2].lo > 0
    sv = (x[0] * x[0] + x[1] * x[1]).sqrt(); vals = dict(st=x[1] / sv, ct=x[0] / sv, sv=sv, cv=x[2])
    for key, e in vals.items(): n += 1; bad += not check(EX[name][key], e, f'view {name}.{key}')
# group elements in the frame (exact body symmetries)
G = rot_group(PF)
GF = []
for g in G:
    perm = [int(np.argmin(np.linalg.norm(PF - g @ x, axis=1))) for x in PF]
    Xb = mat_from_perm(perm)                          # exact, verified on all 60 vertices
    GF.append((perm, mm(mm(R, [[iv(q) for q in row] for row in Xb]), T(R))))
from exact_pts import GE                             # decimal group matrices used by rig_local2 / rig_ball
for (ge, perm) in GE:
    cand = [gf for (p, gf) in GF if p == list(perm)]; assert len(cand) == 1
    for i in range(3):
        for j in range(3): n += 1; bad += not check(ge[i][j], cand[0][i][j], 'GE')
# h2 at C (a group element)
h2 = EX['h2C']; ok = False
for (p, gf) in GF:
    if all(abs(F(D(h2[i][j])) - gf[i][j].lo) < F(1, 10 ** 30) for i in range(3) for j in range(3)):
        for i in range(3):
            for j in range(3): n += 1; bad += not check(h2[i][j], gf[i][j], 'h2C')
        ok = True; break
assert ok, 'h2C does not match a group element'
# half-twists
c36 = phi * F(1, 2); s36 = (Iv(10) - 2 * S5).sqrt() * F(1, 4)
axes = []
for sg in itertools.product([1, -1], repeat=2):
    for v in ([Q5(0), PHI * sg[0], Q5(sg[1])], [PHI * sg[0], Q5(sg[1]), Q5(0)], [Q5(sg[1]), Q5(0), PHI * sg[0]]):
        a = mv(R, [iv(q) for q in v]); a = [y / na for y in a]; axes.append(a)
def rot(a, cs, sn):
    K = [[Iv(0), -a[2], a[1]], [a[2], Iv(0), -a[0]], [-a[1], a[0], Iv(0)]]; KK = mm(K, K)
    return [[(Iv(1) if i == j else Iv(0)) + sn * K[i][j] + (1 - cs) * KK[i][j] for j in range(3)] for i in range(3)]
for name in ('TC', 'TD'):
    Td = EX[name]; Tf = np.array([[float(x) for x in r] for r in Td]); best = None
    for a in axes:
        for sg in (1, -1):
            Ra = rot(a, c36, s36 * sg)
            Raf = np.array([[float(x.lo) for x in r] for r in Ra])
            for (p, gf) in GF:
                gff = np.array([[float(x.lo) for x in r] for r in gf])
                e = np.abs(Raf @ gff - Tf).max()
                if best is None or e < best[0]: best = (e, Ra, gf)
    assert best[0] < 1e-12, (name, best[0])
    Te = mm(best[1], best[2])
    for i in range(3):
        for j in range(3): n += 1; bad += not check(Td[i][j], Te[i][j], name)
print('values checked', n, 'violations', bad)
print('max |d - e| bound = %.3e (at %s)' % (float(worst[0]), worst[1]))
print('OK' if bad == 0 and worst[0] < F(1, 10 ** 55) else 'PROBLEM')
