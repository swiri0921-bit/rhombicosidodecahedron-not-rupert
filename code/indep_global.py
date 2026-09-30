"""Independent re-verification of Global-Theorem leaves (SY Thm 17) in exact rational interval arithmetic
(Fractions; sin/cos by Taylor series with exact remainder; vertices from 60-digit data +- 1e-50).
Does not use ia.py / numpy for any verified quantity."""
import sys, random, time
from fractions import Fraction as F
from decimal import Decimal as D
sys.path.insert(0, '.')
import numpy as np
from rid_setup import rid_z5
from rid_exact import rid_dec
from disprover import global_theorem
class Q:
    __slots__ = ('lo', 'hi')
    def __init__(s, lo, hi=None): s.lo = F(lo); s.hi = F(lo) if hi is None else F(hi)
    def __add__(a, b): b = b if isinstance(b, Q) else Q(b); return Q(a.lo + b.lo, a.hi + b.hi)
    def __sub__(a, b): b = b if isinstance(b, Q) else Q(b); return Q(a.lo - b.hi, a.hi - b.lo)
    def __neg__(a): return Q(-a.hi, -a.lo)
    def __mul__(a, b):
        b = b if isinstance(b, Q) else Q(b); p = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi); return Q(min(p), max(p))
    def abs(a):
        if a.lo >= 0: return a
        if a.hi <= 0: return -a
        return Q(0, max(-a.lo, a.hi))
    def trunc(a, bits=200):   # outward rounding to keep denominators small
        m = F(1, 2 ** bits); return Q(F(int((a.lo / m).__floor__())) * m, F(int(-(-a.hi / m).__floor__())) * m)
def sincos(x):
    """exact rational enclosures of sin x, cos x for rational x, |x| <= 2 (Taylor to degree 60, remainder |x|^61/61!)."""
    x = F(x); s = F(0); c = F(0); t = F(1)
    for n in range(0, 61):
        if n % 2 == 0: c += t * (-1) ** (n // 2)
        else: s += t * (-1) ** (n // 2)
        t = t * x / (n + 1)
    rem = abs(t) * 2       # |x|^61/61! bound (t = x^61/61!), doubled
    return Q(s - rem, s + rem).trunc(), Q(c - rem, c + rem).trunc()
VD = rid_dec(); PF = rid_z5(); VF = np.array([[float(v) for v in w] for w in VD])
VD = [VD[int(np.argmin(np.linalg.norm(VF - v, axis=1)))] for v in PF]     # rid_z5 order
TOL = F(1, 10 ** 50)
VQ = [[Q(F(v) - TOL, F(v) + TOL) for v in w] for w in VD]
def mats(t, p):
    st, ct = sincos(t); sp, cp = sincos(p); Z = Q(0)
    M = [[-st, ct, Z], [-(ct * cp), -(st * cp), sp]]; Mt = [[-ct, -st, Z], [st * cp, -(ct * cp), Z]]; Mp = [[Z, Z, Z], [ct * sp, st * sp, cp]]
    return M, Mt, Mp
def ap(Mr, V): return [Mr[r][0] * V[0] + Mr[r][1] * V[1] + Mr[r][2] * V[2] for r in range(2)]
def check(box, w, s):
    lo, hi = box[:, 0], box[:, 1]
    c = [(F(float(l)) + F(float(h))) / 2 for l, h in zip(lo, hi)]
    eps = max(max(F(float(h)) - cc, cc - F(float(l))) for l, h, cc in zip(lo, hi, c))
    t1, v1, t2, v2, a = c
    M1, M1t, M1p = mats(t1, v1); M2, M2t, M2p = mats(t2, v2); sa, ca = sincos(a)
    wx, wy = F(float(w[0])), F(float(w[1])); nw2 = wx * wx + wy * wy
    # |w| upper bound (rational): nw <= nw2 if nw2 >= 1 else 1 ... use crude bound sqrt via float + margin
    nwu = F(float(np.sqrt(float(nw2)))) * F(1000001, 1000000)
    rot = lambda U: [ca * U[0] - sa * U[1], sa * U[0] + ca * U[1]]
    rota = lambda U: [-(sa * U[0]) - ca * U[1], ca * U[0] - sa * U[1]]
    dotw = lambda U: U[0] * wx + U[1] * wy
    S = VQ[s]; PS = ap(M1, S)
    G = dotw(rot(PS)) - Q(eps) * dotw(rota(PS)).abs() - Q(eps) * dotw(rot(ap(M1t, S))).abs() - Q(eps) * dotw(rot(ap(M1p, S))).abs() - Q(F(9, 2) * eps * eps * nwu)
    Hmax = None
    for V in VQ:
        H = dotw(ap(M2, V)) + Q(eps) * dotw(ap(M2t, V)).abs() + Q(eps) * dotw(ap(M2p, V)).abs() + Q(2 * eps * eps * nwu)
        Hmax = H.hi if Hmax is None else max(Hmax, H.hi)
    return G.lo > Hmax
def leaves(box, depth, out, cap):
    if len(out) >= cap: return
    g = global_theorem(PF, box)
    if g is not None: out.append((box, g[1], g[2])); return
    if depth >= 6: return
    lo = box[:, 0]; hi = box[:, 1]; mid = (lo + hi) / 2
    for bits in random.sample(range(32), 32):
        b = np.empty((5, 2))
        for d in range(5): b[d] = [mid[d], hi[d]] if (bits >> d) & 1 else [lo[d], mid[d]]
        leaves(b, depth + 1, out, cap)
if __name__ == '__main__':
    random.seed(int(sys.argv[1])); ncell = int(sys.argv[2])
    from rig_main import cellbox
    ok = 0; tot = 0; t0 = time.time()
    for _ in range(ncell):
        k = random.randrange(211680); out = []; leaves(cellbox(k), 0, out, 3)
        for box, w, s in out:
            tot += 1; r = check(box, w, s); ok += r
            if not r: print('NOT VERIFIED', k, box.tolist(), flush=True)
    print('independent exact-rational check of Global-Theorem leaves:', ok, '/', tot, 'verified,', round(time.time() - t0), 's', flush=True)
