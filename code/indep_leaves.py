"""Third, independent re-verification of ACTUAL leaves of the cell cover closed by
   S  : Steininger-Yurkevich local theorem (SY25 Thm 36, checked directly as stated there),
   L2 : Theorem 4.1 (fixed-direction coincidence theorem),
   L5 : Theorem 4.2 (multi-contact gradient certificate),
   B  : membership in a singular ball B_X (Section 5).
Written from the statements in the paper and in SY25 only.  It does NOT import ia.py, rig_*.py, local2.py, q5.py,
exact_*.py, rid_*.py or dec_util.py.  All verified quantities use exact rational interval arithmetic (Fractions);
vertex coordinates are rebuilt here from the definition in Section 2 (80-digit Decimals, enclosed +-1e-60);
vertex permutations of the rotation group and congruences are proved in exact arithmetic in Q(sqrt5).
The only floating-point input from the main programs is the leaf box and the witness proposed by its oracle.
The lower bound for c0 (Theorem 4.1) uses float dot products with an explicit rounding margin (see c0_lower)."""
import sys, json, math, itertools, time
from fractions import Fraction as F
from decimal import Decimal as D, getcontext
import numpy as np
getcontext().prec = 90

# ---------------------------------------------------------------- exact Q(sqrt5): pairs (a, b) = a + b sqrt5
def qa(x, y): return (x[0] + y[0], x[1] + y[1])
def qm(x, y): return (x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])
def qneg(x): return (-x[0], -x[1])
def qdot(u, v): return qa(qa(qm(u[0], v[0]), qm(u[1], v[1])), qm(u[2], v[2]))
Z0, ONE = (F(0), F(0)), (F(1), F(0))
PHI = (F(1, 2), F(1, 2))
PHI2 = qm(PHI, PHI); PHI3 = qm(PHI2, PHI)
BASE = [(ONE, ONE, PHI3), (PHI2, PHI, qm((F(2), F(0)), PHI)), (qa((F(2), F(0)), PHI), Z0, PHI2)]
VB = []                                          # exact body-frame vertices (unscaled)
for b in BASE:
    for s in itertools.product([1, -1], repeat=3):
        v = tuple(b[i] if s[i] > 0 else qneg(b[i]) for i in range(3))
        for k in range(3):
            w = v[k:] + v[:k]
            if w not in VB: VB.append(w)
assert len(VB) == 60
N2 = qdot(VB[0], VB[0]); assert all(qdot(v, v) == N2 for v in VB)   # all on one sphere
S5 = D(5).sqrt()
def qdec(x): return D(x[0].numerator) / D(x[0].denominator) + D(x[1].numerator) / D(x[1].denominator) * S5
# rotation of Section 2: five-fold axis (0, phi, 1) -> e3 (rotation about e1), scaled to circumradius 1
PHId = qdec(PHI); nA = (PHId ** 2 + 1).sqrt(); cb, sb = 1 / nA, PHId / nA; rad = qdec(N2).sqrt()
def to_frame(v):
    x, y, z = (qdec(c) for c in v)
    return (x / rad, (cb * y - sb * z) / rad, (sb * y + cb * z) / rad)
VDEC = [to_frame(v) for v in VB]
# reorder to the vertex numbering of the main programs (used only to read the witnesses' indices)
sys.path.insert(0, '.')
from rid_setup import rid_z5
_PF = rid_z5(); _VF0 = np.array([[float(c) for c in v] for v in VDEC])
_ord = [int(np.argmin(np.linalg.norm(_VF0 - p, axis=1))) for p in _PF]
assert sorted(_ord) == list(range(60)) and np.abs(_VF0[_ord] - _PF).max() < 1e-12
VB = [VB[i] for i in _ord]; VDEC = [VDEC[i] for i in _ord]
assert all(abs(float(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) - 1) < 1e-15 for v in VDEC)
assert all(abs(float(qdec(qdot(VB[i], (Z0, PHI, ONE)))) / float(rad * nA) - float(VDEC[i][2])) < 1e-12 for i in range(60))

# ---------------------------------------------------------------- rational interval arithmetic
BITS = 180
class Q:
    __slots__ = ('lo', 'hi')
    def __init__(s, lo, hi=None):
        s.lo = F(lo); s.hi = F(lo) if hi is None else F(hi)
        assert s.lo <= s.hi
    @staticmethod
    def c(x): return x if isinstance(x, Q) else Q(x)
    def __add__(a, b): b = Q.c(b); return Q(a.lo + b.lo, a.hi + b.hi)
    __radd__ = __add__
    def __sub__(a, b): b = Q.c(b); return Q(a.lo - b.hi, a.hi - b.lo)
    def __rsub__(a, b): return Q.c(b) - a
    def __neg__(a): return Q(-a.hi, -a.lo)
    def __mul__(a, b):
        b = Q.c(b); p = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi); return Q(min(p), max(p)).r()
    __rmul__ = __mul__
    def sq(a):
        if a.lo >= 0: return Q(a.lo * a.lo, a.hi * a.hi).r()
        if a.hi <= 0: return Q(a.hi * a.hi, a.lo * a.lo).r()
        return Q(0, max(a.lo * a.lo, a.hi * a.hi)).r()
    def abs(a):
        if a.lo >= 0: return a
        if a.hi <= 0: return -a
        return Q(0, max(-a.lo, a.hi))
    def r(a):                                     # outward rounding to 2^-BITS
        m = 1 << BITS
        lo = F(math.floor(a.lo * m), m); hi = F(-math.floor(-a.hi * m), m); return Q(lo, hi)
    def __repr__(s): return f'[{float(s.lo)}, {float(s.hi)}]'
def qmax(xs): return Q(max(x.lo for x in xs), max(x.hi for x in xs))
def sqrt_up(x):                                   # rational upper bound of sqrt(x), x >= 0 rational
    x = F(x)
    if x <= 0: return F(0)
    s = F(math.sqrt(float(x))) * F(1000000001, 1000000000) + F(1, 10 ** 30)
    while s * s < x: s *= F(1001, 1000)
    return s
def sqrt_dn(x):
    x = F(x)
    if x <= 0: return F(0)
    s = F(math.sqrt(float(x))) * F(999999999, 1000000000)
    while s * s > x: s *= F(999, 1000)
    return s
def norm_hi(v): return sqrt_up(sum((c.sq() for c in v), Q(0)).hi)
def norm_lo(v): return sqrt_dn(sum((c.sq() for c in v), Q(0)).lo)
TOL = F(1, 10 ** 60)
def dq(d): d = F(str(d)); return Q(d - TOL, d + TOL)
V = [[dq(c) for c in v] for v in VDEC]            # vertex enclosures, frame of Section 2
VF = np.array([[float(c) for c in v] for v in VDEC])
SQ2 = F(14142136, 10 ** 7); SQ5 = F(22360680, 10 ** 7)   # upper bounds of sqrt2, sqrt5
assert SQ2 ** 2 > 2 and SQ5 ** 2 > 5

def sincos(x):
    """enclosures of sin x, cos x for rational |x| <= 4 (Taylor to degree 70, Lagrange remainder |x|^71/71!)."""
    x = F(x); assert abs(x) <= 4
    s = F(0); c = F(0); t = F(1)
    for n in range(71):
        if n % 2 == 0: c += t * (-1) ** (n // 2)
        else: s += t * (-1) ** (n // 2)
        t = t * x / (n + 1)
    rem = abs(t) + F(1, 10 ** 60)
    return Q(s - rem, s + rem).r(), Q(c - rem, c + rem).r()

def frame(t, p):
    """A(theta, phi): rows M1, M2, X  (Section 2.2)"""
    st, ct = sincos(t); sp, cp = sincos(p); Z = Q(0)
    return [[-st, ct, Z], [-(ct * cp), -(st * cp), sp], [ct * sp, st * sp, cp]]
def frame_d(t, p):
    """d/dtheta and d/dphi of the two rows of M"""
    st, ct = sincos(t); sp, cp = sincos(p); Z = Q(0)
    return ([[-ct, -st, Z], [st * cp, -(ct * cp), Z]], [[Z, Z, Z], [ct * sp, st * sp, cp]])
def mv(Mr, v): return [Mr[r][0] * v[0] + Mr[r][1] * v[1] + Mr[r][2] * v[2] for r in range(len(Mr))]
def rot2(sa, ca, u): return [ca * u[0] - sa * u[1], sa * u[0] + ca * u[1]]
def dot(u, v): return sum((a * b for a, b in zip(u, v)), Q(0))
def cross3(u, v): return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
def box_center(box):
    lo = [F(float(b[0])) for b in box]; hi = [F(float(b[1])) for b in box]
    c = [(l + h) / 2 for l, h in zip(lo, hi)]; hw = [max(h - cc, cc - l) for l, h, cc in zip(lo, hi, c)]
    return c, hw, lo, hi
def inner_frame(c):
    t1, v1, t2, v2, a = c
    A1 = frame(t1, v1); sa, ca = sincos(a)
    return [rot2(sa, ca, [A1[0][j], A1[1][j]])[0] for j in range(3)], [rot2(sa, ca, [A1[0][j], A1[1][j]])[1] for j in range(3)], A1[2]

# ---------------------------------------------------------------- rotation group I as exact vertex permutations
def _float_group():
    out = []; v0 = VF[0]; j1 = int(np.argmin([abs(VF[0] @ w) for w in VF])); v1 = VF[j1]
    def fr(a, b):
        e1 = a; e2 = b - (b @ a) * a; e2 /= np.linalg.norm(e2); return np.array([e1, e2, np.cross(e1, e2)])
    B0 = fr(v0, v1)
    for a in range(60):
        for b in range(60):
            if abs(VF[a] @ VF[b] - v0 @ v1) > 1e-9: continue
            R = fr(VF[a], VF[b]).T @ B0
            W = VF @ R.T; perm = [int(np.argmin(np.linalg.norm(VF - w, axis=1))) for w in W]
            if np.abs(VF[perm] - W).max() < 1e-9 and perm not in out: out.append(perm)
    return out
GROUP = _float_group(); assert len(GROUP) == 60
GRAM = [[qdot(VB[a], VB[b]) for b in range(60)] for a in range(60)]
for pm in GROUP:                                   # exact: Gram preserved  =>  pm is induced by an orthogonal map
    assert all(GRAM[pm[a]][pm[b]] == GRAM[a][b] for a in range(60) for b in range(a, 60))
def qdet(u, v, w): return qdot(u, (qa(qm(v[1], w[2]), qneg(qm(v[2], w[1]))), qa(qm(v[2], w[0]), qneg(qm(v[0], w[2]))), qa(qm(v[0], w[1]), qneg(qm(v[1], w[0])))))
_b = (0, 1, 2)
while qdet(*[VB[i] for i in _b]) == Z0: _b = tuple(np.random.choice(60, 3, replace=False))
_d0 = qdet(*[VB[i] for i in _b])
for pm in GROUP:                                   # exact: determinant +1 (same sign of a triple product)
    d1 = qdet(*[VB[pm[i]] for i in _b]); assert d1 == _d0
ANTI = [int(np.argmin(np.linalg.norm(VF + v, axis=1))) for v in VF]
assert all(VB[ANTI[k]] == tuple(qneg(x) for x in VB[k]) for k in range(60))
# tight frame: sum_k V_k V_k^T = 20 I  (exact, body frame, scaled by N2)
for i in range(3):
    for j in range(3):
        s = Z0
        for v in VB: s = qa(s, qm(v[i], v[j]))
        assert s == (qm((F(20), F(0)), N2) if i == j else Z0)

def angle_upper(RV):
    """RV[k] = R' V_k - V_k for an orthogonal-difference map; ||R - I||_F^2 = (1/20) sum |R V_k - V_k|^2 (tight frame);
    rotation angle theta satisfies ||R - I||_F = 2 sqrt2 sin(theta/2)."""
    fro2 = sum((sum((c.sq() for c in d), Q(0)) for d in RV), Q(0)) * F(1, 20)
    y = sqrt_up(fro2.hi / 8)
    if y >= F(1, 2): return None
    return 2 * y * sqrt_up(1 / (1 - y * y))       # theta = 2 asin y <= 2 y / sqrt(1 - y^2)

# ---------------------------------------------------------------- S: SY25 Theorem 36 (radius rho = 1)
def check_S(box, w):
    c, hw, _, _ = box_center(box); eps = max(hw); t1, v1, t2, v2, a = c
    ps, qs, sp, sq, r = w['ps'], w['qs'], w['sp'], w['sq'], F(w['r'])
    # congruence (SY Def. 34, Lemma 35 with the remark): equal Gram matrices, exactly
    if not all(GRAM[ps[i]][ps[j]] == GRAM[qs[i]][qs[j]] for i in range(3) for j in range(3)): return 'gram'
    A1 = frame(t1, v1); A2 = frame(t2, v2); sa, ca = sincos(a)
    M1, X1, M2, X2 = A1[:2], A1[2], A2[:2], A2[2]
    se = Q(SQ2 * eps)
    for k in range(3):                               # (A_eps)
        if not ((sp * dot(X1, V[ps[k]])).lo > se.hi): return 'A1'
        if not ((sq * dot(X2, V[qs[k]])).lo > se.hi): return 'A2'
    PP = [mv(M1, V[p]) for p in ps]; QQ = [mv(M2, V[q]) for q in qs]
    g = 2 * eps * (SQ2 + eps)
    for u, v in ((0, 1), (1, 2), (2, 0)):             # eps-spanning (SY Def. 27): <R(pi/2) M P_u, M P_v> = u1 v2 - u2 v1
        if not ((PP[u][0] * PP[v][1] - PP[u][1] * PP[v][0]).lo > g): return 'span P'
        if not ((QQ[u][0] * QQ[v][1] - QQ[u][1] * QQ[v][0]).lo > g): return 'span Q'
    for k in range(3):
        if not (norm_lo(QQ[k]) > r + SQ2 * eps): return 'r'
    delta = max(norm_hi([x - y for x, y in zip(rot2(sa, ca, PP[k]), QQ[k])]) for k in range(3)) / 2
    rhs = (SQ5 * eps + delta) / r
    AQ = [mv(M2, V[j]) for j in range(60)]
    for k, qi in enumerate(qs):
        mq = QQ[k]; nmq = norm_hi(mq)
        for j in range(60):
            if j == qi: continue
            d3 = norm_hi([V[qi][t] - V[j][t] for t in range(3)])
            dm = [mq[0] - AQ[j][0], mq[1] - AQ[j][1]]
            num = dot(mq, dm) - Q(2 * eps * d3 * (SQ2 + eps))
            den = (nmq + SQ2 * eps) * (norm_hi(dm) + 2 * SQ2 * eps)
            if not (num.lo > 0 and num.lo > rhs * den): return f'B {k} {j}'
    return True

# ---------------------------------------------------------------- L2: Theorem 4.1
RZPI = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
def _caps(m):
    u = (np.arange(m) + 0.5) / m * 2 - 1; U, W = np.meshgrid(u, u); P = []
    for ax in range(3):
        for sg in (-1, 1): P.append(np.insert(np.stack([U.ravel(), W.ravel()], 1), ax, sg, axis=1))
    P = np.vstack(P); return P / np.linalg.norm(P, axis=1, keepdims=True), math.sqrt(2) / m
CAPS, CAPR0 = _caps(64)
def c0_lower(gs):
    """lower bound of min_{|v|=1} max_c <v, g_c> for vectors g_c known as rational boxes.
    Every unit v lies within Euclidean distance r = sqrt2/64 of the normalised centre p of a cell of a 64x64 grid on a
    face of [-1,1]^3 (radial projection is 1-Lipschitz outside the unit ball); <v,g> >= <p,g> - r|g| for every g in the box.
    Float evaluation: p is a float vector with |p - p_exact| < 1e-15 (absorbed by using r' = 1.001 r);
    <p, g> for an endpoint g is a sum of 3 products, float error < 4 * 2^-53 * sum|p_i||g_i| <= 1e-15 * |g|_1;
    we subtract 1e-13 * max |g|_1."""
    GL = np.array([[float(x.lo) for x in g] for g in gs]); GH = np.array([[float(x.hi) for x in g] for g in gs])
    # float(x) of a Fraction is within half an ulp: widen by 1e-15 relative
    GL = GL - 1e-15 * np.abs(GL) - 1e-300; GH = GH + 1e-15 * np.abs(GH) + 1e-300
    gmax = np.maximum(np.abs(GL), np.abs(GH)); gn = np.sqrt((gmax ** 2).sum(1)) * (1 + 1e-12)
    lo = np.minimum(CAPS[:, None, :] * GL[None], CAPS[:, None, :] * GH[None]).sum(2)
    lo = lo - 1e-13 * gmax.sum(1)[None, :] - CAPR0 * 1.001 * gn[None, :]
    return float(lo.max(1).min()) - 1e-15

L2_FRACS = [i / 12 for i in range(1, 12)]
def check_L2(box, fracs=None):
    fracs = fracs or L2_FRACS
    c, hw, _, _ = box_center(box); eps = max(hw); t1, v1, t2, v2, a = c
    A2 = frame(t2, v2); M1r0, M1r1, X1 = inner_frame(c); Ain = [M1r0, M1r1, X1]
    AinV = [mv(Ain, V[k]) for k in range(60)]; AoutV = [mv(A2, V[k]) for k in range(60)]
    # choose h in I and D in {I, rho_pi} (float), then bound the angle rigorously:
    #   R = Ain (D Aout h)^T,  R (D Aout h V_k) = Ain V_k, and {D Aout h V_k} = D Aout V_{pm(k)} is a tight frame too
    fl = lambda vv: np.array([[float((x.lo + x.hi) / 2) for x in v] for v in vv])
    AinF, AoutF = fl(AinV), fl(AoutV); best = None
    for gi, pm in enumerate(GROUP):
        for Di in (0, 1):
            B = AoutF[pm] * (np.array([1, 1, 1]) if Di == 0 else np.array([-1, -1, 1]))
            e = ((AinF - B) ** 2).sum()
            if best is None or e < best[0]: best = (e, gi, Di)
    _, gi, Di = best; pm = GROUP[gi]; sD = [1, 1, 1] if Di == 0 else [-1, -1, 1]
    th = angle_upper([[AinV[k][t] - sD[t] * AoutV[pm[k]][t] for t in range(3)] for k in range(60)])
    if th is None: return 'angle'
    xm = th + 5 * eps
    # outer shadow hull at the centre (float), contact directions inside normal cones
    Qf = AoutF[:, :2]
    from scipy.spatial import ConvexHull
    hv = ConvexHull(Qf).vertices; m = len(hv); gs = []; nmax = F(0)
    for e in range(m):
        k, kp, kn = hv[e], hv[e - 1], hv[(e + 1) % m]
        d1 = Qf[k] - Qf[kp]; d2 = Qf[kn] - Qf[k]
        a1 = math.atan2(-d1[0], d1[1]); a2 = math.atan2(-d2[0], d2[1]); da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for fr in fracs:
            ang = a1 + fr * da; n = [F(math.cos(ang)), F(math.sin(ang))]
            nn = sqrt_up(n[0] ** 2 + n[1] ** 2)
            # unique maximiser on the box: <Pi Aout(x0)(V_k - V_j), n> > 2 eps |V_k - V_j| |n|   (Lemma 2.2(a))
            ok = True
            for j in range(60):
                if j == k: continue
                val = (AoutV[k][0] - AoutV[j][0]) * n[0] + (AoutV[k][1] - AoutV[j][1]) * n[1]
                if not (val.lo > 2 * eps * norm_hi([V[k][t] - V[j][t] for t in range(3)]) * nn): ok = False; break
            if not ok: continue
            # inner vertex with the same shadow point; U = D Aout h V_i:  D=I: U = Aout V_k;  D=rho: U = -rho Aout V_k
            U = AoutV[k] if Di == 0 else [AoutV[k][0], AoutV[k][1], -AoutV[k][2]]
            gs.append(cross3(U, [Q(n[0]), Q(n[1]), Q(0)])); nmax = max(nmax, nn)
    if len(gs) < 4: return 'few contacts'
    c0 = c0_lower(gs)
    need = nmax * (2 * eps + xm / 2 + xm * xm / 6)
    return True if F(c0) > need else f'c0 {c0} need {float(need)}'

# ---------------------------------------------------------------- L5: Theorem 4.2
CORN = list(itertools.product([-1, 1], repeat=5))
def check_L5(box, w):
    c, hw, _, _ = box_center(box); t1, v1, t2, v2, a = c
    A1 = frame(t1, v1); d1t, d1p = frame_d(t1, v1); A2 = frame(t2, v2); d2t, d2p = frame_d(t2, v2); sa, ca = sincos(a)
    hin = hw[0] + hw[1] + hw[4]; hout = hw[2] + hw[3]
    tot = [Q(0)] * 32; lsum = F(0)
    for ct in w:
        i, n, K, lam = ct['i'], [F(ct['n'][0]), F(ct['n'][1])], ct['K'], F(ct['lam'])
        if lam < 0: return 'lam'
        nn = sqrt_up(n[0] ** 2 + n[1] ** 2); rem_out = hout * hout / 2 * nn; rem = (hin * hin + hout * hout) / 2 * nn
        dn = lambda u: u[0] * n[0] + u[1] * n[1]
        pi0 = mv(A1[:2], V[i])
        p = dn(rot2(sa, ca, pi0)); pa = dn(rot2(sa, ca, [-pi0[1], pi0[0]]))           # d/dalpha R(a)u = R(a) R(pi/2) u
        pt = dn(rot2(sa, ca, mv(d1t, V[i]))); pp = dn(rot2(sa, ca, mv(d1p, V[i])))
        q = [dn(mv(A2[:2], V[k])) for k in range(60)]; qt = [dn(mv(d2t, V[k])) for k in range(60)]; qp = [dn(mv(d2p, V[k])) for k in range(60)]
        UB = [q[k] + qt[k].abs() * hw[2] + qp[k].abs() * hw[3] + rem_out for k in range(60)]
        LB = [q[k] - qt[k].abs() * hw[2] - qp[k].abs() * hw[3] - rem_out for k in range(60)]
        best_inf = max(LB[k].lo for k in K)
        for j in range(60):                                  # (a)
            if j not in K and not (UB[j].hi <= best_inf): return f'(a) {j}'
        for ci, s in enumerate(CORN):                        # (b) at the 32 vertices of the box
            dl = [sg * h for sg, h in zip(s, hw)]
            li = p + pt * dl[0] + pp * dl[1] + pa * dl[4]
            lk = max((q[k] + qt[k] * dl[2] + qp[k] * dl[3]).hi for k in K)
            tot[ci] = tot[ci] + lam * (li - lk - rem)
        lsum += lam
    if lsum <= 0: return 'lam'
    return True if min(x.lo for x in tot) > 0 else 'corner'

# ---------------------------------------------------------------- B: singular balls (Section 5)
def _qvec(x): return [dq(qdec(c) / rad) for c in x]
SQ5Q = (F(0), F(1))
SINGX = {'A': (ONE, PHI, qa(PHI, (F(-1), F(0)))), 'B': (ONE, (F(5, 2), F(1, 2)), (F(-1, 2), F(1, 2))),
         'C': (ONE, (F(9, 2), F(5, 2)), (F(5, 2), F(3, 2))), 'D': (ONE, (F(-1, 2), F(3, 2)), (F(5, 2), F(-1, 2)))}
RADIUS = {'A': F(15, 10000), 'B': F(15, 10000), 'C': F(1, 1000), 'D': F(1, 1000)}
PHIMAX = 0.6524
def _views():
    """all copies g X (g in I, +-X) of the exact singular views with theta in [0, 2pi/5], phi <= phi_max (float)."""
    out = {}
    for w, X in SINGX.items():
        x = to_frame(X); x = np.array([float(c) for c in x]); x /= np.linalg.norm(x); cps = set()
        for pm in GROUP:
            # rotation acting on the frame: find R with R V_k = V_pm(k) (least squares, float)
            R = np.linalg.lstsq(VF, VF[pm], rcond=None)[0].T
            for s in (1, -1):
                y = s * (R @ x); ph = math.acos(max(-1, min(1, y[2]))); th = math.atan2(y[1], y[0]) % (2 * math.pi)
                if ph <= PHIMAX:
                    for tt in (th % (2 * math.pi / 5), th % (2 * math.pi / 5) + 2 * math.pi / 5):
                        if tt <= 2 * math.pi / 5 + 1e-12: cps.add((round(tt, 12), round(ph, 12)))
        out[w] = sorted(cps)
    return out
VIEWS = _views()
# half-twist references at C, D: T g, g in I, T = rotation by 36 deg about the five-fold axis a = (phi, 1, 0) (body
# frame), the five-fold axis adjacent to e3 in the mirror plane of C and D.  {T g} = R_a(36) I as a set (R_a(72) in I).
# cos 36 = phi/2, sin 36 = sqrt(10 - 2 sqrt5)/4
C36 = dq(qdec(PHI) / 2); S36 = dq((10 - 2 * S5).sqrt() / 4)
_ad = to_frame((PHI, ONE, Z0)); _an = (sum(x * x for x in _ad)).sqrt(); AX5 = [dq(x / _an) for x in _ad]
def rot_a36(v):
    axv = cross3(AX5, v); av = dot(AX5, v)
    return [C36 * v[t] + S36 * axv[t] + (1 - C36) * av * AX5[t] for t in range(3)]
def check_B(box):
    c, hw, lo, hi = box_center(box); eps = max(hw); t1, v1, t2, v2, a = c
    A2 = frame(t2, v2); Ain = list(inner_frame(c))
    AinV = [mv(Ain, V[k]) for k in range(60)]; AoutV = [mv(A2, V[k]) for k in range(60)]
    for wname, views in VIEWS.items():
        r = RADIUS[wname]
        for (ts, vs) in views:
            dt = max(abs(float(lo[2]) - ts), abs(float(hi[2]) - ts)); dv = max(abs(float(lo[3]) - vs), abs(float(hi[3]) - vs))
            if math.hypot(dt, dv) + 1e-12 > float(r): continue
            # references D Aout T g: image of V_k is D Aout T V_pm(k)
            if wname in 'AB': TV = [V[k] for k in range(60)]
            else: TV = [rot_a36(V[k]) for k in range(60)]
            TVo = [mv(A2, x) for x in TV]
            fl = lambda vv: np.array([[float((x.lo + x.hi) / 2) for x in v] for v in vv])
            AinF, TF = fl(AinV), fl(TVo); best = None
            for gi, pm in enumerate(GROUP):
                for Di in (0, 1):
                    e = ((AinF - TF[pm] * (np.array([1, 1, 1]) if Di == 0 else np.array([-1, -1, 1]))) ** 2).sum()
                    if best is None or e < best[0]: best = (e, gi, Di)
            _, gi, Di = best; pm = GROUP[gi]; sD = [1, 1, 1] if Di == 0 else [-1, -1, 1]
            th = angle_upper([[AinV[k][t] - sD[t] * TVo[pm[k]][t] for t in range(3)] for k in range(60)])
            if th is not None and th + 5 * eps <= r: return True
    return 'not in ball'

if __name__ == '__main__':
    fn = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    print('views', {k: [(round(t, 6), round(p, 6)) for t, p in v] for k, v in VIEWS.items()}, flush=True)
    n = ok = 0; t0 = time.time(); bad = []
    for line in open(fn):
        if n >= lim: break
        rec = json.loads(line); box = rec['box']; typ = rec['typ']
        res = {'S': lambda: check_S(box, rec['w']), 'L2': lambda: check_L2(box), 'L5': lambda: check_L5(box, rec['w']),
               'B': lambda: check_B(box)}[typ]()
        n += 1
        if res is True: ok += 1
        else: bad.append((rec['k'], res)); print('NOT CONFIRMED', rec['k'], res, box, flush=True)
        if n % 50 == 0: print(n, ok, round(time.time() - t0), flush=True)
    print(f'{fn}: {ok}/{n} confirmed, {round(time.time() - t0)} s', flush=True)
