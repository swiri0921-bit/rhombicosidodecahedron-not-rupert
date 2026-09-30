"""Rigorous (interval) verification of the SY Local Theorem conditions (as implemented in generate_solutiontree.R /
disprover.local_theorem) for one box. The two vertex triples must be congruent (SY, Definition 34: P_i = L Q_i for an
orthonormal L, which need not be a symmetry of the polyhedron). Congruence is certified either by a symmetry of the
polyhedron (vertex permutation, PERMS) or by exact equality of the Gram matrices of the two triples in Q(sqrt5)
(gram_equal), which is equivalent to the existence of such an L."""
import numpy as np, sys
sys.path.insert(0, '.')
from ia import I, sin, cos, dn, up
from rig_global import VX, mats, proj, rot2
from rid_setup import rid_z5
from disprover import local_oracle
from local2 import rot_group
PF = rid_z5(); RHO = 1.0
G = rot_group(PF); GI = G + [-g for g in G]
PERMS = [np.array([int(np.argmin(np.linalg.norm(PF - (g @ v), axis=1))) for v in PF]) for g in GI]
for pm, g in zip(PERMS, GI):
    assert np.abs(PF[pm] - PF @ g.T).max() < 1e-9
SQ2 = I(dn(np.sqrt(2.0)), up(np.sqrt(2.0)))
_GRAM = {}
def gram_equal(ps, qs):
    key = (ps, qs)
    if key not in _GRAM:
        from exact_check import V as VQ
        from q5 import dot
        _GRAM[key] = all(dot(VQ[ps[a]], VQ[ps[b]]) == dot(VQ[qs[a]], VQ[qs[b]]) for a in range(3) for b in range(a, 3))
    return _GRAM[key]
def vert(i): return [x[i] for x in VX]
def Xrow(t, p):
    st, ct, sp, cp = sin(t), cos(t), sin(p), cos(p)
    return [ct * sp, st * sp, cp]
def dot3(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross2(u, v): return u[0] * v[1] - u[1] * v[0]
def norm2(u): return (u[0].sqr() + u[1].sqr()).sqrt()

def check(box):
    lo, hi = box[:, 0], box[:, 1]
    c = (lo + hi) / 2
    eps = float(np.max(np.maximum(up(hi - c), up(c - lo))))
    res = local_oracle(PF, box, RHO)
    if res is None: return False
    p1, p2, p3, q1, q2, q3, s_p, s_q, r = res
    ps, qs = [p1, p2, p3], [q1, q2, q3]
    if not any(all(pm[p] == q for p, q in zip(ps, qs)) for pm in PERMS):
        if not gram_equal(tuple(ps), tuple(qs)): return False   # exact orthogonal L (Gram matrices equal in Q(sqrt5))
    t1, v1, t2, v2, a = c
    E = I(eps); rho = I(RHO)
    M1, _, _ = mats(t1, v1); M2, _, _ = mats(t2, v2)
    X1 = Xrow(t1, v1); X2 = Xrow(t2, v2)
    Pv = [vert(p) for p in ps]; Qv = [vert(q) for q in qs]
    g = I(2.0) * SQ2 * rho * rho * E + I(2.0) * rho * rho * E * E
    for k in range(3):
        if not ((I(s_p) * dot3(Pv[k], X1)).lo > (SQ2 * rho * E).hi): return False
        if not ((I(s_q) * dot3(Qv[k], X2)).lo > (SQ2 * rho * E).hi): return False
    PP = [proj(M1, P) for P in Pv]; QQ = [proj(M2, Q) for Q in Qv]
    for (u, v) in ((0, 1), (1, 2), (2, 0)):
        if not (cross2(PP[u], PP[v]).lo > g.hi): return False
        if not (cross2(QQ[u], QQ[v]).lo > g.hi): return False
    R = I(r)
    for k in range(3):
        if not (norm2(QQ[k]).lo > (R * rho + I(1.42) * rho * E).hi): return False
    RP = [rot2(a, PP[k]) for k in range(3)]
    delta = None
    for k in range(3):
        d = norm2([RP[k][0] - QQ[k][0], RP[k][1] - QQ[k][1]]) * I(0.5)
        delta = d if delta is None else I(np.maximum(delta.lo, d.lo), np.maximum(delta.hi, d.hi))
    thr = (I(4.5) * rho * E + I(2.0) * delta) * I(1.0)
    AX = proj(M2, VX)          # all vertices projected (interval arrays)
    for k, q in enumerate(qs):
        mq = QQ[k]
        mask = np.arange(60) != q
        A = [x[mask] for x in AX]; Av = [x[mask] for x in VX]
        dq = [Av[j] - Qv[k][j] for j in range(3)]
        ndq = (dq[0].sqr() + dq[1].sqr() + dq[2].sqr()).sqrt()
        noms = (mq[0].sqr() + mq[1].sqr()) - (A[0] * mq[0] + A[1] * mq[1]) - rho * ndq * (I(2.0) * SQ2 * E + I(2.0) * E * E)
        dmq = [A[0] - mq[0], A[1] - mq[1]]
        dens = (norm2(mq) + I(1.42) * rho * E) * (norm2(dmq) + I(2.84) * rho * E)
        # need noms/dens >= thr/(2 r rho)  <=>  noms * 2 r rho >= thr * dens  (dens > 0)
        lhs = noms * (I(2.0) * R * rho); rhs = thr * dens
        if not (dens.lo > 0).all(): return False
        if not (lhs.lo >= rhs.hi).all(): return False
    return True
