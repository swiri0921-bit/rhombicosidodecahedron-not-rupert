"""Rigorous (interval) verification of the Steininger-Yurkevich global theorem for one box,
given float witness (w, s) found by the float search."""
import numpy as np, sys
sys.path.insert(0,'.')
from ia import I, sin, cos, dn, up
from rid_exact import rid_dec
_W = rid_dec()
from rid_setup import rid_z5 as _rz
_Vf = _rz(); _Wf = np.array([[float(x) for x in w] for w in _W])
_W = [_W[int(np.argmin(np.linalg.norm(_Wf - v, axis=1)))] for v in _Vf]    # same order as rid_z5
VLO = np.array([[dn(float(x)) for x in w] for w in _W]); VHI = np.array([[up(float(x)) for x in w] for w in _W])
VX = [I(VLO[:, j], VHI[:, j]) for j in range(3)]           # exact vertices enclosed
VF = (VLO + VHI) / 2

def mats(t, p):
    """interval entries of M, M_t, M_p at exact float angles t,p"""
    st, ct, sp, cp = sin(t), cos(t), sin(p), cos(p)
    Z = I(0.0)
    M = [[-st, ct, Z], [-(ct * cp), -(st * cp), sp]]
    Mt = [[-ct, -st, Z], [st * cp, -(ct * cp), Z]]
    Mp = [[Z, Z, Z], [ct * sp, st * sp, cp]]
    return M, Mt, Mp

def proj(Mrows, X):
    """rows (2 lists of 3 intervals) applied to vertex coords X (list of 3 interval arrays) -> 2 interval arrays"""
    return [Mrows[r][0] * X[0] + Mrows[r][1] * X[1] + Mrows[r][2] * X[2] for r in range(2)]

def rot2(a, U):
    ca, sa = cos(a), sin(a)
    return [ca * U[0] - sa * U[1], sa * U[0] + ca * U[1]]
def rot2a(a, U):   # derivative R_a
    ca, sa = cos(a), sin(a)
    return [-(sa * U[0]) - ca * U[1], ca * U[0] - sa * U[1]]

def check(box, w, s):
    lo, hi = box[:, 0], box[:, 1]
    c = (lo + hi) / 2
    eps = float(np.max(np.maximum(up(hi - c), up(c - lo))))
    t1, v1, t2, v2, a = c
    w = np.asarray(w, float); wx, wy = I(w[0]), I(w[1])
    nw = (wx.sqr() + wy.sqr()).sqrt()
    E = I(eps); E2 = E * E
    M1, M1t, M1p = mats(t1, v1); M2, M2t, M2p = mats(t2, v2)
    S = [x[s] for x in VX]
    PS = proj(M1, S)
    g0 = rot2(a, PS); ga = rot2a(a, PS)
    gt = rot2(a, proj(M1t, S)); gp = rot2(a, proj(M1p, S))
    dotw = lambda U: U[0] * wx + U[1] * wy
    G = dotw(g0) - E * dotw(ga).abs() - E * dotw(gt).abs() - E * dotw(gp).abs() - I(4.5) * E2 * nw
    Q = proj(M2, VX); Qt = proj(M2t, VX); Qp = proj(M2p, VX)
    H = dotw(Q) + E * dotw(Qt).abs() + E * dotw(Qp).abs() + I(2.0) * E2 * nw
    return bool(G.lo > H.hi.max())
