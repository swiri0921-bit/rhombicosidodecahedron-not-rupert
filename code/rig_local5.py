"""Rigorous verification of the local5 certificate (fixed-direction multi-contact minimax) for a box.
Statement: for nonzero n, f_{i,n}(x) = <p_i(x),n> - max_k <q_k(x),n>; if for all x in B there is (i,n) with f >= 0,
B contains no Rupert passage. Lower/upper bounds: first-order Taylor at the centre (all quantities in IA) with
remainder 1/2 |V||n| (sum |delta_d|)^2 (every 2nd partial of R(a)M(t,v) has operator norm <= 1).
Inactive outer vertices k (not in K_n) must satisfy UB_k <= max_{k' in K_n} LB_k' (box-uniform bounds).
Certificate: float LP weights lambda >= 0; verify at all 32 corners (in IA):
   sum_c lambda_c min_{k in K_c}(A_ck + B_ck . delta) - (sum_c lambda_c) R_max > 0."""
import numpy as np, sys, itertools
sys.path.insert(0, '.')
from ia import I, sin, cos, dn, up
from rig_global import VX, mats, proj, rot2, rot2a
from rid_setup import rid_z5
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from disprover import M, R2, M_t, M_p, R_a
CORN = np.array(list(itertools.product([-1, 1], repeat=5)), float)
PF = rid_z5()

def float_cert(box, tau=0.02, tol0=0.02, fan=np.arange(-4, 5)):
    x0 = (box[:, 0] + box[:, 1]) / 2; hw = (box[:, 1] - box[:, 0]) / 2; t1, v1, t2, v2, a = x0
    V = PF; tol = min(tol0, 8 * hw.sum() + 1e-9)
    Mi = R2(a) @ M(t1, v1); Mo = M(t2, v2); P = V @ Mi.T; Q = V @ Mo.T
    dP = [V @ (R2(a) @ M_t(t1, v1)).T, V @ (R2(a) @ M_p(t1, v1)).T, None, None, V @ (R_a(a) @ M(t1, v1)).T]
    dQ = [None, None, V @ M_t(t2, v2).T, V @ M_p(t2, v2).T, None]
    h = ConvexHull(Q).vertices; m = len(h)
    so = hw[2] + hw[3]; move = (so + 0.5 * so * so) * 1.01
    contacts = []
    for e in range(m):
        q, q2 = h[e], h[(e + 1) % m]; d = Q[q2] - Q[q]; n0 = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        for j in fan:
            th = j * hw.max(); n = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ n0
            proj_ = Q @ n; H = proj_.max(); Kn = np.where(proj_ >= H - tau - 2.2 * move)[0]; pp = P @ n
            for i in np.where(pp >= H - tol)[0]:
                gi = np.array([dP[0][i] @ n, dP[1][i] @ n, 0, 0, dP[4][i] @ n])
                A = pp[i] - proj_[Kn]; B = np.array([gi - np.array([0, 0, dQ[2][k] @ n, dQ[3][k] @ n, 0]) for k in Kn])
                contacts.append((i, n, Kn, A, B))
    if not contacts: return None
    F = np.array([(c[3][None, :] + (CORN * hw) @ c[4].T).min(1) for c in contacts]).T
    C = F.shape[1]
    r = linprog(np.r_[np.zeros(C), -1], A_ub=np.hstack([-F, np.ones((len(CORN), 1))]), b_ub=np.zeros(len(CORN)),
                A_eq=np.r_[np.ones(C), 0][None, :], b_eq=[1], bounds=[(0, None)] * C + [(None, None)], method='highs')
    if r.status != 0 or -r.fun <= 0: return None
    lam = r.x[:C]; used = [(contacts[c], lam[c]) for c in range(C) if lam[c] > 1e-12]
    return used

def check(box):
    used = float_cert(box)
    if used is None: return False
    lo, hi = box[:, 0], box[:, 1]; c = (lo + hi) / 2
    hw = np.maximum(up(hi - c), up(c - lo))
    t1, v1, t2, v2, a = c
    M1, M1t, M1p = mats(t1, v1); M2, M2t, M2p = mats(t2, v2)
    Pc = rot2(a, proj(M1, VX)); Pa = rot2a(a, proj(M1, VX)); Pt = rot2(a, proj(M1t, VX)); Pp = rot2(a, proj(M1p, VX))
    Qc = proj(M2, VX); Qt = proj(M2t, VX); Qp = proj(M2p, VX)
    hin = float(up(up(hw[0] + hw[1]) + hw[4])); hout = float(up(hw[2] + hw[3]))
    Rin = I(0.5) * I(hin) * I(hin); Rout = I(0.5) * I(hout) * I(hout)     # |V|<=1 (exact radius 1)
    total = None; lamsum = 0.0
    corner_vals = None
    for (i, n, Kn, _, _), lam in used:
        nx, ny = I(n[0]), I(n[1]); nn = (nx.sqr() + ny.sqr()).sqrt()
        dotn = lambda U, idx: U[0][idx] * nx + U[1][idx] * ny
        # inner vertex i: value and gradient (d t1, d v1, d a)
        pv = dotn(Pc, i); gt = dotn(Pt, i); gp = dotn(Pp, i); ga = dotn(Pa, i)
        # outer: all vertices
        qv = Qc[0] * nx + Qc[1] * ny; qt = Qt[0] * nx + Qt[1] * ny; qp = Qp[0] * nx + Qp[1] * ny
        # box-uniform bounds for outer vertices
        H2 = I(hw[2]); H3 = I(hw[3])
        UB = qv + qt.abs() * H2 + qp.abs() * H3 + Rout * nn
        LB = qv - qt.abs() * H2 - qp.abs() * H3 - Rout * nn
        mask = np.ones(60, bool); mask[Kn] = False
        if mask.any() and not (UB.hi[mask].max() <= LB.lo[Kn].max()): return False
        # corner values of min_k (A_k + B_k . delta) - remainders
        vals = []
        for cs in CORN:
            dl = cs * hw
            lin_i = pv + gt * I(dl[0]) + gp * I(dl[1]) + ga * I(dl[4])
            ub_k = qv[Kn] + qt[Kn] * I(dl[2]) + qp[Kn] * I(dl[3])
            m = lin_i - I(ub_k.hi.max())       # min over k of (lin_i - ub_k)  >=  lin_i.lo - max ub_k.hi
            vals.append(m - (Rin + Rout) * nn)
        lo_vals = np.array([float(v.lo) for v in vals])
        L = I(lam)
        cv = [L * v for v in vals]
        corner_vals = cv if corner_vals is None else [x + y for x, y in zip(corner_vals, cv)]
    return bool(min(float(v.lo) for v in corner_vals) > 0)
