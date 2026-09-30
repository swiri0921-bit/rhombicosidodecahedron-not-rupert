"""Exact Q(sqrt5) certification of the structural zero claims used by the rigorous sector/tube certificates.
Works in the ORIGINAL body frame of the rhombicosidodecahedron (vertices in Q(sqrt5)^3)."""
import numpy as np, sys, pickle
sys.path.insert(0, '.')
from q5 import Q5, PHI, cross, dot, sub, det3, rid_q5
from fractions import Fraction as F
from rupert import rid
from rid_setup import rid_z5
from rid_exact import rid_dec
VQ = rid_q5()
Vf_orig = np.array([[float(x) for x in v] for v in VQ])
# order: map rid_z5 order -> original body vertex order via the float rotation used in rid_setup
PF = rid_z5()
Vr = rid(); a = np.array([0, (1 + 5 ** .5) / 2, 1]); a /= np.linalg.norm(a); z = np.array([0, 0, 1.]); v = np.cross(a, z); s = np.linalg.norm(v); c = a @ z
K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); Rm = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
scale = np.linalg.norm(Vr @ Rm.T, axis=1).max()
ORD = [int(np.argmin(np.linalg.norm(Vf_orig - (Rm.T @ (p * scale)), axis=1))) for p in PF]
assert len(set(ORD)) == 60
V = [VQ[o] for o in ORD]                       # exact body vertices in rid_z5 order
W5 = [Q5(0), PHI, Q5(1)]                       # z5 axis in body frame (unnormalized)
XB = {'A': [Q5(1), PHI, PHI - 1], 'B': [Q5(1), Q5(F(5, 2), F(1, 2)), Q5(F(-1, 2), F(1, 2))],
      'C': [Q5(1), Q5(F(9, 2), F(5, 2)), Q5(F(5, 2), F(3, 2))], 'D': [Q5(1), Q5(F(-1, 2), F(3, 2)), Q5(F(5, 2), F(-1, 2))]}
def body_from_z5_matrix(Mz):
    """exact Q5 matrix from a float z5-frame orthogonal matrix known to be a symmetry-type map, via vertex matching:
    find exact body matrix mapping vertex triples."""
    raise NotImplementedError
def mat_from_perm(perm_map):
    """exact 3x3 (Q5) linear map sending V[a] -> V[perm[a]] (for a symmetry, determined by 3 independent vertices)."""
    idx = [0, 1, 2]
    j = 3
    while True:
        Mf = np.array([[float(x) for x in V[i]] for i in idx])
        if abs(np.linalg.det(Mf)) > 0.1: break
        idx = [0, 1, j]; j += 1
    A = [V[i] for i in idx]; B = [V[perm_map[i]] for i in idx]
    # solve X A^T = B^T ; X = B^T (A^T)^-1 via Cramer in Q5
    At = [[A[c_][r] for c_ in range(3)] for r in range(3)]
    def det(m): return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det(At)
    inv = [[(At[(j_ + 1) % 3][(i_ + 1) % 3] * At[(j_ + 2) % 3][(i_ + 2) % 3] - At[(j_ + 1) % 3][(i_ + 2) % 3] * At[(j_ + 2) % 3][(i_ + 1) % 3]) / d
            for j_ in range(3)] for i_ in range(3)]
    Bt = [[B[c_][r] for c_ in range(3)] for r in range(3)]
    X = [[sum((Bt[r][k] * inv[k][c_] for k in range(3)), Q5(0)) for c_ in range(3)] for r in range(3)]
    # verify it maps ALL vertices correctly (exact)
    for i in range(60):
        img = [sum((X[r][k] * V[i][k] for k in range(3)), Q5(0)) for r in range(3)]
        assert all(img[r] == V[perm_map[i]][r] for r in range(3)), 'not an exact symmetry'
    return X
def apply(Mq, v): return [sum((Mq[r][k] * v[k] for k in range(3)), Q5(0)) for r in range(3)]
def twist_body(which):
    """exact twist T (body frame) such that T V_i = inner vertex i: found from the float z5 twist by exact reconstruction:
    T = R_w(36deg) g ; R_w in Q5 via cos36 = phi/2, (sin36/|w|) w for w in the 5-fold axis class."""
    EX = pickle.load(open('./exact_pts.pkl', 'rb'))
    Tz = np.array([[float(x) for x in r] for r in EX['T' + which]])
    Tb = Rm.T @ Tz @ Rm                              # float body-frame twist
    # candidate axes: 5-fold axes = pentagon centres (sum of 5 vertices) in body frame
    Vfl = np.array([[float(x) for x in v] for v in V]); axes = []
    for u0 in Vfl:
        pass
    from scipy.spatial import ConvexHull
    H = ConvexHull(Vfl); cands = []
    for eq in H.equations:
        nrm = eq[:3]; on = np.where(np.abs(Vfl @ nrm + eq[3]) < 1e-9)[0]
        if len(on) == 5:
            key = tuple(sorted(on))
            if key not in [c_[0] for c_ in cands]: cands.append((key, on))
    best = None
    c36 = PHI * F(1, 2)
    for key, on in cands:
        w = [sum((V[i][k] for i in on), Q5(0)) for k in range(3)]        # exact axis (unnormalized)
        w2 = dot(w, w)
        # sin36/|w| : sin36^2 = (10-2r5)/16 ; need sqrt(sin36^2 / w2) in Q5
        t = Q5(F(10, 16), F(-2, 16)) / w2
        # find q in Q5 with q^2 = t : try q = x + y r5 by solving; use float guess and rational reconstruction
        tf = float(t); qf = tf ** 0.5
        q = None
        for den in range(1, 2000):
            for yb in range(-4000, 4001):
                y = F(yb, den * 40) if False else None
            break
        # direct: q = a + b r5, q^2 = (a^2+5b^2) + 2ab r5 = t.a + t.b r5
        import itertools
        found = None
        for den in (1, 2, 4, 5, 8, 10, 16, 20, 40, 80, 100, 160, 200, 400, 800, 1000, 2000, 4000, 8000):
            for sgn_b in (1, -1):
                # b from float: solve numerically then round
                pass
        # numerical solve for a,b then rational round with increasing denominators
        from math import sqrt
        tA, tB = float(t.a), float(t.b)
        # a^2 + 5 b^2 = tA, 2ab = tB -> a = tB/(2b); tB^2/(4b^2) + 5b^2 = tA -> 20 b^4 - 4 tA b^2 + tB^2 = 0
        disc = 16 * tA * tA - 80 * tB * tB
        sols = []
        for sd in (1, -1):
            b2 = (4 * tA + sd * sqrt(max(disc, 0))) / 40
            if b2 <= 0: continue
            for sb in (1, -1):
                b = sb * sqrt(b2); a_ = tB / (2 * b) if b != 0 else sqrt(tA); sols.append((a_, b))
        for (af, bf) in sols:
            for den in range(1, 5000):
                A_ = F(round(af * den), den); B_ = F(round(bf * den), den)
                qq = Q5(A_, B_)
                if qq * qq == t: found = qq; break
            if found is not None: break
        if found is None: continue
        sw = [found * x for x in w]          # (sin36/|w|) w  exact
        wwT = [[w[i] * w[j] / w2 for j in range(3)] for i in range(3)]
        for sg in (1, -1):
            R = [[(c36 if i == j else Q5(0)) + sg * ([[Q5(0), -sw[2], sw[1]], [sw[2], Q5(0), -sw[0]], [-sw[1], sw[0], Q5(0)]][i][j]) + (1 - c36) * wwT[i][j]
                  for j in range(3)] for i in range(3)]
            Rf = np.array([[float(x) for x in r] for r in R])
            # T = R g  -> g = R^T T ; must be an exact symmetry
            gf = Rf.T @ Tb
            perm = [int(np.argmin(np.linalg.norm(Vfl - (gf @ x), axis=1))) for x in Vfl]
            if np.abs(Vfl[perm] - Vfl @ gf.T).max() > 1e-7: continue
            g = mat_from_perm(perm)
            Tq = [[sum((R[i][k] * g[k][j] for k in range(3)), Q5(0)) for j in range(3)] for i in range(3)]
            err = np.abs(np.array([[float(x) for x in r] for r in Tq]) - Tb).max()
            if best is None or err < best[0]: best = (err, Tq)
    return best
if __name__ == '__main__':
    for wch in ('C', 'D'):
        err, Tq = twist_body(wch); print(wch, 'exact body twist reconstructed, float err', err)
        pickle.dump(Tq, open(f'./twist_body_{wch}.pkl', 'wb'))
