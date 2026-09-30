"""Exact (Decimal) derivative D psi of psi(w) = log(RZ A(w) M A(w)^T) at w=0 for the reflection references,
M = h (A) or Tb h Tb^T (C).  Returns float enclosure (3x2)."""
import numpy as np, sys, pickle
sys.path.insert(0, '.')
from decimal import Decimal as D
from dec_util import mat, T
EX = pickle.load(open('./exact_pts.pkl', 'rb'))
def frames(which):
    vw = EX[which]; st, ct, sv, cv = vw['st'], vw['ct'], vw['sv'], vw['cv']
    A = [[-st, ct, D(0)], [-ct * cv, -st * cv, sv], [ct * sv, st * sv, cv]]
    At = [[-ct, -st, D(0)], [st * cv, -ct * cv, D(0)], [-st * sv, ct * sv, D(0)]]
    Ap = [[D(0), D(0), D(0)], [ct * sv, st * sv, cv], [ct * cv, st * cv, -sv]]
    return A, At, Ap
RZ = [[D(-1), D(0), D(0)], [D(0), D(-1), D(0)], [D(0), D(0), D(1)]]
def dpsi(which, Mbody):
    A, At, Ap = frames(which)
    Phi0 = mat(RZ, mat(mat(A, Mbody), T(A)))
    err = max(abs(Phi0[i][j] - (1 if i == j else 0)) for i in range(3) for j in range(3))
    cols = []
    for Ad in (At, Ap):
        d = mat(RZ, [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(mat(mat(Ad, Mbody), T(A)), mat(mat(A, Mbody), T(Ad)))])
        # skew part vee: (d[2][1]-d[1][2])/2, ...
        cols.append([(d[2][1] - d[1][2]) / 2, (d[0][2] - d[2][0]) / 2, (d[1][0] - d[0][1]) / 2])
    return float(err), [[cols[j][i] for j in range(2)] for i in range(3)]
if __name__ == '__main__':
    from rid_setup import rid_z5; from local2 import rot_group
    import json
    out = {}
    # A: h = 2-fold about the view axis
    from exact_pts import GE, view
    PF = rid_z5(); G = rot_group(PF)
    XA = EX['A']; x = np.array([float(XA['ct'] * XA['sv']), float(XA['st'] * XA['sv']), float(XA['cv'])])
    for (ge, perm), g in zip(GE, G):
        if abs(np.trace(g) + 1) < 1e-9 and abs(abs(np.linalg.eigh((g + np.eye(3)) / 2)[1][:, -1] @ x) - 1) < 1e-6:
            e, Dp = dpsi('A', ge); print('A Phi0-I', e, 'Dpsi', [[float(v) for v in r] for r in Dp]); out['A'] = [[float(v) for v in r] for r in Dp]
    Mc = mat(mat(EX['TC'], EX['h2C']), T(EX['TC']))
    e, Dp = dpsi('C', Mc); print('C Phi0-I', e, 'Dpsi2', [[float(v) for v in r] for r in Dp]); out['C'] = [[float(v) for v in r] for r in Dp]
    json.dump(out, open('./dpsi_exact.json', 'w'))
