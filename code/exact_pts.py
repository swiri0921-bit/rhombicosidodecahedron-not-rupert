"""Exact (60-digit Decimal) data at the singular points A,B,C,D in the z5 frame."""
import numpy as np, sys, json, pickle
sys.path.insert(0, '.')
from decimal import Decimal as D
from dec_util import *
from rid_exact import rid_dec
from rid_setup import rid_z5
from local2 import rot_group
PF = rid_z5()
Wd = rid_dec(); Wf = np.array([[float(x) for x in w] for w in Wd])
VZ = [Wd[int(np.argmin(np.linalg.norm(Wf - v, axis=1)))] for v in PF]         # Decimal z5 vertices, rid_z5 order
# Rm (body->z5 rotation, as in rid_exact) and scale
s5 = D(5).sqrt(); P = (1 + s5) / 2
a = [D(0), P, D(1)]; na = (a[1] ** 2 + a[2] ** 2).sqrt(); a = [x / na for x in a]
v = [a[1], -a[0], D(0)]; s2 = v[0] ** 2 + v[1] ** 2; c = a[2]
K = [[D(0), -v[2], v[1]], [v[2], D(0), -v[0]], [-v[1], v[0], D(0)]]; KK = mat(K, K); f = (1 - c) / s2
RM = [[(D(1) if i == j else D(0)) + K[i][j] + KK[i][j] * f for j in range(3)] for i in range(3)]
XB = {'A': (D(1), P, P - 1), 'B': (D(1), (5 + s5) / 2, (s5 - 1) / 2), 'C': (D(1), (9 + 5 * s5) / 2, (5 + 3 * s5) / 2),
      'D': (D(1), (3 * s5 - 1) / 2, (5 - s5) / 2)}
def view(which):
    x = vec(RM, list(XB[which])); n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2).sqrt(); x = [y / n for y in x]
    if x[2] < 0: x = [-y for y in x]
    sv = (x[0] ** 2 + x[1] ** 2).sqrt(); cv = x[2]; ct = x[0] / sv; st = x[1] / sv
    return dict(st=st, ct=ct, sv=sv, cv=cv)
def frame_d(st, ct, sv, cv):
    return [[-st, ct, D(0)], [-ct * cv, -st * cv, sv], [ct * sv, st * sv, cv]]
def dframe_t(st, ct, sv, cv):
    return [[-ct, -st, D(0)], [st * cv, -ct * cv, D(0)], [-st * sv, ct * sv, D(0)]]
def dframe_v(st, ct, sv, cv):
    return [[D(0), D(0), D(0)], [ct * sv, st * sv, cv], [ct * cv, st * cv, -sv]]
# exact group elements (proper rotations) from vertex permutations
G = rot_group(PF)
def exact_g(g):
    perm = [int(np.argmin(np.linalg.norm(PF - g @ x, axis=1))) for x in PF]
    # pick 3 independent vertices
    idx = [0, 1, 2]
    for j in range(3, 60):
        Mf = PF[idx]
        if abs(np.linalg.det(Mf)) > 0.1: break
        idx = [0, 1, j]
    A_ = [VZ[i] for i in idx]; B_ = [VZ[perm[i]] for i in idx]
    # solve g A^T = B^T  ->  g = B^T (A^T)^{-1}
    At = T(A_); Bt = T(B_)
    # inverse of 3x3 Decimal
    m = At
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    inv = [[(m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3] - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3]) / det
            for j in range(3)] for i in range(3)]
    return mat(Bt, inv), perm
GE = [exact_g(g) for g in G]
def twist_exact(Tb_float):
    """find exact T = Rot(w, +-pi/5) g matching float Tb (z5 coords)."""
    ax = []
    for g in G:
        rv = __import__('scipy.spatial.transform', fromlist=['Rotation']).Rotation.from_matrix(g).as_rotvec()
        if abs(np.linalg.norm(rv) - 2 * np.pi / 5) < 1e-6:
            u = rv / np.linalg.norm(rv)
            if not any(abs(abs(u @ w) - 1) < 1e-9 for w in ax): ax.append(u)
    c36, s36 = dcos(PI / 5), dsin(PI / 5)
    best = None
    for u in ax:
        # exact axis: normalized sum of the 5 vertices of a pentagon around u -> use Decimal from vertex set
        near = [i for i in range(60) if abs(PF[i] @ u - (PF @ u).max()) < 1e-9]
        sv_ = [sum(VZ[i][k] for i in near) for k in range(3)]; nn = (sum(x * x for x in sv_)).sqrt(); w = [x / nn for x in sv_]
        for sgn in (1, -1):
            R = rotaxis(w, c36, sgn * s36)
            for (ge, perm) in GE:
                Tm = mat(R, ge); Tf = np.array([[float(x) for x in r] for r in Tm])
                err = np.abs(Tf - Tb_float).max()
                if best is None or err < best[0]: best = (err, Tm)
    return best
if __name__ == '__main__':
    cfg = json.load(open('./twist_pts.json'))
    out = {}
    for wch, (tf, vf) in {'A': (np.pi / 10, np.arctan(2 / (1 + 5 ** .5))), 'B': (3 * np.pi / 10, np.arctan(0.5)),
                          'C': (cfg['C']['t2'], cfg['C']['v2']), 'D': (cfg['D']['t2'], cfg['D']['v2'])}.items():
        vw = view(wch)
        chk = (abs(float(vw['st']) - np.sin(tf)), abs(float(vw['ct']) - np.cos(tf)), abs(float(vw['sv']) - np.sin(vf)), abs(float(vw['cv']) - np.cos(vf)))
        print(wch, 'view match err', max(chk))
        out[wch] = vw
    for wch in ('C', 'D'):
        err, Tm = twist_exact(np.array(cfg[wch]['Tb'])); print(wch, 'twist exact match err', err); out['T' + wch] = Tm
    h2 = np.array(cfg['C']['h2'])
    for (ge, perm), g in zip(GE, G):
        if np.abs(g - h2).max() < 1e-9: out['h2C'] = ge; print('h2C found')
    out['VZ'] = VZ; out['RM'] = RM
    pickle.dump(out, open('./exact_pts.pkl', 'wb'))
