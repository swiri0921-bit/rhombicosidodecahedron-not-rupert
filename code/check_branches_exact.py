"""Exact (Q(sqrt5)) verification that the second references meet the first ones at the singular points:
 - at A: the symmetry H (vertex permutation permA) is the half-turn about X_A  =>  A(0) H A(0)^T = rho_pi;
 - at C: T_C H_2 T_C^T is the half-turn about X_C (H_2 the symmetry used for the second tube/sector reference).
A rotation R with R X = X and trace R = -1 is the half-turn about X; orthogonality and det = 1 are checked exactly."""
import os, sys, json, pickle
import numpy as np
R_ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, R_)
from q5 import Q5
from exact_check import mat_from_perm, apply, XB, V
from rid_setup import rid_z5
def mul(A, B): return [[sum((A[i][k] * B[k][j] for k in range(3)), Q5(0)) for j in range(3)] for i in range(3)]
def tr(A): return [[A[j][i] for j in range(3)] for i in range(3)]
def det(m): return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
def is_id(M): return all(M[i][j] == Q5(int(i == j)) for i in range(3) for j in range(3))
def half_turn_about(M, X):
    return is_id(mul(M, tr(M))) and det(M) == Q5(1) and all(a == b for a, b in zip(apply(M, X), X)) and (M[0][0] + M[1][1] + M[2][2]) == Q5(-1)
ok = True
H = mat_from_perm(list(np.load(os.path.join(R_, 'permA.npy'))))
r = half_turn_about(H, XB['A']); ok &= r; print('A: H is the half-turn about X_A:', r)
Tq = pickle.load(open(os.path.join(R_, 'twist_body_C.pkl'), 'rb'))
PFz = rid_z5(); h2 = np.array(json.load(open(os.path.join(R_, 'twist_pts.json')))['C']['h2'])
perm = [int(np.argmin(np.linalg.norm(PFz - (h2 @ x), axis=1))) for x in PFz]
H2 = mat_from_perm(perm)
print('C: T_C orthogonal with det 1:', is_id(mul(Tq, tr(Tq))) and det(Tq) == Q5(1))
r = half_turn_about(mul(mul(Tq, H2), tr(Tq)), XB['C']); ok &= r; print('C: T_C H_2 T_C^T is the half-turn about X_C:', r)
print('ALL OK' if ok else 'PROBLEM')
