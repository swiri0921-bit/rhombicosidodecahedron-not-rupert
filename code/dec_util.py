"""Decimal (60-digit) helpers for exact-data preparation; results are enclosed as float intervals."""
from decimal import Decimal as D, getcontext
import numpy as np, sys
sys.path.insert(0, '.')
from ia import dn, up, I
getcontext().prec = 70
def dpi():
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
    def atan_inv(x):
        x = D(x); s = D(0); t = D(1) / x; n = 1; sign = 1; x2 = x * x
        while True:
            term = t / n
            if term < D(10) ** -75: break
            s += sign * term; t /= x2; n += 2; sign = -sign
        return s
    return 16 * atan_inv(5) - 4 * atan_inv(239)
PI = dpi()
def dsin(x):
    x = D(x); s = D(0); t = x; n = 1
    while abs(t) > D(10) ** -75:
        s += t; n += 2; t = -t * x * x / ((n - 1) * n)
    return s
def dcos(x): return dsin(PI / 2 - D(x))
def dsqrt(x): return D(x).sqrt()
TINY = D(10) ** -40
def encl(x):
    """float interval enclosing the exact number approximated by the 70-digit Decimal x (|error| < 1e-60).
    Numerically zero values (|x| < 1e-40) get the absolute enclosure [-1e-40, 1e-40]; otherwise
    [dn(f), up(f)] with f = float(x) contains x +- 1e-60 because ulp(f) > 1e-55."""
    if abs(D(x)) < TINY: return -1e-40, 1e-40
    f = float(x)
    return dn(f), up(f)
def Iarr(A):
    A = np.array(A, dtype=object); lo = np.vectorize(lambda x: encl(x)[0])(A).astype(float); hi = np.vectorize(lambda x: encl(x)[1])(A).astype(float)
    return I(lo, hi)
def mat(A, B): return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def T(A): return [list(r) for r in zip(*A)]
def vec(A, v): return [sum(A[i][k] * v[k] for k in range(len(v))) for i in range(len(A))]
def rotaxis(w, c, s):
    """rotation about unit axis w (Decimals) with cos=c, sin=s"""
    K = [[D(0), -w[2], w[1]], [w[2], D(0), -w[0]], [-w[1], w[0], D(0)]]
    KK = mat(K, K)
    return [[(D(1) if i == j else D(0)) + s * K[i][j] + (1 - c) * KK[i][j] for j in range(3)] for i in range(3)]
