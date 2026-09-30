"""Minimal outward-rounded interval arithmetic on numpy arrays (float64).
Every elementary operation is computed in round-to-nearest and then widened by one ulp in each direction
(np.nextafter), which encloses the exact result. sin/cos are enclosed rigorously (argument reduction and Taylor polynomials in interval arithmetic);
monotonicity pieces are handled explicitly.
Intervals are pairs (lo, hi) of numpy arrays (or floats)."""
import numpy as np

INF = np.inf
def dn(x): return np.nextafter(x, -INF)
def up(x): return np.nextafter(x, INF)

class I:
    __slots__ = ('lo', 'hi')
    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=np.float64)
        self.lo = lo; self.hi = lo if hi is None else np.asarray(hi, dtype=np.float64)
    @staticmethod
    def c(x): return x if isinstance(x, I) else I(x)
    def __add__(a, b):
        # x + 0 is exact in floating point: no outward rounding when one operand is the exact zero interval
        b = I.c(b); lo = a.lo + b.lo; hi = a.hi + b.hi
        ex = ((a.lo == 0) & (a.hi == 0)) | ((b.lo == 0) & (b.hi == 0))
        return I(np.where(ex, lo, dn(lo)), np.where(ex, hi, up(hi)))
    __radd__ = __add__
    def __neg__(a): return I(-a.hi, -a.lo)
    def __sub__(a, b):
        b = I.c(b); return I(dn(a.lo - b.hi), up(a.hi - b.lo))
    def __rsub__(a, b): return I.c(b) - a
    def __mul__(a, b):
        b = I.c(b)
        p1 = a.lo * b.lo; p2 = a.lo * b.hi; p3 = a.hi * b.lo; p4 = a.hi * b.hi
        mn = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4)); mx = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        # x * 0 == 0 exactly: an exact zero factor gives the exact zero interval
        ez = ((a.lo == 0) & (a.hi == 0)) | ((b.lo == 0) & (b.hi == 0))
        return I(np.where(ez, 0.0, dn(mn)), np.where(ez, 0.0, up(mx)))
    __rmul__ = __mul__
    def abs(a):
        lo = np.where(a.lo >= 0, a.lo, np.where(a.hi <= 0, -a.hi, 0.0))
        hi = np.maximum(np.abs(a.lo), np.abs(a.hi))
        return I(lo, hi)
    def sqr(a):
        m = a.abs(); return I(dn(m.lo * m.lo), up(m.hi * m.hi))
    def sqrt(a):
        return I(dn(np.sqrt(np.maximum(a.lo, 0))), up(np.sqrt(a.hi)))
    def __getitem__(a, k): return I(a.lo[k], a.hi[k])
    def __repr__(a): return f'I({a.lo},{a.hi})'
    @property
    def shape(self): return np.shape(self.lo)

def widen(x, k):
    lo, hi = x
    for _ in range(k): lo = dn(lo); hi = up(hi)
    return lo, hi

TWO_PI_HI = up(2 * np.pi)   # used only to bound k ranges

# ---- rigorous sin/cos (no reliance on libm accuracy) ----
from fractions import Fraction as _F
from math import factorial as _fact
def _ienc_frac(q):
    f = float(q)
    return dn(f), up(f)
_SC = [_ienc_frac(_F((-1) ** n, _fact(2 * n + 1))) for n in range(11)]   # sin: r * sum c_n r^(2n)
_CC = [_ienc_frac(_F((-1) ** n, _fact(2 * n))) for n in range(11)]       # cos: sum c_n r^(2n)
_SREM = 1.0 / _fact(23) * (1 + 1e-12); _CREM = 1.0 / _fact(22) * (1 + 1e-12)
PI_LO, PI_HI = dn(np.pi), up(np.pi)            # np.pi < pi < nextafter(np.pi)
PI2_LO, PI2_HI = PI_LO / 2, PI_HI / 2          # exact halving
def _poly(r2, C):
    p = I(np.full(np.shape(r2.lo), C[-1][0]), np.full(np.shape(r2.lo), C[-1][1]))
    for lo, hi in reversed(C[:-1]):
        p = p * r2 + I(np.full(np.shape(r2.lo), lo), np.full(np.shape(r2.lo), hi))
    return p
def _sincos_point(x):
    """rigorous enclosures of sin(x), cos(x) for float arrays x with |x| <= 1e3 (reduction by k*pi/2 in IA, Taylor
    polynomials of degree 21/20 with Lagrange remainder |r|^23/23!, |r|^22/22!, |r| <= 0.8)."""
    x = np.asarray(x, dtype=np.float64)
    k = np.round(x / (np.pi / 2))
    r = I(x) - I(k) * I(np.full(x.shape, PI2_LO), np.full(x.shape, PI2_HI))
    rm = np.maximum(np.abs(r.lo), np.abs(r.hi)); assert (rm < 0.8).all()
    r2 = r.sqr()
    s = r * _poly(r2, _SC); c = _poly(r2, _CC)
    es = up(up(rm ** 23) * _SREM); ec = up(up(rm ** 22) * _CREM)
    s = I(dn(s.lo - es), up(s.hi + es)); c = I(dn(c.lo - ec), up(c.hi + ec))
    m = (k.astype(np.int64) % 4)
    S = [s, c, -s, -c]; C = [c, -s, -c, s]
    slo = np.select([m == j for j in range(4)], [S[j].lo for j in range(4)])
    shi = np.select([m == j for j in range(4)], [S[j].hi for j in range(4)])
    clo = np.select([m == j for j in range(4)], [C[j].lo for j in range(4)])
    chi = np.select([m == j for j in range(4)], [C[j].hi for j in range(4)])
    return (slo, shi), (clo, chi)

import math as _m
_nd = lambda v: _m.nextafter(v, -_m.inf)
_nu = lambda v: _m.nextafter(v, _m.inf)
def _smul(a, b, c, d):
    p = (a * c, a * d, b * c, b * d); return _nd(min(p)), _nu(max(p))
import functools as _ft
@_ft.lru_cache(maxsize=1 << 20)
def _sincos_scalar(x):
    """scalar version of _sincos_point (same algorithm, plain floats with nextafter)."""
    k = round(x / (_m.pi / 2))
    klo, khi = _smul(float(k), float(k), PI2_LO, PI2_HI)
    rlo, rhi = _nd(x - khi), _nu(x - klo)
    rm = max(abs(rlo), abs(rhi)); assert rm < 0.8
    r2lo = 0.0 if rlo <= 0 <= rhi else _nd(min(rlo * rlo, rhi * rhi)); r2hi = _nu(rm * rm)
    def poly(C):
        plo, phi = C[-1]
        for clo, chi in reversed(C[:-1]):
            plo, phi = _smul(plo, phi, r2lo, r2hi); plo, phi = _nd(plo + clo), _nu(phi + chi)
        return plo, phi
    slo, shi = poly(_SC); slo, shi = _smul(rlo, rhi, slo, shi); clo, chi = poly(_CC)
    es = _nu(_nu(rm ** 23) * _SREM); ec = _nu(_nu(rm ** 22) * _CREM)
    slo, shi, clo, chi = _nd(slo - es), _nu(shi + es), _nd(clo - ec), _nu(chi + ec)
    j = k % 4
    if j == 0: return (slo, shi), (clo, chi)
    if j == 1: return (clo, chi), (-shi, -slo)
    if j == 2: return (-shi, -slo), (-chi, -clo)
    return (-chi, -clo), (slo, shi)

def _sin_range(lo, hi):
    """enclosure of sin over [lo, hi] (arrays), assuming hi - lo < 2 (radians) -- plenty for our boxes."""
    if np.ndim(lo) == 0 and np.ndim(hi) == 0:
        return I(*_sin_range_scalar(float(lo), float(hi)))
    return _sin_range_arr(lo, hi)

@_ft.lru_cache(maxsize=1 << 20)
def _sin_range_scalar(lo, hi):
    """pure function of the two floats (memoised; identical results)."""
    if True:
        (alo, ahi), _ = _sincos_scalar(lo); (blo, bhi), _ = _sincos_scalar(hi)
        mn = min(alo, blo); mx = max(ahi, bhi)
        for off, is_max in ((0.5, True), (-0.5, False)):
            k0 = _m.floor((lo - off * PI_HI) / (2 * PI_LO)) - 1
            for dk in range(4):
                kk = k0 + dk; u1 = (off + 2 * kk) * PI_LO; u2 = (off + 2 * kk) * PI_HI
                if _nu(max(u1, u2)) >= lo and _nd(min(u1, u2)) <= hi:
                    if is_max: mx = 1.0
                    else: mn = -1.0
        return (max(mn, -1.0), min(mx, 1.0))

def _sin_range_arr(lo, hi):
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    (a_lo, a_hi), _ = _sincos_point(np.atleast_1d(lo)); (b_lo, b_hi), _ = _sincos_point(np.atleast_1d(hi))
    a_lo = a_lo.reshape(np.shape(lo)); a_hi = a_hi.reshape(np.shape(lo)); b_lo = b_lo.reshape(np.shape(hi)); b_hi = b_hi.reshape(np.shape(hi))
    mn = np.minimum(a_lo, b_lo); mx = np.maximum(a_hi, b_hi)
    # interior maxima at pi/2 + 2 k pi, minima at -pi/2 + 2 k pi (conservative pi bounds, outward-rounded products)
    for off, is_max in ((0.5, True), (-0.5, False)):
        k_lo = np.floor((lo - off * PI_HI) / (2 * PI_LO)) - 1
        for dk in range(4):
            k = k_lo + dk
            xa = dn(np.minimum((off + 2 * k) * PI_LO, (off + 2 * k) * PI_HI))
            xb = up(np.maximum((off + 2 * k) * PI_LO, (off + 2 * k) * PI_HI))
            inside = (xb >= lo) & (xa <= hi)
            if is_max: mx = np.where(inside, 1.0, mx)
            else: mn = np.where(inside, -1.0, mn)
    return I(np.maximum(mn, -1.0), np.minimum(mx, 1.0))

def sin(x): x = I.c(x); return _sin_range(x.lo, x.hi)
def cos(x):
    x = I.c(x)
    # cos t = sin(t + pi/2); shift with outward rounding
    return _sin_range(dn(x.lo + PI2_LO), up(x.hi + PI2_HI))

def dot(A, B):
    """sum of products along last axis for lists of intervals: A, B lists of I with same shapes"""
    s = A[0] * B[0]
    for a, b in zip(A[1:], B[1:]): s = s + a * b
    return s
