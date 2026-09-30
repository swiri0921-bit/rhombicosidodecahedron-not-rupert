"""Exact arithmetic in Q(sqrt5): a + b sqrt5 with Fractions."""
from fractions import Fraction as F
import math
class Q5:
    __slots__ = ('a', 'b')
    def __init__(s, a=0, b=0): s.a = F(a); s.b = F(b)
    @staticmethod
    def c(x): return x if isinstance(x, Q5) else Q5(x)
    def __add__(s, o): o = Q5.c(o); return Q5(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __sub__(s, o): o = Q5.c(o); return Q5(s.a - o.a, s.b - o.b)
    def __rsub__(s, o): return Q5.c(o) - s
    def __neg__(s): return Q5(-s.a, -s.b)
    def __mul__(s, o): o = Q5.c(o); return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__
    def conj(s): return Q5(s.a, -s.b)
    def norm(s): return s.a * s.a - 5 * s.b * s.b
    def __truediv__(s, o):
        o = Q5.c(o); n = o.norm(); cj = o.conj() * s; return Q5(cj.a / n, cj.b / n)
    def __eq__(s, o): o = Q5.c(o); return s.a == o.a and s.b == o.b
    def iszero(s): return s.a == 0 and s.b == 0
    def __float__(s): return float(s.a) + float(s.b) * math.sqrt(5)
    def sign(s):
        # exact sign of a + b sqrt5
        a, b = s.a, s.b
        if b == 0: return (a > 0) - (a < 0)
        if a == 0: return (b > 0) - (b < 0)
        if (a > 0) == (b > 0): return 1 if a > 0 else -1
        # opposite signs: compare a^2 vs 5 b^2
        d = a * a - 5 * b * b
        return (1 if a > 0 else -1) * ((d > 0) - (d < 0))
    def __repr__(s): return f'({s.a}+{s.b}r5)'
PHI = Q5(F(1, 2), F(1, 2))
def cross(u, v): return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
def dot(u, v): return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]
def sub(u, v): return [x-y for x, y in zip(u, v)]
def det3(a, b, c): return dot(a, cross(b, c))
def rid_q5():
    P = PHI
    base = [(Q5(1), Q5(1), P*P*P), (P*P, P, 2*P), (2+P, Q5(0), P*P)]
    out = []
    for (x, y, z) in base:
        for (u, v, w) in [(x, y, z), (y, z, x), (z, x, y)]:
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        p = (u*sx, v*sy, w*sz)
                        if not any(all(p[i] == q[i] for i in range(3)) for q in out): out.append(p)
    return out
