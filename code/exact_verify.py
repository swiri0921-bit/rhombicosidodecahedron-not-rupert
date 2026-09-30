"""Exact Q(sqrt5) verification of the zero claims recorded by RigSector / RigTube."""
import numpy as np, sys, pickle
sys.path.insert(0, '.')
from q5 import Q5, cross, dot, sub
from exact_check import V, XB, W5, apply
I3 = [[Q5(int(i == j)) for j in range(3)] for i in range(3)]
def inner_pt(i, Tq, pm, smul):
    p = apply(Tq, V[pm[i]]); return [x * smul for x in p]
def zero_vec(u): return all(x.iszero() for x in u)
def verify(which, claims, Tq=None, pm=None, smul=1):
    Tq = Tq or I3; pm = pm if pm is not None else list(range(60)); X = XB[which]; bad = []
    for cl in claims:
        typ = cl[0]
        if typ in ('K', 'KI'):
            q, q2 = cl[2]; e = sub(V[q2], V[q])
            pt = V[cl[1]] if typ == 'K' else inner_pt(cl[1], Tq, pm, smul)
            if not dot(sub(pt, V[q]), cross(e, X)).iszero(): bad.append(cl)
        elif typ == 'Kx':
            v = cl[2][0]
            if not zero_vec(cross(sub(V[cl[1]], V[v]), X)): bad.append(cl)
        elif typ == 'KIx':
            v = cl[2][0]; pt = inner_pt(cl[1], Tq, pm, smul)
            if not zero_vec(cross(sub(pt, V[v]), X)): bad.append(cl)
        elif typ == 'Z':           # inner vertex orthogonal to the view (lies in the shadow plane)
            if not dot(inner_pt(cl[1], Tq, pm, smul), X).iszero(): bad.append(cl)
        elif typ == 'ID':          # identical trajectories: V_k == s * T V_i
            if not zero_vec(sub(V[cl[1]], inner_pt(cl[2], Tq, pm, smul))): bad.append(cl)
        elif typ == 'ARC':         # g_k(h) == 0 along the arc (views in span(w5, X)); line through V[q],V[q2]
            k, i, (q, q2) = cl[1], cl[2], cl[3]
            e = sub(V[q2], V[q]); d = sub(V[k], inner_pt(i, Tq, pm, smul))
            m1 = cross(W5, X); m2 = cross(X, m1)
            c1 = dot(m1, e) * dot(X, d); c2 = dot(m1, e) * dot(m2, d); c3 = dot(m2, e) * dot(m1, d)
            if not (c1.iszero() and c2.iszero() and c3.iszero()): bad.append(cl)
    return bad
