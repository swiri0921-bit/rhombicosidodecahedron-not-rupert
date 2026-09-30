"""Sanity check (not part of the proof): random configurations near the singular points must not be passages.
For each sample compute max_i dist(p_i, outer hull) > 0 in float64 (protrusion ~ s^2 >> 1e-16)."""
import numpy as np, sys, pickle, json
sys.path.insert(0, '.')
from rid_setup import rid_z5
from scipy.spatial import ConvexHull
from scipy.linalg import expm
EX = pickle.load(open('./exact_pts.pkl', 'rb'))
V = rid_z5()
def frame(t, p): return np.array([[-np.sin(t), np.cos(t), 0], [-np.cos(t)*np.cos(p), -np.sin(t)*np.cos(p), np.sin(p)], [np.cos(t)*np.sin(p), np.sin(t)*np.sin(p), np.cos(p)]])
def skew(x): return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
def protrusion(Pin, Pout):
    h = ConvexHull(Pout); eq = h.equations          # a.x + b <= 0 inside
    return (Pin @ eq[:, :2].T + eq[:, 2]).max(1).max()   # >0 : some inner point outside
rng = np.random.default_rng(0)
for w, r0 in (('A', 1.5e-3), ('B', 1.5e-3), ('C', 1e-3), ('D', 1e-3)):
    vw = EX[w]; t0 = np.arctan2(float(vw['st']), float(vw['ct'])); p0 = np.arctan2(float(vw['sv']), float(vw['cv']))
    T = np.eye(3) if w in 'AB' else np.array([[float(x) for x in r] for r in EX['T' + w]])
    mn = np.inf; neg = 0
    for _ in range(20000):
        s = r0 * rng.random() ** 0.5
        wv = rng.normal(size=2); xv = rng.normal(size=3)
        if rng.random() < 0.5: wv *= s / np.linalg.norm(wv); xv *= s * rng.random() / np.linalg.norm(xv)
        else: xv *= s / np.linalg.norm(xv); wv *= s * rng.random() / np.linalg.norm(wv)
        A = frame(t0 + wv[0], p0 + wv[1]); Ain = expm(skew(xv)) @ A @ T
        P = protrusion((V @ Ain.T)[:, :2], (V @ A.T)[:, :2])
        mn = min(mn, P / s ** 2); neg += P <= 0
    print(w, 'samples 20000, non-positive protrusions:', neg, ' min protrusion/s^2 = %.3g' % mn, flush=True)
