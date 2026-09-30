"""Rigorous membership test: box inside a certified singular ball.
Ball(X, ref-set, r): outer view (t2,v2) within Euclidean distance r of X's (t*,v*) AND the rotation R = Ain (D Aout Tref)^T
has angle <= r, for some D in {I, RZPI} and Tref in the reference set (Tb g, g in the rotation group; Tb = I for A, B).
Angle bound over the box: theta(x) <= theta(x0) + 5 eps (rotation rates <= 1)."""
import numpy as np, sys, pickle
sys.path.insert(0, '.')
from ia import I, sin, cos, dn, up
from rig_local2 import frame_I, matmul_I, const_I
from local2 import frame, rot_group, RZPI, rot_angle
from disprover import R2
from rid_setup import rid_z5
from dec_util import mat
EX = pickle.load(open('./exact_pts.pkl', 'rb'))
PF = rid_z5(); G = rot_group(PF)
from exact_pts import GE
SING = {}
for w, r in (('A', 1.5e-3), ('B', 1.5e-3), ('C', 1e-3), ('D', 1e-3)):
    vw = EX[w]; t = np.arctan2(float(vw['st']), float(vw['ct'])) % (2 * np.pi); v = np.arctan2(float(vw['sv']), float(vw['cv']))
    if w in ('A', 'B'): refs = [np.array([[float(x) for x in rr] for rr in ge]) for ge, _ in GE]; refsD = [ge for ge, _ in GE]
    else:
        Tb = EX['T' + w]; refsD = [mat(Tb, ge) for ge, _ in GE]; refs = [np.array([[float(x) for x in rr] for rr in m]) for m in refsD]
    SING[w] = dict(t=t, v=v, r=r, refs=refs, refsD=refsD)
from dec_util import encl as _encl
def _encl_mat(MD): return [[I(*_encl(x)) for x in row] for row in MD]
def in_ball(box):
    lo, hi = box[:, 0], box[:, 1]; c = (lo + hi) / 2
    eps = float(np.max(np.maximum(up(hi - c), up(c - lo))))
    for w, S in SING.items():
        # outer view distance (float with margin; t*,v* float approximations within 1e-15)
        dt = max(abs(lo[2] - S['t']), abs(hi[2] - S['t'])); dv = max(abs(lo[3] - S['v']), abs(hi[3] - S['v']))
        if np.hypot(dt, dv) + 1e-12 > S['r']: continue
        t1, v1, t2, v2, a = c
        Aout = frame(t2, v2); Ain = np.block([[R2(a), np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]) @ frame(t1, v1)
        best = None
        for j, Tf in enumerate(S['refs']):
            for Di, Dm in enumerate((np.eye(3), RZPI)):
                th = rot_angle(Ain @ (Dm @ Aout @ Tf).T)
                if best is None or th < best[0]: best = (th, j, Di)
        if best[0] + 5 * eps > S['r'] * 0.999: continue
        # rigorous angle at the centre
        _, j, Di = best
        AoI = frame_I(t2, v2); AiI0 = frame_I(t1, v1); ca, sa = cos(a), sin(a); Z = I(0.0); O = I(1.0)
        AiI = matmul_I([[ca, -sa, Z], [sa, ca, Z], [Z, Z, O]], AiI0)
        B = matmul_I(matmul_I(const_I((np.eye(3), RZPI)[Di]), AoI), _encl_mat(S['refsD'][j]))
        Bt = [[B[q][p] for q in range(3)] for p in range(3)]
        Rm = matmul_I(AiI, Bt); fro = None
        for p in range(3):
            for q in range(3):
                d = (Rm[p][q] - (1.0 if p == q else 0.0)).sqr(); fro = d if fro is None else fro + d
        y = float((fro.sqrt() * I(1 / np.sqrt(8) * (1 + 1e-15))).hi)
        if y >= 0.5: continue
        th_hi = 2 * y / np.sqrt(1 - y * y) * (1 + 1e-12)
        if th_hi + 5 * eps <= S['r'] * 0.999: return w
    return None
