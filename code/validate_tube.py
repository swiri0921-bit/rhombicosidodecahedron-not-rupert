"""Empirical validation (not part of the proof) of the per-contact tube bound P_c/N >= Phi + h Psi - h^2 K at h = r0."""
import numpy as np, sys
sys.path.insert(0, '.')
from rig_tube import RigTube, EX
from dec_util import mat
from decimal import Decimal as D
from rid_setup import rid_z5
from scipy.linalg import expm
V = rid_z5()
def frame(t, p): return np.array([[-np.sin(t), np.cos(t), 0], [-np.cos(t)*np.cos(p), -np.sin(t)*np.cos(p), np.sin(p)], [np.cos(t)*np.sin(p), np.sin(t)*np.sin(p), np.cos(p)]])
def skew(x): return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
RZ = [[D(-1), D(0), D(0)], [D(0), D(-1), D(0)], [D(0), D(0), D(1)]]
def run(which, ref, ntest, r0=1e-3, eps0=0.4, beta=0.5):
    sgn = 1 if which == 'C' else -1
    Tm = EX['T' + which]; Mf = None
    if ref == 'ref2': Tm = mat(EX['TC'], EX['h2C']); Mf = RZ
    T = RigTube(which, Tm, sgn, Mfix=Mf, r0=r0, eps0=eps0, beta=beta)
    vw = EX[which]; t0 = np.arctan2(float(vw['st']), float(vw['ct'])); p0 = np.arctan2(float(vw['sv']), float(vw['cv']))
    Tf = np.array([[float(x) for x in r] for r in Tm]); MF = np.eye(3) if Mf is None else np.diag([-1., -1, 1])
    g0 = np.stack([(x.lo + x.hi) / 2 for x in T.G0], 1); g1 = np.stack([(x.lo + x.hi) / 2 for x in T.G1], 1)
    ui = np.stack([(x.lo + x.hi) / 2 for x in T.UI], 1); nn = np.stack([(x.lo + x.hi) / 2 for x in T.NN], 1); wn = (T.WN.lo + T.WN.hi) / 2
    rng = np.random.default_rng(2); worst = np.inf; nchk = 0
    for _ in range(ntest):
        h = r0; u = eps0 * h * (2 * rng.random() - 1); om = rng.normal(size=3); om /= np.linalg.norm(om); xi = beta * h * rng.random() * om
        N = np.linalg.norm(xi) + abs(u); tau = np.linalg.norm(xi) / N; sig = np.sign(u) if u != 0 else 1
        A = frame(t0 + u, p0 + sgn * h); Ain = expm(skew(xi)) @ MF @ A @ Tf
        Pin = (V @ Ain.T)[:, :2]; Pout = (V @ A.T)[:, :2]
        Dv = T.Dp if sig > 0 else T.Dm
        q = (ui @ om) * (nn @ om) - wn
        Phi = tau * (g0 @ om) - (1 - tau) * Dv
        Psi = tau * (g1 @ om) - tau * eps0 * T.rhoi + 0.5 * tau * beta * np.minimum(q, 0) - (1 - tau) * (T.MX + eps0 * T.MTT / 2)
        bound = Phi + h * Psi - h * h * T.K
        for ci, c in enumerate(T.C):
            i, nx, ny, phi, chi = c['meta']; ang = h * phi + u * chi
            n = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.array([nx, ny])
            P = Pin[i] @ n - (Pout @ n).max()
            worst = min(worst, P / N - bound[ci]); nchk += 1
    print(which, ref, 'checked', nchk, 'contact-configurations; min(true - bound) = %.3g' % worst, flush=True)
run('C', 'ref1', 12); run('C', 'ref2', 12); run('D', 'ref1', 12)
