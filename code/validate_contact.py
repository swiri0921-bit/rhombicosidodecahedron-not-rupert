"""Empirical validation (not part of the proof) of the per-contact lower bound used by the sector certificate:
at random configurations with s = r0 compare the true protrusion P_c (direct geometry) with the bound
   P_c / theta >= <om, g0 + r0 b> + c_r           (xi-mode, theta = |xi|)
   P_c / s     >= tau <om, g0 + r0 b> + c_r       (s-mode)
where c_r is the float value of the certificate's r0-row constant for a degenerate box (a single point)."""
import numpy as np, sys, pickle
sys.path.insert(0, '.')
from rig_sector import RigSector, EX
from rid_setup import rid_z5
from scipy.linalg import expm
V = rid_z5()
def frame(t, p): return np.array([[-np.sin(t), np.cos(t), 0], [-np.cos(t)*np.cos(p), -np.sin(t)*np.cos(p), np.sin(p)], [np.cos(t)*np.sin(p), np.sin(t)*np.sin(p), np.cos(p)]])
def skew(x): return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
def run(which, mode, r0, ntest, **kw):
    R = RigSector(which, mode=mode, r0=r0, **kw)
    vw = EX[which]; t0 = np.arctan2(float(vw['st']), float(vw['ct'])); p0 = np.arctan2(float(vw['sv']), float(vw['cv']))
    T = np.eye(3) if which in 'AB' else np.array([[float(x) for x in r] for r in EX['T' + which]])
    rng = np.random.default_rng(1); worst = np.inf; nchk = 0
    for _ in range(ntest):
        s = r0
        if rng.random() < 0.5:     # piece 1
            tau = 1.0; wh = rng.normal(size=2); wh *= rng.random() / np.linalg.norm(wh)
        else:
            tau = rng.random() * (1.0 if mode == 's' else 0.999) + (0 if mode == 's' else 0.001); wh = rng.normal(size=2); wh /= np.linalg.norm(wh)
        om = rng.normal(size=3); om /= np.linalg.norm(om)
        w = s * wh; xi = s * tau * om; th = s * tau
        A = frame(t0 + w[0], p0 + w[1]); Ain = expm(skew(xi)) @ A @ T
        Pin = (V @ Ain.T)[:, :2]; Pout = (V @ A.T)[:, :2]
        ev, bv, c0, cr, ends = R._float_rows(om, 0.0, wh - 1e-15, wh + 1e-15, tau, tau)
        for ci, c in enumerate(R.C):
            if not np.isfinite(cr[ci]): continue
            i, nx, ny, kind, val, K = c['meta']
            ang = s * val if kind == 'phi' else val * w[R.chidir]
            n = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.array([nx, ny])
            P = Pin[i] @ n - (Pout @ n).max()
            te = 1.0 if mode == 'xi' else tau
            bound = te * ((ev[ci] + r0 * bv[ci]) @ om) + cr[ci]
            val_ = P / (th if mode == 'xi' else s)
            worst = min(worst, val_ - bound); nchk += 1
    print(which, mode, 'checked', nchk, 'contact-configurations; min(true - bound) = %.3g' % worst, '(must be >= 0)', flush=True)
run('A', 'xi', 1.5e-3, 40)
run('B', 'xi', 1.5e-3, 40)
run('C', 's', 1e-3, 30, Vin_mat=EX['TC'])
run('D', 's', 1e-3, 30, Vin_mat=EX['TD'])
