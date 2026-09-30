"""Rigorous (interval) version of the sector blow-up certificate (sector2.py), built from exact (Decimal) data at x*.
Contacts (i, line j, phi) and (i, line j, chi); line normals n_j are the EXACT normals of projected edge lines at x*
(enclosed in intervals); active partners k are the vertices on the exact line (|value| < 1e-40 in 70-digit arithmetic;
structural zeros, see exact_check.py). All per-box inequalities are verified in interval arithmetic with float LP weights."""
import numpy as np, sys, os, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decimal import Decimal as D
from dec_util import mat, T, vec, encl
from ia import I, dn, up
from scipy.optimize import linprog
EX = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exact_pts.pkl'), 'rb'))
VZ = EX['VZ']; NV = 60
import os as _os; DEBUG = bool(_os.environ.get('SECDEBUG')); PHIMARGIN = float(_os.environ.get('PHIMARGIN', '0'))

def Ienc(x):
    lo, hi = encl(x); return I(lo, hi)
def Iarr(L):
    lo = np.array([encl(x)[0] for x in L]); hi = np.array([encl(x)[1] for x in L]); return I(lo, hi)
def ivec(v): return [Ienc(x) for x in v]
def icross(u, v): return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
def idot(u, v): return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
def ihi(x): return float(np.max(x.hi))
def mid(x): return (np.asarray(x.lo) + np.asarray(x.hi)) / 2

class RigSector:
    def __init__(self, which, Vin_mat=None, perm=None, Mfix=None, mode='xi', r0=1.5e-3,
                 fan=np.round(np.arange(-3, 3.001, 0.1), 10), chifan=None, chidir=0, ncone=16):
        vw = EX[which]; st, ct, sv, cv = vw['st'], vw['ct'], vw['sv'], vw['cv']
        A = [[-st, ct, D(0)], [-ct * cv, -st * cv, sv], [ct * sv, st * sv, cv]]
        At = [[-ct, -st, D(0)], [st * cv, -ct * cv, D(0)], [-st * sv, ct * sv, D(0)]]
        Ap = [[D(0), D(0), D(0)], [ct * sv, st * sv, cv], [ct * cv, st * cv, -sv]]
        self.mode = mode; self.r0 = r0; self.chidir = chidir
        # inner body in z5 coords: Vin_i = Tm V_perm(i)
        Tm = Vin_mat if Vin_mat is not None else [[D(int(i == j)) for j in range(3)] for i in range(3)]
        pm = perm if perm is not None else list(range(NV))
        MF = Mfix if Mfix is not None else [[D(int(i == j)) for j in range(3)] for i in range(3)]
        smul = int(MF[0][0])
        VIN = [vec(Tm, VZ[pm[i]]) for i in range(NV)]
        W = [vec(A, v) for v in VZ]; Jt = [vec(At, v) for v in VZ]; Jp = [vec(Ap, v) for v in VZ]
        U = [vec(MF, vec(A, v)) for v in VIN]; JUt = [vec(MF, vec(At, v)) for v in VIN]; JUp = [vec(MF, vec(Ap, v)) for v in VIN]
        Wf = np.array([[float(x) for x in w] for w in W]); Uf = np.array([[float(x) for x in u] for u in U])
        # exact coincidence check: inner shadow inside outer (support) at x*
        from scipy.spatial import ConvexHull
        Q = Wf[:, :2]; hv = ConvexHull(Q).vertices; m = len(hv)
        lines = []    # (n as Decimal pair or None, float n, isextra)
        self.claims = []; self.smul = smul; self.pm = pm; self.which = which
        for e in range(m):
            q, q2 = hv[e], hv[(e + 1) % m]
            dx = W[q2][0] - W[q][0]; dy = W[q2][1] - W[q][1]; L = (dx * dx + dy * dy).sqrt()
            nD = (dy / L, -dx / L); nf = np.array([float(nD[0]), float(nD[1])])
            if any(np.abs(nf - l[1]).max() < 1e-9 for l in lines): continue
            lines.append((nD, nf, False, (q, q2)))
        for vtx in hv:
            nb = [l[1] for l in lines if (Q[vtx] @ l[1]) > (Q @ l[1]).max() - 1e-9]
            if len(nb) != 2: continue
            a0, a1 = [np.arctan2(x[1], x[0]) for x in nb]; dd = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            for f in np.linspace(0, 1, ncone + 2)[1:-1]:
                th = a0 + f * dd; nf = np.array([np.cos(th), np.sin(th)])
                lines.append(((D(float(nf[0])), D(float(nf[1]))), nf, True, (vtx,)))
        rows = []
        for (nD, nf, isextra, ends) in lines:
            nx, ny = nD
            vals = [W[k][0] * nx + W[k][1] * ny for k in range(NV)]; H = max(vals)
            uvals = [U[i][0] * nx + U[i][1] * ny for i in range(NV)]
            assert max(uvals) - H < D(10) ** -40, 'inner protrudes at x*'
            K = [k for k in range(NV) if H - vals[k] < D(10) ** -40]
            KI = [i for i in range(NV) if H - uvals[i] < D(10) ** -40]
            if not isextra:
                self.claims += [('K', k, ends) for k in K] + [('KI', i, ends) for i in KI]
            else:
                assert ends[0] in K
                self.claims += [('Kx', k, ends) for k in K if k != ends[0]] + [('KIx', i, ends) for i in KI]
            inact = [k for k in range(NV) if k not in K]
            dl = min(H - vals[k] for k in inact)
            nI = [Ienc(nx), Ienc(ny)]; N3 = [nI[0], nI[1], I(0.0)]; Np3 = [-nI[1], nI[0], I(0.0)]
            for i in KI:
                Ui = ivec(U[i]); JUti = ivec(JUt[i]); JUpi = ivec(JUp[i])
                g0 = icross(Ui, N3); gwt = icross(JUti, N3); gwp = icross(JUpi, N3); gp = icross(Ui, Np3)
                zflag = abs(U[i][2]) < D(10) ** -45       # inner vertex in the shadow plane: exact claim 'Z'
                if zflag:
                    self.claims.append(('Z', i)); g0 = [I(0.0), I(0.0), g0[2]]
                aw = []; ap = []; dk = []
                for k in K:
                    dvec0 = [VZ[k][j] - smul * VIN[i][j] for j in range(3)]
                    if all(abs(x) < D(10) ** -45 for x in dvec0):     # identical trajectories: W_k == U_i identically
                        self.claims.append(('ID', k, i))
                        aw.append([I(0.0), I(0.0)]); ap.append(I(0.0)); dk.append(0.0); continue
                    aw.append([(Ienc(Jt[k][0] - JUt[i][0]) * nI[0] + Ienc(Jt[k][1] - JUt[i][1]) * nI[1]),
                               (Ienc(Jp[k][0] - JUp[i][0]) * nI[0] + Ienc(Jp[k][1] - JUp[i][1]) * nI[1])])
                    ap.append(Ienc(W[k][0] - U[i][0]) * Np3[0] + Ienc(W[k][1] - U[i][1]) * Np3[1])
                    if smul == 1 and Mfix is None:
                        dvec = [VZ[k][j] - VIN[i][j] for j in range(3)]
                    else:
                        dvec = [VZ[k][j] - smul * VIN[i][j] for j in range(3)]
                    dk.append(float(up(float((sum(x * x for x in dvec)).sqrt()))))
                # inactive partner derivative magnitude bounds (float, rounded up generously)
                am_ = 0.0
                for k in inact:
                    a1 = abs(float(Jt[k][0] - JUt[i][0]) * nf[0] + float(Jt[k][1] - JUt[i][1]) * nf[1])
                    a2 = abs(float(Jp[k][0] - JUp[i][0]) * nf[0] + float(Jp[k][1] - JUp[i][1]) * nf[1])
                    a3 = abs(float(W[k][0] - U[i][0]) * (-nf[1]) + float(W[k][1] - U[i][1]) * nf[0])
                    am_ = max(am_, (a1 + a2) * np.sqrt(2) * 1.01 + 1e-12, a3 * 1.01)
                am_act = 0.0
                for x, a_ in zip(aw, ap):   # active partners: derivative AND distance along the line (rotation term)
                    am_act = max(am_act, (abs(float(ihi(x[0].abs()))) + abs(float(ihi(x[1].abs())))) * np.sqrt(2), abs(float(ihi(a_.abs()))) * 1.01)
                variants = [('phi', p) for p in ([0.0] if isextra else fan)]
                if chifan is not None and not isextra: variants += [('chi', c_) for c_ in chifan if c_ != 0]
                for kind, val in variants:
                    rphi = abs(val) if kind == 'phi' else 0.0; rchi = abs(val) if kind == 'chi' else 0.0
                    fac_max = np.sqrt(2) + rphi + rchi
                    # inactive vertices never reach the active level (rigorous, conservative):
                    lhs = r0 * (am_ + am_act) * (1 + rphi + rchi) + r0 * r0 * 4.0 * fac_max ** 2
                    if not (lhs < float(dl) * 0.999): continue
                    GW = [[gwt[r], gwp[r]] for r in range(3)]
                    GP = [gp[r] * I(val) if kind == 'phi' else I(0.0) for r in range(3)]
                    AW = [[x[0], x[1]] for x in aw]
                    AP = [a * I(val) if kind == 'phi' else I(0.0) for a in ap]
                    if kind == 'chi':
                        GW = [[GW[r][0] + (gp[r] * I(val) if chidir == 0 else I(0.0)), GW[r][1] + (gp[r] * I(val) if chidir == 1 else I(0.0))] for r in range(3)]
                        AW = [[x[0] + (a * I(val) if chidir == 0 else I(0.0)), x[1] + (a * I(val) if chidir == 1 else I(0.0))] for x, a in zip(AW, ap)]
                    rows.append(dict(meta=(i, float(nx), float(ny), kind, float(val), list(K), str(nx), str(ny), isextra), g0=g0, GW=GW, GP=GP, AW=AW, AP=AP, Ui=Ui, N=N3, WN=idot(Ui, N3), dk=np.array(dk), Z=zflag,
                                     rphi=rphi, rchi=rchi))
        self.C = rows
        self._pack()

    def _pack(self):
        C = self.C; n = len(C); ka = max(len(c['AP']) for c in C)
        def stack(get, shape):
            lo = np.zeros((n,) + shape); hi = np.zeros((n,) + shape)
            for ci, c in enumerate(C):
                for idx in itertools.product(*[range(s) for s in shape]):
                    v = get(c, idx); lo[(ci,) + idx] = float(v.lo); hi[(ci,) + idx] = float(v.hi)
            return lo, hi
        self.G0 = I(*stack(lambda c, ix: c['g0'][ix[0]], (3,)))
        self.GW = I(*stack(lambda c, ix: c['GW'][ix[0]][ix[1]], (3, 2)))
        self.GP = I(*stack(lambda c, ix: c['GP'][ix[0]], (3,)))
        self.Ui = I(*stack(lambda c, ix: c['Ui'][ix[0]], (3,)))
        self.Nn = I(*stack(lambda c, ix: c['N'][ix[0]], (3,)))
        self.WN = I(*stack(lambda c, ix: c['WN'], ()))
        AWlo = np.zeros((n, ka, 2)); AWhi = np.zeros((n, ka, 2)); APlo = np.zeros((n, ka)); APhi = np.zeros((n, ka))
        self.AV = np.zeros((n, ka), bool); self.DK = np.zeros((n, ka))
        for ci, c in enumerate(C):
            for kk, (x, a) in enumerate(zip(c['AW'], c['AP'])):
                AWlo[ci, kk] = [float(x[0].lo), float(x[1].lo)]; AWhi[ci, kk] = [float(x[0].hi), float(x[1].hi)]
                APlo[ci, kk] = float(a.lo); APhi[ci, kk] = float(a.hi); self.AV[ci, kk] = True; self.DK[ci, kk] = c['dk'][kk]
        self.AW = I(AWlo, AWhi); self.AP = I(APlo, APhi)
        self.RPHI = np.array([c['rphi'] for c in C]); self.RCHI = np.array([c['rchi'] for c in C])
        self.Z = np.array([c['Z'] for c in C])

    # ---------- float model for LP weights ----------
    def _float_rows(self, om_c, om_rad, wl, wh, tl, th):
        G0 = mid(self.G0); GW = mid(self.GW); GP = mid(self.GP); Ui = mid(self.Ui); Nn = mid(self.Nn); WN = mid(self.WN)
        AW = mid(self.AW); AP = mid(self.AP); r0 = self.r0
        whc = (wl + wh) / 2; whr = (wh - wl) / 2
        w1 = np.maximum(np.abs(wl), np.abs(wh)).sum(); wt = max(abs(wl[self.chidir]), abs(wh[self.chidir]))
        fac = w1 + self.RPHI + self.RCHI * wt
        CG = 0.5 * fac ** 2; CQ = fac; E3 = 0.5 * self.DK * (fac ** 2)[:, None]
        al = np.einsum('cka,a->ck', AW, whc) + np.abs(AW) @ whr + AP
        al0 = np.where(self.AV, al, -np.inf).max(1); alr = np.where(self.AV, al + r0 * E3, -np.inf).max(1)
        bvec = np.einsum('cda,a->cd', GW, whc) + GP
        bsl = np.linalg.norm(GW, axis=(1, 2)) * np.linalg.norm(whr)
        qv = (Ui @ om_c) * (Nn @ om_c) - WN; qs = 2 * om_rad * np.linalg.norm(Ui, axis=1)
        if self.mode == 'xi':
            pen = lambda G: -np.where(np.maximum(G, 0) > 0, np.maximum(G, 0) / tl if tl > 0 else np.inf, 0.0)
            c0 = pen(al0); cr = pen(alr) + r0 * (-bsl + np.minimum(tl * (qv - qs), th * (qv - qs)) / 2) - r0 ** 2 * (CG + th * CQ / 2 + th ** 2 / 6)
            return G0, bvec, c0, cr, [1.0]
        c0 = -al0; cr = -alr + r0 * (-bsl * th + np.minimum(tl ** 2 * (qv - qs), th ** 2 * (qv - qs)) / 2) - r0 ** 2 * (th * CG + th ** 2 * CQ / 2 + th ** 3 / 6)
        return G0, bvec, c0, cr, [tl, th]

    def _lp(self, om_c, om_rad, wl, wh, tl, th, margin=0.0):
        evec, bvec, c0, cr, ends = self._float_rows(om_c, om_rad, wl, wh, tl, th)
        ok = np.isfinite(c0) & np.isfinite(cr)
        if not ok.any(): return None
        idx = np.where(ok)[0]; evec, bvec, c0, cr = evec[idx], bvec[idx], c0[idx], cr[idx]
        r0 = self.r0; n = len(c0); nv = n + 6 * len(ends) + 1; rows = []; bub = []
        for ei, te in enumerate(ends):
            base = n + 6 * ei; Hv = evec + r0 * bvec
            r = np.zeros(nv); r[:n] = -(te * (evec @ om_c) + c0); r[base:base + 3] = te * om_rad; rows.append(r); bub.append(-margin)
            r = np.zeros(nv); r[:n] = -(te * (Hv @ om_c) + cr); r[base + 3:base + 6] = te * om_rad; r[-1] = 1; rows.append(r); bub.append(0.0)
            for d in range(3):
                for sg in (1, -1):
                    r = np.zeros(nv); r[:n] = sg * evec[:, d]; r[base + d] = -1; rows.append(r); bub.append(0.0)
                    r = np.zeros(nv); r[:n] = sg * Hv[:, d]; r[base + 3 + d] = -1; rows.append(r); bub.append(0.0)
        c = np.zeros(nv); c[-1] = -1; Aeq = np.zeros((1, nv)); Aeq[0, :n] = 1
        res = linprog(c, A_ub=np.array(rows), b_ub=np.array(bub), A_eq=Aeq, b_eq=[1],
                      bounds=[(0, None)] * (nv - 1) + [(None, None)], method='highs')
        if res.status != 0 or -res.fun <= 0: return None
        lam = np.zeros(len(self.C)); lam[idx] = np.maximum(res.x[:n], 0)
        return lam

    # ---------- rigorous verification ----------
    def _snap(self, lam):
        """Degenerate first order: support {p, q} with g0 = (0,0,a), (0,0,-b) exactly (claims 'Z'). Exact weights
        (b, a)/(a+b) (enclosed) give sum lam* g0 == 0 exactly."""
        S = np.where(lam > 1e-14)[0]
        if len(S) != 2 or not self.Z[S].all(): return None
        z = self.G0[:, 2]; a = I(z.lo[S[0]], z.hi[S[0]]); b = I(z.lo[S[1]], z.hi[S[1]])
        if a.lo > 0 and b.hi < 0: b = I(0.0) - b
        elif a.hi < 0 and b.lo > 0: a = I(0.0) - a
        else: return None
        s = a + b
        l1 = I(dn(b.lo / s.hi), up(b.hi / s.lo)); l2 = I(dn(a.lo / s.hi), up(a.hi / s.lo))
        return S, I(np.array([float(l1.lo), float(l2.lo)]), np.array([float(l1.hi), float(l2.hi)]))

    def certify(self, om_c, om_rad, wl, wh, tl, th):
        # first try weights with a small first-order margin (robust to rounding), then the exact-zero route
        lam = self._lp(om_c, om_rad, wl, wh, tl, th, margin=PHIMARGIN) if PHIMARGIN > 0 else None
        if lam is not None:
            S = np.where(lam > 1e-14)[0]
            if len(S) and self._cert(lam, S, I(lam[S]), None, om_c, om_rad, wl, wh, tl, th): self.last = (list(S), I(lam[S]), False); return True
        lam = self._lp(om_c, om_rad, wl, wh, tl, th)
        if lam is None: return False
        S = np.where(lam > 1e-14)[0]
        if len(S) == 0: return False
        if self._cert(lam, S, I(lam[S]), None, om_c, om_rad, wl, wh, tl, th): self.last = (list(S), I(lam[S]), False); return True
        sn = self._snap(lam)
        if sn is not None and self._cert(lam, sn[0], sn[1], True, om_c, om_rad, wl, wh, tl, th): self.last = (list(sn[0]), sn[1], True); return True
        # support = opposite-signed Z pair + small extra weights: try the exact-zero pair alone (heaviest pairs first)
        S = np.where(lam > 1e-14)[0]; SZ = [c for c in S if self.Z[c]]
        if len(SZ) >= 2 and len(SZ) < len(S) + 1:
            z = mid(self.G0)[:, 2]
            pairs = sorted(((p_, q_) for p_ in SZ for q_ in SZ if z[p_] > 0 > z[q_]), key=lambda t: -(lam[t[0]] + lam[t[1]]))
            for p_, q_ in pairs[:4]:
                lam2 = np.zeros_like(lam); lam2[p_] = lam[p_]; lam2[q_] = lam[q_]
                sn = self._snap(lam2)
                if sn is not None and self._cert(lam2, sn[0], sn[1], True, om_c, om_rad, wl, wh, tl, th): self.last = (list(sn[0]), sn[1], True); return True
        # LP solver tolerance (~1e-7) may return weights with a slightly negative first-order value:
        # re-solve with an explicit first-order margin (weights are only a guess; _cert is rigorous)
        for mg in (1e-8, 1e-7, 1e-6):
            lam = self._lp(om_c, om_rad, wl, wh, tl, th, margin=mg)
            if lam is None: return False
            S = np.where(lam > 1e-14)[0]
            if len(S) and self._cert(lam, S, I(lam[S]), None, om_c, om_rad, wl, wh, tl, th): self.last = (list(S), I(lam[S]), False); return True
        return False

    def _cert(self, lam, S, L, snapped, om_c, om_rad, wl, wh, tl, th):
        r0 = I(self.r0); R0 = self.r0
        sub = lambda X: I(X.lo[S], X.hi[S])
        G0 = sub(self.G0); GW = sub(self.GW); GP = sub(self.GP); Ui = sub(self.Ui); Nn = sub(self.Nn); WN = sub(self.WN)
        AW = sub(self.AW); AP = sub(self.AP); AV = self.AV[S]; DK = self.DK[S]
        wl = np.asarray(wl, float); wh = np.asarray(wh, float); whc = (wl + wh) / 2
        whr = np.maximum(up(wh - whc), up(whc - wl))
        w1 = float(up(np.maximum(np.abs(wl), np.abs(wh)).sum() * (1 + 1e-15)))
        wt = max(abs(wl[self.chidir]), abs(wh[self.chidir]))
        fac = I(w1) + I(self.RPHI[S]) + I(self.RCHI[S]) * I(wt)
        fac2 = fac * fac
        NSL = I(1.0 + 1e-12)       # |N| <= 1 + 1e-15 for the extra (vertex-cone) directions
        CG = I(0.5) * fac2 * NSL; CQ = fac * NSL
        # alpha upper bound over the wh-box: AW . whc + |AW| . whr + AP
        whcI = [I(whc[0]), I(whc[1])]; whrI = [I(whr[0]), I(whr[1])]
        AW0 = I(AW.lo[..., 0], AW.hi[..., 0]); AW1 = I(AW.lo[..., 1], AW.hi[..., 1])
        al = AW0 * whcI[0] + AW1 * whcI[1] + AW0.abs() * whrI[0] + AW1.abs() * whrI[1] + AP
        E3 = I(0.5) * I(DK) * I(fac2.lo[:, None], fac2.hi[:, None]) * I(1.0 + 1e-12)
        alr = al + r0 * E3
        G0u = np.where(AV, al.hi, -np.inf).max(1); Gru = np.where(AV, alr.hi, -np.inf).max(1)     # upper bounds of maxima
        # bvec interval
        GWt = I(GW.lo[..., 0], GW.hi[..., 0]); GWp = I(GW.lo[..., 1], GW.hi[..., 1])
        bvec = GWt * whcI[0] + GWp * whcI[1] + GP                    # (n,3)
        gwn = (GW.abs().sqr()); gwn = I(gwn.lo.sum((1, 2)), gwn.hi.sum((1, 2))).sqrt()
        bsl = gwn * I(float(up(np.sqrt(whr[0] ** 2 + whr[1] ** 2) * (1 + 1e-15))))
        omc = [I(om_c[0]), I(om_c[1]), I(om_c[2])]
        ui = [I(Ui.lo[:, r], Ui.hi[:, r]) for r in range(3)]; nn = [I(Nn.lo[:, r], Nn.hi[:, r]) for r in range(3)]
        qv = (ui[0] * omc[0] + ui[1] * omc[1] + ui[2] * omc[2]) * (nn[0] * omc[0] + nn[1] * omc[1] + nn[2] * omc[2]) - WN
        uin = (ui[0].sqr() + ui[1].sqr() + ui[2].sqr()).sqrt()
        qs = I(2.0) * I(om_rad) * uin * I(1.0 + 1e-12)
        qlo_v = qv - qs
        tlI, thI = I(tl), I(th)
        if self.mode == 'xi':
            if tl <= 0 and (np.maximum(G0u, 0) > 0).any(): return False
            if tl <= 0 and (np.maximum(Gru, 0) > 0).any(): return False
            if tl > 0:
                inv = float(up(1.0 / tl))
                p0 = I(-np.maximum(G0u, 0) * inv * (1 + 1e-15)); pr = I(-np.maximum(Gru, 0) * inv * (1 + 1e-15))
            else:
                p0 = I(np.zeros(len(S))); pr = I(np.zeros(len(S)))
            qterm = I(np.minimum((tlI * qlo_v).lo, (thI * qlo_v).lo)) * I(0.5)
            c0 = p0
            cr = pr + r0 * (I(0.0) - bsl + qterm) - r0 * r0 * (CG + thI * CQ * I(0.5) + thI * thI * I((1.0 + R0 / 4) / 6 * (1 + 1e-12)))
            ends = [1.0]
        else:
            c0 = I(-G0u)
            qterm = I(np.minimum((tlI * tlI * qlo_v).lo, (thI * thI * qlo_v).lo)) * I(0.5)
            cr = I(-Gru) + r0 * (I(0.0) - bsl * thI + qterm) - r0 * r0 * (thI * CG + thI * thI * CQ * I(0.5) + thI * thI * thI * I((1.0 + R0 / 4) / 6 * (1 + 1e-12)))
            ends = [tl, th]
        ev = [I(G0.lo[:, r], G0.hi[:, r]) for r in range(3)]
        bv = [I(bvec.lo[:, r], bvec.hi[:, r]) for r in range(3)]
        def comb(x):   # rigorous enclosure of sum_c lam_c x_c (float summation error <= n u sum|x|)
            p = L * x; m = len(S) * 2.3e-16
            return I(float(dn(p.lo.sum() - m * np.abs(p.lo).sum())), float(up(p.hi.sum() + m * np.abs(p.hi).sum())))
        if snapped:     # sum lam* g0 == 0 exactly, hence E = 0 and Hs = r0 sum lam* bvec
            E = [I(0.0)] * 3; Hs = [comb(r0 * b) for b in bv]
            if not (G0u <= 0).all():
                if DEBUG: print('G0u', G0u)
                return False     # then c0 >= 0 exactly (both modes), so Phi = <om,0> + c0 >= 0
        else:
            E = [comb(e) for e in ev]; Hs = [comb(e + r0 * b) for e, b in zip(ev, bv)]
        C0 = comb(c0); CR = comb(cr)
        def lowdot(vecI, te):   # lower bound of te*<omega, v> over the cap: te*(<om_c, v> - rad*|v|_2)
            d = vecI[0] * omc[0] + vecI[1] * omc[1] + vecI[2] * omc[2]
            nrm = (vecI[0].sqr() + vecI[1].sqr() + vecI[2].sqr()).sqrt()
            return I(te) * (d - I(om_rad) * nrm)
        for te in ends:
            if not snapped and not float((lowdot(E, te) + C0).lo) >= 0: return False
            if not float((lowdot(Hs, te) + CR).lo) > 0:
                if DEBUG: print('row2', lowdot(Hs, te), CR, Hs, 'C0', C0)
                return False
        return True
