"""Rigorous arc-tube certificate (interval arithmetic, exact Decimal data at the arc endpoint x*).
Tube: w = (u, sgn*h), 0 < h <= r0, |u| <= eps0 h, |xi| <= beta h (xi relative to the reference inner body
U(w) = Mfix A(w) Vin).  N = |xi| + |u|, tau = |xi|/N, sigma = sign u, omega = xi/|xi|.
Contact (i, j, phi, chi): n = R(h phi + u chi) n_j.  Exact partners k: g_k(0,h) == 0 on the arc (identical shadow
trajectories, or collinear along the arc: certified in exact arithmetic).  Non-exact partners must stay below
the exact ones throughout the tube (checked with margins).
   P/N >= Phi + h Psi - h^2 K,  Phi = tau<om,g0> - (1-tau) D_sigma,
   Psi = tau<om,g1> - tau eps0 |V_i|(1+|chi|) + 1/2 tau beta min(q,0) - (1-tau)(MX + eps0 MTT/2),
   K   = 1/2 fac^2 + beta(1+eps0)(1+|phi|+|chi|) + beta^2/6 (1+beta r0/4)   [|R3| <= t^3/6 + t^4/24],   fac = 1 + eps0 + |phi| + eps0|chi|.
Concave in h: need Phi >= 0 and Phi + r0 Psi - r0^2 K > 0 over the (omega, tau, sigma) box."""
import numpy as np, sys, pickle
sys.path.insert(0, '.')
from decimal import Decimal as D
from dec_util import vec, encl
from ia import I, dn, up
from rig_sector import EX, VZ, NV, Ienc, ivec, icross, idot
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

class RigTube:
    def __init__(self, which, Tm, sgn, Mfix=None, r0=1e-3, eps0=0.4, beta=0.5,
                 fan=np.round(np.arange(-3, 3.001, 0.1), 10), chifan=np.round(np.arange(-3, 3.001, 0.1), 10)):
        vw = EX[which]; st, ct, sv, cv = vw['st'], vw['ct'], vw['sv'], vw['cv']
        A = [[-st, ct, D(0)], [-ct * cv, -st * cv, sv], [ct * sv, st * sv, cv]]
        At = [[-ct, -st, D(0)], [st * cv, -ct * cv, D(0)], [-st * sv, ct * sv, D(0)]]
        Ap = [[D(0), D(0), D(0)], [ct * sv, st * sv, cv], [ct * cv, st * cv, -sv]]
        self.r0, self.eps0, self.beta = r0, eps0, beta; self.sep_disagree = 0
        MF = Mfix if Mfix is not None else [[D(int(i == j)) for j in range(3)] for i in range(3)]
        smul = int(MF[0][0])
        VIN = [vec(Tm, v) for v in VZ]
        W = [vec(A, v) for v in VZ]; Jt = [vec(At, v) for v in VZ]; Jv = [[sgn * x for x in vec(Ap, v)] for v in VZ]
        U = [vec(MF, vec(A, v)) for v in VIN]; JUt = [vec(MF, vec(At, v)) for v in VIN]; JUv = [[sgn * x for x in vec(MF, vec(Ap, v))] for v in VIN]
        self.claims = []
        Wf = np.array([[float(x) for x in w] for w in W]); Q = Wf[:, :2]
        hv = ConvexHull(Q).vertices; m = len(hv); lines = []
        for e in range(m):
            q, q2 = hv[e], hv[(e + 1) % m]
            dx = W[q2][0] - W[q][0]; dy = W[q2][1] - W[q][1]; L = (dx * dx + dy * dy).sqrt()
            nD = (dy / L, -dx / L); nf = np.array([float(nD[0]), float(nD[1])])
            if not any(np.abs(nf - l[1]).max() < 1e-9 for l in lines): lines.append((nD, nf, (q, q2)))
        # arc exactness test data: frames at a few arc points (Decimal) for collinear partners
        rows = []; eps = D(10) ** -40
        for (nD, nf, ends) in lines:
            nx, ny = nD; nI = [Ienc(nx), Ienc(ny)]; N3 = [nI[0], nI[1], I(0.0)]; Np3 = [-nI[1], nI[0], I(0.0)]
            vals = [W[k][0] * nx + W[k][1] * ny for k in range(NV)]; H = max(vals)
            uvals = [U[i][0] * nx + U[i][1] * ny for i in range(NV)]
            assert max(uvals) - H < eps
            KI = [i for i in range(NV) if H - uvals[i] < eps]
            self.claims += [('KI', i, ends) for i in KI]
            for i in KI:
                Ui = ivec(U[i]); JUvi = ivec(JUv[i])
                g0 = icross(Ui, N3)
                zflag = abs(U[i][2]) < D('1e-45')          # inner vertex in the shadow plane (exact claim 'Z')
                if zflag:
                    self.claims.append(('Z', i)); g0 = [I(0.0), I(0.0), g0[2]]
                for phi in fan:
                    exact = []; bad = False; nonexact = []
                    for k in range(NV):
                        dvec = [VZ[k][j] - smul * VIN[i][j] for j in range(3)]
                        dist = float((sum(x * x for x in dvec)).sqrt())
                        g0k = (W[k][0] - U[i][0]) * nx + (W[k][1] - U[i][1]) * ny            # Decimal, <= 0
                        alpha = float((Jv[k][0] - JUv[i][0]) * nx + (Jv[k][1] - JUv[i][1]) * ny) + phi * float(-(W[k][0] - U[i][0]) * ny + (W[k][1] - U[i][1]) * nx)
                        dktD = (Jt[k][0] - JUt[i][0]) * nx + (Jt[k][1] - JUt[i][1]) * ny
                        pkD = -(W[k][0] - U[i][0]) * ny + (W[k][1] - U[i][1]) * nx
                        dkt = float(dktD); pk = float(pkD)
                        identical = dist < 1e-30
                        collinear_arc = (not identical) and abs(g0k) < eps and abs(alpha) < 1e-30 and phi == 0 and self._arc_zero(which, k, i, Tm, MF, sgn, nD)
                        if identical or collinear_arc:
                            exact.append((k, dkt, pk, dist, dktD, pkD))
                            self.claims.append(('ID', k, i) if identical else ('ARC', k, i, ends))
                        else:
                            alphaD = (Jv[k][0] - JUv[i][0]) * nx + (Jv[k][1] - JUv[i][1]) * ny + D(float(phi)) * (-(W[k][0] - U[i][0]) * ny + (W[k][1] - U[i][1]) * nx)
                            nonexact.append((float(g0k), alpha, dkt, dist, pk, (g0k, alphaD, dktD, (sum(x * x for x in dvec)).sqrt(), pkD)))
                    if not exact: continue
                    for chi in chifan:
                        Dk = np.array([e[1] + chi * e[2] for e in exact])
                        # sigma-maxima from the 60-digit data (error < 1e-50), rounded up to float
                        chiD = D(float(chi))   # exact binary value of chi
                        DkD = [e[4] + chiD * e[5] for e in exact]
                        DpD = max(DkD) + D('1e-45'); DmD = max(-x for x in DkD) + D('1e-45')
                        hasid = any(e[3] < 1e-30 for e in exact)      # identical partner => D_k == 0 exactly
                        tiny = D('1e-45')
                        Dp0 = hasid and all(e[3] < 1e-30 or x < -tiny for e, x in zip(exact, DkD))
                        Dm0 = hasid and all(e[3] < 1e-30 or x > tiny for e, x in zip(exact, DkD))
                        Dmax = np.abs(Dk).max()
                        fac = 1 + eps0 + abs(phi) + eps0 * abs(chi)
                        ok = True
                        maxdist0 = max(e[3] for e in exact)
                        # |e_k(h,u)| <= |u| (|D_k| + h (MX + eps0 MTT/2)) for exact partners (same bound as in Phi/Psi);
                        # non-exact partners must stay below min_e <W_e,n>, i.e. g_k/h + margin < 0 with |u| <= eps0 h
                        margin = eps0 * (Dmax + r0 * (maxdist0 * (1 + abs(phi)) * (1 + abs(chi)) + eps0 * maxdist0 * (1 + abs(chi)) ** 2 / 2)) * 1.0001
                        # separation of non-exact vertices: decided in 60-digit decimal arithmetic (error < 1e-50,
                        # thresholds 1e-12 / 1e-13); the float evaluation is kept only to count disagreements
                        chiA = D(float(chi)); r0D = D(float(r0)); e0D = D(float(eps0)); phD = abs(D(float(phi)))
                        facD = 1 + e0D + phD + e0D * abs(chiA)
                        DmaxD = max(abs(e[4] + chiA * e[5]) for e in exact)
                        md0 = max((sum(x * x for x in [VZ[e[0]][jj] - smul * VIN[i][jj] for jj in range(3)])).sqrt() for e in exact)
                        marginD = e0D * (DmaxD + r0D * (md0 * (1 + phD) * (1 + abs(chiA)) + e0D * md0 * (1 + abs(chiA)) ** 2 / 2)) * D('1.0001')
                        for (g0k, alpha, dkt, dist, pk, dec) in nonexact:
                            dd = abs(dkt) + abs(chi) * abs(pk)
                            if g0k > -1e-30:
                                okf = alpha + r0 * 0.5 * dist * fac * fac + eps0 * dd + margin < -1e-12
                            else:
                                okf = g0k + r0 * (abs(alpha) + r0 * 0.5 * dist * fac * fac + eps0 * dd + margin) < -1e-13
                            gD, aD, tD, distD, pD = dec
                            ddD = abs(tD) + abs(chiA) * abs(pD)
                            if gD > D('-1e-30'):
                                okd = aD + r0D * D('0.5') * distD * facD * facD + e0D * ddD + marginD < D('-1e-12')
                            else:
                                okd = gD + r0D * (abs(aD) + r0D * D('0.5') * distD * facD * facD + e0D * ddD + marginD) < D('-1e-13')
                            if okf != okd: self.sep_disagree += 1
                            if not okd: ok = False; break
                        if not ok: continue
                        maxdist = max(e[3] for e in exact)
                        g1 = [icross(JUvi, N3)[r] + icross(Ui, Np3)[r] * I(phi) for r in range(3)]
                        rows.append(dict(meta=(i, float(nx), float(ny), float(phi), float(chi), str(nx), str(ny), [e[0] for e in exact]), g0=g0, g1=g1, Dp=0.0 if Dp0 else float(up(float(DpD))), Dm=0.0 if Dm0 else float(up(float(DmD))),
                                         Dp0=Dp0, Dm0=Dm0, Z=zflag,
                                         MX=maxdist * (1 + abs(phi)) * (1 + abs(chi)) * 1.0001, MTT=maxdist * (1 + abs(chi)) ** 2 * 1.0001,
                                         Ui=Ui, N=N3, WN=idot(Ui, N3), rhoi=(1 + abs(chi)),
                                         K=(0.5 * fac * fac + beta * (1 + eps0) * (1 + abs(phi) + abs(chi)) + beta * beta / 6 * (1 + beta * r0 / 4) * 1.000001) * (1 + 1e-12)))   # float sum: relative factor covers rounding
        self.C = rows; self._pack()

    _arc_cache = {}
    def _arc_zero(self, which, k, i, Tm, MF, sgn, nD):
        """exactness of g_k along the arc: collinearity persists for all arc views <=> the 3D vector
        V_k - s Vin_i is parallel to the mirror plane normal direction... certified numerically at 70 digits
        at 3 arc points (structural identity; exact Q(sqrt5) certificate in exact_check.py)."""
        from dec_util import dsin, dcos
        vw = EX[which]
        # arc: outer view (t*, v* + sgn h); check g(h) = <Pi A(h)(V_k - s Vin_i), n(h)> with n(h) the rotated line normal? (phi=0 -> n fixed)
        smul = int(MF[0][0]); VIN_i = vec(Tm, VZ[i]); dv = [VZ[k][j] - smul * VIN_i[j] for j in range(3)]
        st, ct, sv, cv = vw['st'], vw['ct'], vw['sv'], vw['cv']
        for hh in (D('1e-4'), D('5e-4'), D('1e-3')):
            s_h, c_h = dsin(hh), dcos(hh)
            sv2 = sv * c_h + sgn * cv * s_h; cv2 = cv * c_h - sgn * sv * s_h
            A = [[-st, ct, D(0)], [-ct * cv2, -st * cv2, sv2]]
            p = vec(A, dv)
            if abs(p[0] * nD[0] + p[1] * nD[1]) > D(10) ** -40: return False
        return True

    def _pack(self):
        C = self.C
        def arr(key, r=None):
            lo = np.array([float(c[key][r].lo) if r is not None else float(c[key].lo) for c in C])
            hi = np.array([float(c[key][r].hi) if r is not None else float(c[key].hi) for c in C])
            return I(lo, hi)
        self.G0 = [arr('g0', r) for r in range(3)]; self.G1 = [arr('g1', r) for r in range(3)]
        self.UI = [arr('Ui', r) for r in range(3)]; self.NN = [arr('N', r) for r in range(3)]; self.WN = arr('WN')
        for key in ('Dp', 'Dm', 'MX', 'MTT', 'rhoi', 'K', 'Dp0', 'Dm0', 'Z'): setattr(self, key, np.array([c[key] for c in C]))

    def _lp(self, om_c, om_rad, tl, th, sigma):
        g0 = np.stack([(x.lo + x.hi) / 2 for x in self.G0], 1); g1 = np.stack([(x.lo + x.hi) / 2 for x in self.G1], 1)
        ui = np.stack([(x.lo + x.hi) / 2 for x in self.UI], 1); nn = np.stack([(x.lo + x.hi) / 2 for x in self.NN], 1)
        wn = (self.WN.lo + self.WN.hi) / 2
        r0, eps0, beta = self.r0, self.eps0, self.beta
        Dv = self.Dp if sigma > 0 else self.Dm
        qv = (ui @ om_c) * (nn @ om_c) - wn - 2 * om_rad * np.linalg.norm(ui, axis=1)
        n = len(Dv); rows = []; nv = n + 13
        for ei, te in enumerate([tl, th]):
            base = n + 6 * ei; c0 = -(1 - te) * Dv; Hv = g0 + r0 * g1
            cr = c0 + r0 * (np.minimum(0, te * beta * qv / 2) - te * eps0 * self.rhoi - (1 - te) * (self.MX + eps0 * self.MTT / 2)) - r0 ** 2 * self.K
            r = np.zeros(nv); r[:n] = -(te * (g0 @ om_c) + c0); r[base:base + 3] = te * om_rad; rows.append(r)
            r = np.zeros(nv); r[:n] = -(te * (Hv @ om_c) + cr); r[base + 3:base + 6] = te * om_rad; r[-1] = 1; rows.append(r)
            for d in range(3):
                for sg in (1, -1):
                    r = np.zeros(nv); r[:n] = sg * g0[:, d]; r[base + d] = -1; rows.append(r)
                    r = np.zeros(nv); r[:n] = sg * Hv[:, d]; r[base + 3 + d] = -1; rows.append(r)
        c = np.zeros(nv); c[-1] = -1; Aeq = np.zeros((1, nv)); Aeq[0, :n] = 1
        res = linprog(c, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=Aeq, b_eq=[1], bounds=[(0, None)] * (nv - 1) + [(None, None)], method='highs')
        if res.status != 0 or -res.fun <= 0: return None
        return np.maximum(res.x[:n], 0)

    def _snap(self, lam, sigma):
        """Degenerate first order: support {p, q} with g0 = (0,0,a), (0,0,-b) exactly (claims 'Z') and D_sigma == 0 exactly.
        Returns interval enclosures of the exact weights (b, a)/(a+b), for which sum lam g0 == 0 exactly."""
        S = np.where(lam > 1e-14)[0]
        if len(S) != 2: return None
        D0 = self.Dp0 if sigma > 0 else self.Dm0
        if not (self.Z[S].all() and D0[S].all()): return None
        z = self.G0[2]; a = I(z.lo[S[0]], z.hi[S[0]]); b = I(z.lo[S[1]], z.hi[S[1]])
        if a.lo > 0 and b.hi < 0: b = I(0.0) - b
        elif a.hi < 0 and b.lo > 0: a = I(0.0) - a
        else: return None
        s = a + b                                    # > 0
        l1 = I(dn(b.lo / s.hi), up(b.hi / s.lo)); l2 = I(dn(a.lo / s.hi), up(a.hi / s.lo))
        return S, I(np.array([float(l1.lo), float(l2.lo)]), np.array([float(l1.hi), float(l2.hi)]))

    def certify(self, om_c, om_rad, tl, th, sigma):
        lam = self._lp(om_c, om_rad, tl, th, sigma)
        if lam is None: return False
        ok = self._certify(lam, om_c, om_rad, tl, th, sigma, None)
        if ok: return True
        sn = self._snap(lam, sigma)
        return sn is not None and self._certify(lam, om_c, om_rad, tl, th, sigma, sn)

    def _certify(self, lam, om_c, om_rad, tl, th, sigma, snap):
        if snap is None:
            S = np.where(lam > 1e-14)[0]; L = I(lam[S])
        else:
            S, L = snap
        r0 = I(self.r0); m = len(S) * 2.3e-16
        sub = lambda X: I(X.lo[S], X.hi[S])
        G0 = [sub(x) for x in self.G0]; G1 = [sub(x) for x in self.G1]; UI = [sub(x) for x in self.UI]; NN = [sub(x) for x in self.NN]; WN = sub(self.WN)
        Dv = I((self.Dp if sigma > 0 else self.Dm)[S]); MX = I(self.MX[S]); MTT = I(self.MTT[S]); RH = I(self.rhoi[S]); KK = I(self.K[S])
        om = [I(om_c[0]), I(om_c[1]), I(om_c[2])]
        qv = (UI[0] * om[0] + UI[1] * om[1] + UI[2] * om[2]) * (NN[0] * om[0] + NN[1] * om[1] + NN[2] * om[2]) - WN \
             - I(2.0) * I(om_rad) * (UI[0].sqr() + UI[1].sqr() + UI[2].sqr()).sqrt()
        qmin = I(np.minimum(qv.lo, 0.0))
        def comb(x):
            p = L * x; return I(float(dn(p.lo.sum() - m * np.abs(p.lo).sum())), float(up(p.hi.sum() + m * np.abs(p.hi).sum())))
        if snap is None:
            E = [comb(x) for x in G0]; Hs = [comb(x + r0 * y) for x, y in zip(G0, G1)]
        else:   # sum lam* g0 == 0 exactly (g0 = (0,0,+-a)), hence E = 0 and Hs = r0 sum lam* g1
            E = [I(0.0)] * 3; Hs = [comb(r0 * y) for y in G1]
        eps0, beta = I(self.eps0), I(self.beta)
        for te in (tl, th):
            T_ = I(te); T1 = I(1.0) - T_
            c0 = comb(I(0.0) - T1 * Dv)
            cr = comb(I(0.0) - T1 * Dv + r0 * (I(0.5) * T_ * beta * qmin - T_ * eps0 * RH - T1 * (MX + eps0 * MTT * I(0.5))) - r0 * r0 * KK)
            def lowdot(v):
                d = v[0] * om[0] + v[1] * om[1] + v[2] * om[2]; nr = (v[0].sqr() + v[1].sqr() + v[2].sqr()).sqrt()
                return T_ * (d - I(om_rad) * nr)
            if snap is None:
                if not float((lowdot(E) + c0).lo) >= 0: return False
            # snapped: Phi == tau<om, 0> - (1-tau) * 0 == 0 exactly (D_sigma == 0 exactly by the flags checked in _snap)
            if not float((lowdot(Hs) + cr).lo) > 0: return False
        return True
