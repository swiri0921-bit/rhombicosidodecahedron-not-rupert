"""Independent re-verification of arc-tube certificates (Theorem tube) in exact rational interval arithmetic.
Combinatorial data (contacts, exact partner lists) and LP weights come from rig_tube; every number is recomputed here."""
import sys, time, pickle
import numpy as np
from fractions import Fraction as F
sys.path.insert(0, '.')
from indep_sector import Q, dq, sqrt_ub, norm_ub, dot, cross, mv, sub, scal, add
from rig_tube import RigTube
from dec_util import mat
from decimal import Decimal as D
EX = pickle.load(open('./exact_pts.pkl', 'rb'))
Z40 = F(1, 10 ** 38)
class IndepTube:
    def __init__(self, which, ref, T, r0, eps0, beta):
        self.T = T; self.r0 = F(r0); self.e0 = F(eps0); self.beta = F(beta)
        sgn = 1 if which == 'C' else -1; vw = EX[which]
        st, ct, sv, cv = dq(vw['st']), dq(vw['ct']), dq(vw['sv']), dq(vw['cv']); Zr = Q(0)
        A = [[-st, ct, Zr], [-(ct * cv), -(st * cv), sv], [ct * sv, st * sv, cv]]
        At = [[-ct, -st, Zr], [st * cv, -(ct * cv), Zr], [-(st * sv), ct * sv, Zr]]
        Ap = [[Zr, Zr, Zr], [ct * sv, st * sv, cv], [ct * cv, st * cv, -sv]]
        Tm = EX['T' + which]; sig = 1
        if ref == 'ref2': Tm = mat(EX['TC'], EX['h2C']); sig = -1
        TmQ = [[dq(x) for x in r] for r in Tm]
        V = [[dq(x) for x in v] for v in EX['VZ']]; TV = [mv(TmQ, v) for v in V]
        MF = [[Q(sig if (i == j and i < 2) else int(i == j)) for j in range(3)] for i in range(3)]
        self.W = [mv(A, v) for v in V]; self.Wt = [mv(At, v) for v in V]; self.Wv = [scal(Q(sgn), mv(Ap, v)) for v in V]
        self.U = [mv(MF, mv(A, v)) for v in TV]; self.Ut = [mv(MF, mv(At, v)) for v in TV]; self.Uv = [scal(Q(sgn), mv(MF, mv(Ap, v))) for v in TV]
        self.d = [[norm_ub([a - F(sig) * b for a, b in zip(V[k], TV[i])]) for k in range(60)] for i in range(60)]
        self.ident = [[all((a - F(sig) * b).abs().hi < Z40 for a, b in zip(V[k], TV[i])) for k in range(60)] for i in range(60)]
        # arc test data: first image axis M1 (fixed along the arc), horizontal direction H=(cos t, sin t, 0), e3
        self.M1 = A[0]; self.Hd = [ct, st, Zr]; self.sig = sig
        self.TV = TV; self.V = V
    def pk(self, i, k, N): return (self.W[k][0] - self.U[i][0]) * N[0] + (self.W[k][1] - self.U[i][1]) * N[1]
    def contact(self, c):
        i, _, _, phi, chi, snx, sny, exact = self.T.C[c]['meta']
        nx, ny = dq(snx), dq(sny); N = [nx, ny, Q(0)]; Np = [-ny, nx, Q(0)]
        phi = F(phi); chi = F(chi); U = self.U[i]
        # exactness of the partners along the arc (numerical zero test; exact version: exact_verify ARC/ID claims)
        for k in exact:
            if self.ident[i][k]: continue
            assert phi == 0
            dv = [a - F(self.sig) * b for a, b in zip(self.V[k], self.TV[i])]
            c1 = (nx * dot(self.M1, dv)).abs().hi < Z40 or nx.abs().hi < Z40
            c2 = ny.abs().hi < Z40 or (dot(self.Hd, dv).abs().hi < Z40 and dv[2].abs().hi < Z40)
            assert c1 and c2 and self.pk(i, k, N).abs().hi < Z40, 'partner not exact along the arc'
        return dict(i=i, N=N, Np=Np, U=U, phi=phi, chi=chi, exact=exact)
    def quantities(self, ct):
        i, N, Np, U, phi, chi, ex = ct['i'], ct['N'], ct['Np'], ct['U'], ct['phi'], ct['chi'], ct['exact']
        e0, beta, r0 = self.e0, self.beta, self.r0
        g0 = cross(U, N); g1 = add(cross(self.Uv[i], N), scal(Q(phi), cross(U, Np)))
        Dk = []
        for k in ex:
            if self.ident[i][k]: Dk.append(Q(0)); continue
            dkt = (self.Wt[k][0] - self.Ut[i][0]) * N[0] + (self.Wt[k][1] - self.Ut[i][1]) * N[1]
            pk = (self.W[k][0] - U[0]) * Np[0] + (self.W[k][1] - U[1]) * Np[1]
            Dk.append(dkt + Q(chi) * pk)
        Dp = max(x.hi for x in Dk); Dm = max((-x).hi for x in Dk); Dmax = max(max(abs(x.lo), abs(x.hi)) for x in Dk)
        md = max(self.d[i][k] for k in ex)
        MX = md * (1 + abs(phi)) * (1 + abs(chi)); MTT = md * (1 + abs(chi)) ** 2
        f = 1 + e0 + abs(phi) + e0 * abs(chi)
        K = F(1, 2) * f * f + beta * (1 + e0) * (1 + abs(phi) + abs(chi)) + beta * beta / 6 * (1 + beta * r0 / 4)
        # non-exact partners must stay below the exact ones (Theorem tube, displayed condition)
        margin = e0 * (Dmax + r0 * (MX + e0 * MTT / 2))
        for k in range(60):
            if k in ex: continue
            gk = self.pk(i, k, N).hi * -1 if False else ((self.W[k][0] - U[0]) * N[0] + (self.W[k][1] - U[1]) * N[1])
            ak = ((self.Wv[k][0] - self.Uv[i][0]) * N[0] + (self.Wv[k][1] - self.Uv[i][1]) * N[1]) + Q(phi) * ((self.W[k][0] - U[0]) * Np[0] + (self.W[k][1] - U[1]) * Np[1])
            duk = (self.Wt[k][0] - self.Ut[i][0]) * N[0] + (self.Wt[k][1] - self.Ut[i][1]) * N[1]
            pkk = (self.W[k][0] - U[0]) * Np[0] + (self.W[k][1] - U[1]) * Np[1]
            dd = duk.abs().hi + abs(chi) * pkk.abs().hi
            rem = F(1, 2) * r0 * self.d[i][k] * f * f + e0 * dd + margin
            if gk.hi > -Z40:
                if not (gk.hi <= Z40 and ak.hi + rem < 0): return None
            else:
                if not (gk.hi + r0 * (ak.abs().hi + rem) < 0): return None
        return dict(g0=g0, g1=g1, Dp=Dp, Dm=Dm, MX=MX, MTT=MTT, K=K, chi=chi, U=U, N=N)
    def verify(self, S, lam, om_c, rho, tl, th, sigma, snapped=False):
        om = [Q(F(x)) for x in om_c]; rho = F(rho); tl = F(tl); th = F(th)
        qs = [self.quantities(self.contact(c)) for c in S]
        if any(q is None for q in qs): return False
        L = [Q(F(l)) if not isinstance(l, Q) else l for l in lam]; r0, e0, beta = self.r0, self.e0, self.beta
        G = [sum((L[j] * qs[j]['g0'][r] for j in range(len(S))), Q(0)) for r in range(3)]
        H = [sum((L[j] * (qs[j]['g0'][r] + r0 * qs[j]['g1'][r]) for j in range(len(S))), Q(0)) for r in range(3)]
        if snapped:
            G = [Q(0)] * 3; H = [sum((L[j] * (r0 * qs[j]['g1'][r]) for j in range(len(S))), Q(0)) for r in range(3)]
        for te in (tl, th):
            Dsum = sum((L[j] * Q(qs[j]['Dp'] if sigma > 0 else qs[j]['Dm']) for j in range(len(S))), Q(0)).hi
            if snapped: assert Dsum <= Z40 and all((qs[j]['Dp'] if sigma > 0 else qs[j]['Dm']) <= 0 for j in range(len(S)))
            lowG = F(0) if snapped else te * (dot(om, G).lo - rho * norm_ub(G))
            if not (snapped or lowG - (1 - te) * Dsum >= 0): return False
            rest = F(0)
            for j in range(len(S)):
                q = qs[j]; qc = dot(om, q['U']) * dot(om, q['N']) - dot(q['U'], q['N'])
                qmin = min(qc.lo - 2 * rho * norm_ub(q['U']) * norm_ub(q['N']), F(0))
                Dv = q['Dp'] if sigma > 0 else q['Dm']
                if snapped: Dv = F(0)
                v = -(1 - te) * Dv + r0 * (F(1, 2) * te * beta * qmin - te * e0 * (1 + abs(q['chi'])) - (1 - te) * (q['MX'] + e0 * q['MTT'] / 2)) - r0 * r0 * q['K']
                rest += (L[j] * Q(v)).lo
            lowH = te * (dot(om, H).lo - rho * norm_ub(H))
            if not lowH + rest > 0: return False
        return True
def omega_box(face, lo, hi):
    a, sg = face; c = (lo + hi) / 2; v = np.insert(c, a, sg); return v / np.linalg.norm(v), float(np.linalg.norm((hi - lo) / 2)) * (1 + 1e-12) + 1e-15
if __name__ == '__main__':
    which, ref, nsamp, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    sgn = 1 if which == 'C' else -1; Tm = EX['T' + which]; Mfix = None
    if ref == 'ref2': Tm = mat(EX['TC'], EX['h2C']); Mfix = [[D(-1), D(0), D(0)], [D(0), D(-1), D(0)], [D(0), D(0), D(1)]]
    T = RigTube(which, Tm, sgn, Mfix=Mfix, r0=1e-3, eps0=0.4, beta=0.5)
    X = IndepTube(which, ref, T, 1e-3, 0.4, 0.5)
    rng = np.random.default_rng(seed); cert = 0; ok = 0; tried = 0; t0 = time.time()
    while cert < nsamp and tried < 100 * nsamp:
        tried += 1; face = (int(rng.integers(3)), int(rng.choice([-1, 1]))); dep = int(rng.integers(int(sys.argv[5]) if len(sys.argv) > 5 else 3, int(sys.argv[6]) if len(sys.argv) > 6 else 9)); cell = 2.0 ** (1 - dep)
        lo = np.array([-1 + cell * rng.integers(0, 2 ** dep), -1 + cell * rng.integers(0, 2 ** dep), rng.integers(0, 2 ** dep) / 2 ** dep])
        hi = lo + np.array([cell, cell, 1.0 / 2 ** dep]); sigma = int(rng.choice([-1, 1]))
        oc, orad = omega_box(face, lo[:2], hi[:2])
        if not T.certify(oc, orad, lo[2], hi[2], sigma): continue
        cert += 1
        lam = T._lp(oc, orad, lo[2], hi[2], sigma); S = list(np.where(lam > 1e-14)[0])
        good = X.verify(S, [lam[j] for j in S], oc, orad, lo[2], hi[2], sigma)
        if not good:
            sn = T._snap(lam, sigma)
            if sn is not None:
                Ss, Li = sn; good = X.verify(list(Ss), [Q(F(Li.lo[j]), F(Li.hi[j])) for j in range(len(Ss))], oc, orad, lo[2], hi[2], sigma, snapped=True)
        ok += good
        if not good: print('NOT CONFIRMED', face, lo.tolist(), hi.tolist(), sigma, flush=True)
    print(which, ref, 'tube: independently confirmed', ok, '/', cert, 'certified boxes', round(time.time() - t0), 's', flush=True)
