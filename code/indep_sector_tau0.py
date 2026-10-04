"""Independent re-verification of sector certificates (Theorem 4.x) in exact rational interval arithmetic.
Only combinatorial data are taken from rig_sector (the list of contacts: inner vertex, line normal (70-digit),
kind, parameter, active set) and the LP weights (witnesses).  All numerical quantities are recomputed here from the
60-digit data with Fractions, following the statements in the paper (Lemmas contact expansion / inactive vertices,
Theorem sector), independently of ia.py and of rig_sector's numerics."""
import sys, random, time, pickle
from fractions import Fraction as F
from decimal import Decimal as D
import numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_sector import RigSector
EX = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exact_pts.pkl'), 'rb'))
TOL = F(1, 10 ** 55)
class Q:
    __slots__ = ('lo', 'hi')
    def __init__(s, lo, hi=None): s.lo = F(lo); s.hi = s.lo if hi is None else F(hi)
    @staticmethod
    def c(x): return x if isinstance(x, Q) else Q(x)
    def __add__(a, b): b = Q.c(b); return Q(a.lo + b.lo, a.hi + b.hi)
    __radd__ = __add__
    def __sub__(a, b): b = Q.c(b); return Q(a.lo - b.hi, a.hi - b.lo)
    def __rsub__(a, b): return Q.c(b) - a
    def __neg__(a): return Q(-a.hi, -a.lo)
    def __mul__(a, b):
        b = Q.c(b); p = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi); return Q(min(p), max(p))
    __rmul__ = __mul__
    def abs(a): return a if a.lo >= 0 else (-a if a.hi <= 0 else Q(0, max(-a.lo, a.hi)))
    def r(a, bits=240):
        m = 2 ** bits; return Q(F((a.lo * m).__floor__(), m), F((a.hi * m).__ceil__(), m))
def dq(x): f = F(D(x)); return Q(f - TOL, f + TOL)
def sqrt_ub(x):       # rational upper bound of sqrt(x), x >= 0 rational
    if x <= 0: return F(0)
    g = F(float(x) ** 0.5) * F(1000000001, 1000000000) + F(1, 10 ** 30)
    while g * g < x: g *= 2
    return g
def norm_ub(v): return sqrt_ub(sum(max(c.lo * c.lo, c.hi * c.hi) for c in v))
def dot(u, v): return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
def cross(u, v): return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
def mv(M, v): return [M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2] for r in range(3)]
def sub(u, v): return [a - b for a, b in zip(u, v)]
def scal(a, v): return [a * x for x in v]
def add(u, v): return [a + b for a, b in zip(u, v)]

class Indep:
    def __init__(self, which, ref, mode, r0, R):
        self.R = R; self.mode = mode; self.r0 = F(r0); vw = EX[which]
        st, ct, sv, cv = dq(vw['st']), dq(vw['ct']), dq(vw['sv']), dq(vw['cv']); Z = Q(0)
        A = [[-st, ct, Z], [-(ct * cv), -(st * cv), sv], [ct * sv, st * sv, cv]]
        At = [[-ct, -st, Z], [st * cv, -(ct * cv), Z], [-(st * sv), ct * sv, Z]]
        Ap = [[Z, Z, Z], [ct * sv, st * sv, cv], [ct * cv, st * cv, -sv]]
        V = [[dq(x) for x in v] for v in EX['VZ']]
        T = [[Q(int(i == j)) for j in range(3)] for i in range(3)]
        if which in 'CD': T = [[dq(x) for x in r] for r in EX['T' + which]]
        pm = list(range(60)); sig = 1
        if ref == 'ref2': pm = [int(x) for x in np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'permA.npy'))]; sig = -1
        MF = [[Q(sig if (i == j and i < 2) else int(i == j)) for j in range(3)] for i in range(3)]
        self.W = [mv(A, v) for v in V]; self.Wt = [mv(At, v) for v in V]; self.Wp = [mv(Ap, v) for v in V]
        TV = [mv(T, V[pm[i]]) for i in range(60)]
        self.U = [mv(MF, mv(A, v)) for v in TV]; self.Ut = [mv(MF, mv(At, v)) for v in TV]; self.Up = [mv(MF, mv(Ap, v)) for v in TV]
        self.d = [[norm_ub([a - F(sig) * b for a, b in zip(V[k], TV[i])]) for k in range(60)] for i in range(60)]
        # identical trajectories (V_k = sigma T V_i; verified exactly in Q(sqrt5) by exact_verify): alpha == 0, d == 0
        self.ident = [[all((a - F(sig) * b).abs().hi < F(1, 10 ** 40) for a, b in zip(V[k], TV[i])) for k in range(60)] for i in range(60)]
        self.chidir = R.chidir
    def contact(self, c):
        i, _, _, kind, val, K, snx, sny, isextra = self.R.C[c]['meta']
        nx, ny = dq(snx), dq(sny); N = [nx, ny, Q(0)]; Np = [-ny, nx, Q(0)]
        U = self.U[i]; phi = F(val) if kind == 'phi' else F(0); chi = F(val) if kind == 'chi' else F(0)
        # active set checks (independent): partners on the line, others strictly below with gap Delta
        vals = [self.W[k][0] * nx + self.W[k][1] * ny for k in range(60)]; ui = U[0] * nx + U[1] * ny
        for k in K: assert (vals[k] - ui).abs().hi < F(1, 10 ** 38), 'active partner not on line'
        Delta = min((ui - vals[k]).lo for k in range(60) if k not in K)
        return dict(i=i, N=N, Np=Np, U=U, phi=phi, chi=chi, K=K, Delta=Delta, isextra=isextra)
    def inactive_ok(self, ct):
        i, N, Np, K = ct['i'], ct['N'], ct['Np'], ct['K']; U, Ut, Up = self.U[i], self.Ut[i], self.Up[i]
        a_in = F(0); a_act = F(0)
        for k in range(60):
            e1 = (self.Wt[k][0] - Ut[0]) * N[0] + (self.Wt[k][1] - Ut[1]) * N[1]
            e2 = (self.Wp[k][0] - Up[0]) * N[0] + (self.Wp[k][1] - Up[1]) * N[1]
            e3 = (self.W[k][0] - U[0]) * Np[0] + (self.W[k][1] - U[1]) * Np[1]
            a = max((e1.abs() + e2.abs()).hi, e3.abs().hi)
            if k in K: a_act = max(a_act, a)
            else: a_in = max(a_in, a)
        r = abs(ct['phi']) + abs(ct['chi']); s2 = sqrt_ub(F(2))
        return self.r0 * (a_in + a_act) * (1 + r) + 4 * self.r0 ** 2 * (s2 + r) ** 2 < ct['Delta']
    def terms(self, ct, wl, wh, om, rho, tl, th):
        """lower bounds for one contact over the box: returns (g0, b_c (at centre), bslack, e0, er, beta, C)."""
        i, N, Np, U, phi, chi, K = ct['i'], ct['N'], ct['Np'], ct['U'], ct['phi'], ct['chi'], ct['K']
        Ut, Up = self.Ut[i], self.Up[i]; wc = [(a + b) / 2 for a, b in zip(wl, wh)]; wr = [(b - a) / 2 for a, b in zip(wl, wh)]
        g0 = cross(U, N)
        def bvec(w):
            rate = phi + chi * w[self.chidir]
            JU = add(scal(Q(w[0]), Ut), scal(Q(w[1]), Up))
            return add(cross(JU, N), scal(Q(rate), cross(U, Np)))
        bc = bvec(wc)
        # linear part of b in w: columns
        B0 = sub(bvec([F(1), F(0)]), bvec([F(0), F(0)])); B1 = sub(bvec([F(0), F(1)]), bvec([F(0), F(0)]))
        dbn = sqrt_ub(sum(max(x.lo ** 2, x.hi ** 2) for x in B0 + B1))
        bsl = dbn * sqrt_ub(wr[0] ** 2 + wr[1] ** 2)
        corners = [[a, b] for a in (wl[0], wh[0]) for b in (wl[1], wh[1])]
        f = max(abs(w[0]) + abs(w[1]) for w in corners) + abs(phi) + abs(chi) * max(abs(wl[self.chidir]), abs(wh[self.chidir]))
        NS = F(1) + F(1, 10 ** 12)
        amax0 = None; amaxr = None
        for k in K:
            def alpha(w):
                rate = phi + chi * w[self.chidir]
                d1 = (self.Wt[k][0] - Ut[0]) * N[0] + (self.Wt[k][1] - Ut[1]) * N[1]
                d2 = (self.Wp[k][0] - Up[0]) * N[0] + (self.Wp[k][1] - Up[1]) * N[1]
                d3 = (self.W[k][0] - U[0]) * Np[0] + (self.W[k][1] - U[1]) * Np[1]
                return (d1 * w[0] + d2 * w[1] + d3 * rate).hi
            if self.ident[i][k]: am = F(0); E3 = F(0)
            else:
                am = max(alpha(w) for w in corners)          # linear in w: max at a corner
                E3 = F(1, 2) * self.d[i][k] * f * f * NS
            amax0 = am if amax0 is None else max(amax0, am); amaxr = am + self.r0 * E3 if amaxr is None else max(amaxr, am + self.r0 * E3)
        qc = dot(om, U) * dot(om, N) - dot(U, N); qlo = qc.lo - 2 * rho * norm_ub(U) * norm_ub(N)
        r0 = self.r0; cub = (1 + r0 / 4) / 6 * NS
        if self.mode == 'xi':
            if tl <= 0 and (amax0 > 0 or amaxr > 0): return None
            e0 = -max(amax0, 0) / tl if tl > 0 else F(0); er = -max(amaxr, 0) / tl if tl > 0 else F(0)
            beta = min(tl * qlo, th * qlo) / 2; C = (F(1, 2) * f * f + th * f / 2) * NS + th * th * cub
        else:
            e0 = -amax0; er = -amaxr
            beta = min(tl * tl * qlo, th * th * qlo) / 2; C = (th * f * f / 2 + th * th * f / 2) * NS + th ** 3 * cub
        return dict(g0=g0, bc=bc, bsl=bsl, e0=e0, er=er, beta=beta, C=C)
    def verify(self, S, lam, om_c, rho, wl, wh, tl, th, snapped=False):
        om = [Q(F(x)) for x in om_c]; rho = F(rho)
        wl = [F(x) for x in wl]; wh = [F(x) for x in wh]; tl = F(tl); th = F(th)
        cts = [self.contact(c) for c in S]
        if not all(self.inactive_ok(ct) for ct in cts): return False
        T = [self.terms(ct, wl, wh, om, rho, tl, th) for ct in cts]
        if any(t is None for t in T): return False
        L = [Q(F(l)) if not isinstance(l, Q) else l for l in lam]
        G = [sum((L[j] * T[j]['g0'][r] for j in range(len(S))), Q(0)) for r in range(3)]
        H = [sum((L[j] * (T[j]['g0'][r] + self.r0 * T[j]['bc'][r]) for j in range(len(S))), Q(0)) for r in range(3)]
        if snapped:
            # exact-zero first order: two contacts whose g0 = (0,0,+a), (0,0,-b) exactly (claims 'Z', verified in Q(sqrt5));
            # here we check consistency of the enclosures and that the weights are the exact ratio (b,a)/(a+b) enclosed
            assert len(S) == 2 and all(self.R.Z[c] for c in S)
            g = [T[j]['g0'] for j in range(2)]
            for j in range(2):
                for r in range(2): assert g[j][r].lo <= 0 <= g[j][r].hi
            a_, b_ = g[0][2], g[1][2]
            if a_.hi < 0: a_, b_ = -a_, -b_
            assert a_.lo > 0 and b_.hi < 0
            b_ = -b_; s_ = a_ + b_
            w0 = Q(b_.lo / s_.hi, b_.hi / s_.lo); w1 = Q(a_.lo / s_.hi, a_.hi / s_.lo)
            assert L[0].lo <= w0.hi and w0.lo <= L[0].hi and L[1].lo <= w1.hi and w1.lo <= L[1].hi
            G = [Q(0)] * 3; H = [sum((L[j] * (self.r0 * T[j]['bc'][r]) for j in range(len(S))), Q(0)) for r in range(3)]
        e0 = sum((L[j] * T[j]['e0'] for j in range(len(S))), Q(0)).lo
        rest = sum((L[j] * (T[j]['er'] + self.r0 * T[j]['beta'] - self.r0 ** 2 * T[j]['C']) for j in range(len(S))), Q(0)).lo
        bsl = sum((L[j] * T[j]['bsl'] for j in range(len(S))), Q(0)).hi
        ends = [F(1)] if self.mode == 'xi' else [tl, th]
        for te in ends:
            lowG = te * (dot(om, G).lo - rho * norm_ub(G)); lowH = te * (dot(om, H).lo - rho * norm_ub(H))
            if snapped: lowG = F(0); assert e0 >= 0
            if not lowG + e0 >= 0: return False
            if not lowH + rest - self.r0 * te * bsl > 0: return False
        return True

def omega_box(fc, lo, hi):
    a, sg = fc; c = (lo + hi) / 2; v = np.insert(c, a, sg); return v / np.linalg.norm(v), float(np.linalg.norm((hi - lo) / 2)) * (1 + 1e-12) + 1e-15
if __name__ == '__main__':
    which, ref, mode, r0, nsamp, seed = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    from decimal import Decimal as Dd
    RZ = [[Dd(-1), Dd(0), Dd(0)], [Dd(0), Dd(-1), Dd(0)], [Dd(0), Dd(0), Dd(1)]]
    if which in 'AB': R = RigSector(which, mode=mode, r0=r0) if ref == 'ref1' else RigSector(which, perm=list(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'permA.npy'))), Mfix=RZ, mode=mode, r0=r0)
    else: R = RigSector(which, Vin_mat=EX['T' + which], mode=mode, r0=r0)
    X = Indep(which, ref, mode, r0, R)
    rng = np.random.default_rng(seed); tried = 0; cert = 0; ok = 0; t0 = time.time(); kinds = {}
    import os, json
    BOX = json.load(open(os.environ['BOXFILE'])) if os.environ.get('BOXFILE') else None
    while cert < nsamp and tried < 50 * nsamp:
        tried += 1; fc = (int(rng.integers(3)), int(rng.choice([-1, 1]))); piece = 2; dep = int(rng.integers(6, 12)); cell = 2.0 ** (1 - dep)
        if BOX is not None:     # boxes from a saved list (e.g. the hard spots), piece given by env PIECE
            b = BOX[int(rng.integers(len(BOX)))]; fc = tuple(b[0]); lo = np.array(b[1], float); hi = np.array(b[2], float); piece = int(os.environ.get('PIECE', '1'))
            if piece == 1: wl, wh, tl, th = lo[2:], hi[2:], 1.0, 1.0
        if BOX is not None and piece == 1: pass
        elif piece == 1:
            lo = -1 + cell * rng.integers(0, 2 ** dep, 4); hi = lo + cell
            if np.linalg.norm(np.clip(0, lo[2:], hi[2:])) > 1: continue
            wl, wh, tl, th = lo[2:], hi[2:], 1.0, 1.0
        else:
          if BOX is None:
            lo = np.array([-1 + cell * rng.integers(0, 2 ** dep), -1 + cell * rng.integers(0, 2 ** dep), 2 * np.pi * rng.integers(0, 2 ** dep) / 2 ** dep, 0.0])
            hi = lo + np.array([cell, cell, 2 * np.pi / 2 ** dep, 1.0 / 2 ** dep])
          if True:
            ps = np.linspace(lo[2], hi[2], 9); pts = np.stack([np.cos(ps), np.sin(ps)], 1); wl = pts.min(0) - 1e-9; wh = pts.max(0) + 1e-9
            for k in range(-2, 6):
                ang = k * np.pi / 2
                if lo[2] <= ang <= hi[2]:
                    dim = k % 2; val = np.cos(ang) if dim == 0 else np.sin(ang)
                    if val > 0: wh[dim] = 1
                    else: wl[dim] = -1
            tl, th = lo[3], hi[3]
            assert tl == 0
        oc, orad = omega_box(fc, lo[:2], hi[:2])
        if not R.certify(oc, orad, wl, wh, tl, th): continue
        cert += 1
        Ss, Li, snapped = R.last       # the certificate actually used by rig_sector.certify
        try:
            good = X.verify(list(Ss), [Q(F(Li.lo[j]), F(Li.hi[j])) for j in range(len(Ss))], oc, orad, wl, wh, tl, th, snapped=snapped)
        except AssertionError: good = False
        kinds[(snapped, len(Ss))] = kinds.get((snapped, len(Ss)), 0) + 1
        ok += good
        if not good: print('NOT CONFIRMED', fc, lo.tolist(), hi.tolist(), flush=True)
    print(which, ref, mode, 'independently confirmed', ok, '/', cert, 'certified boxes', round(time.time() - t0), 's', 'kinds(snapped,|S|):', kinds, flush=True)
