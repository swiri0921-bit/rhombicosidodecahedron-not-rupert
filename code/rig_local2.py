"""Rigorous local2 (identity/RZPI-type coincidence) theorem, fixed-direction version.
Setting: Ain(x) = R(x) D Aout(x) h (world frame), h in Ih-rotations, D in {I, RZPI}; inner vertex i sits at R U_i(x),
U_i = D Aout h V_i, whose shadow equals that of outer vertex k(i) (k = pi(i) for D=I, antipode of pi(i) for D=RZPI).
For fixed unit n and outer vertex k that is the unique maximiser of <Pi W_j(x), n> over the whole box:
   f = <Pi R U_i, n> - h_Q(n) = <Pi (R - I) U_i, n> >= <xi, U_i x N> - (|xi|^2/2 + |xi|^3/6)|U_i|.
|g(x) - g(x0)| <= |U_i(x) - U_i(x0)| <= 2 eps (outer frame rotation rates <= 1, |V| <= 1).
|xi(x)| <= theta(x0) + 5 eps.  If min_{|v|=1} max_c <v, g_c(x0)> =: c0 and c0 - 2 eps > (xm/2 + xm^2/6) with xm = theta + 5eps,
then for every x in the box with xi != 0 some f > 0; for xi = 0 the shadows coincide (no strict containment)."""
import numpy as np, sys
sys.path.insert(0, '.')
from ia import I, sin, cos, dn, up
from rig_global import VX
from rid_setup import rid_z5
from local2 import frame, rot_group, RZPI, rot_angle
from disprover import R2, box_mid_eps
from scipy.spatial import ConvexHull
NUP = I(1.0 + 1e-12)
PF = rid_z5(); G = rot_group(PF)
from exact_pts import GE
from dec_util import encl as _encl
HEX = [[[I(*_encl(x)) for x in row] for row in ge] for ge, _ in GE]   # exact h enclosed
assert all(list(pp) == [int(np.argmin(np.linalg.norm(PF - g @ v, axis=1))) for v in PF] for (_, pp), g in zip(GE, G))
PERM = [np.array([int(np.argmin(np.linalg.norm(PF - g @ v, axis=1))) for v in PF]) for g in G]
ANTI = np.array([int(np.argmin(np.linalg.norm(PF + v, axis=1))) for v in PF])
# sphere cap cover (cube faces)
_m = 48
_u = (np.arange(_m) + 0.5) / _m * 2 - 1
_U, _Vv = np.meshgrid(_u, _u); CAPS = []
for ax in range(3):
    for sg in (-1, 1):
        P3 = np.insert(np.stack([_U.ravel(), _Vv.ravel()], 1), ax, sg, axis=1); CAPS.append(P3)
CAPS = np.vstack(CAPS); CAPS = CAPS / np.linalg.norm(CAPS, axis=1, keepdims=True)
CAPR = float(up(np.sqrt(2) * (1.0 / _m) * 1.001))

def frame_I(t, p):
    st, ct, sp, cp = sin(t), cos(t), sin(p), cos(p); Z = I(0.0)
    return [[-st, ct, Z], [-(ct * cp), -(st * cp), sp], [ct * sp, st * sp, cp]]
def matmul_I(A, B):
    return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3)] for i in range(3)]
def const_I(M): return [[I(float(M[i][j])) for j in range(3)] for i in range(3)]

def check(box, ndir=5):
    lo, hi = box[:, 0], box[:, 1]; c = (lo + hi) / 2
    eps = float(np.max(np.maximum(up(hi - c), up(c - lo))))
    t1, v1, t2, v2, a = c
    Aout = frame(t2, v2); Ain = np.block([[R2(a), np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]) @ frame(t1, v1)
    best = (np.inf, None)
    for hi_, h in enumerate(G):
        for Di, D in enumerate((np.eye(3), RZPI)):
            th = rot_angle(Ain @ (D @ Aout @ h).T)
            if th < best[0]: best = (th, (hi_, Di))
    if best[0] + 5 * eps > 0.3: return False
    hi_, Di = best[1]; h = G[hi_]; D = (np.eye(3), RZPI)[Di]
    # rigorous rotation angle bound of R = Ain (D Aout h)^T at the centre
    AoI = frame_I(t2, v2); AiI0 = frame_I(t1, v1)
    ca, sa = cos(a), sin(a); Z = I(0.0); O = I(1.0)
    Ra = [[ca, -sa, Z], [sa, ca, Z], [Z, Z, O]]
    AiI = matmul_I(Ra, AiI0)
    B = matmul_I(matmul_I(const_I(D), AoI), HEX[hi_])            # D Aout h, h enclosed from exact (60-digit) data
    Bt = [[B[j][i] for j in range(3)] for i in range(3)]
    Rm = matmul_I(AiI, Bt)
    fro = None
    for i in range(3):
        for j in range(3):
            d = (Rm[i][j] - (1.0 if i == j else 0.0)).sqr(); fro = d if fro is None else fro + d
    y = fro.sqrt() * I(1 / np.sqrt(8) * (1 + 1e-15))   # ||R-I||_F = 2 sqrt2 sin(theta/2)
    if not (y.hi < 0.5): return False
    th_hi = float((I(2.0) * y * (O - y.sqr()).sqrt().__rtruediv__(1.0) if False else I(2.0) * y * I(1.0 / np.sqrt(1 - float(y.hi) ** 2) * (1 + 1e-12))).hi)
    xm = float(up(th_hi + 5 * eps))
    # outer projected vertices at centre, derivatives for robust extremeness
    W = PF @ Aout.T; Q = W[:, :2]
    hv = ConvexHull(Q).vertices; mh = len(hv)
    kmap = PERM[hi_] if Di == 0 else ANTI[PERM[hi_]]
    # U_i = D Aout h V_i  -> world coords of inner vertex i at R=I ; shadow equals outer vertex kmap[i]
    inv = np.empty(60, int); inv[kmap] = np.arange(60)
    Wx = VX  # interval exact vertices (body coords)
    AoutI = AoI
    def world(Mt, Vint): return [Mt[r][0] * Vint[0] + Mt[r][1] * Vint[1] + Mt[r][2] * Vint[2] for r in range(3)]
    Wo = world(AoutI, Wx)                        # outer vertices world (interval arrays)
    UB_ = world(B, Wx)                           # D Aout h V_i for all i
    gs = []
    for e in range(mh):
        k = hv[e]; kp = hv[e - 1]; kn = hv[(e + 1) % mh]
        d1 = Q[k] - Q[kp]; d2 = Q[kn] - Q[k]
        n1 = np.array([d1[1], -d1[0]]) / np.linalg.norm(d1); n2 = np.array([d2[1], -d2[0]]) / np.linalg.norm(d2)
        a1, a2 = np.arctan2(n1[1], n1[0]), np.arctan2(n2[1], n2[0]); da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        for f in np.linspace(0, 1, ndir + 2)[1:-1]:
            ang = a1 + f * da; n = np.array([np.cos(ang), np.sin(ang)])
            nx, ny = I(n[0]), I(n[1])
            # robust: <Pi(W_k - W_j)(x), n> > 0 for all j != k, x in box. value at centre minus grad*hw minus remainder
            dk = [Wx[r][k] - Wx[r] for r in range(3)]
            val = (Wo[0][k] - Wo[0]) * nx + (Wo[1][k] - Wo[1]) * ny
            nd = (dk[0].sqr() + dk[1].sqr() + dk[2].sqr()).sqrt()
            e2 = I(2 * eps)
            lb = val - (nd * e2 + I(0.5) * nd * e2 * e2) * NUP    # |n| <= 1 + 1e-12 (float unit vector)
            mask = np.arange(60) != k
            if not (lb.lo[mask] > 0).all(): continue
            i = inv[k]
            Ui = [UB_[r][i] for r in range(3)]
            g = [Ui[1] * I(0.0) - Ui[2] * ny, Ui[2] * nx - Ui[0] * I(0.0), Ui[0] * ny - Ui[1] * nx]   # U x (nx,ny,0)
            gs.append(g)
    if len(gs) < 4: return False
    GL = np.array([[float(g[r].lo) for r in range(3)] for g in gs]); GH = np.array([[float(g[r].hi) for r in range(3)] for g in gs])
    # c0 >= min over caps of max_c ( <cap, g_c>_lo - CAPR |g_c| )
    Gm = np.maximum(np.abs(GL), np.abs(GH)); gn = up(np.sqrt((Gm ** 2).sum(1)))
    prod_lo = np.minimum(CAPS[:, None, :] * GL[None], CAPS[:, None, :] * GH[None]).sum(2)
    prod_lo = dn(prod_lo - 1e-15 * 3)
    capv = (prod_lo - CAPR * gn[None, :]).max(1)
    c0 = float(capv.min())
    need = float(up((2 * eps + xm / 2 + xm * xm / 6) * (1 + 1e-12)))   # all remainder terms scale with |n| <= 1 + 1e-12
    return c0 > need
