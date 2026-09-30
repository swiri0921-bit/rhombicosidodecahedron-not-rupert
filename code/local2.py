"""New local theorem (edge/support-function version) for near-coincident configurations.
For configuration u: inner set = Pi0 e^{[xi]} W (as sets), W = A_out V, xi = log of relative rotation
to nearest coincidence. For each outer hull edge j (normal n_j, endpoints q,q') and m_j = Pi0^T n_j:
   h_P(n_j) - h_Q(n_j) >= max_{i in {q,q'}} <xi, W_i x m_j> - (|xi|^2/2)(1+|xi|/3) rho.
Hence if 0<|xi| < 2c/(rho(1+|xi|/3)), c = min_{|v|=1} max_{j,i} <v, W_i x m_j>, the inner shadow is not
inside the outer one. Box version: |xi(u)| <= |xi_c| + 5 eps, and c(u) >= c_c - dg (perturbation of g-vectors)."""
import numpy as np, itertools
from scipy.spatial import ConvexHull
from disprover import M, X, R2, box_mid_eps

def frame(t, p):
    B = np.vstack([M(t, p), X(t, p)])
    return B

def rot_group(V, tol=1e-7):
    """all proper rotations mapping vertex set V to itself"""
    n = len(V)
    D = np.linalg.norm(V[:, None] - V[None], axis=2)
    a = 0
    nb = np.argsort(D[a])[1:3]
    b, c = nb
    B0 = np.stack([V[a], V[b], V[c]], 1)
    Binv = np.linalg.inv(B0)
    out = []
    for a2 in range(n):
        for b2 in range(n):
            if abs(D[a2, b2] - D[a, b]) > 1e-6: continue
            for c2 in range(n):
                if abs(D[a2, c2] - D[a, c]) > 1e-6 or abs(D[b2, c2] - D[b, c]) > 1e-6: continue
                Rm = np.stack([V[a2], V[b2], V[c2]], 1) @ Binv
                if np.abs(Rm @ Rm.T - np.eye(3)).max() > 1e-6 or np.linalg.det(Rm) < 0: continue
                W = V @ Rm.T
                if np.min(np.linalg.norm(W[:, None] - V[None], axis=2), 1).max() < 1e-6:
                    out.append(Rm)
    U = []
    for Rm in out:
        if not any(np.abs(Rm - S).max() < 1e-8 for S in U): U.append(Rm)
    return U

RZPI = np.diag([-1.0, -1.0, 1.0])

def rot_angle(Rm):
    return np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))

def g_vectors(W, gap_needed=0.0):
    Q = W[:, :2]
    h = ConvexHull(Q); hv = list(h.vertices); m = len(hv)
    G = []; min_gap = np.inf; min_edge = np.inf; edges = []
    for i in range(m):
        q, q2 = hv[i], hv[(i + 1) % m]
        a, b = Q[q], Q[q2]
        d = b - a; L = np.linalg.norm(d)
        n = np.array([d[1], -d[0]]) / L
        s = (Q - a) @ n
        on = np.where(np.abs(s) < 1e-9)[0]
        others = np.setdiff1d(np.arange(len(Q)), on)
        min_gap = min(min_gap, -s[others].max())
        min_edge = min(min_edge, L)
        mvec = np.array([n[0], n[1], 0.0])
        for k in on:
            G.append(np.cross(W[k], mvec))
        edges.append((q, q2, L, np.linalg.norm(W[q2] - W[q])))
    return np.array(G), min_gap, edges

def inradius(G):
    try:
        H = ConvexHull(G)
    except Exception:
        return 0.0
    return max(0.0, (-H.equations[:, 3]).min())

class Local2:
    def __init__(self, V):
        self.V = V
        self.rho = np.linalg.norm(V, axis=1).max()
        self.grp = rot_group(V)
    def check(self, box):
        (t1, v1, t2, v2, a), eps = box_mid_eps(box)
        Aout = frame(t2, v2)
        Ain = np.block([[R2(a), np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]) @ frame(t1, v1)
        Om = Ain @ Aout.T
        best = np.inf
        for h in self.grp:
            S = Aout @ h @ Aout.T
            for D in (np.eye(3), RZPI):
                best = min(best, rot_angle(Om @ (D @ S).T))
        xi_max = best + 5 * eps                 # |xi(u)| over the box
        W = self.V @ Aout.T
        G, gap, edges = g_vectors(W)
        c = inradius(G)
        eta = 2 * eps                           # outer orientation change over box
        rho = self.rho
        # combinatorial stability: vertices move <= eta*rho (+2nd order) in the projection
        move = eta * rho * (1 + eta)
        if gap <= 2.5 * move:
            return False
        # perturbation of g-vectors: |dW| <= move ; edge-normal rotation <= |d edge| / (L - |d edge|)
        dn = 0.0
        for (q, q2, L, L3) in edges:
            de = eta * L3 * (1 + eta)
            if de >= L / 2: return False
            dn = max(dn, de / (L - de))
        dg = move + rho * dn
        c_low = c - dg
        if c_low <= 0: return False
        return xi_max < 2 * c_low / (rho * (1 + xi_max / 3))


def _poly_g(W, poly):
    """g-vectors and edge data for a given ccw vertex cycle `poly` of projected W."""
    Q = W[:, :2]; G = []; edges = []
    m = len(poly)
    for i in range(m):
        q, q2 = poly[i], poly[(i + 1) % m]
        d = Q[q2] - Q[q]; L = np.linalg.norm(d)
        if L < 1e-12: return None, None
        n = np.array([d[1], -d[0]]) / L
        mvec = np.array([n[0], n[1], 0.0])
        G.append(np.cross(W[q], mvec)); G.append(np.cross(W[q2], mvec))
        edges.append((L, np.linalg.norm(W[q2] - W[q])))
    return np.array(G), edges


def variants(W, thr):
    """possible hull vertex cycles when vertices move by <= thr/4."""
    Q = W[:, :2]
    hv = list(ConvexHull(Q).vertices); m = len(hv)
    mods = []
    for i in range(m):          # insertions onto edges
        q, q2 = hv[i], hv[(i + 1) % m]
        d = Q[q2] - Q[q]; L = np.linalg.norm(d); n = np.array([d[1], -d[0]]) / L
        s = (Q - Q[q]) @ n; tpar = (Q - Q[q]) @ d / L ** 2
        for k in range(len(Q)):
            if k in (q, q2) or k in hv: continue
            if -thr < s[k] <= 1e-12 and 0 < tpar[k] < 1:
                mods.append(('ins', i, k, tpar[k]))
    for i in range(m):          # removals of nearly-flat hull vertices
        p, k, nx = hv[i - 1], hv[i], hv[(i + 1) % m]
        d = Q[nx] - Q[p]; L = np.linalg.norm(d); n = np.array([d[1], -d[0]]) / L
        if (Q[k] - Q[p]) @ n < thr:
            mods.append(('rem', i, k, 0))
    if len(mods) > 8:
        return None
    out = []
    for mask in range(1 << len(mods)):
        chosen = [mods[j] for j in range(len(mods)) if (mask >> j) & 1]
        rem = {c[2] for c in chosen if c[0] == 'rem'}
        ins = {}
        for c in chosen:
            if c[0] == 'ins': ins.setdefault(c[1], []).append((c[3], c[2]))
        poly = []
        for i in range(m):
            if hv[i] not in rem: poly.append(hv[i])
            for _, k in sorted(ins.get(i, [])): poly.append(k)
        if len(poly) >= 3: out.append(poly)
    return out


def check_robust(self, box):
    (t1, v1, t2, v2, a), eps = box_mid_eps(box)
    Aout = frame(t2, v2)
    Ain = np.block([[R2(a), np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]) @ frame(t1, v1)
    Om = Ain @ Aout.T
    best = min(rot_angle(Om @ (D @ Aout @ h @ Aout.T).T) for h in self.grp for D in (np.eye(3), RZPI))
    xi_max = best + 5 * eps
    rho = self.rho; eta = 2 * eps
    move = eta * rho * (1 + eta)
    W = self.V @ Aout.T
    polys = variants(W, 4 * move + 1e-9)
    if polys is None: return False
    for poly in polys:
        G, edges = _poly_g(W, poly)
        if G is None: return False
        dn = 0.0
        for (L, L3) in edges:
            de = eta * L3 * (1 + eta)
            if de >= L / 2: return False
            dn = max(dn, de / (L - de))
        c_low = inradius(G) - (move + rho * dn)
        if c_low <= 0 or not (xi_max < 2 * c_low / (rho * (1 + xi_max / 3))):
            return False
    return True

Local2.check = check_robust


def vertex_face_normals(V):
    H = ConvexHull(V)
    N = []
    for eq in H.equations:
        n = eq[:3] / np.linalg.norm(eq[:3])
        if not any(np.abs(n - m).max() < 1e-7 for m in N): N.append(n)
    N = np.array(N)
    adj = []
    for v in V:
        d = N @ v; mx = (N[:, None, :] @ V.T[None]).max(-1).ravel() if False else None
        adj.append(None)
    # face k contains vertex i iff n_k.v_i == max_j n_k.v_j
    S = N @ V.T; mx = S.max(1, keepdims=True)
    on = np.abs(S - mx) < 1e-7
    return N, [np.where(on[:, i])[0] for i in range(len(V))]


def silhouette_status(X, N, adj, eta):
    """per vertex: 1 = surely on silhouette, 0 = surely not, -1 = uncertain (for view dirs within eta of X)."""
    d = N @ X
    st = []
    for fs in adj:
        dd = d[fs]
        if dd.min() < -eta and dd.max() > eta: st.append(1)
        elif dd.min() > eta or dd.max() < -eta: st.append(0)
        else: st.append(-1)
    return np.array(st)


def check_robust2(self, box):
    (t1, v1, t2, v2, a), eps = box_mid_eps(box)
    Aout = frame(t2, v2)
    Ain = np.block([[R2(a), np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]) @ frame(t1, v1)
    Om = Ain @ Aout.T
    best = min(rot_angle(Om @ (D @ Aout @ h @ Aout.T).T) for h in self.grp for D in (np.eye(3), RZPI))
    xi_max = best + 5 * eps
    rho = self.rho; eta = 2 * eps
    move = eta * rho * (1 + eta)
    if not hasattr(self, 'N'):
        self.N, self.adj = vertex_face_normals(self.V)
    X2 = Aout[2]
    st = silhouette_status(X2, self.N, self.adj, 1.01 * eta * (1 + eta))
    W = self.V @ Aout.T
    Q = W[:, :2]
    sure = np.where(st == 1)[0]; unc = np.where(st == -1)[0]
    if len(unc) > 10: return False
    cen = Q[np.concatenate([sure, unc])].mean(0)
    ang = np.arctan2(Q[:, 1] - cen[1], Q[:, 0] - cen[0])
    for mask in range(1 << len(unc)):
        S = list(sure) + [unc[j] for j in range(len(unc)) if (mask >> j) & 1]
        if len(S) < 3: return False
        poly = sorted(S, key=lambda i: ang[i])
        # drop exact duplicates in projection
        G, edges = _poly_g(W, poly)
        if G is None: return False
        dn = 0.0
        for (L, L3) in edges:
            de = eta * L3 * (1 + eta)
            if de >= L / 2: return False
            dn = max(dn, de / (L - de))
        c_low = inradius(G) - (move + rho * dn)
        if c_low <= 0 or not (xi_max < 2 * c_low / (rho * (1 + xi_max / 3))):
            return False
    return True

Local2.check = check_robust2
