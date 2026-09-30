"""Python port of Steininger & Yurkevich's Rupert disprover (global + local theorem),
from generate_solutiontree.R (github.com/Jakob256/Rupert). Floating-point exploration
version (the authors re-verify the final tree in exact arithmetic with SageMath)."""
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

SQ2 = np.sqrt(2.0)


def M(t, p):
    return np.array([[-np.sin(t), np.cos(t), 0.0],
                     [-np.cos(t) * np.cos(p), -np.sin(t) * np.cos(p), np.sin(p)]])


def X(t, p):
    return np.array([np.cos(t) * np.sin(p), np.sin(t) * np.sin(p), np.cos(p)])


def R2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def M_t(t, p):
    return np.array([[-np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t) * np.cos(p), -np.cos(t) * np.cos(p), 0.0]])


def M_p(t, p):
    return np.array([[0.0, 0.0, 0.0],
                     [np.cos(t) * np.sin(p), np.sin(t) * np.sin(p), np.cos(p)]])


def R_a(a):
    return np.array([[-np.sin(a), -np.cos(a)], [np.cos(a), -np.sin(a)]])


ROT90 = R2(np.pi / 2)


def hull_idx(P2):
    """indices of convex hull vertices, counterclockwise"""
    return ConvexHull(P2).vertices


def box_mid_eps(box):
    lo = box[:, 0]; hi = box[:, 1]
    mid = (lo + hi) / 2
    eps = (hi - lo).max() / 2
    return mid, eps


def w_candidates(Outside, Qh):
    """minimal displacement vectors from each outside point to polygon Qh (ccw)."""
    d = Outside[:, None, :] - Qh[None, :, :]
    d2 = (d ** 2).sum(-1)
    k = d2.argmin(1)
    res1 = -d[np.arange(len(Outside)), k]
    E0 = Qh; E1 = np.roll(Qh, -1, axis=0)
    N = np.stack([E0[:, 1] - E1[:, 1], E1[:, 0] - E0[:, 0]], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    sc = Outside @ N.T - (E0 * N).sum(1)[None, :]
    px = Outside[:, 0:1] - sc * N[:, 0][None, :]
    py = Outside[:, 1:2] - sc * N[:, 1][None, :]
    okx = ((E0[:, 0] <= px) & (px <= E1[:, 0])) | ((E0[:, 0] >= px) & (px >= E1[:, 0]))
    oky = ((E0[:, 1] <= py) & (py <= E1[:, 1])) | ((E0[:, 1] >= py) & (py >= E1[:, 1]))
    L = np.abs(sc); L[~(okx | oky)] = np.inf
    be = L.argmin(1); valid = np.isfinite(L.min(1))
    res2 = np.stack([px[np.arange(len(Outside)), be] - Outside[:, 0],
                     py[np.arange(len(Outside)), be] - Outside[:, 1]], 1)
    res2[~valid] = np.inf
    take = np.linalg.norm(res1, axis=1) > np.linalg.norm(res2, axis=1)
    res = res1.copy(); res[take] = res2[take]
    return res


def global_theorem(Pts, box):
    (t1, v1, t2, v2, a), eps = box_mid_eps(box)
    P = Pts @ M(t1, v1).T @ R2(a).T
    Q = Pts @ M(t2, v2).T
    qh = hull_idx(Q); ph = hull_idx(P)
    Qh = Q[qh]; Ph = P[ph]
    inside = Path(Qh).contains_points(Ph)
    Outside = Ph[~inside]
    if len(Outside) == 0:
        return None
    w = w_candidates(Outside, Qh)
    w = w[w[:, 0] >= 0]
    if len(w) == 0:
        return None
    w = w[np.argsort(-np.linalg.norm(w, axis=1))]
    Rm = R2(a) @ M(t1, v1)
    Ra = R_a(a) @ M(t1, v1); Rt = R2(a) @ M_t(t1, v1); Rp = R2(a) @ M_p(t1, v1)
    M2 = M(t2, v2); M2t = M_t(t2, v2); M2p = M_p(t2, v2)
    for vec in w[:4]:
        if np.all(vec == 0) or not np.all(np.isfinite(vec)):
            return None
        vec = vec / np.linalg.norm(vec)
        s = np.argmax((Pts @ Rm.T) @ vec)
        S = Pts[s]
        G = (Rm @ S) @ vec - eps * abs((Ra @ S) @ vec) - eps * abs((Rt @ S) @ vec) \
            - eps * abs((Rp @ S) @ vec) - 4.5 * eps ** 2
        hv = Pts @ M2.T @ vec + eps * np.abs(Pts @ M2t.T @ vec) + eps * np.abs(Pts @ M2p.T @ vec) + 2 * eps ** 2
        if G > hv.max() + 1e-7:
            return ('global', vec, s)
    return None


def local_oracle(Pts, box, rho):
    (t1, v1, t2, v2, a), eps = box_mid_eps(box)
    g = 2 * SQ2 * rho ** 2 * eps + 2 * rho ** 2 * eps ** 2
    M1 = M(t1, v1); M2 = M(t2, v2)
    P = Pts @ M1.T @ R2(a).T
    Q = Pts @ M2.T
    X1 = X(t1, v1); X2 = X(t2, v2)
    pc = np.sort(hull_idx(P)); qc = np.sort(hull_idx(Q))
    r = (np.linalg.norm(Q[qc], axis=1).min() - 1.42 * rho * eps) / rho * 0.99
    r = np.floor(r * 1000) / 1000
    if r <= 0:
        return None
    margin = 1e-7
    pc = pc[(Pts[pc] @ X1) > 1.42 * rho * eps + margin]
    qc = qc[np.abs(Pts[qc] @ X2) > 1.42 * rho * eps + margin]
    if len(pc) < 3 or len(qc) < 3:
        return None
    d2 = ((P[pc][:, None, :] - Q[qc][None, :, :]) ** 2).sum(-1)
    pq = qc[d2.argmin(1)]
    delta = np.linalg.norm(P[pc] - Q[pq], axis=1) / 2
    typ = np.where(Pts[pq] @ X2 > 0, 1, -1)
    maxd = np.empty(len(pc))
    for i, q in enumerate(pq):
        tries = np.arange(len(Pts)) != q
        noms = Q[q] @ Q[q] - Q[tries] @ Q[q] - np.linalg.norm(Pts[tries] - Pts[q], axis=1) * (2 * SQ2 * rho * eps + 2 * rho * eps ** 2)
        noms = noms - 1e-6
        dens = (np.linalg.norm(Q[q]) + 1.42 * rho * eps) * (np.linalg.norm(Q[q] - Q[tries], axis=1) + 2.84 * rho * eps)
        fr = noms / dens * 0.95
        maxd[i] = min(np.inf, ((fr - 4.5 * eps / (2 * r)) * (r * rho)).min()) * 0.95
    keep = maxd >= delta
    pc, pq, delta, typ, maxd = pc[keep], pq[keep], delta[keep], typ[keep], maxd[keep]
    n = len(pc)
    if n < 3:
        return None
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]   # = <R(pi/2) u, v>
    for i in range(n - 1):
        for j in range(i + 1, n):
            if cr(P[pc[i]], P[pc[j]]) <= g + margin:
                continue
            for k in range(i + 1, n):
                if not (typ[i] == typ[j] == typ[k]):
                    continue
                if cr(P[pc[j]], P[pc[k]]) <= g + margin or cr(P[pc[k]], P[pc[i]]) <= g + margin:
                    continue
                q1, q2, q3 = pq[i], pq[j], pq[k]
                if cr(Q[q1], Q[q2]) <= g + margin or cr(Q[q2], Q[q3]) <= g + margin or cr(Q[q3], Q[q1]) <= g + margin:
                    continue
                if max(delta[[i, j, k]]) > min(maxd[[i, j, k]]):
                    continue
                A1 = Pts[[pc[i], pc[j], pc[k]]]; A2 = Pts[[q1, q2, q3]]
                try:
                    L = np.linalg.solve(A1, A2)
                except np.linalg.LinAlgError:
                    continue
                if np.abs(L @ L.T - np.eye(3)).max() > 1e-5:
                    continue
                return (pc[i], pc[j], pc[k], q1, q2, q3, 1, typ[i], r)
    return None


def local_theorem(Pts, box, rho):
    res = local_oracle(Pts, box, rho)
    if res is None:
        return None
    p1, p2, p3, q1, q2, q3, s_p, s_q, r = res
    (t1, v1, t2, v2, a), eps = box_mid_eps(box)
    M1 = M(t1, v1); M2 = M(t2, v2); Ra = R2(a)
    X1 = X(t1, v1); X2 = X(t2, v2)
    Pi = Pts[[p1, p2, p3]]; Qi = Pts[[q1, q2, q3]]
    rr = [(Ra @ M1 @ Pi[k] - M2 @ Qi[k]) / 2 for k in range(3)]
    delta = max(np.linalg.norm(x) for x in rr)
    g = 2 * SQ2 * rho ** 2 * eps + 2 * rho ** 2 * eps ** 2
    L = np.linalg.solve(Pi, Qi)
    if np.abs(L @ L.T - np.eye(3)).max() > 1e-5:
        return None
    if np.any(s_p * (Pi @ X1) <= SQ2 * rho * eps) or np.any(s_q * (Qi @ X2) <= SQ2 * rho * eps):
        return None
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]
    PP = Pi @ M1.T; QQ = Qi @ M2.T
    for (u, v) in ((0, 1), (1, 2), (2, 0)):
        if cr(PP[u], PP[v]) <= g or cr(QQ[u], QQ[v]) <= g:
            return None
    if np.any(np.linalg.norm(QQ, axis=1) <= r * rho + 1.42 * rho * eps):
        return None
    for jdx, q in enumerate((q1, q2, q3)):
        As = np.delete(Pts, q, axis=0)
        mq = M2 @ Pts[q]
        noms = mq @ mq - (As @ M2.T) @ mq - rho * np.linalg.norm(Pts[q] - As, axis=1) * (2 * SQ2 * eps + 2 * eps ** 2)
        dens = (np.linalg.norm(mq) + 1.42 * rho * eps) * (np.linalg.norm((As - Pts[q]) @ M2.T, axis=1) + 2.84 * rho * eps)
        if np.any(noms / dens < (4.5 * rho * eps + 2 * delta) / (2 * r * rho)):
            return None
    return ('local', res)


def classify(Pts, box, rho=None):
    if rho is None:
        rho = np.linalg.norm(Pts, axis=1).max()
    g = global_theorem(Pts, box)
    if g is not None:
        return 1
    l = local_theorem(Pts, box, rho)
    if l is not None:
        return 2
    return 0


def noperthedron():
    def Rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    C = [np.array([152024884, 0, 210152163]) / 259375205,
         np.array([6632738028, 6106948881, 3980949609]) / 1e10,
         np.array([8193990033, 5298215096, 1230614493]) / 1e10]
    P = np.zeros((90, 3))
    for i in range(3):
        for k in range(15):
            for l in range(2):
                P[k + 15 * i + 45 * l] = (-1) ** l * Rz(2 * np.pi * k / 15) @ C[i]
    return P
