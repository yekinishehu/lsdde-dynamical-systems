"""Section 5.4: full-scale variant confirmation (Table tab:variant-comp-full).

SCAD-LS m=2000, d=5000, kappa(A^T A)=1e3, mu=5e-4, a=3.7, s=50 nonzeros.
Lean inner solves (8 warm-started FISTA iterations per subproblem) -> iteration counts
are NOT comparable with Table 2 (tuned implementation); the table reports the
stationarity trajectory dist(0, dE(x^n)) every 25 iterations.
"""
import numpy as np


def scad_p(t, lam, a):
    at = np.abs(t)
    return np.where(at <= lam, lam * at,
           np.where(at <= a * lam, (2 * a * lam * at - at**2 - lam**2) / (2 * (a - 1)),
                    (a + 1) * lam**2 / 2))


def scad_dp(t, lam, a):
    s = np.sign(t); at = np.abs(t)
    return s * np.where(at <= lam, lam, np.where(at <= a * lam, (a * lam - at) / (a - 1), 0.0))


def fista(y0, G, c, Mdiag, lam, xi, L, iters=8):
    y = y0.copy(); z = y0.copy(); tk = 1.0
    for _ in range(iters):
        g = G @ z - c + Mdiag * (z - y0) - xi
        ynew = np.sign(z - g / L) * np.maximum(np.abs(z - g / L) - lam / L, 0)
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        z = ynew + ((tk - 1) / tnew) * (ynew - y)
        y = ynew; tk = tnew
    return y


def run_hist(mode, lam, a=3.7, lam_fix=0.9, N=250, seed=0):
    rng = np.random.default_rng(seed)
    m, d, kap = 2000, 5000, 1e3
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.geomspace(1.0, 1.0 / np.sqrt(kap), m)
    A = U * s @ V[:m]
    G = A.T @ A
    xt = np.zeros(d); sup = rng.choice(d, 50, replace=False)
    xt[sup] = rng.standard_normal(50)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    c = A.T @ b
    del A, U, V, s
    Mdiag = np.diag(G) + lam
    v = rng.standard_normal(d)
    for _ in range(12):
        v = G @ v; v /= np.linalg.norm(v)
    Lmax = np.linalg.norm(G @ v) + Mdiag.max()
    xi_g = lambda x: lam * np.sign(x) - scad_dp(x, lam, a)
    dist = lambda x: np.linalg.norm(np.where(x != 0, G @ x - c - xi_g(x) + lam * np.sign(x),
                                             np.clip(np.abs(G @ x - c - xi_g(x)) - lam, 0, None)))
    E = lambda x: 0.5 * x @ (G @ x) - c @ x + scad_p(x, lam, a).sum()
    x = rng.standard_normal(d); xm = x.copy(); beta = 0.0
    lam_trial = 2.0 * 0.5 ** np.arange(5)
    hist = []
    for n in range(N):
        y = x + beta * (x - xm)
        xb = fista(y, G, c, Mdiag, lam, xi_g(x), Lmax)
        dd = xb - x; dd2 = dd @ dd
        if dd2 < 1e-16: break
        nu = 0.1 / (n + 1) * dd2
        if mode == 'const':
            lams = lam_fix
        else:
            Eb = E(xb); lams = 0.0
            for l in ([lam_fix] if mode == 'fixed' else lam_trial):
                if E(xb + l * dd) <= Eb - 0.3 * l * dd2 + nu: lams = l; break
        xn = xb + lams * dd
        beta = 1 / (1 + 1e-3 + lams) if lams > 0 else 0.0
        xm, x = x, xn
        if (n + 1) % 25 == 0 or n == N - 1:
            hist.append((n + 1, dist(x)))
    return hist


if __name__ == "__main__":
    for mode in ['fixed', 'const']:
        h = run_hist(mode, lam=5e-4)
        print(mode, ":", " ".join(f"{n}:{d:.2e}" for n, d in h))
