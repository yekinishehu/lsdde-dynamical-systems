"""Section 5.4: variant comparison on synthetic SCAD-LS (Table tab:variant-comp).

m=50, d=100, kappa(A^T A)=1e2, SCAD lambda=0.1, a=3.7; 5 random instances.
Variants: line search (Algorithm 1), fixed-trial lambda=0.9 with fallback,
constant-step lambda=0.9 without Armijo check.
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


def make_A(m, d, kappa, rng):
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.geomspace(1.0, 1.0 / np.sqrt(kappa), m)
    return U * s @ V[:m]


def fista(y0, G, c, Mdiag, lam, xi, L, iters=80):
    y = y0.copy(); z = y0.copy(); tk = 1.0
    for _ in range(iters):
        g = G @ z - c + Mdiag * (z - y0) - xi
        ynew = np.sign(z - g / L) * np.maximum(np.abs(z - g / L) - lam / L, 0)
        tnew = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        z = ynew + ((tk - 1) / tnew) * (ynew - y)
        y = ynew; tk = tnew
    return y


def run(mode, kappa=1e2, lam=0.1, a=3.7, lam_fix=0.9, N=400, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    m, d = 50, 100
    A = make_A(m, d, kappa, rng)
    G = A.T @ A
    xt = np.where(rng.standard_normal(d) > 1.15, rng.standard_normal(d), 0.0)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    c = A.T @ b
    del A
    Mdiag = np.diag(G) + lam
    Lmax = np.linalg.norm(G, 2) + Mdiag.max()
    xi_g = lambda x: lam * np.sign(x) - scad_dp(x, lam, a)
    dist = lambda x: np.linalg.norm(np.where(x != 0, G @ x - c - xi_g(x) + lam * np.sign(x),
                                             np.clip(np.abs(G @ x - c - xi_g(x)) - lam, 0, None)))
    E = lambda x: 0.5 * x @ (G @ x) - c @ x + scad_p(x, lam, a).sum()
    x = rng.standard_normal(d); xm = x.copy(); beta = 0.0; n_eval = 0; brk = 'cap'
    lam_trial = 2.0 * 0.5 ** np.arange(5)
    for n in range(N):
        y = x + beta * (x - xm)
        xb = fista(y, G, c, Mdiag, lam, xi_g(x), Lmax)
        dd = xb - x; dd2 = dd @ dd
        if dd2 < 1e-16: brk = 'dzero'; break
        nu = 0.1 / (n + 1) * dd2
        if mode == 'const':
            lams, ne = lam_fix, 1
        else:
            trials = [lam_fix] if mode == 'fixed' else lam_trial
            Eb = E(xb); lams, ne = 0.0, 1
            for l in trials:
                ne += 1
                if E(xb + l * dd) <= Eb - 0.3 * l * dd2 + nu: lams = l; break
        n_eval += ne
        xn = xb + lams * dd
        beta = 1 / (1 + 1e-3 + lams) if lams > 0 else 0.0
        xm, x = x, xn
        dv = dist(x)
        if dv <= tol: brk = 'conv'; break
        if dv > 1e3: brk = 'blowup'; break
    return dict(iters=n + 1, dist=dist(x), n_eval=n_eval, brk=brk)


if __name__ == "__main__":
    for mode, name in [('ls', 'line search'), ('fixed', 'fixed-trial 0.9'), ('const', 'constant-step 0.9')]:
        rs = [run(mode, seed=s) for s in range(5)]
        brks = [r['brk'] for r in rs]
        print(f"{name:>18}: iters={np.mean([r['iters'] for r in rs]):6.0f}+-{np.std([r['iters'] for r in rs]):4.0f} "
              f" E-evals={np.mean([r['n_eval'] for r in rs]):6.0f}  dist={np.mean([r['dist'] for r in rs]):.2e}  {brks}")
