"""Section 5.8: Moreau-smoothed capped-l1 least squares (Table tab:smooth-g2).

min_x (1/2m)||Ax-b||^2 + mu sum_i min(|x_i|, theta)   with g_2 = mu sum_i (|x_i|-theta)_+ NONSMOOTH.
Run Algorithm 1 on E_rho = f + g_1 - e_rho(g_2) (Theorem 4.28) and verify the transfer
inequality dist(0, dE(u)) <= dist(0, dE_rho(x)) + L_f Lip(g_2)/rho at u = prox_{g_2/rho}(x).
m=50, d=100, kappa=1e2, mu=0.05, theta=0.5; 3 instances per rho.
"""
import numpy as np


def scad_dp_unused(t, lam, a):  # not used here; kept for uniformity
    return np.zeros_like(t)


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


def run(rho, mu=0.05, theta=0.5, kappa=1e2, N=400, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    m, d = 50, 100
    A = make_A(m, d, kappa, rng)
    G = A.T @ A
    xt = np.where(rng.standard_normal(d) > 1.15, rng.standard_normal(d), 0.0)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    c = A.T @ b
    del A
    Mdiag = np.diag(G) + mu
    Lmax = np.linalg.norm(G, 2) + Mdiag.max()

    def prox(v):  # prox of g_2/rho, componentwise
        t = np.abs(v)
        w = np.where(t < theta, t, np.where(t <= theta + mu / rho, theta, t - mu / rho))
        return np.sign(v) * w

    xi_g = lambda x: rho * (x - prox(x))           # grad of the Moreau envelope

    def env_g2(x):
        u = prox(x)
        return mu * np.maximum(np.abs(u) - theta, 0).sum() + 0.5 * rho * np.sum((x - u) ** 2)

    dist_rho = lambda x: np.linalg.norm(np.where(x != 0, G @ x - c - xi_g(x) + mu * np.sign(x),
                                                 np.clip(np.abs(G @ x - c - xi_g(x)) - mu, 0, None)))
    E_rho = lambda x: 0.5 * x @ (G @ x) - c @ x + mu * np.abs(x).sum() - env_g2(x)
    x = rng.standard_normal(d); xm = x.copy(); beta = 0.0
    lam_trial = 2.0 * 0.5 ** np.arange(5); n_eval = 0
    for n in range(N):
        y = x + beta * (x - xm)
        xb = fista(y, G, c, Mdiag, mu, xi_g(x), Lmax)
        dd = xb - x; dd2 = dd @ dd
        if dd2 < 1e-16: break
        nu = 0.1 / (n + 1) * dd2
        Eb = E_rho(xb); lams = 0.0
        for l in lam_trial:
            n_eval += 1
            if E_rho(xb + l * dd) <= Eb - 0.3 * l * dd2 + nu: lams = l; break
        xn = xb + lams * dd
        beta = 1 / (1 + 1e-3 + lams) if lams > 0 else 0.0
        xm, x = x, xn
        if dist_rho(x) <= tol: break
    u = prox(x)
    zeta = mu * np.sign(u) * (np.abs(u) > theta)
    v = G @ u - c - zeta
    dist_true = np.linalg.norm(np.where(u != 0, v + mu * np.sign(u), np.clip(np.abs(v) - mu, 0, None)))
    bias = np.linalg.norm(G, 2) * mu * np.sqrt(d) / rho
    return dict(iters=n + 1, dist_rho=dist_rho(x), gap=np.linalg.norm(x - u),
                bias=bias, dist_true=dist_true, holds=dist_true <= dist_rho(x) + bias)


if __name__ == "__main__":
    for rho in [1.0, 10.0, 100.0]:
        rs = [run(rho, seed=s) for s in range(3)]
        print(f"rho={rho:>5}: iters={np.mean([r['iters'] for r in rs]):5.0f}  "
              f"dist(dE_rho)={np.mean([r['dist_rho'] for r in rs]):.2e}  "
              f"||x-u||={rs[0]['gap']:.2e}  bound={rs[0]['bias']:.2e}  "
              f"dist_true={np.mean([r['dist_true'] for r in rs]):.2e}  holds={all(r['holds'] for r in rs)}")
