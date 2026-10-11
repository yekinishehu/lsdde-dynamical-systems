"""Section 5.7: validation of the period-2 line-search failure cycle (Table tab:cycle, Figure fig_cycle).

Model: E(x) = t x^2 / 2, M = 1, g_2 = 0 (Algorithm 1 with sigma=1).
Parameters: lambda_max=2, rho=0.5, N_max=5, b_1=1e-3, b_2=0, omega=0.1, eta=0.3.
"""
import numpy as np
import matplotlib.pyplot as plt

PARAMS = dict(lam_max=2.0, shr=0.5, Nmax=5, b1=1e-3, b2=0.0, omega=0.1, eta=0.3)


def simulate(t, x0, xm0, N=4000, **kw):
    """Exact simulation of Algorithm 1 on the 1-D quadratic; scale-invariant renormalization."""
    P = dict(PARAMS); P.update(kw)
    grid = P['lam_max'] * P['shr'] ** np.arange(P['Nmax'])
    alpha = 1.0 / (1.0 + t)
    x = float(x0); xm = float(xm0); beta = 0.0
    lams = np.full(N, np.nan); xs = np.zeros(N + 1); rs = np.full(N, np.nan)
    xs[0] = x
    for n in range(N):
        if max(abs(x), abs(xm)) < 1e-150:      # acceptance test is homogeneous in (x, xm)
            x *= 1e150; xm *= 1e150
        y = x + beta * (x - xm)
        xb = alpha * y
        d = xb - x
        if d == 0.0:
            break
        nu = P['omega'] / (n + 1) * d * d
        lam = 0.0
        for l in grid:
            if l > 0 and (t * (xb / d) + t * l / 2.0 + P['eta']) * l * d * d <= nu:
                lam = l
                break
        lams[n] = lam; rs[n] = xb / d
        xn = xb + lam * d
        beta = 1.0 / (1.0 + P['b1'] + lam) if lam > 0 else P['b2']
        xm, x = x, xn
        xs[n + 1] = x
    return lams, xs, rs


def cycle_formulas(t, lam_star, **kw):
    """Exact closed forms from Appendix A: c(t), m_2(t), r_fail."""
    P = dict(PARAMS); P.update(kw)
    beta_star = 1.0 / (1.0 + P['b1'] + lam_star)
    K = beta_star + (1.0 + beta_star) * lam_star
    m2 = (1.0 - K * t) / (1.0 + t) ** 2
    c = (1.0 - lam_star * t) / (1.0 + t)
    return dict(c=c, m2=m2, r_fail=m2 / (m2 - c))


def cycle_observed(lams, lam_star):
    last = lams[-200:]
    if np.isnan(last).any():
        return False
    n = len(last)
    p1 = np.mean(np.abs(last - np.resize(np.array([lam_star, 0.0]), n)))
    p2 = np.mean(np.abs(last - np.resize(np.array([0.0, lam_star]), n)))
    return min(p1, p2) < 1e-12


def effective_lam_star(t, seed_count=3):
    """lambda* as the modal positive accepted step in the simulation tail (both strict and
    boundary cases, e.g. t=0.7 where g(lambda*;-1/t)=0 passes)."""
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(seed_count):
        lams, _, _ = simulate(t, *rng.standard_normal(2))
        lw = lams[3000:4000]; lw = lw[~np.isnan(lw) & (lw > 0)]
        vals.extend(lw.tolist())
    u, cnt = np.unique(vals, return_counts=True)
    return u[cnt.argmax()]


def table_main():
    rng = np.random.default_rng(42)
    configs = [0.25, 0.35, 0.5, 0.8, 1.0, 2.0, 4.0, 8.0]
    print(f"{'t':>5} {'lam*':>5} {'|m2| pred':>10} {'2-step meas':>12} {'fail rate':>10} "
          f"{'r_fail pred':>12} {'r_fail meas':>12} {'r_succ':>9} {'cycle':>7}")
    for t in configs:
        lam_star = effective_lam_star(t)
        p = cycle_formulas(t, lam_star)
        fr, ts, rf, rsucc, cyc = [], [], [], [], 0
        for _ in range(30):
            lams, xs, rs = simulate(t, *rng.standard_normal(2))
            lw = lams[3500:4000]; lw = lw[~np.isnan(lw)]
            fr.append(np.mean(lw == 0.0) if len(lw) else np.nan)
            idx = np.where((lams[:-1] == lam_star) & (lams[1:] == 0.0))[0]
            idx = idx[(idx > 200) & (idx < 3999)]
            good = [k for k in idx if abs(xs[k]) > 1e-120]
            if len(good) >= 50:
                ts.append(np.mean([abs(xs[k + 2] / xs[k]) for k in good]))
                rf.append(np.mean([rs[k + 1] for k in good]))
                rsucc.append(np.mean([rs[k] for k in good]))
            cyc += cycle_observed(lams, lam_star)
        print(f"{t:>5} {lam_star:>5.3f} {abs(p['m2']):>10.5f} {np.nanmean(ts):>12.5f} "
              f"{np.nanmean(fr):>10.3f} {p['r_fail']:>12.4f} {np.nanmean(rf):>12.4f} "
              f"{np.nanmean(rsucc):>9.4f} {cyc:>4d}/30")


def margin_threshold():
    """Threshold t_hat where the margin condition g(lambda_ls; r_fail) = 0 (Appendix A)."""
    grid = PARAMS['lam_max'] * PARAMS['shr'] ** np.arange(PARAMS['Nmax'])
    def g(t):
        below = grid[t * grid < 2 * (1 - PARAMS['eta'])]
        if len(below) == 0:
            return 1.0
        p = cycle_formulas(t, below.max())
        return t * p['r_fail'] + t * 0.125 / 2 + PARAMS['eta']
    lo, hi = 0.05, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0: hi = mid
        else: lo = mid
    print(f"margin threshold t_hat ~ {0.5*(lo+hi):.4f} (paper: 0.1597)")
    rng = np.random.default_rng(7)
    for t in [0.10, 0.15, 0.16, 0.20, 0.25, 0.70]:
        lam_star = effective_lam_star(t)
        cyc = 0; rates = []
        for _ in range(30):
            lams, _, _ = simulate(t, *rng.standard_normal(2))
            cyc += cycle_observed(lams, lam_star)
            lw = lams[3500:4000]; lw = lw[~np.isnan(lw)]
            rates.append(np.mean(lw == 0.0) if len(lw) else np.nan)
        print(f"  t={t:<5} lam*={lam_star:<5} cycle {cyc}/30 fail_rate={np.nanmean(rates):.3f}")


def two_dimensional():
    """Commuting 2-D model t=(1,1/16): period-3 orbit (2,2,0) with failure rate 1/3."""
    tvec = np.array([1.0, 1.0 / 16])
    alpha = 1.0 / (1.0 + tvec)
    grid = PARAMS['lam_max'] * PARAMS['shr'] ** np.arange(PARAMS['Nmax'])
    rng = np.random.default_rng(11)
    x = rng.standard_normal(2); xm = rng.standard_normal(2); beta = 0.0
    lams = []
    for n in range(6000):
        y = x + beta * (x - xm)
        xb = alpha * y
        d = xb - x
        dd = d @ d
        if dd == 0: break
        nu = PARAMS['omega'] / (n + 1) * dd
        lam = 0.0
        for l in grid:
            if l > 0 and (tvec @ (xb * d) + (l / 2.0) * (tvec @ (d * d)) + PARAMS['eta'] * dd) * l <= nu:
                lam = l; break
        lams.append(lam)
        xn = xb + lam * d
        beta = 1.0 / (1.0 + PARAMS['b1'] + lam) if lam > 0 else PARAMS['b2']
        xm, x = x, xn
    lams = np.array(lams)
    tail = lams[-4000:]
    u, cnt = np.unique(tail[~np.isnan(tail)], return_counts=True)
    print("2D t=(1,1/16): step histogram in last 4000 iters:", dict(zip(u.tolist(), cnt.tolist())),
          "(period-3 orbit (2,2,0), failure rate 1/3)")


def constant_step_check():
    """Constant-step variant on hard coordinate t=1/16: measured contraction vs frozen radius."""
    t = 1.0 / 16; lam = 1.0; b1 = 0.25
    alpha = 1.0 / (1.0 + t); beta = 1.0 / (1.0 + b1 + lam)
    S = (1 + lam) * alpha * (1 + beta) - lam
    P = (1 + lam) * alpha * beta
    rho_frozen = max(abs(np.roots([1, -S, P])))
    x, xm = 0.7, -0.3
    hist = [x]
    for _ in range(1500):
        y = x + beta * (x - xm)
        xb = alpha * y
        xn = xb + lam * (xb - x)
        xm, x = x, xn
        hist.append(x)
    hist = np.array(hist)
    sel = np.arange(200, 1200)
    slope = np.polyfit(sel, np.log(np.abs(hist[sel])), 1)[0]
    print(f"constant step t=1/16: frozen rho = {rho_frozen:.5f} | measured = {np.exp(slope):.5f}")


def make_figure(path="figures/fig_cycle.pdf"):
    lams, xs, _ = simulate(1.0, 0.7, -0.3, N=80)
    m2_abs = 0.5 / 2.001
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].step(range(60), lams[:60], where='post', color='tab:blue')
    ax[0].set_xlabel(r'iteration $n$'); ax[0].set_ylabel(r'accepted step $\lambda_n$')
    ax[0].set_title(r'(a) step sequence: period-2 cycle $(\lambda^*,0)$, $\lambda^*=1$')
    ax[0].set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    k = np.arange(0, 61, 2)
    ax[1].semilogy(k, np.abs(xs[k]), 'o-', color='tab:blue', label=r'$\|x^n\|$, even phase')
    ax[1].semilogy(np.arange(0, 61), 0.7 * np.sqrt(m2_abs) ** np.arange(0, 61), '--',
                   color='tab:red', label=r'slope $|m_2|^{n/2}$, $|m_2|=0.24988$')
    ax[1].set_xlabel(r'iteration $n$'); ax[1].legend()
    ax[1].set_title(r'(b) two-step contraction $|m_2|$')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close(fig)
    print("figure saved:", path)


if __name__ == "__main__":
    table_main()
    margin_threshold()
    two_dimensional()
    constant_step_check()
    make_figure()
