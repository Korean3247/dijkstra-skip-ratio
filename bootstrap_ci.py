"""
bootstrap_ci.py
===============
Computes bootstrap confidence intervals for the power-law fit parameters
c and alpha in: skip_ratio(k) ≈ 1 - c / k^alpha

Uses the existing skip_ratio_by_dist CSV (theory_probe_v4 output).
Bootstrap: resample k-values with replacement, refit OLS each time.
Reports 95% CI for c and alpha.
"""

import csv
import math
import random
import os
import glob
from datetime import datetime

# ── load data ──────────────────────────────────────────────────────────
data_dir = "/Users/gwpark/Downloads/Algorithm Research/resultsTheory"
csv_files = sorted(glob.glob(f"{data_dir}/skip_ratio_by_dist_*.csv"))
if not csv_files:
    raise FileNotFoundError("No skip_ratio_by_dist CSV found")
csv_path = csv_files[-1]
print(f"Loading: {csv_path}")

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# ── helper: OLS fit in log-log ─────────────────────────────────────────
def fit_powerlaw(k_vals, sr_vals):
    """
    Fit skip_ratio = 1 - c/k^alpha via OLS on log(1-sr) = log(c) - alpha*log(k)
    Returns (c, alpha, R2)
    """
    n = len(k_vals)
    X = [math.log(k) for k in k_vals]
    Y = [math.log(1 - sr) for sr in sr_vals]
    Xm = sum(X)/n
    Ym = sum(Y)/n
    Sxx = sum((x-Xm)**2 for x in X)
    Sxy = sum((x-Xm)*(y-Ym) for x,y in zip(X,Y))
    alpha = -Sxy / Sxx
    log_c = Ym + alpha * Xm
    c = math.exp(log_c)
    # R2
    Y_pred = [log_c - alpha*x for x in X]
    ss_res = sum((y-yp)**2 for y,yp in zip(Y,Y_pred))
    ss_tot = sum((y-Ym)**2 for y in Y)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return c, alpha, R2

# ── for each distribution, fit on k∈[4,64] and bootstrap ──────────────
distributions = list(dict.fromkeys(r['distribution'] for r in rows))
K_RANGE = (4, 64)  # practical range used in paper
B = 10000          # bootstrap iterations
RNG_SEED = 42

results = {}
for dist in distributions:
    dist_rows = [r for r in rows
                 if r['distribution'] == dist
                 and K_RANGE[0] <= int(float(r['k'])) <= K_RANGE[1]]
    k_vals = [float(r['k']) for r in dist_rows]
    sr_vals = [float(r['skip_ratio_mean']) for r in dist_rows]
    n = len(k_vals)

    # point estimate
    c_hat, alpha_hat, r2_hat = fit_powerlaw(k_vals, sr_vals)

    # bootstrap
    rng = random.Random(RNG_SEED)
    boot_c = []
    boot_alpha = []
    for _ in range(B):
        idx = [rng.randint(0, n-1) for _ in range(n)]
        bk = [k_vals[i] for i in idx]
        bsr = [sr_vals[i] for i in idx]
        # need at least 2 distinct k values for meaningful fit
        if len(set(bk)) < 2:
            continue
        try:
            bc, ba, _ = fit_powerlaw(bk, bsr)
            boot_c.append(bc)
            boot_alpha.append(ba)
        except (ValueError, ZeroDivisionError):
            pass

    boot_c.sort()
    boot_alpha.sort()
    ci_lo = int(0.025 * len(boot_c))
    ci_hi = int(0.975 * len(boot_c))

    results[dist] = {
        'dist': dist,
        'n_k': n,
        'c': round(c_hat, 4),
        'alpha': round(alpha_hat, 4),
        'R2': round(r2_hat, 4),
        'c_ci_lo': round(boot_c[ci_lo], 4),
        'c_ci_hi': round(boot_c[ci_hi], 4),
        'alpha_ci_lo': round(boot_alpha[ci_lo], 4),
        'alpha_ci_hi': round(boot_alpha[ci_hi], 4),
        'boot_n': len(boot_c),
    }

# ── also compute OLS standard errors (analytical) ──────────────────────
def fit_with_se(k_vals, sr_vals):
    n = len(k_vals)
    X = [math.log(k) for k in k_vals]
    Y = [math.log(1 - sr) for sr in sr_vals]
    Xm = sum(X)/n
    Ym = sum(Y)/n
    Sxx = sum((x-Xm)**2 for x in X)
    Sxy = sum((x-Xm)*(y-Ym) for x,y in zip(X,Y))
    alpha = -Sxy / Sxx
    log_c = Ym + alpha * Xm
    Y_pred = [log_c - alpha*x for x in X]
    ss_res = sum((y-yp)**2 for y,yp in zip(Y,Y_pred))
    s2 = ss_res / (n - 2) if n > 2 else 0
    se_alpha = math.sqrt(s2 / Sxx) if Sxx > 0 else 0
    se_logc  = math.sqrt(s2 * (1/n + Xm**2/Sxx)) if Sxx > 0 else 0
    return alpha, se_alpha, math.exp(log_c), se_logc

# ── print results ───────────────────────────────────────────────────────
print(f"\nBootstrap 95% CI  (B={B}, k∈[{K_RANGE[0]},{K_RANGE[1]}])\n")
print(f"{'Distribution':<28} {'c':>6} {'α':>6} {'R²':>6}  "
      f"{'CI(c)':^16}  {'CI(α)':^16}")
print("-"*80)
for dist, r in results.items():
    print(f"{dist:<28} {r['c']:>6.4f} {r['alpha']:>6.4f} {r['R2']:>6.4f}  "
          f"[{r['c_ci_lo']:.4f},{r['c_ci_hi']:.4f}]  "
          f"[{r['alpha_ci_lo']:.4f},{r['alpha_ci_hi']:.4f}]")

# OLS SE
print(f"\nOLS Standard Errors  (k∈[{K_RANGE[0]},{K_RANGE[1]}])\n")
for dist in distributions:
    dist_rows = [r for r in rows
                 if r['distribution'] == dist
                 and K_RANGE[0] <= int(float(r['k'])) <= K_RANGE[1]]
    k_vals = [float(r['k']) for r in dist_rows]
    sr_vals = [float(r['skip_ratio_mean']) for r in dist_rows]
    alpha, se_alpha, c, se_logc = fit_with_se(k_vals, sr_vals)
    print(f"{dist:<28}  c={c:.4f}±{se_logc*c:.4f}  α={alpha:.4f}±{se_alpha:.4f}")

# ── save CSV ─────────────────────────────────────────────────────────────
out_dir = "/Users/gwpark/Downloads/Algorithm Research/resultsTheory"
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = f"{out_dir}/bootstrap_ci_{ts}.csv"
with open(out_path, 'w', newline='') as f:
    fieldnames = ['dist','n_k','c','alpha','R2',
                  'c_ci_lo','c_ci_hi','alpha_ci_lo','alpha_ci_hi','boot_n']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in results.values():
        w.writerow(r)
print(f"\nCSV: {out_path}")

# ── primary result for paper ──────────────────────────────────────────
r = results.get('U[1,100]    (original)', list(results.values())[0])
print(f"\n=== PRIMARY RESULT (U[1,100], k∈[4,64]) ===")
print(f"  c     = {r['c']:.4f}  95% CI [{r['c_ci_lo']:.4f}, {r['c_ci_hi']:.4f}]")
print(f"  alpha = {r['alpha']:.4f}  95% CI [{r['alpha_ci_lo']:.4f}, {r['alpha_ci_hi']:.4f}]")
print(f"  R2    = {r['R2']:.4f}")
