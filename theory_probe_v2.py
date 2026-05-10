"""
Theory Probe v2: α의 분포 의존성 체계적 분석
══════════════════════════════════════════════════════════════════
핵심 질문: skip_ratio(k) ≈ 1 - c/k^α 에서 α를 결정하는
           가중치 분포의 통계량은 무엇인가?

실험 1: U[1, W_max] → α(W_max) 함수형 매핑
  W_max ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}

실험 2: Exp(mean) → α(mean) 함수형 매핑
  mean ∈ {2, 5, 10, 20, 50, 100, 200}  (truncated to ≥1)

실험 3: 동일 분산 비교 (분산이 α를 결정하는지 검증)
  Pair: U[1,W]   vs   Exp(σ=W/√12) — 같은 표준편차, 다른 분포형

실험 4: 이봉분포 (Two-point) — 분포 형태의 역할
  P(W=1) = P(W=W_high) = 0.5  for W_high ∈ {10, 100, 1000}

모든 실험:
  k ∈ {4, 8, 16, 32, 64, 128}
  V = 3000, N_REPS = 30 (통계 안정성 확보)
"""

import heapq
import random
import math
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("resultsTheory")
RESULTS_DIR.mkdir(exist_ok=True)

V      = 3000
N_REPS = 30
SEED   = 42
K_VALS = [4, 8, 16, 32, 64, 128]


# ── Graph & Dijkstra ─────────────────────────────────────────────────

def make_graph(V, k_target, weight_fn, rng):
    """ER random graph with custom weight distribution."""
    adj = [[] for _ in range(V)]
    n_edges = V * k_target // 2
    for _ in range(n_edges):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            w = weight_fn(rng)
            adj[u].append((v, w))
            adj[v].append((u, w))
    # spanning path for connectivity
    nodes = list(range(V))
    rng.shuffle(nodes)
    for i in range(V - 1):
        u, v = nodes[i], nodes[i + 1]
        w = weight_fn(rng)
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def dijkstra_skip(adj, source, V):
    dist = [float("inf")] * V
    dist[source] = 0
    heap = [(0, source)]
    push_c = pop_c = skip_c = 1, 0, 0
    push_c = 1
    pop_c = 0
    skip_c = 0
    while heap:
        d, u = heapq.heappop(heap)
        pop_c += 1
        if d > dist[u]:
            skip_c += 1
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                push_c += 1
    return skip_c / pop_c if pop_c else 0.0


# ── Fitting ──────────────────────────────────────────────────────────

def fit_power_law(k_vals, sr_vals):
    """
    skip_ratio ≈ 1 - c/k^α
    log(1 - skip_ratio) = log(c) - α*log(k)
    → OLS on (log k, log(1-skip_ratio))
    Returns (c, alpha, R²)
    """
    valid = [(k, s) for k, s in zip(k_vals, sr_vals) if 0 < s < 1]
    if len(valid) < 3:
        return None, None, None
    lk  = np.array([math.log(k) for k, _ in valid])
    ly  = np.array([math.log(1.0 - s) for _, s in valid])
    slope, intercept, r, p, se = stats.linregress(lk, ly)
    alpha = -slope
    c     = math.exp(intercept)
    return c, alpha, r ** 2


# ── Experiment runner ────────────────────────────────────────────────

def run_distribution(label, weight_fn, k_vals=K_VALS, verbose=True):
    """Run one weight distribution; return dict with α, c, R², skip_ratio per k."""
    sr_per_k = []
    for k in k_vals:
        srs = []
        for rep in range(N_REPS):
            rng = random.Random(SEED + rep * 1009 + k * 7)
            adj = make_graph(V, k, weight_fn, rng)
            src = rng.randint(0, V - 1)
            srs.append(dijkstra_skip(adj, src, V))
        sr_per_k.append(float(np.mean(srs)))

    c, alpha, r2 = fit_power_law(k_vals, sr_per_k)
    if verbose:
        if alpha is None:
            print(f"  {label:<30s}  α=N/A  c=N/A  R²=N/A  ⚠ fit failed")
        else:
            flag = "" if (r2 is not None and r2 > 0.97) else "  ⚠ low R²"
            print(f"  {label:<30s}  α={alpha:.4f}  c={c:.4f}  R²={r2:.4f}{flag}")
    return {"label": label, "alpha": alpha, "c": c, "r2": r2,
            "sr_per_k": sr_per_k}


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: U[1, W_max]  →  α(W_max)
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("실험 1: U[1, W_max] — α vs W_max")
print("═"*65)

W_max_vals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
exp1_rows = []

for W in W_max_vals:
    if W == 1:
        fn = lambda rng, W=W: 1   # constant
    else:
        fn = lambda rng, W=W: rng.randint(1, W)
    row = run_distribution(f"U[1,{W}]", fn)
    row["W_max"] = W
    exp1_rows.append(row)

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Exp(mean)  →  α(mean)
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("실험 2: Exp(mean) — α vs mean  (truncated to ≥1)")
print("═"*65)

def make_exp_fn(mean):
    """Exponential distribution truncated to integers ≥ 1."""
    lam = 1.0 / mean
    def fn(rng, lam=lam):
        return max(1, int(-math.log(max(rng.random(), 1e-15)) / lam))
    return fn

exp_means = [2, 5, 10, 20, 50, 100, 200]
exp2_rows = []

for mean in exp_means:
    fn = make_exp_fn(mean)
    row = run_distribution(f"Exp(mean={mean})", fn)
    row["mean"] = mean
    exp2_rows.append(row)

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: 동일 분산 비교
#   U[1,W] has std ≈ W/√12  (for large W)
#   Match with Exp(σ) where σ = W/√12  →  mean = W/√12
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("실험 3: 동일 표준편차 비교  U[1,W] vs Exp(σ=W/√12)")
print("═"*65)

W_pairs = [10, 100, 1000]
exp3_rows = []

for W in W_pairs:
    sigma = W / math.sqrt(12)  # std of U[1,W]

    fn_u = lambda rng, W=W: rng.randint(1, W)
    r_u  = run_distribution(f"U[1,{W}]  (σ≈{sigma:.1f})", fn_u)
    r_u["group"] = W; r_u["dist_type"] = "Uniform"
    exp3_rows.append(r_u)

    fn_e = make_exp_fn(sigma)
    r_e  = run_distribution(f"Exp(σ={sigma:.1f})", fn_e)
    r_e["group"] = W; r_e["dist_type"] = "Exp_matched"
    exp3_rows.append(r_e)

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Two-point distribution
#   P(W=1) = P(W=W_high) = 0.5
#   Mean = (1+W_high)/2,  Var = (W_high-1)²/4
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("실험 4: Two-point (이봉분포)  P(W=1)=P(W=W_high)=0.5")
print("═"*65)

W_highs = [10, 100, 1000]
exp4_rows = []

for W_high in W_highs:
    fn = lambda rng, W=W_high: (1 if rng.random() < 0.5 else W)
    row = run_distribution(f"Two-point{{1,{W_high}}}", fn)
    row["W_high"] = W_high
    sigma_tp = (W_high - 1) / 2
    row["sigma"] = sigma_tp
    exp4_rows.append(row)

# ─────────────────────────────────────────────────────────────────────
# EXTRA: U[1,W] actual mean of skip_ratio per (k, W) grid for heatmap
# ─────────────────────────────────────────────────────────────────────
# (already computed in exp1_rows above — extract the sr_per_k grid)

# ─────────────────────────────────────────────────────────────────────
# ANALYSIS & STATISTICS
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("분석: α vs 분포 통계량")
print("═"*65)

# Fit α(W_max) for U[1,W]
alpha1 = [r["alpha"] for r in exp1_rows if r["alpha"] is not None and r["W_max"] > 1]
W1     = [r["W_max"] for r in exp1_rows if r["alpha"] is not None and r["W_max"] > 1]
log_W1 = np.log(W1)

slope_a, intercept_a, r_a, _, _ = stats.linregress(log_W1, alpha1)
print(f"\nU[1,W]: α ≈ {intercept_a:.4f} + {slope_a:.4f} * log(W_max)  (R={r_a:.4f})")

# Fit α(mean) for Exp
alpha2 = [r["alpha"] for r in exp2_rows if r["alpha"] is not None]
mean2  = [r["mean"]  for r in exp2_rows if r["alpha"] is not None]
log_m2 = np.log(mean2)

slope_b, intercept_b, r_b, _, _ = stats.linregress(log_m2, alpha2)
print(f"Exp(μ): α ≈ {intercept_b:.4f} + {slope_b:.4f} * log(mean)  (R={r_b:.4f})")

# Same-variance comparison
print("\n동일 표준편차 비교:")
for i in range(0, len(exp3_rows), 2):
    r1, r2 = exp3_rows[i], exp3_rows[i+1]
    print(f"  W={r1['group']:4d}: {r1['label']:<22s} α={r1['alpha']:.4f}   "
          f"{r2['label']:<22s} α={r2['alpha']:.4f}   "
          f"Δα={abs(r1['alpha']-r2['alpha']):.4f}")

# Two-point vs Uniform same-max comparison
print("\nTwo-point vs U[1,W] (같은 최대값 W_high):")
for r4 in exp4_rows:
    W = r4["W_high"]
    r1_match = next((r for r in exp1_rows if r["W_max"] == W), None)
    if r1_match:
        print(f"  W={W:4d}: Two-point α={r4['alpha']:.4f}   "
              f"U[1,{W}] α={r1_match['alpha']:.4f}   "
              f"σ_tp={r4['sigma']:.1f} vs σ_U={W/math.sqrt(12):.1f}")

# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ─ Panel (0,0): α vs W_max for U[1,W] ───────────────────────────────
ax = fig.add_subplot(gs[0, 0])
W1_plot  = [r["W_max"] for r in exp1_rows if r["alpha"] is not None and r["W_max"] > 1]
a1_plot  = [r["alpha"] for r in exp1_rows if r["alpha"] is not None and r["W_max"] > 1]

ax.scatter(W1_plot, a1_plot, s=80, color="#1565C0", zorder=5, label="실측 α")
W_fit = np.array(W1_plot)
a_fit_line = intercept_a + slope_a * np.log(W_fit)
ax.plot(W_fit, a_fit_line, "r--", lw=1.5,
        label=f"log fit: α={intercept_a:.3f}+{slope_a:.3f}·ln(W)")
ax.set_xscale("log")
ax.set_xlabel("W_max", fontsize=11)
ax.set_ylabel("α", fontsize=11)
ax.set_title("U[1,W]: α vs W_max", fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ─ Panel (0,1): α vs mean for Exp ───────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
m2_plot = [r["mean"]  for r in exp2_rows if r["alpha"] is not None]
a2_plot = [r["alpha"] for r in exp2_rows if r["alpha"] is not None]

ax.scatter(m2_plot, a2_plot, s=80, color="#E53935", zorder=5, label="실측 α")
m_fit = np.array(m2_plot)
a_fit_exp = intercept_b + slope_b * np.log(m_fit)
ax.plot(m_fit, a_fit_exp, "b--", lw=1.5,
        label=f"log fit: α={intercept_b:.3f}+{slope_b:.3f}·ln(μ)")
ax.set_xscale("log")
ax.set_xlabel("Exponential mean μ", fontsize=11)
ax.set_ylabel("α", fontsize=11)
ax.set_title("Exp(μ): α vs mean", fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ─ Panel (0,2): α vs log(σ) — ALL distributions overlaid ────────────
ax = fig.add_subplot(gs[0, 2])

# U[1,W]: σ = W/√12
sigma_u = [r["W_max"]/math.sqrt(12) for r in exp1_rows
           if r["alpha"] is not None and r["W_max"] > 1]
alpha_u = [r["alpha"] for r in exp1_rows
           if r["alpha"] is not None and r["W_max"] > 1]
ax.scatter(sigma_u, alpha_u, s=80, color="#1565C0",
           label="U[1,W]", zorder=5, marker="o")

# Exp(μ): σ = μ
sigma_e = [r["mean"] for r in exp2_rows if r["alpha"] is not None]
alpha_e = [r["alpha"] for r in exp2_rows if r["alpha"] is not None]
ax.scatter(sigma_e, alpha_e, s=80, color="#E53935",
           label="Exp(μ)", zorder=5, marker="^")

# Two-point: σ = (W_high-1)/2
sigma_t = [r["sigma"] for r in exp4_rows if r["alpha"] is not None]
alpha_t = [r["alpha"] for r in exp4_rows if r["alpha"] is not None]
ax.scatter(sigma_t, alpha_t, s=80, color="#388E3C",
           label="Two-point{1,W}", zorder=5, marker="s")

# Fit joint log-σ → α
all_sigma = sigma_u + sigma_e + sigma_t
all_alpha = alpha_u + alpha_e + alpha_t
valid_pairs = [(s, a) for s, a in zip(all_sigma, all_alpha)
               if a is not None and s > 0]
if valid_pairs:
    ls = np.log([p[0] for p in valid_pairs])
    la = [p[1] for p in valid_pairs]
    sl, ic, rv, _, _ = stats.linregress(ls, la)
    sv_arr = np.array(sorted([p[0] for p in valid_pairs]))
    ax.plot(sv_arr, ic + sl * np.log(sv_arr), "k--", lw=1.5,
            label=f"joint log fit (R={rv:.3f})")

ax.set_xscale("log")
ax.set_xlabel("Standard deviation σ", fontsize=11)
ax.set_ylabel("α", fontsize=11)
ax.set_title("α vs σ  (모든 분포 통합)", fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ─ Panel (1,0): skip_ratio curves per W for U[1,W] ──────────────────
ax = fig.add_subplot(gs[1, 0])
cmap = plt.cm.viridis
W_show = [10, 100, 1000]
for i, W in enumerate(W_show):
    row = next((r for r in exp1_rows if r["W_max"] == W), None)
    if row:
        sr = row["sr_per_k"]
        col = cmap(i / max(len(W_show)-1, 1))
        ax.plot(K_VALS[:len(sr)], sr, "o-", color=col,
                lw=2, markersize=7, label=f"U[1,{W}]")
        # formula overlay
        if row["alpha"] and row["c"]:
            k_fine = np.linspace(K_VALS[0], K_VALS[-1], 100)
            ax.plot(k_fine, 1 - row["c"]/k_fine**row["alpha"],
                    "--", color=col, lw=1, alpha=0.6)

ax.set_xscale("log")
ax.set_xlabel("k (average degree)", fontsize=11)
ax.set_ylabel("skip_ratio", fontsize=11)
ax.set_title("skip_ratio vs k: U[1,W] 비교", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

# ─ Panel (1,1): same-variance comparison ────────────────────────────
ax = fig.add_subplot(gs[1, 1])
colors_sv = {"Uniform": "#1565C0", "Exp_matched": "#E53935"}
markers_sv = {"Uniform": "o", "Exp_matched": "^"}

for row in exp3_rows:
    dt = row["dist_type"]
    col = colors_sv[dt]
    mk  = markers_sv[dt]
    sr  = row["sr_per_k"]
    ax.plot(K_VALS[:len(sr)], sr, marker=mk, color=col,
            lw=2, markersize=7,
            label=row["label"] + f" α={row['alpha']:.3f}")

ax.set_xscale("log")
ax.set_xlabel("k (average degree)", fontsize=11)
ax.set_ylabel("skip_ratio", fontsize=11)
ax.set_title("동일 σ 비교: Uniform vs Exponential", fontsize=12)
ax.legend(fontsize=7, ncol=1); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

# ─ Panel (1,2): α summary table / heatmap (W_max × k) ───────────────
ax = fig.add_subplot(gs[1, 2])

W_list_heat = [r["W_max"] for r in exp1_rows if r["W_max"] > 1]
data_heat = np.array([r["sr_per_k"] for r in exp1_rows if r["W_max"] > 1])

im = ax.imshow(data_heat, aspect="auto", cmap="RdYlGn",
               vmin=0, vmax=0.85)
ax.set_xticks(range(len(K_VALS)))
ax.set_xticklabels([f"k={k}" for k in K_VALS], fontsize=8)
ax.set_yticks(range(len(W_list_heat)))
ax.set_yticklabels([f"W={w}" for w in W_list_heat], fontsize=8)
ax.set_title("skip_ratio 히트맵: U[1,W] × k", fontsize=12)
plt.colorbar(im, ax=ax, label="skip_ratio")

for i in range(len(W_list_heat)):
    for j in range(len(K_VALS)):
        ax.text(j, i, f"{data_heat[i,j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="black" if data_heat[i,j] < 0.6 else "white")

plt.suptitle("Theory Probe v2: α의 분포 의존성 체계적 분석\n"
             "skip_ratio(k) ≈ 1 − c/k^α", fontsize=14, y=1.01)

out = RESULTS_DIR / f"theory_v2_{ts}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ─────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("최종 요약: 모든 분포의 (c, α, R²)")
print("═"*65)
print(f"{'분포':<30s}  {'α':>7s}  {'c':>7s}  {'R²':>7s}")
print("-"*55)

all_rows = (
    [(r["label"], r["alpha"], r["c"], r["r2"]) for r in exp1_rows] +
    [(r["label"], r["alpha"], r["c"], r["r2"]) for r in exp2_rows] +
    [(r["label"], r["alpha"], r["c"], r["r2"]) for r in exp3_rows
     if r["dist_type"] == "Exp_matched"] +
    [(r["label"], r["alpha"], r["c"], r["r2"]) for r in exp4_rows]
)
for label, alpha, c, r2 in all_rows:
    if alpha is None:
        print(f"  {label:<28s}  {'N/A':>7s}  {'N/A':>7s}  {'N/A':>7s}")
    else:
        print(f"  {label:<28s}  {alpha:>7.4f}  {c:>7.4f}  {r2:>7.4f}")

print(f"\n저장 위치: {RESULTS_DIR.resolve()}/")
