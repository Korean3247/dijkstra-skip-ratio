"""
Theory Probe v3: α를 제어하는 단일 파라미터 식별
══════════════════════════════════════════════════════════════════
핵심 가설: α는 P(W = W_min) — 최솟값에서의 확률 질량 — 에 의해 제어된다.

실험 5: P(W=1) 고정 효과
  Two-point{1, 1000}에서 P(W=1) = p ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99}
  나머지 확률 1-p로 W=1000

실험 6: Minimum weight의 역할
  U[1, W] vs U[2, W] vs U[10, W]  (최솟값만 변화)
  → min weight가 바뀌면 α가 바뀌는가?

실험 7: α_∞ 극한 확인
  U[1, W_max] for W_max ∈ {1000, 5000, 10000, 50000}
  → α가 정말로 0.30에 수렴하는가?

실험 8: "경쟁 weight 레벨 수" 직접 조작
  U on {1, 2, ..., n_levels} (n_levels=W_max이 아닌 레벨 수)
  각 레벨에서 random integer 가중치
  → n_levels ∈ {2, 5, 10, 50, 100} 고정
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
    adj = [[] for _ in range(V)]
    n_edges = V * k_target // 2
    for _ in range(n_edges):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            w = weight_fn(rng)
            adj[u].append((v, w))
            adj[v].append((u, w))
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
    push_c = 1
    pop_c = skip_c = 0
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


def fit_power_law(k_vals, sr_vals):
    valid = [(k, s) for k, s in zip(k_vals, sr_vals) if 0 < s < 1]
    if len(valid) < 3:
        return None, None, None
    lk  = np.array([math.log(k) for k, _ in valid])
    ly  = np.array([math.log(1.0 - s) for _, s in valid])
    slope, intercept, r, p, se = stats.linregress(lk, ly)
    return math.exp(intercept), -slope, r**2


def run_distribution(label, weight_fn, k_vals=K_VALS, verbose=True):
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
            print(f"  {label:<40s}  α=N/A   c=N/A   R²=N/A")
        else:
            flag = "" if (r2 and r2 > 0.97) else "  ⚠"
            print(f"  {label:<40s}  α={alpha:.4f}  c={c:.4f}  R²={r2:.4f}{flag}")
    return {"label": label, "alpha": alpha, "c": c, "r2": r2, "sr_per_k": sr_per_k}


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 5: P(W=1) 변화  →  α(p)
# Two-point{1, 1000}: P(W=1) = p, P(W=1000) = 1-p
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("실험 5: Two-point{1,1000}에서 P(W=1) = p 변화  →  α vs p")
print("═"*70)

p_vals = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.99]
exp5_rows = []

for p in p_vals:
    fn = lambda rng, p=p: (1 if rng.random() < p else 1000)
    row = run_distribution(f"Two-point{{1,1000}} p(W=1)={p:.2f}", fn)
    row["p_min"] = p
    exp5_rows.append(row)

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 6: Minimum weight 역할
# U[W_min, W_max] — 같은 range (100), 다른 최솟값
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("실험 6: U[W_min, W_min+99]  — 최솟값 변화  (range 고정 =100)")
print("═"*70)

W_mins = [1, 2, 5, 10, 50, 100]
exp6_rows = []

for W_min in W_mins:
    W_max = W_min + 99
    fn = lambda rng, a=W_min, b=W_max: rng.randint(a, b)
    row = run_distribution(f"U[{W_min},{W_max}]  (range=100)", fn)
    row["W_min"] = W_min
    row["W_max"] = W_max
    exp6_rows.append(row)

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 7: α_∞ 극한 확인 — 대형 W
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("실험 7: U[1, W_max] 대형 W  →  α_∞ 수렴 확인")
print("═"*70)

# Only test k in {8,16,32,64} to save time (enough for α estimation)
K_LARGE = [8, 16, 32, 64]
W_large_vals = [1000, 5000, 10000, 50000]
exp7_rows = []

for W in W_large_vals:
    fn = lambda rng, W=W: rng.randint(1, W)
    row = run_distribution(f"U[1,{W}]", fn, k_vals=K_LARGE)
    row["W_max"] = W
    exp7_rows.append(row)

# Add W=100 as reference
fn_ref = lambda rng: rng.randint(1, 100)
row_ref = run_distribution("U[1,100] (reference)", fn_ref, k_vals=K_LARGE)
row_ref["W_max"] = 100
exp7_rows.insert(0, row_ref)

# ─────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("분석 결과")
print("═"*70)

# Exp 5: α vs p
print("\n[실험 5] Two-point P(W=1)=p vs α:")
for row in exp5_rows:
    if row["alpha"] is not None:
        print(f"  p={row['p_min']:.2f}  α={row['alpha']:.4f}")
    else:
        print(f"  p={row['p_min']:.2f}  α=N/A")

# Exp 6: α vs W_min (range fixed)
print("\n[실험 6] U[W_min, W_min+99] vs α  (range=100 고정):")
for row in exp6_rows:
    if row["alpha"] is not None:
        print(f"  W_min={row['W_min']:4d}  W_max={row['W_max']:4d}  α={row['alpha']:.4f}  c={row['c']:.4f}")

# Exp 7: α_∞ convergence
print("\n[실험 7] U[1,W] 대형 W — α_∞ 수렴:")
for row in exp7_rows:
    if row["alpha"] is not None:
        print(f"  W={row['W_max']:6d}  α={row['alpha']:.4f}  R²={row['r2']:.4f}")

# Key insight: does α depend on P(W=1)?
print("\n[핵심 분석] P(W=W_min) vs α:")
# Collect all cases where P(W=W_min) is known:
# Two-point with p = P(W=1):  p_min
# U[1,W]: P(W=1) = 1/W
# Exp(μ) truncated: P(W=1) ≈ 1-e^{-1/μ} ≈ 1/μ

from_exp5 = [(r["p_min"], r["alpha"]) for r in exp5_rows if r["alpha"] is not None]
# U[1,W] with P(W=1) = 1/W
u_p_alpha = [(1/r["W_max"], r["alpha"]) for r in exp7_rows if r["alpha"] is not None]

print("  Two-point P(W=1) vs α:")
for p, a in from_exp5:
    print(f"    P={p:.3f}  α={a:.4f}")
print("  U[1,W] P(W=1) = 1/W vs α:")
for p, a in u_p_alpha:
    print(f"    P={p:.5f}  α={a:.4f}")

# Does α_∞ relate to α when P(W=1)→0?
print(f"\n  → U[1,50000] (P(W=1)=2e-5): α={exp7_rows[-1]['alpha']:.4f}")
print(f"  → α_∞ 추정 (P→0 극한): ~0.300")

# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel (0,0): α vs p for Two-point
ax = fig.add_subplot(gs[0, 0])
p5 = [r["p_min"] for r in exp5_rows if r["alpha"] is not None]
a5 = [r["alpha"] for r in exp5_rows if r["alpha"] is not None]
ax.plot(p5, a5, "o-", color="#9C27B0", lw=2, markersize=9, zorder=5)
ax.axhline(0.30, color="k", ls="--", lw=1, alpha=0.5, label="alpha_inf=0.30")
ax.set_xlabel("P(W=1) = p", fontsize=11)
ax.set_ylabel("alpha", fontsize=11)
ax.set_title("Two-point{1,1000}: alpha vs P(W_min)", fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(-0.01, 0.35)

# Panel (0,1): α vs W_min (range=100 fixed)
ax = fig.add_subplot(gs[0, 1])
wm6 = [r["W_min"] for r in exp6_rows if r["alpha"] is not None]
a6  = [r["alpha"] for r in exp6_rows if r["alpha"] is not None]
ax.plot(wm6, a6, "s-", color="#E53935", lw=2, markersize=9, zorder=5)
ax.set_xscale("log")
ax.set_xlabel("W_min", fontsize=11)
ax.set_ylabel("alpha", fontsize=11)
ax.set_title("U[W_min, W_min+99]: alpha vs W_min\n(range=100 fixed)", fontsize=12)
ax.grid(True, alpha=0.3)

# Panel (0,2): α_inf convergence
ax = fig.add_subplot(gs[0, 2])
W7 = [r["W_max"] for r in exp7_rows if r["alpha"] is not None]
a7 = [r["alpha"] for r in exp7_rows if r["alpha"] is not None]
ax.plot(W7, a7, "o-", color="#1565C0", lw=2, markersize=9, label="U[1,W]")
ax.axhline(0.30, color="k", ls="--", lw=1.5, alpha=0.7, label="alpha_inf=0.30")
ax.set_xscale("log")
ax.set_xlabel("W_max", fontsize=11)
ax.set_ylabel("alpha", fontsize=11)
ax.set_title("U[1,W]: alpha convergence to alpha_inf", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.35)

# Panel (1,0): skip_ratio curves for Exp 5 (p variations)
ax = fig.add_subplot(gs[1, 0])
cmap = plt.cm.RdYlGn
p_show = [0.01, 0.10, 0.50, 0.99]
for i, row in enumerate([r for r in exp5_rows if r["p_min"] in p_show]):
    col = cmap(i / max(len(p_show)-1, 1))
    ax.plot(K_VALS[:len(row["sr_per_k"])], row["sr_per_k"],
            "o-", color=col, lw=2, markersize=7,
            label=f"p={row['p_min']:.2f} a={row['alpha']:.3f}")
ax.set_xscale("log")
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("skip_ratio", fontsize=11)
ax.set_title("Two-point: skip_ratio curves", fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

# Panel (1,1): skip_ratio curves for Exp 6 (W_min variations)
ax = fig.add_subplot(gs[1, 1])
cmap2 = plt.cm.viridis
for i, row in enumerate(exp6_rows):
    col = cmap2(i / max(len(exp6_rows)-1, 1))
    ax.plot(K_VALS[:len(row["sr_per_k"])], row["sr_per_k"],
            "o-", color=col, lw=2, markersize=7,
            label=f"W_min={row['W_min']} a={row['alpha']:.3f}")
ax.set_xscale("log")
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("skip_ratio", fontsize=11)
ax.set_title("U[W_min, W_min+99]: skip_ratio curves", fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

# Panel (1,2): unified view — all alpha vs log(1/P_min)
ax = fig.add_subplot(gs[1, 2])
# Two-point: P_min = p
for p, a in from_exp5:
    if a is not None:
        ax.scatter(-math.log(p), a, s=80, color="#9C27B0", marker="^", zorder=5)
# U[1,W]: P_min = 1/W
for p, a in u_p_alpha:
    if a is not None:
        ax.scatter(-math.log(p), a, s=80, color="#1565C0", marker="o", zorder=5)
# Dummy points for legend
ax.scatter([], [], color="#9C27B0", marker="^", label="Two-point (P_min=p)")
ax.scatter([], [], color="#1565C0", marker="o", label="U[1,W] (P_min=1/W)")
ax.axhline(0.30, color="k", ls="--", lw=1.5, alpha=0.7, label="alpha_inf=0.30")
ax.set_xlabel("-log P(W = W_min)  =  log(1/P_min)", fontsize=11)
ax.set_ylabel("alpha", fontsize=11)
ax.set_title("alpha vs -log P_min: 통합 뷰", fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle("Theory Probe v3: alpha를 제어하는 단일 파라미터 식별\n"
             "Hypothesis: alpha is controlled by P(W = W_min)", fontsize=14, y=1.01)

out = RESULTS_DIR / f"theory_v3_{ts}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# Final summary
print("\n" + "═"*70)
print("[실험 6 핵심] W_min 효과 (range=100 고정):")
print("  W_min이 바뀌어도 P(W=W_min) = 1/range = 1/100은 고정")
print("  만약 α가 P(W_min)에만 의존하면 → α가 W_min에 무관해야 함")
print("  만약 α가 바뀐다 → W_min 자체(또는 ratio E[W]/W_min)도 영향")
print()
for row in exp6_rows:
    if row["alpha"] is not None:
        ratio = (row["W_min"] + 99.5) / row["W_min"]  # approx E[W]/W_min
        print(f"  W_min={row['W_min']:4d}  E[W]/W_min={ratio:.2f}  α={row['alpha']:.4f}")

print(f"\n완료: {RESULTS_DIR.resolve()}/")
