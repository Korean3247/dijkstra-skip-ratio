"""
Theory Probe v4: α_∞ 정밀 측정 및 고-k 거동 분석
══════════════════════════════════════════════════════════════════
핵심 질문:
  1. k=256, 512까지 확장 시 α_∞ 추정값이 바뀌는가?
  2. 멱법칙은 k≈350에서 H_k 상한을 위반한다 (이론) — 데이터로 확인
  3. 연속 float 가중치 U[0,1]에서의 α 측정
  4. 올바른 점근 형태: 1-c/k^α vs 1-1/H_k — 어느 쪽이 더 정확한가?

이론적 예측:
  push_count/V = k^α/c vs H_k = ln(k) + 0.577
  → k^0.309/0.945 > H_k 는 k≈350부터 발생
  → 실제 데이터에서 k≥256에서 α 감소 또는 H_k 굴절이 관측될 것
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
import time

RESULTS_DIR = Path("resultsTheory")
RESULTS_DIR.mkdir(exist_ok=True)

V      = 10_000   # 고-k에서 포화 방지
N_REPS = 20
SEED   = 42
K_VALS = [4, 8, 16, 32, 64, 128, 256, 512]


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


def dijkstra_full(adj, source, V):
    """Return (skip_ratio, push_count/V, pop_count/V)."""
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
    skip_ratio = skip_c / pop_c if pop_c else 0.0
    return skip_ratio, push_c / V, pop_c / V


def harmonic(k):
    return sum(1.0/i for i in range(1, k+1))


def fit_power_law(k_vals, sr_vals):
    valid = [(k, s) for k, s in zip(k_vals, sr_vals) if 0 < s < 1]
    if len(valid) < 3:
        return None, None, None
    lk = np.array([math.log(k) for k, _ in valid])
    ly = np.array([math.log(1.0 - s) for _, s in valid])
    slope, intercept, r, p, se = stats.linregress(lk, ly)
    return math.exp(intercept), -slope, r**2


def run_dist(label, weight_fn, k_vals=K_VALS, verbose=True):
    """Run one weight distribution, collecting skip_ratio AND push_count/V."""
    sr_per_k   = []
    push_per_k = []

    for k in k_vals:
        srs, pushes = [], []
        t0 = time.perf_counter()
        for rep in range(N_REPS):
            rng = random.Random(SEED + rep * 1009 + k * 7)
            adj = make_graph(V, k, weight_fn, rng)
            src = rng.randint(0, V - 1)
            sr, push_v, pop_v = dijkstra_full(adj, src, V)
            srs.append(sr)
            pushes.append(push_v)
        elapsed = time.perf_counter() - t0
        sr_per_k.append(float(np.mean(srs)))
        push_per_k.append(float(np.mean(pushes)))
        if verbose:
            print(f"  {label}  k={k:4d}  sr={sr_per_k[-1]:.4f}"
                  f"  push/V={push_per_k[-1]:.3f}"
                  f"  H_k={harmonic(k):.3f}"
                  f"  t={elapsed:.1f}s")

    return {"label": label, "sr": sr_per_k, "push_v": push_per_k}


# ─────────────────────────────────────────────────────────────────────
# DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────

dists = {
    "U[1,100]    (original)": lambda rng: rng.randint(1, 100),
    "U[1,100000] (large W)":  lambda rng: rng.randint(1, 100_000),
    "U[0,1] float (cont.)":   lambda rng: rng.random(),
}

results = {}
for label, fn in dists.items():
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    results[label] = run_dist(label, fn)

# ─────────────────────────────────────────────────────────────────────
# α FITS OVER DIFFERENT k-RANGES
# ─────────────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print("  α 추정: k 범위별 비교")
print(f"{'═'*60}")

K_RANGES = {
    "k ∈ [4, 64]":   K_VALS[:5],   # indices 0..4
    "k ∈ [4, 128]":  K_VALS[:6],
    "k ∈ [4, 256]":  K_VALS[:7],
    "k ∈ [4, 512]":  K_VALS[:8],
}

alpha_table = {}   # label → {range_name → alpha}

for label, res in results.items():
    alpha_table[label] = {}
    print(f"\n  {label}:")
    for range_name, k_sub in K_RANGES.items():
        k_idx = [K_VALS.index(k) for k in k_sub]
        sr_sub = [res["sr"][i] for i in k_idx]
        c, alpha, r2 = fit_power_law(k_sub, sr_sub)
        alpha_table[label][range_name] = alpha
        flag = "  ⚠ low R²" if (r2 is None or r2 < 0.99) else ""
        if alpha is not None:
            print(f"    {range_name}: α={alpha:.4f}  c={c:.4f}  R²={r2:.4f}{flag}")
        else:
            print(f"    {range_name}: α=N/A")

# ─────────────────────────────────────────────────────────────────────
# HARMONIC BOUND COMPARISON: push_count/V vs H_k vs k^α/c
# ─────────────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print("  조화 상한 비교: push_count/V vs H_k vs 멱법칙 예측")
print(f"{'═'*60}")

for label, res in results.items():
    # fit α over full range
    c, alpha, r2 = fit_power_law(K_VALS, res["sr"])
    print(f"\n  {label}  (α={alpha:.4f}, c={c:.4f}):")
    print(f"  {'k':>5s}  {'push/V':>7s}  {'H_k':>7s}  {'k^α/c':>7s}  {'ratio':>7s}")
    for i, k in enumerate(K_VALS):
        pv   = res["push_v"][i]
        hk   = harmonic(k)
        pred = (k**alpha / c) if (alpha and c) else float("nan")
        ratio = pv / hk
        flag = "  ⚠ >H_k" if pv > hk else ""
        print(f"  {k:5d}  {pv:7.3f}  {hk:7.3f}  {pred:7.3f}  {ratio:7.3f}{flag}")

# ─────────────────────────────────────────────────────────────────────
# CSV OUTPUT  (reproducibility)
# ─────────────────────────────────────────────────────────────────────

import csv, math

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. skip_ratio and push_count/V by distribution × k
csv1 = RESULTS_DIR / f"skip_ratio_by_dist_{ts}.csv"
with open(csv1, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["distribution", "k", "skip_ratio_mean", "push_per_V", "H_k",
                "push_per_V_over_Hk"])
    for label, res in results.items():
        for i, k in enumerate(K_VALS):
            hk = harmonic(k)
            w.writerow([label, k,
                        round(res["sr"][i], 6),
                        round(res["push_v"][i], 6),
                        round(hk, 6),
                        round(res["push_v"][i] / hk, 6)])
print(f"Saved: {csv1}")

# 2. Alpha fits by distribution × k-range
csv2 = RESULTS_DIR / f"alpha_fits_{ts}.csv"
with open(csv2, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["distribution", "k_range", "alpha", "c", "R2"])
    for label, res in results.items():
        for range_name, k_sub in K_RANGES.items():
            k_idx = [K_VALS.index(k) for k in k_sub]
            sr_sub = [res["sr"][i] for i in k_idx]
            c_fit, alpha_fit, r2_fit = fit_power_law(k_sub, sr_sub)
            w.writerow([label, range_name,
                        round(alpha_fit, 6) if alpha_fit else "NA",
                        round(c_fit, 6)     if c_fit     else "NA",
                        round(r2_fit, 6)    if r2_fit    else "NA"])
print(f"Saved: {csv2}")

# 3. Local slope α(k) — numerical derivative
csv3 = RESULTS_DIR / f"local_slope_{ts}.csv"
with open(csv3, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["distribution", "k_lo", "k_hi", "k_geomean",
                "local_alpha", "one_over_Hk_geomean", "ratio"])
    for label, res in results.items():
        sr = res["sr"]
        for i in range(len(K_VALS) - 1):
            k1, k2 = K_VALS[i], K_VALS[i + 1]
            s1, s2 = sr[i], sr[i + 1]
            if 0 < s1 < 1 and 0 < s2 < 1:
                la = -(math.log(1-s2) - math.log(1-s1)) / \
                     (math.log(k2)    - math.log(k1))
                kg = math.sqrt(k1 * k2)
                inv_hk = 1.0 / harmonic(int(kg))
                w.writerow([label, k1, k2, round(kg, 3),
                            round(la, 6), round(inv_hk, 6),
                            round(la / inv_hk, 6)])
print(f"Saved: {csv3}")

# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

colors_dist = {
    "U[1,100]    (original)": "#1565C0",
    "U[1,100000] (large W)":  "#E53935",
    "U[0,1] float (cont.)":   "#388E3C",
}

# ─ Panel (0,0): skip_ratio vs k — all distributions ─────────────────
ax = fig.add_subplot(gs[0, 0])
k_fine = np.logspace(np.log10(4), np.log10(512), 200)

for label, res in results.items():
    col = colors_dist[label]
    ax.plot(K_VALS[:len(res["sr"])], res["sr"],
            "o-", color=col, lw=2, markersize=7, label=label)

# Harmonic bound 1 - 1/H_k
ax.plot(k_fine, [1 - 1/harmonic(int(k)) for k in k_fine],
        "k--", lw=1.5, label="Harmonic: $1-1/H_k$")

# Power law with original α
ax.plot(k_fine, 1 - 0.884 * k_fine**(-0.262),
        "gray", lw=1, ls=":", label="Original: $1-0.884/k^{0.262}$")

ax.set_xscale("log")
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("skip_ratio", fontsize=11)
ax.set_title("skip_ratio vs k: distributions + bounds", fontsize=12)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0.2, 1)

# ─ Panel (0,1): push_count/V vs H_k ─────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
H_vals = [harmonic(k) for k in K_VALS]
ax.plot(K_VALS, H_vals, "k--", lw=2, label="$H_k$ (upper bound)")

for label, res in results.items():
    col = colors_dist[label]
    ax.plot(K_VALS[:len(res["push_v"])], res["push_v"],
            "o-", color=col, lw=2, markersize=7, label=label)

ax.set_xscale("log")
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("push_count / V", fontsize=11)
ax.set_title("push_count/V vs Harmonic bound $H_k$", fontsize=12)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ─ Panel (0,2): α vs k-range — how α changes with range ─────────────
ax = fig.add_subplot(gs[0, 2])
range_labels = list(K_RANGES.keys())
x = np.arange(len(range_labels))
width = 0.25

for j, (label, col) in enumerate(colors_dist.items()):
    alphas_per_range = [alpha_table[label].get(rn) for rn in range_labels]
    valid_a = [a if a is not None else 0 for a in alphas_per_range]
    ax.bar(x + j*width, valid_a, width, label=label[:15], color=col, alpha=0.8)

ax.set_xticks(x + width)
ax.set_xticklabels([rn.replace("k ∈ ", "").replace(",", "\n", 1)
                    for rn in range_labels], fontsize=8)
ax.set_ylabel("alpha (fitted)", fontsize=11)
ax.set_title("alpha fitted over different k-ranges", fontsize=12)
ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)
ax.axhline(0.309, color="k", ls="--", lw=1, alpha=0.5)
ax.axhline(0.262, color="gray", ls=":", lw=1, alpha=0.5)

# ─ Panel (1,0): residuals — actual vs power law, large W ─────────────
ax = fig.add_subplot(gs[1, 0])
for label, res in results.items():
    col = colors_dist[label]
    c, alpha, r2 = fit_power_law(K_VALS, res["sr"])
    if alpha is None:
        continue
    residuals = [actual - (1 - c/k**alpha)
                 for k, actual in zip(K_VALS, res["sr"])]
    ax.plot(K_VALS, residuals, "o-", color=col, lw=2, markersize=7,
            label=f"{label[:15]} (alpha={alpha:.3f})")

ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xscale("log")
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("actual - power_law", fontsize=11)
ax.set_title("Power-law residuals: positive=underprediction", fontsize=12)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ─ Panel (1,1): local slope α(k) — numerical derivative ─────────────
ax = fig.add_subplot(gs[1, 1])
for label, res in results.items():
    col = colors_dist[label]
    sr = res["sr"]
    local_alphas = []
    k_mids = []
    for i in range(len(K_VALS)-1):
        k1, k2 = K_VALS[i], K_VALS[i+1]
        s1, s2 = sr[i], sr[i+1]
        if 0 < s1 < 1 and 0 < s2 < 1:
            ls1, ls2 = math.log(1-s1), math.log(1-s2)
            lk1, lk2 = math.log(k1), math.log(k2)
            local_alpha = -(ls2-ls1)/(lk2-lk1)
            local_alphas.append(local_alpha)
            k_mids.append(math.sqrt(k1*k2))  # geometric mean

    ax.plot(k_mids, local_alphas, "o-", color=col, lw=2, markersize=7,
            label=label[:20])

ax.axhline(0.309, color="k", ls="--", lw=1, label="alpha_inf=0.309")
ax.axhline(0.262, color="gray", ls=":", lw=1, label="original 0.262")
ax.set_xscale("log")
ax.set_xlabel("k (geometric mean of segment)", fontsize=11)
ax.set_ylabel("local slope alpha(k)", fontsize=11)
ax.set_title("Local slope: is alpha decreasing at high k?", fontsize=12)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ─ Panel (1,2): power law prediction vs harmonic at k=4..1024 ────────
ax = fig.add_subplot(gs[1, 2])
k_ext = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
H_ext = np.array([harmonic(k) for k in k_ext])
sr_H  = 1 - 1.0/H_ext

ax.plot(k_ext, 1 - 0.884/k_ext**0.262, "gray", ls=":", lw=2,
        label="Power law $\\alpha=0.262$ (original)")
ax.plot(k_ext, 1 - 0.945/k_ext**0.309, "r--", lw=2,
        label="Power law $\\alpha=0.309$ (large-W)")
ax.plot(k_ext, sr_H, "k-", lw=2, label="Harmonic approx $1-1/H_k$")

# shade where power law > harmonic bound (violation zone)
vio = k_ext[0.945/k_ext**0.309 < 1.0/H_ext]
if len(vio):
    ax.axvline(vio[0], color="orange", lw=2, ls="--",
               label=f"Violation start k≈{vio[0]}")
ax.set_xscale("log")
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("skip_ratio", fontsize=11)
ax.set_title("Power law vs Harmonic: extrapolation to k=2048", fontsize=12)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0.2, 1)

plt.suptitle("Theory Probe v4: alpha_inf precision & high-k behavior\n"
             "Does power law violate H_k at k>=256?", fontsize=14, y=1.01)

out = RESULTS_DIR / f"theory_v4_{ts}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ─────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print("최종 결론")
print(f"{'═'*60}")

for label, res in results.items():
    c_full, alpha_full, r2_full = fit_power_law(K_VALS, res["sr"])
    c_lo, alpha_lo, r2_lo = fit_power_law(K_VALS[:5], res["sr"][:5])  # k<=64
    print(f"\n  {label}")
    if alpha_lo and alpha_full:
        print(f"    k∈[4,64]:  α={alpha_lo:.4f}  R²={r2_lo:.4f}")
        print(f"    k∈[4,512]: α={alpha_full:.4f}  R²={r2_full:.4f}  "
              f"Δα={alpha_lo-alpha_full:+.4f}")
    # check harmonic violation
    max_pv = max(res["push_v"])
    max_k  = K_VALS[res["push_v"].index(max_pv)]
    hk_max = harmonic(max_k)
    print(f"    Max push/V={max_pv:.3f} at k={max_k}, H_k={hk_max:.3f}  "
          f"{'VIOLATION' if max_pv > hk_max else 'within bound'}")

print(f"\n  완료: {RESULTS_DIR.resolve()}/")
