"""
Dijkstra 3차 실험: 포화 곡선 확인 및 수식 파라미터 피팅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
후보 수식:
  (A) skip_ratio(k) = 1 - c / k^α          (power law)
  (B) skip_ratio(k) = a * log(k) + b        (log)
  (C) skip_ratio(k) = L / (1 + exp(-r*(k - k0)))  (sigmoid)

k 범위: 4, 8, 16, 32, 64 (2차 데이터 재활용) + 128, 256, 512 (신규)
V      : 1,000 / 10,000 (V 무관성 이미 확인됨)
반복   : 10회
"""

import heapq
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS  = 10
SEED     = 42

V_LIST   = [1_000, 10_000]
K_NEW    = [128, 256, 512]          # 3차 신규 측정
K_PRIOR  = [4, 8, 16, 32, 64]      # 2차에서 이미 측정
K_ALL    = K_PRIOR + K_NEW         # 전체 범위 (시각화용)

# 2차 실험 CSV 경로 (있으면 합쳐서 전체 범위 시각화)
PRIOR_CSV = Path("results/experiment2_20260506_074451.csv")


# ─────────────────────────────────────────────────────────────────────
# Graph generator
# ─────────────────────────────────────────────────────────────────────

def make_graph(V: int, k: int, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    adj = {u: [] for u in range(V)}
    for _ in range(V * k // 2):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            w = round(rng.uniform(0.1, 10.0), 4)
            adj[u].append((v, w))
            adj[v].append((u, w))
    # 연결성 보장
    nodes = list(range(V))
    rng.shuffle(nodes)
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        w = round(rng.uniform(0.1, 10.0), 4)
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# ─────────────────────────────────────────────────────────────────────
# Dijkstra (계측)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_counted(adj: dict, source: int, V: int) -> dict:
    dist = [float("inf")] * V
    dist[source] = 0.0
    heap = [(0.0, source)]
    push_count = 1
    pop_count  = 0
    skip_count = 0
    while heap:
        d, u = heapq.heappop(heap)
        pop_count += 1
        if d > dist[u]:
            skip_count += 1
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                push_count += 1
    return {
        "push_count":    push_count,
        "pop_count":     pop_count,
        "skip_count":    skip_count,
        "skip_ratio":    skip_count / pop_count if pop_count else 0.0,
        "push_per_node": push_count / V,
    }


def dijkstra_plain(adj: dict, source: int, V: int) -> None:
    dist = [float("inf")] * V
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))


# ─────────────────────────────────────────────────────────────────────
# Curve fitting models
# ─────────────────────────────────────────────────────────────────────

def model_power(k, c, alpha):
    """skip_ratio = 1 - c / k^alpha"""
    return 1.0 - c / (np.asarray(k, dtype=float) ** alpha)

def model_log(k, a, b):
    """skip_ratio = a * log(k) + b"""
    return a * np.log(np.asarray(k, dtype=float)) + b

def model_sigmoid(k, L, r, k0):
    """skip_ratio = L / (1 + exp(-r*(k - k0)))"""
    return L / (1.0 + np.exp(-r * (np.asarray(k, dtype=float) - k0)))


def fit_and_score(model_fn, k_arr, y_arr, p0, bounds=(-np.inf, np.inf)):
    try:
        popt, _ = curve_fit(model_fn, k_arr, y_arr, p0=p0, bounds=bounds,
                            maxfev=10000)
        y_pred = model_fn(k_arr, *popt)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return popt, r2
    except Exception:
        return None, -1.0


# ─────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────

def run_new_measurements() -> pd.DataFrame:
    """K_NEW × V_LIST 조합만 새로 측정 (K_PRIOR는 2차 CSV 재활용)."""
    rows = []
    total = len(V_LIST) * len(K_NEW)
    print(f"신규 측정: V {V_LIST} × k {K_NEW} = {total}가지\n")

    with tqdm(total=total, desc="3차 실험") as pbar:
        for V in V_LIST:
            for k in K_NEW:
                adj = make_graph(V, k)
                E   = sum(len(nb) for nb in adj.values()) // 2
                ops = dijkstra_counted(adj, 0, V)

                times = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    dijkstra_plain(adj, 0, V)
                    times.append(time.perf_counter() - t0)

                row = {
                    "V":             V,
                    "k_target":      k,
                    "k_actual":      round(sum(len(nb) for nb in adj.values()) / V, 2),
                    "edge_count":    E,
                    "skip_ratio":    round(ops["skip_ratio"], 6),
                    "push_per_node": round(ops["push_per_node"], 4),
                    "push_count":    ops["push_count"],
                    "pop_count":     ops["pop_count"],
                    "skip_count":    ops["skip_count"],
                    "time_mean":     round(float(np.mean(times)), 6),
                    "time_std":      round(float(np.std(times)),  6),
                }
                rows.append(row)
                pbar.update(1)
                pbar.write(
                    f"  V={V:>6,}  k={k:>3}  "
                    f"skip={ops['skip_ratio']:.4f}  "
                    f"push/V={ops['push_per_node']:.3f}  "
                    f"time={np.mean(times)*1000:.1f}ms"
                )

    return pd.DataFrame(rows)


def load_prior(V_list: list[int]) -> pd.DataFrame:
    if not PRIOR_CSV.exists():
        print(f"[경고] 2차 데이터 없음: {PRIOR_CSV}  (신규 데이터만 사용)")
        return pd.DataFrame()
    df = pd.read_csv(PRIOR_CSV)
    return df[df["V"].isin(V_list)][
        ["V","k_target","k_actual","edge_count",
         "skip_ratio","push_per_node","push_count","pop_count","skip_count",
         "time_mean","time_std"]
    ].copy()


# ─────────────────────────────────────────────────────────────────────
# Visualization + curve fitting
# ─────────────────────────────────────────────────────────────────────

V_COLORS = {1_000: "#2196F3", 10_000: "#F44336"}
FIT_STYLE = {
    "power":   {"ls": "--", "lw": 1.5},
    "log":     {"ls": ":",  "lw": 1.5},
    "sigmoid": {"ls": "-.", "lw": 1.5},
}


def plot_and_fit(df_full: pd.DataFrame, timestamp: str) -> dict:
    fit_results = {}

    # ── Fig 1: skip_ratio vs k 전체 범위 + curve fits ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("skip_ratio vs Average Degree k  (k=4–512)\nCurve Fitting: Power / Log / Sigmoid",
                 fontsize=12, fontweight="bold")

    for ax_idx, V in enumerate(V_LIST):
        ax  = axes[ax_idx]
        sub = df_full[df_full["V"] == V].sort_values("k_target")
        k_arr = sub["k_target"].values.astype(float)
        y_arr = sub["skip_ratio"].values

        color = V_COLORS[V]
        # 데이터 포인트: 2차(기존) vs 3차(신규) 구분
        mask_new = sub["k_target"].isin(K_NEW)
        ax.scatter(k_arr[~mask_new.values], y_arr[~mask_new.values],
                   color=color, s=60, zorder=5, label="2차 데이터 (k≤64)")
        ax.scatter(k_arr[mask_new.values],  y_arr[mask_new.values],
                   color=color, s=90, marker="*", zorder=5, label="3차 데이터 (k≥128)")

        k_smooth = np.linspace(k_arr.min(), k_arr.max(), 300)
        fit_results[V] = {}

        # Power fit
        popt, r2 = fit_and_score(model_power, k_arr, y_arr,
                                  p0=[1.0, 0.5], bounds=([0, 0], [10, 5]))
        if popt is not None:
            fit_results[V]["power"] = {"c": popt[0], "alpha": popt[1], "r2": r2}
            ax.plot(k_smooth, model_power(k_smooth, *popt), color="gray",
                    label=f"Power: 1−{popt[0]:.3f}/k^{popt[1]:.3f}  R²={r2:.4f}",
                    **FIT_STYLE["power"])

        # Log fit
        popt_l, r2_l = fit_and_score(model_log, k_arr, y_arr, p0=[0.1, 0.2])
        if popt_l is not None:
            fit_results[V]["log"] = {"a": popt_l[0], "b": popt_l[1], "r2": r2_l}
            ax.plot(k_smooth, model_log(k_smooth, *popt_l), color="purple",
                    label=f"Log: {popt_l[0]:.3f}·ln(k)+{popt_l[1]:.3f}  R²={r2_l:.4f}",
                    **FIT_STYLE["log"])

        # Sigmoid fit
        popt_s, r2_s = fit_and_score(model_sigmoid, k_arr, y_arr,
                                      p0=[0.9, 0.02, 50],
                                      bounds=([0.5, 0, 0], [1.0, 1, 600]))
        if popt_s is not None:
            fit_results[V]["sigmoid"] = {"L": popt_s[0], "r": popt_s[1],
                                          "k0": popt_s[2], "r2": r2_s}
            ax.plot(k_smooth, model_sigmoid(k_smooth, *popt_s), color="green",
                    label=f"Sigmoid  R²={r2_s:.4f}",
                    **FIT_STYLE["sigmoid"])

        ax.set_title(f"V = {V:,}")
        ax.set_xlabel("Average Degree k")
        ax.set_ylabel("skip_ratio")
        ax.set_xscale("log")
        ax.set_xticks(K_ALL)
        ax.set_xticklabels(K_ALL)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"curve_fit_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: push_per_node vs k ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="gray", lw=1, ls=":", label="ideal (1×)")
    for V in V_LIST:
        sub = df_full[df_full["V"] == V].sort_values("k_target")
        ax.plot(sub["k_target"], sub["push_per_node"],
                color=V_COLORS[V], marker="o", lw=2, label=f"V={V:,}")
    ax.set_title("push_count / V  vs  k  (k=4–512)")
    ax.set_xlabel("Average Degree k")
    ax.set_ylabel("push_count / V")
    ax.set_xscale("log")
    ax.set_xticks(K_ALL); ax.set_xticklabels(K_ALL)
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"push_per_node_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: 잔차 플롯 (best fit 모델 기준) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Residuals — Power Law Fit", fontsize=11)
    for ax_idx, V in enumerate(V_LIST):
        ax  = axes[ax_idx]
        sub = df_full[df_full["V"] == V].sort_values("k_target")
        k_arr = sub["k_target"].values.astype(float)
        y_arr = sub["skip_ratio"].values
        if "power" in fit_results.get(V, {}):
            p = fit_results[V]["power"]
            y_pred = model_power(k_arr, p["c"], p["alpha"])
            residuals = y_arr - y_pred
            ax.bar(range(len(k_arr)), residuals, color=V_COLORS[V], alpha=0.7)
            ax.axhline(0, color="gray", lw=1)
            ax.set_xticks(range(len(k_arr)))
            ax.set_xticklabels([int(k) for k in k_arr])
            ax.set_title(f"V={V:,}  (R²={p['r2']:.4f})")
            ax.set_xlabel("k"); ax.set_ylabel("Residual")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"residuals_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved → {RESULTS_DIR}/")
    return fit_results


def print_fit_summary(fit_results: dict) -> None:
    print("\n" + "="*60)
    print("Curve Fitting Summary")
    print("="*60)
    for V, models in fit_results.items():
        print(f"\nV = {V:,}")
        print(f"  {'Model':<10}  {'Parameters':<40}  R²")
        print(f"  {'-'*8}  {'-'*40}  {'-'*6}")
        if "power" in models:
            p = models["power"]
            print(f"  {'Power':<10}  c={p['c']:.5f}  alpha={p['alpha']:.5f}       {p['r2']:.6f}")
        if "log" in models:
            p = models["log"]
            print(f"  {'Log':<10}  a={p['a']:.5f}  b={p['b']:.5f}              {p['r2']:.6f}")
        if "sigmoid" in models:
            p = models["sigmoid"]
            print(f"  {'Sigmoid':<10}  L={p['L']:.5f}  r={p['r']:.5f}  k0={p['k0']:.2f}  {p['r2']:.6f}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 3차 실험 — 포화 곡선 + 수식 피팅")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # 신규 측정 (k=128, 256, 512)
    df_new = run_new_measurements()
    df_new.to_csv(RESULTS_DIR / f"experiment3_new_{ts}.csv", index=False)
    print(f"\n신규 데이터 저장 → results/experiment3_new_{ts}.csv")

    # 2차 데이터 합치기
    df_prior = load_prior(V_LIST)
    df_full  = pd.concat([df_prior, df_new], ignore_index=True).sort_values(["V","k_target"])
    df_full.to_csv(RESULTS_DIR / f"experiment3_full_{ts}.csv", index=False)
    print(f"전체 데이터 저장 → results/experiment3_full_{ts}.csv")

    # 시각화 + 피팅
    fit_results = plot_and_fit(df_full, ts)
    print_fit_summary(fit_results)

    print(f"\nDone. All outputs in {RESULTS_DIR.resolve()}/")
