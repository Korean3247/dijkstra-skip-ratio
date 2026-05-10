"""
Dijkstra 5차 실험: Threshold 변형 — 속도 vs 정확성 트레이드오프
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[핵심 아이디어]
  기존 lazy deletion: nd < dist[v] 이면 무조건 push
  Threshold 변형:     nd < dist[v] 이고 in_heap[v] < θ 일 때만 push
                      → 동일 노드의 heap 중복을 θ개로 제한

[정확성 손실 메커니즘]
  dist[v]는 항상 업데이트되지만, θ 초과 시 heap에 추가 안 됨
  → 이전 stale entry(더 긴 거리)만 heap에 남음
  → 그 entry가 pop될 때 d > dist[v]로 skip
  → v가 올바른 거리로 처리될 기회를 영구히 잃음
  → v를 경유하는 경로들이 전파되지 않아 오답 발생

[실험 설계]
  그래프: sparse(k=10) / dense-mid(k=32) / dense-high(k=64)
  V     : 1,000 / 10,000
  θ     : inf (기존) / 1 / 2 / 3
  seed  : 5가지 (정확성 분포 측정)
  반복  : 10회 (타이밍)
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
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS      = 10
SEEDS        = [42, 123, 777, 2024, 9999]   # 정확성 분포용 다중 seed

V_LIST       = [1_000, 10_000]
GRAPH_TYPES  = {"sparse": 10, "dense-mid": 32, "dense-high": 64}
THETAS       = [float("inf"), 1, 2, 3]       # inf = 기존 lazy deletion

COLORS_THETA = {
    float("inf"): "#2196F3",
    1:            "#F44336",
    2:            "#FF9800",
    3:            "#4CAF50",
}
LABELS_THETA = {
    float("inf"): "Standard (θ=∞)",
    1:            "θ=1",
    2:            "θ=2",
    3:            "θ=3",
}


# ─────────────────────────────────────────────────────────────────────
# Graph generator
# ─────────────────────────────────────────────────────────────────────

def make_graph(V: int, k: int, seed: int) -> dict:
    rng = random.Random(seed)
    adj = {u: [] for u in range(V)}
    for _ in range(V * k // 2):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            w = round(rng.uniform(0.1, 10.0), 4)
            adj[u].append((v, w))
            adj[v].append((u, w))
    nodes = list(range(V))
    rng.shuffle(nodes)
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        w = round(rng.uniform(0.1, 10.0), 4)
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# ─────────────────────────────────────────────────────────────────────
# Dijkstra 구현
# ─────────────────────────────────────────────────────────────────────

def dijkstra_standard(adj: dict, source: int, V: int):
    """기존 lazy deletion — 항상 정확."""
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
    return dist, push_count, pop_count, skip_count, 0


def dijkstra_threshold(adj: dict, source: int, V: int, theta: int):
    """
    Threshold 변형.
    in_heap[v] >= theta 이면 nd < dist[v] 여도 push 차단.
    dist[v]는 업데이트하지만 heap 전파가 끊김 → 오답 가능.
    """
    dist      = [float("inf")] * V
    dist[source] = 0.0
    heap      = [(0.0, source)]
    in_heap   = [0] * V
    in_heap[source] = 1

    push_count    = 1
    pop_count     = 0
    skip_count    = 0
    blocked_push  = 0      # θ 초과로 차단된 push 횟수

    while heap:
        d, u = heapq.heappop(heap)
        pop_count += 1
        if in_heap[u] > 0:
            in_heap[u] -= 1

        if d > dist[u]:
            skip_count += 1
            continue

        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                if in_heap[v] < theta:
                    heapq.heappush(heap, (nd, v))
                    in_heap[v] += 1
                    push_count += 1
                else:
                    blocked_push += 1    # ← 이 순간 v의 올바른 거리 전파가 끊김

    return dist, push_count, pop_count, skip_count, blocked_push


def run_dijkstra(adj, source, V, theta):
    if theta == float("inf"):
        return dijkstra_standard(adj, source, V)
    return dijkstra_threshold(adj, source, V, int(theta))


# ─────────────────────────────────────────────────────────────────────
# 정확성 측정
# ─────────────────────────────────────────────────────────────────────

def measure_accuracy(dist_ref: list, dist_approx: list, V: int) -> dict:
    """
    dist_ref    : 기존 standard Dijkstra 결과 (정답)
    dist_approx : threshold 변형 결과
    """
    reachable = [i for i in range(V) if dist_ref[i] < float("inf")]
    if not reachable:
        return {"wrong_nodes": 0, "wrong_ratio": 0.0,
                "max_abs_err": 0.0, "mean_abs_err": 0.0,
                "max_rel_err": 0.0}

    errors     = []
    wrong      = 0
    for i in reachable:
        ref  = dist_ref[i]
        approx = dist_approx[i]
        ae   = abs(ref - approx)
        if ae > 1e-9:
            wrong += 1
            errors.append(ae)

    n_reach = len(reachable)
    return {
        "wrong_nodes":   wrong,
        "wrong_ratio":   wrong / n_reach,
        "max_abs_err":   float(max(errors))       if errors else 0.0,
        "mean_abs_err":  float(np.mean(errors))   if errors else 0.0,
        "max_rel_err":   float(max(ae / dist_ref[i]
                                   for i, ae in zip(
                                       [j for j in reachable
                                        if abs(dist_ref[j]-dist_approx[j])>1e-9],
                                       errors)
                                   )) if errors else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# 실험 루프
# ─────────────────────────────────────────────────────────────────────

def run_experiment() -> pd.DataFrame:
    rows  = []
    total = len(V_LIST) * len(GRAPH_TYPES) * len(THETAS) * len(SEEDS)
    print(f"총 조합: {total}  (V×graph×θ×seed)\n")

    with tqdm(total=len(V_LIST)*len(GRAPH_TYPES)*len(THETAS), desc="5차 실험") as pbar:
        for V in V_LIST:
            for gtype, k in GRAPH_TYPES.items():
                for theta in THETAS:

                    # 다중 seed → 정확성 분포 측정
                    acc_wrong_ratios = []
                    acc_max_err      = []
                    all_push, all_pop, all_skip, all_blocked = [], [], [], []
                    times = []

                    for seed in SEEDS:
                        adj      = make_graph(V, k, seed)
                        dist_ref, *_ = dijkstra_standard(adj, 0, V)

                        # heap ops (결정론적 → seed당 1회)
                        dist_t, pc, poc, sc, bc = run_dijkstra(adj, 0, V, theta)

                        acc = measure_accuracy(dist_ref, dist_t, V)
                        acc_wrong_ratios.append(acc["wrong_ratio"])
                        acc_max_err.append(acc["max_abs_err"])
                        all_push.append(pc)
                        all_pop.append(poc)
                        all_skip.append(sc)
                        all_blocked.append(bc)

                        # 타이밍 (10회 평균)
                        t_list = []
                        for _ in range(REPEATS):
                            t0 = time.perf_counter()
                            run_dijkstra(adj, 0, V, theta)
                            t_list.append(time.perf_counter() - t0)
                        times.append(np.mean(t_list))

                    row = {
                        "V":              V,
                        "graph_type":     gtype,
                        "k":              k,
                        "theta":          theta if theta != float("inf") else "inf",
                        "push_mean":      round(float(np.mean(all_push)),   1),
                        "pop_mean":       round(float(np.mean(all_pop)),    1),
                        "skip_mean":      round(float(np.mean(all_skip)),   1),
                        "blocked_mean":   round(float(np.mean(all_blocked)),1),
                        "skip_ratio":     round(float(np.mean(all_skip)) /
                                               max(float(np.mean(all_pop)),1), 4),
                        "push_reduction": 0.0,    # 나중에 채움
                        "wrong_ratio_mean": round(float(np.mean(acc_wrong_ratios)), 5),
                        "wrong_ratio_max":  round(float(np.max(acc_wrong_ratios)),  5),
                        "max_abs_err_mean": round(float(np.mean(acc_max_err)),      4),
                        "max_abs_err_max":  round(float(np.max(acc_max_err)),       4),
                        "time_mean":       round(float(np.mean(times)), 6),
                        "time_std":        round(float(np.std(times)),  6),
                    }
                    rows.append(row)
                    pbar.update(1)
                    pbar.write(
                        f"  V={V:>6,}  {gtype:<12}  θ={str(theta):>4}  "
                        f"push={row['push_mean']:>7.0f}  "
                        f"blocked={row['blocked_mean']:>6.0f}  "
                        f"wrong={row['wrong_ratio_mean']*100:>6.2f}%  "
                        f"time={row['time_mean']*1000:>7.2f}ms"
                    )

    df = pd.DataFrame(rows)

    # push_reduction: 동일 (V, graph_type)에서 standard 대비 감소율
    for V in V_LIST:
        for gtype in GRAPH_TYPES:
            mask_std = (df["V"]==V) & (df["graph_type"]==gtype) & (df["theta"]=="inf")
            base_push = df.loc[mask_std, "push_mean"].values[0]
            mask_all  = (df["V"]==V) & (df["graph_type"]==gtype)
            df.loc[mask_all, "push_reduction"] = (
                (base_push - df.loc[mask_all, "push_mean"]) / base_push
            ).round(4)

    return df


# ─────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, timestamp: str) -> None:

    gtypes = list(GRAPH_TYPES.keys())
    theta_vals = [float("inf"), 1, 2, 3]
    theta_labels = [LABELS_THETA[t] for t in theta_vals]
    theta_str = ["inf", "1", "2", "3"]

    # ── Fig 1: 실행 시간 비교 (V × graph_type × θ) ───────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    fig.suptitle("Execution Time: Standard vs Threshold Variants", fontsize=12, fontweight="bold")
    for row_i, V in enumerate(V_LIST):
        for col_i, gtype in enumerate(gtypes):
            ax  = axes[row_i][col_i]
            sub = df[(df["V"]==V) & (df["graph_type"]==gtype)]
            sub = sub.set_index("theta").reindex(theta_str)
            times  = sub["time_mean"].values * 1000
            colors = [COLORS_THETA[t] for t in theta_vals]
            bars   = ax.bar(theta_labels, times, color=colors, alpha=0.85, edgecolor="white")
            for bar, t in zip(bars, times):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                        f"{t:.1f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(f"V={V:,}  {gtype}  (k={GRAPH_TYPES[gtype]})", fontsize=9)
            ax.set_ylabel("Time (ms)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p5_time_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: 정확성 — wrong_ratio (핵심) ───────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Accuracy: Wrong Node Ratio  (0 = perfect, 1 = all wrong)", fontsize=12, fontweight="bold")
    for row_i, V in enumerate(V_LIST):
        for col_i, gtype in enumerate(gtypes):
            ax  = axes[row_i][col_i]
            sub = df[(df["V"]==V) & (df["graph_type"]==gtype)]
            sub = sub.set_index("theta").reindex(theta_str)
            wrong_mean = sub["wrong_ratio_mean"].values
            wrong_max  = sub["wrong_ratio_max"].values
            colors     = [COLORS_THETA[t] for t in theta_vals]
            bars = ax.bar(theta_labels, wrong_mean * 100, color=colors, alpha=0.85, edgecolor="white")
            # max 표시 (오차 막대)
            for i, (bar, wmax) in enumerate(zip(bars, wrong_max)):
                ax.errorbar(bar.get_x()+bar.get_width()/2,
                            wrong_mean[i]*100,
                            yerr=[[0], [(wmax-wrong_mean[i])*100]],
                            fmt="none", color="black", capsize=4, lw=1.2)
                if wrong_mean[i] > 0:
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+0.5,
                            f"{wrong_mean[i]*100:.1f}%", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"V={V:,}  {gtype}  (k={GRAPH_TYPES[gtype]})", fontsize=9)
            ax.set_ylabel("Wrong nodes (%)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p5_accuracy_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: 핵심 — Speedup vs Wrong_ratio 트레이드오프 ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tradeoff: Speedup vs Accuracy Loss  (ideal = top-left)", fontsize=12, fontweight="bold")

    for ax_i, V in enumerate(V_LIST):
        ax = axes[ax_i]
        for gtype in gtypes:
            sub_std = df[(df["V"]==V)&(df["graph_type"]==gtype)&(df["theta"]=="inf")]
            t_std   = sub_std["time_mean"].values[0]
            for theta, ts in zip(theta_str[1:], theta_vals[1:]):   # θ=1,2,3
                sub = df[(df["V"]==V)&(df["graph_type"]==gtype)&(df["theta"]==theta)]
                if sub.empty:
                    continue
                speedup    = t_std / sub["time_mean"].values[0]
                wrong_pct  = sub["wrong_ratio_mean"].values[0] * 100
                ax.scatter(wrong_pct, speedup,
                           s=100, alpha=0.85,
                           label=f"{gtype} θ={ts}")
                ax.annotate(f"θ={ts}",
                            (wrong_pct, speedup),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=7)
        ax.axhline(1.0, color="gray", lw=1, ls=":")
        ax.axvline(0.0, color="gray", lw=1, ls=":")
        ax.text(0.5, 1.02, "ideal zone", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform(), ha="center")
        ax.set_title(f"V={V:,}")
        ax.set_xlabel("Wrong nodes (%)")
        ax.set_ylabel("Speedup vs Standard")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p5_tradeoff_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 4: push 감소율 vs wrong_ratio ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Push Reduction vs Wrong Node Ratio", fontsize=12, fontweight="bold")
    for ax_i, V in enumerate(V_LIST):
        ax = axes[ax_i]
        for gtype in gtypes:
            for theta, ts in zip(theta_str[1:], theta_vals[1:]):
                sub = df[(df["V"]==V)&(df["graph_type"]==gtype)&(df["theta"]==theta)]
                if sub.empty:
                    continue
                pr  = sub["push_reduction"].values[0] * 100
                wr  = sub["wrong_ratio_mean"].values[0] * 100
                ax.scatter(pr, wr, s=90, alpha=0.85)
                ax.annotate(f"{gtype} θ={ts}",
                            (pr, wr), textcoords="offset points",
                            xytext=(3, 3), fontsize=7)
        ax.set_title(f"V={V:,}")
        ax.set_xlabel("Push reduction (%)")
        ax.set_ylabel("Wrong nodes (%)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p5_push_vs_accuracy_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# 요약 출력
# ─────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "="*90)
    print("5차 실험 결과 요약")
    print("="*90)
    print(f"{'V':>7}  {'graph':>12}  {'θ':>4}  "
          f"{'push':>7}  {'blocked':>8}  {'push_red%':>9}  "
          f"{'wrong%':>7}  {'max_err':>8}  {'time_ms':>8}")
    print("-"*90)
    for _, r in df.sort_values(["V","graph_type","theta"]).iterrows():
        theta_disp = "inf" if r["theta"] == "inf" else str(r["theta"])
        print(
            f"{int(r.V):>7,}  {r.graph_type:>12}  {theta_disp:>4}  "
            f"{r.push_mean:>7.0f}  {r.blocked_mean:>8.0f}  "
            f"{r.push_reduction*100:>8.1f}%  "
            f"{r.wrong_ratio_mean*100:>6.2f}%  "
            f"{r.max_abs_err_mean:>8.4f}  "
            f"{r.time_mean*1000:>8.2f}"
        )


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 5차 실험 — Threshold 변형 속도/정확성 트레이드오프")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = run_experiment()
    csv_path = RESULTS_DIR / f"experiment5_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n데이터 저장 → {csv_path}")

    print_summary(df)
    plot_results(df, ts)

    print(f"\nDone. All outputs in {RESULTS_DIR.resolve()}/")
