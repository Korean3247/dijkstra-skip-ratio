"""
Dijkstra 2차 실험: skip_ratio 원인 분리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 질문: skip_ratio는 그래프 밀도(density) 때문인가,
          평균 차수(average degree k) 때문인가?

설계: V × k 격자 (5×5 = 25가지 조합)
  V: 500 / 1,000 / 5,000 / 10,000 / 50,000
  k: 4 / 8 / 16 / 32 / 64

각 조합마다 랜덤 그래프(고정 seed) 생성 후
  - skip_ratio  = skip_pop / total_pop
  - push_per_node = push_count / V
  - wall-clock time (5회 평균)
측정.

기대: skip_ratio ∝ k 패턴 확인 → 수식화 근거 확보
"""

import heapq
import math
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
from scipy import stats
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS = 5
SEED    = 42

V_SIZES = [500, 1_000, 5_000, 10_000, 50_000]
K_LIST  = [4, 8, 16, 32, 64]   # 평균 차수

# 시각화용 색상 (V별 구분)
V_COLORS = {
    500:    "#1565C0",
    1_000:  "#2196F3",
    5_000:  "#4CAF50",
    10_000: "#FF9800",
    50_000: "#F44336",
}


# ─────────────────────────────────────────────────────────────────────
# Graph generator: 평균 차수 k 고정 랜덤 그래프
# ─────────────────────────────────────────────────────────────────────

def make_graph_fixed_degree(V: int, k: int, seed: int = SEED) -> dict:
    """
    V개 노드, 평균 차수 ≈ k 인 랜덤 무방향 그래프.
    edge_count ≈ V*k/2 개의 엣지를 균등 랜덤 샘플링으로 생성.
    연결성 보장을 위해 랜덤 spanning path 추가.
    """
    rng = random.Random(seed)
    adj = {u: [] for u in range(V)}

    target_edges = V * k // 2
    for _ in range(target_edges):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            w = round(rng.uniform(0.1, 10.0), 4)
            adj[u].append((v, w))
            adj[v].append((u, w))

    # 연결성 보장 (spanning path)
    nodes = list(range(V))
    rng.shuffle(nodes)
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        w = round(rng.uniform(0.1, 10.0), 4)
        adj[u].append((v, w))
        adj[v].append((u, w))

    return adj


def actual_avg_degree(adj: dict, V: int) -> float:
    return sum(len(nbrs) for nbrs in adj.values()) / V


# ─────────────────────────────────────────────────────────────────────
# Dijkstra (계측 버전)
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
        "push_count": push_count,
        "pop_count":  pop_count,
        "skip_count": skip_count,
        "skip_ratio": skip_count / pop_count if pop_count else 0.0,
        "push_per_node": push_count / V,
    }


def dijkstra_plain(adj: dict, source: int, V: int) -> list:
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
    return dist


# ─────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────

def run_experiment() -> pd.DataFrame:
    rows = []
    total = len(V_SIZES) * len(K_LIST)

    print(f"V × k 격자 실험: {len(V_SIZES)} × {len(K_LIST)} = {total}가지 조합")
    print(f"반복 횟수: {REPEATS}회\n")

    with tqdm(total=total, desc="진행") as pbar:
        for V in V_SIZES:
            for k in K_LIST:
                adj  = make_graph_fixed_degree(V, k)
                k_actual = actual_avg_degree(adj, V)
                E    = sum(len(nbrs) for nbrs in adj.values()) // 2
                density = E / (V * (V - 1) / 2)

                # heap 연산 카운트 (결정론적 → 1회)
                ops = dijkstra_counted(adj, source=0, V=V)

                # wall-clock (5회 평균)
                times = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    dijkstra_plain(adj, 0, V)
                    times.append(time.perf_counter() - t0)

                row = {
                    "V":             V,
                    "k_target":      k,
                    "k_actual":      round(k_actual, 2),
                    "edge_count":    E,
                    "density":       round(density, 6),
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
                    f"  V={V:>6,}  k={k:>2}  "
                    f"skip={ops['skip_ratio']:.3f}  "
                    f"push/V={ops['push_per_node']:.3f}  "
                    f"time={np.mean(times)*1000:.2f}ms"
                )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, timestamp: str) -> None:

    # ── Fig 1: skip_ratio vs k (V별 분리, 회귀선 포함) ───────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    for V in V_SIZES:
        sub = df[df["V"] == V].sort_values("k_target")
        color = V_COLORS[V]

        ax.plot(sub["k_target"], sub["skip_ratio"],
                color=color, marker="o", lw=2, label=f"V={V:,}")

        # 선형 회귀선
        slope, intercept, r, *_ = stats.linregress(sub["k_target"], sub["skip_ratio"])
        x_fit = np.array(K_LIST)
        ax.plot(x_fit, intercept + slope * x_fit,
                color=color, lw=1, ls="--", alpha=0.5)

    ax.set_title("skip_ratio vs Average Degree k\n(점선: 선형 회귀)", fontsize=12)
    ax.set_xlabel("Average Degree k")
    ax.set_ylabel("skip_ratio  (skip_pop / total_pop)")
    ax.legend(fontsize=9, title="Node count")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"skip_ratio_vs_k_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: push_per_node vs k ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(1.0, color="gray", lw=1, ls=":", label="ideal (1×)")

    for V in V_SIZES:
        sub = df[df["V"] == V].sort_values("k_target")
        ax.plot(sub["k_target"], sub["push_per_node"],
                color=V_COLORS[V], marker="s", lw=2, label=f"V={V:,}")

    ax.set_title("push_count / V  vs  Average Degree k\n(1에 가까울수록 이론적으로 효율적)", fontsize=12)
    ax.set_xlabel("Average Degree k")
    ax.set_ylabel("push_count / V")
    ax.legend(fontsize=9, title="Node count")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"push_per_node_vs_k_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: skip_ratio vs density (V별) ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for V in V_SIZES:
        sub = df[df["V"] == V].sort_values("density")
        ax.plot(sub["density"], sub["skip_ratio"],
                color=V_COLORS[V], marker="o", lw=2, label=f"V={V:,}")

    ax.set_title("skip_ratio vs Graph Density\n(같은 k라도 V가 클수록 density가 낮아짐)", fontsize=12)
    ax.set_xlabel("Edge Density  (E / V(V-1)/2)")
    ax.set_ylabel("skip_ratio")
    ax.set_xscale("log")
    ax.legend(fontsize=9, title="Node count")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"skip_ratio_vs_density_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 4: heatmap — skip_ratio (V × k 격자) ─────────────────────
    pivot = df.pivot(index="V", columns="k_target", values="skip_ratio")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(K_LIST)));   ax.set_xticklabels(K_LIST)
    ax.set_yticks(range(len(V_SIZES))); ax.set_yticklabels([f"{v:,}" for v in V_SIZES])
    ax.set_xlabel("Average Degree k")
    ax.set_ylabel("Node count V")
    ax.set_title("skip_ratio Heatmap  (V × k)\n붉을수록 낭비 심함", fontsize=12)
    plt.colorbar(im, ax=ax, label="skip_ratio")
    for i in range(len(V_SIZES)):
        for j in range(len(K_LIST)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if pivot.values[i, j] > 0.65 else "black")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"heatmap_skip_ratio_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# 회귀 분석 요약 출력
# ─────────────────────────────────────────────────────────────────────

def print_regression_summary(df: pd.DataFrame) -> None:
    print("\n── 선형 회귀: skip_ratio ~ k  (V별) ──")
    print(f"{'V':>8}  {'slope':>8}  {'intercept':>10}  {'R²':>6}")
    print("-" * 42)
    for V in V_SIZES:
        sub = df[df["V"] == V].sort_values("k_target")
        slope, intercept, r, *_ = stats.linregress(sub["k_target"], sub["skip_ratio"])
        print(f"{V:>8,}  {slope:>8.5f}  {intercept:>10.5f}  {r**2:>6.4f}")

    print("\n── 선형 회귀: skip_ratio ~ k  (전체) ──")
    slope, intercept, r, p, _ = stats.linregress(df["k_target"], df["skip_ratio"])
    print(f"slope={slope:.5f}  intercept={intercept:.5f}  R²={r**2:.4f}  p={p:.2e}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 2차 실험 — skip_ratio 원인 분리")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = run_experiment()

    csv_path = RESULTS_DIR / f"experiment2_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData saved → {csv_path}")

    print_regression_summary(df)
    plot_results(df, ts)

    print(f"\nDone. All outputs in {RESULTS_DIR.resolve()}/")
