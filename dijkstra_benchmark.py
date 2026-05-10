"""
Dijkstra Algorithm Benchmark  —  v2 (heap-ops instrumentation)

실험 구성
─────────
Phase 1 · Baseline
  그래프 유형(sparse / dense / grid) × 노드 수(100 ~ 100k)
  → wall-clock time 측정 (heapq vs NetworkX)

Phase 2 · Heap Operation Analysis  ← 핵심 추가
  동일 조건에서 heap push / pop / skip(lazy-delete) / relax 횟수 측정
  → skip_ratio = skip/pop : "이론 대비 낭비된 연산 비율"

Phase 3 · Weight Distribution Study
  동일 토폴로지에 가중치 분포만 교체
    - uniform  : 균등 분포 (기준선)
    - biased   : 지수 분포 (소수의 큰 가중치, 다수의 작은 가중치)
    - clustered: 클러스터 내부는 극소 / 클러스터 간은 극대
  → 가중치 패턴이 skip_ratio를 어떻게 변화시키는지 분석
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
import networkx as nx
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS      = 5    # wall-clock 반복 횟수 (heap 카운터는 결정론적 → 1회)
SPARSE_DEGREE = 10
DENSE_PROB    = 0.10
N_CLUSTERS    = 8   # Phase 3 clustered weight 클러스터 수
SEED          = 42

NODE_SIZES = {
    "sparse": [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000],
    "dense":  [100, 500, 1_000, 2_000, 5_000, 10_000],
    "grid":   [100, 400, 900, 2_500, 10_000, 40_000, 90_000],
}

# Phase 3: sparse 그래프에서 가중치 분포만 교체
WEIGHT_STUDY_SIZES = [500, 1_000, 5_000, 10_000, 50_000]

COLORS_GRAPH  = {"sparse": "#2196F3", "dense": "#F44336", "grid": "#4CAF50"}
COLORS_WEIGHT = {"uniform": "#607D8B", "biased": "#FF9800", "clustered": "#9C27B0"}


# ─────────────────────────────────────────────────────────────────────
# Weight distribution functions
#   signature: fn(rng, u, v, n) → float
# ─────────────────────────────────────────────────────────────────────

def wt_uniform(rng: random.Random, u: int, v: int, n: int) -> float:
    return round(rng.uniform(0.1, 10.0), 4)


def wt_biased(rng: random.Random, u: int, v: int, n: int) -> float:
    # 지수분포: 작은 가중치가 압도적으로 많고, 간헐적으로 매우 큰 값
    return round(min(rng.expovariate(1.5) + 0.01, 200.0), 4)


def wt_clustered(rng: random.Random, u: int, v: int, n: int) -> float:
    # 같은 클러스터: 0.01~0.5 (매우 작음)  →  내부 relaxation 폭발 유도
    # 다른 클러스터: 50~200  (매우 큼)      →  cross-edge 통한 재완화로 stale entry 급증
    cu = u * N_CLUSTERS // n
    cv = v * N_CLUSTERS // n
    if cu == cv:
        return round(rng.uniform(0.01, 0.5), 4)
    return round(rng.uniform(50.0, 200.0), 4)


WEIGHT_FNS = {
    "uniform":   wt_uniform,
    "biased":    wt_biased,
    "clustered": wt_clustered,
}


# ─────────────────────────────────────────────────────────────────────
# Graph generators
# ─────────────────────────────────────────────────────────────────────

def _ensure_connected(adj: dict, n: int, rng: random.Random,
                      wt_fn=wt_uniform) -> None:
    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        w = wt_fn(rng, u, v, n)
        adj[u].append((v, w))
        adj[v].append((u, w))


def make_sparse(n: int, seed: int = SEED, wt_fn=wt_uniform) -> dict:
    rng = random.Random(seed)
    adj = {u: [] for u in range(n)}
    for _ in range(n * SPARSE_DEGREE // 2):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            w = wt_fn(rng, u, v, n)
            adj[u].append((v, w))
            adj[v].append((u, w))
    _ensure_connected(adj, n, rng, wt_fn)
    return adj


def make_dense(n: int, seed: int = SEED, wt_fn=wt_uniform) -> dict:
    rng = random.Random(seed)
    adj = {u: [] for u in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < DENSE_PROB:
                w = wt_fn(rng, u, v, n)
                adj[u].append((v, w))
                adj[v].append((u, w))
    _ensure_connected(adj, n, rng, wt_fn)
    return adj


def make_grid(n: int, seed: int = SEED, wt_fn=wt_uniform) -> tuple[dict, int]:
    rng = random.Random(seed)
    side = int(math.isqrt(n))
    actual_n = side * side
    adj = {u: [] for u in range(actual_n)}
    for r in range(side):
        for c in range(side):
            u = r * side + c
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr < side and nc < side:
                    v = nr * side + nc
                    w = wt_fn(rng, u, v, actual_n)
                    adj[u].append((v, w))
                    adj[v].append((u, w))
    return adj, actual_n


# ─────────────────────────────────────────────────────────────────────
# Dijkstra implementations
# ─────────────────────────────────────────────────────────────────────

def dijkstra_heapq(adj: dict, source: int, n: int) -> list:
    dist = [float("inf")] * n
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


def dijkstra_heapq_counted(adj: dict, source: int, n: int) -> dict:
    """
    heap 연산을 카운트하는 계측 버전.

    반환 키:
      push_count  : heappush 총 횟수 (초기 push 1 포함)
      pop_count   : heappop 총 횟수
      skip_count  : d > dist[u] 로 버려진 pop (= stale / lazy-delete overhead)
      relax_count : 성공적으로 완화된 edge 수 (= push_count - 1)
      skip_ratio  : skip_count / pop_count  — "낭비 연산 비율"  ← 핵심 지표
    """
    dist = [float("inf")] * n
    dist[source] = 0.0
    heap = [(0.0, source)]
    push_count  = 1
    pop_count   = 0
    skip_count  = 0
    relax_count = 0

    while heap:
        d, u = heapq.heappop(heap)
        pop_count += 1
        if d > dist[u]:          # stale entry — lazy deletion
            skip_count += 1
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                push_count  += 1
                relax_count += 1

    return {
        "push_count":  push_count,
        "pop_count":   pop_count,
        "skip_count":  skip_count,
        "relax_count": relax_count,
        "skip_ratio":  skip_count / pop_count if pop_count else 0.0,
    }


def dijkstra_networkx(G_nx: nx.Graph, source: int) -> dict:
    return nx.single_source_dijkstra_path_length(G_nx, source, weight="weight")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def adj_to_nx(adj: dict, n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    seen = set()
    for u in range(n):
        for v, w in adj[u]:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                G.add_edge(u, v, weight=w)
    return G


def _time(fn, repeats: int) -> tuple[float, float, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times)), float(np.min(times))


# ─────────────────────────────────────────────────────────────────────
# Phase 1+2  benchmark runner
# ─────────────────────────────────────────────────────────────────────

def run_one(adj: dict, n: int, source: int = 0) -> dict:
    edge_count = sum(len(v) for v in adj.values()) // 2
    stats = {"n": n, "edge_count": edge_count}

    # wall-clock: heapq
    m, s, mn = _time(lambda: dijkstra_heapq(adj, source, n), REPEATS)
    stats.update(heapq_mean=m, heapq_std=s, heapq_min=mn)

    # wall-clock: NetworkX
    G_nx = adj_to_nx(adj, n)
    m, s, mn = _time(lambda: dijkstra_networkx(G_nx, source), REPEATS)
    stats.update(nx_mean=m, nx_std=s, nx_min=mn)

    # heap operation counts (결정론적 → 1회로 충분)
    ops = dijkstra_heapq_counted(adj, source, n)
    stats.update(ops)

    # 파생 지표
    stats["push_per_node"] = ops["push_count"] / n        # 노드당 평균 push 횟수
    stats["pop_per_edge"]  = ops["pop_count"] / max(edge_count, 1)

    return stats


# ─────────────────────────────────────────────────────────────────────
# Phase 3  weight distribution study
# ─────────────────────────────────────────────────────────────────────

def run_weight_study(timestamp: str) -> pd.DataFrame:
    """
    동일 sparse 토폴로지(고정 seed) + 가중치 분포만 교체
    → skip_ratio 변화를 통해 "왜 느린가"의 원인 분리
    """
    rows = []
    print(f"\n{'='*50}")
    print("Phase 3 · Weight Distribution Study (sparse topology)")
    print(f"{'='*50}")

    for wt_name, wt_fn in WEIGHT_FNS.items():
        print(f"\n  Weight: {wt_name}")
        for n in tqdm(WEIGHT_STUDY_SIZES, desc=f"  {wt_name}"):
            # 토폴로지는 동일 (SEED 고정), 가중치 함수만 교체
            adj = make_sparse(n, seed=SEED, wt_fn=wt_fn)
            ops = dijkstra_heapq_counted(adj, source=0, n=n)
            m, _, _ = _time(lambda: dijkstra_heapq(adj, 0, n), REPEATS)
            row = {
                "weight_dist": wt_name,
                "n":           n,
                "edge_count":  sum(len(v) for v in adj.values()) // 2,
                "heapq_mean":  m,
            }
            row.update(ops)
            row["push_per_node"] = ops["push_count"] / n
            rows.append(row)

            print(
                f"    n={n:>7,}  skip_ratio={ops['skip_ratio']:.3f}  "
                f"push/node={row['push_per_node']:.2f}  "
                f"time={m*1000:.2f}ms"
            )

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / f"weight_study_{timestamp}.csv"
    df.to_csv(path, index=False)
    print(f"\nWeight study saved → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────
# Phase 1+2  main loop
# ─────────────────────────────────────────────────────────────────────

def run_baseline(timestamp: str) -> pd.DataFrame:
    rows = []
    tasks = [
        ("sparse", NODE_SIZES["sparse"], make_sparse, False),
        ("dense",  NODE_SIZES["dense"],  make_dense,  False),
        ("grid",   NODE_SIZES["grid"],   make_grid,   True),
    ]

    for graph_type, sizes, generator, is_grid in tasks:
        print(f"\n{'='*50}")
        print(f"Phase 1+2 · {graph_type.upper()}")
        print(f"{'='*50}")
        for target_n in tqdm(sizes, desc=graph_type):
            adj, actual_n = generator(target_n) if is_grid else (generator(target_n), target_n)
            stats = run_one(adj, actual_n)
            stats.update(graph_type=graph_type, target_n=target_n)
            rows.append(stats)

            print(
                f"  n={actual_n:>7,}  E={stats['edge_count']:>9,}  "
                f"skip={stats['skip_ratio']:.3f}  "
                f"push/node={stats['push_per_node']:.2f}  "
                f"heapq={stats['heapq_mean']*1000:7.2f}ms  "
                f"nx={stats['nx_mean']*1000:7.2f}ms"
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"baseline_{timestamp}.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_baseline(df: pd.DataFrame, timestamp: str) -> None:

    # ── Fig 1: wall-clock time ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase 1 · Wall-Clock Time", fontsize=13, fontweight="bold")
    for ax, gtype in zip(axes, ["sparse", "dense", "grid"]):
        sub   = df[df["graph_type"] == gtype].sort_values("n")
        color = COLORS_GRAPH[gtype]
        ax.plot(sub["n"], sub["heapq_mean"], color=color, marker="o", lw=2, label="heapq")
        ax.fill_between(sub["n"],
                        sub["heapq_mean"] - sub["heapq_std"],
                        sub["heapq_mean"] + sub["heapq_std"],
                        alpha=0.15, color=color)
        ax.plot(sub["n"], sub["nx_mean"], color=color, marker="s",
                lw=2, ls="--", alpha=0.7, label="NetworkX")
        ax.set_title(gtype.capitalize()); ax.set_xlabel("V"); ax.set_ylabel("sec")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p1_time_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: skip_ratio — 핵심 지표 ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 2 · Heap Operation Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    for gtype in ["sparse", "dense", "grid"]:
        sub = df[df["graph_type"] == gtype].sort_values("n")
        ax.plot(sub["n"], sub["skip_ratio"], color=COLORS_GRAPH[gtype],
                marker="o", lw=2, label=gtype.capitalize())
    ax.set_title("skip_ratio = skip_pop / total_pop\n(높을수록 heap 낭비 심함)")
    ax.set_xlabel("V"); ax.set_ylabel("skip ratio (0–1)")
    ax.set_xscale("log"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax = axes[1]
    for gtype in ["sparse", "dense", "grid"]:
        sub = df[df["graph_type"] == gtype].sort_values("n")
        ax.plot(sub["n"], sub["push_per_node"], color=COLORS_GRAPH[gtype],
                marker="o", lw=2, label=gtype.capitalize())
    ax.axhline(1.0, color="gray", lw=1, ls=":", label="ideal (1×)")
    ax.set_title("push_count / V\n(1에 가까울수록 이론적으로 효율적)")
    ax.set_xlabel("V"); ax.set_ylabel("pushes per node")
    ax.set_xscale("log"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p2_heap_ops_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: skip_ratio vs heapq time (scatter — 상관관계) ─────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for gtype in ["sparse", "dense", "grid"]:
        sub = df[df["graph_type"] == gtype]
        ax.scatter(sub["skip_ratio"], sub["heapq_mean"],
                   color=COLORS_GRAPH[gtype], s=60, alpha=0.85,
                   edgecolors="white", lw=0.5, label=gtype.capitalize())
    ax.set_title("Phase 2 · skip_ratio vs Runtime\n(상관관계가 높으면 skip이 병목임을 의미)")
    ax.set_xlabel("skip_ratio"); ax.set_ylabel("heapq time (sec)")
    ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p2_skip_vs_time_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Baseline plots saved → {RESULTS_DIR}/p1_* p2_*")


def plot_weight_study(df: pd.DataFrame, timestamp: str) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase 3 · Weight Distribution  ×  Heap Behavior",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("skip_ratio",    "skip_ratio",     "skip_ratio (낭비 연산 비율)"),
        ("push_per_node", "push_per_node",  "push_count / V"),
        ("heapq_mean",    "heapq_mean",     "Runtime (sec)"),
    ]
    for ax, (col, _, ylabel) in zip(axes, metrics):
        for wt_name in ["uniform", "biased", "clustered"]:
            sub = df[df["weight_dist"] == wt_name].sort_values("n")
            ax.plot(sub["n"], sub[col],
                    color=COLORS_WEIGHT[wt_name], marker="o", lw=2,
                    label=wt_name.capitalize())
        ax.set_title(ylabel)
        ax.set_xlabel("V"); ax.set_xscale("log")
        if col == "heapq_mean":
            ax.set_yscale("log")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    path = RESULTS_DIR / f"p3_weight_study_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Weight study plot → {path}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra Benchmark  v2")
    print(f"Repeats : {REPEATS}  |  Seed : {SEED}  |  Started : {datetime.now():%Y-%m-%d %H:%M:%S}")

    df_base   = run_baseline(ts)
    df_weight = run_weight_study(ts)

    plot_baseline(df_base, ts)
    plot_weight_study(df_weight, ts)

    print(f"\n{'='*50}")
    print("All outputs saved to:", RESULTS_DIR.resolve())
    print(f"{'='*50}")
