"""
Dijkstra 8차 실험: Dial's stale_skip 메커니즘 정량화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
목적: k가 커질수록 Dial's stale_ratio가 감소하는 현상의 원인 수식화

추가 측정:
  - stale_ratio   = stale_skip / (stale_skip + settled)  [실제 pop 대비]
  - raw_stale     = stale_skip / bucket_push_count
  - avg_nodes_per_bucket = settled / occupied_buckets
  - bucket_traversals   = current_dist 진행 횟수
  - avg_dist_spread     = max_settled_dist / V  (경로 거리 분포 지표)
"""

import heapq
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

RESULTS_DIR = Path("results8")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS = 10
SEED    = 42
W_MAX   = 100

V_LIST = [1_000, 10_000]
K_LIST = [4, 8, 16, 32, 64, 128]

K_COLORS = {
    4:  "#1565C0", 8:  "#2196F3", 16: "#4CAF50",
    32: "#FF9800", 64: "#F44336", 128:"#9C27B0"
}


# ─────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────

def make_graph_int(V, k, seed=SEED):
    rng = random.Random(seed)
    adj = [[] for _ in range(V)]
    for _ in range(V * k // 2):
        u = rng.randint(0, V-1); v = rng.randint(0, V-1)
        if u != v:
            w = rng.randint(1, W_MAX)
            adj[u].append((v,w)); adj[v].append((u,w))
    nodes = list(range(V)); rng.shuffle(nodes)
    for i in range(len(nodes)-1):
        u,v = nodes[i],nodes[i+1]; w = rng.randint(1, W_MAX)
        adj[u].append((v,w)); adj[v].append((u,w))
    return adj


# ─────────────────────────────────────────────────────────────────────
# heapq (reference + skip_ratio)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_heapq(adj, source, V):
    dist = [float("inf")] * V
    dist[source] = 0
    heap = [(0, source)]
    push_c = pop_c = skip_c = 0
    push_c = 1
    while heap:
        d, u = heapq.heappop(heap); pop_c += 1
        if d > dist[u]: skip_c += 1; continue
        for v,w in adj[u]:
            nd = d+w
            if nd < dist[v]:
                dist[v]=nd; heapq.heappush(heap,(nd,v)); push_c+=1
    return dist, skip_c/pop_c if pop_c else 0.0


# ─────────────────────────────────────────────────────────────────────
# Dial's (extended instrumentation)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_dial_instrumented(adj, source, V, W=W_MAX):
    INF = float("inf")
    dist = [INF] * V
    dist[source] = 0
    nb = W + 1
    buckets = [deque() for _ in range(nb)]
    buckets[0].append(source)

    bucket_push_count  = 1
    stale_skip_count   = 0
    settled_count      = 0
    occupied_buckets   = 0    # buckets that had at least one valid settlement
    bucket_traversals  = 0    # number of times current_dist advances
    max_settled_dist   = 0

    cur = 0
    max_possible = V * W

    while settled_count < V and cur <= max_possible:
        b = cur % nb
        had_valid = False
        while buckets[b]:
            u = buckets[b].popleft()
            if dist[u] != cur:
                stale_skip_count += 1
                continue
            settled_count += 1
            if not had_valid:
                had_valid = True
                occupied_buckets += 1
            max_settled_dist = max(max_settled_dist, cur)
            for v,w in adj[u]:
                nd = cur + w
                if nd < dist[v]:
                    dist[v] = nd
                    buckets[nd % nb].append(v)
                    bucket_push_count += 1
        cur += 1
        bucket_traversals += 1

    total_pops = stale_skip_count + settled_count
    stale_ratio_pop  = stale_skip_count / total_pops if total_pops else 0.0
    raw_stale        = stale_skip_count / bucket_push_count if bucket_push_count else 0.0
    avg_nodes_per_bucket = settled_count / occupied_buckets if occupied_buckets else 0.0
    avg_dist_spread  = max_settled_dist / V if V else 0.0

    return dist, {
        "bucket_push_count":  bucket_push_count,
        "stale_skip_count":   stale_skip_count,
        "settled_count":      settled_count,
        "occupied_buckets":   occupied_buckets,
        "bucket_traversals":  bucket_traversals,
        "stale_ratio_pop":    round(stale_ratio_pop, 6),
        "raw_stale":          round(raw_stale, 6),
        "avg_nodes_per_bucket": round(avg_nodes_per_bucket, 4),
        "avg_dist_spread":    round(avg_dist_spread, 4),
        "max_settled_dist":   max_settled_dist,
    }


# ─────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────

def run_experiment():
    rows = []
    total = len(V_LIST) * len(K_LIST)
    with tqdm(total=total, desc="진행") as pbar:
        for V in V_LIST:
            for k in K_LIST:
                adj = make_graph_int(V, k)
                E   = sum(len(nb2) for nb2 in adj) // 2

                # heapq reference
                ref_dist, hq_skip_ratio = dijkstra_heapq(adj, 0, V)

                # Dial's (1회 카운트, REPEATS회 시간)
                _, ops = dijkstra_dial_instrumented(adj, 0, V)

                times = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    dijkstra_dial_instrumented(adj, 0, V)
                    times.append(time.perf_counter() - t0)

                row = {
                    "V":     V, "k": k, "E": E,
                    "hq_skip_ratio": round(hq_skip_ratio, 6),
                    "time_mean_ms": round(float(np.mean(times))*1000, 3),
                    "time_std_ms":  round(float(np.std(times))*1000, 3),
                }
                row.update(ops)
                rows.append(row)
                pbar.update(1)
                pbar.write(
                    f"  V={V:>6,} k={k:>3}  "
                    f"stale_pop={ops['stale_ratio_pop']:.3f}  "
                    f"raw_stale={ops['raw_stale']:.3f}  "
                    f"avg_npb={ops['avg_nodes_per_bucket']:.3f}"
                )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_results(df, ts):

    V_vals = sorted(df["V"].unique())
    ls_map = {V_vals[0]: "-", V_vals[1]: "--"}

    # ── Fig 1: k vs stale_ratio_pop (+ heapq skip_ratio 비교) ────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for V in V_vals:
        sub = df[df["V"]==V].sort_values("k")
        ls = ls_map[V]
        ax.plot(sub["k"], sub["stale_ratio_pop"],
                color="#FF9800", ls=ls, marker="o", lw=2,
                label=f"Dial stale_ratio V={V:,}")
        ax.plot(sub["k"], sub["hq_skip_ratio"],
                color="#2196F3", ls=ls, marker="s", lw=2,
                label=f"heapq skip_ratio V={V:,}")

    ax.set_title("k vs Stale/Skip Ratio\n(Dial stale_ratio vs heapq skip_ratio)", fontsize=12)
    ax.set_xlabel("Average Degree k")
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"stale_vs_k_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: k vs avg_nodes_per_bucket ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for V in V_vals:
        sub = df[df["V"]==V].sort_values("k")
        ax.plot(sub["k"], sub["avg_nodes_per_bucket"],
                color=K_COLORS.get(V % 200, "#888"),
                ls=ls_map[V], marker="o", lw=2, label=f"V={V:,}")
    ax.set_title("k vs avg_nodes_per_bucket\n(settled 노드 / 방문 버킷 수)", fontsize=12)
    ax.set_xlabel("Average Degree k")
    ax.set_ylabel("avg nodes per occupied bucket")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"avg_npb_vs_k_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: scatter avg_nodes_per_bucket vs stale_ratio ───────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for V in V_vals:
        sub = df[df["V"]==V].sort_values("k")
        sc = ax.scatter(sub["avg_nodes_per_bucket"], sub["stale_ratio_pop"],
                        c=[K_LIST.index(k) for k in sub["k"]],
                        cmap="viridis", s=80, marker="o" if V==V_vals[0] else "^",
                        label=f"V={V:,}", zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(f"k={int(row['k'])}",
                        (row["avg_nodes_per_bucket"], row["stale_ratio_pop"]),
                        textcoords="offset points", xytext=(5,3), fontsize=7)
    # 상관관계
    slope, intercept, r, p, _ = stats.linregress(df["avg_nodes_per_bucket"], df["stale_ratio_pop"])
    x_fit = np.linspace(df["avg_nodes_per_bucket"].min(), df["avg_nodes_per_bucket"].max(), 50)
    ax.plot(x_fit, intercept+slope*x_fit, "r--", lw=1.5,
            label=f"Linear fit R²={r**2:.3f}")
    ax.set_title("avg_nodes_per_bucket vs stale_ratio_pop\n(상관관계 분석)", fontsize=12)
    ax.set_xlabel("avg_nodes_per_bucket")
    ax.set_ylabel("stale_ratio_pop")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"scatter_npb_stale_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 4: bucket_traversals / V vs k ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for V in V_vals:
        sub = df[df["V"]==V].sort_values("k")
        ax.plot(sub["k"], sub["bucket_traversals"]/V,
                ls=ls_map[V], marker="o", lw=2, label=f"V={V:,}")
    ax.set_title("bucket_traversals / V  vs  k\n(경로 거리 분포 지표)", fontsize=12)
    ax.set_xlabel("Average Degree k")
    ax.set_ylabel("bucket_traversals / V  (≈ max distance / V)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"traversals_vs_k_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 8차 실험 — Dial's stale_skip 분석")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = run_experiment()

    csv_path = RESULTS_DIR / f"experiment8_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData → {csv_path}")

    # 상관관계 요약
    slope, intercept, r, p, _ = stats.linregress(df["avg_nodes_per_bucket"], df["stale_ratio_pop"])
    print(f"\n── 상관: avg_nodes_per_bucket vs stale_ratio_pop ──")
    print(f"  slope={slope:.4f}  intercept={intercept:.4f}  R²={r**2:.4f}  p={p:.2e}")
    print("\n── stale_ratio_pop 표 (k × V) ──")
    print(df.pivot(index="V", columns="k", values="stale_ratio_pop").to_string())

    plot_results(df, ts)
    print(f"\nDone → {RESULTS_DIR.resolve()}/")
