"""
Dijkstra 실험 C: Barabási–Albert scale-free 그래프 검증
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
목적: 수식 skip_ratio(k) ≈ 1 - 0.884/k^0.262가
      ER(Erdős–Rényi) 그래프뿐만 아니라 BA(Barabási–Albert)
      scale-free 그래프에서도 성립하는지 검증

설계:
  - BA 그래프: barabasi_albert_graph(n, m)  →  k_avg ≈ 2m
  - m ∈ {4, 8, 16, 32}  →  k_avg ∈ {8, 16, 32, 64}
  - V ∈ {1,000, 10,000}
  - 가중치: uniform int [1, 100]
  - 반복: 10회 평균 (rep마다 다른 그래프 seed)

시각화:
  - 예측 곡선 + ER 포인트 + BA 포인트 오버레이
  - 상대 오차 비교 막대 그래프

패키지 설치 (실행 전):
  pip install networkx tqdm
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
from tqdm import tqdm

try:
    import networkx as nx
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "networkx", "-q"])
    import networkx as nx

RESULTS_DIR = Path("resultsC")
RESULTS_DIR.mkdir(exist_ok=True)

SEED    = 42
N_REPS  = 10
W_MAX   = 100
C_FIT   = 0.884
ALPHA   = 0.262


# ── Helpers ─────────────────────────────────────────────────────────

def predict_skip_ratio(k):
    return 1.0 - C_FIT / (k ** ALPHA)


def dijkstra_counted(adj, source, V):
    """Identical to experiment 10 dijkstra_counted."""
    dist = [float("inf")] * V
    dist[source] = 0
    heap = [(0, source)]
    push_c = 1
    pop_c  = 0
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
    skip_ratio = skip_c / pop_c if pop_c else 0.0
    return skip_ratio, push_c, pop_c, skip_c


def nx_to_weighted_adj(G, V, rng):
    """NetworkX Graph → weighted adjacency list with random int weights."""
    adj = [[] for _ in range(V)]
    for u, v in G.edges():
        w = rng.randint(1, W_MAX)
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def make_er_adj(V, k_target, rng):
    """Generate Erdős–Rényi random graph adjacency list."""
    adj = [[] for _ in range(V)]
    n_edges = V * k_target // 2
    for _ in range(n_edges):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            w = rng.randint(1, W_MAX)
            adj[u].append((v, w))
            adj[v].append((u, w))
    # spanning path for connectivity
    nodes = list(range(V))
    rng.shuffle(nodes)
    for i in range(V - 1):
        u, v = nodes[i], nodes[i + 1]
        w = rng.randint(1, W_MAX)
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# ── Experiment ───────────────────────────────────────────────────────

def run_experiment():
    rows = []
    m_vals     = [4, 8, 16, 32]          # BA parameter m → k_avg = 2m
    k_er_vals  = [8, 16, 32, 64]         # ER target k (matching BA k_avg)
    V_vals     = [1000, 10000]

    for V in V_vals:
        print(f"\n{'='*60}")
        print(f"  V = {V:,}")
        print(f"{'='*60}")

        # ── BA graphs ─────────────────────────────────────────────
        for m in m_vals:
            k_label = 2 * m
            desc = f"BA V={V} m={m} (k_avg≈{k_label})"
            print(f"\n── {desc} ──")

            skip_ratios = []
            k_actuals   = []

            for rep in tqdm(range(N_REPS), desc=f"  {desc}"):
                G = nx.barabasi_albert_graph(V, m, seed=SEED + rep)
                k_actual = 2 * G.number_of_edges() / V
                k_actuals.append(k_actual)

                rng_wt = random.Random(SEED + rep * 7919)
                adj = nx_to_weighted_adj(G, V, rng_wt)

                src = random.Random(SEED + rep).randint(0, V - 1)
                sr, push_c, pop_c, skip_c = dijkstra_counted(adj, src, V)
                skip_ratios.append(sr)

            k_actual_mean = float(np.mean(k_actuals))
            sr_mean = float(np.mean(skip_ratios))
            sr_std  = float(np.std(skip_ratios))
            k_pred  = predict_skip_ratio(k_actual_mean)
            rel_err = abs(sr_mean - k_pred) / k_pred if k_pred > 0 else 0.0

            print(f"  k_actual={k_actual_mean:.2f}  "
                  f"skip_ratio={sr_mean:.4f}±{sr_std:.4f}  "
                  f"predicted={k_pred:.4f}  rel_err={rel_err*100:.1f}%")

            rows.append({
                "graph_type": "BA",
                "V": V,
                "param": m,
                "k_avg": round(k_actual_mean, 4),
                "skip_ratio_mean": round(sr_mean, 6),
                "skip_ratio_std":  round(sr_std, 6),
                "predicted": round(k_pred, 6),
                "rel_err":   round(rel_err, 6),
            })

        # ── ER graphs (same k values for overlay comparison) ──────
        for k in k_er_vals:
            desc = f"ER V={V} k={k}"
            print(f"\n── {desc} ──")

            skip_ratios = []
            k_actuals   = []

            for rep in tqdm(range(N_REPS), desc=f"  {desc}"):
                rng = random.Random(SEED + rep * 3571)
                adj = make_er_adj(V, k, rng)
                k_actual = sum(len(a) for a in adj) / V
                k_actuals.append(k_actual)

                src = rng.randint(0, V - 1)
                sr, push_c, pop_c, skip_c = dijkstra_counted(adj, src, V)
                skip_ratios.append(sr)

            k_actual_mean = float(np.mean(k_actuals))
            sr_mean = float(np.mean(skip_ratios))
            sr_std  = float(np.std(skip_ratios))
            k_pred  = predict_skip_ratio(k_actual_mean)
            rel_err = abs(sr_mean - k_pred) / k_pred if k_pred > 0 else 0.0

            print(f"  k_actual={k_actual_mean:.2f}  "
                  f"skip_ratio={sr_mean:.4f}±{sr_std:.4f}  "
                  f"predicted={k_pred:.4f}  rel_err={rel_err*100:.1f}%")

            rows.append({
                "graph_type": "ER",
                "V": V,
                "param": k,
                "k_avg": round(k_actual_mean, 4),
                "skip_ratio_mean": round(sr_mean, 6),
                "skip_ratio_std":  round(sr_std, 6),
                "predicted": round(k_pred, 6),
                "rel_err":   round(rel_err, 6),
            })

    return pd.DataFrame(rows)


# ── Visualization ────────────────────────────────────────────────────

def plot_results(df, ts):
    if df.empty:
        print("No data — skip plot"); return

    k_range = np.linspace(4, 80, 300)
    pred_curve = [predict_skip_ratio(k) for k in k_range]

    V_vals = sorted(df["V"].unique())

    # ─ Fig 1: formula curve + ER + BA overlay (one panel per V) ─────
    fig, axes = plt.subplots(1, len(V_vals), figsize=(14, 6), sharey=True)
    if len(V_vals) == 1:
        axes = [axes]

    for ax, V in zip(axes, V_vals):
        ax.plot(k_range, pred_curve, "k-", lw=2,
                label=f"수식: 1−{C_FIT}/k^{ALPHA}")

        sub = df[df["V"] == V]

        er  = sub[sub["graph_type"] == "ER"]
        ba  = sub[sub["graph_type"] == "BA"]

        ax.errorbar(er["k_avg"], er["skip_ratio_mean"],
                    yerr=er["skip_ratio_std"],
                    fmt="o", markersize=9, capsize=4,
                    color="#1976D2", label="ER (Erdős–Rényi)", zorder=5)

        ax.errorbar(ba["k_avg"], ba["skip_ratio_mean"],
                    yerr=ba["skip_ratio_std"],
                    fmt="^", markersize=9, capsize=4,
                    color="#E53935", label="BA (Barabási–Albert)", zorder=5)

        ax.set_title(f"V = {V:,}", fontsize=12)
        ax.set_xlabel("Average degree k", fontsize=11)
        ax.set_ylabel("skip_ratio", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle("skip_ratio Formula: ER vs BA Scale-Free Graphs\n"
                 "공식 skip_ratio(k) ≈ 1 − 0.884/k^0.262", fontsize=13)
    plt.tight_layout()
    out1 = RESULTS_DIR / f"ER_vs_BA_overlay_{ts}.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # ─ Fig 2: relative error bar chart ER vs BA ──────────────────────
    fig, axes = plt.subplots(1, len(V_vals), figsize=(14, 5))
    if len(V_vals) == 1:
        axes = [axes]

    for ax, V in zip(axes, V_vals):
        sub = df[df["V"] == V]
        er  = sub[sub["graph_type"] == "ER"].sort_values("k_avg")
        ba  = sub[sub["graph_type"] == "BA"].sort_values("k_avg")

        x = np.arange(len(er))
        width = 0.35

        bar1 = ax.bar(x - width/2, er["rel_err"].values * 100,
                      width, label="ER", color="#1976D2", alpha=0.8)
        bar2 = ax.bar(x + width/2, ba["rel_err"].values * 100,
                      width, label="BA", color="#E53935", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"k≈{int(k)}" for k in er["k_avg"].values],
                            fontsize=9)
        ax.set_ylabel("Relative Error (%)")
        ax.set_title(f"V = {V:,}")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("수식 예측 상대 오차: ER vs BA\n"
                 "|실측 − 예측| / 예측", fontsize=13)
    plt.tight_layout()
    out2 = RESULTS_DIR / f"rel_err_ER_vs_BA_{ts}.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    print(f"\nAll plots → {RESULTS_DIR.resolve()}/")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 실험 C — BA scale-free 그래프 검증")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"수식: skip_ratio = 1 − {C_FIT}/k^{ALPHA}\n")

    df = run_experiment()

    if not df.empty:
        csv_path = RESULTS_DIR / f"experimentC_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nData → {csv_path}")

        print("\n══ 결과 요약 ══════════════════════════════════════════")
        print(df[["graph_type","V","k_avg",
                   "skip_ratio_mean","predicted","rel_err"]].to_string(index=False))

    plot_results(df, ts)
    print(f"\nDone → {RESULTS_DIR.resolve()}/")
