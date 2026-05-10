"""
Dijkstra 4차 실험: k=32 근처 체계적 편향 원인 파악
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
k=20~48 세밀 측정 + heap 내 중복 항목 실시간 집계

[heap 중복 추적 방법]
  in_heap[v]: 현재 heap에 들어있는 v의 항목 수
  total_extra = Σ max(0, in_heap[v]-1): 중복(stale 예정) 항목 총합

  push v 시:  in_heap[v] >= 1 이면 total_extra += 1
  pop  v 시:  in_heap[v] >= 2 이면 total_extra -= 1

  → 모든 연산 O(1), 매 pop 시점에 total_extra를 샘플링

[잠정 가설: k ≈ sqrt(V) 근방 parallel-edge 효과]
  V=1,000 → sqrt(V) ≈ 31.6 ≈ 32
  V=10,000 → sqrt(V) ≈ 100  ← 이 구간에서 편향이 없으면 가설 지지
  랜덤 edge 샘플링 시 k ≈ sqrt(V)부터 parallel edge 충돌 확률이 비선형 증가
  → parallel edge 수도 별도 집계
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
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REPEATS = 20
SEED    = 42

V_LIST  = [1_000, 10_000]
K_FINE  = [20, 24, 28, 32, 36, 40, 44, 48]   # 4차 신규 세밀 측정

# 3차 전체 데이터 (오버레이용)
FULL_CSV = Path("results/experiment3_full_20260506_080059.csv")

V_COLORS = {1_000: "#2196F3", 10_000: "#F44336"}


# ─────────────────────────────────────────────────────────────────────
# Graph generator — parallel edge 수 집계 포함
# ─────────────────────────────────────────────────────────────────────

def make_graph(V: int, k: int, seed: int = SEED):
    """
    Returns (adj, parallel_edge_count)
    parallel_edge_count: 동일 (u,v) 쌍이 두 번 이상 연결된 횟수
    """
    rng = random.Random(seed)
    adj = {u: [] for u in range(V)}
    edge_set: dict[tuple, int] = {}   # (min,max) → 연결 횟수
    parallel = 0

    for _ in range(V * k // 2):
        u = rng.randint(0, V - 1)
        v = rng.randint(0, V - 1)
        if u != v:
            key = (min(u, v), max(u, v))
            edge_set[key] = edge_set.get(key, 0) + 1
            if edge_set[key] > 1:
                parallel += 1
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

    return adj, parallel


# ─────────────────────────────────────────────────────────────────────
# Dijkstra — heap 중복 실시간 집계 (O(1) per op)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_full_instrumented(adj: dict, source: int, V: int) -> dict:
    """
    측정 항목:
      push_count, pop_count, skip_count, skip_ratio, push_per_node
      avg_heap_extra : 각 pop 시점의 중복 항목 수 평균
      max_heap_extra : 중복 항목 수 최댓값
      avg_heap_size  : 각 pop 시점의 heap 전체 크기 평균
      max_heap_size  : heap 전체 크기 최댓값
    """
    dist       = [float("inf")] * V
    dist[source] = 0.0
    heap       = [(0.0, source)]
    in_heap    = [0] * V
    in_heap[source] = 1
    total_extra = 0          # Σ max(0, in_heap[v]-1)

    push_count  = 1
    pop_count   = 0
    skip_count  = 0

    extra_samples    = []
    heap_size_samples = []

    while heap:
        # 샘플링 (pop 직전)
        extra_samples.append(total_extra)
        heap_size_samples.append(len(heap))

        d, u = heapq.heappop(heap)
        pop_count += 1

        # in_heap[u]: n → n-1
        if in_heap[u] >= 2:
            total_extra -= 1
        in_heap[u] -= 1

        if d > dist[u]:
            skip_count += 1
            continue

        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                # in_heap[v]: m → m+1
                if in_heap[v] >= 1:
                    total_extra += 1
                in_heap[v] += 1
                heapq.heappush(heap, (nd, v))
                push_count += 1

    return {
        "push_count":     push_count,
        "pop_count":      pop_count,
        "skip_count":     skip_count,
        "skip_ratio":     skip_count / pop_count if pop_count else 0.0,
        "push_per_node":  push_count / V,
        "avg_heap_extra": float(np.mean(extra_samples))  if extra_samples else 0.0,
        "max_heap_extra": int(np.max(extra_samples))     if extra_samples else 0,
        "avg_heap_size":  float(np.mean(heap_size_samples)) if heap_size_samples else 0.0,
        "max_heap_size":  int(np.max(heap_size_samples))    if heap_size_samples else 0,
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
# Curve fit (Power law — 3차에서 확정된 모델)
# ─────────────────────────────────────────────────────────────────────

def model_power(k, c, alpha):
    return 1.0 - c / (np.asarray(k, dtype=float) ** alpha)


def fit_power(k_arr, y_arr):
    try:
        popt, _ = curve_fit(model_power, k_arr, y_arr,
                            p0=[0.884, 0.262], bounds=([0, 0], [10, 5]),
                            maxfev=10000)
        y_pred = model_power(k_arr, *popt)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return popt, r2
    except Exception:
        return None, -1.0


# ─────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────

def run_experiment() -> pd.DataFrame:
    rows = []
    total = len(V_LIST) * len(K_FINE)
    print(f"세밀 측정: V {V_LIST} × k {K_FINE} = {total}가지  (반복 {REPEATS}회)\n")

    with tqdm(total=total, desc="4차 실험") as pbar:
        for V in V_LIST:
            for k in K_FINE:
                adj, parallel = make_graph(V, k)
                E = sum(len(nb) for nb in adj.values()) // 2
                k_actual = sum(len(nb) for nb in adj.values()) / V

                # 계측 (결정론적 → 1회)
                ops = dijkstra_full_instrumented(adj, 0, V)

                # wall-clock (20회)
                times = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    dijkstra_plain(adj, 0, V)
                    times.append(time.perf_counter() - t0)

                parallel_ratio = parallel / max(E, 1)

                row = {
                    "V":               V,
                    "k_target":        k,
                    "k_actual":        round(k_actual, 2),
                    "edge_count":      E,
                    "parallel_edges":  parallel,
                    "parallel_ratio":  round(parallel_ratio, 5),
                    "skip_ratio":      round(ops["skip_ratio"],     6),
                    "push_per_node":   round(ops["push_per_node"],  4),
                    "push_count":      ops["push_count"],
                    "pop_count":       ops["pop_count"],
                    "skip_count":      ops["skip_count"],
                    "avg_heap_extra":  round(ops["avg_heap_extra"], 3),
                    "max_heap_extra":  ops["max_heap_extra"],
                    "avg_heap_size":   round(ops["avg_heap_size"],  2),
                    "max_heap_size":   ops["max_heap_size"],
                    "time_mean":       round(float(np.mean(times)), 6),
                    "time_std":        round(float(np.std(times)),  6),
                }
                rows.append(row)
                pbar.update(1)
                pbar.write(
                    f"  V={V:>6,}  k={k:>2}  "
                    f"skip={ops['skip_ratio']:.4f}  "
                    f"avg_extra={ops['avg_heap_extra']:5.1f}  "
                    f"max_extra={ops['max_heap_extra']:4d}  "
                    f"parallel={parallel_ratio:.3f}  "
                    f"time={np.mean(times)*1000:.1f}ms"
                )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_results(df4: pd.DataFrame, df_full: pd.DataFrame, timestamp: str) -> None:

    # ── Fig 1: skip_ratio — 전체 범위 오버레이 + 세밀 구간 확대 ──────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("skip_ratio: 3차 전체 + 4차 세밀 측정 오버레이", fontsize=12, fontweight="bold")

    for ax_idx, V in enumerate(V_LIST):
        ax = axes[ax_idx]

        # 3차 전체 데이터 (배경)
        if df_full is not None:
            bg = df_full[df_full["V"] == V].sort_values("k_target")
            ax.plot(bg["k_target"], bg["skip_ratio"],
                    color=V_COLORS[V], lw=1.5, alpha=0.35,
                    marker="o", ms=4, label="3차 데이터 (k=4–512)")

            # Power law fit (3차 파라미터 고정)
            k_bg = bg["k_target"].values.astype(float)
            y_bg = bg["skip_ratio"].values
            popt, r2 = fit_power(k_bg, y_bg)
            if popt is not None:
                k_sm = np.linspace(k_bg.min(), k_bg.max(), 500)
                ax.plot(k_sm, model_power(k_sm, *popt),
                        color="gray", lw=1, ls="--", alpha=0.5,
                        label=f"Power fit (R²={r2:.4f})")

        # 4차 세밀 데이터
        sub4 = df4[df4["V"] == V].sort_values("k_target")
        ax.scatter(sub4["k_target"], sub4["skip_ratio"],
                   color=V_COLORS[V], s=80, zorder=5,
                   edgecolors="black", lw=0.8,
                   label="4차 세밀 측정 (k=20–48)")
        ax.plot(sub4["k_target"], sub4["skip_ratio"],
                color=V_COLORS[V], lw=2, zorder=4)

        # k=32 편향 구간 강조
        ax.axvspan(30, 34, alpha=0.10, color="red", label="편향 관측 구간")
        ax.axvline(32, color="red", lw=1, ls=":", alpha=0.7)

        ax.set_title(f"V = {V:,}")
        ax.set_xlabel("Average Degree k")
        ax.set_ylabel("skip_ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p4_overlay_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: 잔차 (4차 데이터 vs Power law fit) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Residuals: 4차 skip_ratio − Power Law Prediction", fontsize=11)

    for ax_idx, V in enumerate(V_LIST):
        ax  = axes[ax_idx]
        sub = df4[df4["V"] == V].sort_values("k_target")
        k_arr = sub["k_target"].values.astype(float)
        y_arr = sub["skip_ratio"].values

        # 3차 파라미터로 예측값 계산 (c=0.884, alpha=0.262)
        y_pred = model_power(k_arr, 0.884, 0.262)
        residuals = y_arr - y_pred

        colors = ["#F44336" if abs(r) == max(abs(residuals)) else V_COLORS[V]
                  for r in residuals]
        bars = ax.bar(k_arr, residuals, width=2.5, color=colors, alpha=0.8)
        ax.axhline(0, color="gray", lw=1)
        ax.axvline(32, color="red", lw=1, ls=":", alpha=0.7)

        # 최대 편향 k 표시
        max_idx = np.argmax(np.abs(residuals))
        ax.annotate(f"k={int(k_arr[max_idx])}\n(Δ={residuals[max_idx]:+.4f})",
                    xy=(k_arr[max_idx], residuals[max_idx]),
                    xytext=(k_arr[max_idx]+3, residuals[max_idx]*1.2),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

        ax.set_title(f"V={V:,}  (기준: c=0.884, α=0.262)")
        ax.set_xlabel("k"); ax.set_ylabel("Residual")
        ax.set_xticks(K_FINE)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p4_residuals_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: heap 중복 항목 + parallel edge vs k ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Heap 중복 항목 및 Parallel Edge 분석 (k=20–48)", fontsize=11, fontweight="bold")

    metrics = [
        (axes[0, 0], "avg_heap_extra", "avg_heap_extra\n(중복 항목 수 평균)"),
        (axes[0, 1], "max_heap_extra", "max_heap_extra\n(중복 항목 수 최댓값)"),
        (axes[1, 0], "parallel_ratio", "parallel_ratio\n(중복 엣지 비율)"),
        (axes[1, 1], "avg_heap_size",  "avg_heap_size\n(heap 전체 크기 평균)"),
    ]
    for ax, col, title in metrics:
        for V in V_LIST:
            sub = df4[df4["V"] == V].sort_values("k_target")
            ax.plot(sub["k_target"], sub[col],
                    color=V_COLORS[V], marker="o", lw=2, label=f"V={V:,}")
        ax.axvline(32, color="red", lw=1, ls=":", alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("k"); ax.set_xticks(K_FINE)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"p4_heap_duplicate_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 4차 실험 — k=32 편향 구간 집중 분석")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df4 = run_experiment()
    csv_path = RESULTS_DIR / f"experiment4_{ts}.csv"
    df4.to_csv(csv_path, index=False)
    print(f"\n데이터 저장 → {csv_path}")

    # 3차 전체 데이터 로드
    df_full = pd.read_csv(FULL_CSV) if FULL_CSV.exists() else None
    if df_full is None:
        print(f"[경고] 3차 데이터 없음: {FULL_CSV}")

    plot_results(df4, df_full, ts)

    # 요약 출력
    print("\n── 4차 세밀 측정 요약 ──")
    print(f"{'V':>7}  {'k':>3}  {'skip':>7}  {'avg_extra':>10}  {'max_extra':>10}  "
          f"{'parallel%':>10}  {'time_ms':>8}")
    print("-" * 70)
    for _, r in df4.sort_values(["V","k_target"]).iterrows():
        print(f"{int(r.V):>7,}  {int(r.k_target):>3}  {r.skip_ratio:>7.4f}  "
              f"{r.avg_heap_extra:>10.2f}  {int(r.max_heap_extra):>10}  "
              f"{r.parallel_ratio*100:>9.2f}%  {r.time_mean*1000:>8.1f}")

    print(f"\nDone. All outputs in {RESULTS_DIR.resolve()}/")
