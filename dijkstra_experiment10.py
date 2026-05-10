"""
Dijkstra 10차 실험: 실제 그래프 데이터셋 검증
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
목적: 랜덤 그래프 수식 skip_ratio ≈ 1 - 0.884/k^0.262 이
      실제 그래프에서도 성립하는지 검증

데이터셋 (SNAP Stanford):
  1. ca-GrQc        — 협업 네트워크 (5K 노드, 28K 엣지)
  2. ego-Facebook   — 소셜 그래프   (4K 노드, 88K 엣지)
  3. roadNet-CA     — 도로망        (1.97M 노드, 2.76M 엣지)

측정:
  - 실제 평균 차수 k_actual
  - heapq skip_ratio 측정 (랜덤 소스 10개 평균)
  - 수식 예측값: 1 - 0.884/k^0.262
  - 오차: |실측 - 예측| / 예측

시각화:
  - 3차 power law curve 위에 실제 데이터셋 포인트 오버레이
  - 각 데이터셋 오차 막대 그래프
"""

import heapq
import gzip
import os
import random
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

RESULTS_DIR = Path("results10")
DATA_DIR    = Path("snap_data")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

SEED     = 42
N_SOURCES = 10    # 소스 10개 평균

# 3차 실험에서 확정된 수식 파라미터
C_FIT    = 0.884
ALPHA    = 0.262

# ─── power law 예측값 ────────────────────────────────────────────────

def predict_skip_ratio(k):
    return 1.0 - C_FIT / (k ** ALPHA)


# ─────────────────────────────────────────────────────────────────────
# 데이터셋 정보
# ─────────────────────────────────────────────────────────────────────

DATASETS = {
    "ca-GrQc": {
        "url":  "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "file": DATA_DIR / "ca-GrQc.txt.gz",
        "directed": False,
        "skip_header": 4,   # 주석 줄 수
        "desc": "협업 네트워크 (Arxiv GrQc)",
    },
    "ego-Facebook": {
        "url":  "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "file": DATA_DIR / "ego-Facebook.txt.gz",
        "directed": False,
        "skip_header": 0,
        "desc": "소셜 네트워크 (Facebook ego)",
    },
    "roadNet-CA": {
        "url":  "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
        "file": DATA_DIR / "roadNet-CA.txt.gz",
        "directed": False,
        "skip_header": 4,
        "desc": "도로망 (California)",
    },
}


# ─────────────────────────────────────────────────────────────────────
# 데이터 다운로드
# ─────────────────────────────────────────────────────────────────────

def download_dataset(name, info):
    fpath = info["file"]
    if fpath.exists():
        print(f"  {name}: 이미 존재 ({fpath.stat().st_size//1024:,} KB)")
        return True
    print(f"  {name}: 다운로드 중 ... {info['url']}")
    try:
        urllib.request.urlretrieve(info["url"], fpath)
        print(f"  {name}: 완료 ({fpath.stat().st_size//1024:,} KB)")
        return True
    except Exception as e:
        print(f"  {name}: 실패 — {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# 그래프 로드 (무방향, 무가중치 → 가중치=1 균등)
# ─────────────────────────────────────────────────────────────────────

def load_snap_graph(name, info, weight_seed=42):
    """
    SNAP edge-list 형식 파일을 로드해 인접 리스트 반환.
    가중치: uniform int [1, 100] (랜덤)
    """
    print(f"  {name}: 로드 중 ...")
    rng = random.Random(weight_seed)

    edges = []
    node_set = set()

    with gzip.open(info["file"], "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            node_set.add(u); node_set.add(v)
            edges.append((u, v))

    # node re-mapping to 0-indexed
    nodes    = sorted(node_set)
    node_map = {n: i for i, n in enumerate(nodes)}
    V        = len(nodes)

    adj = [[] for _ in range(V)]
    seen = set()
    for u_raw, v_raw in edges:
        u = node_map[u_raw]; v = node_map[v_raw]
        key = (min(u,v), max(u,v))
        if key in seen:
            continue
        seen.add(key)
        w = rng.randint(1, 100)
        adj[u].append((v, w))
        adj[v].append((u, w))

    E = len(seen)
    k_actual = 2 * E / V if V > 0 else 0
    print(f"  {name}: V={V:,}  E={E:,}  k_actual={k_actual:.2f}")
    return adj, V, E, k_actual


# ─────────────────────────────────────────────────────────────────────
# Dijkstra (계측)
# ─────────────────────────────────────────────────────────────────────

def dijkstra_counted(adj, source, V):
    dist = [float("inf")] * V
    dist[source] = 0
    heap = [(0, source)]
    push_c = pop_c = skip_c = 0
    push_c = 1
    while heap:
        d, u = heapq.heappop(heap); pop_c += 1
        if d > dist[u]: skip_c += 1; continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(heap, (nd, v)); push_c += 1
    skip_ratio = skip_c / pop_c if pop_c else 0.0
    return skip_ratio, push_c, pop_c, skip_c


# ─────────────────────────────────────────────────────────────────────
# 실험 실행
# ─────────────────────────────────────────────────────────────────────

def run_experiment():
    rows = []
    rng_src = random.Random(SEED)

    for name, info in DATASETS.items():
        print(f"\n── {name} ({info['desc']}) ──")
        if not download_dataset(name, info):
            print(f"  SKIP: {name}")
            continue
        try:
            adj, V, E, k_actual = load_snap_graph(name, info)
        except Exception as e:
            print(f"  로드 실패: {e}"); continue

        k_pred = predict_skip_ratio(k_actual)

        # 랜덤 소스 N_SOURCES 개 선택
        sources = rng_src.sample(range(V), min(N_SOURCES, V))

        skip_ratios = []
        t_total = 0.0
        for src in tqdm(sources, desc=f"  {name} sources"):
            t0 = time.perf_counter()
            sr, push_c, pop_c, skip_c = dijkstra_counted(adj, src, V)
            t_total += time.perf_counter() - t0
            skip_ratios.append(sr)

        sr_mean = float(np.mean(skip_ratios))
        sr_std  = float(np.std(skip_ratios))
        rel_err = abs(sr_mean - k_pred) / k_pred if k_pred > 0 else 0.0

        row = {
            "dataset":       name,
            "desc":          info["desc"],
            "V":             V,
            "E":             E,
            "k_actual":      round(k_actual, 4),
            "skip_ratio_mean": round(sr_mean, 6),
            "skip_ratio_std":  round(sr_std,  6),
            "predicted":     round(k_pred, 6),
            "rel_err":       round(rel_err, 6),
            "time_per_src_ms": round(t_total / len(sources) * 1000, 2),
        }
        rows.append(row)
        print(f"  skip_ratio = {sr_mean:.4f} ± {sr_std:.4f}")
        print(f"  예측값    = {k_pred:.4f}")
        print(f"  상대 오차  = {rel_err*100:.2f}%")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_results(df, ts):

    if df.empty:
        print("데이터 없음 — 시각화 생략"); return

    # ── Fig 1: power law curve + real dataset overlay ─────────────────
    k_range = np.linspace(2, max(df["k_actual"].max()*1.2, 200), 300)
    sk_pred  = [predict_skip_ratio(k) for k in k_range]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, sk_pred, "k-", lw=2,
            label=f"수식: 1 - {C_FIT}/k^{ALPHA}  (3차 fit)")

    colors = {"ca-GrQc": "#E91E63", "ego-Facebook": "#9C27B0", "roadNet-CA": "#009688"}
    for _, row in df.iterrows():
        name = row["dataset"]
        ax.errorbar(row["k_actual"], row["skip_ratio_mean"],
                    yerr=row["skip_ratio_std"],
                    fmt="o", markersize=10, capsize=5,
                    color=colors.get(name, "#333"),
                    label=f"{name}\n  k={row['k_actual']:.1f}  "
                          f"실측={row['skip_ratio_mean']:.3f}  "
                          f"오차={row['rel_err']*100:.1f}%",
                    zorder=5)
        ax.annotate(name,
                    xy=(row["k_actual"], row["skip_ratio_mean"]),
                    xytext=(15, -10), textcoords="offset points",
                    fontsize=9, color=colors.get(name, "#333"))

    ax.set_title("skip_ratio 수식 vs 실제 데이터셋\n3차 power law fit 검증", fontsize=13)
    ax.set_xlabel("Average Degree k (실제 그래프)")
    ax.set_ylabel("skip_ratio")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"formula_validation_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: formula_validation")

    # ── Fig 2: 오차 막대 그래프 ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    names = df["dataset"].tolist()
    x     = np.arange(len(names))
    rel_errs = df["rel_err"].values * 100

    bars = ax.bar(x, rel_errs, color=[colors.get(n, "#888") for n in names])
    for bar, err in zip(bars, rel_errs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{err:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("수식 예측 상대 오차 |실측 - 예측| / 예측\n낮을수록 수식이 실제 그래프에 잘 적합", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"rel_error_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rel_error")

    # ── Fig 3: 실측 vs 예측 산점도 ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    lo = min(df["predicted"].min(), df["skip_ratio_mean"].min()) - 0.05
    hi = max(df["predicted"].max(), df["skip_ratio_mean"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="완벽 예측선")
    for _, row in df.iterrows():
        name = row["dataset"]
        ax.scatter(row["predicted"], row["skip_ratio_mean"],
                   s=120, color=colors.get(name,"#888"),
                   zorder=5, label=name)
        ax.annotate(name,
                    xy=(row["predicted"], row["skip_ratio_mean"]),
                    xytext=(8, 0), textcoords="offset points", fontsize=9)
    ax.set_xlabel("수식 예측 skip_ratio")
    ax.set_ylabel("실측 skip_ratio")
    ax.set_title("실측 vs 예측 산점도")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"scatter_pred_vs_actual_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: scatter")

    print(f"\nAll plots → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Dijkstra 10차 실험 — 실제 그래프 수식 검증")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"수식: skip_ratio = 1 - {C_FIT}/k^{ALPHA}\n")

    print("── 데이터셋 다운로드 ──")
    for name, info in DATASETS.items():
        download_dataset(name, info)

    df = run_experiment()

    if not df.empty:
        csv_path = RESULTS_DIR / f"experiment10_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nData → {csv_path}")

        print("\n══ 결과 요약 ══════════════════════════════════════")
        print(df[["dataset","V","E","k_actual",
                   "skip_ratio_mean","predicted","rel_err",
                   "time_per_src_ms"]].to_string(index=False))

    plot_results(df, ts)
    print(f"\nDone → {RESULTS_DIR.resolve()}/")
