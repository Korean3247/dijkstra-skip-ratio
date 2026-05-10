"""
Dijkstra 실험 B: Dial's W 민감도 — C++ 컴파일 + 실행 + 시각화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법: python3 dijkstra_experimentB_plot.py
출력:
  • W vs Dial's speedup 곡선 (k별 3개 선)
  • W* 임계값 자동 탐지 (speedup < 1.0 교차점)
"""

import subprocess
import glob
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

RESULTS_DIR = Path("resultsB")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Step 1: compile ──────────────────────────────────────────────────

print("=== Compiling dijkstra_experimentB.cpp ===")
ret = subprocess.run(
    ["g++", "-O2", "-std=c++17", "-o", "experimentB_bin",
     "dijkstra_experimentB.cpp"],
    capture_output=True, text=True
)
if ret.returncode != 0:
    print("COMPILE ERROR:", ret.stderr)
    sys.exit(1)
print("Compiled OK")

# ── Step 2: run ──────────────────────────────────────────────────────

print("\n=== Running experiment B ===")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
ret = subprocess.run(["./experimentB_bin"], capture_output=True, text=True)
print(ret.stderr)
if ret.returncode != 0:
    print("RUN ERROR:", ret.stdout)
    sys.exit(1)

# ── Step 3: load CSV ─────────────────────────────────────────────────

csv_files = sorted(glob.glob("resultsB/experimentB_*.csv"))
if not csv_files:
    print("No CSV in resultsB/"); sys.exit(1)
latest = csv_files[-1]
print(f"Loading: {latest}")
df = pd.read_csv(latest)

dial_df  = df[df["algo"] == "dial"].copy()
heapq_df = df[df["algo"] == "heapq"].copy()

print("\n── Dial's speedup table ──")
pivot = dial_df.pivot_table(
    index="W", columns="k", values="speedup", aggfunc="mean")
print(pivot.round(3))

# ── Step 4: find W* threshold ────────────────────────────────────────

print("\n── W* thresholds (speedup < 1.0) ──")
k_vals = sorted(dial_df["k"].unique())
W_star = {}

for k in k_vals:
    sub = dial_df[dial_df["k"] == k].sort_values("W")
    W_arr = sub["W"].values
    sp_arr = sub["speedup"].values
    # find crossing: speedup drops below 1.0
    cross = None
    for i in range(len(sp_arr) - 1):
        if sp_arr[i] >= 1.0 and sp_arr[i+1] < 1.0:
            # linear interpolation
            frac = (1.0 - sp_arr[i]) / (sp_arr[i+1] - sp_arr[i])
            cross = W_arr[i] + frac * (W_arr[i+1] - W_arr[i])
            break
    if cross is None and sp_arr[-1] >= 1.0:
        cross = float("inf")  # never crossed in range
    W_star[k] = cross
    print(f"  k={k}: W* = {cross}")

# ── Step 5: visualize ────────────────────────────────────────────────

colors = {8: "#E91E63", 32: "#9C27B0", 64: "#009688"}

# ─ Fig 1: W vs speedup (main figure) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in k_vals:
    sub = dial_df[dial_df["k"] == k].sort_values("W")
    ax.plot(sub["W"], sub["speedup"],
            "o-", color=colors.get(k, "#888"),
            lw=2, markersize=8, label=f"k={k}")

    # annotate W* if finite
    ws = W_star.get(k)
    if ws and ws != float("inf"):
        ax.axvline(ws, color=colors.get(k, "#888"),
                   lw=1.5, ls=":", alpha=0.7)
        ax.text(ws, 0.05, f"W*≈{int(ws)}",
                color=colors.get(k, "#888"), fontsize=9,
                rotation=90, va="bottom", ha="right")

ax.axhline(1.0, color="k", lw=1.5, ls="--", label="speedup = 1 (breakeven)")
ax.set_xscale("log")
ax.set_xlabel("Maximum edge weight W", fontsize=12)
ax.set_ylabel("Speedup (heapq time / Dial's time)", fontsize=12)
ax.set_title("Dial's vs heapq: Speedup as a Function of W\n"
             "V=10,000, k∈{8,32,64}, 10 repetitions", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out1 = RESULTS_DIR / f"W_sensitivity_{ts}.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ─ Fig 2: bucket_scans / nonempty ratio vs W ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in k_vals:
    sub = dial_df[dial_df["k"] == k].sort_values("W")
    if "bucket_scans" in sub.columns and sub["bucket_scans"].sum() > 0:
        ratio = sub["nonempty_bucket_scans"] / sub["bucket_scans"]
        ax.plot(sub["W"], ratio,
                "s--", color=colors.get(k, "#888"),
                lw=1.5, markersize=7, label=f"k={k}")

ax.set_xscale("log")
ax.set_xlabel("Maximum edge weight W", fontsize=12)
ax.set_ylabel("Nonempty bucket fraction", fontsize=12)
ax.set_title("Dial's Bucket Efficiency vs W\n"
             "Fraction of bucket-scans that find at least one vertex", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out2 = RESULTS_DIR / f"bucket_efficiency_{ts}.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ─ Fig 3: heapq time vs Dial's time (absolute) ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in k_vals:
    sub_h = heapq_df[heapq_df["k"] == k].sort_values("W")
    sub_d = dial_df[dial_df["k"] == k].sort_values("W")
    ax.plot(sub_h["W"], sub_h["time_mean_ms"],
            "o-", color=colors.get(k, "#888"), lw=2, markersize=7,
            label=f"heapq k={k}")
    ax.plot(sub_d["W"], sub_d["time_mean_ms"],
            "s--", color=colors.get(k, "#888"), lw=1.5, markersize=7,
            alpha=0.7, label=f"Dial's k={k}")

ax.set_xscale("log")
ax.set_xlabel("Maximum edge weight W", fontsize=12)
ax.set_ylabel("Mean time (ms)", fontsize=12)
ax.set_title("Absolute Runtime vs W\nheapq (solid) vs Dial's (dashed)", fontsize=12)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out3 = RESULTS_DIR / f"absolute_time_{ts}.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ── Step 6: W* summary for paper ─────────────────────────────────────
print("\n══ W* 임계값 요약 (Table IV 조건 명시용) ══════════════════════")
for k, ws in W_star.items():
    if ws == float("inf"):
        print(f"  k={k}: Dial's 항상 우세 (W={max(W_arr)}까지)")
    else:
        print(f"  k={k}: W* ≈ {int(ws)}  (이 이상이면 speedup < 1.0)")

print(f"\nAll plots → {RESULTS_DIR.resolve()}/")
