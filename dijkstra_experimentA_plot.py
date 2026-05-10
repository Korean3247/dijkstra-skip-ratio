"""
Dijkstra 실험 A: V 범위 확장 — C++ 컴파일 + 실행 + 시각화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법: python3 dijkstra_experimentA_plot.py
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

RESULTS_DIR = Path("resultsA")
RESULTS_DIR.mkdir(exist_ok=True)

C_FIT = 0.884
ALPHA = 0.262


def predict_skip_ratio(k):
    return 1.0 - C_FIT / (k ** ALPHA)


# ── Step 1: compile ──────────────────────────────────────────────────

print("=== Compiling dijkstra_experimentA.cpp ===")
ret = subprocess.run(
    ["g++", "-O2", "-std=c++17", "-o", "experimentA_bin",
     "dijkstra_experimentA.cpp"],
    capture_output=True, text=True
)
if ret.returncode != 0:
    print("COMPILE ERROR:", ret.stderr)
    sys.exit(1)
print("Compiled OK")

# ── Step 2: run ──────────────────────────────────────────────────────

print("\n=== Running experiment A ===")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
ret = subprocess.run(["./experimentA_bin"], capture_output=True, text=True)
print(ret.stderr)
if ret.returncode != 0:
    print("RUN ERROR:", ret.stdout)
    sys.exit(1)

# ── Step 3: load CSV ─────────────────────────────────────────────────

csv_files = sorted(glob.glob("resultsA/experimentA_*.csv"))
if not csv_files:
    print("No CSV found in resultsA/"); sys.exit(1)
latest = csv_files[-1]
print(f"Loading: {latest}")
df = pd.read_csv(latest)
print(df[["V","k","skip_ratio","push_per_V","time_mean_ms"]].to_string(index=False))

# Also load existing small-V data from experiments 2-3 if available
small_v_files = sorted(glob.glob("results2/experiment2_*.csv") +
                        glob.glob("results3/experiment3_*.csv"))
df_small = None
if small_v_files:
    frames = [pd.read_csv(f) for f in small_v_files]
    df_all_raw = pd.concat(frames, ignore_index=True)
    # Keep only heapq rows with relevant k values if column exists
    if "algo" in df_all_raw.columns:
        df_small = df_all_raw[df_all_raw["algo"] == "heapq"].copy()
    elif "skip_ratio" in df_all_raw.columns:
        df_small = df_all_raw.copy()
    print(f"Loaded small-V data from {len(small_v_files)} file(s)")

# ── Step 4: visualize ────────────────────────────────────────────────

colors = {8: "#E91E63", 32: "#9C27B0", 64: "#009688"}
k_vals = sorted(df["k"].unique())

# ─ Fig 1: skip_ratio vs V (log-log) for each k ──────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for k in k_vals:
    subset = df[df["k"] == k].sort_values("V")
    ax.plot(subset["V"], subset["skip_ratio"],
            "o-", color=colors.get(k, "#888"),
            label=f"k={k} (large V)", lw=2, markersize=8)
    # formula horizontal line
    pred = predict_skip_ratio(k)
    ax.axhline(pred, color=colors.get(k, "#888"),
               lw=1, ls="--", alpha=0.5,
               label=f"k={k} 수식 예측 {pred:.3f}")

ax.set_xscale("log")
ax.set_xlabel("Graph size V", fontsize=12)
ax.set_ylabel("skip_ratio", fontsize=12)
ax.set_title("skip_ratio vs V (Large-Scale Extension)\n"
             "V-independence validation: V=200K ~ 1M", fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
out1 = RESULTS_DIR / f"skip_vs_V_large_{ts}.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ─ Fig 2: push_count/V vs k (bar chart) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (metric, ylabel, title) in enumerate([
    ("skip_ratio",   "skip_ratio",    "skip_ratio by k × V"),
    ("push_per_V",   "push_count / V", "push_count/V by k × V"),
]):
    ax = axes[i]
    V_vals = sorted(df["V"].unique())
    V_labels = [f"{int(v)//1000}K" for v in V_vals]
    x = np.arange(len(k_vals))
    width = 0.25

    for j, V in enumerate(V_vals):
        vals = [df[(df["k"]==k) & (df["V"]==V)][metric].values[0]
                for k in k_vals if len(df[(df["k"]==k) & (df["V"]==V)]) > 0]
        ax.bar(x[:len(vals)] + j*width, vals, width,
               label=f"V={V_labels[j]}",
               color=plt.cm.viridis(j / max(len(V_vals)-1, 1)))

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

plt.suptitle("V-independence Verification (Large V)", fontsize=13, y=1.01)
plt.tight_layout()
out2 = RESULTS_DIR / f"V_independence_large_{ts}.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ─ Fig 3: formula overlay (power-law curve + all V sizes) ────────────
fig, axes = plt.subplots(1, len(k_vals), figsize=(15, 5))

k_range_fine = np.linspace(4, 80, 200)

for ax, k in zip(axes, k_vals):
    # formula curve (horizontal)
    pred_val = predict_skip_ratio(k)
    subset = df[df["k"] == k].sort_values("V")

    ax.axhline(pred_val, color="k", lw=2, ls="--",
               label=f"수식 예측: {pred_val:.4f}")
    ax.plot(subset["V"], subset["skip_ratio"],
            "o", color=colors.get(k, "#888"), markersize=10,
            label="실측 (large V)")

    ax.set_xlabel("V")
    ax.set_ylabel("skip_ratio")
    ax.set_title(f"k = {k}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

plt.suptitle("skip_ratio V-independence: Large-V Confirmation", fontsize=13)
plt.tight_layout()
out3 = RESULTS_DIR / f"formula_overlay_large_{ts}.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ── Step 5: print summary table ──────────────────────────────────────
print("\n══ 결과 요약 ══════════════════════════════════════════════════")
summary = df[["V","k","E","skip_ratio","push_per_V","pop_per_V","time_mean_ms"]].copy()
summary["predicted"]  = summary["k"].apply(predict_skip_ratio).round(4)
summary["rel_err_%"]  = ((summary["skip_ratio"] - summary["predicted"]).abs()
                          / summary["predicted"] * 100).round(2)
print(summary.to_string(index=False))
print(f"\nAll plots → {RESULTS_DIR.resolve()}/")
