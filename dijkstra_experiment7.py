"""
Dijkstra 7차 실험: C++ 세 자료구조 비교 + Python 6차 대조
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. dijkstra_experiment7.cpp 를 g++ -O2 로 컴파일
2. C++ binary 실행 → results7/experiment7_CPP_*.csv
3. Python 6차 결과(results6/experiment6_*.csv) 로드
4. 비교 시각화 4종
"""

import subprocess
import os
import sys
import glob
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS7 = Path("results7")
RESULTS7.mkdir(exist_ok=True)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

DS_COLORS = {"heapq": "#2196F3", "fibonacci": "#4CAF50", "dial": "#FF9800"}
LANG_COLORS = {"Python": "#E91E63", "C++": "#009688"}

# ─────────────────────────────────────────────────────────────────────
# 1. Compile & Run C++
# ─────────────────────────────────────────────────────────────────────

def compile_and_run():
    print("── C++ 컴파일 중 ──")
    ret = subprocess.run(
        ["g++", "-O2", "-std=c++17", "-o", "dijkstra7_bin",
         "dijkstra_experiment7.cpp", "-lm"],
        capture_output=True, text=True
    )
    if ret.returncode != 0:
        print("컴파일 실패:\n", ret.stderr)
        sys.exit(1)
    print("컴파일 완료. 실행 중 ...")
    ret2 = subprocess.run(["./dijkstra7_bin"], capture_output=True, text=True)
    print(ret2.stderr)
    if ret2.returncode != 0:
        print("실행 실패:", ret2.stderr)
        sys.exit(1)
    return sorted(RESULTS7.glob("experiment7_*.csv"))[-1]


# ─────────────────────────────────────────────────────────────────────
# 2. Load Data
# ─────────────────────────────────────────────────────────────────────

def load_cpp(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["correct"] == True].copy()
    return df


def load_python6():
    files = sorted(Path("results6").glob("experiment6_*.csv"))
    if not files:
        print("⚠ results6/experiment6_*.csv 없음 — Python 비교 생략")
        return None
    df = pd.read_csv(files[-1])
    return df


# ─────────────────────────────────────────────────────────────────────
# 3. Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_all(df_cpp, df_py, ts):

    k_list  = [8, 32, 64]
    titles  = ["sparse (k=8)", "dense-mid (k=32)", "dense-high (k=64)"]

    # ── Fig 1: C++ time comparison (bars, V per group) ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, k_val, title in zip(axes, k_list, titles):
        sub  = df_cpp[df_cpp["k"] == k_val].sort_values("V")
        V_vals = sorted(sub["V"].unique())
        algos  = sorted(sub["algo"].unique())
        x = np.arange(len(V_vals))
        w = 0.25
        offs = np.linspace(-(len(algos)-1)/2, (len(algos)-1)/2, len(algos)) * w
        for off, algo in zip(offs, algos):
            t_vals = [
                sub[(sub["V"]==V) & (sub["algo"]==algo)]["time_mean_ms"].values[0]
                if len(sub[(sub["V"]==V) & (sub["algo"]==algo)]) else 0
                for V in V_vals
            ]
            ax.bar(x+off, t_vals, width=w*0.9,
                   label=algo, color=DS_COLORS.get(algo, "#888"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:,}" for v in V_vals])
        ax.set_xlabel("V"); ax.set_ylabel("Time (ms)"); ax.set_title(f"C++ — {title}")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("C++ Wall-clock Time by Algorithm", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS7 / f"cpp_time_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cpp_time")

    # ── Fig 2: C++ speedup vs heapq ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"fibonacci": "^", "dial": "s"}
    for k_val in k_list:
        sub_k  = df_cpp[df_cpp["k"] == k_val]
        sub_hq = sub_k[sub_k["algo"]=="heapq"].set_index("V")
        for algo in ["dial", "fibonacci"]:
            sub_a = sub_k[sub_k["algo"]==algo].sort_values("V")
            if sub_a.empty: continue
            sp = [sub_hq.loc[V,"time_mean_ms"] / sub_a[sub_a["V"]==V]["time_mean_ms"].values[0]
                  if V in sub_hq.index else np.nan
                  for V in sub_a["V"]]
            ax.plot(sub_a["V"], sp,
                    marker=markers.get(algo,"o"), lw=2,
                    color=DS_COLORS.get(algo,"#888"),
                    label=f"{algo} k={k_val}", linestyle="--" if k_val>8 else "-")
    ax.axhline(1.0, color="gray", lw=1, ls=":", label="heapq baseline")
    ax.set_xscale("log")
    ax.set_title("C++ Speedup vs heapq baseline")
    ax.set_xlabel("V"); ax.set_ylabel("Speedup")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS7 / f"cpp_speedup_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cpp_speedup")

    if df_py is None:
        print("Python 비교 스킵 (results6 없음)")
        return

    # ── Fig 3: Python vs C++ — heapq ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, k_val, title in zip(axes, k_list, titles):
        sub_cpp = df_cpp[(df_cpp["k"]==k_val) & (df_cpp["algo"]=="heapq")].sort_values("V")
        sub_py  = df_py [(df_py ["k"]==k_val) & (df_py ["algo"]=="heapq")].sort_values("V")
        common_V = sorted(set(sub_cpp["V"]) & set(sub_py["V"]))
        if not common_V: continue
        t_cpp = [sub_cpp[sub_cpp["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        t_py  = [sub_py [sub_py ["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        x = np.arange(len(common_V))
        ax.bar(x-0.2, t_cpp, 0.35, label="C++", color=LANG_COLORS["C++"])
        ax.bar(x+0.2, t_py,  0.35, label="Python", color=LANG_COLORS["Python"])
        ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in common_V])
        ax.set_title(f"heapq — {title}"); ax.set_xlabel("V"); ax.set_ylabel("ms")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("heapq: Python vs C++", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS7 / f"py_vs_cpp_heapq_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: py_vs_cpp_heapq")

    # ── Fig 4: Python vs C++ — fibonacci ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, k_val, title in zip(axes, k_list, titles):
        sub_cpp = df_cpp[(df_cpp["k"]==k_val) & (df_cpp["algo"]=="fibonacci")].sort_values("V")
        sub_py  = df_py [(df_py ["k"]==k_val) & (df_py ["algo"]=="fibonacci")].sort_values("V")
        common_V = sorted(set(sub_cpp["V"]) & set(sub_py["V"]))
        if not common_V: ax.set_title(f"fibonacci — {title}\n(데이터 없음)"); continue
        t_cpp = [sub_cpp[sub_cpp["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        t_py  = [sub_py [sub_py ["V"]==V]["time_mean_ms"].values[0] for V in common_V]
        speedup_lang = [p/c for p,c in zip(t_py, t_cpp)]
        x = np.arange(len(common_V))
        ax.bar(x-0.2, t_cpp, 0.35, label="C++",    color=LANG_COLORS["C++"])
        ax.bar(x+0.2, t_py,  0.35, label="Python",  color=LANG_COLORS["Python"])
        for xi, sp in zip(x, speedup_lang):
            ax.text(xi+0.2, t_py[xi-x[0]], f"{sp:.1f}×", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in common_V])
        ax.set_title(f"fibonacci — {title}"); ax.set_xlabel("V"); ax.set_ylabel("ms")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Fibonacci heap: Python vs C++ (숫자=Python/C++ ratio)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS7 / f"py_vs_cpp_fibonacci_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: py_vs_cpp_fibonacci")

    # ── Fig 5: C++ Fibonacci vs C++ heapq ratio ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for k_val in k_list:
        sub_k  = df_cpp[df_cpp["k"]==k_val]
        sub_hq = sub_k[sub_k["algo"]=="heapq"].sort_values("V")
        sub_fi = sub_k[sub_k["algo"]=="fibonacci"].sort_values("V")
        common_V = sorted(set(sub_hq["V"]) & set(sub_fi["V"]))
        if not common_V: continue
        ratio = [sub_fi[sub_fi["V"]==V]["time_mean_ms"].values[0] /
                 sub_hq[sub_hq["V"]==V]["time_mean_ms"].values[0]
                 for V in common_V]
        ax.plot(common_V, ratio, marker="o", lw=2, label=f"k={k_val}")
    ax.axhline(1.0, ls="--", color="gray", lw=1, label="heapq parity")
    ax.set_xscale("log")
    ax.set_title("C++: Fibonacci / heapq time ratio\n(< 1 → Fibonacci faster)")
    ax.set_xlabel("V"); ax.set_ylabel("Fibonacci_time / heapq_time")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS7 / f"cpp_fib_vs_heapq_ratio_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cpp_fib_vs_heapq_ratio")
    print(f"\nAll plots → {RESULTS7}/")


# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────

def print_summary(df_cpp, df_py):
    print("\n══ C++ Results ════════════════════════════════════")
    print(f"{'V':>8} {'k':>4} {'algo':>10} {'mean_ms':>9} {'std_ms':>8}")
    for _, r in df_cpp.sort_values(["k","V","algo"]).iterrows():
        print(f"{r['V']:>8,} {r['k']:>4} {r['algo']:>10} {r['time_mean_ms']:>9.3f} {r['time_std_ms']:>8.3f}")

    if df_py is not None:
        print("\n══ Python/C++ speedup ratio (heapq) ═══════════")
        for k_val in [8,32,64]:
            sc = df_cpp[(df_cpp["k"]==k_val)&(df_cpp["algo"]=="heapq")].set_index("V")
            sp = df_py [(df_py ["k"]==k_val)&(df_py ["algo"]=="heapq")].set_index("V")
            for V in sorted(set(sc.index)&set(sp.index)):
                ratio = sp.loc[V,"time_mean_ms"] / sc.loc[V,"time_mean_ms"]
                print(f"  k={k_val} V={V:>7,}  Python/C++ heapq = {ratio:.1f}×")

        print("\n══ Python/C++ speedup ratio (fibonacci) ════════")
        for k_val in [8,32,64]:
            sc = df_cpp[(df_cpp["k"]==k_val)&(df_cpp["algo"]=="fibonacci")].set_index("V")
            sp = df_py [(df_py ["k"]==k_val)&(df_py ["algo"]=="fibonacci")].set_index("V")
            for V in sorted(set(sc.index)&set(sp.index)):
                ratio = sp.loc[V,"time_mean_ms"] / sc.loc[V,"time_mean_ms"]
                print(f"  k={k_val} V={V:>7,}  Python/C++ fibonacci = {ratio:.1f}×")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Dijkstra 7차 실험 — C++ 비교")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    cpp_csv = compile_and_run()
    print(f"C++ CSV: {cpp_csv}")

    df_cpp = load_cpp(cpp_csv)
    df_py  = load_python6()

    print_summary(df_cpp, df_py)
    plot_all(df_cpp, df_py, TS)

    print(f"\nDone. All outputs → {RESULTS7.resolve()}/")
