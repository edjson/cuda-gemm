"""Plot GEMM benchmark results from `./gemm --sweep > results.csv`.

Usage:
    ./gemm --sweep > results.csv
    python plot.py          # writes gflops_vs_size.png, pct_cublas.png
"""

import sys
import csv 
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["naive", "tiled", "register-tiled", "vectorized", "wmma", "cuBLAS", "cuBLAS-fp16"]

FP32_KERNALS = ["naive", "tiled", "register-tiled", "vectorized"]

STYLE = {
    "cuBLAS": dict(linestyle="--", color="black", marker ="s"),
    "cuBLAS-fp16": dict(linestyle="--", color="#76b900", marker="s"),
    "wmma": dict(color="#76b900", marker="^"),
}

def load(path):
    data = defaultdict(list)
    pct = defaultdict(dict)
    with open(path, newline="")as f:
        reader = csv.DictReader
        for row in csv.DictReader(f):
            k, size = row["kernel"], int(row["size"])
            data[k].append((size, float(row["gflops"])))
            pct[k][size] = float(row["pct_cublas"])
    for k in data:
        data[k].sort()
    return data, pct

def plot_gflops(data, out="gflops_vs_size.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in [k for k in ORDER if k in data]:
        sizes = [s for s, _ in data[k]]
        gf = [g for _, g in data[k]]
        style = dict(marker="o", linewidth=2)
        style.update(STYLE.get(k, {}))
        ax.plot(sizes, gf, label=k, **style)
    ax.set_xscale("log", base = 2)
    ax.set_xlabel("Matrix size (M = N =K)")
    ax.set_ylabel("Gflop/s")
    ax.set_title("SGEMM throughput vs. problem size\n(FP32 CUDA-core vs. FP16 tensor-core)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)

def plot_pct(pct, out="pct_cublas.png"):
    custom = [k for k in FP32_KERNALS if k in pct]
    if not custom:
        return
    largest = max(s for k in custom for s in pct[k])
    vals = [pct[k].get(largest, 0.0) for k in custom]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(custom, vals, color="#76b900")
    ax.axhline(100, linestyle="--", color="black", linewidth=1)
    ax.set_ylabel("% of FP32 cuBLS thoughput")
    ax.set_title(f"FP32 kernals vs. cuBLAS at {largest} x {largest} x {largest}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    data, pct = load(path)
    plot_gflops(data)
    plot_pct(pct)

if __name__ == "__main__":
    main()