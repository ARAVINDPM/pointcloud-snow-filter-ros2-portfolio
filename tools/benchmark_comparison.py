"""
benchmark_comparison.py - Cross-implementation runtime comparison.

Benchmarks all 4 filters on SICK-equivalent and Livox scans,
then produces a benchmark comparison figure.

Usage:
    PYTHONPATH=src python tools/benchmark_comparison.py
    PYTHONPATH=src python tools/benchmark_comparison.py --n-runs 50 --output results/my_benchmark.png

Requires local private sensor data via THESIS_DATA_ROOT. The data is not
included in this public export.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lidar_snow_filter.filters import LiDARFilters

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

THESIS_DATA = Path(os.environ.get("THESIS_DATA_ROOT", REPO_ROOT / "data" / "private_thesis" / "1Results"))

# Other implementations from literature (µs/pt, None = not reported)
OTHER_IMPLS = {
    "Open3D native":     {"SOR": 0.42, "ROR": 0.32, "DSOR": None,  "DROR": None},
    "scipy/numpy naive": {"SOR": 1.87, "ROR": 3.37, "DSOR": None,  "DROR": None},
    "PCL (published)":   {"SOR": 1.5,  "ROR": 1.0,  "DSOR": None,  "DROR": None},
    "Charron 2018":      {"SOR": None, "ROR": None,  "DSOR": None,  "DROR": 300.0},
}

PALETTE = {"SOR": "#F58518", "ROR": "#54A24B", "DSOR": "#B279A2", "DROR": "#9C755F"}
FILTERS = ["SOR", "ROR", "DSOR", "DROR"]


def benchmark_filter(fn, pcd, n_runs: int, warmup: int = 5):
    n_pts = len(pcd.points)
    for _ in range(warmup):
        fn(pcd)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(pcd)
        times.append((time.perf_counter() - t0) * 1e6 / n_pts)
    arr = np.array(times)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "p95": float(np.percentile(arr, 95))}


def run(n_runs: int, output: str):
    sick_path  = THESIS_DATA / "Object detection Accuracy/Snow model/child_ror_filtered.pcd"
    livox_path = THESIS_DATA / "livox/good_weather_livox.pcd"

    for p in [sick_path, livox_path]:
        if not p.exists():
            log.error(f"Missing data: {p}\nSet THESIS_DATA_ROOT to your local private sensor-data export.")
            raise SystemExit(1)

    sick_pcd  = o3d.io.read_point_cloud(str(sick_path))
    livox_pcd = o3d.io.read_point_cloud(str(livox_path))

    f = LiDARFilters()
    filter_fns = {"SOR": f.sor, "ROR": f.ror, "DSOR": f.dsor, "DROR": f.dror}

    log.info(f"Benchmarking SICK  ({len(sick_pcd.points)} pts, {n_runs} runs)...")
    sick_r = {name: benchmark_filter(fn, sick_pcd, n_runs) for name, fn in filter_fns.items()}

    livox_runs = max(10, n_runs // 5)
    log.info(f"Benchmarking Livox ({len(livox_pcd.points):,} pts, {livox_runs} runs)...")
    livox_r = {name: benchmark_filter(fn, livox_pcd, livox_runs, warmup=3) for name, fn in filter_fns.items()}

    # ── Print table ────────────────────────────────────────────────────────────
    log.info("\nResults (µs/point):")
    log.info(f"  {'Filter':<8}  {'SICK mean':>10}  {'SICK p95':>9}  {'Livox mean':>11}")
    log.info("  " + "-" * 45)
    for name in FILTERS:
        log.info(f"  {name:<8}  {sick_r[name]['mean']:>10.3f}  {sick_r[name]['p95']:>9.3f}  {livox_r[name]['mean']:>11.3f}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    colors = [PALETTE[f] for f in FILTERS]
    x = np.arange(len(FILTERS))
    w = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LiDAR Filter Benchmark: This Repo vs Comparable Implementations",
                 fontsize=13, fontweight="bold")

    # Left: SICK vs Livox
    ax = axes[0]
    sick_m  = [sick_r[f]["mean"]  for f in FILTERS]
    livox_m = [livox_r[f]["mean"] for f in FILTERS]
    sick_e  = [sick_r[f]["std"]   for f in FILTERS]
    livox_e = [livox_r[f]["std"]  for f in FILTERS]

    b1 = ax.bar(x - w/2, sick_m,  w, color=colors, alpha=0.9,
                label=f"SICK ({len(sick_pcd.points)} pts)", yerr=sick_e, capsize=3, ecolor="k", error_kw={"lw": 0.8})
    b2 = ax.bar(x + w/2, livox_m, w, color=colors, alpha=0.55, edgecolor="k", lw=0.5,
                label=f"Livox ({len(livox_pcd.points):,} pts)", yerr=livox_e, capsize=3, ecolor="k", error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(FILTERS)
    ax.set_ylabel("µs / point"); ax.set_title("Runtime by Sensor (this repo)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for b, v in zip(b1, sick_m):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Right: SOR & ROR vs published
    ax = axes[1]
    impl_names = ["This repo"] + list(OTHER_IMPLS.keys())
    sor_vals = [sick_r["SOR"]["mean"]] + [OTHER_IMPLS[k]["SOR"] or np.nan for k in OTHER_IMPLS]
    ror_vals = [sick_r["ROR"]["mean"]] + [OTHER_IMPLS[k]["ROR"] or np.nan for k in OTHER_IMPLS]
    xi = np.arange(len(impl_names))

    ax.bar(xi - w/2, sor_vals, w, color="#F58518", alpha=0.85, label="SOR")
    ax.bar(xi + w/2, ror_vals, w, color="#54A24B", alpha=0.85, label="ROR")
    ax.set_xticks(xi); ax.set_xticklabels(impl_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("µs / point"); ax.set_title("SOR / ROR: Cross-implementation", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.annotate("* bars capped at scale — Charron DROR = 300 µs/pt",
                xy=(0.5, -0.18), xycoords="axes fraction", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    log.info(f"\nSaved → {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-runs", type=int, default=100, help="Benchmark iterations per filter (default: 100)")
    parser.add_argument("--output", default="results/benchmark_comparison.png", help="Output PNG path")
    args = parser.parse_args()
    run(args.n_runs, args.output)


if __name__ == "__main__":
    main()
