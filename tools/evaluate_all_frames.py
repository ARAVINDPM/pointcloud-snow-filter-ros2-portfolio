#!/usr/bin/env python3
"""
Multi-frame statistical evaluation of filters.

Generates 10+ synthetic frames and computes mean ± std metrics
with statistical significance tests (Wilcoxon signed-rank).

Usage:
    python tools/evaluate_all_frames.py --num_frames 10 --seed 42
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import open3d as o3d
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lidar_snow_filter.filters import LiDARFilters
from lidar_snow_filter.metrics import ComprehensiveEvaluation
from lidar_snow_filter.synthetic_data_generator import (
    SyntheticMannequinGenerator,
    SnowContaminationSimulator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_multi_frame_dataset(
    num_frames: int = 10,
    seed: int = 42,
    sensor: str = "livox"
) -> Tuple[List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud]]:
    """
    Generate multiple clean and contaminated frames for statistical evaluation.

    Args:
        num_frames: Number of frames to generate
        seed: Random seed (for reproducibility)
        sensor: "livox" or "sick"

    Returns:
        (clean_frames, contaminated_frames) lists
    """
    logger.info(f"Generating {num_frames} frames (seed={seed}, sensor={sensor})...")

    contaminator = SnowContaminationSimulator(seed=seed)

    clean_frames = []
    contaminated_frames = []

    for i in range(num_frames):
        # Generate unique frame with distinct seed (no global state pollution)
        gen = SyntheticMannequinGenerator(sensor=sensor, seed=seed + i)

        clean = gen.generate_single_scan()
        contaminated = contaminator.contaminate(clean, snow_density=0.25)

        clean_frames.append(clean)
        contaminated_frames.append(contaminated)

        logger.info(f"  Frame {i+1}/{num_frames}: {len(clean.points)} pts → {len(contaminated.points)} pts")

    return clean_frames, contaminated_frames


def evaluate_filter_on_frames(
    filter_name: str,
    filter_func,
    contaminated_frames: List[o3d.geometry.PointCloud],
    clean_frames: List[o3d.geometry.PointCloud],
    **filter_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single filter across multiple frames.

    Returns:
        Dictionary mapping metric_name → {mean, std, min, max}
    """
    results = {
        "retention_pct": [],
        "aabb_iou": [],
        "voxel_iou": [],
        "chamfer_distance_cm": [],
        "centroid_displacement_mm": [],
    }

    logger.info(f"\nEvaluating {filter_name}...")

    for i, (contam, clean) in enumerate(zip(contaminated_frames, clean_frames)):
        try:
            # Apply filter
            filtered, meta = filter_func(contam, **filter_kwargs)

            # Evaluate
            metrics = ComprehensiveEvaluation.evaluate(
                filtered, clean, filter_name=f"{filter_name}_frame{i}",
                original_input_points=len(contam.points)
            )

            results["retention_pct"].append(metrics["retention_pct"])
            results["aabb_iou"].append(metrics["aabb_iou"])
            results["voxel_iou"].append(metrics["voxel_iou"])
            results["chamfer_distance_cm"].append(metrics["chamfer_distance_cm"])
            results["centroid_displacement_mm"].append(metrics["centroid_displacement_m"] * 1000)

        except Exception as e:
            logger.error(f"  Frame {i}: {filter_name} failed: {e}")

    # Compute statistics
    stats_dict = {}
    for metric, values in results.items():
        if len(values) > 0:
            stats_dict[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values)
            }

    return stats_dict


def compute_statistical_significance(
    filter1_results: Dict[str, List[float]],
    filter2_results: Dict[str, List[float]],
    metric: str
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test comparing two filter results.

    Args:
        filter1_results: Frame-by-frame results from filter 1
        filter2_results: Frame-by-frame results from filter 2
        metric: Metric name to test

    Returns:
        (statistic, p_value)
    """
    if len(filter1_results[metric]) < 3:
        return np.nan, np.nan

    stat, pval = stats.wilcoxon(
        filter1_results[metric],
        filter2_results[metric]
    )
    return stat, pval


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-frame statistical evaluation of LiDAR filters"
    )
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sensor", choices=["livox", "sick"], default="livox")
    parser.add_argument("--output", type=str, default="results/statistical_evaluation.json")
    args = parser.parse_args()

    # Generate dataset
    clean_frames, contaminated_frames = generate_multi_frame_dataset(
        num_frames=args.num_frames,
        seed=args.seed,
        sensor=args.sensor
    )

    # Initialize filters
    filters = LiDARFilters()
    filter_configs = {
        "SOR": (filters.sor, {}),
        "ROR": (filters.ror, {}),
        "HA-DSOR": (filters.dsor, {}),
        "HA-DROR": (filters.dror, {}),
    }

    # Evaluate all filters
    all_results = {}
    for filter_name, (filter_func, kwargs) in filter_configs.items():
        stats_results = evaluate_filter_on_frames(
            filter_name, filter_func, contaminated_frames, clean_frames, **kwargs
        )
        all_results[filter_name] = stats_results

    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL SUMMARY (10 frames)")
    logger.info("="*80)

    summary_data = []
    for filter_name, stats_dict in all_results.items():
        row = {"Filter": filter_name}
        for metric, stat_values in stats_dict.items():
            if np.isnan(stat_values["mean"]):
                row[f"{metric}"] = "N/A"
            else:
                mean = stat_values["mean"]
                std = stat_values["std"]
                row[f"{metric}"] = f"{mean:.2f} ± {std:.2f}"
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    logger.info("\n" + df.to_string(index=False))

    # Statistical tests: ROR vs HA-DSOR (most relevant comparison)
    logger.info("\n" + "="*80)
    logger.info("WILCOXON SIGNED-RANK TEST: ROR vs HA-DSOR")
    logger.info("="*80)
    logger.info("Null hypothesis: ROR and HA-DSOR have identical distributions")
    logger.info("Significance level: α = 0.05")

    comparison_data = []
    for metric in ["aabb_iou", "voxel_iou", "chamfer_distance_cm", "centroid_displacement_mm"]:
        # Reconstruct per-frame lists for Wilcoxon test
        ror_results = {metric: []}
        hadsor_results = {metric: []}

        for clean, contam in zip(clean_frames, contaminated_frames):
            try:
                f_ror, _ = filters.ror(contam)
                f_hadsor, _ = filters.dsor(contam)

                if metric == "aabb_iou":
                    from lidar_snow_filter.metrics import GeometryMetrics
                    ror_results[metric].append(GeometryMetrics.aabb_iou(f_ror, clean))
                    hadsor_results[metric].append(GeometryMetrics.aabb_iou(f_hadsor, clean))
                elif metric == "voxel_iou":
                    from lidar_snow_filter.metrics import GeometryMetrics
                    ror_results[metric].append(GeometryMetrics.voxel_iou(f_ror, clean))
                    hadsor_results[metric].append(GeometryMetrics.voxel_iou(f_hadsor, clean))
                elif metric == "chamfer_distance_cm":
                    from lidar_snow_filter.metrics import GeometryMetrics
                    _, cm = GeometryMetrics.chamfer_distance(f_ror, clean)
                    ror_results[metric].append(cm if not np.isnan(cm) else 0)
                    _, cm = GeometryMetrics.chamfer_distance(f_hadsor, clean)
                    hadsor_results[metric].append(cm if not np.isnan(cm) else 0)
                elif metric == "centroid_displacement_mm":
                    from lidar_snow_filter.metrics import StabilityMetrics
                    ror_results[metric].append(StabilityMetrics.centroid_displacement(f_ror, clean) * 1000)
                    hadsor_results[metric].append(StabilityMetrics.centroid_displacement(f_hadsor, clean) * 1000)
            except Exception as e:
                logger.debug(f"Skipping frame due to error: {e}")
                continue

        if len(ror_results[metric]) >= 3:
            stat, pval = stats.wilcoxon(ror_results[metric], hadsor_results[metric])
            significance = "***" if pval < 0.01 else ("**" if pval < 0.05 else "ns")
            comparison_data.append({
                "Metric": metric,
                "Statistic": f"{stat:.2f}",
                "p-value": f"{pval:.4f}",
                "Significant?": significance
            })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        logger.info("\n" + comp_df.to_string(index=False))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
