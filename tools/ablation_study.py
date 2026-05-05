"""
Ablation Study: Thesis vs Variant Implementations
Compare DSOR and DROR thesis originals against your variant implementations

Metrics:
- Point retention %
- Runtime (ms)
- Stability (variance across runs)
- Overall effectiveness

Output:
- JSON results with detailed metrics
- Comparison tables
- Statistical analysis
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import json
import numpy as np
import open3d as o3d
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from lidar_snow_filter.filters import LiDARFilters

try:
    from filters_thesis_originals import ThesisOriginalFilters
    THESIS_FILTERS_AVAILABLE = True
except ImportError:
    THESIS_FILTERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("filters_thesis_originals not available (not included in this repository)")


class AblationStudy:
    """Comprehensive ablation study comparing thesis vs variant implementations."""

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def generate_synthetic_scenario(self, name: str, scenario_type: str,
                                    n_object_points: int = 2000,
                                    n_noise_points: int = 500) -> o3d.geometry.PointCloud:
        """Generate synthetic point clouds for ablation study."""

        if scenario_type == "baseline_cylinder":
            # Clean cylinder
            theta = np.random.uniform(0, 2*np.pi, n_object_points)
            z = np.random.uniform(-0.5, 0.5, n_object_points)
            x = 0.3 * np.cos(theta)
            y = 0.3 * np.sin(theta)
            points = np.column_stack([x, y, z])

        elif scenario_type == "dense_clustered":
            # Dense object + clustered noise (harder than uniform)
            theta = np.random.uniform(0, 2*np.pi, n_object_points)
            z = np.random.uniform(-0.5, 0.5, n_object_points)
            x = 0.3 * np.cos(theta) + np.random.normal(0, 0.01, n_object_points)
            y = 0.3 * np.sin(theta) + np.random.normal(0, 0.01, n_object_points)
            object_pts = np.column_stack([x, y, z])

            # Clustered noise (concentrated regions)
            n_clusters = 5
            cluster_centers = np.random.uniform(-1, 1, (n_clusters, 3))
            noise_per_cluster = n_noise_points // n_clusters
            noise_pts = []
            for center in cluster_centers:
                cluster_noise = center + np.random.normal(0, 0.05, (noise_per_cluster, 3))
                noise_pts.extend(cluster_noise)
            noise_pts = np.array(noise_pts)

            points = np.vstack([object_pts, noise_pts[:n_noise_points]])

        elif scenario_type == "sparse_far_field":
            # Object + very sparse noise (easy case)
            theta = np.random.uniform(0, 2*np.pi, n_object_points)
            z = np.random.uniform(-0.5, 0.5, n_object_points)
            x = 0.3 * np.cos(theta) + np.random.normal(0, 0.005, n_object_points)
            y = 0.3 * np.sin(theta) + np.random.normal(0, 0.005, n_object_points)
            object_pts = np.column_stack([x, y, z])

            # Sparse noise
            noise_pts = np.random.uniform(-2, 2, (n_noise_points, 3))

            points = np.vstack([object_pts, noise_pts[:n_noise_points]])

        elif scenario_type == "high_noise_ratio":
            # Object + high noise (70% noise)
            theta = np.random.uniform(0, 2*np.pi, n_object_points)
            z = np.random.uniform(-0.5, 0.5, n_object_points)
            x = 0.3 * np.cos(theta)
            y = 0.3 * np.sin(theta)
            object_pts = np.column_stack([x, y, z])

            # Very high noise ratio
            noise_pts = np.random.uniform(-1, 1, (n_noise_points * 2, 3))

            points = np.vstack([object_pts, noise_pts])

        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        logger.info(f"Generated {scenario_type}: {len(points)} points ({n_object_points} object, {n_noise_points} noise)")
        return pcd

    def benchmark_filter(self, pcd: o3d.geometry.PointCloud,
                        filter_func, filter_name: str,
                        repeats: int = 10, **kwargs) -> dict:
        """Benchmark a single filter implementation."""

        times = []
        retention_values = []

        for i in range(repeats):
            start = time.perf_counter()
            try:
                filtered, metadata = filter_func(pcd, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # ms

                times.append(elapsed)
                retention = metadata["retention_pct"]
                retention_values.append(retention)

            except Exception as e:
                logger.error(f"{filter_name} iteration {i} failed: {e}")
                continue

        if not times:
            return {
                "error": f"All iterations failed for {filter_name}",
                "n_runs": 0
            }

        times = np.array(times)
        retention_values = np.array(retention_values)

        return {
            "filter": filter_name,
            "n_runs": len(times),
            "time_ms": {
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "stdev": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
            },
            "retention_pct": {
                "mean": float(np.mean(retention_values)),
                "median": float(np.median(retention_values)),
                "stdev": float(np.std(retention_values)),
                "min": float(np.min(retention_values)),
                "max": float(np.max(retention_values)),
            },
            "stability_coefficient": float(np.std(times) / np.mean(times))  # Lower is better
        }

    def run_ablation(self, scenarios: dict, repeats_per_filter: int = 10):
        """Run full ablation study across scenarios and filter variants."""

        logger.info(f"Starting ablation study with {len(scenarios)} scenarios, {repeats_per_filter} repeats each")

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "study_type": "DSOR and DROR - Thesis vs Variant Ablation",
            "scenarios": {},
            "summary": {}
        }

        for scenario_name, (scenario_type, n_obj, n_noise) in scenarios.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Scenario: {scenario_name}")
            logger.info(f"{'='*70}")

            pcd = self.generate_synthetic_scenario(
                scenario_name, scenario_type,
                n_object_points=n_obj,
                n_noise_points=n_noise
            )

            scenario_results = {
                "description": f"{scenario_type} ({n_obj} object pts, {n_noise} noise pts)",
                "total_points": len(pcd.points),
                "filters": {}
            }

            # Test DSOR variants
            logger.info("\n--- DSOR Variants ---")

            # Thesis DSOR
            logger.info("Testing DSOR_THESIS...")
            if THESIS_FILTERS_AVAILABLE:
                dsor_thesis_result = self.benchmark_filter(
                    pcd, ThesisOriginalFilters.dsor_thesis,
                    "DSOR_THESIS",
                    repeats=repeats_per_filter,
                    rho_max=3.0,
                    nb_neighbors=20
                )
            else:
                dsor_thesis_result = {"error": "Thesis filters not available", "n_runs": 0}
                logger.warning("  ⊘ Skipped: filters_thesis_originals module not found")
            scenario_results["filters"]["DSOR_THESIS"] = dsor_thesis_result
            if "error" not in dsor_thesis_result:
                logger.info(f"  Retention: {dsor_thesis_result['retention_pct']['mean']:.2f}% "
                          f"Runtime: {dsor_thesis_result['time_ms']['mean']:.2f}ms")

            # Variant DSOR (height-based)
            logger.info("Testing DSOR_VARIANT (height-based)...")
            dsor_variant_result = self.benchmark_filter(
                pcd, LiDARFilters.dsor,
                "DSOR_VARIANT",
                repeats=repeats_per_filter,
                min_ratio=1.5,
                sector_count=8
            )
            scenario_results["filters"]["DSOR_VARIANT"] = dsor_variant_result
            if "error" not in dsor_variant_result:
                logger.info(f"  Retention: {dsor_variant_result['retention_pct']['mean']:.2f}% "
                          f"Runtime: {dsor_variant_result['time_ms']['mean']:.2f}ms")

            # Test DROR variants
            logger.info("\n--- DROR Variants ---")

            # Thesis DROR
            logger.info("Testing DROR_THESIS...")
            if THESIS_FILTERS_AVAILABLE:
                dror_thesis_result = self.benchmark_filter(
                    pcd, ThesisOriginalFilters.dror_thesis,
                    "DROR_THESIS",
                    repeats=repeats_per_filter,
                    R_0=0.05,
                    N_min=5,
                    sector_count=12
                )
            else:
                dror_thesis_result = {"error": "Thesis filters not available", "n_runs": 0}
                logger.warning("  ⊘ Skipped: filters_thesis_originals module not found")
            scenario_results["filters"]["DROR_THESIS"] = dror_thesis_result
            if "error" not in dror_thesis_result:
                logger.info(f"  Retention: {dror_thesis_result['retention_pct']['mean']:.2f}% "
                          f"Runtime: {dror_thesis_result['time_ms']['mean']:.2f}ms")

            # Variant DROR (cubic-root)
            logger.info("Testing DROR_VARIANT (cubic-root)...")
            dror_variant_result = self.benchmark_filter(
                pcd, LiDARFilters.dror,
                "DROR_VARIANT",
                repeats=repeats_per_filter,
                sector_count=12,
                scale_factor=1.5
            )
            scenario_results["filters"]["DROR_VARIANT"] = dror_variant_result
            if "error" not in dror_variant_result:
                logger.info(f"  Retention: {dror_variant_result['retention_pct']['mean']:.2f}% "
                          f"Runtime: {dror_variant_result['time_ms']['mean']:.2f}ms")

            all_results["scenarios"][scenario_name] = scenario_results

        # Compute summary statistics
        all_results["summary"] = self._compute_summary(all_results["scenarios"])

        return all_results

    def _compute_summary(self, scenarios: dict) -> dict:
        """Compute aggregate statistics across all scenarios."""

        summary = {
            "DSOR": {
                "thesis_vs_variant": {
                    "retention_diff_mean": None,
                    "retention_diff_max": None,
                    "runtime_ratio": None,
                    "variant_wins_count": 0,
                    "thesis_wins_count": 0
                }
            },
            "DROR": {
                "thesis_vs_variant": {
                    "retention_diff_mean": None,
                    "retention_diff_max": None,
                    "runtime_ratio": None,
                    "variant_wins_count": 0,
                    "thesis_wins_count": 0
                }
            }
        }

        dsor_retention_diffs = []
        dsor_runtime_ratios = []
        dror_retention_diffs = []
        dror_runtime_ratios = []

        for scenario_name, scenario in scenarios.items():
            filters = scenario["filters"]

            # DSOR comparison
            if "DSOR_THESIS" in filters and "DSOR_VARIANT" in filters:
                thesis = filters["DSOR_THESIS"]
                variant = filters["DSOR_VARIANT"]

                if "error" not in thesis and "error" not in variant:
                    ret_diff = variant["retention_pct"]["mean"] - thesis["retention_pct"]["mean"]
                    dsor_retention_diffs.append(ret_diff)
                    runtime_ratio = variant["time_ms"]["mean"] / thesis["time_ms"]["mean"]
                    dsor_runtime_ratios.append(runtime_ratio)

                    if ret_diff > 0:
                        summary["DSOR"]["thesis_vs_variant"]["variant_wins_count"] += 1
                    else:
                        summary["DSOR"]["thesis_vs_variant"]["thesis_wins_count"] += 1

            # DROR comparison
            if "DROR_THESIS" in filters and "DROR_VARIANT" in filters:
                thesis = filters["DROR_THESIS"]
                variant = filters["DROR_VARIANT"]

                if "error" not in thesis and "error" not in variant:
                    ret_diff = variant["retention_pct"]["mean"] - thesis["retention_pct"]["mean"]
                    dror_retention_diffs.append(ret_diff)
                    runtime_ratio = variant["time_ms"]["mean"] / thesis["time_ms"]["mean"]
                    dror_runtime_ratios.append(runtime_ratio)

                    if ret_diff > 0:
                        summary["DROR"]["thesis_vs_variant"]["variant_wins_count"] += 1
                    else:
                        summary["DROR"]["thesis_vs_variant"]["thesis_wins_count"] += 1

        # Calculate summary statistics
        if dsor_retention_diffs:
            summary["DSOR"]["thesis_vs_variant"]["retention_diff_mean"] = float(np.mean(dsor_retention_diffs))
            summary["DSOR"]["thesis_vs_variant"]["retention_diff_max"] = float(np.max(np.abs(dsor_retention_diffs)))
            summary["DSOR"]["thesis_vs_variant"]["runtime_ratio"] = float(np.mean(dsor_runtime_ratios))

        if dror_retention_diffs:
            summary["DROR"]["thesis_vs_variant"]["retention_diff_mean"] = float(np.mean(dror_retention_diffs))
            summary["DROR"]["thesis_vs_variant"]["retention_diff_max"] = float(np.max(np.abs(dror_retention_diffs)))
            summary["DROR"]["thesis_vs_variant"]["runtime_ratio"] = float(np.mean(dror_runtime_ratios))

        return summary

    def save_results(self, results: dict) -> Path:
        """Save ablation study results to JSON."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ablation_study_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")
        return output_file

    def generate_comparison_table(self, results: dict) -> str:
        """Generate markdown comparison table."""

        table = "# Ablation Study Results: Thesis vs Variant\n\n"

        for scenario_name, scenario in results["scenarios"].items():
            table += f"## {scenario_name}\n"
            table += f"Total points: {scenario['total_points']}\n\n"

            table += "| Filter | Retention (%) | Runtime (ms) | Stability |\n"
            table += "|--------|---------------|--------------|----------|\n"

            for filter_name, metrics in scenario["filters"].items():
                if "error" in metrics:
                    table += f"| {filter_name} | ERROR | - | - |\n"
                else:
                    ret = metrics["retention_pct"]["mean"]
                    rt = metrics["time_ms"]["mean"]
                    stab = metrics["stability_coefficient"]
                    table += f"| {filter_name} | {ret:.2f} | {rt:.2f} | {stab:.3f} |\n"

            table += "\n"

        # Summary
        table += "## Summary\n"
        table += "### DSOR (Thesis vs Variant)\n"
        dsor_summary = results["summary"]["DSOR"]["thesis_vs_variant"]
        table += f"- **Variant Wins**: {dsor_summary['variant_wins_count']} scenarios\n"
        table += f"- **Thesis Wins**: {dsor_summary['thesis_wins_count']} scenarios\n"
        if dsor_summary["retention_diff_mean"] is not None:
            table += f"- **Avg Retention Difference**: {dsor_summary['retention_diff_mean']:.2f}%\n"
            table += f"- **Runtime Ratio**: {dsor_summary['runtime_ratio']:.2f}x\n"

        table += "\n### DROR (Thesis vs Variant)\n"
        dror_summary = results["summary"]["DROR"]["thesis_vs_variant"]
        table += f"- **Variant Wins**: {dror_summary['variant_wins_count']} scenarios\n"
        table += f"- **Thesis Wins**: {dror_summary['thesis_wins_count']} scenarios\n"
        if dror_summary["retention_diff_mean"] is not None:
            table += f"- **Avg Retention Difference**: {dror_summary['retention_diff_mean']:.2f}%\n"
            table += f"- **Runtime Ratio**: {dror_summary['runtime_ratio']:.2f}x\n"

        return table


if __name__ == "__main__":
    # Define test scenarios
    scenarios = {
        "Baseline_Cylinder": ("baseline_cylinder", 2000, 300),
        "Dense_Clustered": ("dense_clustered", 2000, 500),
        "Sparse_FarField": ("sparse_far_field", 2000, 400),
        "High_Noise_70pct": ("high_noise_ratio", 2000, 1400),
    }

    # Run ablation study
    ablation = AblationStudy(output_dir=REPO_ROOT / "results" / "ablation_study")
    results = ablation.run_ablation(scenarios, repeats_per_filter=10)

    # Save results
    output_file = ablation.save_results(results)

    # Generate comparison table
    comparison = ablation.generate_comparison_table(results)
    print("\n" + "="*70)
    print(comparison)
    print("="*70)

    # Print summary
    print("\n✅ Ablation study complete!")
    print(f"Full results: {output_file}")
