"""
Real-World Validation: Thesis vs Variant Implementations on Actual SICK/Livox Data
Compare algorithms on real sensor data from master's thesis experiments
"""

import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from datetime import datetime
import time
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import implementations
from filters import LiDARFilters

# Attempt to import thesis original filters (not included in public repo)
try:
    from filters_thesis_originals import ThesisOriginalFilters
    THESIS_FILTERS_AVAILABLE = True
except ImportError:
    THESIS_FILTERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("filters_thesis_originals not available (not included in this repository)")


class RealWorldValidator:
    """Validate filter implementations on actual thesis sensor data."""

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _repo_root = Path(__file__).resolve().parent
        self.thesis_data_root = Path(os.environ.get("THESIS_DATA_ROOT", _repo_root / "thesis/thesis/thesis/1Results"))
        self.results = {}

    def find_data_files(self) -> dict:
        """Locate actual thesis data files."""
        logger.info("Searching for thesis data files...")

        data_files = {
            "livox_clear": [],
            "livox_snow": [],
            "livox_gt": None
        }

        # Find Livox clear scans
        livox_clear_dir = self.thesis_data_root / "livox" / "clear 10 scans"
        if livox_clear_dir.exists():
            clear_files = sorted(livox_clear_dir.glob("*.pcd"))
            data_files["livox_clear"] = clear_files
            logger.info(f"Found {len(clear_files)} Livox clear scans")

        # Find Livox snow scans
        livox_mannequin_dir = self.thesis_data_root / "livox" / "mannenquin clusters"
        if livox_mannequin_dir.exists():
            snow_files = sorted(livox_mannequin_dir.glob("*.pcd"))
            data_files["livox_snow"] = snow_files
            logger.info(f"Found {len(snow_files)} Livox snow scans")

        # Find ground truth model
        livox_dir = self.thesis_data_root / "livox"
        gt_file = livox_dir / "livox_gt_model_merged_downsampled.pcd"
        if gt_file.exists():
            data_files["livox_gt"] = gt_file
            logger.info(f"Found ground truth model: {gt_file.name}")

        return data_files

    def load_and_inspect(self, pcd_path: Path) -> Tuple[o3d.geometry.PointCloud, dict]:
        """Load and inspect a point cloud."""
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        info = {
            "file": pcd_path.name,
            "points": len(pcd.points),
            "has_colors": pcd.has_colors(),
            "has_normals": pcd.has_normals(),
            "bounds": {
                "x": [float(pcd.get_min_bound()[0]), float(pcd.get_max_bound()[0])],
                "y": [float(pcd.get_min_bound()[1]), float(pcd.get_max_bound()[1])],
                "z": [float(pcd.get_min_bound()[2]), float(pcd.get_max_bound()[2])]
            }
        }

        logger.info(f"Loaded {info['file']}: {info['points']} points")
        logger.info(f"  Bounds X: {info['bounds']['x']}")
        logger.info(f"  Bounds Y: {info['bounds']['y']}")
        logger.info(f"  Bounds Z: {info['bounds']['z']}")

        return pcd, info

    def run_filter_comparison(self, pcd: o3d.geometry.PointCloud,
                             dataset_name: str, file_name: str) -> dict:
        """Run all filter variants on a single point cloud."""

        input_size = len(pcd.points)
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {dataset_name} / {file_name} ({input_size} points)")
        logger.info(f"{'='*70}")

        results = {
            "dataset": dataset_name,
            "file": file_name,
            "input_points": input_size,
            "filters": {}
        }

        # DSOR Thesis
        logger.info("\nDSOR_THESIS...")
        if THESIS_FILTERS_AVAILABLE:
            try:
                start = time.perf_counter()
                filtered, metadata = ThesisOriginalFilters.dsor_thesis(pcd)
                elapsed = (time.perf_counter() - start) * 1000

                results["filters"]["DSOR_THESIS"] = {
                    "retention_pct": metadata["retention_pct"],
                    "output_points": metadata["output_points"],
                    "runtime_ms": elapsed,
                    "status": "success"
                }
                logger.info(f"  ✓ Retention: {metadata['retention_pct']:.2f}%, Runtime: {elapsed:.2f}ms")
            except Exception as e:
                results["filters"]["DSOR_THESIS"] = {"error": str(e), "status": "failed"}
                logger.error(f"  ✗ Failed: {e}")
        else:
            results["filters"]["DSOR_THESIS"] = {"error": "Thesis filters not available", "status": "skipped"}
            logger.warning(f"  ⊘ Skipped: filters_thesis_originals module not found")

        # DSOR Variant
        logger.info("DSOR_VARIANT...")
        try:
            start = time.perf_counter()
            filtered, metadata = LiDARFilters.dsor(pcd)
            elapsed = (time.perf_counter() - start) * 1000

            results["filters"]["DSOR_VARIANT"] = {
                "retention_pct": metadata["retention_pct"],
                "output_points": metadata["output_points"],
                "runtime_ms": elapsed,
                "status": "success"
            }
            logger.info(f"  ✓ Retention: {metadata['retention_pct']:.2f}%, Runtime: {elapsed:.2f}ms")
        except Exception as e:
            results["filters"]["DSOR_VARIANT"] = {"error": str(e), "status": "failed"}
            logger.error(f"  ✗ Failed: {e}")

        # DROR Thesis
        logger.info("DROR_THESIS...")
        if THESIS_FILTERS_AVAILABLE:
            try:
                start = time.perf_counter()
                filtered, metadata = ThesisOriginalFilters.dror_thesis(pcd)
                elapsed = (time.perf_counter() - start) * 1000

                results["filters"]["DROR_THESIS"] = {
                    "retention_pct": metadata["retention_pct"],
                    "output_points": metadata["output_points"],
                    "runtime_ms": elapsed,
                    "status": "success"
                }
                logger.info(f"  ✓ Retention: {metadata['retention_pct']:.2f}%, Runtime: {elapsed:.2f}ms")
            except Exception as e:
                results["filters"]["DROR_THESIS"] = {"error": str(e), "status": "failed"}
                logger.error(f"  ✗ Failed: {e}")
        else:
            results["filters"]["DROR_THESIS"] = {"error": "Thesis filters not available", "status": "skipped"}
            logger.warning(f"  ⊘ Skipped: filters_thesis_originals module not found")

        # DROR Variant
        logger.info("DROR_VARIANT...")
        try:
            start = time.perf_counter()
            filtered, metadata = LiDARFilters.dror(pcd)
            elapsed = (time.perf_counter() - start) * 1000

            results["filters"]["DROR_VARIANT"] = {
                "retention_pct": metadata["retention_pct"],
                "output_points": metadata["output_points"],
                "runtime_ms": elapsed,
                "status": "success"
            }
            logger.info(f"  ✓ Retention: {metadata['retention_pct']:.2f}%, Runtime: {elapsed:.2f}ms")
        except Exception as e:
            results["filters"]["DROR_VARIANT"] = {"error": str(e), "status": "failed"}
            logger.error(f"  ✗ Failed: {e}")

        # Comparison
        filters = results["filters"]
        if all("error" not in f for f in filters.values()):
            dsor_thesis_ret = filters["DSOR_THESIS"]["retention_pct"]
            dsor_variant_ret = filters["DSOR_VARIANT"]["retention_pct"]
            dror_thesis_ret = filters["DROR_THESIS"]["retention_pct"]
            dror_variant_ret = filters["DROR_VARIANT"]["retention_pct"]

            logger.info(f"\nSUMMARY:")
            logger.info(f"  DSOR: Thesis {dsor_thesis_ret:.2f}% vs Variant {dsor_variant_ret:.2f}% "
                       f"(Δ {dsor_variant_ret - dsor_thesis_ret:+.2f}%)")
            logger.info(f"  DROR: Thesis {dror_thesis_ret:.2f}% vs Variant {dror_variant_ret:.2f}% "
                       f"(Δ {dror_variant_ret - dror_thesis_ret:+.2f}%)")

        return results

    def run_validation(self):
        """Run full real-world validation."""

        data_files = self.find_data_files()

        if not data_files["livox_clear"] and not data_files["livox_snow"]:
            logger.error("No data files found! Check thesis folder structure.")
            return

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "study_type": "Real-World Validation: Thesis vs Variant on Actual Sensor Data",
            "test_results": []
        }

        # Test clear scans (up to 3 samples)
        clear_samples = data_files["livox_clear"][:3]
        for pcd_path in clear_samples:
            pcd, info = self.load_and_inspect(pcd_path)
            result = self.run_filter_comparison(pcd, "Livox_Clear_Weather", pcd_path.name)
            result["metadata"] = info
            all_results["test_results"].append(result)

        # Test snow scans
        snow_samples = data_files["livox_snow"][:2]
        for pcd_path in snow_samples:
            pcd, info = self.load_and_inspect(pcd_path)
            result = self.run_filter_comparison(pcd, "Livox_Snow", pcd_path.name)
            result["metadata"] = info
            all_results["test_results"].append(result)

        # Test ground truth model
        if data_files["livox_gt"]:
            logger.info("\n" + "="*70)
            logger.info("Testing Ground Truth Model (Merged Clear Scans)")
            logger.info("="*70)
            pcd, info = self.load_and_inspect(data_files["livox_gt"])
            result = self.run_filter_comparison(pcd, "Livox_GroundTruth", data_files["livox_gt"].name)
            result["metadata"] = info
            all_results["test_results"].append(result)

        # Compute summary
        all_results["summary"] = self._compute_summary(all_results["test_results"])

        return all_results

    def _compute_summary(self, results: List[dict]) -> dict:
        """Compute aggregate statistics."""

        summary = {
            "total_tests": len(results),
            "dsor_wins_variant": 0,
            "dror_wins_variant": 0,
            "retention_improvements": {
                "dsor": [],
                "dror": []
            },
            "runtime_ratios": {
                "dsor": [],
                "dror": []
            }
        }

        for result in results:
            filters = result["filters"]

            # DSOR comparison
            if "DSOR_THESIS" in filters and "DSOR_VARIANT" in filters:
                thesis = filters["DSOR_THESIS"]
                variant = filters["DSOR_VARIANT"]

                if thesis.get("status") == "success" and variant.get("status") == "success":
                    ret_diff = variant["retention_pct"] - thesis["retention_pct"]
                    summary["retention_improvements"]["dsor"].append(ret_diff)
                    runtime_ratio = variant["runtime_ms"] / thesis["runtime_ms"]
                    summary["runtime_ratios"]["dsor"].append(runtime_ratio)

                    if ret_diff > 0:
                        summary["dsor_wins_variant"] += 1

            # DROR comparison
            if "DROR_THESIS" in filters and "DROR_VARIANT" in filters:
                thesis = filters["DROR_THESIS"]
                variant = filters["DROR_VARIANT"]

                if thesis.get("status") == "success" and variant.get("status") == "success":
                    ret_diff = variant["retention_pct"] - thesis["retention_pct"]
                    summary["retention_improvements"]["dror"].append(ret_diff)
                    runtime_ratio = variant["runtime_ms"] / thesis["runtime_ms"]
                    summary["runtime_ratios"]["dror"].append(runtime_ratio)

                    if ret_diff > 0:
                        summary["dror_wins_variant"] += 1

        # Calculate averages
        if summary["retention_improvements"]["dsor"]:
            summary["dsor_avg_retention_improvement"] = float(np.mean(summary["retention_improvements"]["dsor"]))
            summary["dsor_avg_runtime_ratio"] = float(np.mean(summary["runtime_ratios"]["dsor"]))

        if summary["retention_improvements"]["dror"]:
            summary["dror_avg_retention_improvement"] = float(np.mean(summary["retention_improvements"]["dror"]))
            summary["dror_avg_runtime_ratio"] = float(np.mean(summary["runtime_ratios"]["dror"]))

        return summary

    def save_results(self, results: dict) -> Path:
        """Save validation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"real_world_validation_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")
        return output_file

    def generate_report(self, results: dict) -> str:
        """Generate markdown report."""

        report = "# Real-World Validation Report\n"
        report += "## Thesis vs Variant Implementations on Actual Sensor Data\n\n"

        report += f"**Timestamp**: {results['timestamp']}\n"
        report += f"**Total Tests**: {results['summary']['total_tests']}\n\n"

        # Summary table
        report += "## Summary Statistics\n\n"
        report += "### DSOR Performance\n"
        if "dsor_avg_retention_improvement" in results["summary"]:
            report += f"- **Variant Wins**: {results['summary']['dsor_wins_variant']}/{results['summary']['total_tests']} tests\n"
            report += f"- **Avg Retention Improvement**: {results['summary']['dsor_avg_retention_improvement']:.2f}%\n"
            report += f"- **Avg Runtime Ratio**: {results['summary']['dsor_avg_runtime_ratio']:.2f}x\n"

        report += "\n### DROR Performance\n"
        if "dror_avg_retention_improvement" in results["summary"]:
            report += f"- **Variant Wins**: {results['summary']['dror_wins_variant']}/{results['summary']['total_tests']} tests\n"
            report += f"- **Avg Retention Improvement**: {results['summary']['dror_avg_retention_improvement']:.2f}%\n"
            report += f"- **Avg Runtime Ratio**: {results['summary']['dror_avg_runtime_ratio']:.2f}x\n"

        # Detailed results
        report += "\n## Detailed Results\n\n"
        for i, test_result in enumerate(results["test_results"], 1):
            report += f"### Test {i}: {test_result['dataset']} / {test_result['file']}\n"
            report += f"**Input Points**: {test_result['input_points']:,}\n\n"

            report += "| Filter | Output Points | Retention | Runtime |\n"
            report += "|--------|---------------|-----------|----------|\n"

            for filter_name, metrics in test_result["filters"].items():
                if metrics.get("status") == "success":
                    output = metrics["output_points"]
                    retention = metrics["retention_pct"]
                    runtime = metrics["runtime_ms"]
                    report += f"| {filter_name} | {output:,} | {retention:.2f}% | {runtime:.2f}ms |\n"
                else:
                    report += f"| {filter_name} | ERROR | - | - |\n"

            report += "\n"

        return report


if __name__ == "__main__":
    validator = RealWorldValidator()
    results = validator.run_validation()

    if results:
        # Save results
        output_file = validator.save_results(results)

        # Generate and display report
        report = validator.generate_report(results)
        print("\n" + "="*70)
        print(report)
        print("="*70)

        logger.info("\n✅ Real-world validation complete!")
        logger.info(f"Full results: {output_file}")
    else:
        logger.error("❌ Validation failed - no data found")
