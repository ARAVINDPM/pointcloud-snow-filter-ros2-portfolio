"""
Test and visualization script for LiDAR filters.

Generates synthetic point clouds and produces comparison plots.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import logging
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from lidar_snow_filter.config import RESULTS_DIR
from lidar_snow_filter.filters import LiDARFilters
from lidar_snow_filter.benchmarking import RobustBenchmark
from lidar_snow_filter.metrics import ComprehensiveEvaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_cloud(n_points: int = 50000,
                          noise_ratio: float = 0.15) -> tuple:
    """
    Create synthetic point cloud: mannequin shape + snow noise.

    Args:
        n_points: Total points
        noise_ratio: Fraction of noise points

    Returns:
        (cloud_with_noise, ground_truth_cloud) tuple
    """
    logger.info(f"Creating synthetic cloud: {n_points} points, {noise_ratio*100:.0f}% noise")

    # Create synthetic object: cylinder (mannequin torso)
    n_object = int(n_points * (1 - noise_ratio))

    # Cylinder: radius 0.15m, height 1.7m, centered at origin
    angles = np.random.uniform(0, 2*np.pi, n_object)
    heights = np.random.uniform(-0.85, 0.85, n_object)
    radii = np.random.uniform(0, 0.15, n_object)

    object_points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles),
        heights
    ])

    # Create ground truth cloud
    gt_cloud = o3d.geometry.PointCloud()
    gt_cloud.points = o3d.utility.Vector3dVector(object_points)

    # Add noise (snow): random points in nearby space
    n_noise = int(n_points * noise_ratio)
    noise_points = np.random.uniform(-0.5, 0.5, (n_noise, 3))
    noise_points[:, 2] = np.random.uniform(-1.2, 1.2, n_noise)  # Vertical spread

    all_points = np.vstack([object_points, noise_points])
    noisy_cloud = o3d.geometry.PointCloud()
    noisy_cloud.points = o3d.utility.Vector3dVector(all_points)

    logger.info(f"  Ground truth: {len(gt_cloud.points)} points")
    logger.info(f"  With noise: {len(noisy_cloud.points)} points")

    return noisy_cloud, gt_cloud


def apply_all_filters(noisy_cloud: o3d.geometry.PointCloud) -> dict:
    """Apply all 4 filters to the noisy cloud."""
    logger.info("\nApplying filters...")

    filters_results = {}

    filters_list = [
        ('SOR', lambda pcd: LiDARFilters.sor(pcd)),
        ('ROR', lambda pcd: LiDARFilters.ror(pcd)),
        ('DSOR', lambda pcd: LiDARFilters.dsor(pcd)),
        ('DROR', lambda pcd: LiDARFilters.dror(pcd)),
    ]

    for name, filter_func in filters_list:
        try:
            filtered, metadata = filter_func(noisy_cloud)
            filters_results[name] = filtered
            logger.info(f"  {name}: {metadata['output_points']} points "
                       f"({metadata['retention_pct']:.1f}% retention)")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    return filters_results


def benchmark_filters(noisy_cloud: o3d.geometry.PointCloud) -> dict:
    """Benchmark all filters."""
    logger.info("\nBenchmarking filters...")

    benchmarks = {}

    filters_list = [
        ('SOR', lambda p: LiDARFilters.sor(p)[0]),
        ('ROR', lambda p: LiDARFilters.ror(p)[0]),
        ('DSOR', lambda p: LiDARFilters.dsor(p)[0]),
        ('DROR', lambda p: LiDARFilters.dror(p)[0]),
    ]

    for name, filter_func in filters_list:
        try:
            bench = RobustBenchmark(repeats=30, warmup=2)
            median_time, stats = bench.run(filter_func, noisy_cloud)
            benchmarks[name] = stats
            logger.info(f"  {name}: {stats['microseconds_per_point']:.3f} µs/pt "
                       f"(±{stats['stdev_ms']:.2f}ms)")
        except Exception as e:
            logger.warning(f"  Benchmark failed for {name}: {e}")

    return benchmarks


def evaluate_filters(filtered_clouds: dict,
                     input_cloud: o3d.geometry.PointCloud,
                     ground_truth: o3d.geometry.PointCloud) -> dict:
    """Evaluate filter quality against ground truth."""
    logger.info("\nEvaluating filter quality...")

    evaluations = {}

    for name, filtered in filtered_clouds.items():
        try:
            results = ComprehensiveEvaluation.evaluate(filtered, ground_truth, name,
                                               original_input_points=len(input_cloud.points))
            evaluations[name] = results
        except Exception as e:
            logger.warning(f"  Evaluation failed for {name}: {e}")

    return evaluations


def plot_point_clouds(noisy_cloud: o3d.geometry.PointCloud,
                      filtered_clouds: dict,
                      ground_truth: o3d.geometry.PointCloud) -> str:
    """Create 3D point cloud comparison plot."""
    logger.info("\nGenerating point cloud plots...")

    fig = plt.figure(figsize=(16, 12))

    # Original noisy cloud
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    pts = np.asarray(noisy_cloud.points)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=1, alpha=0.5)
    ax1.set_title(f'Noisy Cloud\n({len(pts)} points)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Ground truth
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    gt_pts = np.asarray(ground_truth.points)
    ax2.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='green', s=1, alpha=0.8)
    ax2.set_title(f'Ground Truth\n({len(gt_pts)} points)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Filtered clouds
    positions = [3, 4, 5, 6]
    colors_list = ['blue', 'purple', 'orange', 'brown']

    for idx, (name, cloud) in enumerate(filtered_clouds.items()):
        ax = fig.add_subplot(2, 3, positions[idx], projection='3d')
        fpts = np.asarray(cloud.points)
        retention = (len(fpts) / len(pts)) * 100
        ax.scatter(fpts[:, 0], fpts[:, 1], fpts[:, 2],
                  c=colors_list[idx], s=1, alpha=0.8)
        ax.set_title(f'{name}\n({len(fpts)} points, {retention:.1f}%)',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plot_file = RESULTS_DIR / 'point_clouds_3d.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: {plot_file}")
    return str(plot_file)


def plot_metrics(evaluations: dict) -> str:
    """Create metrics comparison plots."""
    logger.info("Generating metrics plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    filter_names = list(evaluations.keys())

    # AABB IoU
    aabb_values = [evaluations[name].get('aabb_iou', 0) for name in filter_names]
    axes[0, 0].bar(filter_names, aabb_values, color=['blue', 'purple', 'orange', 'brown'])
    axes[0, 0].set_ylabel('IoU')
    axes[0, 0].set_title('AABB IoU (Macro-scale Geometry)', fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(aabb_values):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')

    # Voxel IoU
    voxel_values = [evaluations[name].get('voxel_iou', 0) for name in filter_names]
    axes[0, 1].bar(filter_names, voxel_values, color=['blue', 'purple', 'orange', 'brown'])
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Voxel IoU (Micro-scale Detail)', fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(voxel_values):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')

    # Chamfer Distance
    chamfer_values = [evaluations[name].get('chamfer_distance_cm', 0) for name in filter_names]
    axes[1, 0].bar(filter_names, chamfer_values, color=['blue', 'purple', 'orange', 'brown'])
    axes[1, 0].set_ylabel('Distance (cm)')
    axes[1, 0].set_title('Chamfer Distance (Surface Accuracy)', fontweight='bold')
    for i, v in enumerate(chamfer_values):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center')

    # Centroid Displacement
    centroid_values = [evaluations[name].get('centroid_displacement_mm', 0)
                      for name in filter_names]
    axes[1, 1].bar(filter_names, centroid_values, color=['blue', 'purple', 'orange', 'brown'])
    axes[1, 1].set_ylabel('Displacement (mm)')
    axes[1, 1].set_title('Centroid Displacement (Stability)', fontweight='bold')
    for i, v in enumerate(centroid_values):
        axes[1, 1].text(i, v + 0.1, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plot_file = RESULTS_DIR / 'metrics_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: {plot_file}")
    return str(plot_file)


def plot_performance(benchmarks: dict) -> str:
    """Create performance comparison plots."""
    logger.info("Generating performance plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    filter_names = list(benchmarks.keys())

    # Runtime per point
    runtime_values = [benchmarks[name].get('microseconds_per_point', 0)
                     for name in filter_names]
    axes[0].bar(filter_names, runtime_values, color=['blue', 'purple', 'orange', 'brown'])
    axes[0].set_ylabel('Time (µs/point)')
    axes[0].set_title('Runtime Efficiency', fontweight='bold')
    for i, v in enumerate(runtime_values):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center')

    # Median runtime with error bars
    median_ms = [benchmarks[name].get('median_ms', 0) for name in filter_names]
    stdev_ms = [benchmarks[name].get('stdev_ms', 0) for name in filter_names]
    axes[1].bar(filter_names, median_ms, yerr=stdev_ms, capsize=5,
               color=['blue', 'purple', 'orange', 'brown'], alpha=0.7)
    axes[1].set_ylabel('Time (ms)')
    axes[1].set_title('Runtime Distribution (Median ± Stdev)', fontweight='bold')
    for i, v in enumerate(median_ms):
        axes[1].text(i, v + stdev_ms[i] + 0.2, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plot_file = RESULTS_DIR / 'performance_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: {plot_file}")
    return str(plot_file)


def plot_retention(filtered_clouds: dict, noisy_cloud: o3d.geometry.PointCloud) -> str:
    """Create point retention analysis."""
    logger.info("Generating retention plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    input_size = len(noisy_cloud.points)
    filter_names = list(filtered_clouds.keys())
    output_sizes = [len(filtered_clouds[name].points) for name in filter_names]
    retention_pcts = [(size / input_size * 100) for size in output_sizes]

    bars = ax.bar(filter_names, retention_pcts, color=['blue', 'purple', 'orange', 'brown'])

    # Add labels
    for i, (bar, pct, size) in enumerate(zip(bars, retention_pcts, output_sizes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{pct:.1f}%\n({size:,} pts)', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    ax.set_ylabel('Point Retention (%)')
    ax.set_title(f'Point Cloud Retention (Input: {input_size:,} points)', fontweight='bold')
    ax.set_ylim([0, 110])
    ax.legend()

    plt.tight_layout()
    plot_file = RESULTS_DIR / 'retention_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: {plot_file}")
    return str(plot_file)


def main():
    """Run complete test and visualization pipeline."""
    logger.info("="*70)
    logger.info("LiDAR Filters Test & Visualization")
    logger.info("="*70)

    # 1. Generate synthetic data
    noisy_cloud, gt_cloud = create_synthetic_cloud(n_points=50000, noise_ratio=0.20)

    # 2. Apply filters
    filtered_clouds = apply_all_filters(noisy_cloud)

    # 3. Benchmark
    benchmarks = benchmark_filters(noisy_cloud)

    # 4. Evaluate
    evaluations = evaluate_filters(filtered_clouds, noisy_cloud, gt_cloud)

    # 5. Visualizations
    logger.info("\n" + "="*70)
    logger.info("Generating Visualizations")
    logger.info("="*70)

    plot_paths = {}
    plot_paths['clouds'] = plot_point_clouds(noisy_cloud, filtered_clouds, gt_cloud)
    plot_paths['metrics'] = plot_metrics(evaluations)
    plot_paths['performance'] = plot_performance(benchmarks)
    plot_paths['retention'] = plot_retention(filtered_clouds, noisy_cloud)

    # 6. Summary table
    logger.info("\n" + "="*70)
    logger.info("Summary Results")
    logger.info("="*70)

    print(f"\n{'Filter':<8} {'Points':<10} {'Retention':<12} {'AABB IoU':<12} "
          f"{'Centroid':<12} {'Runtime':<12}")
    print("-" * 80)

    for name in filtered_clouds.keys():
        pts = len(filtered_clouds[name].points)
        ret = (pts / len(noisy_cloud.points)) * 100
        aabb = evaluations.get(name, {}).get('aabb_iou', 0)
        cent = evaluations.get(name, {}).get('centroid_displacement_mm', 0)
        rtime = benchmarks.get(name, {}).get('microseconds_per_point', 0)

        print(f"{name:<8} {pts:<10,} {ret:<11.1f}% {aabb:<11.4f}  "
              f"{cent:<11.2f}mm {rtime:<11.3f}µs/pt")

    logger.info(f"\n✓ All plots saved to: {RESULTS_DIR}")
    return plot_paths


if __name__ == "__main__":
    plot_paths = main()
    logger.info("\n✅ Test and visualization complete!")
