"""
Synthetic Snow Contamination Generator.

Adds realistic snow noise patterns to clean LiDAR scans for testing filter robustness.

Usage:
    python -m lidar_snow_filter.contaminate_with_synthetic_snow \
        --input_dir data/synthetic_clear_scans \
        --output_dir data/synthetic_snow_scans \
        --snow_density 0.3
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import logging
import argparse
import json
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealisticSnowSimulator:
    """Simulate realistic snow noise patterns."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def add_snow_noise(
        self,
        pcd: o3d.geometry.PointCloud,
        snow_density: float = 0.25,
        noise_std: float = 0.8,
        max_distance: float = 25.0,
    ) -> o3d.geometry.PointCloud:
        """
        Add realistic snow contamination to point cloud.

        Snow is modeled as:
        1. Random outliers in surrounding region
        2. Gaussian noise on existing points
        3. Occasional clustering (snowflakes)

        Args:
            pcd: Clean point cloud
            snow_density: Ratio of outliers to original points
            noise_std: Standard deviation of Gaussian noise (cm)
            max_distance: Maximum distance of outliers from object center (cm)

        Returns:
            Contaminated point cloud
        """
        points = np.asarray(pcd.points)
        n_points = len(points)

        # Calculate object center
        center = np.mean(points, axis=0)

        # 1. Add random outlier points (sparse snow)
        n_outliers = int(n_points * snow_density)

        # Generate outliers in spherical region around object
        radii = np.random.uniform(0, max_distance, n_outliers)
        theta = np.random.uniform(0, 2 * np.pi, n_outliers)
        phi = np.random.uniform(0, np.pi, n_outliers)

        outliers_x = radii * np.sin(phi) * np.cos(theta) + center[0]
        outliers_y = radii * np.sin(phi) * np.sin(theta) + center[1]
        outliers_z = radii * np.cos(phi) + center[2]

        outliers = np.column_stack([outliers_x, outliers_y, outliers_z])

        # 2. Add Gaussian noise to existing points (surface corruption)
        point_noise = np.random.normal(0, noise_std, points.shape)
        noisy_points = points + point_noise

        # 3. Add occasional snow clusters
        n_clusters = int(n_outliers * 0.1)
        cluster_points = []
        for _ in range(n_clusters):
            # Random cluster center
            cx = np.random.uniform(center[0] - max_distance, center[0] + max_distance)
            cy = np.random.uniform(center[1] - max_distance, center[1] + max_distance)
            cz = np.random.uniform(center[2], center[2] + max_distance)

            # Dense cluster (10-50 points)
            n_cluster = np.random.randint(10, 50)
            cluster = np.random.normal([cx, cy, cz], 2.0, (n_cluster, 3))
            cluster_points.append(cluster)

        if cluster_points:
            cluster_points = np.vstack(cluster_points)
        else:
            cluster_points = np.empty((0, 3))

        # Combine all points
        all_points = np.vstack([noisy_points, outliers, cluster_points])

        # Create output point cloud
        contaminated_pcd = o3d.geometry.PointCloud()
        contaminated_pcd.points = o3d.utility.Vector3dVector(all_points)

        return contaminated_pcd

    def add_sensor_artifacts(
        self,
        pcd: o3d.geometry.PointCloud,
        dropout_rate: float = 0.02,
    ) -> o3d.geometry.PointCloud:
        """
        Add sensor artifacts: dropout (missing returns), multipath.

        Args:
            pcd: Point cloud
            dropout_rate: Fraction of points to randomly drop

        Returns:
            Point cloud with artifacts
        """
        points = np.asarray(pcd.points)
        n_points = len(points)

        # Random dropout
        keep_mask = np.random.uniform(0, 1, n_points) > dropout_rate
        filtered_points = points[keep_mask]

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

        return pcd_filtered


def process_directory(
    input_dir: str,
    output_dir: str,
    snow_density: float = 0.25,
    seed: int = 42,
) -> None:
    """
    Process all PCD files in input directory.

    Args:
        input_dir: Directory containing clean PCD files
        output_dir: Where to save contaminated PCDs
        snow_density: Snow contamination level
        seed: Random seed
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator = RealisticSnowSimulator(seed=seed)

    pcd_files = sorted(input_path.glob("*.pcd"))
    if not pcd_files:
        logger.warning(f"No PCD files found in {input_dir}")
        return

    logger.info(f"Found {len(pcd_files)} PCD files")

    results = []
    for i, pcd_file in enumerate(pcd_files):
        logger.info(f"Processing {i+1}/{len(pcd_files)}: {pcd_file.name}")

        # Load clean point cloud
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        n_clean = len(pcd.points)

        # Add snow contamination
        contaminated = simulator.add_snow_noise(pcd, snow_density=snow_density)

        # Add sensor artifacts
        contaminated = simulator.add_sensor_artifacts(contaminated, dropout_rate=0.02)

        n_contaminated = len(contaminated.points)

        # Save
        output_file = output_path / pcd_file.name.replace(".pcd", "_snow.pcd")
        o3d.io.write_point_cloud(str(output_file), contaminated)

        logger.info(f"  Clean: {n_clean} points → Contaminated: {n_contaminated} points")

        results.append({
            "clean_file": pcd_file.name,
            "snow_file": output_file.name,
            "clean_points": n_clean,
            "snow_points": n_contaminated,
            "added_points": n_contaminated - n_clean,
            "snow_density": snow_density,
        })

    # Save metadata
    metadata = {
        "source_dir": str(input_path),
        "output_dir": str(output_path),
        "snow_density": snow_density,
        "seed": seed,
        "total_processed": len(results),
        "results": results,
    }

    metadata_file = output_path / "contamination_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Contamination complete! Processed {len(results)} files")
    logger.info(f"Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Add synthetic snow to LiDAR scans")
    parser.add_argument("--input_dir", required=True, help="Input directory with clean PCD files")
    parser.add_argument("--output_dir", required=True, help="Output directory for contaminated PCDs")
    parser.add_argument("--snow_density", type=float, default=0.25, help="Snow contamination level (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir,
        snow_density=args.snow_density,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
