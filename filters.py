"""
LiDAR point-cloud filtering implementations for research prototypes.

This module provides error-handling implementations of four snow-filtering
methods for evaluating geometry-preservation trade-offs.

Methods:
    - SOR (Statistical Outlier Removal)
    - ROR (Radius Outlier Removal)
    - HA-DSOR (Height-Adaptive Statistical Outlier Removal)
    - HA-DROR (Azimuth-Adaptive Radius Outlier Removal)

Height-Adaptive (HA) variants are project-specific adaptive formulations. They
partition the cloud spatially and apply per-sector thresholds to handle varying
point density across altitude and azimuth.
"""

import logging
import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PointCloudValidator:
    """Validates point cloud integrity and properties."""

    @staticmethod
    def validate_input(pcd: o3d.geometry.PointCloud, name: str = "input") -> bool:
        """
        Validate point cloud properties.

        Args:
            pcd: Open3D point cloud object
            name: Name for logging

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If cloud is empty or invalid
        """
        if not isinstance(pcd, o3d.geometry.PointCloud):
            raise TypeError(f"{name} must be o3d.geometry.PointCloud")

        n_points = len(pcd.points)
        if n_points == 0:
            raise ValueError(f"{name} is empty (0 points)")

        # Check for NaNs
        points_array = np.asarray(pcd.points)
        if np.isnan(points_array).any():
            raise ValueError(f"{name} contains NaN values")

        logger.info(f"{name}: {n_points} points, bounds: {pcd.get_axis_aligned_bounding_box()}")
        return True

    @staticmethod
    def validate_output(pcd: o3d.geometry.PointCloud,
                       input_size: int,
                       name: str = "output") -> None:
        """
        Validate output cloud properties.

        Args:
            pcd: Output point cloud
            input_size: Original point count
            name: Name for logging

        Raises:
            ValueError: If output is invalid
        """
        output_size = len(pcd.points)

        if output_size == 0:
            logger.warning(f"{name}: All points filtered out!")
            return

        if output_size > input_size:
            raise ValueError(f"{name}: Created {output_size} points from {input_size} input")

        retention = (output_size / input_size) * 100
        logger.info(f"{name}: {output_size} points, retention: {retention:.1f}%")


class LiDARFilters:
    """Production-grade implementations of snow-filtering methods."""

    @staticmethod
    def sor(pcd: o3d.geometry.PointCloud,
            nb_neighbors: int = 20,
            std_ratio: float = 2.0) -> Tuple[o3d.geometry.PointCloud, dict]:
        """
        Statistical Outlier Removal (SOR).

        Global kNN-based method: removes points whose distance to kth nearest
        neighbor exceeds mean + std_ratio * stddev of all distances.

        Args:
            pcd: Input point cloud
            nb_neighbors: Number of nearest neighbors to consider
            std_ratio: Standard deviation threshold multiplier

        Returns:
            (filtered_cloud, metadata) tuple

        Raises:
            ValueError: If cloud is invalid or parameters out of range
        """
        PointCloudValidator.validate_input(pcd, "SOR input")

        if nb_neighbors < 2:
            raise ValueError(f"nb_neighbors must be >= 2, got {nb_neighbors}")
        if std_ratio <= 0:
            raise ValueError(f"std_ratio must be > 0, got {std_ratio}")

        input_size = len(pcd.points)
        logger.info(f"SOR: Starting with {input_size} points")

        try:
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            filtered = pcd.select_by_index(ind)
            PointCloudValidator.validate_output(filtered, input_size, "SOR output")

            metadata = {
                "method": "SOR",
                "input_points": input_size,
                "output_points": len(filtered.points),
                "retention_pct": (len(filtered.points) / input_size * 100),
                "parameters": {"nb_neighbors": nb_neighbors, "std_ratio": std_ratio}
            }
            return filtered, metadata

        except Exception as e:
            logger.error(f"SOR failed: {e}")
            raise

    @staticmethod
    def ror(pcd: o3d.geometry.PointCloud,
            nb_points: int = 5,
            radius: float = 0.05) -> Tuple[o3d.geometry.PointCloud, dict]:
        """
        Radius Outlier Removal (ROR).

        Fixed-radius neighborhood method: removes points with fewer than
        nb_points neighbors within radius.

        Args:
            pcd: Input point cloud
            nb_points: Minimum neighbors required within radius
            radius: Search radius (meters)

        Returns:
            (filtered_cloud, metadata) tuple

        Raises:
            ValueError: If cloud is invalid or parameters out of range
        """
        PointCloudValidator.validate_input(pcd, "ROR input")

        if nb_points < 1:
            raise ValueError(f"nb_points must be >= 1, got {nb_points}")
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        input_size = len(pcd.points)
        logger.info(f"ROR: Starting with {input_size} points")

        try:
            cl, ind = pcd.remove_radius_outlier(
                nb_points=nb_points,
                radius=radius
            )
            filtered = pcd.select_by_index(ind)
            PointCloudValidator.validate_output(filtered, input_size, "ROR output")

            metadata = {
                "method": "ROR",
                "input_points": input_size,
                "output_points": len(filtered.points),
                "retention_pct": (len(filtered.points) / input_size * 100),
                "parameters": {"nb_points": nb_points, "radius": radius}
            }
            return filtered, metadata

        except Exception as e:
            logger.error(f"ROR failed: {e}")
            raise

    @staticmethod
    def dsor(pcd: o3d.geometry.PointCloud,
             min_ratio: float = 1.5,
             sector_count: int = 8) -> Tuple[o3d.geometry.PointCloud, dict]:
        """
        Height-Adaptive Statistical Outlier Removal (HA-DSOR).

        Project-specific variant: partitions point cloud into N height zones and applies
        SOR with per-zone adaptive std_ratio. Addresses varying point density
        across altitude (denser near ground, sparser aloft).

        Formula: σ_threshold[i] = min_ratio + (i / sector_count)
        where i = zone index (0 = lowest altitude)

        This is distinct from Charron 2018's range-adaptive ROR, which scales
        by distance-from-sensor instead of altitude.

        Args:
            pcd: Input point cloud
            min_ratio: Base std_ratio for lowest zone (tuned to ~1.5 for LiDAR)
            sector_count: Number of height sectors (tuned to 8 for ~10-130cm range)

        Returns:
            (filtered_cloud, metadata) tuple
        """
        PointCloudValidator.validate_input(pcd, "DSOR input")

        if sector_count < 1:
            raise ValueError(f"sector_count must be >= 1, got {sector_count}")

        points = np.asarray(pcd.points)
        input_size = len(points)
        logger.info(f"DSOR: Starting with {input_size} points, {sector_count} height sectors")

        try:
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            z_bins = np.linspace(z_min, z_max, sector_count + 1)

            all_indices = []
            for i in range(sector_count):
                z_low, z_high = z_bins[i], z_bins[i + 1]
                if i == sector_count - 1:
                    mask = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
                else:
                    mask = (points[:, 2] >= z_low) & (points[:, 2] < z_high)
                sector_indices = np.where(mask)[0]

                if len(sector_indices) < 5:
                    all_indices.extend(sector_indices)
                    continue

                sector_cloud = pcd.select_by_index(sector_indices)
                adaptive_ratio = min_ratio + (i / sector_count)

                cl, ind = sector_cloud.remove_statistical_outlier(
                    nb_neighbors=20,
                    std_ratio=adaptive_ratio
                )
                sector_filtered_indices = sector_indices[ind]
                all_indices.extend(sector_filtered_indices)

            filtered = pcd.select_by_index(sorted(all_indices))
            PointCloudValidator.validate_output(filtered, input_size, "DSOR output")

            metadata = {
                "method": "DSOR",
                "input_points": input_size,
                "output_points": len(filtered.points),
                "retention_pct": (len(filtered.points) / input_size * 100),
                "parameters": {"min_ratio": min_ratio, "sector_count": sector_count}
            }
            return filtered, metadata

        except Exception as e:
            logger.error(f"DSOR failed: {e}")
            raise

    @staticmethod
    def dror(pcd: o3d.geometry.PointCloud,
             sector_count: int = 12,
             scale_factor: float = 1.5) -> Tuple[o3d.geometry.PointCloud, dict]:
        """
        Azimuth-Adaptive Radius Outlier Removal (HA-DROR).

        Project-specific variant: partitions point cloud into N azimuthal sectors and applies
        ROR with per-sector adaptive radius. Handles varying point density across
        horizontal scanning patterns (sparse regions need larger search radii).

        Each sector computes: radius = 0.05 * (density / mean_density)^(-1/3)
        Then applies: ROR with radius *= scale_factor

        This is distinct from Charron 2018's range-adaptive ROR, which scales
        by distance-from-sensor. HA-DROR scales by local azimuthal density.

        Args:
            pcd: Input point cloud
            sector_count: Number of azimuthal sectors (tuned to 12 for 360° / 30° each)
            scale_factor: Multiplier for adaptive radius (tuned to 1.5 for snow filtering)

        Returns:
            (filtered_cloud, metadata) tuple
        """
        PointCloudValidator.validate_input(pcd, "DROR input")

        if sector_count < 4:
            raise ValueError(f"sector_count must be >= 4, got {sector_count}")

        points = np.asarray(pcd.points)
        input_size = len(points)
        logger.info(f"DROR: Starting with {input_size} points, {sector_count} azimuth sectors")

        try:
            # Compute azimuth angles
            azimuths = np.arctan2(points[:, 1], points[:, 0])
            azimuth_bins = np.linspace(-np.pi, np.pi, sector_count + 1)

            all_indices = []
            for i in range(sector_count):
                az_low, az_high = azimuth_bins[i], azimuth_bins[i + 1]
                mask = (azimuths >= az_low) & (azimuths < az_high)
                sector_indices = np.where(mask)[0]

                if len(sector_indices) < 5:
                    all_indices.extend(sector_indices)
                    continue

                sector_cloud = pcd.select_by_index(sector_indices)
                sector_points = points[sector_indices]

                # Adaptive radius based on local density
                density = len(sector_indices) / max(1, (az_high - az_low))
                radius = 0.05 * (density / (input_size / sector_count)) ** (-1 / 3)
                radius = np.clip(radius, 0.02, 0.15)

                cl, ind = sector_cloud.remove_radius_outlier(
                    nb_points=5,
                    radius=radius * scale_factor
                )
                sector_filtered_indices = sector_indices[ind]
                all_indices.extend(sector_filtered_indices)

            filtered = pcd.select_by_index(sorted(all_indices))
            PointCloudValidator.validate_output(filtered, input_size, "DROR output")

            metadata = {
                "method": "DROR",
                "input_points": input_size,
                "output_points": len(filtered.points),
                "retention_pct": (len(filtered.points) / input_size * 100),
                "parameters": {"sector_count": sector_count, "scale_factor": scale_factor}
            }
            return filtered, metadata

        except Exception as e:
            logger.error(f"DROR failed: {e}")
            raise


# Convenience functions
def load_and_filter(input_path: str,
                   method: str = "sor",
                   **kwargs) -> Tuple[o3d.geometry.PointCloud, dict]:
    """
    Load PCD file and apply filtering in one call.

    Args:
        input_path: Path to input PCD file
        method: Filter method ("sor", "ror", "dsor", "dror")
        **kwargs: Method-specific parameters

    Returns:
        (filtered_cloud, metadata) tuple
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"PCD file not found: {input_path}")

    logger.info(f"Loading: {input_path}")
    pcd = o3d.io.read_point_cloud(str(input_path))

    if method.lower() == "sor":
        return LiDARFilters.sor(pcd, **kwargs)
    elif method.lower() == "ror":
        return LiDARFilters.ror(pcd, **kwargs)
    elif method.lower() == "dsor":
        return LiDARFilters.dsor(pcd, **kwargs)
    elif method.lower() == "dror":
        return LiDARFilters.dror(pcd, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Example usage (requires a valid PCD file)
    logger.info("LiDAR Filters module loaded successfully")
