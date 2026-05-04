"""
Configuration and path management for LiDAR Snow Filtering project.

This module centralizes all path management, making the repo reproducible
without hardcoded paths.
"""

from pathlib import Path
import os

# ============================================================================
# PROJECT ROOT
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()

# ============================================================================
# DATA DIRECTORIES (for local testing; paths should be set relative to repo)
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Subdirectories for different dataset types
SNOW_DATA_DIR = DATA_DIR / "snow_scans"
CLEAR_DATA_DIR = DATA_DIR / "clear_scans"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories as needed
SNOW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEAR_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SENSOR DATA SAMPLES (example paths - users should provide their own)
# ============================================================================
# To use: Download your PCD files and place them in data/snow_scans or data/clear_scans
# Then reference them as: config.SNOW_DATA_DIR / "your_file.pcd"

EXAMPLE_SNOW_PCD = SNOW_DATA_DIR / "snow_cloud_sample.pcd"
EXAMPLE_CLEAR_PCD = CLEAR_DATA_DIR / "clear_cloud_sample.pcd"

# ============================================================================
# OUTPUT & RESULTS
# ============================================================================
BENCHMARK_RESULTS = RESULTS_DIR / "benchmark_results.csv"
EVALUATION_METRICS = RESULTS_DIR / "evaluation_metrics.json"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# FILTER PARAMETERS (defaults used for synthetic/public examples)
# ============================================================================
# Statistical Outlier Removal (SOR)
# Checks: for each point, are there ≥N neighbors within 1 std dev?
SOR_NB_NEIGHBORS = 20          # Typical density for LiDAR scans (tuned on real data)
SOR_STD_RATIO = 2.0            # Points >2σ from neighborhood mean are outliers

# Radius Outlier Removal (ROR)
# Checks: does each point have ≥N points within radius R?
ROR_NB_POINTS = 5              # Minimum cluster membership
ROR_RADIUS = 0.05              # Search radius in meters (matches sensor noise margin)

# Height-Adaptive Statistical Outlier Removal (HA-DSOR)
# Project-specific variant: applies different std_ratio thresholds per altitude zone
DSOR_MIN_RATIO = 1.5           # Base ratio for lowest zone (density adaptation)
DSOR_SECTOR_COUNT = 8          # Number of height sectors (0-130cm range ÷ 8 ≈ 16cm each)

# Azimuth-Adaptive Radius Outlier Removal (HA-DROR)
# Project-specific variant: scales ROR radius per azimuthal sector based on local density
DROR_SECTOR_COUNT = 12         # Number of azimuthal sectors (360° / 12 = 30° per sector)
DROR_SCALE_FACTOR = 1.5        # Multiplier for computed adaptive radius

# ============================================================================
# TIMING & BENCHMARKING
# ============================================================================
TIMEIT_REPEAT = 1000  # Number of repeats for %timeit
TIMEIT_LOOPS = 3      # Loops per repeat (keep 1 for large point clouds)

# ============================================================================
# HELPER FUNCTION: GET DATA PATH
# ============================================================================
def get_data_path(filename: str, data_type: str = "snow") -> Path:
    """
    Resolve a data filename to its full path.

    Args:
        filename: Name of the file (e.g., "cloud_output.pcd")
        data_type: Either "snow" or "clear"

    Returns:
        Path object pointing to the file

    Example:
        >>> cloud_path = get_data_path("snow_cloud_feb14.pcd", "snow")
    """
    if data_type == "snow":
        return SNOW_DATA_DIR / filename
    elif data_type == "clear":
        return CLEAR_DATA_DIR / filename
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Use 'snow' or 'clear'.")


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
