"""
Density Proxy vs Height-Based DSOR Comparison
Test on real Livox data to validate proxy claim
"""

import os
import numpy as np
import open3d as o3d
from pathlib import Path
import json
from datetime import datetime
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from lidar_snow_filter.filters import LiDARFilters

print("\n" + "="*80)
print("DENSITY PROXY vs HEIGHT-BASED DSOR ANALYSIS")
print("="*80)

def dsor_density_proxy(pcd, min_ratio=1.5, nb_neighbors=20, rho_max=3.0):
    """True density-proxy DSOR using local point density."""
    points = np.asarray(pcd.points)
    input_size = len(points)

    print(f"  DSOR Density-Proxy: {input_size} points")

    # Compute k-nearest neighbor distances
    distances = []
    for i in range(input_size):
        dists = np.linalg.norm(points - points[i], axis=1)
        nearest_k = np.sort(dists)[1:nb_neighbors + 1]
        distances.append(np.mean(nearest_k))

    distances = np.array(distances)
    d_bar = np.mean(distances)
    sigma = np.std(distances)

    # Compute local density for each point
    densities = []
    for i in range(input_size):
        dists = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.sum(dists < 0.1)
        density = neighbors / (4/3 * np.pi * 0.1**3)
        densities.append(density)

    densities = np.array(densities)
    density_mean = np.mean(densities)

    # Apply density-adaptive threshold
    retained_indices = []
    for i in range(input_size):
        d_i = distances[i]
        density_ratio = densities[i] / density_mean
        rho_i = np.clip(1.0 / density_ratio, 1.0, rho_max)
        threshold = d_bar + rho_i * sigma

        if d_i <= threshold:
            retained_indices.append(i)

    filtered = pcd.select_by_index(retained_indices)
    retention = len(filtered.points) / input_size * 100

    metadata = {
        "method": "DSOR_DENSITY_PROXY",
        "input_points": input_size,
        "output_points": len(filtered.points),
        "retention_pct": retention,
        "stats": {
            "d_bar": float(d_bar),
            "sigma": float(sigma),
            "density_mean": float(density_mean)
        }
    }

    return filtered, metadata


# Test on real Livox clear scans
livox_dir = Path(os.environ.get("LIVOX_CLEAR_DIR", REPO_ROOT / "data" / "private_thesis" / "1Results" / "livox" / "clear 10 scans"))
scan_files = sorted(list(livox_dir.glob("*clear*.pcd")))[:3]

print(f"\nFound {len(scan_files)} scan files for testing\n")

all_results = []
comparison_data = []

for scan_idx, scan_file in enumerate(scan_files, 1):
    print(f"\n{'='*80}")
    print(f"SCAN {scan_idx}: {scan_file.name}")
    print(f"{'='*80}")

    # Load scan
    pcd = o3d.io.read_point_cloud(str(scan_file))
    n_points = len(pcd.points)
    print(f"  Points: {n_points:,}")

    # Get points for correlation analysis
    points_array = np.asarray(pcd.points)
    heights = points_array[:, 2]

    # Compute density for correlation
    print("  Computing density for correlation analysis...")
    densities = []
    for i in range(min(1000, len(points_array))):  # Sample for speed
        dists = np.linalg.norm(points_array - points_array[i], axis=1)
        neighbors = np.sum(dists < 0.1)
        density = neighbors / (4/3 * np.pi * 0.1**3)
        densities.append(density)

    densities = np.array(densities)
    heights_sample = heights[:min(1000, len(heights))]

    correlation = np.corrcoef(heights_sample, densities)[0, 1]
    print(f"  Height-Density Correlation: {correlation:.3f}")

    # Height-based DSOR
    print("  Running Height-Based DSOR...")
    try:
        _, meta_h = LiDARFilters.dsor(pcd, min_ratio=1.5, sector_count=8)
        ret_height = meta_h['retention_pct']
        print(f"    ✓ Retention: {ret_height:.2f}%")
    except Exception as e:
        ret_height = None
        print(f"    ✗ Error: {e}")

    # Density-proxy DSOR
    print("  Running Density-Proxy DSOR...")
    try:
        _, meta_d = dsor_density_proxy(pcd)
        ret_density = meta_d['retention_pct']
        print(f"    ✓ Retention: {ret_density:.2f}%")
    except Exception as e:
        ret_density = None
        print(f"    ✗ Error: {e}")

    # Compare
    if ret_height is not None and ret_density is not None:
        diff = ret_density - ret_height
        print("\n  COMPARISON:")
        print(f"    Height-Based:   {ret_height:6.2f}%")
        print(f"    Density-Proxy:  {ret_density:6.2f}%")
        print(f"    Difference:     {diff:+6.2f}%")

        comparison_data.append({
            "scan": scan_file.name,
            "points": n_points,
            "correlation": float(correlation),
            "height_retention": ret_height,
            "density_retention": ret_density,
            "difference": diff
        })

# Analysis Summary
print(f"\n{'='*80}")
print("CONSISTENCY ANALYSIS ACROSS SCANS")
print(f"{'='*80}\n")

if comparison_data:
    import pandas as pd
    df = pd.DataFrame(comparison_data)

    print(df[['scan', 'points', 'correlation', 'height_retention', 'density_retention', 'difference']].to_string(index=False))

    print("\nSTATISTICS:")
    print(f"  Avg Height-Based Retention:  {df['height_retention'].mean():.2f}%")
    print(f"  Avg Density-Proxy Retention: {df['density_retention'].mean():.2f}%")
    print(f"  Avg Difference:              {df['difference'].mean():+.2f}%")
    print(f"  Std Dev Difference:          {df['difference'].std():.2f}%")
    print(f"  Avg Height-Density Corr:     {df['correlation'].mean():.3f}")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")

    avg_diff = df['difference'].mean()
    avg_corr = df['correlation'].mean()

    print(f"Height-Density Correlation: {avg_corr:.3f}")
    if abs(avg_corr) > 0.7:
        print("  ✅ STRONG: Height is a valid proxy for density")
    elif abs(avg_corr) > 0.5:
        print("  ⚠️  MODERATE: Height provides reasonable proxy")
    else:
        print("  ❌ WEAK: Height is NOT a good density proxy")

    print(f"\nRetention Difference: {avg_diff:+.2f}%")
    if abs(avg_diff) < 2:
        print("  ✅ NEGLIGIBLE: Both methods are equivalent")
        print("  → Height-based IS a valid practical simplification of density-aware")
    elif abs(avg_diff) < 5:
        print("  ⚠️  MODERATE: Some difference in behavior")
        print("  → Height-based and density-aware are different approaches")
    else:
        print("  ❌ SIGNIFICANT: Methods behave very differently")
        print("  → Height-based does NOT approximate density-aware")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION FOR GITHUB/PUBLICATION")
    print(f"{'='*80}\n")

    if abs(avg_corr) > 0.6 and abs(avg_diff) < 3:
        print("CLAIM: Height-aware sectioning as practical density proxy")
        print(f"Evidence: {abs(avg_corr):.3f} correlation, {avg_diff:+.2f}% difference")
        print("Position: 'We use height-based partitioning as an efficient proxy for")
        print("          range-dependent density, leveraging LiDAR's natural sampling pattern'")
    else:
        print("CLAIM: Alternative sectoring strategy (not a density proxy)")
        print(f"Evidence: {abs(avg_corr):.3f} correlation, {avg_diff:+.2f}% difference")
        print("Position: 'We propose height-based partitioning as an alternative to")
        print("          thesis density-aware approach, with different trade-offs'")

    # Save results
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "Density-Proxy vs Height-Based DSOR on Real Livox Data",
        "comparison_data": comparison_data,
        "statistics": {
            "avg_height_retention": float(df['height_retention'].mean()),
            "avg_density_retention": float(df['density_retention'].mean()),
            "avg_difference": float(df['difference'].mean()),
            "std_difference": float(df['difference'].std()),
            "avg_correlation": float(df['correlation'].mean())
        }
    }

    output_dir = REPO_ROOT / "results" / "density_vs_height"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "density_vs_height_results.json"

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Results saved: {output_file}")
else:
    print("✗ No comparison data collected. Check errors above.")

print(f"\n{'='*80}\n")
