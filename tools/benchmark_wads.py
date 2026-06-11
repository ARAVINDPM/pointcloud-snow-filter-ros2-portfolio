#!/usr/bin/env python3
"""Benchmark the filters on WADS (Winter Adverse Driving dataSet) frames.

WADS provides real snowfall LiDAR scans with per-point labels
(https://digitalcommons.mtu.edu/wads/). It is several GB and not
redistributed here: download frames yourself, then point this script at a
directory of .pcd/.bin files.

Snow label IDs follow the WADS/SemanticKITTI convention:
110 = falling snow, 111 = accumulated snow.

Usage:
    python tools/benchmark_wads.py /path/to/wads/velodyne [--labels /path/to/labels]

Without labels the script reports retention + runtime per filter.
With labels it additionally reports snow removal precision/recall.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lidar_snow_filter.filters import LiDARFilters  # noqa: E402

SNOW_LABELS = {110, 111}


def load_frame(path: Path) -> np.ndarray:
    if path.suffix == ".bin":  # KITTI-style float32 x,y,z,intensity
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def load_labels(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32) & 0xFFFF


def evaluate(xyz: np.ndarray, labels, filter_name: str) -> dict:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    t0 = time.perf_counter()
    filtered, meta = getattr(LiDARFilters, filter_name)(pcd)
    dt = time.perf_counter() - t0

    out = {
        "filter": filter_name.upper(),
        "input_points": len(xyz),
        "kept_points": len(filtered.points),
        "retention_pct": meta["retention_pct"],
        "us_per_point": dt / len(xyz) * 1e6,
    }
    if labels is not None:
        # Recover kept indices by exact coordinate match on the original order
        kept = np.zeros(len(xyz), dtype=bool)
        kept_pts = np.asarray(filtered.points)
        # Filters select by index, so use a set of row hashes for robustness
        view = {pt.tobytes(): i for i, pt in enumerate(xyz)}
        for pt in kept_pts:
            idx = view.get(pt.tobytes())
            if idx is not None:
                kept[idx] = True
        is_snow = np.isin(labels[: len(xyz)], list(SNOW_LABELS))
        removed = ~kept
        tp = int(np.sum(removed & is_snow))
        fp = int(np.sum(removed & ~is_snow))
        fn = int(np.sum(kept & is_snow))
        out["snow_removal_precision"] = tp / max(1, tp + fp)
        out["snow_removal_recall"] = tp / max(1, tp + fn)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", help="Directory of WADS .pcd or .bin frames")
    parser.add_argument("--labels", default=None, help="Directory of .label files")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()

    frames = sorted(
        list(Path(args.frames_dir).glob("*.pcd")) + list(Path(args.frames_dir).glob("*.bin"))
    )[: args.max_frames]
    if not frames:
        sys.exit(f"No .pcd/.bin frames found in {args.frames_dir}")

    rows = []
    for frame in frames:
        xyz = load_frame(frame)
        labels = None
        if args.labels:
            label_file = Path(args.labels) / (frame.stem + ".label")
            if label_file.exists():
                labels = load_labels(label_file)
        for name in ("sor", "ror", "dsor", "dror"):
            rows.append({"frame": frame.name, **evaluate(xyz, labels, name)})

    # Aggregate
    print(f"\n{'filter':6} {'retention%':>10} {'us/pt':>8}", end="")
    has_labels = any("snow_removal_recall" in r for r in rows)
    if has_labels:
        print(f" {'precision':>10} {'recall':>8}")
    else:
        print()
    for name in ("SOR", "ROR", "DSOR", "DROR"):
        sub = [r for r in rows if r["filter"] == name]
        ret = np.mean([r["retention_pct"] for r in sub])
        us = np.mean([r["us_per_point"] for r in sub])
        line = f"{name:6} {ret:10.1f} {us:8.2f}"
        if has_labels:
            line += (f" {np.mean([r['snow_removal_precision'] for r in sub]):10.2f}"
                     f" {np.mean([r['snow_removal_recall'] for r in sub]):8.2f}")
        print(line)


if __name__ == "__main__":
    main()
