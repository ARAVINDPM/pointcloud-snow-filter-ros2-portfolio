# LiDAR Snow Filtering Research Prototype

Clean portfolio export of a master's thesis project on classical point-cloud
outlier filtering for snow-contaminated LiDAR scans.

This repository is intentionally a clean, fresh-history export. It contains the
reusable Python code, synthetic-data pipeline, tests, and documentation needed to
review the technical approach. Raw thesis sensor data, notebooks, vendor
documents, binary result files, and the original private git history are not
included.

## Scope

- Python filtering library built on Open3D; no ROS2 runtime dependency.
- Baseline filters: Statistical Outlier Removal (SOR) and Radius Outlier
  Removal (ROR).
- Project-specific adaptive variants: height-adaptive SOR and
  azimuth-adaptive ROR.
- Synthetic mannequin point-cloud generation and reproducible snow-noise tests.
- Geometry and stability metrics for comparing filtered point clouds.

The repository name keeps `ros2` because the original data collection context
used ROS2. The code in this export can run without ROS2 once the Python/Open3D
environment is installed. It is tested for Python 3.10 with Open3D 0.17.0.

## Public/Private Boundary

Included:

- `filters.py`, `metrics.py`, `benchmarking.py`, `config.py`
- `synthetic_data_generator.py`, `contaminate_with_synthetic_snow.py`
- Selected tools under `tools/`
- Tests under `tests/`
- Packaging and environment files

Excluded:

- Real SICK/Livox thesis point clouds and ROS bag/database files
- Thesis notebooks and executed notebook outputs
- Thesis draft/PDF, presentation files, screenshots, generated plots
- Vendor manuals and any third-party documents without redistribution review
- Original private git history

Real-sensor results from the thesis are not independently reproducible from this
clean export because the underlying measurement data is not redistributed.

## Quick Start

```bash
git clone https://github.com/ARAVINDPM/pointcloud-snow-filter-ros2-portfolio.git
cd pointcloud-snow-filter-ros2-portfolio
python -m pip install -r requirements-lock.txt
```

Generate synthetic clear and snow-contaminated point clouds:

```bash
python synthetic_data_generator.py --num_scans 2 --seed 42 --contaminate
```

Run the lightweight unit tests:

```bash
python -m unittest tests.test_bug_fixes -v
```

If `pytest`, Open3D, and the scientific stack are available, run:

```bash
python -m pytest tests/test_bug_fixes.py -q
python -m pytest tests/test_integration.py -q
python tests/test_reproducibility.py
```

Run the example pipeline on generated synthetic data:

```bash
python tools/example_workflow.py data/synthetic_snow_scans/synthetic_mannequin_000_snow.pcd results/example/
```

## Main Files

| File | Purpose |
| --- | --- |
| `filters.py` | SOR, ROR, height-adaptive SOR, and azimuth-adaptive ROR implementations |
| `metrics.py` | AABB IoU, voxel IoU, Chamfer distance, centroid displacement, and retention metrics |
| `benchmarking.py` | Repeatable runtime measurement helpers |
| `synthetic_data_generator.py` | Synthetic mannequin and snow-contamination generator |
| `tools/example_workflow.py` | End-to-end filtering and metric export on a `.pcd` file |
| `tests/` | Lightweight unit tests plus Open3D integration/reproducibility tests |

## Notes For Reviewers

This is best read as a research prototype and technical portfolio artifact, not
as a production perception stack. The code is useful for evaluating classical
filter behavior, reproducibility, and trade-offs between geometry preservation,
point retention, and runtime on synthetic point clouds.

## Technical Decisions And Trade-Offs

- Classical filters were used because they are explainable, parameter-light, and
  suitable for inspecting geometry-preservation trade-offs.
- Synthetic data is included so reviewers can reproduce the mechanics without
  access to private thesis measurements.
- Real-sensor validation belongs to the thesis context and is intentionally kept
  outside this export.
- Runtime, retention, and geometry metrics are treated as competing signals; the
  repo is not claiming one universal best filter.
