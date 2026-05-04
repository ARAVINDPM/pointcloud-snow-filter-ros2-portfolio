# Reproducibility Guide

This clean export is reproducible through synthetic point-cloud generation and
deterministic tests. Raw real-world thesis data is intentionally not included.

## Environment

Recommended:

```bash
python -m pip install -r requirements-lock.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate lidar-snow-filtering
```

## Synthetic Dataset

Generate two deterministic synthetic scans and snow-contaminated versions:

```bash
python synthetic_data_generator.py --num_scans 2 --seed 42 --contaminate
```

This writes generated `.pcd` files under `data/`. The `data/` directory and
point-cloud files are ignored by git.

## Tests

Fast unit test:

```bash
python -m pytest tests/test_bug_fixes.py -q
```

Open3D integration tests:

```bash
python -m pytest tests/test_integration.py -q
python tests/test_reproducibility.py
```

Example pipeline:

```bash
python tools/example_workflow.py data/synthetic_snow_scans/synthetic_mannequin_000_snow.pcd results/example/
```

## What Cannot Be Reproduced From This Export

The original thesis used private real sensor measurements. Those files are not
redistributed here, so exact real-sensor thesis metrics and plots cannot be
recreated from this repository alone.
