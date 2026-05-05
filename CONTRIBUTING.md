# Contributing

Thanks for taking a look. This repository is a clean research-prototype export,
so contributions should keep the publication boundary intact.

## Rules For Public Changes

- Do not commit real sensor data, ROS bags, database dumps, point clouds, or
  generated result binaries.
- Do not commit thesis drafts, presentation files, screenshots, vendor manuals,
  or executed notebooks with embedded outputs.
- Use synthetic data or user-provided local data for examples and tests.
- Keep paths relative to the repository root; avoid machine-specific paths.
- Keep claims in docs tied to reproducible code or clearly label them as
  thesis-context results.

## Development Checks

```bash
PYTHONPATH=src python -m py_compile src/lidar_snow_filter/*.py tools/*.py
PYTHONPATH=src python -m pytest tests/test_bug_fixes.py -q
PYTHONPATH=src python -m pytest tests/test_integration.py -q
PYTHONPATH=src python tests/test_reproducibility.py
```

The integration and reproducibility tests require Open3D and the scientific
Python stack from `requirements-lock.txt`.
