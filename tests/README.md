# Test Suite

The public test suite focuses on synthetic and deterministic validation.

## Fast Checks

```bash
PYTHONPATH=src python -m py_compile src/lidar_snow_filter/*.py tools/*.py
PYTHONPATH=src python -m unittest tests.test_bug_fixes -v
```

## Integration Checks

These require Open3D and the scientific stack:

```bash
PYTHONPATH=src python -m pytest tests/test_bug_fixes.py -q
PYTHONPATH=src python -m pytest tests/test_integration.py -q
PYTHONPATH=src python tests/test_reproducibility.py
```

The tests do not require private thesis data. Generated `data/` and `results/`
outputs are ignored by git.
