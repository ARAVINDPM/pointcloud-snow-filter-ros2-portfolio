# Test Suite

The public test suite focuses on synthetic and deterministic validation.

## Fast Checks

```bash
python -m py_compile filters.py metrics.py benchmarking.py synthetic_data_generator.py
python -m unittest tests.test_bug_fixes -v
```

## Integration Checks

These require Open3D and the scientific stack:

```bash
python -m pytest tests/test_bug_fixes.py -q
python -m pytest tests/test_integration.py -q
python tests/test_reproducibility.py
```

The tests do not require private thesis data. Generated `data/` and `results/`
outputs are ignored by git.
