# Difference Report

This report summarizes the publication boundary for this clean portfolio
export.

## Repository Scope

This export keeps only the reusable code, synthetic-data pipeline, tests, and
review-oriented documentation. It intentionally excludes private working files,
source datasets, and previous git history.

| Item | Clean portfolio export |
| --- | --- |
| Source scope | Reusable Python code, tests, and docs |
| Data scope | Synthetic generation only |
| High-risk tracked artifacts | None found |
| History scope | Clean export history only |

High-risk patterns checked: `.pcd`, `.npy`, `.db3`, `.bag`, `.xlsx`, `.pdf`,
`.jpg`, `.jpeg`, `.png`, `.ipynb`, `.pptx`, `.docx`, and `.DS_Store`.

## Removed From This Export

- Previous `.git` history
- Private thesis working files
- Real SICK/Livox point clouds and ROS bag/database files
- Thesis notebooks and executed notebook outputs
- Thesis draft/PDF, presentation files, reports, screenshots, generated plots
- Vendor manuals and third-party documents
- Non-project planning notes
- Scripts that depend directly on private data paths

## Kept In This Export

- Core filtering code: `filters.py`
- Metrics and benchmarking helpers: `metrics.py`, `benchmarking.py`
- Synthetic data generation and contamination scripts
- Selected tools that work with synthetic or user-provided local `.pcd` files
- Unit, integration, and reproducibility tests
- Packaging, environment, license, citation, and contribution files
- Documentation focused on reproducible synthetic validation

## Verification Summary

Commands run locally on the clean export:

```bash
git ls-files '*.pcd' '*.npy' '*.db3' '*.bag' '*.xlsx' '*.pdf' '*.jpg' '*.jpeg' '*.png' '*.ipynb' '*.pptx' '*.docx' '.DS_Store'
python3 -m py_compile filters.py metrics.py benchmarking.py config.py synthetic_data_generator.py contaminate_with_synthetic_snow.py tools/example_workflow.py tools/test_and_visualize.py tools/evaluate_all_frames.py tools/visualize_and_animate.py
python -m unittest tests.test_bug_fixes -v
```

Results:

- No high-risk data/binary/notebook files were tracked.
- A text scan found no machine-home paths, placeholder email/DOI values,
  credential-like markers, unsupported production wording, private data paths,
  or ignored-CI markers.
- Python syntax compilation passed.
- Lightweight unit tests passed: 12 tests.

Open3D integration tests were retained but not run in this local environment
because Open3D is not installed here.
