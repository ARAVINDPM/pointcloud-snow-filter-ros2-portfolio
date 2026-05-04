# Difference Report

This report compares the original private thesis working repository with this
clean portfolio export.

## Repository Scope

| Item | Private thesis repo | Clean portfolio export |
| --- | ---: | ---: |
| Tracked files | 330 | 27 |
| Files under `thesis/` | 268 | 0 |
| Tracked high-risk data/binary/notebook files | 265 | 0 |
| Git object store | 111.46 MiB packed | 46.56 KiB packed |
| Git history | Full thesis working history | Fresh single commit |
| Remote visibility target | Private source repo | Private portfolio repo |

High-risk patterns checked: `.pcd`, `.npy`, `.db3`, `.bag`, `.xlsx`, `.pdf`,
`.jpg`, `.jpeg`, `.png`, `.ipynb`, `.pptx`, `.docx`, and `.DS_Store`.

## Removed From This Export

- Original `.git` history
- Entire `thesis/` directory
- Real SICK/Livox point clouds and ROS bag/database files
- Thesis notebooks and executed notebook outputs
- Thesis draft/PDF, presentation files, reports, screenshots, generated plots
- Vendor manuals and third-party documents
- Promotion/application strategy notes
- Scripts that depend directly on private thesis data paths

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
  credential-like markers, unsupported production wording, private nested thesis
  paths, or ignored-CI markers.
- Python syntax compilation passed.
- Lightweight unit tests passed: 12 tests.

Open3D integration tests were retained but not run in this local environment
because Open3D is not installed here.
