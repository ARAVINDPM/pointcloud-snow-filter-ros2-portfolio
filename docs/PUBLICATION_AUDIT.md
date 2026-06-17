# Publication Audit

This repository is a self-contained public export of the reusable code and
synthetic pipeline from a private thesis repository.

## Included

- Reusable Python source code
- Synthetic-data generation and contamination scripts
- Selected tools that work with synthetic or user-provided local `.pcd` files
- Unit, integration, and reproducibility tests
- Packaging, environment, license, citation, and contribution files

## Excluded

- Original `.git` history
- Entire `thesis/` tree
- Real SICK/Livox point clouds and ROS bag/database files
- Notebooks and executed notebook outputs
- Thesis drafts, presentation files, reports, screenshots, generated plots
- Vendor manuals and third-party documents
- Any files matching high-risk patterns such as `.pcd`, `.npy`, `.db3`,
  `.bag`, `.pdf`, `.pptx`, `.docx`, `.xlsx`, `.png`, `.jpg`, and `.ipynb`

## Verification Commands

```bash
git ls-files
git ls-files '*.pcd' '*.npy' '*.db3' '*.bag' '*.xlsx' '*.pdf' '*.jpg' '*.png' '*.ipynb'
git count-objects -vH
```

Also scan for absolute home-directory paths, placeholder emails/DOIs, private
data markers, and credential-like terms before publishing.

All tracked files should be explainable from source code, synthetic tests, or
project metadata.
