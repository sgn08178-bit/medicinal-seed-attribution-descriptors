# Data access

## Included in this repository

This public repository is a code-only release. It includes analysis scripts,
configuration files, environment specifications, and documentation. It does
not include research data or derived numerical results.

## Available through controlled access

The following files are not included in the public repository:

- original seed images;
- processed images and foreground masks;
- model checkpoints;
- Integrated Gradients and Grad-CAM arrays;
- per-image descriptor maps;
- split files, predictions, association tables, and descriptor summary features;
- figure source data; and
- the 24-image manual-orientation correction manifest.

Access to image-derived files is subject to source-provider and institutional
redistribution permissions. Requests for non-commercial research or peer-review
access should be directed to the corresponding author at `daehyun@khu.ac.kr`
and may require approval by NIKOM and the relevant institution.

When access is approved, place the supplied files under the relative paths
described in `data/README.md` and `configs/`. The validation script additionally
expects the supplied derived tables under `source_data/`. No source-code path
changes are required.
