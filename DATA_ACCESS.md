# Data access

## Included in this repository

This public repository is a code-only release. It includes analysis scripts,
configuration files, environment specifications, and documentation. It does
not include research data or derived numerical results.

## Companion public dataset

A separate data release has been prepared for the 1,124 study samples. It
contains original images, processed images, foreground masks, sample metadata,
the fixed train/test split, the 24-image correction manifest, preprocessing
logs, predictions, and principal result tables. The DOI will be inserted here
after the corresponding author confirms the data license and the repository is
published:

> Dataset DOI: pending

The extra source-folder image `PJNA_0229.jpg` was not used in the study and is
not included in the release.

## Large derived intermediates

Model checkpoints, per-image Integrated Gradients and Grad-CAM arrays, and
per-image descriptor maps are not duplicated in the code repository or the
core image release. They can be regenerated with this repository. Questions
about additional peer-review material should be directed to the corresponding
author at `daehyun@khu.ac.kr`.

After downloading the data release, place or link its extracted root at
`data/`, or set `MEDICINAL_SEED_DATA_ROOT` to its location. Figure-only source
tables can be placed under `source_data/` or selected with
`MEDICINAL_SEED_SOURCE_DATA_ROOT`; no source-code path edits are required.
