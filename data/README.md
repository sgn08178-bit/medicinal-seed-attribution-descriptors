# Local data layout

No research data are tracked in Git. Download the companion dataset from
[Zenodo](https://doi.org/10.5281/zenodo.21537568). The dataset is released
under CC BY 4.0.

Extract all companion archives into a common dataset root. Place or link that
root here, or set `MEDICINAL_SEED_DATA_ROOT` to the extracted dataset root:

```text
data/
|-- raw_images/
|-- processed_images/
|-- foreground_masks/
|-- metadata/
|   |-- sample_metadata.csv
|   |-- train_split.csv
|   |-- test_split.csv
|   |-- manual_orientation_corrections.csv
|   `-- convnext_small_test_predictions.csv
|-- model_checkpoints/              # optional regenerated artifact
|-- attribution_arrays/             # optional regenerated artifact
`-- descriptor_maps/                # optional regenerated artifact
```

The untracked paths above are excluded by `.gitignore`. Configuration files use
these relative paths. Figure source data used by `figure_generation/` and
`validation/` belong under `source_data/` unless
`MEDICINAL_SEED_SOURCE_DATA_ROOT` points elsewhere.
