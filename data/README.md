# Local data layout

No research data are tracked in Git. After controlled access is approved, place
the supplied files under this directory:

```text
data/
|-- manual_orientation_corrections.csv
|-- raw_images/
|-- processed_images/
|-- foreground_masks/
|-- model_checkpoints/
|-- attribution_arrays/
`-- descriptor_maps/
```

The untracked paths above are excluded by `.gitignore`. Configuration files
under `configs/` use these relative paths. Separately supplied result tables and
figure source data used by `validation/` belong under `source_data/` at the
repository root.
