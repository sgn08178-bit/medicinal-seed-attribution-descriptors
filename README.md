# Medicinal seed attribution-descriptor analysis

This repository supports the analyses reported in **Contextualizing CNN attribution maps using predefined image-derived descriptors in medicinal plant seed classification**. It contains analysis code, reproducibility specifications, environment files, and validation scripts. Research data, derived result tables, the manuscript, and publication figures are maintained separately.

## Workflow

1. `preprocessing/`: background removal, orientation normalization, foreground centering, manual-QC correction, and mask generation.
2. `model_training/`: stratified split, five-fold cross-validation, final training, and independent-test evaluation for ConvNeXt-Small, ResNet50, and EfficientNet-B0.
3. `attribution/`: zero- and Gaussian-blurred-baseline Integrated Gradients and selected-layer Grad-CAM.
4. `descriptor_generation/`: 29 predefined image-derived descriptor maps.
5. `statistics/`: foreground-restricted Spearman association, classwise summaries, and figure-ready summaries.
6. `descriptor_classifier/`: classification from foreground descriptor summary features.
7. `figure_generation/`: supplementary figure and table generation scripts.

Each stage reads the preceding stage and writes to a new `results/` subdirectory. No research data are bundled. Obtain the controlled input and derived-data package separately and place it under the relative paths documented in `data/README.md` and `configs/`. See `DATA_ACCESS.md` for the access scope.

## Manual orientation corrections

The preprocessing code expects the canonical machine-readable correction record at `data/manual_orientation_corrections.csv`. This controlled data file is not included in the public code-only repository. It records 24 unique images: 15 initial horizontal flips, 4 initial vertical flips, and 5 initial rotation corrections, matching the Methods. Four of the same 24 images also have a recorded final-stage postprocessing adjustment. Preprocessing stops with an error if the separately supplied manifest is missing or does not pass its count and operation checks.

## Execution order

Run commands from the repository root with the final environment activated.

```bash
python model_training/scripts/01_make_splits.py --config configs/stage1_default.yaml --output-dir results/stage1
python model_training/scripts/02_train_cv.py --config configs/stage1_default.yaml --output-dir results/stage1 --model-name convnext_small
python model_training/scripts/03_train_final.py --config configs/stage1_default.yaml --output-dir results/stage1 --model-name convnext_small
python model_training/scripts/04_evaluate_test.py --config configs/stage1_default.yaml --output-dir results/stage1 --model-name convnext_small
python model_training/scripts/05_collect_model_comparison.py --config configs/stage1_default.yaml --output-dir results/stage1

python attribution/scripts/01_prepare_stage2_inputs.py --config configs/stage2_attribution.yaml --output-dir results/stage2
python attribution/scripts/08_compute_ig_canonical_rawrgb.py --run-dir results/stage2/stage2_attribution_final
python attribution/scripts/09_compute_gradcam_final_selected_layers.py --run-dir results/stage2/stage2_attribution_final

python descriptor_generation/scripts/00_validate_stage3_inputs.py --config configs/stage3_descriptor_association.yaml --run-dir results/stage3/final
python descriptor_generation/scripts/01_generate_descriptor_maps.py --config configs/stage3_descriptor_association.yaml --run-dir results/stage3/final
python statistics/scripts/02_compute_ig_descriptor_association.py --config configs/stage3_descriptor_association.yaml --run-dir results/stage3/final --mode absolute
python statistics/scripts/02_compute_ig_descriptor_association.py --config configs/stage3_descriptor_association.yaml --run-dir results/stage3/final --mode positive
python statistics/scripts/03_compute_gradcam_descriptor_association.py --config configs/stage3_descriptor_association.yaml --run-dir results/stage3/final
python statistics/scripts/04_make_stage3_figures.py --config configs/stage3_descriptor_association.yaml --run-dir results/stage3/final
python statistics/scripts/05_write_stage3_summary.py --config configs/stage3_descriptor_association.yaml --run-dir results/stage3/final
```

The descriptor classifier scripts require the train/test descriptor-map roots listed at the top of each script. Set `MEDICINAL_SEED_PROJECT_ROOT` to the repository root after placing the separately supplied data package.

## Tables, figures, and validation

```bash
python figure_generation/create_submission_ready_supplementary_tables.py
python figure_generation/generate_supplementary_figures.py
python validation/validate_key_results.py
```

The figure scripts preserve the exact final source mapping but require a local working copy of the separately supplied source tables and arrays under `source_data/`. After that controlled package is placed locally, `validation/validate_key_results.py` reproduces the split counts, descriptor count, ConvNeXt accuracy, manual-correction counts, and three principal IG association values.

Figures 1 and 2 are author-designed final composite figures and are not generated automatically by the analysis scripts. Generation code for analytical panels is included where applicable, while quantitative source data are available through the controlled-access process. Legacy figure-development helpers are excluded from this public release.

## Compute environment

- Final recorded hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, CUDA 12.8, cuDNN 9.19.0.
- Model training and attribution generation require a CUDA-capable GPU. Descriptor generation, statistics, table generation, and source-data validation can run on CPU.
- Python and package versions are in `environment/requirements_frozen.txt` and `environment/environment.yml`.
- A legacy conflicting `pip freeze` record is retained in the private submission audit as provenance evidence only; it is not an installation requirement.
- Random seed: 42.
- End-to-end wall-clock time was not preserved in the final logs. No runtime estimate is supplied because inventing one would be misleading; record measured times when rerunning on the target system.

## Checksums

The repository file inventory and SHA-256 checksums are in `MANIFEST.csv`.

After changing any public file, rebuild the repository manifest from the repository root:

```bash
python validation/rebuild_manifests.py
```

## Data and release limitations

No research data are included. This covers raw and processed images, masks, checkpoints, attribution arrays, descriptor maps, result tables, figure source data, and the manual-correction manifest. Access is subject to source-provider and institutional permissions; see `DATA_ACCESS.md`.

## License

No open-source license has been granted yet. The repository is publicly visible for scientific review and transparency, but reuse requires prior permission until the corresponding author and institution select a license. See `LICENSE_STATUS.md`.
