from __future__ import annotations

from pathlib import Path


def validate_allowed_stage2_paths(config: dict) -> None:
    blocked = ["01_ig_convnext/convnext_small", "gradcam_layer_selection", "02_gradcam_candidate_layers"]
    keys = [
        "canonical_ig_zero_absolute_dir",
        "canonical_ig_zero_positive_dir",
        "convnext_gradcam_final_dir",
    ]
    for key in keys:
        value = str(config.get(key, ""))
        for token in blocked:
            if token in value:
                raise ValueError(f"Blocked non-final Stage 2 path in {key}: {value}")
    if "01_ig_convnext_canonical_rawrgb_baseline" not in str(config.get("canonical_ig_zero_absolute_dir", "")):
        raise ValueError("canonical_ig_zero_absolute_dir must point to canonical raw-RGB-baseline IG.")
    if "03_gradcam_final_selected_layers" not in str(config.get("convnext_gradcam_final_dir", "")):
        raise ValueError("convnext_gradcam_final_dir must point to final selected-layer Grad-CAM.")


def build_input_manifest(config: dict, run_dir: str | Path) -> dict:
    return {
        "run_dir": str(run_dir),
        "test_csv": config["test_csv"],
        "image_root": config["image_root"],
        "mask_root": config["mask_root"],
        "stage2_run_dir": config["stage2_run_dir"],
        "canonical_ig_zero_absolute_dir": config["canonical_ig_zero_absolute_dir"],
        "canonical_ig_zero_positive_dir": config["canonical_ig_zero_positive_dir"],
        "convnext_gradcam_final_dir": config["convnext_gradcam_final_dir"],
        "deprecated_ig_used": False,
        "gradcam_candidate_used": False,
    }
