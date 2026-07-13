#!/usr/bin/env python3
"""Export real preprocessing intermediate images for the manuscript workflow figure.

This script mirrors the current PNG v3 preprocessing settings in
process_png_v2.py, but it only processes one representative sample and
does not modify the training dataset, split files, or existing preprocessing
outputs.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from rembg import new_session, remove

from correction_manifest import DEFAULT_CORRECTION_CSV, load_operations


ROOT = Path(os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
INPUT_BASE_DIR = ROOT / "data" / "raw_images"
OUTPUT_DIR = ROOT / "outputs" / "preprocessing_workflow_ARSE"

SAMPLE_CLASS = "ARSE"
CANVAS_SIZE = 512
PADDING = 30
MAX_OBJ_SIZE = CANVAS_SIZE - (PADDING * 2)
MODEL_INPUT_SIZE = 224
BACKGROUND_COLOR = (0, 0, 0)
MASK_THRESHOLD = 25

# Canonical 24-image list shared with the production preprocessing script.
MANUAL_CORRECTIONS = load_operations("initial_operation", DEFAULT_CORRECTION_CSV)


def select_sample(sample_class: str = SAMPLE_CLASS) -> Path:
    class_dirs = sorted([p for p in INPUT_BASE_DIR.iterdir() if p.is_dir() and p.name.endswith(f"_{sample_class}")])
    if not class_dirs:
        raise FileNotFoundError(f"No input class folder ending with _{sample_class}: {INPUT_BASE_DIR}")

    files = []
    for class_dir in class_dirs:
        files.extend(
            sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            )
        )
    if not files:
        raise FileNotFoundError(f"No valid image files found for class {sample_class}")

    preferred = [p for p in files if p.stem in {f"{sample_class}_0000", f"{sample_class}_0001"}]
    return sorted(preferred or files)[0]


def crop_alpha_bbox(rgba_img: Image.Image) -> Image.Image:
    alpha = rgba_img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError("No valid alpha bounding box found.")
    return rgba_img.crop(bbox)


def rotate_rgba(rgba_img: Image.Image, angle_cw: float) -> Image.Image:
    rotated = rgba_img.rotate(
        -angle_cw,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )
    return crop_alpha_bbox(rotated)


def apply_manual_correction(rgba_img: Image.Image, operation: str | None) -> Image.Image:
    if operation in ["none", "", None]:
        return rgba_img
    if operation == "hflip":
        return crop_alpha_bbox(rgba_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    if operation == "vflip":
        return crop_alpha_bbox(rgba_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    if operation == "rot90cw":
        return rotate_rgba(rgba_img, 90)
    if operation == "rot90ccw":
        return rotate_rgba(rgba_img, 270)
    if operation == "rot135cw":
        return rotate_rgba(rgba_img, 135)
    if operation == "rot225cw":
        return rotate_rgba(rgba_img, 225)
    raise ValueError(f"Unknown manual correction operation: {operation}")


def rgba_to_rgb_black_background(rgba_img: Image.Image) -> Image.Image:
    black_bg = Image.new("RGBA", rgba_img.size, BACKGROUND_COLOR + (255,))
    return Image.alpha_composite(black_bg, rgba_img).convert("RGB")


def extract_binary_mask_from_rgba(rgba_img: Image.Image, threshold: int = MASK_THRESHOLD) -> Image.Image:
    alpha = np.array(rgba_img.split()[-1])
    binary = (alpha > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def find_largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def alpha_bbox_size(rgba_img: Image.Image) -> tuple[int, int]:
    bbox = rgba_img.split()[-1].getbbox()
    if bbox is None:
        return 0, 0
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def ensure_horizontal(rgba_img: Image.Image) -> Image.Image:
    width, height = alpha_bbox_size(rgba_img)
    if height > width:
        return crop_alpha_bbox(rgba_img.transpose(Image.Transpose.ROTATE_90))
    return rgba_img


def final_canvas_from_aligned(aligned_rgba: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    cw, ch = aligned_rgba.size
    if cw <= 0 or ch <= 0:
        raise RuntimeError("Invalid aligned image size.")

    ratio = min(MAX_OBJ_SIZE / cw, MAX_OBJ_SIZE / ch)
    new_w, new_h = int(cw * ratio), int(ch * ratio)
    resized_img = aligned_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    final_rgba = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    paste_x = (CANVAS_SIZE - new_w) // 2
    paste_y = (CANVAS_SIZE - new_h) // 2
    final_rgba.paste(resized_img, (paste_x, paste_y), resized_img)
    return final_rgba, (new_w, new_h)


def checkerboard(size: tuple[int, int], tile: int = 24) -> Image.Image:
    w, h = size
    board = Image.new("RGB", size, (230, 230, 230))
    draw = ImageDraw.Draw(board)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle((x, y, min(x + tile, w), min(y + tile, h)), fill=(185, 185, 185))
    return board.convert("RGBA")


def contour_bbox(contour) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(w), int(h)


def make_contour_overlay(raw_rgb: Image.Image, contour, ellipse) -> Image.Image:
    """Draw contour diagnostics on an unrotated, uncropped raw RGB copy.

    The contour and ellipse must be computed from an alpha mask with the same
    width, height, and coordinate system as raw_rgb.
    """
    overlay = np.array(raw_rgb.copy())
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 8)
    if ellipse is not None:
        cv2.ellipse(overlay, ellipse, (0, 80, 255), 7)
        (cx, cy), (axis_a, axis_b), angle = ellipse
        major = max(axis_a, axis_b)
        theta = np.deg2rad(angle + (90 if axis_b > axis_a else 0))
        dx = np.cos(theta) * major / 2
        dy = np.sin(theta) * major / 2
        p1 = (int(round(cx - dx)), int(round(cy - dy)))
        p2 = (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(overlay, p1, p2, (80, 255, 0), 7)
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 10, (80, 255, 0), -1)
    return Image.fromarray(overlay, mode="RGB")


def make_debug_mask_overlay(raw_rgb: Image.Image, binary_mask: np.ndarray) -> Image.Image:
    raw = np.array(raw_rgb.convert("RGB"), dtype=np.uint8)
    mask_bool = binary_mask > 0
    overlay = raw.copy()
    red = np.zeros_like(raw)
    red[..., 0] = 255
    overlay[mask_bool] = (0.62 * raw[mask_bool] + 0.38 * red[mask_bool]).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB")


def save_image(img: Image.Image, name: str, saved: list[Path]) -> Path:
    path = OUTPUT_DIR / name
    img.save(path, "PNG")
    saved.append(path)
    return path


def mask_unique_values(img: Image.Image) -> list[int]:
    return sorted(int(v) for v in np.unique(np.array(img)))


def print_image_report(path: Path) -> None:
    with Image.open(path) as im:
        extra = ""
        if im.mode == "L":
            extra = f", unique values {mask_unique_values(im)}"
        print(f"{path.name}: {im.mode}, {im.size[0]} x {im.size[1]}{extra}")


def verify_horizontal(path: Path) -> tuple[bool, tuple[int, int]]:
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        width, height = alpha_bbox_size(rgba)
        return width >= height, (width, height)


def export_preprocessing_steps() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = select_sample(SAMPLE_CLASS)
    stem = sample_path.stem
    manual_operation = MANUAL_CORRECTIONS.get(stem, "none")
    saved: list[Path] = []

    print(f"Selected sample: {sample_path}")
    print(f"Selected class: {SAMPLE_CLASS}")
    print(f"Manual correction: {manual_operation}")

    try:
        # Keep rembg input identical to process_png_v2.py, while saving the raw
        # figure panel in the image's EXIF display orientation. rembg returns
        # the same display-oriented coordinate system, so contour overlay uses
        # the EXIF-transposed raw copy without rotating/cropping/resizing it.
        raw_pil = Image.open(sample_path)
        raw_rgba_for_pipeline = raw_pil.convert("RGBA")
        raw_display_rgba = ImageOps.exif_transpose(raw_pil).convert("RGBA")
        raw_rgb = raw_display_rgba.convert("RGB")
        save_image(raw_rgb, "01_raw_rgb.png", saved)

        session = new_session("u2net")
        no_bg_img = remove(raw_rgba_for_pipeline, session=session, alpha_matting=False).convert("RGBA")
        if no_bg_img.size != raw_rgb.size:
            raise RuntimeError(
                "Coordinate mismatch before contour overlay: "
                f"raw_rgb size={raw_rgb.size}, background_removed_rgba size={no_bg_img.size}"
            )
        save_image(no_bg_img, "02_background_removed_rgba.png", saved)

        fg_mask = extract_binary_mask_from_rgba(no_bg_img)
        if fg_mask.size != raw_rgb.size:
            raise RuntimeError(
                "Coordinate mismatch before mask export: "
                f"raw_rgb size={raw_rgb.size}, foreground_mask size={fg_mask.size}"
            )
        save_image(fg_mask, "03_foreground_mask.png", saved)

        alpha = np.array(no_bg_img.split()[-1])
        binary_for_contour = (alpha > MASK_THRESHOLD).astype(np.uint8) * 255
        if binary_for_contour.shape[:2] != (raw_rgb.size[1], raw_rgb.size[0]):
            raise RuntimeError(
                "Coordinate mismatch before contour detection: "
                f"raw_rgb WxH={raw_rgb.size}, binary_mask shape={binary_for_contour.shape}"
            )
        overlay_contour = find_largest_contour(binary_for_contour)
        if overlay_contour is None:
            raise RuntimeError("No foreground contour found after background removal.")

        overlay_ellipse = cv2.fitEllipse(overlay_contour) if len(overlay_contour) >= 5 else None
        x, y, w, h = contour_bbox(overlay_contour)
        nonzero = np.where(binary_for_contour > 0)
        mask_bbox = None
        if nonzero[0].size:
            yy, xx = nonzero
            mask_bbox = (int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1))
            ix1 = max(x, mask_bbox[0])
            iy1 = max(y, mask_bbox[1])
            ix2 = min(x + w, mask_bbox[0] + mask_bbox[2])
            iy2 = min(y + h, mask_bbox[1] + mask_bbox[3])
            if ix2 <= ix1 or iy2 <= iy1:
                print("WARNING: contour bounding box does not overlap binary foreground mask bbox.")

        print("\nContour coordinate debug:")
        print(f"raw_rgb size: {raw_rgb.size[0]} x {raw_rgb.size[1]}")
        print(f"background_removed_rgba size: {no_bg_img.size[0]} x {no_bg_img.size[1]}")
        print(f"binary_mask size: {binary_for_contour.shape[1]} x {binary_for_contour.shape[0]}")
        print(f"largest contour bounding box: x={x}, y={y}, w={w}, h={h}")
        if mask_bbox is not None:
            print(f"binary foreground bounding box: x={mask_bbox[0]}, y={mask_bbox[1]}, w={mask_bbox[2]}, h={mask_bbox[3]}")
        if overlay_ellipse is not None:
            (cx, cy), (axis_a, axis_b), angle = overlay_ellipse
            print(
                "fitted ellipse for contour overlay: "
                f"center=({cx:.3f}, {cy:.3f}), axes=({axis_a:.3f}, {axis_b:.3f}), angle={angle:.6f}"
            )
        print("same coordinate system confirmed: raw_rgb, alpha mask, and contour overlay use identical width and height.")

        contour_overlay = make_contour_overlay(raw_rgb, overlay_contour, overlay_ellipse)
        save_image(contour_overlay, "04_contour_overlay.png", saved)
        save_image(make_debug_mask_overlay(raw_rgb, binary_for_contour), "debug_mask_overlay_same_coordinates.png", saved)

        # From this point onward, keep the original process_png_v2.py behavior.
        # It estimates the preprocessing rotation from the raw alpha contour,
        # before the thresholded contour used for the manuscript overlay.
        pipeline_contour = find_largest_contour(alpha)
        if pipeline_contour is None:
            raise RuntimeError("No foreground contour found in alpha channel for orientation normalization.")
        pipeline_ellipse = cv2.fitEllipse(pipeline_contour) if len(pipeline_contour) >= 5 else None
        rotation_angle = float(pipeline_ellipse[2]) if pipeline_ellipse is not None else 0.0
        if pipeline_ellipse is not None:
            (pcx, pcy), (paxis_a, paxis_b), pangle = pipeline_ellipse
            print(
                "fitted ellipse for pipeline rotation: "
                f"center=({pcx:.3f}, {pcy:.3f}), axes=({paxis_a:.3f}, {paxis_b:.3f}), angle={pangle:.6f}"
            )
        rotated_img = no_bg_img.rotate(
            rotation_angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )
        orientation_rgba = ensure_horizontal(rotated_img)
        save_image(orientation_rgba, "05_orientation_normalized_rgba.png", saved)

        rotated_mask = np.array(rotated_img.split()[-1])
        _, thresh = cv2.threshold(rotated_mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        contour_after_rotation = find_largest_contour(thresh)
        if contour_after_rotation is not None:
            x, y, w, h = cv2.boundingRect(contour_after_rotation)
            cropped_img = rotated_img.crop((x, y, x + w, y + h))
        else:
            cropped_img = crop_alpha_bbox(rotated_img)

        cropped_img = ensure_horizontal(cropped_img)
        alpha_arr = np.array(cropped_img.split()[-1])
        fw, _ = cropped_img.size
        left_weight = np.sum(alpha_arr[:, : fw // 2])
        right_weight = np.sum(alpha_arr[:, fw // 2 :])
        if right_weight > left_weight:
            cropped_img = crop_alpha_bbox(cropped_img.transpose(Image.Transpose.ROTATE_180))
        cropped_img = apply_manual_correction(cropped_img, manual_operation)
        cropped_img = ensure_horizontal(cropped_img)
        save_image(cropped_img, "06_cropped_aligned_rgba.png", saved)

        final_rgba, object_size = final_canvas_from_aligned(cropped_img)
        save_image(final_rgba, "07_centered_canvas_512_rgba.png", saved)

        model_input = rgba_to_rgb_black_background(final_rgba).resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            Image.Resampling.BILINEAR,
        )
        save_image(model_input, "08_model_input_224_rgb.png", saved)

        final_mask_224 = extract_binary_mask_from_rgba(final_rgba).resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            Image.Resampling.NEAREST,
        )
        final_mask_224 = Image.fromarray((np.array(final_mask_224) > 0).astype(np.uint8) * 255, mode="L")
        save_image(final_mask_224, "09_final_foreground_mask_224.png", saved)

        # Figure-only preview. This checkerboard is not used for analysis.
        checker = checkerboard((CANVAS_SIZE, CANVAS_SIZE))
        checker.alpha_composite(final_rgba)
        save_image(checker.convert("RGB"), "10_checkerboard_preview.png", saved)

        readme_path = OUTPUT_DIR / "README.txt"
        readme_path.write_text(
            "\n".join(
                [
                    "Preprocessing workflow figure export",
                    "",
                    f"selected sample filename: {sample_path.name}",
                    f"selected sample path: {sample_path}",
                    f"selected class: {SAMPLE_CLASS}",
                    f"date and time of export: {datetime.now().isoformat(timespec='seconds')}",
                    "",
                    "preprocessing parameters:",
                    "- rembg session: u2net",
                    "- alpha matting: False",
                    "- orientation normalization: contour/largest-contour ellipse-based rotation",
                    f"- mask threshold: {MASK_THRESHOLD}",
                    f"- padding: {PADDING} pixels",
                    f"- max object size: {MAX_OBJ_SIZE} pixels",
                    f"- final canvas size: {CANVAS_SIZE} x {CANVAS_SIZE}",
                    f"- model input size: {MODEL_INPUT_SIZE} x {MODEL_INPUT_SIZE}",
                    "- object resize interpolation: Lanczos",
                    "- final mask resize interpolation: nearest-neighbor",
                    "- model-input preview background: RGB black, no ImageNet normalization",
                    f"- manual correction for selected sample: {manual_operation}",
                    f"- fitted ellipse rotation angle: {rotation_angle}",
                    f"- object size before 512 canvas: {object_size[0]} x {object_size[1]}",
                    "",
                    "saved files and intended figure panel usage:",
                    "- 01_raw_rgb.png: panel A, raw RGB image",
                    "- 02_background_removed_rgba.png: panel B, background-removed RGBA with transparent background",
                    "- 03_foreground_mask.png: panel C, binary foreground mask from alpha channel",
                    "- 04_contour_overlay.png: panel D, largest contour, fitted ellipse, and major-axis overlay",
                    "- 05_orientation_normalized_rgba.png: panel D/E, orientation-normalized RGBA image",
                    "- 06_cropped_aligned_rgba.png: reference before final centered canvas",
                    "- 07_centered_canvas_512_rgba.png: panel E, centered 512 x 512 transparent RGBA canvas",
                    "- 08_model_input_224_rgb.png: panel F, black-background 224 x 224 RGB model-input preview",
                    "- 09_final_foreground_mask_224.png: panel F or methods inset, final 224 x 224 binary mask",
                    "- 10_checkerboard_preview.png: figure-only preview of transparent placement; not used for analysis",
                    "",
                    "note: These images are for manuscript preprocessing workflow figure preparation.",
                    "They are exported from the actual preprocessing steps and should not be treated as new training data.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print("\nQuality report:")
        for path in saved:
            print_image_report(path)

        print("\nOrientation verification:")
        for name in ["05_orientation_normalized_rgba.png", "07_centered_canvas_512_rgba.png"]:
            ok, bbox_size = verify_horizontal(OUTPUT_DIR / name)
            print(f"{name}: horizontal={ok}, foreground bbox={bbox_size[0]} x {bbox_size[1]}")
        mask_224 = Image.open(OUTPUT_DIR / "09_final_foreground_mask_224.png")
        arr = np.array(mask_224)
        ys, xs = np.where(arr > 0)
        bbox_size = (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) if xs.size else (0, 0)
        print(f"08_model_input_224_rgb.png / 09 mask: horizontal={bbox_size[0] >= bbox_size[1]}, foreground bbox={bbox_size[0]} x {bbox_size[1]}")

        print("\nSaved output paths:")
        for path in saved:
            print(path)
        print(readme_path)

        return saved + [readme_path]
    except Exception as exc:
        raise RuntimeError(f"Preprocessing workflow export failed at sample {sample_path}: {exc}") from exc


if __name__ == "__main__":
    export_preprocessing_steps()
