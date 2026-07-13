import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from correction_manifest import DEFAULT_CORRECTION_CSV, load_operations


INPUT_IMAGE_ROOT = Path("data/processed_images_v2")
INPUT_MASK_ROOT = Path("data/foreground_masks_v2")
OUTPUT_ROOT = Path("data")
OUTPUT_IMAGE_ROOT = OUTPUT_ROOT / "images"
OUTPUT_MASK_ROOT = OUTPUT_ROOT / "masks"
LOG_PATH = OUTPUT_ROOT / "postprocess_log.csv"
CONTACT_SHEET_PATH = OUTPUT_ROOT / "manual_corrections_contact_sheet.png"

# Additional final-stage operations for four images already included in the
# canonical 24-image correction manifest.
MANUAL_OPS = load_operations("postprocess_operation", DEFAULT_CORRECTION_CSV)

EXPECTED_COUNTS = {
    "ARSE": 205,
    "ARSS": 223,
    "PJNA": 228,
    "PRDA": 228,
    "PRPE": 240,
}

LOG_FIELDS = [
    "class_folder",
    "code",
    "input_image",
    "input_mask",
    "output_image",
    "output_mask",
    "manual_operation",
    "background_set_to_black",
    "image_size",
    "mask_size",
    "status",
    "error",
]


def class_prefix(code):
    return code.split("_", 1)[0]


def should_include(code):
    prefix = class_prefix(code)
    if prefix not in EXPECTED_COUNTS:
        return True
    try:
        number = int(code.split("_", 1)[1])
    except (IndexError, ValueError):
        return True
    return number <= EXPECTED_COUNTS[prefix]


def find_mask(class_folder, code):
    candidates = [
        INPUT_MASK_ROOT / class_folder / f"{code}_mask.png",
        INPUT_MASK_ROOT / class_folder / f"{code}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def apply_operation_image(img, operation):
    if operation == "none":
        return img
    if operation == "hflip":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if operation == "rot90cw":
        return img.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"Unknown operation: {operation}")


def apply_operation_mask(mask, operation):
    if operation == "none":
        return mask
    if operation == "hflip":
        return mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if operation == "rot90cw":
        return mask.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"Unknown operation: {operation}")


def binary_mask(mask_img):
    arr = np.array(mask_img.convert("L"))
    return ((arr > 0).astype(np.uint8) * 255)


def set_background_black(rgb_img, mask_binary):
    arr = np.array(rgb_img.convert("RGB")).copy()
    arr[mask_binary == 0] = (0, 0, 0)
    return Image.fromarray(arr, mode="RGB")


def mask_stem_matches(code, mask_path):
    stem = mask_path.stem
    return stem == code or stem == f"{code}_mask"


def save_contact_sheet(records):
    if not records:
        return

    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.array([axes])

    titles = ["before image", "before mask", "after image", "after mask"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=10)

    for i, rec in enumerate(records):
        before_img = Image.open(rec["input_image"]).convert("RGB")
        before_mask = Image.open(rec["input_mask"]).convert("L")
        after_img = Image.open(rec["output_image"]).convert("RGB")
        after_mask = Image.open(rec["output_mask"]).convert("L")

        panels = [before_img, before_mask, after_img, after_mask]
        for j, panel in enumerate(panels):
            cmap = "gray" if j in (1, 3) else None
            axes[i, j].imshow(panel, cmap=cmap)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"{rec['code']}\n{rec['manual_operation']}", fontsize=9)

    plt.tight_layout()
    fig.savefig(CONTACT_SHEET_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_ROOT.mkdir(parents=True, exist_ok=True)

    log_rows = []
    manual_records = []
    size_mismatches = []
    stem_mismatches = []

    image_paths = sorted(INPUT_IMAGE_ROOT.glob("*/*.png"))

    for image_path in image_paths:
        class_folder = image_path.parent.name
        code = image_path.stem
        operation = MANUAL_OPS.get(code, "none")
        mask_path = find_mask(class_folder, code)

        output_image_path = OUTPUT_IMAGE_ROOT / class_folder / image_path.name
        output_mask_path = OUTPUT_MASK_ROOT / class_folder / f"{code}_mask.png"
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "class_folder": class_folder,
            "code": code,
            "input_image": str(image_path),
            "input_mask": str(mask_path) if mask_path else "",
            "output_image": str(output_image_path),
            "output_mask": str(output_mask_path),
            "manual_operation": operation,
            "background_set_to_black": False,
            "image_size": "",
            "mask_size": "",
            "status": "",
            "error": "",
        }

        try:
            if not should_include(code):
                row["status"] = "skipped_excluded_to_match_expected_counts"
                log_rows.append(row)
                continue

            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for code={code}")

            if not mask_stem_matches(code, mask_path):
                stem_mismatches.append((code, str(mask_path)))

            before_image_size = Image.open(image_path).size
            before_mask_size = Image.open(mask_path).size
            if before_image_size != before_mask_size:
                size_mismatches.append((code, "before", before_image_size, before_mask_size))

            img = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            img = apply_operation_image(img, operation)
            mask = apply_operation_mask(mask, operation)

            if img.size != mask.size:
                size_mismatches.append((code, "after", img.size, mask.size))
                raise RuntimeError(f"Image/mask size mismatch after operation: {img.size} vs {mask.size}")

            mask_arr = binary_mask(mask)
            out_img = set_background_black(img, mask_arr)
            out_mask = Image.fromarray(mask_arr, mode="L")

            out_img.save(output_image_path, "PNG")
            out_mask.save(output_mask_path, "PNG")

            row["background_set_to_black"] = True
            row["image_size"] = f"{out_img.size[0]}x{out_img.size[1]}"
            row["mask_size"] = f"{out_mask.size[0]}x{out_mask.size[1]}"
            row["status"] = "success"

            if operation != "none":
                manual_records.append(row.copy())

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)

        log_rows.append(row)

    with open(LOG_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(log_rows)

    save_contact_sheet(manual_records)

    output_images = sorted(OUTPUT_IMAGE_ROOT.glob("*/*.png"))
    output_masks = sorted(OUTPUT_MASK_ROOT.glob("*/*.png"))
    failed = [r for r in log_rows if r["status"] == "failed"]
    manual_non_none = [r for r in log_rows if r["manual_operation"] != "none" and r["status"] == "success"]

    print("1. 전체 image PNG 개수:", len(output_images))
    print("2. 전체 mask PNG 개수:", len(output_masks))
    print("3. 클래스별 image 개수")
    for class_dir in sorted(OUTPUT_IMAGE_ROOT.iterdir()):
        if class_dir.is_dir():
            print(f"  {class_dir.name}: {len(list(class_dir.glob('*.png')))}")
    print("4. 클래스별 mask 개수")
    for class_dir in sorted(OUTPUT_MASK_ROOT.iterdir()):
        if class_dir.is_dir():
            print(f"  {class_dir.name}: {len(list(class_dir.glob('*.png')))}")
    print("5. failed 항목 수:", len(failed))
    print("6. manual_operation이 none이 아닌 항목 수:", len(manual_non_none))
    print("7. 수동 보정 적용 파일 목록")
    for rec in manual_non_none:
        print(f"  {rec['code']}: {rec['manual_operation']}")
    print("8. image와 mask 크기 불일치 여부:", "있음" if size_mismatches else "없음")
    if size_mismatches:
        for item in size_mismatches:
            print(" ", item)
    print("9. image와 mask stem 불일치 여부:", "있음" if stem_mismatches else "없음")
    if stem_mismatches:
        for item in stem_mismatches:
            print(" ", item)

    skipped = [r for r in log_rows if r["status"].startswith("skipped")]
    if skipped:
        print("추가 참고: expected count 일치를 위해 제외한 항목")
        for rec in skipped:
            print(f"  {rec['code']}")

    print(f"postprocess_log.csv: {LOG_PATH}")
    print(f"contact sheet: {CONTACT_SHEET_PATH}")


if __name__ == "__main__":
    main()
