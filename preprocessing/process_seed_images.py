import os
import csv
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from correction_manifest import DEFAULT_CORRECTION_CSV, load_operations

# =========================
# Path settings
# =========================
input_base_dir = "data/raw_images"

# 새 PNG 전처리 결과 폴더
output_base_dir = "data/processed_images"
output_image_base_dir = os.path.join(output_base_dir, "images")
output_mask_base_dir = os.path.join(output_base_dir, "masks")

# 수동 보정 기록 파일. 누락되면 24개 보정이 조용히 생략되지 않도록
# 전처리를 중단한다.
manual_correction_csv = DEFAULT_CORRECTION_CSV

os.makedirs(output_image_base_dir, exist_ok=True)
os.makedirs(output_mask_base_dir, exist_ok=True)

# =========================
# Preprocessing settings
# =========================
CANVAS_SIZE = 512
PADDING = 30
MAX_OBJ_SIZE = CANVAS_SIZE - (PADDING * 2)

# 배경은 모델 학습용 RGB 이미지에서 흰색으로 고정
BACKGROUND_COLOR = (255, 255, 255)

# rembg session
session = new_session("u2net")


# =========================
# Manual correction loader
# =========================
def load_manual_corrections(csv_path):
    """Load the canonical 24-image initial correction mapping by image stem."""
    corrections = load_operations("initial_operation", csv_path)
    print(f"✅ Loaded manual corrections: {len(corrections)} entries")
    return corrections


def apply_manual_correction(rgba_img, operation):
    """
    Apply manual orientation correction to RGBA image.
    """
    if operation in ["none", "", None]:
        return rgba_img

    if operation == "hflip":
        return rgba_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if operation == "vflip":
        return rgba_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    rotation_angles_cw = {
        "rot90cw": 90,
        "rot90ccw": 270,
        "rot135cw": 135,
        "rot225cw": 225,
    }
    if operation in rotation_angles_cw:
        rotated = rgba_img.rotate(
            -rotation_angles_cw[operation],
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )
        bbox = rotated.getbbox()
        return rotated.crop(bbox) if bbox is not None else rotated

    raise ValueError(f"Unknown manual correction operation: {operation}")


def rgba_to_rgb_white_background(rgba_img):
    """
    Convert RGBA image to RGB image with white background.
    This prevents transparent background from becoming black during model loading.
    """
    white_bg = Image.new("RGBA", rgba_img.size, BACKGROUND_COLOR + (255,))
    composed = Image.alpha_composite(white_bg, rgba_img)
    return composed.convert("RGB")


def extract_binary_mask_from_rgba(rgba_img):
    """
    Extract binary foreground mask from RGBA alpha channel.
    Output: PIL grayscale image, 0 background / 255 foreground.
    """
    alpha = np.array(rgba_img.split()[-1])
    binary = (alpha > 0).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


# =========================
# Main preprocessing function
# =========================
def process_seed_image(img_path, image_save_path, mask_save_path, manual_operation="none"):
    """
    Full preprocessing:
    1. Load original as RGBA
    2. rembg U2Net background removal
    3. extract alpha mask and largest contour
    4. ellipse-based rotation
    5. crop by largest contour
    6. horizontal orientation normalization
    7. left/right mass-based 180 rotation
    8. apply manual correction if needed
    9. resize to fit 512 canvas with 30 px padding
    10. save RGB PNG with white background
    11. save binary foreground mask PNG
    """
    try:
        # 1. Load original image at original resolution
        img = Image.open(img_path).convert("RGBA")

        # 2. Background removal
        # alpha_matting=False to avoid excessive smoothing of seed boundary/texture
        no_bg_img = remove(img, session=session, alpha_matting=False)

        # 3. Extract mask and find largest contour
        mask = np.array(no_bg_img.split()[-1])
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise RuntimeError("No foreground contour found after background removal.")

        c = max(contours, key=cv2.contourArea)

        # 4. Ellipse-based orientation correction
        rotation_angle = 0.0
        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            rotation_angle = ellipse[2]

        rotated_img = no_bg_img.rotate(
            rotation_angle,
            resample=Image.Resampling.BICUBIC,
            expand=True
        )

        # 5. Smart crop after rotation
        rotated_mask = np.array(rotated_img.split()[-1])

        # threshold only for locating object, not modifying original RGBA pixels
        _, thresh = cv2.threshold(rotated_mask, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped_img = rotated_img.crop((x, y, x + w, y + h))
        else:
            bbox = rotated_img.getbbox()
            if bbox is None:
                raise RuntimeError("No valid bounding box found after rotation.")
            cropped_img = rotated_img.crop(bbox)

        # 6. Make object horizontal if height > width
        cw, ch = cropped_img.size
        if ch > cw:
            cropped_img = cropped_img.transpose(Image.Transpose.ROTATE_90)
            bbox = cropped_img.getbbox()
            if bbox is not None:
                cropped_img = cropped_img.crop(bbox)

        # 7. Left-right orientation normalization using foreground mass
        final_mask = np.array(cropped_img.split()[-1])
        fw, fh = cropped_img.size

        left_weight = np.sum(final_mask[:, :fw // 2])
        right_weight = np.sum(final_mask[:, fw // 2:])

        if right_weight > left_weight:
            cropped_img = cropped_img.transpose(Image.Transpose.ROTATE_180)

        # 8. Apply manual correction after automatic orientation normalization
        cropped_img = apply_manual_correction(cropped_img, manual_operation)

        # 9. Resize object to fit 512 canvas with padding
        cw, ch = cropped_img.size
        ratio = min(MAX_OBJ_SIZE / cw, MAX_OBJ_SIZE / ch)
        new_w, new_h = int(cw * ratio), int(ch * ratio)

        resized_img = cropped_img.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )

        # 10. Place on transparent 512 canvas
        final_rgba = Image.new(
            "RGBA",
            (CANVAS_SIZE, CANVAS_SIZE),
            (0, 0, 0, 0)
        )

        paste_x = (CANVAS_SIZE - new_w) // 2
        paste_y = (CANVAS_SIZE - new_h) // 2
        final_rgba.paste(resized_img, (paste_x, paste_y), resized_img)

        # 11. Save model-training image as RGB PNG with white background
        final_rgb = rgba_to_rgb_white_background(final_rgba)
        final_rgb.save(image_save_path, "PNG")

        # 12. Save binary foreground mask as PNG
        final_mask_img = extract_binary_mask_from_rgba(final_rgba)
        final_mask_img.save(mask_save_path, "PNG")

        return {
            "status": "success",
            "rotation_angle": rotation_angle,
            "manual_operation": manual_operation,
            "object_size_before_canvas": f"{new_w}x{new_h}",
            "image_path": image_save_path,
            "mask_path": mask_save_path,
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "manual_operation": manual_operation,
            "image_path": image_save_path,
            "mask_path": mask_save_path,
        }


# =========================
# Run preprocessing
# =========================
manual_corrections = load_manual_corrections(manual_correction_csv)

log_rows = []

for folder_name in sorted(os.listdir(input_base_dir)):
    folder_path = os.path.join(input_base_dir, folder_name)

    if not os.path.isdir(folder_path):
        continue

    output_image_folder = os.path.join(output_image_base_dir, folder_name)
    output_mask_folder = os.path.join(output_mask_base_dir, folder_name)

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        img_path = os.path.join(folder_path, file_name)

        # PNG output filename
        stem = os.path.splitext(file_name)[0]
        image_save_name = stem + ".png"
        mask_save_name = stem + "_mask.png"

        image_save_path = os.path.join(output_image_folder, image_save_name)
        mask_save_path = os.path.join(output_mask_folder, mask_save_name)

        manual_operation = manual_corrections.get(stem, "none")

        result = process_seed_image(
            img_path=img_path,
            image_save_path=image_save_path,
            mask_save_path=mask_save_path,
            manual_operation=manual_operation
        )

        log_row = {
            "folder": folder_name,
            "filename": file_name,
            "output_image": image_save_path,
            "output_mask": mask_save_path,
            "status": result.get("status"),
            "rotation_angle": result.get("rotation_angle", ""),
            "manual_operation": result.get("manual_operation", "none"),
            "object_size_before_canvas": result.get("object_size_before_canvas", ""),
            "error": result.get("error", ""),
        }

        log_rows.append(log_row)

        if result["status"] == "success":
            print(f"✅ 처리 완료: {folder_name}/{file_name} -> {image_save_name} | manual={manual_operation}")
        else:
            print(f"❌ 에러: {folder_name}/{file_name} | {result.get('error')}")

# =========================
# Save preprocessing log
# =========================
log_path = os.path.join(output_base_dir, "preprocessing_log.csv")

with open(log_path, "w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "folder",
        "filename",
        "output_image",
        "output_mask",
        "status",
        "rotation_angle",
        "manual_operation",
        "object_size_before_canvas",
        "error",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(log_rows)

success_count = sum(row["status"] == "success" for row in log_rows)
failed_count = sum(row["status"] == "failed" for row in log_rows)
manual_count = sum(row["manual_operation"] not in ["none", "", None] for row in log_rows)

print("\n🎉 PNG 기반 전처리 데이터셋 구축 완료")
print(f"📁 Output image dir: {output_image_base_dir}")
print(f"📁 Output mask dir: {output_mask_base_dir}")
print(f"📄 Log file: {log_path}")
print(f"✅ Success: {success_count}")
print(f"❌ Failed: {failed_count}")
print(f"🔧 Manual corrections applied: {manual_count}")
