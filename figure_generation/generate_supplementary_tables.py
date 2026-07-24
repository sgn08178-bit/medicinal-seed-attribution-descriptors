#!/usr/bin/env python3
"""Create submission-ready Supplementary Tables S1-S5 from existing source files."""

from __future__ import annotations

import os

import html
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SOURCE_DATA_ROOT = Path(
    os.environ.get("MEDICINAL_SEED_SOURCE_DATA_ROOT", ROOT / "source_data")
).resolve()
RESULTS_ROOT = Path(os.environ.get("MEDICINAL_SEED_RESULTS_ROOT", ROOT / "results")).resolve()
OUT = Path(
    os.environ.get(
        "MEDICINAL_SEED_SUPPLEMENTARY_TABLE_OUTPUT",
        RESULTS_ROOT / "supplementary_tables",
    )
).resolve()
STAGE = SOURCE_DATA_ROOT
INV_SRC = Path(
    os.environ.get(
        "MEDICINAL_SEED_DESCRIPTOR_INVENTORY_CSV",
        SOURCE_DATA_ROOT / "descriptor_map_inventory.csv",
    )
).resolve()
PDF_TEXT = Path(
    os.environ.get(
        "MEDICINAL_SEED_MANUSCRIPT_TEXT",
        SOURCE_DATA_ROOT / "manuscript_text.txt",
    )
).resolve()


DISPLAY_REPLACEMENTS = {
    "Saturation_HSV": "HSV Saturation",
    "HSV saturation": "HSV Saturation",
    "FourierDescriptor": "Fourier descriptor",
    "Curvature_Laplacian": "Laplacian-based local variation",
    "Edge_Sobel": "Sobel edge response",
    "Local binary pattern (LBP)": "Local binary pattern",
}


def clean_name(value: str) -> str:
    s = str(value)
    s = s.replace("θ", "theta").replace("°", " deg")
    s = re.sub(r"\s+", " ", s).strip()
    replacements = {
        "Saturation_HSV": "HSV Saturation",
        "HSV saturation": "HSV Saturation",
        "FourierDescriptor": "Fourier descriptor",
        "Curvature_Laplacian": "Laplacian-based local variation",
        "Edge_Sobel": "Sobel edge response",
        "Local binary pattern (LBP)": "Local binary pattern",
    }
    return replacements.get(s, s)


def final_name(value: str) -> str:
    s = clean_name(value)
    return s


def name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", final_name(value).lower())


def descriptor_mappings() -> tuple[dict[str, str], dict[str, str]]:
    inv = pd.read_csv(INV_SRC)
    display_to_key = {}
    key_to_display = {}
    for _, row in inv.iterrows():
        display = final_name(row["display_name"])
        display_to_key[name_key(display)] = row["descriptor"]
        key_to_display[row["descriptor"]] = display
    return display_to_key, key_to_display


DISPLAY_TO_KEY, KEY_TO_DISPLAY = descriptor_mappings()


def descriptor_property_and_params(key: str, name: str) -> tuple[str, str, str]:
    if key == "Brightness":
        return "Mean RGB intensity", "RGB image", "Mean RGB/grayscale intensity."
    if key == "LAB_L":
        return "Lightness in CIELAB color space", "RGB image converted to CIELAB", "CIELAB L channel."
    if key == "LAB_Chroma":
        return "Chroma magnitude in CIELAB color space", "RGB image converted to CIELAB", "Chroma magnitude from LAB color components."
    if key == "Saturation_HSV":
        return "Saturation in HSV color space", "RGB image converted to HSV", "HSV saturation channel."
    if key == "FFT_LowPass":
        return "Low spatial-frequency intensity structure", "Grayscale intensity image", "FFT low-pass filtering; low-pass ratio = 0.25."
    if key == "FFT_HighPass":
        return "High spatial-frequency intensity structure", "Grayscale intensity image", "FFT high-pass component using low-pass ratio = 0.25."
    if key.startswith("Wavelet_"):
        parts = key.split("_")
        level = parts[1].replace("L", "")
        direction = {"H": "horizontal", "V": "vertical", "D": "diagonal"}[parts[2]]
        return "Wavelet detail coefficients", "Grayscale intensity image", f"Wavelet db2; level {level}; {direction} detail."
    if key.startswith("Gabor_"):
        m = re.search(r"f([0-9.]+)_t([0-9]+)", key)
        freq = m.group(1) if m else "[MISSING]"
        theta = m.group(2) if m else "[MISSING]"
        return "Orientation- and frequency-selective texture response", "Grayscale intensity image", f"Gabor response; frequency = {freq}; theta = {theta} deg."
    if key == "LBP":
        return "Local binary texture pattern", "Grayscale intensity image", "LBP radius = 2; n_points = 16; method = uniform."
    if key == "Edge_Sobel":
        return "First-order edge response", "Grayscale intensity image", "Sobel gradient magnitude."
    if key == "Curvature_Laplacian":
        return "Second-order intensity response", "Grayscale intensity image", "Laplacian-based local variation."
    if key == "DistanceTransform":
        return "Mask-derived interior distance", "Foreground mask", "Distance of each foreground pixel from the seed boundary."
    if key == "FourierDescriptor":
        return "Contour-derived shape representation", "Foreground contour", "Fourier descriptor; number of components = 20."
    return "[MISSING]", "[MISSING]", "[MISSING]"


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write a minimal valid XLSX workbook without external dependencies."""
    sheet_xmls: list[tuple[str, str]] = []
    workbook_sheets = []
    workbook_rels = []
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]

    for idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        safe_name = sheet_name[:31]
        workbook_sheets.append(f'<sheet name="{html.escape(safe_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        rows = []
        values = [list(df.columns)] + df.astype(object).where(pd.notna(df), "NA").values.tolist()
        for r_idx, row in enumerate(values, start=1):
            cells = []
            for c_idx, value in enumerate(row, start=1):
                col = column_letter(c_idx)
                text = html.escape(str(value))
                cells.append(f'<c r="{col}{r_idx}" t="inlineStr"><is><t>{text}</t></is></c>')
            rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
        xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>{''.join(rows)}</sheetData>
</worksheet>
"""
        sheet_xmls.append((f"xl/worksheets/sheet{idx}.xml", xml))

    workbook_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{''.join(workbook_sheets)}</sheets>
</workbook>
"""
    rels_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(workbook_rels)}
</Relationships>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {''.join(overrides)}
</Types>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Times New Roman"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="1"><xf/></cellXfs></styleSheet>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        zf.writestr("xl/styles.xml", styles)
        for name, xml in sheet_xmls:
            zf.writestr(name, xml)


def column_letter(n: int) -> str:
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def make_s1() -> pd.DataFrame:
    inv = pd.read_csv(INV_SRC)
    rows = []
    for _, row in inv.iterrows():
        key = row["descriptor"]
        name = final_name(row["display_name"])
        visual, input_type, params = descriptor_property_and_params(key, name)
        rows.append(
            {
                "Category": row["category"],
                "Descriptor key": key,
                "Descriptor name": name,
                "Visual property represented": visual,
                "Input image type": input_type,
                "Main parameters": params,
                "Normalization": "Map-wise min-max normalization to [0, 1]",
                "Output size": "224 x 224 pixels",
                "Used in spatial association": "Yes",
                "Used in descriptor summary classification": "Yes" if bool(row["valid_for_stage7c"]) else "No",
            }
        )
    return pd.DataFrame(rows)


def make_assoc(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    rows = []
    for i, row in df.iterrows():
        name = final_name(row["Descriptor map"])
        key = DISPLAY_TO_KEY.get(name_key(name), "[MISSING]")
        rows.append(
            {
                "Rank": i + 1,
                "Descriptor key": key,
                "Descriptor name": name,
                "Category": row["category"],
                "Mean Spearman r": row["mean_spearman_r"],
                "SD": row["sd"],
                "n": int(row["n"]),
                "Test statistic": row["t_statistic"],
                "p-value": row["p_value"],
                "FDR-adjusted p-value": row["fdr_adjusted_p_value"],
                "Significant after FDR": "Yes" if bool(row["significant_fdr_0.05"]) else "No",
            }
        )
    return pd.DataFrame(rows)


def make_s5() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(STAGE / "supplementary_tables/Supplementary_Table_S5_descriptor_summary_classifier_all_subsets.csv")
    subset_cols = ["subset_id", "subset_label", "included_descriptor_names", "n_descriptor_maps", "n_numeric_features"]
    subset_df = src[subset_cols].drop_duplicates("subset_id").copy()
    subset_df["included_descriptor_names"] = subset_df["included_descriptor_names"].map(clean_descriptor_list)
    s5a = subset_df.rename(
        columns={
            "subset_label": "Feature set",
            "included_descriptor_names": "Descriptor maps included",
            "n_descriptor_maps": "Number of descriptor maps",
            "n_numeric_features": "Number of features",
        }
    )[["Feature set", "Descriptor maps included", "Number of descriptor maps", "Number of features"]]

    s5b = src.rename(
        columns={
            "subset_label": "Feature set",
            "n_descriptor_maps": "Number of descriptor maps",
            "n_numeric_features": "Number of features",
            "model_label": "Classifier",
            "test_accuracy": "Accuracy",
            "test_macro_precision": "Macro precision",
            "test_macro_recall": "Macro recall",
            "test_macro_f1": "Macro F1",
        }
    )[
        [
            "Feature set",
            "Number of descriptor maps",
            "Number of features",
            "Classifier",
            "Accuracy",
            "Macro precision",
            "Macro recall",
            "Macro F1",
        ]
    ].copy()

    combined_rows = []
    for _, row in s5a.iterrows():
        combined_rows.append({"Section": "S5a. Descriptor feature subsets", **row.to_dict()})
    for _, row in s5b.iterrows():
        combined_rows.append({"Section": "S5b. Classification performance", **row.to_dict()})
    combined = pd.DataFrame(combined_rows).fillna("NA")
    return s5a, s5b, combined


def clean_descriptor_list(value: str) -> str:
    parts = [final_name(p.strip()) for p in str(value).split(";")]
    return "; ".join(parts)


def save_table(df: pd.DataFrame, basename: str, sheets: dict[str, pd.DataFrame] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{basename}.csv", index=False)
    write_xlsx(OUT / f"{basename}.xlsx", sheets or {basename[:31]: df})


def fmt_value(value, decimals: int = 4) -> str:
    if value == "NA" or value == "[MISSING]":
        return str(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return "NA"
    return f"{x:.{decimals}f}"


def fmt_p(value) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return "NA"
    if x != 0 and x < 0.0001:
        return "<0.0001"
    return f"{x:.4f}"


def compact_assoc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Mean Spearman r", "SD", "Test statistic"]:
        out[c] = out[c].map(fmt_value)
    for c in ["p-value", "FDR-adjusted p-value"]:
        out[c] = out[c].map(fmt_p)
    return out


def compact_s5b(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]:
        out[c] = out[c].map(fmt_value)
    return out


def md_table(df: pd.DataFrame, max_col_width: int = 80) -> str:
    tmp = df.copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].astype(str).map(lambda s: s if len(s) <= max_col_width else s[: max_col_width - 3] + "...")
    headers = [str(c) for c in tmp.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in tmp.iterrows():
        vals = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in tmp.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def create_word_md(s1, s2, s3, s4, s5a, s5b, missing: list[str], warnings: list[str]) -> None:
    note_s1 = "Descriptor maps were generated in the 224 × 224 pixel coordinate system and normalized to the range [0, 1] using map-wise min–max normalization unless otherwise specified."
    note_assoc = "Spearman correlations were calculated between attribution maps and descriptor maps using only foreground pixels. Image-level coefficients from the independent test set were summarized for each descriptor. n = 225 unless otherwise specified. Adjusted p-values were calculated using the Benjamini-Hochberg false discovery rate procedure."
    note_s5 = "Six foreground-restricted summary statistics were calculated for each descriptor map: mean, standard deviation, median, 90th percentile, top 10% mean, and interquartile range. Classification metrics were calculated on the independent test set."

    parts = [
        "# Supplementary Tables for Word",
        "",
        "## Supplementary Table S1. Full descriptor inventory",
        "",
        md_table(s1),
        "",
        f"Table note: {note_s1}",
        "",
        "## Supplementary Table S2. Full descriptor association results for zero-baseline absolute IG",
        "",
        md_table(compact_assoc(s2)),
        "",
        f"Table note: {note_assoc}",
        "",
        "## Supplementary Table S3. Full descriptor association results for positive IG",
        "",
        md_table(compact_assoc(s3)),
        "",
        f"Table note: {note_assoc}",
        "",
        "## Supplementary Table S4. Full descriptor association results for ConvNeXt-Small Grad-CAM",
        "",
        md_table(compact_assoc(s4)),
        "",
        f"Table note: {note_assoc}",
        "",
        "## Supplementary Table S5a. Descriptor feature subsets",
        "",
        md_table(s5a, max_col_width=100),
        "",
        "## Supplementary Table S5b. Classification performance",
        "",
        md_table(compact_s5b(s5b)),
        "",
        f"Table note: {note_s5}",
        "",
        "## Missing Values and Source Warnings",
        "",
    ]
    if missing:
        parts.extend(f"- {m}" for m in missing)
    else:
        parts.append("- No required values were missing.")
    if warnings:
        parts.append("")
        parts.extend(f"- {w}" for w in warnings)
    (OUT / "Supplementary_Tables_for_Word.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_manuscript_values(s2: pd.DataFrame) -> list[str]:
    warnings = []
    if PDF_TEXT.exists():
        text = PDF_TEXT.read_text(encoding="utf-8")
        expected = [
            ("LAB L", "0.4857"),
            ("Brightness", "0.4849"),
            ("FFT low-pass", "0.4804"),
        ]
        for desc, rounded in expected:
            row = s2.loc[s2["Descriptor name"] == desc]
            if row.empty:
                warnings.append(f"Could not find {desc} in Supplementary Table S2 for manuscript cross-check.")
                continue
            table_value = f"{float(row.iloc[0]['Mean Spearman r']):.4f}"
            if table_value != rounded:
                warnings.append(f"Manuscript-check mismatch for {desc}: source table rounds to {table_value}, expected manuscript value {rounded}.")
            elif rounded not in text:
                warnings.append(f"Source value for {desc} rounds to {table_value}, but that rounded value was not detected in extracted PDF text.")
    else:
        warnings.append("Extracted PDF text not found; manuscript value cross-check was limited to source tables.")
    return warnings


def main() -> None:
    missing: list[str] = []
    warnings: list[str] = []

    s1 = make_s1()
    s2 = make_assoc(STAGE / "supplementary_tables/Supplementary_Table_S2_full_ig_descriptor_association.csv")
    s3 = make_assoc(STAGE / "supplementary_tables/Supplementary_Table_S3_positive_ig_descriptor_association.csv")
    s4 = make_assoc(STAGE / "supplementary_tables/Supplementary_Table_S4_gradcam_descriptor_association.csv")
    s5a, s5b, s5 = make_s5()

    for table_name, df in [("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4), ("S5a", s5a), ("S5b", s5b)]:
        if (df == "[MISSING]").any().any():
            missing.append(f"{table_name} contains [MISSING] values.")

    if set(s4["n"].unique()) != {225}:
        warnings.append(f"Supplementary Table S4 uses n values {sorted(s4['n'].unique().tolist())}; this differs from the default n = 225 note and is therefore shown explicitly in the n column.")
    warnings.extend(check_manuscript_values(s2))

    save_table(s1, "Supplementary_Table_S1_descriptor_inventory")
    save_table(s2, "Supplementary_Table_S2_zero_baseline_absolute_IG_descriptor_association")
    save_table(s3, "Supplementary_Table_S3_positive_IG_descriptor_association")
    save_table(s4, "Supplementary_Table_S4_convnext_gradcam_descriptor_association")
    save_table(
        s5,
        "Supplementary_Table_S5_descriptor_summary_classification_results",
        {"S5a_feature_subsets": s5a, "S5b_classification": s5b, "S5_combined": s5},
    )
    create_word_md(s1, s2, s3, s4, s5a, s5b, missing, warnings)

    source_report = f"""# Supplementary Table Source Report

## Source Files Used

- `{INV_SRC}`
- `{STAGE / 'supplementary_tables/Supplementary_Table_S2_full_ig_descriptor_association.csv'}`
- `{STAGE / 'supplementary_tables/Supplementary_Table_S3_positive_ig_descriptor_association.csv'}`
- `{STAGE / 'supplementary_tables/Supplementary_Table_S4_gradcam_descriptor_association.csv'}`
- `{STAGE / 'supplementary_tables/Supplementary_Table_S5_descriptor_summary_classifier_all_subsets.csv'}`
- `{STAGE / 'source_data/stage3_descriptor_association/final_stage3_config.yaml'}`
- `{ROOT / 'manuscript_tables/main_tables/Table2_descriptor_maps.csv'}`

## Missing Values

{chr(10).join('- ' + m for m in missing) if missing else '- No required values were missing.'}

## Source Warnings / Manuscript Cross-check

{chr(10).join('- ' + w for w in warnings) if warnings else '- No mismatches detected for checked manuscript values.'}

## Layout Warnings

- Supplementary Table S1 has 10 columns and should be placed on a landscape page or split if inserted directly into a Word/PDF supplementary file.
- Supplementary Tables S2-S4 have 11 columns each and should be placed on landscape pages.
- Supplementary Table S5a contains long descriptor-list cells and should be placed on a landscape page or kept as a full source table with a compact summary in the PDF.
"""
    (OUT / "Supplementary_Table_source_report.md").write_text(source_report, encoding="utf-8")


if __name__ == "__main__":
    main()
