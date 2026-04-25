#Simplified version of AAZ_disks_regions_to_excel.py

#!/usr/bin/env python3
#source venv/bin/activate       ───> Command to activate the relevant python environment

"""
AAZ Disks Region Stats — Simplified version
============================================
Automatically processes all *_disk.vtk files in a folder and outputs
per-region statistics (Median, P10, Q1, Q3, IQR) to separate Excel sheets.

Usage (terminal commands):
-----
Basic (no metadata):
    python AAZ_Disks_region_stats.py --vtk-dir /path/to/vtk/folder --out results.xlsx

With optional metadata (recurrence / FAtype columns):
    python AAZ_Disks_region_stats.py --vtk-dir /path/to/vtk/folder --student-xlsx metadata.xlsx --out results.xlsx 
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CODE INDEX

    # [50]  Name normalization (mild): consistent underscores, remove extension, etc.
    # [60]  Per region stat calculations: median, P10, Q1, Q3 and IQR
    # [160] Mesh processing
    # [195] Metadata loader: loads recurrence and FAtype from an optional Excel file, matching by normalized map name.
    # [135] Row builder
    # [250] Main: running the previously defined functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


# ── Constants ─────────────────────────────────────────────────────────────────

REGION_COUNT = 24


# ── Name normalisation (kept for clean output labels) ─────────────────────────

def normalize_name(s: str) -> str:
    """Produce a clean, consistent map name from a filename or string."""
    s = s.strip()
    s = re.sub(r"\.vtk$",   "", s, flags=re.IGNORECASE)
    s = re.sub(r"_disk$",   "", s, flags=re.IGNORECASE)
    s = s.replace(" ", "_")
    return s

# ── Per-region stat functions ─────────────────────────────────────────────────

def compute_region_median(scar_c: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ Median of cell-averaged scar per region """
    result = np.full(REGION_COUNT, np.nan, dtype=float)
    for r in range(1, REGION_COUNT + 1):
        mask = labels == r
        if np.any(mask):
            result[r - 1] = np.nanmedian(scar_c[mask])
    return result


def compute_region_p10(scar_c: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ P10 of cell-averaged scar per region """
    result = np.full(REGION_COUNT, np.nan, dtype=float)
    for r in range(1, REGION_COUNT + 1):
        mask = labels == r
        if np.any(mask):
            result[r - 1] = np.nanpercentile(scar_c[mask], 10)
    return result


def compute_region_q1(scar_c: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ Q1 (P25) of cell-averaged scar per region """
    result = np.full(REGION_COUNT, np.nan, dtype=float)
    for r in range(1, REGION_COUNT + 1):
        mask = labels == r
        if np.any(mask):
            result[r - 1] = np.nanpercentile(scar_c[mask], 25)
    return result


def compute_region_q3(scar_c: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ Q3 (P75) of cell-averaged scar per region """
    result = np.full(REGION_COUNT, np.nan, dtype=float)
    for r in range(1, REGION_COUNT + 1):
        mask = labels == r
        if np.any(mask):
            result[r - 1] = np.nanpercentile(scar_c[mask], 75)
    return result


def compute_region_iqr(scar_c: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ IQR (P75 - P25) of cell-averaged scar per region """
    result = np.full(REGION_COUNT, np.nan, dtype=float)
    for r in range(1, REGION_COUNT + 1):
        mask = labels == r
        if np.any(mask):
            p25, p75 = np.nanpercentile(scar_c[mask], [25, 75])
            result[r - 1] = p75 - p25
    return result


def compute_global_q2(all_median_rows: list[dict]) -> dict:
    """
    Compute the cross-patient Q2 (median) per region.
    Takes the per-file median rows and computes the median across all files
    for each region column (R01..R24).
    Returns a single-row dict suitable for a summary sheet.
    """
    df = pd.DataFrame(all_median_rows)
    region_cols_list = [f"R{r:02d}" for r in range(1, REGION_COUNT + 1)]
    summary = {"mapNames": "GLOBAL_Q2", "recurrence": np.nan, "FAtype": np.nan}
    for col in region_cols_list:
        if col in df.columns:
            summary[col] = np.nanmedian(df[col].values.astype(float))
        else:
            summary[col] = np.nan
    summary["Q1"] = np.nan
    summary["Q2"] = np.nan
    summary["Q3"] = np.nan
    return summary


def compute_global_iqr(all_median_rows: list[dict]) -> dict:
    """
    Compute the cross-patient IQR per region.
    Takes the per-file median rows and computes IQR (P75 - P25) across all
    files for each region column (R01..R24).
    Returns a single-row dict suitable for a summary sheet.
    """
    df = pd.DataFrame(all_median_rows)
    region_cols_list = [f"R{r:02d}" for r in range(1, REGION_COUNT + 1)]
    summary = {"mapNames": "GLOBAL_IQR", "recurrence": np.nan, "FAtype": np.nan}
    for col in region_cols_list:
        if col in df.columns:
            vals = df[col].values.astype(float)
            p25, p75 = np.nanpercentile(vals[~np.isnan(vals)], [25, 75]) if np.any(~np.isnan(vals)) else (np.nan, np.nan)
            summary[col] = p75 - p25
        else:
            summary[col] = np.nan
    summary["Q1"] = np.nan
    summary["Q2"] = np.nan
    summary["Q3"] = np.nan
    return summary





# ── Mesh processing ───────────────────────────────────────────────────────────

def process_mesh(mesh: pv.DataSet) -> tuple:
    """
    Extract per-region stats from a single VTK mesh.

    Returns
    -------
    median[24], p10[24], q1[24], q3[24], iqr[24]  — per-region arrays
    (gq1, gq2, gq3)                               — global point-scar quantiles
    """
    if "scar" not in mesh.point_data:
        raise KeyError("Missing point_data array 'scar'")
    if "sumlabels" not in mesh.cell_data:
        raise KeyError("Missing cell_data array 'sumlabels'")

    # Global quantiles from point data
    scar_p = np.asarray(mesh.point_data["scar"], dtype=float)
    gq1, gq2, gq3 = np.nanpercentile(scar_p, [25, 50, 75])

    # Convert point data to cell-averaged values
    mesh_c = mesh.point_data_to_cell_data()
    scar_c = np.asarray(mesh_c.cell_data["scar"],      dtype=float)
    labels = np.asarray(mesh_c.cell_data["sumlabels"], dtype=float)

    median = compute_region_median(scar_c, labels)
    p10    = compute_region_p10(scar_c, labels)
    q1     = compute_region_q1(scar_c, labels)
    q3     = compute_region_q3(scar_c, labels)
    iqr    = compute_region_iqr(scar_c, labels)

    return median, p10, q1, q3, iqr, (float(gq1), float(gq2), float(gq3))



# ── Optional metadata loader ──────────────────────────────────────────────────

def load_metadata(student_xlsx: Path | None) -> dict[str, tuple[float, float]]:
    """
    Load recurrence and FAtype columns from an optional Excel file.
    Returns a dict: normalised_map_name -> (recurrence, FAtype).
    If no file is provided, returns an empty dict (columns will be NaN).
    """
    if student_xlsx is None:
        return {}

    df = pd.read_excel(student_xlsx)

    name_col = next(
        (c for c in ["mapNames", "mapName", "mapname", "MapNames", "MapName"] if c in df.columns),
        None
    )
    if name_col is None:
        raise ValueError(f"Could not find map name column in {student_xlsx}. Columns: {list(df.columns)}")

    rec_col = next(
        (c for c in ["recurrence", "Recurrencia", "recurrencia", "Recurrence"] if c in df.columns),
        None
    )
    fa_col = next(
        (c for c in ["FAtype", "FA_type", "tipoFA", "FAType"] if c in df.columns),
        None
    )

    meta: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        nm  = normalize_name(str(row[name_col]))
        rec = float(row[rec_col]) if rec_col and pd.notna(row[rec_col]) else np.nan
        fa  = float(row[fa_col])  if fa_col  and pd.notna(row[fa_col])  else np.nan
        meta[nm] = (rec, fa)

    return meta



# ── Row-building helpers ──────────────────────────────────────────────────────


def make_base_row(nm_norm: str, rec: float, fa: float) -> dict:
    return {"mapNames": nm_norm, "recurrence": rec, "FAtype": fa}


def region_cols(values: np.ndarray) -> dict:
    return {f"R{r:02d}": values[r - 1] for r in range(1, REGION_COUNT + 1)}


def global_quantile_cols(gq1: float, gq2: float, gq3: float) -> dict:
    return {"Q1": gq1, "Q2": gq2, "Q3": gq3}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-region scar statistics from all *_disk.vtk files in a folder."
    )
    ap.add_argument("--vtk-dir",      type=Path, required=True,
                    help="Folder containing *_disk.vtk files.")
    ap.add_argument("--student-xlsx", type=Path, default=None,
                    help="Optional Excel file with recurrence / FAtype metadata.")
    ap.add_argument("--out",          type=Path, required=True,
                    help="Output Excel file path (e.g. results.xlsx).")
    args = ap.parse_args()

    # --- Auto-discover all VTK files (no .txt needed) ---
    vtk_files = sorted(args.vtk_dir.glob("*_disk.vtk"))

    if not vtk_files:
        print(f"WARNING: No *_disk.vtk files found in {args.vtk_dir}")
        return

    print(f"Found {len(vtk_files)} VTK file(s) to process.")

    meta = load_metadata(args.student_xlsx)

    nan_regions = np.full(REGION_COUNT, np.nan)
    
    median_rows, p10_rows, q1_rows, q3_rows, iqr_rows = [], [], [], [], []
    errors = []

    for vtk_path in vtk_files:
        nm_norm = normalize_name(vtk_path.stem)   # stem strips .vtk extension
        rec, fa = meta.get(nm_norm, (np.nan, np.nan))

        try:
            mesh = pv.read(str(vtk_path))
            median, p10, q1, q3, iqr, (gq1, gq2, gq3) = process_mesh(mesh)
        except Exception as e:
            print(f"  ERROR processing {vtk_path.name}: {e}")
            errors.append(vtk_path.name)
            median = p10 = q1 = q3 = iqr = nan_regions
            gq1 = gq2 = gq3 = np.nan

        base = make_base_row(nm_norm, rec, fa)
        gq   = global_quantile_cols(gq1, gq2, gq3)

        median_rows.append({**base, **region_cols(median), **gq})
        p10_rows.append(   {**base, **region_cols(p10),    **gq})
        q1_rows.append(    {**base, **region_cols(q1),     **gq})
        q3_rows.append(    {**base, **region_cols(q3),     **gq})
        iqr_rows.append(   {**base, **region_cols(iqr),    **gq})

        print(f"  Processed: {vtk_path.name}")

    # Column order shared across all sheets
    cols = (
        ["mapNames", "recurrence", "FAtype"]
        + [f"R{r:02d}" for r in range(1, REGION_COUNT + 1)]
        + ["Q1", "Q2", "Q3"]
    )

    # Write one sheet per statistic calculation
    global_q2_row  = compute_global_q2(median_rows)
    global_iqr_row = compute_global_iqr(median_rows)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        pd.DataFrame(median_rows)[cols].to_excel(writer, sheet_name="Median",     index=False)
        pd.DataFrame(p10_rows)[cols].to_excel(   writer, sheet_name="P10",        index=False)
        pd.DataFrame(q1_rows)[cols].to_excel(    writer, sheet_name="Q1",         index=False)
        pd.DataFrame(q3_rows)[cols].to_excel(    writer, sheet_name="Q3",         index=False)
        pd.DataFrame(iqr_rows)[cols].to_excel(   writer, sheet_name="IQR",        index=False)
        pd.DataFrame([global_q2_row])[cols].to_excel( writer, sheet_name="Global_Q2",  index=False)
        pd.DataFrame([global_iqr_row])[cols].to_excel(writer, sheet_name="Global_IQR", index=False)

    if errors:
        print(f"\nWARNING: {len(errors)} file(s) failed to process: {errors}")

    print(f"\nDone. 7 excel sheets (Median, P10, Q1, Q3, IQR, Global_Q2, Global_IQR) have been written and stored into: {args.out}")


if __name__ == "__main__":
    main()



"""
AAZ Disks Region Stats Excel, Terminal Commands
============================================
Activate python environment:
    source venv/bin/activate 

Usage (terminal commands):
-----
Basic (no metadata):
    python3 AAZ_Simplified_PLUS_VTK_to_excel.py 
    --vtk-dir /home/data/Documents/DATA_MAPS/MapDisks_NEW
    --out /home/data/Documents/DATA_MAPS/RESULTS_FILE_NAME.xlsx

 #   python3 AAZ_Simplified_PLUS_VTK_to_excel.py --vtk-dir /home/data/Documents/DATA_MAPS/MapDisks_NEW --out /home/data/Documents/DATA_MAPS/RESULTS_FILE_NAME.xlsx
 
With optional metadata (recurrence / FAtype columns):
    python3 AAZ_Simplified_PLUS_VTK_to_excel.py 
    --vtk-dir /home/data/Documents/DATA_MAPS/MapDisks_NEW
    --student-xlsx /home/data/Documents/DATA_MAPS/data_disks_medianvolt_redo_FAtype_PYTHON_V02.xlsx 
    --out /home/data/Documents/DATA_MAPS/RESULTS_FILE_NAME.xlsx

 #   python3 AAZ_Simplified_PLUS_VTK_to_excel.py --vtk-dir /home/data/Documents/DATA_MAPS/MapDisks_NEW --student-xlsx /home/data/Documents/DATA_MAPS/data_disks_medianvolt_redo_FAtype_PYTHON_V02.xlsx --out /home/data/Documents/DATA_MAPS/RESULTS_FILE_NAME.xlsx

Output sheets (7 total):
    Median      — per-file median per region
    P10         — per-file P10 per region
    Q1          — per-file Q1 per region
    Q3          — per-file Q3 per region
    IQR         — per-file IQR per region
    Global_Q2   — single summary row: median across ALL files per region (R01–R24)
    Global_IQR  — single summary row: IQR across ALL files per region (R01–R24)
"""