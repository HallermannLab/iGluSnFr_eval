import os
import numpy as np
import pandas as pd

from read_roi import read_roi_zip
from skimage import io

from diff_image import calculate_diff_image
from roi_utils import build_roi_masks, load_image_2d
from plotting_roi_pdf import plot_roi_pdf
from mito_intensity import mean_intensity_in_mask

import shutil
import subprocess


SPECIAL_INDUCTION_TIF = "ind.tif"
SPECIAL_INDUCTION_CSV = "ind.csv"

VIDEO_TIFS = [SPECIAL_INDUCTION_TIF]


def _video_exists(tif_path: str) -> bool:
    if os.path.exists(tif_path):
        return True
    mp4_path = os.path.splitext(tif_path)[0] + ".mp4"
    return os.path.exists(mp4_path)


def _read_stack_prefer_tif_else_mp4(tif_path: str) -> np.ndarray:
    if os.path.exists(tif_path):
        stk = io.imread(tif_path)
        if stk.ndim == 2:
            stk = stk[None, ...]
        return stk

    mp4_path = os.path.splitext(tif_path)[0] + ".mp4"
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"No video found: {tif_path} (also checked {mp4_path})")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH (required to decode MP4 for analysis).")

    probe_cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "default=noprint_wrappers=1",
        mp4_path,
    ]
    probe_out = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
    info = {}
    for line in probe_out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()

    w = int(info["width"])
    h = int(info["height"])

    dec_cmd = [
        ffmpeg,
        "-hide_banner",
        "-v",
        "error",
        "-i",
        mp4_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-",
    ]
    raw = subprocess.check_output(dec_cmd, stderr=subprocess.STDOUT)

    frame_bytes = w * h * 2
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(
            f"Decoded MP4 size not divisible by frame size: {mp4_path} "
            f"({len(raw)} bytes, {frame_bytes} bytes/frame)"
        )

    n_frames = len(raw) // frame_bytes
    return np.frombuffer(raw, dtype=np.uint16).reshape((n_frames, h, w))


def extract_roi_csvs(block_path, output_folder_csvs):
    """
    Generate CSVs from ROIs for each video tif in VIDEO_TIFS.
    Returns dict csv_name -> DataFrame
    """
    os.makedirs(output_folder_csvs, exist_ok=True)

    rois_zip_path = os.path.join(block_path, "ROIs.zip")
    rois_data = read_roi_zip(rois_zip_path)

    image_shape_hw = None
    for tif in VIDEO_TIFS:
        p = os.path.join(block_path, tif)
        if _video_exists(p):
            stk = _read_stack_prefer_tif_else_mp4(p)
            image_shape_hw = (stk.shape[1], stk.shape[2]) if stk.ndim == 3 else stk.shape
            break
    if image_shape_hw is None:
        raise FileNotFoundError(f"Could not determine image shape (missing {SPECIAL_INDUCTION_TIF} or ind.mp4).")

    roi_masks = build_roi_masks(rois_data, image_shape_hw)
    if not roi_masks:
        raise ValueError("No valid ROI masks produced.")

    dict_csv_dfs = {}
    for tif in VIDEO_TIFS:
        p = os.path.join(block_path, tif)
        if not _video_exists(p):
            continue

        stk = _read_stack_prefer_tif_else_mp4(p)
        if stk.ndim == 2:
            stk = stk[None, ...]
        if stk.ndim != 3:
            continue

        data_cols = {}
        for roi_name, mask in roi_masks.items():
            if mask.shape != stk.shape[1:]:
                h = min(mask.shape[0], stk.shape[1])
                w = min(mask.shape[1], stk.shape[2])
                m = mask[:h, :w]
                sub = stk[:, :h, :w]
            else:
                m = mask
                sub = stk

            if not m.any():
                continue
            data_cols[roi_name] = sub[:, m].mean(axis=1)

        if not data_cols:
            continue

        df_out = pd.DataFrame(data_cols)
        csv_name = tif.replace(".tif", ".csv")  # ind.tif -> ind.csv
        df_out.to_csv(os.path.join(output_folder_csvs, csv_name), index=False)
        dict_csv_dfs[csv_name] = df_out

    return rois_data, roi_masks, dict_csv_dfs


def process_special_block(
    *,
    block_path,
    block_name,
    recording_params,
    output_folder_experiment,
):
    """
    Special blocks (names length > 1), do ONLY:
      a) diff image with *_Induction params (from ind.tif)
      b) if ROIs.zip exists: csvs from ROIs (from ind.tif)
      c) if ROIs.zip + mito.tif exist: ROI PDFs
      d) no release prob / amplitude analysis
    """
    output_rois = os.path.join(output_folder_experiment, "ROIs")
    output_diff = os.path.join(output_folder_experiment, "DiffImage")
    output_csvs = os.path.join(output_folder_experiment, "CSVs", block_name)

    os.makedirs(output_rois, exist_ok=True)
    os.makedirs(output_diff, exist_ok=True)
    os.makedirs(output_csvs, exist_ok=True)

    ind_path = os.path.join(block_path, SPECIAL_INDUCTION_TIF)
    if not _video_exists(ind_path):
        print(f"    Warning: {ind_path} (or ind.mp4) not found. Skipping special block.")
        return {"mito_rows": []}

    # Always compute diff image (does not require ROIs.zip)
    diff_path = os.path.join(output_diff, f"{block_name}_diff.tif")
    try:
        calculate_diff_image(ind_path, diff_path, recording_params, param_suffix="_Induction")
    except Exception as e:
        print(f"    Warning: Failed to compute diff image for special block {block_name}: {e}")
        return {"mito_rows": []}

    rois_zip_path = os.path.join(block_path, "ROIs.zip")
    if not os.path.exists(rois_zip_path):
        print(f"    Info: {rois_zip_path} not found. Skipping ROI extraction and ROI PDFs for special block {block_name}.")
        return {"mito_rows": []}

    mito_path = os.path.join(block_path, "mito.tif")
    if not os.path.exists(mito_path):
        print(f"    Info: {mito_path} not found. Skipping ROI PDFs and mito intensity for special block {block_name}.")
        return {"mito_rows": []}

    # ROI-based work (CSVs + PDFs + mito means)
    try:
        rois_data, roi_masks, dict_csv_dfs = extract_roi_csvs(block_path, output_csvs)
    except Exception as e:
        print(f"    Warning: ROI extraction failed for special block {block_name}: {e}")
        return {"mito_rows": []}

    try:
        mito_img = load_image_2d(mito_path, mode="first")
        diff_img = load_image_2d(diff_path, mode="first")
    except Exception as e:
        print(f"    Warning: Failed to load images for special block {block_name}: {e}")
        return {"mito_rows": []}

    acq_time = float(recording_params["acquisition time (ms)"])
    zoom_size = float(recording_params.get("ZoomSize", 40))

    mito_rows = []
    for roi_name, roi_geom in rois_data.items():
        if roi_name not in roi_masks:
            continue

        # Full trace: ind.csv
        if SPECIAL_INDUCTION_CSV in dict_csv_dfs and roi_name in dict_csv_dfs[SPECIAL_INDUCTION_CSV].columns:
            full_y = dict_csv_dfs[SPECIAL_INDUCTION_CSV][roi_name].to_numpy(dtype=float)
            full_t = np.arange(len(full_y)) * acq_time
        else:
            full_y = np.array([], dtype=float)
            full_t = np.array([], dtype=float)

        pdf_path = os.path.join(output_rois, f"{roi_name}_{block_name}.pdf")
        plot_roi_pdf(
            save_path=pdf_path,
            block_name=block_name,
            roi_name=roi_name,
            roi_geom=roi_geom,
            diff_img=diff_img,
            mito_img=mito_img,
            zoom_size=zoom_size,
            full_train_time_ms=full_t,
            full_train_trace=full_y,
        )

        mito_mean = mean_intensity_in_mask(mito_img, roi_masks[roi_name])
        mito_rows.append([roi_name, mito_mean])

    return {"mito_rows": mito_rows}