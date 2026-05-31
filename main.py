try:
    import config
except ImportError:
    print(
        "\nERROR: 'config.py' not found.\n"
        "Please create a local 'config.py' by copying 'config_template.py' and "
        "adjusting the paths for your system.\n"
    )
    raise SystemExit(1)

import os
import time
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from read_roi import read_roi_zip
from skimage import io
from scipy.signal import butter, filtfilt

import git_save as myGit

from roi_utils import load_image_2d, build_roi_masks
from mito_intensity import mean_intensity_in_mask
from diff_image import calculate_diff_image
from averages_report import save_block_averages_pdf, save_special_block_averages
from special_blocks import process_special_block

import shutil
import subprocess


VIDEO_FILES = ["ap1+train.tif", "ap2.tif", "ap3.tif", "ap4.tif", "ap5.tif"]
VIDEO_CSV_FILES = [v.replace(".tif", ".csv") for v in VIDEO_FILES]


def _metadata_flag_is_true(recording_params, key: str, default: int = 1) -> bool:
    """
    Read metadata flags such as Recalc_CSV robustly from Excel.

    Treats 1, "1", "true", "yes", "y" as True.
    Treats 0, "0", "false", "no", "n" as False.
    """
    value = recording_params.get(key, default)

    if pd.isna(value):
        value = default

    if isinstance(value, str):
        value = value.strip().lower()
        return value in {"1", "true", "yes", "y"}

    return bool(int(value))


def _load_existing_csvs(csv_folder: str, expected_csv_files: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load existing CSV files from csv_folder.

    Returns an empty dict if none of the expected CSVs exist.
    """
    dict_csv_dfs = {}

    for csv_name in expected_csv_files:
        csv_path = os.path.join(csv_folder, csv_name)
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                dict_csv_dfs[csv_name] = df
        except Exception as e:
            print(f"    Warning: could not read cached CSV {csv_path}: {e}")

    return dict_csv_dfs


def _save_csv_cache_copy(df: pd.DataFrame, csv_name: str, *folders: str):
    """
    Save one CSV DataFrame to multiple folders.
    """
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        df.to_csv(os.path.join(folder, csv_name), index=False)

def _get_float_param(recording_params, key: str, default=np.nan) -> float:
    value = recording_params.get(key, default)
    if pd.isna(value):
        return default
    return float(value)


def _get_int_param(recording_params, key: str, default=0) -> int:
    value = recording_params.get(key, default)
    if pd.isna(value):
        return default
    return int(value)


def _iqr(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)

def _calculate_train_qc_for_block(dict_csv_dfs, block_name, recording_params):
    """
    Calculate Train_Bsl_median, Train_Bsl_IRQ, Train_Peak, Train_fold per ROI
    from ap1+train.csv for one block.

    Uses metadata columns:
      Train_Bsl_Start, Train_Bsl_End, Train_Peak_Start, Train_Peak_End
    """
    train_key = "ap1+train.csv"
    if train_key not in dict_csv_dfs:
        return {}

    df_train = dict_csv_dfs[train_key]
    if not isinstance(df_train, pd.DataFrame) or df_train.empty:
        return {}

    cols_to_drop = [c for c in df_train.columns if c in ["Average", "Err", " "]]
    if cols_to_drop:
        df_train = df_train.drop(cols_to_drop, axis=1)

    if df_train.empty:
        return {}

    acq_time = float(recording_params["acquisition time (ms)"])

    bsl_start = int(_get_float_param(recording_params, "Train_Bsl_Start") / acq_time)
    bsl_end = int(_get_float_param(recording_params, "Train_Bsl_End") / acq_time)
    peak_start = int(_get_float_param(recording_params, "Train_Peak_Start") / acq_time)
    peak_end = int(_get_float_param(recording_params, "Train_Peak_End") / acq_time)

    out = {}

    for roi in df_train.columns:
        y = df_train[roi].to_numpy(dtype=float)

        bsl_values = y[bsl_start:bsl_end]
        peak_values = y[peak_start:peak_end]

        if bsl_values.size == 0 or peak_values.size == 0:
            out[roi] = {
                f"Train_Bsl_median_{block_name}": "",
                f"Train_Bsl_IRQ_{block_name}": "",
                f"Train_Peak_{block_name}": "",
                f"Train_fold_{block_name}": "",
            }
            continue

        bsl_median = float(np.median(bsl_values))
        bsl_iqr = _iqr(bsl_values)
        peak = float(np.max(peak_values))

        if pd.isna(bsl_iqr) or bsl_iqr == 0:
            train_fold = np.nan
        else:
            train_fold = (peak - bsl_median) / bsl_iqr

        out[roi] = {
            f"Train_Bsl_median_{block_name}": bsl_median,
            f"Train_Bsl_IRQ_{block_name}": bsl_iqr,
            f"Train_Peak_{block_name}": peak,
            f"Train_fold_{block_name}": train_fold,
        }

    return out

def _append_qc_columns(df, qc_by_roi):
    """
    Add train QC columns and excluded-stimulus counts to a result dataframe.

    df must contain column ROI_number.

    The columns are detected dynamically from qc_by_roi instead of being
    hard-coded to blocks A/B/C. This prevents missing Excel values when
    block folder names are lowercase or use names other than A, B, C.
    """
    if df.empty or "ROI_number" not in df.columns:
        return df

    df = df.copy()

    qc_columns = []
    for roi_qc in qc_by_roi.values():
        for col in roi_qc.keys():
            if col not in qc_columns:
                qc_columns.append(col)

    train_prefix_order = [
        "Train_Bsl_median",
        "Train_Bsl_IRQ",
        "Train_Peak",
        "Train_fold",
        "Exclude_Stim_number",
    ]

    def _qc_column_sort_key(col):
        for prefix_index, prefix in enumerate(train_prefix_order):
            prefix_with_sep = f"{prefix}_"
            if col.startswith(prefix_with_sep):
                block_name = col[len(prefix_with_sep):]
                return block_name, prefix_index, col
        return col, len(train_prefix_order), col

    qc_columns = sorted(qc_columns, key=_qc_column_sort_key)

    for col in qc_columns:
        df[col] = df["ROI_number"].map(
            lambda roi: qc_by_roi.get(roi, {}).get(col, "")
        )

    return df

def _roi_should_be_excluded_from_clean(roi, qc_by_roi, rel_prob_df, recording_params):
    """
    Decide whether one ROI should be removed from results_clean.
    """
    roi_qc = qc_by_roi.get(roi, {})

    train_bc_threshold = _get_float_param(
        recording_params,
        "Exclude_ROI_when_trainBC<xx-fold",
        default=np.nan,
    )
    pr_a_threshold = _get_float_param(
        recording_params,
        "Exclude_ROI_when_PrA>xx",
        default=np.nan,
    )
    excluded_stim_threshold = _get_float_param(
        recording_params,
        "Exclude_ROI_when_#excludedStim>xx",
        default=np.nan,
    )

    train_fold_b = roi_qc.get("Train_fold_B", np.nan)
    train_fold_c = roi_qc.get("Train_fold_C", np.nan)

    if not pd.isna(train_bc_threshold):
        if not pd.isna(train_fold_b) and not pd.isna(train_fold_c):
            if train_fold_b < train_bc_threshold and train_fold_c < train_bc_threshold:
                return True

    if not pd.isna(pr_a_threshold):
        if "A" in rel_prob_df.columns and roi in rel_prob_df.index:
            pr_a = rel_prob_df.loc[roi, "A"]
            if not pd.isna(pr_a) and pr_a > pr_a_threshold:
                return True

    if not pd.isna(excluded_stim_threshold):
        total_excluded = 0
        found_any = False
        for block in ["A", "B", "C"]:
            value = roi_qc.get(f"Exclude_Stim_number_{block}", "")
            if value != "" and not pd.isna(value):
                total_excluded += int(value)
                found_any = True

        if found_any and total_excluded > excluded_stim_threshold:
            return True

    return False


def _filter_clean_result_df(df_with_qc, rel_prob_df, qc_by_roi, recording_params):
    """
    Remove excluded ROIs from one result dataframe.
    """
    if df_with_qc.empty or "ROI_number" not in df_with_qc.columns:
        return df_with_qc

    keep_rows = []
    for _, row in df_with_qc.iterrows():
        roi = row["ROI_number"]
        exclude = _roi_should_be_excluded_from_clean(
            roi=roi,
            qc_by_roi=qc_by_roi,
            rel_prob_df=rel_prob_df,
            recording_params=recording_params,
        )
        keep_rows.append(not exclude)

    return df_with_qc.loc[keep_rows].copy()


def _video_exists(tif_path: str) -> bool:
    if os.path.exists(tif_path):
        return True
    mp4_path = os.path.splitext(tif_path)[0] + ".mp4"
    return os.path.exists(mp4_path)


def _read_stack_prefer_tif_else_mp4(tif_path: str) -> np.ndarray:
    """
    Returns stack shaped (frames, H, W) as uint16.

    - If tif exists: read via skimage.io.imread
    - Else if mp4 exists: decode via ffmpeg to gray16le
    """
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

def run_analysis(
    block_path,
    recording_params,
    output_folder_ROIs,
    block_name,
    diff_image_path_and_name,
    dict_data_signal,
):
    """
    NOTE: This is your existing analysis, with the requested additions:
      - Add mito row (mito + zoom) under the diff row in each ROI PDF, both with ROI outline
      - Compute mean mito intensity per ROI and return it for Excel export
    """
    #print(f"    Running analysis for block: {block_name}")

    VIDEO_CSV_FILES = ["ap1+train.csv", "ap2.csv", "ap3.csv", "ap4.csv", "ap5.csv"]

    exclude_stim_baseline_percent = _get_float_param(
        recording_params,
        "Exclude_stim_when_baseline>xx%_of_ min_bsl_per_trace",
        default=np.nan,
    )

    output_dir = output_folder_ROIs
    os.makedirs(output_dir, exist_ok=True)

    rois_path = os.path.join(block_path, "ROIs.zip")
    rois_data = {}
    if os.path.exists(rois_path):
        try:
            rois_data = read_roi_zip(rois_path)
            #print(f"      Loaded {len(rois_data)} ROIs from {rois_path}")
        except Exception as e:
            print(f"      Failed to load ROIs: {e}")

    diff_img = None
    if rois_data and os.path.exists(diff_image_path_and_name):
        try:
            diff_img = io.imread(diff_image_path_and_name)
            if diff_img.ndim == 3:
                diff_img = diff_img[0]
        except Exception as e:
            print(f"      Failed to load diff image: {e}")

    mito_img = None
    mito_path = os.path.join(block_path, "mito.tif")
    if os.path.exists(mito_path):
        try:
            mito_img = load_image_2d(mito_path, mode="first")
        except Exception as e:
            print(f"      Failed to load mito image: {e}")

    # 1. Load Data (in-memory)
    if not isinstance(dict_data_signal, dict) or not dict_data_signal:
        print("      No CSV data provided in-memory. Aborting analysis.")
        return [], [], [], []

    list_ROIs = None
    cleaned_signal = {}
    for csv_name, df in dict_data_signal.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols_to_drop = [c for c in df.columns if c in ["Average", "Err", " "]]
        if cols_to_drop:
            df = df.drop(cols_to_drop, axis=1)
        cleaned_signal[csv_name] = df
        if list_ROIs is None:
            list_ROIs = list(df.columns)

    dict_data_signal = cleaned_signal
    if not dict_data_signal or not list_ROIs:
        print("      In-memory CSV dataframes are empty/invalid. Aborting analysis.")
        return [], [], [], []

    # 2. Parameters
    try:
        acq_time = float(recording_params["acquisition time (ms)"])
        butter_cutoff = float(recording_params["butter cutoff freq)"]) if "butter cutoff freq)" in recording_params else float(recording_params["butter cutoff freq"])

        baseline_dur = float(recording_params["baseline dur (ms)"])
        trace_start_offset = float(recording_params["trace start offset dur (ms)"])
        max_value_dur = float(recording_params["max value dur (ms)"])

        num_stim = int(recording_params["number of stim"])
        first_stim_time = float(recording_params["first stim timepoint (ms)"])
        inter_stim_dur = float(recording_params["inter stimulus dur (ms)"])

        train_plot_start = float(recording_params.get("trainPlotStart", 0))
        train_plot_end = float(recording_params.get("trainPlotEnd", 2000))
        zoom_size = float(recording_params.get("ZoomSize", 40))
    except KeyError as e:
        print(f"      Missing recording parameter: {e}")
        return [], [], [], []

    def get_time_idx(ms):
        return int(ms / acq_time)

    # 3. Filter Data
    fs = 1 / (acq_time / 1000)

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    dict_data_filtered = {}
    for fname, df in dict_data_signal.items():
        if butter_cutoff == 0:
            dict_data_filtered[fname] = df
        else:
            df_filt = df.copy()
            for col in df_filt.columns:
                df_filt[col] = butter_lowpass_filter(df_filt[col], butter_cutoff, fs)
            dict_data_filtered[fname] = df_filt

    # 4. Indices
    idx_first_stim = get_time_idx(first_stim_time)
    idx_offset = get_time_idx(trace_start_offset)
    idx_start_first_stim = idx_first_stim - idx_offset
    idx_dur_stim = get_time_idx(inter_stim_dur)

    stim_indices = []
    for i in range(num_stim):
        start = idx_start_first_stim + (i * idx_dur_stim)
        end = start + idx_dur_stim
        stim_indices.append((start, end))

    # For mito means we need ROI masks
    mito_means = []
    roi_masks = {}
    if mito_img is not None and rois_data:
        try:
            roi_masks = build_roi_masks(rois_data, mito_img.shape[:2])
        except Exception:
            roi_masks = {}

    # 5. Existing analysis outputs
    export_data_rel_prob = []
    export_data_w_mean_amp = []
    export_data_excluded_stim = []

    idx_baseline_end = get_time_idx(baseline_dur)
    idx_max_start = get_time_idx(trace_start_offset)
    idx_max_end = get_time_idx(trace_start_offset + max_value_dur)

    for roi in list_ROIs:
        # Compute mito mean (even if ROI PDF fails)
        if mito_img is not None and roi in roi_masks:
            mito_means.append([roi, mean_intensity_in_mask(mito_img, roi_masks[roi])])

        roi_events = []
        roi_traces_info = {}
        roi_excluded_stim_count = 0

        for fname in VIDEO_CSV_FILES:
            if fname not in dict_data_filtered:
                continue

            df_sig = dict_data_filtered[fname]
            if roi not in df_sig.columns:
                continue

            roi_sig = df_sig[roi].values

            baseline_sds = []
            baseline_maxs = []

            for (start, _) in stim_indices:
                baseline_seg = roi_sig[start: start + idx_baseline_end]
                baseline_sds.append(np.std(baseline_seg))
                baseline_maxs.append(np.max(baseline_seg))

            min_bsl_per_trace = np.min(baseline_maxs) if baseline_maxs else np.nan

            excluded_stim_for_trace = []
            for baseline_max in baseline_maxs:
                exclude_this_stim = False

                if not pd.isna(exclude_stim_baseline_percent) and not pd.isna(min_bsl_per_trace):
                    threshold_baseline = (
                                                 1 + 0.01 * exclude_stim_baseline_percent
                                         ) * min_bsl_per_trace
                    exclude_this_stim = baseline_max > threshold_baseline

                excluded_stim_for_trace.append(exclude_this_stim)

            roi_excluded_stim_count += int(np.sum(excluded_stim_for_trace))

            valid_baseline_sds = [
                value
                for value, excluded in zip(baseline_sds, excluded_stim_for_trace)
                if not excluded
            ]
            valid_baseline_maxs = [
                value
                for value, excluded in zip(baseline_maxs, excluded_stim_for_trace)
                if not excluded
            ]

            if not valid_baseline_sds or not valid_baseline_maxs:
                continue

            baseline_average_sd = np.median(valid_baseline_sds)
            baseline_average_max = np.median(valid_baseline_maxs)
            threshold = 3 * baseline_average_sd

            file_traces = []
            file_amps = []
            file_success = []
            file_baselines = []
            file_maxs = []
            file_excluded = []

            for stim_i, (start, end) in enumerate(stim_indices):
                trace_seg = roi_sig[start:end]
                file_traces.append(trace_seg)

                m_seg = roi_sig[start + idx_max_start: start + idx_max_end]
                max_val = np.max(m_seg) if len(m_seg) > 0 else 0

                amp = max_val - baseline_average_max
                is_success = amp >= threshold
                is_excluded = excluded_stim_for_trace[stim_i]

                file_amps.append(amp)
                file_success.append(is_success)
                file_baselines.append(baseline_average_max)
                file_maxs.append(max_val)
                file_excluded.append(is_excluded)

                if not is_excluded:
                    roi_events.append((amp, is_success))

            roi_traces_info[fname] = {
                "traces": file_traces,
                "amps": file_amps,
                "success": file_success,
                "baselines": file_baselines,
                "maxs": file_maxs,
                "excluded": file_excluded,
                "baseline_sd_val": baseline_average_sd,
                "threshold_val": threshold,
            }

        all_amps = [x[0] for x in roi_events]
        if not all_amps:
            export_data_excluded_stim.append([roi, roi_excluded_stim_count])
            continue

        counts, bin_edges = np.histogram(all_amps, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        counts_success = np.zeros_like(counts)
        counts_failures = np.zeros_like(counts)

        inds = np.digitize(all_amps, bin_edges)
        for (amp, is_succ), bin_idx in zip(roi_events, inds):
            idx = bin_idx - 1
            idx = max(0, min(idx, len(counts) - 1))
            if is_succ:
                counts_success[idx] += 1
            else:
                counts_failures[idx] += 1

        sum_success = np.sum(counts_success)
        sum_failures = np.sum(counts_failures)
        rel_prob = sum_success / (sum_success + sum_failures) if (sum_success + sum_failures) > 0 else 0

        weighted_sum_amp = np.sum(bin_centers * counts_success)
        w_mean_amp = weighted_sum_amp / sum_success if sum_success > 0 else np.nan

        export_data_rel_prob.append([roi, rel_prob])
        export_data_w_mean_amp.append([roi, w_mean_amp])
        export_data_excluded_stim.append([roi, roi_excluded_stim_count])

        # ---- Plotting (add mito row) ----
        current_roi_data = rois_data.get(roi)
        if current_roi_data is None:
            s = str(roi)
            if s.startswith("Mean(") and s.endswith(")"):
                base = s[len("Mean("):-1]
            else:
                base = s
            current_roi_data = rois_data.get(base)

        show_diff = (current_roi_data is not None) and (diff_img is not None)
        show_mito = (current_roi_data is not None) and (mito_img is not None)

        train_trace_key = "ap1+train.csv"
        show_train_trace = train_trace_key in dict_data_signal and roi in dict_data_signal[train_trace_key].columns

        extra_rows = 0
        if show_diff:
            extra_rows += 1
        if show_mito:
            extra_rows += 1

        train_row = 1 if show_train_trace else 0
        num_files = len(roi_traces_info)
        total_rows = num_files + 2 + extra_rows + train_row

        fig = plt.figure(figsize=(10, 20 + (5 * extra_rows) + (2 if show_train_trace else 0)))
        gs = fig.add_gridspec(total_rows, 2)

        row_offset = 0

        def _plot_roi(ax, roi_geom):
            roi_type = roi_geom.get("type")
            if roi_type in ("polygon", "freehand"):
                x = roi_geom["x"]
                y = roi_geom["y"]
                ax.plot(x + [x[0]], y + [y[0]], linewidth=2, color="yellow")
            elif roi_type == "rectangle":
                x = roi_geom["left"]
                y = roi_geom["top"]
                w = roi_geom["width"]
                h = roi_geom["height"]
                ax.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], linewidth=2, color="yellow")
            elif roi_type == "oval":
                rx = roi_geom["width"] / 2
                ry = roi_geom["height"] / 2
                cx = roi_geom["left"] + rx
                cy = roi_geom["top"] + ry
                theta = np.linspace(0, 2 * np.pi, 200)
                ax.plot(cx + rx * np.cos(theta), cy + ry * np.sin(theta), linewidth=2, color="yellow")

        def _roi_center(roi_geom):
            if roi_geom.get("type") in ("polygon", "freehand"):
                return float(np.mean(roi_geom["x"])), float(np.mean(roi_geom["y"]))
            return (
                float(roi_geom["left"] + roi_geom["width"] / 2),
                float(roi_geom["top"] + roi_geom["height"] / 2),
            )

        def _add_img_row(img, title_left, title_right):
            nonlocal row_offset
            vmin, vmax = np.percentile(img, [1, 99])
            axL = fig.add_subplot(gs[row_offset, 0])
            axL.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            _plot_roi(axL, current_roi_data)
            axL.set_title(title_left)
            axL.axis("off")

            cx, cy = _roi_center(current_roi_data)
            half = zoom_size / 2
            h_img, w_img = img.shape
            x_start = max(0, cx - half)
            x_end = min(w_img, cx + half)
            y_start = max(0, cy - half)
            y_end = min(h_img, cy + half)

            zoom_slice = img[int(y_start):int(y_end), int(x_start):int(x_end)]
            if zoom_slice.size > 0:
                vmin_z, vmax_z = np.percentile(zoom_slice, [1, 99])
            else:
                vmin_z, vmax_z = vmin, vmax

            axR = fig.add_subplot(gs[row_offset, 1])
            axR.imshow(img, cmap="gray", vmin=vmin_z, vmax=vmax_z)
            axR.set_xlim(x_start, x_end)
            axR.set_ylim(y_end, y_start)
            _plot_roi(axR, current_roi_data)
            axR.set_title(title_right)
            axR.axis("off")

            row_offset += 1

        if show_diff:
            _add_img_row(diff_img, "Diff Image with ROI", f"Diff Zoom (Size: {zoom_size})")
        if show_mito:
            _add_img_row(mito_img, "Mito Image with ROI", f"Mito Zoom (Size: {zoom_size})")

        if show_train_trace:
            ax_train = fig.add_subplot(gs[row_offset, :])
            full_trace = dict_data_signal[train_trace_key][roi].values
            t_axis = np.arange(0, len(full_trace) * acq_time, acq_time)[: len(full_trace)]
            ax_train.plot(t_axis, full_trace, color="black")
            ax_train.set_xlim(train_plot_start, train_plot_end)
            ax_train.set_title("Train Trace (not filtered)")
            ax_train.set_xlabel("time (ms)")
            ax_train.set_ylabel("Signal")
            row_offset += 1

        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1

        ax_hist1 = fig.add_subplot(gs[row_offset, :])
        ax_hist1.bar(bin_centers, counts, width=bin_width, color="lightgrey")
        ax_hist1.set_xlabel("Signal Intensity")
        ax_hist1.set_ylabel("Frequency")

        ax_hist2 = fig.add_subplot(gs[row_offset + 1, :])
        ax_hist2.bar(bin_centers, counts_failures, width=bin_width, color="blue", label="failures")
        ax_hist2.bar(
            bin_centers,
            counts_success,
            width=bin_width,
            color="orange",
            bottom=counts_failures,
            label="success",
        )
        ax_hist2.legend(loc="upper right")
        ax_hist2.text(
            0.02,
            0.95,
            f"release prob= {rel_prob:.2f}\nweighted mean amplitude= {w_mean_amp:.2f}",
            transform=ax_hist2.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_hist2.set_xlabel("Signal Intensity")
        ax_hist2.set_ylabel("Frequency")

        plot_idx = 2
        for fname in VIDEO_CSV_FILES:
            if fname not in roi_traces_info:
                continue

            info = roi_traces_info[fname]
            ax = fig.add_subplot(gs[row_offset + plot_idx, :])
            plot_idx += 1

            for i, trace in enumerate(info["traces"]):
                start_idx = stim_indices[i][0]
                time_arr = np.arange(
                    start_idx * acq_time,
                    (start_idx * acq_time) + (len(trace) * acq_time),
                    acq_time,
                )[: len(trace)]

                N = len(info["traces"])
                cmap = plt.colormaps.get_cmap("Dark2")
                colormap = cmap(np.linspace(0, 1, max(N, 1)))
                color = colormap[i % len(colormap)]
                ax.plot(time_arr, trace, color=color)

                t_start = time_arr[0]
                t_base_end = t_start + baseline_dur
                t_max_start = t_start + trace_start_offset
                t_max_end = t_max_start + max_value_dur

                ax.hlines(info["baselines"][i], t_start, t_base_end, color="black")
                ax.hlines(info["maxs"][i], t_max_start, t_max_end, color="black")

                if info.get("excluded", [False] * len(info["traces"]))[i]:
                    ax.axvspan(
                        t_start,
                        time_arr[-1],
                        color="red",
                        alpha=0.12,
                        label="excluded stim" if i == 0 else None,
                    )

                average_sd = info["baseline_sd_val"]
                thresh = info["threshold_val"]
                base = info["baselines"][i]
                ax.hlines(average_sd + base, t_start, t_base_end, color="black", linestyles="dashed")
                ax.hlines(thresh + base, t_start, t_base_end, color="black", linestyles="dashed")

            ax.set_xlabel("time (ms)")
            ax.set_ylabel("Signal Intensity")
            ax.set_title(f"{fname}")

        plt.suptitle(f"Block: {block_name}; ROI: {roi}", y=0.98)
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])

        save_path = os.path.join(output_dir, f"{roi}_{block_name}.pdf")
        plt.savefig(save_path)
        plt.close(fig)
        #print(f"      Saved figure for ROI {roi}")
        print(f"      {roi}")

    #print("    Analysis completed.")
    return export_data_rel_prob, export_data_w_mean_amp, mito_means, export_data_excluded_stim

def process_block(block_path, output_folder_experiment, recording_params):
    """
    Processes a single block folder.
    Special blocks: name length > 1 -> use special pipeline only.
    """
    block_name = os.path.basename(block_path)
    is_special = len(str(block_name)) > 1

    train_qc = {}
    excluded_stim = []

    output_folder_ROIs = os.path.join(output_folder_experiment, "ROIs")
    os.makedirs(output_folder_ROIs, exist_ok=True)

    output_folder_DiffImage = os.path.join(output_folder_experiment, "DiffImage")
    os.makedirs(output_folder_DiffImage, exist_ok=True)

    output_folder_CSVs = os.path.join(output_folder_experiment, "CSVs", block_name)
    os.makedirs(output_folder_CSVs, exist_ok=True)

    external_folder_CSVs = os.path.join(block_path, "CSVs")
    os.makedirs(external_folder_CSVs, exist_ok=True)

    output_folder_averages = os.path.join(output_folder_experiment, "averages")
    os.makedirs(output_folder_averages, exist_ok=True)

    recalc_csv = _metadata_flag_is_true(recording_params, "Recalc_CSV", default=1)

    print(f"  Processing block: {block_name}")

    # --- handle special blocks before checking ap1+train.tif ---
    if is_special:
        ind_path = os.path.join(block_path, "ind.tif")
        if not _video_exists(ind_path):
            print(f"    Warning: ind.tif/mp4 not found in {block_path}. Skipping special block.")
            return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}

        out = process_special_block(
            block_path=block_path,
            block_name=block_name,
            recording_params=recording_params,
            output_folder_experiment=output_folder_experiment,
            recalc_csv=recalc_csv,
        )
        # Averages PDF + Excel for special blocks
        ind_csv_path = os.path.join(output_folder_CSVs, "ind.csv")
        if os.path.exists(ind_csv_path):
            try:
                df_ind = pd.read_csv(ind_csv_path)
                if not df_ind.empty:
                    save_special_block_averages(
                        output_folder_averages=output_folder_averages,
                        block_name=block_name,
                        recording_params=recording_params,
                        df_ind=df_ind,
                    )
            except Exception as e:
                print(f"    Warning: could not generate averages for {block_name}: {e}")

        return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}

    ap1_path = os.path.join(block_path, "ap1+train.tif")
    diff_image_path_and_name = os.path.join(output_folder_DiffImage, f"{block_name}_diff.tif")

    if not _video_exists(ap1_path):
        print(f"    Warning: ap1+train.tif/mp4 not found in {block_path}. Skipping block.")
        return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}

    # Standard blocks: compute diff with standard params
    calculate_diff_image(ap1_path, diff_image_path_and_name, recording_params, param_suffix="")

    generated_csvs = []
    dict_csv_dfs = {}

    expected_existing_csvs = [
        vid_file.replace(".tif", ".csv")
        for vid_file in VIDEO_FILES
        if _video_exists(os.path.join(block_path, vid_file))
    ]

    if not recalc_csv:
        dict_csv_dfs = _load_existing_csvs(external_folder_CSVs, expected_existing_csvs)

        if dict_csv_dfs:
            print(f"    Using cached CSV files from {external_folder_CSVs}")

            for csv_name, df_cached in dict_csv_dfs.items():
                _save_csv_cache_copy(df_cached, csv_name, output_folder_CSVs)

            generated_csvs = list(dict_csv_dfs.keys())
        else:
            print(f"    No cached CSV files found in {external_folder_CSVs}. Recalculating from videos.")

    if recalc_csv or not dict_csv_dfs:
        rois_zip_path = os.path.join(block_path, "ROIs.zip")
        if not os.path.exists(rois_zip_path):
            print(f"    Warning: ROIs.zip not found at {rois_zip_path}. Skipping intensity extraction.")
            return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}

        try:
            rois_data = read_roi_zip(rois_zip_path)
            if not rois_data:
                print("    Warning: No ROIs found in ROIs.zip. Skipping intensity extraction.")
                return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}
        except Exception as e:
            print(f"    Error reading ROIs.zip: {e}")
            return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}

        image_shape_hw = None
        for vid_file in VIDEO_FILES:
            vid_path_probe = os.path.join(block_path, vid_file)
            if _video_exists(vid_path_probe):
                probe_stack = _read_stack_prefer_tif_else_mp4(vid_path_probe)
                if probe_stack.ndim == 3:
                    image_shape_hw = (probe_stack.shape[1], probe_stack.shape[2])
                elif probe_stack.ndim == 2:
                    image_shape_hw = probe_stack.shape
                break

        if image_shape_hw is None:
            print("    Warning: Could not determine image shape (no videos found). Skipping intensity extraction.")
            return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}

        roi_masks = build_roi_masks(rois_data, image_shape_hw)
        if not roi_masks:
            print("    Warning: No valid ROI masks produced. Skipping intensity extraction.")
            return {"rel": [], "wma": [], "mito": [], "train_qc": train_qc, "excluded_stim": excluded_stim}

        for vid_file in VIDEO_FILES:
            vid_path = os.path.join(block_path, vid_file)
            if not _video_exists(vid_path):
                continue

            stack = _read_stack_prefer_tif_else_mp4(vid_path)
            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]
            if stack.ndim != 3:
                continue

            data_cols = {}
            for roi_name, mask in roi_masks.items():
                if mask.shape != stack.shape[1:]:
                    hh = min(stack.shape[1], mask.shape[0])
                    ww = min(stack.shape[2], mask.shape[1])
                    sub_mask = mask[:hh, :ww]
                    sub_stack = stack[:, :hh, :ww]
                else:
                    sub_mask = mask
                    sub_stack = stack

                if not sub_mask.any():
                    continue
                data_cols[roi_name] = sub_stack[:, sub_mask].mean(axis=1)

            if not data_cols:
                continue

            df_out = pd.DataFrame(data_cols)
            csv_name = vid_file.replace(".tif", ".csv")

            _save_csv_cache_copy(
                df_out,
                csv_name,
                output_folder_CSVs,
                external_folder_CSVs,
            )

            generated_csvs.append(csv_name)
            dict_csv_dfs[csv_name] = df_out

    # Standard analysis + mito means
    rel_data, wma_data, mito_data = [], [], []
    if generated_csvs:
        train_qc = _calculate_train_qc_for_block(
            dict_csv_dfs=dict_csv_dfs,
            block_name=block_name,
            recording_params=recording_params,
        )
        rel_data, wma_data, mito_data, excluded_stim = run_analysis(
            block_path=block_path,
            recording_params=recording_params,
            output_folder_ROIs=output_folder_ROIs,
            block_name=block_name,
            diff_image_path_and_name=diff_image_path_and_name,
            dict_data_signal=dict_csv_dfs,
        )

        # Block averages PDF (only for one-letter blocks)
        mito_path = os.path.join(block_path, "mito.tif")
        if os.path.exists(mito_path):
            save_block_averages_pdf(
                output_folder_averages=output_folder_averages,
                block_name=block_name,
                recording_params=recording_params,
                diff_image_path=diff_image_path_and_name,
                mito_path=mito_path,
                dict_data_signal=dict_csv_dfs,
            )

    return {
        "rel": rel_data,
        "wma": wma_data,
        "mito": mito_data,
        "train_qc": train_qc,
        "excluded_stim": excluded_stim,
    }

def iGluSnFr_eval():
    run_start_time = time.perf_counter()

    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    output_folder = os.path.join(config.ROOT_FOLDER, f"output_{config.MY_INITIAL}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    output_folder_used_data_and_code = os.path.join(output_folder, "used_data_and_code")
    os.makedirs(output_folder_used_data_and_code, exist_ok=True)

    try:
        metadata_df = pd.read_excel(config.METADATA_FILE)
        metadata_df.to_excel(os.path.join(output_folder_used_data_and_code, "my_data.xlsx"), index=False)
    except Exception as e:
        print(f"Error loading metadata file {config.METADATA_FILE}: {e}")
        return

    script_path = __file__ if "__file__" in globals() else None
    myGit.save_git_info(output_folder_used_data_and_code, script_path)

    for experiment_count, row in metadata_df.iterrows():
        experimentName = row["experimentName"]
        print(f"Processing experiment {experiment_count + 1}: {experimentName}")

        recording_params = row.to_dict()

        exp_folder_path = os.path.join(config.EXTERNAL_DATA_FOLDER, str(experimentName))

        output_folder_experiment = os.path.join(output_folder, experimentName)
        os.makedirs(output_folder_experiment, exist_ok=True)

        output_folder_results = os.path.join(output_folder_experiment, "results")
        os.makedirs(output_folder_results, exist_ok=True)

        output_folder_results_clean = os.path.join(output_folder_experiment, "results_clean")
        os.makedirs(output_folder_results_clean, exist_ok=True)

        if not os.path.exists(exp_folder_path):
            print(f"  Experiment folder not found: {exp_folder_path}")
            continue

        exp_rel_prob_df = pd.DataFrame()
        exp_wma_df = pd.DataFrame()
        exp_mito_df = pd.DataFrame()
        exp_qc_by_roi = {}

        for block_name in sorted(os.listdir(exp_folder_path)):
            block_path = os.path.join(exp_folder_path, block_name)
            if not os.path.isdir(block_path):
                continue

            out = process_block(block_path, output_folder_experiment, recording_params)

            rel_data = out.get("rel", [])
            wma_data = out.get("wma", [])
            mito_data = out.get("mito", [])
            train_qc = out.get("train_qc", {})
            excluded_stim = out.get("excluded_stim", [])

            for roi, qc_values in train_qc.items():
                if roi not in exp_qc_by_roi:
                    exp_qc_by_roi[roi] = {}
                exp_qc_by_roi[roi].update(qc_values)

            for roi, excluded_count in excluded_stim:
                if roi not in exp_qc_by_roi:
                    exp_qc_by_roi[roi] = {}
                exp_qc_by_roi[roi][f"Exclude_Stim_number_{block_name}"] = excluded_count

            if rel_data:
                current_rel_df = pd.DataFrame(rel_data, columns=["ROI_number", block_name]).set_index("ROI_number")
                exp_rel_prob_df = current_rel_df if exp_rel_prob_df.empty else exp_rel_prob_df.join(current_rel_df, how="outer")

            if wma_data:
                current_wma_df = pd.DataFrame(wma_data, columns=["ROI_number", block_name]).set_index("ROI_number")
                exp_wma_df = current_wma_df if exp_wma_df.empty else exp_wma_df.join(current_wma_df, how="outer")

            if mito_data:
                current_mito_df = pd.DataFrame(mito_data, columns=["ROI_number", block_name]).set_index("ROI_number")
                exp_mito_df = current_mito_df if exp_mito_df.empty else exp_mito_df.join(current_mito_df, how="outer")

        rel_with_qc = pd.DataFrame()
        wma_with_qc = pd.DataFrame()
        mito_with_qc = pd.DataFrame()

        if not exp_rel_prob_df.empty:
            rel_plain = exp_rel_prob_df.sort_index().reset_index()
            rel_with_qc = _append_qc_columns(rel_plain, exp_qc_by_roi)

            rel_with_qc.to_excel(
                os.path.join(output_folder_results, "release_probability.xlsx"),
                index=False,
            )

        if not exp_wma_df.empty:
            wma_plain = exp_wma_df.sort_index().reset_index()
            wma_with_qc = _append_qc_columns(wma_plain, exp_qc_by_roi)

            wma_with_qc.to_excel(
                os.path.join(output_folder_results, "weighted_amplitude.xlsx"),
                index=False,
            )

        if not exp_mito_df.empty:
            mito_plain = exp_mito_df.sort_index().reset_index()
            mito_with_qc = _append_qc_columns(mito_plain, exp_qc_by_roi)

            mito_with_qc.to_excel(
                os.path.join(output_folder_results, "mito_intensity.xlsx"),
                index=False,
            )

        # Clean outputs: same columns, but ROIs failing exclusion criteria are removed.
        if not exp_rel_prob_df.empty:
            rel_prob_for_filter = exp_rel_prob_df.copy()

            rel_clean = _filter_clean_result_df(
                rel_with_qc,
                rel_prob_for_filter,
                exp_qc_by_roi,
                recording_params,
            )
            rel_clean.to_excel(
                os.path.join(output_folder_results_clean, "release_probability.xlsx"),
                index=False,
            )

            if not wma_with_qc.empty:
                wma_clean = _filter_clean_result_df(
                    wma_with_qc,
                    rel_prob_for_filter,
                    exp_qc_by_roi,
                    recording_params,
                )
                wma_clean.to_excel(
                    os.path.join(output_folder_results_clean, "weighted_amplitude.xlsx"),
                    index=False,
                )

            if not mito_with_qc.empty:
                mito_clean = _filter_clean_result_df(
                    mito_with_qc,
                    rel_prob_for_filter,
                    exp_qc_by_roi,
                    recording_params,
                )
                mito_clean.to_excel(
                    os.path.join(output_folder_results_clean, "mito_intensity.xlsx"),
                    index=False,
                )

    elapsed_seconds = time.perf_counter() - run_start_time
    elapsed_minutes = elapsed_seconds / 60

    print("\nAnalysis completed.")
    print(f"Total runtime: {elapsed_seconds:.1f} seconds ({elapsed_minutes:.2f} minutes)")

if __name__ == "__main__":
    iGluSnFr_eval()