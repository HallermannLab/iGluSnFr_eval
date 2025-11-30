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

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct
from matplotlib.patches import Rectangle, Polygon
from read_roi import read_roi_file

from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

import git_save as myGit
from skimage import io
import zipfile
from read_roi import read_roi_zip


VIDEO_FILES = ["ap1+train.tif", "ap2.tif", "ap3.tif", "ap4.tif", "ap5.tif"]


def calculate_diff_image(tif_path, output_path, recording_params):
    """
    Calculates difference image (Stim - Baseline) similar to the notebook logic.
    """
    try:
        img = io.imread(tif_path)

        # Convert ms to frame numbers using acquisition time
        acq_time = float(recording_params["acquisition time (ms)"])
        Start_BL = int(float(recording_params["Diff_BL_Start"]) / acq_time)
        End_BL = int(float(recording_params["Diff_BL_End"]) / acq_time)
        Start_Stim = int(float(recording_params["Diff_Stim_Start"]) / acq_time)
        End_Stim = int(float(recording_params["Diff_Stim_End"]) / acq_time)

        # Safety check for frame indices
        if img.shape[0] <= max(Start_BL, End_BL, Start_Stim, End_Stim):
            print(f"Warning: Image at {tif_path} has fewer frames than expected indices. Using defaults.")

        # Calculate averages
        BL = img[Start_BL:End_BL].mean(axis=0).astype(np.float32)
        Stim = img[Start_Stim:End_Stim].mean(axis=0).astype(np.float32)

        # Subtract (Stim - BL), clip negative values
        img_responding = np.clip(Stim - BL, 0, 65535).astype(np.uint16)

        # Save
        io.imsave(output_path, img_responding, check_contrast=False)
        print(f"Saved Diff image: {output_path}")
        return True
    except Exception as e:
        print(f"Error calculating diff image for {tif_path}: {e}")
        return False


def run_analysis(block_path, recording_params, output_folder_ROIs, block_name,diff_image_path_and_name):
    print(f"    Running analysis for block: {block_name}")

    # Constants / Config
    VIDEO_CSV_FILES = ["ap1+train.csv", "ap2.csv", "ap3.csv", "ap4.csv", "ap5.csv"]

    # Ensure output directory exists
    output_dir = output_folder_ROIs
    os.makedirs(output_dir, exist_ok=True)

    # 0. Check for ROIs.zip and Diff Image
    rois_path = os.path.join(block_path, "ROIs.zip")
    rois_data = {}
    if os.path.exists(rois_path):
        try:
            # Use read_roi library directly
            rois_data = read_roi_zip(rois_path)
            print(f"      Loaded {len(rois_data)} ROIs from {rois_path}")
        except Exception as e:
            print(f"      Failed to load ROIs: {e}")

    diff_img = None
    if rois_data:
        if os.path.exists(diff_image_path_and_name):
            try:
                diff_img = io.imread(diff_image_path_and_name)
            except Exception as e:
                print(f"      Failed to load diff image: {e}")

    # 1. Load Data
    dict_data_signal = {}
    list_ROIs = None

    # Check files availability
    files_found = []
    for csv_file in VIDEO_CSV_FILES:
        full_path = os.path.join(block_path, csv_file)
        if os.path.exists(full_path):
            try:
                # Assuming CSVs have ROIs as columns.
                df = pd.read_csv(full_path)
                # If "Average" or "Err" columns exist, drop them (backward compatibility)
                cols_to_drop = [c for c in df.columns if c in ["Average", "Err", " "]]
                if cols_to_drop:
                    df = df.drop(cols_to_drop, axis=1)

                dict_data_signal[csv_file] = df
                files_found.append(csv_file)

                if list_ROIs is None:
                    list_ROIs = list(df.columns)
            except Exception as e:
                print(f"      Error reading {csv_file}: {e}")
        else:
            print(f"      File not found: {full_path}")

    if not files_found:
        print("      No CSV files found. Aborting analysis.")
        return [], []

    # 2. Pre-calculations & Constants
    try:
        acq_time = float(recording_params["acquisition time (ms)"])
        butter_cutoff = float(recording_params["butter cutoff freq"])

        baseline_dur = float(recording_params["baseline dur (ms)"])
        trace_start_offset = float(recording_params["trace start offset dur (ms)"])
        max_value_dur = float(recording_params["max value dur (ms)"])

        num_stim = int(recording_params["number of stim"])
        first_stim_time = float(recording_params["first stim timepoint (ms)"])
        inter_stim_dur = float(recording_params["inter stimulus dur (ms)"])
        # trace_dur = float(recording_params["trace dur (ms)"]) # Not strictly needed if calculated from inter_stim
        train_plot_start = float(recording_params.get("trainPlotStart", 0))
        train_plot_end = float(recording_params.get("trainPlotEnd", 2000))

    except KeyError as e:
        print(f"      Missing recording parameter: {e}")
        return [], []

    def get_time_idx(ms):
        return int(ms / acq_time)

    # 3. Filter Data (Butterworth)
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

    # 4. Get Start/End Indices for Traces
    idx_first_stim = get_time_idx(first_stim_time)
    idx_offset = get_time_idx(trace_start_offset)
    idx_start_first_stim = idx_first_stim - idx_offset
    idx_dur_stim = get_time_idx(inter_stim_dur)

    stim_indices = []
    for i in range(num_stim):
        start = idx_start_first_stim + (i * idx_dur_stim)
        end = start + idx_dur_stim
        stim_indices.append((start, end))

    # 5. Calculate Amplitudes & Classify Success (SD)
    # We need per ROI:
    #  - Release Prob
    #  - Weighted Mean Amp
    #  - Histogram data (bin centers, counts, success counts, fail counts)

    # Structures to hold results for exporting
    export_data_rel_prob = []
    export_data_w_mean_amp = []

    idx_baseline_end = get_time_idx(baseline_dur)
    idx_max_start = get_time_idx(trace_start_offset)
    idx_max_end = get_time_idx(trace_start_offset + max_value_dur)

    for roi in list_ROIs:
        roi_events = []  # List of (amp, is_success)
        roi_traces_info = {}  # fname -> {traces: [], amps: [], success: [], baselines: [], maxs: []}
        # 1. Gather Data & Classify
        for fname in VIDEO_CSV_FILES:
            if fname not in dict_data_filtered: continue

            df_sig = dict_data_filtered[fname]
            roi_sig = df_sig[roi].values

            # Baseline SD
            baseline_sds = []
            for (start, _) in stim_indices:
                baseline_seg = roi_sig[start: start + idx_baseline_end]
                baseline_sds.append(np.std(baseline_seg))

            if not baseline_sds:
                continue

            mean_sd = np.mean(baseline_sds)
            threshold = 3 * mean_sd

            file_traces = []
            file_amps = []
            file_success = []
            file_baselines = []  # Need for plotting (hlines)
            file_maxs = []  # Need for plotting (hlines)

            for (start, end) in stim_indices:
                trace_seg = roi_sig[start:end]
                file_traces.append(trace_seg)

                b_seg = roi_sig[(start):(start + idx_baseline_end)]
                baseline_val = np.max(b_seg) if len(b_seg) > 0 else 0

                m_seg = roi_sig[(start + idx_max_start):(start + idx_max_end)]
                max_val = np.max(m_seg) if len(m_seg) > 0 else 0

                amp = max_val - baseline_val
                is_success = (amp >= threshold)

                file_amps.append(amp)
                file_success.append(is_success)
                file_baselines.append(baseline_val)
                file_maxs.append(max_val)

                roi_events.append((amp, is_success))

            roi_traces_info[fname] = {
                "traces": file_traces,
                "amps": file_amps,
                "success": file_success,
                "baselines": file_baselines,
                "maxs": file_maxs,
                "baseline_sd_val": mean_sd,  # needed for plotting line
                "threshold_val": threshold  # needed for plotting line
            }

        # 2. Binning & Stats
        all_amps = [x[0] for x in roi_events]
        if not all_amps: continue

        counts, bin_edges = np.histogram(all_amps, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        counts_success = np.zeros_like(counts)
        counts_failures = np.zeros_like(counts)

        # Assign events to bins
        inds = np.digitize(all_amps, bin_edges)

        for (amp, is_succ), bin_idx in zip(roi_events, inds):
            # bin_idx is 1-based index of bin_edges.
            idx = bin_idx - 1
            if idx >= len(counts): idx = len(counts) - 1
            if idx < 0: idx = 0

            if is_succ:
                counts_success[idx] += 1
            else:
                counts_failures[idx] += 1

        sum_success = np.sum(counts_success)
        sum_failures = np.sum(counts_failures)
        rel_prob = sum_success / (sum_success + sum_failures) if (sum_success + sum_failures) > 0 else 0

        weighted_sum_amp = np.sum(bin_centers * counts_success)
        w_mean_amp = weighted_sum_amp / sum_success if sum_success > 0 else np.nan

        # Store for Excel
        export_data_rel_prob.append([roi, rel_prob])
        export_data_w_mean_amp.append([roi, w_mean_amp])

        # 3. Plotting
        current_roi_data = rois_data.get(roi)

        # Determine if we plot the specific Train Trace
        train_trace_key = "ap1+train.csv"
        show_train_trace = train_trace_key in roi_traces_info

        if current_roi_data is None:
            s = str(roi)

            # If the column is named like "Mean(ROI001)" → "ROI001"
            if s.startswith("Mean(") and s.endswith(")"):
                base = s[len("Mean("):-1]  # remove "Mean(" and trailing ")"
            else:
                base = s

            current_roi_data = rois_data.get(base)

            show_roi_images = (current_roi_data is not None) and (diff_img is not None)

        # Rows: Optional Image Row + Optional Train Trace Row + 2 Histogram Rows + N Trace Rows
        extra_rows = 1 if show_roi_images else 0
        train_row = 1 if show_train_trace else 0
        num_files = len(roi_traces_info)
        total_rows = num_files + 2 + extra_rows + train_row

        fig = plt.figure(figsize=(10, 20 + (5 if show_roi_images else 0) + (2 if show_train_trace else 0)))
        gs = fig.add_gridspec(total_rows, 2)

        row_offset = 0

        # Plot ROI Images (Row 0)
        if show_roi_images:
            # Calculate robust contrast limits (1st and 99th percentiles)
            vmin, vmax = np.percentile(diff_img, [1, 99])

            # Diff Image with ROI Highlight
            ax_diff = fig.add_subplot(gs[0, 0])
            ax_diff.imshow(diff_img, cmap='gray', vmin=vmin, vmax=vmax)

            # Draw ROI outline
            roi_type = current_roi_data['type']

            # Polygon ROI
            if roi_type == 'polygon':
                x = current_roi_data['x']
                y = current_roi_data['y']
                ax_diff.plot(x + [x[0]], y + [y[0]], linewidth=2, color='yellow')

            # Rectangle ROI
            elif roi_type == 'rectangle':
                x = current_roi_data["left"]
                y = current_roi_data["top"]
                w = current_roi_data["width"]
                h = current_roi_data["height"]
                rect_x = [x, x + w, x + w, x, x]
                rect_y = [y, y, y + h, y + h, y]
                ax_diff.plot(rect_x, rect_y, linewidth=2, color='yellow')

            # Oval ROI (approximate with a polygon)
            elif roi_type == 'oval':
                rx = current_roi_data["width"] / 2
                ry = current_roi_data["height"] / 2
                cx = current_roi_data["left"] + rx
                cy = current_roi_data["top"] + ry
                theta = np.linspace(0, 2*np.pi, 200)
                x = cx + rx * np.cos(theta)
                y = cy + ry * np.sin(theta)
                ax_diff.plot(x, y, linewidth=2, color='yellow')

            ax_diff.set_title("Diff Image with ROI")
            ax_diff.axis('off')

            # Zoom Image ------------
            # Determine ROI type and center for zooming
            roi_type = current_roi_data['type']

            if roi_type == 'polygon':
                cx = np.mean(current_roi_data['x'])
                cy = np.mean(current_roi_data['y'])
            else:
                # Rectangle or Oval
                cx = current_roi_data['left'] + current_roi_data['width'] / 2
                cy = current_roi_data['top'] + current_roi_data['height'] / 2

            # Get ZoomSize
            try:
                zoom_size = float(recording_params.get('ZoomSize', 40))
            except:
                zoom_size = 40

            # Calculate zoom limits
            half_size = zoom_size / 2
            h_img, w_img = diff_img.shape
            x_start = max(0, cx - half_size)
            x_end = min(w_img, cx + half_size)
            y_start = max(0, cy - half_size)
            y_end = min(h_img, cy + half_size)

            # Calculate separate contrast for zoom
            zoom_slice = diff_img[int(y_start):int(y_end), int(x_start):int(x_end)]
            if zoom_slice.size > 0:
                vmin_zoom, vmax_zoom = np.percentile(zoom_slice, [1, 99])
            else:
                vmin_zoom, vmax_zoom = vmin, vmax

            # Zoom Image (Show full image but limit view to create zoom effect)
            ax_zoom = fig.add_subplot(gs[0, 1])
            ax_zoom.imshow(diff_img, cmap='gray', vmin=vmin_zoom, vmax=vmax_zoom)

            # Apply Zoom (invert Y for images so smaller Y is at top)
            ax_zoom.set_xlim(x_start, x_end)
            ax_zoom.set_ylim(y_end, y_start)

            ax_zoom.set_title(f"Zoom (Size: {zoom_size})")
            ax_zoom.axis('off')

            # Draw ROI outline on BOTH axes
            for ax in [ax_diff, ax_zoom]:
                if roi_type == 'polygon':
                    x = current_roi_data['x']
                    y = current_roi_data['y']
                    ax.plot(x + [x[0]], y + [y[0]], linewidth=2, color='yellow')

                elif roi_type == 'rectangle':
                    x = current_roi_data["left"]
                    y = current_roi_data["top"]
                    w = current_roi_data["width"]
                    h = current_roi_data["height"]
                    rect_x = [x, x + w, x + w, x, x]
                    rect_y = [y, y, y + h, y + h, y]
                    ax.plot(rect_x, rect_y, linewidth=2, color='yellow')

                elif roi_type == 'oval':
                    rx = current_roi_data["width"] / 2
                    ry = current_roi_data["height"] / 2
                    cx_ov = current_roi_data["left"] + rx
                    cy_ov = current_roi_data["top"] + ry
                    theta = np.linspace(0, 2 * np.pi, 200)
                    x_ov = cx_ov + rx * np.cos(theta)
                    y_ov = cy_ov + ry * np.sin(theta)
                    ax.plot(x_ov, y_ov, linewidth=2, color='yellow')

            row_offset = 1

        # Plot Train Trace (Row 1 or 0)
        if show_train_trace:
            ax_train = fig.add_subplot(gs[row_offset, :])

            # Get data
            # We access the filtered data dictionary directly to get the full trace for this ROI
            if train_trace_key in dict_data_filtered:
                #full_trace = dict_data_filtered[train_trace_key][roi].values
                full_trace = dict_data_signal[train_trace_key][roi].values

                # Time axis
                t_axis = np.arange(0, len(full_trace) * acq_time, acq_time)
                # Truncate if size mismatch slightly
                t_axis = t_axis[:len(full_trace)]

                ax_train.plot(t_axis, full_trace, color='black')

                # Set limits
                ax_train.set_xlim(train_plot_start, train_plot_end)
                ax_train.set_title("Train Trace (Zoomed)")
                ax_train.set_xlabel("time (ms)")
                ax_train.set_ylabel("Signal")

            row_offset += 1

        # Plot Histogram (Row 0/1 & 1/2)
        bin_width = bin_centers[1] - bin_centers[0]

        # Frequency
        ax_hist1 = fig.add_subplot(gs[row_offset, :])
        ax_hist1.bar(bin_centers, counts, width=bin_width, color="lightgrey")
        ax_hist1.set_xlabel("Signal Intensity")
        ax_hist1.set_ylabel("Frequency")

        # Breakdown
        ax_hist2 = fig.add_subplot(gs[row_offset + 1, :])
        ax_hist2.bar(bin_centers, counts_failures, width=bin_width, color="blue", label='failures')
        ax_hist2.bar(bin_centers, counts_success, width=bin_width, color="orange", bottom=counts_failures,
                     label='success')
        ax_hist2.legend(loc="upper right")
        ax_hist2.text(0.02, 0.95,
                      f"release prob= {rel_prob:.2f}\nweighted mean amplitude= {w_mean_amp:.2f}",
                      transform=ax_hist2.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        ax_hist2.set_xlabel("Signal Intensity")
        ax_hist2.set_ylabel("Frequency")

        # Traces (Rows 2+ to N)
        plot_idx = 2
        for fname in VIDEO_CSV_FILES:
            if fname not in roi_traces_info: continue

            info = roi_traces_info[fname]
            ax = fig.add_subplot(gs[row_offset + plot_idx, :])
            plot_idx += 1

            # Plot traces
            for i, trace in enumerate(info["traces"]):
                # Time
                start_idx = stim_indices[i][0]
                time_arr = np.arange(start_idx * acq_time, (start_idx * acq_time) + (len(trace) * acq_time), acq_time)
                time_arr = time_arr[:len(trace)]

                # Color
                N = len(info["traces"])
                cmap = plt.colormaps.get_cmap('Dark2')
                colormap = cmap(np.linspace(0, 1, N))
                color = colormap[i]

                ax.plot(time_arr, trace, color=color)

                # Lines
                t_start = time_arr[0]
                t_base_end = t_start + baseline_dur
                t_max_start = t_start + trace_start_offset
                t_max_end = t_max_start + max_value_dur

                ax.hlines(info["baselines"][i], t_start, t_base_end, color="black")
                ax.hlines(info["maxs"][i], t_max_start, t_max_end, color="black")

                # SD Lines (Dashed)
                mean_sd = info["baseline_sd_val"]
                thresh = info["threshold_val"]
                base = info["baselines"][i]

                ax.hlines(mean_sd + base, t_start, t_base_end, color="black", linestyles='dashed')
                ax.hlines(thresh + base, t_start, t_base_end, color="black", linestyles='dashed')

            ax.set_xlabel("time (ms)")
            ax.set_ylabel("Signal Intensity")
            ax.set_title(f"{fname}")

        # Title
        plt.suptitle(f"Block: {block_name}; ROI: {roi}", y=0.98)
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])

        # Save
        save_path = os.path.join(output_dir, f"{roi}_{block_name}.pdf")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"      Saved figure for ROI {roi}")

    # 4. Export Excel
    #df_rel = pd.DataFrame(export_data_rel_prob, columns=['ROI', 'release_probability'])
    #df_rel.to_excel(os.path.join(output_dir, f"{block_name}_release_probability.xlsx"), index=False)

    #df_wma = pd.DataFrame(export_data_w_mean_amp, columns=['ROI', 'weighted_mean_amplitude'])
    #df_wma.to_excel(os.path.join(output_dir, f"{block_name}_weighted_mean_amplitude.xlsx"), index=False)

    print("    Analysis completed.")

    return export_data_rel_prob, export_data_w_mean_amp


def process_block(block_path, output_folder_experiment, recording_params):
    """
    Processes a single block folder ("A", "B", etc).
    """
    block_name = os.path.basename(block_path)

    output_folder_ROIs = os.path.join(output_folder_experiment, "ROIs")
    os.makedirs(output_folder_ROIs, exist_ok=True)

    output_folder_DiffImage = os.path.join(output_folder_experiment, "DiffImage")
    os.makedirs(output_folder_DiffImage, exist_ok=True)

    print(f"  Processing block: {block_name}")

    # 1. Diff Image from ap1+train.tif
    ap1_path = os.path.join(block_path, "ap1+train.tif")
    diff_image_path_and_name = os.path.join(output_folder_DiffImage, f"{block_name}_diff.tif")

    if not os.path.exists(ap1_path):
        print(f"    Warning: {ap1_path} not found. Skipping block.")
        return [], []

    if not calculate_diff_image(ap1_path, diff_image_path_and_name, recording_params):
        return [], []

    """
    # 2. Detect ROIs using ImageJ
    roi_zip_path = os.path.join(output_block_folder, "RoiSet.zip")
    if not get_rois_fiji(diff_image_path_and_name, roi_zip_path, ij):
        return [], []

    # 3. Extract Intensities (Measure) from all 5 videos
    generated_csvs = []

    for vid_file in VIDEO_FILES:
        vid_path = os.path.join(block_path, vid_file)
        csv_name = vid_file.replace(".tif", ".csv")
        csv_output_path = os.path.join(output_block_folder, csv_name)

        if not os.path.exists(vid_path):
            print(f"    Video {vid_path} missing. Skipping.")
            continue

        if ij:
            # Macro: open video, open ROIs, measure (Multi Measure), save results
            macro_measure = f###
            open("{vid_path.replace(os.sep, '/')}");
            roiManager("Open", "{roi_zip_path.replace(os.sep, '/')}");
            run("Set Measurements...", "mean redirect=None decimal=3");
            roiManager("Multi Measure");
            saveAs("Results", "{csv_output_path.replace(os.sep, '/')}");
            run("Close"); // Close Results window
            close(); // Close Image
            roiManager("Delete"); // Clear ROIs for next run
            ###
            try:
                ij.py.run_macro(macro_measure)
                if os.path.exists(csv_output_path):
                    generated_csvs.append(csv_name)
            except Exception as e:
                print(f"    Error extracting intensities for {vid_file}: {e}")
    """

    # use manual csv
    generated_csvs = []
    for vid_file in VIDEO_FILES:
        #vid_path = os.path.join(block_path, vid_file)
        csv_name = vid_file.replace(".tif", ".csv")
        #csv_output_path = os.path.join(output_block_folder, csv_name)
        generated_csvs.append(csv_name)

    # 4. Run Analysis
    if generated_csvs:
        export_data_rel_prob, export_data_w_mean_amp = run_analysis(block_path, recording_params, output_folder_ROIs, block_name, diff_image_path_and_name)

    return export_data_rel_prob, export_data_w_mean_amp


def iGluSnFr_eval():

    # --- Create Output Folders ---
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    output_folder = os.path.join(config.ROOT_FOLDER, f"output_{config.MY_INITIAL}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    output_folder_used_data_and_code = os.path.join(output_folder, "used_data_and_code")
    os.makedirs(output_folder_used_data_and_code, exist_ok=True)

    # --- Load Metadata ---
    # Use read_excel as per user description, though previous code used read_csv in Image class.
    # User specified .xlsx file in description.
    try:
        metadata_df = pd.read_excel(config.METADATA_FILE)
        # Also save copy of metadata
        metadata_df.to_excel(os.path.join(output_folder_used_data_and_code, "my_data.xlsx"), index=False)
    except Exception as e:
        print(f"Error loading metadata file {config.METADATA_FILE}: {e}")
        return

    # === GIT SAVE ===
    script_path = __file__ if '__file__' in globals() else None
    myGit.save_git_info(output_folder_used_data_and_code, script_path)

    # --- Process Each Experiment ---
    for experiment_count, row in metadata_df.iterrows():
        experimentName = row['experimentName']
        print(f"Processing experiment {experiment_count + 1}: {experimentName}")

        # Extract recording parameters for this experiment
        recording_params = row.to_dict()

        # Locate Experiment Folder
        exp_folder_path = os.path.join(config.EXTERNAL_DATA_FOLDER, str(experimentName))

        output_folder_experiment = os.path.join(output_folder, experimentName)
        os.makedirs(output_folder_experiment, exist_ok=True)

        output_folder_results = os.path.join(output_folder_experiment, "results")
        os.makedirs(output_folder_results, exist_ok=True)

        if not os.path.exists(exp_folder_path):
            print(f"  Experiment folder not found: {exp_folder_path}")
            continue

        # Accumulate results for this experiment
        exp_rel_prob_df = pd.DataFrame()
        exp_wma_df = pd.DataFrame()

        # Iterate over subfolders (Blocks: "A", "B", etc.)
        # We assume all directories in the experiment folder are blocks
        for block_name in sorted(os.listdir(exp_folder_path)):
            block_path = os.path.join(exp_folder_path, block_name)
            if os.path.isdir(block_path):
                # You might want to filter for specific block names if needed,
                # e.g., if len(block_name) == 1:
                rel_data, wma_data = process_block(block_path, output_folder_experiment, recording_params)

                if rel_data:
                    # Convert list to DataFrame, set column name to Block Name (e.g., "A")
                    current_rel_df = pd.DataFrame(rel_data, columns=['ROI_number', block_name])
                    current_rel_df.set_index('ROI_number', inplace=True)

                    if exp_rel_prob_df.empty:
                        exp_rel_prob_df = current_rel_df
                    else:
                        # Join on ROI_number (index) to align ROIs across blocks
                        exp_rel_prob_df = exp_rel_prob_df.join(current_rel_df, how='outer')

                if wma_data:
                    current_wma_df = pd.DataFrame(wma_data, columns=['ROI_number', block_name])
                    current_wma_df.set_index('ROI_number', inplace=True)

                    if exp_wma_df.empty:
                        exp_wma_df = current_wma_df
                    else:
                        exp_wma_df = exp_wma_df.join(current_wma_df, how='outer')

            # Save Experiment Results
            out_path = os.path.join(output_folder_results, "release_probability.xlsx")
            # Sort by ROI name, reset index to make ROI_number a column, and don't save the numerical index
            exp_rel_prob_df.sort_index().reset_index().to_excel(out_path, index=False)

            out_path = os.path.join(output_folder_results, "wheighted_amplitude.xlsx")
            exp_wma_df.sort_index().reset_index().to_excel(out_path, index=False)

if __name__ == '__main__':
    iGluSnFr_eval()