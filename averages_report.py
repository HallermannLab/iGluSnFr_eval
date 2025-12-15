import os
import numpy as np
import matplotlib.pyplot as plt

from roi_utils import load_image_2d


VIDEO_CSV_FILES = ["ap1+train.csv", "ap2.csv", "ap3.csv", "ap4.csv", "ap5.csv"]


def _avg_trace(df):
    # df columns are ROIs
    return df.mean(axis=1).to_numpy(dtype=float)


def save_block_averages_pdf(
    *,
    output_folder_averages,
    block_name,
    recording_params,
    diff_image_path,
    mito_path,
    dict_data_signal,
):
    os.makedirs(output_folder_averages, exist_ok=True)

    diff_img = load_image_2d(diff_image_path, mode="first")
    mito_img = load_image_2d(mito_path, mode="first")

    acq_time = float(recording_params["acquisition time (ms)"])

    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(1 + 1 + len(VIDEO_CSV_FILES), 2)

    # Row 0: images (no zoom, no ROI overlay)
    ax0 = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(diff_img, [1, 99])
    ax0.imshow(diff_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax0.set_title("Diff image")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    vmin, vmax = np.percentile(mito_img, [1, 99])
    ax1.imshow(mito_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("Mito image")
    ax1.axis("off")

    # Row 1: average full train trace (no analysis)
    train_plot_start = float(recording_params.get("trainPlotStart", 0))
    train_plot_end = float(recording_params.get("trainPlotEnd", 2000))
    train_key = "ap1+train.csv"
    ax_train = fig.add_subplot(gs[1, :])
    if train_key in dict_data_signal and not dict_data_signal[train_key].empty:
        y = _avg_trace(dict_data_signal[train_key])
        t = np.arange(len(y)) * acq_time
        ax_train.plot(t, y, color="black", linewidth=1)
    ax_train.set_xlim(train_plot_start, train_plot_end)
    ax_train.set_xlabel("time (ms)")
    ax_train.set_ylabel("mean ROI intensity")
    ax_train.set_title("Average full train trace across ROIs (no analysis)")

    # Rows 2..: average per-file traces (plotted as full duration, no analysis overlays)
    row = 2
    for fname in VIDEO_CSV_FILES:
        ax = fig.add_subplot(gs[row, :])
        if fname in dict_data_signal and not dict_data_signal[fname].empty:
            y = _avg_trace(dict_data_signal[fname])
            t = np.arange(len(y)) * acq_time
            ax.plot(t, y, color="black", linewidth=1)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("mean ROI intensity")
        ax.set_title(f"Average trace across ROIs: {fname}")
        row += 1

    fig.suptitle(f"Block averages: {block_name}", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(output_folder_averages, f"{block_name}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path