import numpy as np
import matplotlib.pyplot as plt

from roi_utils import roi_center, roi_outline_xy


def _imshow_with_percentiles(ax, img, title):
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return vmin, vmax


def _plot_roi_outline(ax, roi, color="yellow", lw=2):
    x, y = roi_outline_xy(roi)
    if x and y:
        ax.plot(x, y, linewidth=lw, color=color)


def _zoom_limits(cx, cy, zoom_size, shape_hw):
    h, w = shape_hw
    half = zoom_size / 2.0
    x_start = max(0, cx - half)
    x_end = min(w, cx + half)
    y_start = max(0, cy - half)
    y_end = min(h, cy + half)
    return x_start, x_end, y_start, y_end


def plot_roi_pdf(
    *,
    save_path,
    block_name,
    roi_name,
    roi_geom,
    diff_img,
    mito_img,
    zoom_size,
    full_train_time_ms,
    full_train_trace,
    extra_panels=None,
):
    """
    Creates a 2x2 image header:
      row 1: diff + zoom (both with ROI outline)
      row 2: mito + zoom (both with ROI outline)
    Then one trace panel: full ROI intensity over full video duration.

    extra_panels: optional list of (title, time_ms, y_values) to append as additional rows
    """
    cx, cy = roi_center(roi_geom)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3 + (len(extra_panels) if extra_panels else 0), 2)

    # Row 0: diff + zoom
    ax00 = fig.add_subplot(gs[0, 0])
    _imshow_with_percentiles(ax00, diff_img, "Diff image (ROI overlay)")
    _plot_roi_outline(ax00, roi_geom)

    x1, x2, y1, y2 = _zoom_limits(cx, cy, zoom_size, diff_img.shape[:2])
    ax01 = fig.add_subplot(gs[0, 1])
    zoom = diff_img[int(y1):int(y2), int(x1):int(x2)]
    if zoom.size:
        vmin, vmax = np.percentile(zoom, [1, 99])
        ax01.imshow(diff_img, cmap="gray", vmin=vmin, vmax=vmax)
    else:
        ax01.imshow(diff_img, cmap="gray")
    ax01.set_xlim(x1, x2)
    ax01.set_ylim(y2, y1)
    ax01.set_title(f"Diff zoom (size={zoom_size})")
    ax01.axis("off")
    _plot_roi_outline(ax01, roi_geom)

    # Row 1: mito + zoom
    ax10 = fig.add_subplot(gs[1, 0])
    _imshow_with_percentiles(ax10, mito_img, "Mito image (ROI overlay)")
    _plot_roi_outline(ax10, roi_geom)

    ax11 = fig.add_subplot(gs[1, 1])
    zoom_m = mito_img[int(y1):int(y2), int(x1):int(x2)]
    if zoom_m.size:
        vmin, vmax = np.percentile(zoom_m, [1, 99])
        ax11.imshow(mito_img, cmap="gray", vmin=vmin, vmax=vmax)
    else:
        ax11.imshow(mito_img, cmap="gray")
    ax11.set_xlim(x1, x2)
    ax11.set_ylim(y2, y1)
    ax11.set_title(f"Mito zoom (size={zoom_size})")
    ax11.axis("off")
    _plot_roi_outline(ax11, roi_geom)

    # Row 2: full trace
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(full_train_time_ms, full_train_trace, color="black", linewidth=1)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("ROI intensity")
    ax2.set_title("Full ROI intensity trace (no analysis)")

    # Extra panels if needed
    row = 3
    if extra_panels:
        for title, t_ms, y in extra_panels:
            ax = fig.add_subplot(gs[row, :])
            ax.plot(t_ms, y, color="black", linewidth=1)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("ROI intensity")
            ax.set_title(title)
            row += 1

    fig.suptitle(f"Block: {block_name} | ROI: {roi_name}", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path)
    plt.close(fig)