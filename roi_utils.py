import numpy as np
from skimage.draw import polygon as draw_polygon, ellipse as draw_ellipse


def load_image_2d(path, *, mode="first"):
    """
    Loads an image that may be 2D or 3D stack.
    Returns a 2D array.

    mode:
      - "first": use first frame if stack
      - "mean": average across frames if stack
    """
    from skimage import io

    img = io.imread(path)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if mode == "mean":
            return img.mean(axis=0)
        return img[0]
    raise ValueError(f"Unsupported image shape {img.shape} for {path}")


def build_roi_masks(roi_dict, image_shape_hw):
    """
    roi_dict is from read_roi_zip().
    image_shape_hw is (h, w).
    Returns: dict roi_name -> boolean mask (h, w)
    """
    h, w = image_shape_hw
    masks = {}

    for roi_name, roi in roi_dict.items():
        rtype = roi.get("type")
        mask = np.zeros((h, w), dtype=bool)

        try:
            if rtype in ("polygon", "freehand"):
                xs = np.asarray(roi["x"], dtype=np.float32)
                ys = np.asarray(roi["y"], dtype=np.float32)
                rr, cc = draw_polygon(ys, xs, shape=(h, w))
                mask[rr, cc] = True

            elif rtype == "rectangle":
                x = int(round(roi["left"]))
                y = int(round(roi["top"]))
                ww = int(round(roi["width"]))
                hh = int(round(roi["height"]))
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + ww)
                y2 = min(h, y + hh)
                if x1 < x2 and y1 < y2:
                    mask[y1:y2, x1:x2] = True

            elif rtype == "oval":
                rx = max(int(round(roi["width"] / 2)), 1)
                ry = max(int(round(roi["height"] / 2)), 1)
                cx = int(round(roi["left"] + rx))
                cy = int(round(roi["top"] + ry))
                rr, cc = draw_ellipse(cy, cx, ry, rx, shape=(h, w))
                mask[rr, cc] = True

            else:
                # Fallback: if it still has x/y arrays, treat like polygon
                if "x" in roi and "y" in roi:
                    xs = np.asarray(roi["x"], dtype=np.float32)
                    ys = np.asarray(roi["y"], dtype=np.float32)
                    rr, cc = draw_polygon(ys, xs, shape=(h, w))
                    mask[rr, cc] = True

        except Exception:
            # Skip problematic ROI silently (caller can warn if needed)
            continue

        if mask.any():
            masks[roi_name] = mask

    return masks


def roi_center(roi):
    rtype = roi.get("type")
    if rtype in ("polygon", "freehand"):
        cx = float(np.mean(roi["x"]))
        cy = float(np.mean(roi["y"]))
        return cx, cy
    return (
        float(roi["left"] + roi["width"] / 2),
        float(roi["top"] + roi["height"] / 2),
    )


def roi_outline_xy(roi):
    """
    Returns (x_list, y_list) that traces the outline, closed if needed.
    """
    rtype = roi.get("type")
    if rtype in ("polygon", "freehand"):
        x = list(roi["x"])
        y = list(roi["y"])
        if x and y:
            x = x + [x[0]]
            y = y + [y[0]]
        return x, y

    if rtype == "rectangle":
        x = float(roi["left"])
        y = float(roi["top"])
        w = float(roi["width"])
        h = float(roi["height"])
        return [x, x + w, x + w, x, x], [y, y, y + h, y + h, y]

    if rtype == "oval":
        rx = float(roi["width"]) / 2.0
        ry = float(roi["height"]) / 2.0
        cx = float(roi["left"]) + rx
        cy = float(roi["top"]) + ry
        theta = np.linspace(0, 2 * np.pi, 200)
        return (cx + rx * np.cos(theta)).tolist(), (cy + ry * np.sin(theta)).tolist()

    # Fallback: try polygon if x/y exist
    if "x" in roi and "y" in roi:
        x = list(roi["x"])
        y = list(roi["y"])
        if x and y:
            x = x + [x[0]]
            y = y + [y[0]]
        return x, y

    return [], []