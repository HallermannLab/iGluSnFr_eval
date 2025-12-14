import numpy as np


def mean_intensity_in_mask(img2d, mask2d):
    if img2d.shape != mask2d.shape:
        h = min(img2d.shape[0], mask2d.shape[0])
        w = min(img2d.shape[1], mask2d.shape[1])
        img2d = img2d[:h, :w]
        mask2d = mask2d[:h, :w]

    pix = img2d[mask2d]
    if pix.size == 0:
        return float("nan")
    return float(np.mean(pix))