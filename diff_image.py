import numpy as np
from skimage import io


def _get_ms_param(recording_params, key, default=None):
    val = recording_params.get(key, default)
    if val is None:
        raise KeyError(key)
    return float(val)


def calculate_diff_image(tif_path, output_path, recording_params, *, param_suffix=""):
    """
    Computes (Stim - Baseline) and clips negatives to 0.

    param_suffix:
      - "" for standard keys: Diff_BL_Start, Diff_BL_End, Diff_Stim_Start, Diff_Stim_End
      - "_Induction" for special keys: Diff_BL_Start_Induction, ... etc
    """
    img = io.imread(tif_path)
    if img.ndim == 2:
        img = img[None, ...]
    if img.ndim != 3:
        raise ValueError(f"Unsupported image dims {img.shape} for {tif_path}")

    acq_time = float(recording_params["acquisition time (ms)"])

    k_bl_s = f"Diff_BL_Start{param_suffix}"
    k_bl_e = f"Diff_BL_End{param_suffix}"
    k_st_s = f"Diff_Stim_Start{param_suffix}"
    k_st_e = f"Diff_Stim_End{param_suffix}"

    start_bl = int(_get_ms_param(recording_params, k_bl_s) / acq_time)
    end_bl = int(_get_ms_param(recording_params, k_bl_e) / acq_time)
    start_st = int(_get_ms_param(recording_params, k_st_s) / acq_time)
    end_st = int(_get_ms_param(recording_params, k_st_e) / acq_time)

    max_idx = max(start_bl, end_bl, start_st, end_st)
    if img.shape[0] <= max_idx:
        raise ValueError(
            f"Diff indices exceed stack length for {tif_path}. "
            f"stack_frames={img.shape[0]}, max_idx={max_idx}"
        )

    bl = img[start_bl:end_bl].mean(axis=0).astype(np.float32)
    st = img[start_st:end_st].mean(axis=0).astype(np.float32)

    diff = np.clip(st - bl, 0, 65535).astype(np.uint16)
    io.imsave(output_path, diff, check_contrast=False)
    return True