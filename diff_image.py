import os
import numpy as np
from skimage import io

import shutil
import subprocess


def _get_ms_param(recording_params, key, default=None):
    val = recording_params.get(key, default)
    if val is None:
        raise KeyError(key)
    return float(val)


def _read_stack_prefer_tif_else_mp4(path: str) -> np.ndarray:
    """
    Loads a (frames, H, W) stack from:
      - TIFF if present
      - otherwise MP4 with same basename (decoded via ffmpeg to gray16le)
    """
    if os.path.exists(path):
        img = io.imread(path)
        if img.ndim == 2:
            img = img[None, ...]
        return img

    mp4_path = os.path.splitext(path)[0] + ".mp4"
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"No input video found: {path} (also checked {mp4_path})")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH (required to decode MP4 for diff image).")

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



def calculate_diff_image(tif_path, output_path, recording_params, *, param_suffix=""):
    """
    Computes (Stim - Baseline) and clips negatives to 0.

    param_suffix:
      - "" for standard keys: Diff_BL_Start, Diff_BL_End, Diff_Stim_Start, Diff_Stim_End
      - "_Induction" for special keys: Diff_BL_Start_Induction, ... etc
    """
    img = _read_stack_prefer_tif_else_mp4(tif_path)
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