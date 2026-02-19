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
import shutil
import subprocess
from pathlib import Path

import numpy as np
from skimage import io


VIDEO_FILES = ["ap1+train.tif", "ap2.tif", "ap3.tif", "ap4.tif", "ap5.tif"]
EXTRA_VIDEO_FILES = ["ind.tif"]  # special_blocks.py uses this as a video stack


def _is_video_file(path: Path, *, video_basenames: set[str]) -> bool:
    """
    Decide whether a file should be treated as a "video" that must NOT be copied
    and instead must be compressed.
    """
    return path.is_file() and path.name in video_basenames


def _copy_non_video_files(src_root: str, dst_root: str, *, video_basenames: set[str]) -> None:
    """
    Copies all NON-video files/folders from src_root into dst_root, preserving structure.
    Video files (by basename) are skipped so they can be compressed instead.
    """
    src_root_p = Path(src_root)
    dst_root_p = Path(dst_root)
    dst_root_p.mkdir(parents=True, exist_ok=True)

    for dirpath, _, filenames in os.walk(src_root):
        rel_dir = Path(dirpath).relative_to(src_root_p)
        out_dir = dst_root_p / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in filenames:
            src_file = Path(dirpath) / fname
            if _is_video_file(src_file, video_basenames=video_basenames):
                continue
            dst_file = out_dir / fname
            shutil.copy2(src_file, dst_file)


def _iter_video_files(root: str, video_basenames: set[str]):
    """
    Yields Paths of video files under root whose basename matches video_basenames.
    """
    root_p = Path(root)
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname in video_basenames:
                yield root_p / Path(dirpath).relative_to(root_p) / fname


def _compress_lossless_to_zarr(tif_path: Path, zarr_path: Path, *, chunk_hw: int = 256) -> None:
    """
    Stores TIFF stack as Zarr using Zstd + Bitshuffle (via numcodecs.Blosc).

    Note: numcodecs compressors like Blosc are the Zarr v2 codec pipeline.
    With zarr>=3, explicitly writing zarr_format=2 prevents metadata/layout mismatches.
    """
    try:
        import zarr  # type: ignore
        from numcodecs import Blosc  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing packages for lossless Zarr compression. Install with: pip install zarr numcodecs"
        ) from e

    arr = io.imread(str(tif_path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    y = arr.shape[1]
    x = arr.shape[2]
    chunks = (1, min(chunk_hw, y), min(chunk_hw, x))

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    # Write atomically: create in a temp folder, then replace the target.
    tmp_path = zarr_path.with_name(zarr_path.name + ".tmp")

    if tmp_path.exists():
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path)
        else:
            tmp_path.unlink()

    if zarr_path.exists():
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            zarr_path.unlink()

    z = zarr.open(
        str(tmp_path),
        mode="w",
        shape=arr.shape,
        chunks=chunks,
        dtype=arr.dtype,
        compressor=compressor,
        zarr_format=2,
    )
    z[:] = arr

    if not (tmp_path / ".zarray").exists():
        raise RuntimeError(f"Zarr write failed: missing metadata {(tmp_path / '.zarray')}")

    tmp_path.replace(zarr_path)


def _compress_for_viewing_h264(tif_path: Path, mp4_path: Path, *, fps: float = 25.0) -> None:
    """
    Encodes a TIFF stack (scientific camera) to H.264 MP4 for viewing.

    Note: We decode the TIFF with skimage (more robust for multi-page scientific TIFFs than ffmpeg),
    then stream frames to ffmpeg so H.264 can use temporal compression.
    """
    import shutil
    import subprocess

    import numpy as np
    from skimage import io

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install ffmpeg and ensure the 'ffmpeg' command works."
        )

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if mp4_path.exists():
        mp4_path.unlink()

    stack = io.imread(str(tif_path))
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"Unsupported image dims {stack.shape} for {tif_path}")

    n_frames, h, w = stack.shape
    if n_frames <= 1:
        raise RuntimeError(
            f"{tif_path} decoded to {n_frames} frame(s). "
            "This cannot produce a real movie. The TIFF may be single-frame or unreadable as a stack."
        )

    # Viewing conversion: 12-bit-in-16-bit -> 8-bit (fast, compatible).
    # If your data is truly 16-bit, this will compress dynamic range; adapt if needed.
    if stack.dtype != np.uint16:
        stack = stack.astype(np.uint16, copy=False)
    stack8 = (stack >> 4).astype(np.uint8, copy=False)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s:v",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        "format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "10",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]

    # IMPORTANT: Don't manually close stdin before communicate() on Python 3.13+.
    # Provide input via communicate(input=...) so subprocess manages stdin safely.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=stack8.tobytes(order="C"))

    if proc.returncode != 0:
        err_text = (err or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {tif_path}:\n{err_text}")


def _compress_strong_12bit_hevc(
    tif_path: Path,
    mp4_path: Path,
    *,
    fps: float = 30.0,
    crf: int = 12,
    preset: str = "slow",
) -> None:
    """
    Strong lossy 12-bit compression using HEVC/H.265 Main12.

    Intended for analysis where you want to stay in a 12-bit domain (values 0..4095)
    and keep ROI mean intensity vs time stable.

    Notes:
      - Uses libx265 with Main12 profile and gray12le output.
      - Disables psy + AQ to avoid perceptual re-allocation of error (more "numeric-friendly").
      - If your TIFF values are 12-bit stored in uint16 (common), sending gray16le + format=gray12le
        preserves values as long as they're already in 0..4095.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install ffmpeg and ensure the 'ffmpeg' command works."
        )

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if mp4_path.exists():
        mp4_path.unlink()

    stack = io.imread(str(tif_path))
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"Unsupported image dims {stack.shape} for {tif_path}")

    n_frames, h, w = stack.shape
    if n_frames <= 1:
        raise RuntimeError(
            f"{tif_path} decoded to {n_frames} frame(s). "
            "This cannot produce a real movie. The TIFF may be single-frame or unreadable as a stack."
        )

    if stack.dtype != np.uint16:
        stack = stack.astype(np.uint16, copy=False)

    x265_params = f"profile=main12:crf={crf}:aq-mode=0:psy-rd=0:psy-rdoq=0"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-s:v",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        "format=gray12le",
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-x265-params",
        x265_params,
        "-pix_fmt",
        "gray12le",
        "-color_range",
        "2",
        "-tag:v",
        "hvc1",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=stack.tobytes(order="C"))

    if proc.returncode != 0:
        err_text = (err or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {tif_path}:\n{err_text}")


def _compress_strong_16bit_hevc_main12(
    tif_path: Path,
    mp4_path: Path,
    *,
    fps: float = 30.0,
    crf: int = 10,
    preset: str = "slow",
) -> None:
    """
    "MP4-like" strong compression for 16-bit sources using HEVC Main12 (CRF-based).

    Important limitation:
      - True 16-bit video encoding in MP4 with HEVC is not supported (x265 is up to 12-bit).
      - This function therefore converts uint16 -> 12-bit (0..4095) with rounding,
        then encodes as HEVC Main12 (gray12le).

    This is the closest analogue to the previous 12-bit pipeline, but starting from real 16-bit inputs.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install ffmpeg and ensure the 'ffmpeg' command works."
        )

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if mp4_path.exists():
        mp4_path.unlink()

    stack = io.imread(str(tif_path))
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"Unsupported image dims {stack.shape} for {tif_path}")

    n_frames, h, w = stack.shape
    if n_frames <= 1:
        raise RuntimeError(
            f"{tif_path} decoded to {n_frames} frame(s). "
            "This cannot produce a real movie. The TIFF may be single-frame or unreadable as a stack."
        )

    if stack.dtype != np.uint16:
        stack = stack.astype(np.uint16, copy=False)

    # Convert to 12-bit with rounding: (val + 8) >> 4
    # This preserves the top 12 bits (best effort) and makes the stream truly gray12le.
    stack12 = ((stack.astype(np.uint32) + 8) >> 4).astype(np.uint16, copy=False)

    x265_params = f"profile=main12:crf={crf}:aq-mode=0:psy-rd=0:psy-rdoq=0"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",  # we feed 16le containers holding 0..4095 values
        "-s:v",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        "format=gray12le",
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-x265-params",
        x265_params,
        "-pix_fmt",
        "gray12le",
        "-color_range",
        "2",
        "-tag:v",
        "hvc1",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=stack12.tobytes(order="C"))

    if proc.returncode != 0:
        err_text = (err or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {tif_path}:\n{err_text}")


def _compress_strong_16bit_ffv1(
    tif_path: Path,
    mkv_path: Path,
    *,
    fps: float = 30.0,
    level: int = 3,
) -> None:
    """
    Strong 16-bit compression for true uint16 video stacks using FFV1 (lossless) in Matroska.

    Why FFV1 here:
      - libx265 does not practically support true 16-bit grayscale (x265 is typically 8/10/12-bit).
      - FFV1 supports gray16le well and stays numerically exact (lossless), which is usually what you want
        when your source actually uses the full 16-bit domain.

    Output:
      - Writes .mkv (Matroska), which is the common/robust container for FFV1.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install ffmpeg and ensure the 'ffmpeg' command works."
        )

    mkv_path.parent.mkdir(parents=True, exist_ok=True)
    if mkv_path.exists():
        mkv_path.unlink()

    stack = io.imread(str(tif_path))
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"Unsupported image dims {stack.shape} for {tif_path}")

    n_frames, h, w = stack.shape
    if n_frames <= 1:
        raise RuntimeError(
            f"{tif_path} decoded to {n_frames} frame(s). "
            "This cannot produce a real movie. The TIFF may be single-frame or unreadable as a stack."
        )

    if stack.dtype != np.uint16:
        stack = stack.astype(np.uint16, copy=False)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-s:v",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "ffv1",
        "-level",
        str(level),
        "-g",
        "1",
        "-pix_fmt",
        "gray16le",
        str(mkv_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=stack.tobytes(order="C"))

    if proc.returncode != 0:
        err_text = (err or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {tif_path}:\n{err_text}")


def compress_external_data_only() -> None:
    """
    Creates two compressed mirrors of EXTERNAL_DATA_FOLDER:
      - *_compress_loss_less : Zarr(Zstd+Bitshuffle) for video files
      - *_compress_for_viewing : H.264 CRF 10 MP4 for video files
    All other files are copied unchanged into both mirrors.

    Important: Video files are NOT copied as .tif into the mirrors at all.
    They are read from SRC and written directly as compressed outputs into each mirror.
    """
    src = config.EXTERNAL_DATA_FOLDER
    if not os.path.isdir(src):
        raise RuntimeError(f"EXTERNAL_DATA_FOLDER not found or not a directory: {src}")

    dst_lossless = f"{src}_compress_loss_less"
    dst_view = f"{src}_compress_for_viewing"
    dst_strong16 = f"{src}_compress_strong_16bit"
    dst_strong12 = f"{src}_compress_strong_12bit"

    video_set = set(VIDEO_FILES) | set(EXTRA_VIDEO_FILES)

    print(
        "Copying non-video data:\n"
        f"  SRC : {src}\n"
        f"  DST1: {dst_lossless}\n"
        f"  DST2: {dst_view}\n"
        f"  DST3: {dst_strong16}\n"
        f"  (unused for now) DST4: {dst_strong12}"
    )
    _copy_non_video_files(src, dst_lossless, video_basenames=video_set)
    _copy_non_video_files(src, dst_view, video_basenames=video_set)
    _copy_non_video_files(src, dst_strong16, video_basenames=video_set)
    _copy_non_video_files(src, dst_strong12, video_basenames=video_set)

    src_p = Path(src)
    dst_lossless_p = Path(dst_lossless)
    dst_view_p = Path(dst_view)
    dst_strong16_p = Path(dst_strong16)
    dst_strong12_p = Path(dst_strong12)

    """
    print("\nCompressing loss_less (Zarr + Zstd + Bitshuffle) from SRC into mirror...")
    for tif in _iter_video_files(src, video_set):
        rel = tif.relative_to(src_p)
        out_zarr = (dst_lossless_p / rel).with_suffix(".zarr")
        out_zarr.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} -> {out_zarr.relative_to(dst_lossless_p)}")
        _compress_lossless_to_zarr(tif, out_zarr)

    print("\nCompressing for_viewing (H.264 CRF 10) from SRC into mirror...")
    for tif in _iter_video_files(src, video_set):
        rel = tif.relative_to(src_p)
        out_mp4 = (dst_view_p / rel).with_suffix(".mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} -> {out_mp4.relative_to(dst_view_p)}")
        _compress_for_viewing_h264(tif, out_mp4)
    """


    print("\nCompressing strong_16bit (HEVC Main12 CRF; best-effort from 16-bit sources) from SRC into mirror...")
    for tif in _iter_video_files(src, video_set):
        rel = tif.relative_to(src_p)
        out_mp4 = (dst_strong16_p / rel).with_suffix(".mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} -> {out_mp4.relative_to(dst_strong16_p)}")
        _compress_strong_16bit_hevc_main12(tif, out_mp4, fps=30.0, crf=8, preset="slow")

    # If you ever want *true* 16-bit (lossless), use this instead:
    # print("\nCompressing strong_16bit (FFV1 lossless gray16le) from SRC into mirror...")
    # for tif in _iter_video_files(src, video_set):
    #     rel = tif.relative_to(src_p)
    #     out_mkv = (dst_strong16_p / rel).with_suffix(".mkv")
    #     out_mkv.parent.mkdir(parents=True, exist_ok=True)
    #     print(f"  {rel} -> {out_mkv.relative_to(dst_strong16_p)}")
    #     _compress_strong_16bit_ffv1(tif, out_mkv, fps=30.0, level=3)

    # --- Keep for later (12-bit pipeline), but do not execute for 16-bit sources ---
    # print("\nCompressing strong_12bit (HEVC Main12) from SRC into mirror...")
    # for tif in _iter_video_files(src, video_set):
    #     rel = tif.relative_to(src_p)
    #     out_mp4 = (dst_strong12_p / rel).with_suffix(".mp4")
    #     out_mp4.parent.mkdir(parents=True, exist_ok=True)
    #     print(f"  {rel} -> {out_mp4.relative_to(dst_strong12_p)}")
    #     _compress_strong_12bit_hevc(tif, out_mp4, fps=30.0, crf=8, preset="slow")

    print("\nDone.")
    print(f"Loss_less mirror:        {dst_lossless}")
    print(f"For_viewing mirror:      {dst_view}")
    print(f"Strong_16bit mirror:     {dst_strong16}")
    print(f"(unused) Strong_12bit mirror:{dst_strong12}")


if __name__ == "__main__":
    compress_external_data_only()