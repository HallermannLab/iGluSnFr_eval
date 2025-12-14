# iGluSnFR_eval

This project processes iGluSnFR imaging experiments organized in *experiment folders* containing multiple *block folders* (e.g. `A`, `B`, … or special blocks like `X1`). It generates:

- **Difference images** (stimulus – baseline) per block
- **ROI-based intensity traces** exported as CSV
- **Per-ROI PDF reports** (images + traces)
- **Per-block “averages” PDF reports** (one-letter blocks only)
- **Excel summaries** (release probability, weighted amplitude, mito intensity)

The workflow is driven by a **metadata Excel file** (“metafile”). A template with the required column names and typical values is included as:

- `metadata_example.xlsx`


---

## 1) Inputs

### 1.1 Configuration (`config.py`)
You must provide a local `config.py` (not committed) containing at least:

- `ROOT_FOLDER`: output root directory for generated results
- `EXTERNAL_DATA_FOLDER`: base folder containing experiment folders
- `METADATA_FILE`: path to the metadata Excel file to use
- `MY_INITIAL`: initials used in the output folder name

(See `config_template.py`.)

> Start by copying `config_template.py` to `config.py` and updating the paths.

### 1.2 Metadata Excel (“metafile”)
The program reads the Excel file pointed to by `config.METADATA_FILE`. Each row describes one experiment to process.

The required column names and meanings are documented by example in:

- `metadata_example.xlsx`

**Key idea:** Each row provides:
- the experiment identifier (used as folder name),
- acquisition timing parameters,
- difference-image window parameters (baseline/stimulus windows),
- plotting parameters (zoom size, optional plot limits),
- and analysis parameters (filter cutoff, baseline/max windows, stimulation timing).

#### Difference-image parameters
The program computes a 2D difference image per block as:

> **Diff = mean(stim window) − mean(baseline window)**, clipped at 0

There are two parameter sets in the metafile:

**Standard blocks** (one-letter blocks like `A`, `B`, …):
- `Diff_BL_Start`, `Diff_BL_End`
- `Diff_Stim_Start`, `Diff_Stim_End`

**Special blocks** (block name has more than one character, e.g. `X1`):
- `Diff_BL_Start_Induction`, `Diff_BL_End_Induction`
- `Diff_Stim_Start_Induction`, `Diff_Stim_End_Induction`

All values are given in **milliseconds** and converted to frame indices using:
- `acquisition time (ms)`

### 1.3 Experiment folder layout

The program expects the following folder structure:

#### Standard blocks (one-letter block names)
Required files in each block folder:
- `ap1+train.tif` (required)
- `ap2.tif`, `ap3.tif`, `ap4.tif`, `ap5.tif` (optional but typically present)
- `mito.tif` (recommended; used for mito overlay + mito intensity)
- `ROIs.zip` (required for ROI intensity extraction and ROI PDFs)

#### Special blocks (block name length > 1, e.g. `X1`)
Required files:
- `ind.tif` (required)

Optional files:
- `ROIs.zip` (if missing, ROI extraction + plotting are skipped)
- `mito.tif` (needed for mito overlay + mito intensity; if missing, ROI PDFs/mito output are skipped)

### 1.4 ROI definitions (`ROIs.zip`)
If present, ROIs are read from `ROIs.zip` (ImageJ/Fiji ROI zip). Supported ROI types:
- polygon/freehand
- rectangle
- oval

---

## 2) What the program does

Run entry point:

- `python main.py`

For each experiment (row in the metadata Excel), the program:
1. Iterates through each block folder.
2. Computes a difference image:
   - standard blocks: from `ap1+train.tif` using `Diff_*`
   - special blocks: from `ind.tif` using `Diff_*_Induction`
3. If `ROIs.zip` exists:
   - extracts ROI mean intensity over time and writes CSV files
   - generates PDF reports
4. For standard blocks only:
   - runs the existing event-based analysis to compute:
     - release probability
     - weighted mean amplitude
   - generates an “averages” PDF (mean across ROIs)

---

## 3) Outputs

Each run creates a timestamped output folder:

### 3.1 `DiffImage/`
Per block, a computed difference image:

- Standard blocks: `DiffImage/<block>_diff.tif`
- Special blocks: `DiffImage/<block>_diff.tif` (computed from `ind.tif`)

### 3.2 `CSVs/`
ROI intensity traces extracted from the video stacks.

Standard blocks:
- `CSVs/<block>/ap1+train.csv`
- `CSVs/<block>/ap2.csv`
- `CSVs/<block>/ap3.csv`
- `CSVs/<block>/ap4.csv`
- `CSVs/<block>/ap5.csv`

Special blocks:
- `CSVs/<block>/ind.csv`

Each CSV contains:
- columns = ROI names
- rows = frames
- values = mean pixel intensity inside each ROI for that frame

### 3.3 `ROIs/` (per-ROI PDFs)
Per ROI *and* block, a PDF is generated:

- `ROIs/<ROI_name>_<block>.pdf`

#### Standard blocks (one-letter)
These PDFs include:
- Diff image + zoom, both with ROI outline overlay
- Mito image + zoom, both with ROI outline overlay (if `mito.tif` exists)
- A “train” trace panel (ROI intensity over time) and additional analysis panels from the original workflow

#### Special blocks (e.g. `X1`)
If `ROIs.zip` exists, each ROI PDF includes:
- Row 1: diff image + zoom (ROI overlay)
- Row 2: mito image + zoom (ROI overlay) (requires `mito.tif`)
- Bottom: **one** full-duration trace (ROI intensity vs time from `ind.csv`)

If `ROIs.zip` is missing:
- ROI CSV extraction and ROI PDFs are skipped for that block.

### 3.4 `averages/` (block-level averages PDFs; standard blocks only)
For each one-letter block:

- `averages/<block>.pdf`

Contains:
- Diff image (no zoom, no ROI overlay)
- Mito image (no zoom, no ROI overlay)
- Mean trace across all ROIs for `ap1+train.csv`
- Mean traces across all ROIs for `ap2.csv` … `ap5.csv`

No release probability / amplitude analysis overlays are shown in these average plots.

### 3.5 `results/` (Excel summaries)
Created per experiment:

- `results/release_probability.xlsx`
- `results/wheighted_amplitude.xlsx`
- `results/mito_intensity.xlsx`

Each file is shaped like:
- rows = ROI names
- columns = block names
- values = computed metric for that ROI in that block

Notes:
- Special blocks only contribute to `mito_intensity.xlsx` **if** `ROIs.zip` and `mito.tif` exist.
- If a block lacks ROIs (no `ROIs.zip`), that block won’t appear in ROI-based tables.

---

## 4) Common warnings and what they mean

- **“ap1+train.tif not found. Skipping block.”**
  - Standard block is missing `ap1+train.tif`.
  - Special blocks use `ind.tif` instead.

- **“ROIs.zip not found … Skipping ROI extraction …”**
  - Block has no ROIs. The program still may create the diff image, but will not produce ROI CSVs/PDFs or ROI-based Excel entries for that block.

- **“Diff indices exceed stack length …”**
  - One of the Diff_* time windows exceeds the number of frames in the `.tif` stack.
  - Fix by adjusting Diff_* values in the metadata Excel or verifying `acquisition time (ms)`.

---

## 5) How to run

1. Create `config.py` from `config_template.py` and edit paths.
2. Ensure your metadata Excel file exists and matches the columns shown in `metadata_example.xlsx`.
3. Run:

---

## 6) Notes on dependencies

Dependencies are listed in `requirements.txt`. The project expects a Python environment that can import:
- numpy, pandas, scipy, matplotlib
- scikit-image (`skimage`)
- read-roi (for ImageJ ROI zip parsing)
