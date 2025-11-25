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
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import heka_reader
import git_save as myGit
from collections import defaultdict
import json
from statistics import median


def iGluSnFr_eval():

    # --- Create Output Folders ---
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    output_folder = os.path.join(config.ROOT_FOLDER, f"output_{config.MY_INITIAL}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    output_folder_results = os.path.join(output_folder, "results")
    os.makedirs(output_folder_results, exist_ok=True)

    output_folder_ROIs = os.path.join(output_folder, "ROIs")
    os.makedirs(output_folder_traces, exist_ok=True)

    output_folder_used_data_and_code = os.path.join(output_folder, "used_data_and_code")
    os.makedirs(output_folder_used_data_and_code, exist_ok=True)

    # --- Load Metadata ---
    metadata_df = pd.read_excel(config.METADATA_FILE)
    # save used data
    metadata_df.to_excel(os.path.join(output_folder_used_data_and_code, "my_data.xlsx"), index=False)

        # === GIT SAVE ===
    # Provide the current script path (only works in .py, not notebooks)
    script_path = __file__ if '__file__' in globals() else None
    myGit.save_git_info(output_folder_used_data_and_code, script_path)

    # --- Results Storage ---
    results = []


    # --- Process Each Experiment ---
    for experiment_count, row in metadata_df.iterrows():
        experimentName = row['experimentName']
        print(f"Processing experiment {experiment_count + 1}: {experimentName}")


if __name__ == '__main__':
    iGluSnFr_eval()
