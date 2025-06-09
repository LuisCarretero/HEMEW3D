# Documentation for GenCFD-prep

This document provides detailed documentation for the `GenCFD-prep` repository, which contains the preprocessing pipeline for the HemewS-3D dataset to prepare it for training with GenCFD-derived models.

## Overview

The preprocessing pipeline consists of two main stages:
1.  **Data Download**: Downloading the raw velocity trace data from the data host.
2.  **Data Preprocessing**: Processing the raw velocity traces and material property data into a format suitable for training GenCFD models. This involves filtering, subsampling, spatial interpolation, and organizing the data into HDF5 files.

## Data Download

The raw data for the HemewS-3D dataset is hosted on the [recherche.data.gouv.fr website](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/LAI6YU). The velocity trace data can be downloaded using the scripts provided in the `data-download` directory.

The underground velocity fields (material data) are assumed to be already downloaded and placed in the `data-download` directory as they are small.

### Scripts

The `data-download` directory contains the following key files:
-   `download_all.py`: The main Python script for downloading the data.
-   `utils.py`: Contains helper functions for parsing HTML and downloading files.
-   `download_all.sh`: A convenience shell script that wraps `download_all.py`.
-   `raw_table_of_contents_V*-velocity.txt`: Raw HTML content of the data repository's file listing, used to extract download links.
-   `download_urls_V*-velocity.csv`: CSV files where extracted download URLs are stored.

The `download_all.py` script has two main functions, triggered by command-line arguments:
-   `--extract`: Parses the `raw_table_of_contents...` file to extract the download links for the velocity data files and saves them to a `.csv` file.
-   `--download`: Downloads the files listed in the `.csv` file.

### Usage

You can run the download script directly or use the provided shell script.

**Using python:**

1.  **Extract links**:
    ```bash
    python GenCFD-prep/data-download/download_all.py --extract --versions 1 2
    ```
    This will create `download_urls_V1-velocity.csv` and `download_urls_V2-velocity.csv`.

2.  **Download files**:
    ```bash
    python GenCFD-prep/data-download/download_all.py --download --versions 1 --rawdata-dirpath /path/to/save/data
    ```

**Command-line arguments for `download_all.py`:**

| Argument                 | Description                                                                  | Default                                               |
| ------------------------ | ---------------------------------------------------------------------------- | ----------------------------------------------------- |
| `--extract`              | Extract links from the table of contents HTML file.                          | `False`                                               |
| `--download`             | Download velocity files from the extracted links.                            | `False`                                               |
| `--versions`             | List of dataset versions to process (1, 2, or both).                         | `1 2`                                                 |
| `--metadata-dirpath`     | Directory where metadata files (TOC and URL files) are stored.               | `./`                                                  |
| `--rawdata-dirpath`      | Base directory to save downloaded data files.                                | `/cluster/work/math/camlab-data/Wave_HemewS-3D`       |
| `--max-downloads`        | Maximum number of files to download (for testing purposes).                  | `None` (all files)                                    |


## Data Preprocessing

Once the raw data is downloaded, the next step is to preprocess it. The scripts in the `data-preprocessing` directory handle this. The preprocessing pipeline transforms the raw `.feather` trace files and `.npy` material files into structured HDF5 files.

### Scripts

-   `preprocess.py`: The main script for running the preprocessing pipeline.
-   `preprocessing_utils.py`: Contains the core functions for filtering, interpolation, and data transformation.
-   `preprocess.sh`: A convenience wrapper for `preprocess.py`.

### Processing Steps

1.  **Velocity Trace Processing**:
    -   **Filtering**: The raw velocity data, originally sampled at a predefined frequency, is low-pass filtered with a Butterworth filter. The default maximum frequency (`fmax`) is 5Hz.
    -   **Subsampling**: After filtering, the data is subsampled to a target frequency `f` (default 10Hz) and a fixed number of temporal steps `Nt` (default 64).
    -   **Spatial Interpolation**: The velocity fields, originally on a predefined grid, are spatially interpolated to a `S_out x S_out` grid (default 32x32).
    -   **Output**: The processed traces are saved in HDF5 files (`shard*.h5`). Each file contains data for multiple simulation runs, organized by velocity components (`uE`, `uN`, `uZ`).

2.  **Material Data Processing**:
    -   The material property data (from `.npy` files) is interpolated to the desired output grid (`S_out x S_out x Z_out`). Note that `Z_out` is currently hardcoded to `Nt` in `preprocessing_utils.py` as this is required by the GenCFD architecture.
    -   The processed material data is then added to the corresponding HDF5 shard files that contain the velocity traces.

3.  **Metadata**:
    -   A `metadata.json` file is created in the processed data directory.
    -   This file stores all the parameters used during the preprocessing run (`S_out`, `Nt`, `Z_out`, `f`, `fmax`, etc.), which is essential for the training phase to understand the data structure.
    -   Note: The `samples_per_file` parameter is currently hardcoded to `100` in `preprocessing_utils.py`, which is the number of velocity traces per file on the data host.

### Usage

**Using python:**

```bash
python GenCFD-prep/data-preprocessing/preprocess.py \
    --raw_data_path /path/to/raw/data/version1 \
    --processed_data_path /path/to/save/processed/data/version1 \
    --S_out 32 \
    --Nt 64 \
    --Z_out 64
```

This will create a directory `/path/to/save/processed/data/version1/inputs3D_S32_T64_fmax5.0` containing the processed HDF5 files and the `metadata.json` file.

**Command-line arguments for `preprocess.py`:**

| Argument                       | Description                                                                 | Default                                                      |
| ------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `--raw_data_path`              | Path to the raw data directory for a specific version.                      | `/cluster/work/math/camlab-data/Wave_HemewS-3D/version1`     |
| `--processed_data_path`        | Path where the processed data will be stored.                               | `/cluster/work/math/camlab-data/Wave_HemewS-3D/processed/version1` |
| `--S_out`                      | Spatial output dimension for the X and Y axes.                              | `32`                                                         |
| `--Nt`                         | Temporal output dimension (number of time steps).                           | `64`                                                         |
| `--Z_out`                      | Spatial output dimension for the Z axis (vertical).                         | `64`                                                         |
| `--f`                          | Target sampling frequency in Hz.                                            | `10`                                                         |
| `--fmax`                       | Maximum frequency for the low-pass filter in Hz.                            | `5`                                                          |
| `--max_files`                  | Maximum number of raw files to process.                                     | `9999` (all files)                                           |
| `--allow_different_Z_T_size`   | Allow `Z_out` and `Nt` to have different values. GenCFD requires `Z_out=Nt`. | `False`                                                      |

## How to Run the Full Pipeline

The provided shell scripts (`download_all.sh` and `preprocess.sh`) can be used to simplify the process.

1.  **Configure paths**: Before running, you might need to adjust the default paths within the `.sh` scripts or pass them as arguments if the scripts are extended to accept them. As they are now, they use the default paths in the python scripts.

2.  **Download the data**:
    ```bash
    cd GenCFD-prep/data-download
    ./download_all.sh
    ```
    This will first extract the links and then download the data for both V1 and V2 of the dataset.

3.  **Preprocess the data**:
    ```bash
    cd ../data-preprocessing
    ./preprocess.sh
    ```
    This will run the preprocessing pipeline on the downloaded data with the default parameters. Make sure the `raw_data_path` in `preprocess.py` points to where you downloaded the data.
