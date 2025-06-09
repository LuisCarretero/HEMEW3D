from typing import Union, Tuple, List

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import RegularGridInterpolator
import h5py
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
import json
from datetime import datetime


def process_trace_data(
    raw_data_path: str,
    processed_data_path: str,
    S_out: int = 32,
    Nt: int = 320,
    f: int = 50,
    fmax: float = 5,
    max_files: int = None,
) -> None:
    """
    Process trace data from raw data directory to processed data directory.
    This function handles the complete pipeline for processing velocity trace data:
    1. Sets up directories and validates parameters
    2. Identifies files that need processing
    3. Processes each file with spatial interpolation, filtering, and subsampling
    4. Organizes data into HDF5 format for efficient storage and access

    Args:
        raw_data_path: Path to raw data directory
        processed_data_path: Path to store processed data
        S_out: Spatial output dimension
        Nt: Temporal output dimension
        f: Sampling frequency in Hz
        fmax: Maximum frequency for filtering in Hz
        max_files: Maximum number of files to process
    """
    F_ORIG = 100  # Recording frequency of numerical simulation

    assert (
        F_ORIG % f == 0
    ), f"Requested sampling frequency {f}Hz that is not a divider of the recording frequency of 100Hz."
    assert (
        fmax <= 5
    ), f"Requested maximum frequency {fmax}Hz that is greater than the maximum frequency of the mesh (5Hz)."

    # Create processed data dir
    processed_subdir = f"inputs3D_S{S_out}_T{Nt}_fmax{fmax}"
    os.makedirs(os.path.join(processed_data_path, processed_subdir), exist_ok=True)

    # Get file paths and check for files already processed
    fpaths_raw = np.array(
        sorted(
            Path(raw_data_path).joinpath("velocity").iterdir(),
            key=lambda x: int(x.name.strip("velocity").strip(".feather").split("-")[0]),
        )
    )[:max_files]
    fpaths_processed = np.array(
        [
            os.path.join(processed_data_path, processed_subdir, f"shard{fnum}.h5")
            for fnum in range(len(fpaths_raw))
        ]
    )
    file_exists = np.array(
        [os.path.exists(fpath_processed) for fpath_processed in fpaths_processed]
    )
    print(
        f"There are {len(fpaths_raw)} files to process. {np.sum(file_exists)} files already processed. "
        f"Processing the remaining {np.sum(~file_exists)} files."
    )
    fpaths_raw, fpaths_processed = (
        fpaths_raw[~file_exists],
        fpaths_processed[~file_exists],
    )

    # Process (remaining) files
    for fpath_raw, fpath_processed in tqdm(
        zip(fpaths_raw, fpaths_processed),
        desc="Processing files",
        total=len(fpaths_raw),
    ):
        try:
            process_trace_file(
                fpath_raw=fpath_raw,
                fpath_processed=fpath_processed,
                interpolate=True,
                f_orig=F_ORIG,
                S_out=S_out,
                Nt=Nt,
                f=f,
                fmax=fmax,
            )
        except Exception as e:
            print(f"Error processing file {fpath_raw}: {e}")
            continue


def process_trace_file(
    fpath_raw: str,
    fpath_processed: str,
    interpolate: bool = True,
    f_orig: int = 100,
    S_out: int = 32,
    Nt: int = 320,
    f: int = 50,
    fmax: float = 5,
) -> None:
    """
    Process a single trace file from raw format to processed format.
    This function performs the core processing steps for individual velocity trace files:
    1. Prepares interpolation grid points if spatial interpolation is requested
    2. Loads raw data from feather files
    3. Applies frequency filtering and temporal subsampling
    4. Performs spatial interpolation to the desired output resolution
    5. Saves the processed data as HDF5 datasets with proper organization

    Args:
        fpath_raw: Path to raw trace file
        fpath_processed: Path to save processed file
        interpolate: Whether to perform spatial interpolation
        f_orig: Original sampling frequency in Hz
        S_out: Spatial output dimension
        Nt: Temporal output dimension
        f: Target sampling frequency in Hz
        fmax: Maximum frequency for filtering in Hz
    """
    if interpolate:
        x, z, t, points = calc_interpolation_points(S_out, Nt, f)

    # Load data
    trace_df = pd.read_feather(fpath_raw)

    # Filter and subsample
    trace_df = filter_and_subsample(trace_df, f_orig, f, fmax, Nt)

    # Interpolate and write to file
    with h5py.File(fpath_processed, "w") as file:
        file.attrs["sampleIDs"] = trace_df.run.unique().astype(int)
        for comp in ["E", "N", "Z"]:
            comp_trace = (
                trace_df.loc[(trace_df.field == f"Veloc {comp}")]
                .iloc[:, 2:]
                .values.reshape(-1, 16, 16, Nt)
            )
            file.create_group(f"u{comp}")
            for i in range(comp_trace.shape[0]):

                # Spatial interpolation
                if interpolate:
                    f_interp = RegularGridInterpolator(
                        points=(x, z, t),
                        values=comp_trace[i],
                        bounds_error=True,
                        method="nearest",
                    )
                    u = f_interp(points).reshape(S_out, S_out, Nt)
                else:
                    u = comp_trace[i]

                # Write to file
                file[f"u{comp}"].create_dataset(f"sample{i}", data=u.astype(np.float32))


def process_material_data(
    raw_data_path: str,
    processed_data_path: str,
    S_out: int = 32,
    Z_out: int = 64,
    Nt: int = 320,
    fmax: float = 5,
) -> None:
    """
    Process material data from raw data directory to processed data directory.
    This function handles the processing of material property data:
    1. Identifies material files and their corresponding velocity trace data
    2. Maps material data to the corresponding velocity trace shards
    3. Processes each material file with appropriate interpolation
    4. Adds the material data to the existing HDF5 files containing velocity traces

    Args:
        raw_data_path: Path to raw data directory
        processed_data_path: Path to store processed data
        S_out: Spatial output dimension (X,Y)
        Z_out: Vertical spatial output dimension
        Nt: Temporal dimension
        fmax: Maximum frequency for filtering in Hz
    """
    SAMPLES_PER_SHARD = 100

    processed_subdir = f"inputs3D_S{S_out}_T{Nt}_fmax{fmax}"

    # Get materials data fpaths and sort by index
    fpaths_raw = np.array(list(Path(raw_data_path).joinpath("materials").iterdir()))
    indices = np.array(
        list(
            map(
                lambda x: tuple(
                    map(int, x.name.strip("materials").strip(".npy").split("-"))
                ),
                fpaths_raw,
            )
        )
    )
    order = np.argsort(indices[:, 0])
    fpaths_raw, indices = fpaths_raw[order], indices[order]

    # Get processed trace data shard ranges
    tmp = []
    for fpath in Path(processed_data_path).joinpath(processed_subdir).glob("shard*.h5"):
        with h5py.File(fpath, "r") as f:
            tmp.extend(
                [(str(fpath), sample_idx) for sample_idx in f.attrs["sampleIDs"]]
            )
    df = pd.DataFrame(tmp, columns=["fpath", "sample_idx"])

    # Process each material file
    for (trace_idx_start, trace_idx_end), fpath in tqdm(
        zip(indices, fpaths_raw), desc="Processing material files", total=len(indices)
    ):
        contained_sample_ids = set(range(trace_idx_start, trace_idx_end + 1))
        id_intersection = set(df.sample_idx).intersection(contained_sample_ids)
        assert (
            len(id_intersection) > 0
        ), f"No intersection between sample IDs in material file {fpath} and the ones from {processed_data_path}."

        shard_ranges = []
        unique_shards = df[df.sample_idx.isin(id_intersection)]["fpath"].unique()

        for shard_path in unique_shards:
            # Get all sample IDs in this shard that are also in our material file
            shard_samples = set(
                df[df.fpath == shard_path].sample_idx.values
            ).intersection(id_intersection)
            assert (
                len(shard_samples) == SAMPLES_PER_SHARD
            ), f"Shard {shard_path} contains {len(shard_samples)} samples instead of {SAMPLES_PER_SHARD}."

            shard_ranges.append((shard_path, np.array(sorted(shard_samples))))

        process_material_file(
            fpath_raw=fpath, shard_idx_ranges=shard_ranges, S_out=S_out, Z_out=Z_out
        )


def process_material_file(
    fpath_raw: str,
    shard_idx_ranges: List[Tuple[str, np.ndarray]],
    S_out: int = 32,
    Z_out: int = 64,
) -> None:
    """
    Process a single material file and write to corresponding shards.
    This function processes a material property file and integrates it with velocity data:
    1. Loads the raw material data
    2. Applies spatial interpolation to match desired dimensions
    3. Updates multiple HDF5 shard files with the corresponding material data
    4. Handles indexing to ensure correct association between materials and velocity traces

    Args:
        fpath_raw: Path to raw material file
        shard_idx_ranges: List of tuples (shard_path, sample_indices)
        S_out: Spatial output dimension (X,Y)
        Z_out: Vertical spatial output dimension
    """
    # Load and reshape data
    arr = np.load(fpath_raw)
    arr = reshape_vel_batch(arr[:, :32, :32, :32], (S_out, S_out, Z_out))

    # Write to shards
    for shard_path, shard_indices in shard_idx_ranges:
        with h5py.File(shard_path, "a") as f:
            idx_offset = f.attrs["sampleIDs"][0]
            if f.get("material"):
                del f["material"]
            f.create_group(f"material")
            local_indices = shard_indices - idx_offset
            for loc_idx in local_indices:
                f["material"].create_dataset(f"sample{loc_idx}", data=arr[loc_idx])


def calc_interpolation_points(
    S_out: int, Nt: int, f: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate interpolation points for spatial interpolation.
    This function sets up the grid points and query points for spatial interpolation:
    1. Defines the original and target grid coordinates
    2. Creates a multiindex for efficient interpolation
    3. Returns all necessary arrays for the RegularGridInterpolator

    Args:
        S_out: Spatial output dimension
        Nt: Temporal output dimension
        f: Sampling frequency in Hz

    Returns:
        Tuple of (x, z, t, points) arrays for interpolation
    """
    x = np.linspace(150, 9450, 16)
    x2 = np.linspace(150, 9450, S_out)
    z = np.linspace(150, 9450, 16)
    z2 = np.linspace(150, 9450, S_out)
    t = np.linspace(
        0, (Nt - 1) / f, Nt
    )  # values are not important because interpolation is only on x,y

    # array of points containing the new values of the grid points
    indices = pd.MultiIndex.from_product([x2, z2, t])
    indices = indices.to_frame(name=["x", "z", "t"])
    points = indices.sort_values(by=["x", "z", "t"]).values

    return x, z, t, points


def filter_and_subsample(
    df: pd.DataFrame, f_orig: int, f: int, fmax: float, Nt: int
) -> pd.DataFrame:
    """
    Filter and subsample trace data.
    This function performs temporal processing of the velocity data:
    1. Organizes and cleans the input dataframe
    2. Applies a low-pass Butterworth filter to remove high frequencies
    3. Subsamples the data to the desired temporal resolution
    4. Returns the processed data maintaining the original structure

    Args:
        df: DataFrame containing trace data
        f_orig: Original sampling frequency in Hz
        f: Target sampling frequency in Hz
        fmax: Maximum frequency for filtering in Hz
        Nt: Target number of time steps

    Returns:
        Filtered and subsampled DataFrame
    """
    # Clean df
    df.sort_values(by=["run", "y", "x"], inplace=True)
    df.drop(["x", "y", "z"], axis=1, inplace=True)

    # Filter out higher frequencies using low-pass butter filter
    raw_traces = df.iloc[:, 2:]  # Trace data, shape=(nruns*3*Nx*Ny, Nt)
    traces_filtered = pd.DataFrame(
        butter_lowpass_filter(raw_traces, fmax, sample_rate=f_orig, order=4),
        index=df.index,
        columns=df.columns[2:],
    )
    traces_filtered = pd.concat([df.loc[:, ["run", "field"]], traces_filtered], axis=1)

    # Subsample
    traces_short = pd.concat(
        [
            df.loc[:, ["run", "field"]],
            # traces_filtered.iloc[f_orig:(Nt+f)*subsampling_fact:subsampling_fact]
            traces_filtered.iloc[:, :: int(100 / f)].iloc[:, f : Nt + f],
        ],
        axis=1,
    )
    """
    s = pd.Series(np.arange(10032))
    a = s.iloc[::int(100/f)].iloc[f:Nt+f]
    b = s.iloc[f_orig:(Nt+f)*subsampling_fact:subsampling_fact]
    """

    return traces_short


def butter_lowpass_filter(
    data: Union[np.ndarray, pd.Series],
    cutoff: float,
    sample_rate: float = 100,
    order: int = 4,
) -> Union[np.ndarray, pd.Series]:
    """
    Apply a Butterworth lowpass filter to the input data.
    This function implements a standard signal processing filter:
    1. Calculates the normalized cutoff frequency
    2. Designs the Butterworth filter with specified parameters
    3. Applies the filter using forward-backward filtering for zero phase shift
    4. Preserves the input data format (Series or ndarray)

    Args:
        data: Input data to be filtered
        cutoff: Cutoff frequency in Hz
        sample_rate: Sampling rate in Hz
        order: Filter order

    Returns:
        Filtered data in the same format as input
    """
    if isinstance(data, pd.Series):
        idx = data.index

    nyquist_freq = sample_rate / 2
    normalized_cutoff = cutoff / nyquist_freq

    # Get the filter coefficients
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)

    y = filtfilt(b, a, data)

    if isinstance(data, pd.Series):
        return pd.Series(y, index=idx)

    return y


def read_processed_trace_data(
    fpath: str, comp: str = "E", sample: int = 0
) -> np.ndarray:
    """
    Read processed trace data from HDF5 file.
    This utility function provides a simple interface to access processed data:
    1. Opens the HDF5 file and navigates to the requested component and sample
    2. Loads the data into memory as a numpy array
    3. Ensures proper cleanup of file resources

    Args:
        fpath: Path to HDF5 file
        comp: Component ('E', 'N', or 'Z')
        sample: Sample index

    Returns:
        Numpy array of trace data
    """
    with h5py.File(fpath, "r") as f:
        data = f[f"u{comp}"][f"sample{sample}"][:]
    return data


def reshape_vel_batch(vel_batch: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Vectorized trilinear interpolation of a batch of 3D velocity fields.
    This function uses PyTorch's efficient interpolation for 3D data:
    1. Converts numpy arrays to PyTorch tensors
    2. Applies trilinear interpolation to reach target dimensions
    3. Converts back to numpy arrays for compatibility with other functions

    Args:
        vel_batch: Array of shape (N, X, Y, Z) containing N samples
        target_shape: Desired output shape (X', Y', Z')

    Returns:
        Array of shape (N, X', Y', Z') with interpolated fields
    """
    # vel_batch -> torch tensor with shape (N, 1, X, Y, Z)
    t = torch.from_numpy(vel_batch).unsqueeze(1).float()

    # interpolate to (N, 1, X', Y', Z')
    t_up = F.interpolate(
        t,
        size=target_shape,
        mode="trilinear",
        align_corners=True,  # preserve endpoints like linspace(0,1)
        recompute_scale_factor=False,
    )

    # back to numpy, squeeze out the channel dim
    return t_up.squeeze(1).numpy()


def write_metadata(
    processed_data_path: str,
    S_out: int,
    Nt: int,
    Z_out: int,
    f: int,
    fmax: float,
    samples_per_file: int,
    additional_params: dict = None,
) -> None:
    """
    Write metadata about the preprocessing parameters to a JSON file.

    Args:
        processed_data_path: Path to the directory where processed data is stored
        S_out: Spatial output dimension (X and Y)
        Nt: Temporal output dimension
        Z_out: Vertical spatial output dimension
        f: Sampling frequency used in preprocessing
        fmax: Maximum frequency for filtering
        samples_per_file: Number of samples per file
        additional_params: Any additional parameters to include in the metadata

    Returns:
        None
    """

    # Create metadata dictionary
    metadata = {
        "creation_date": datetime.now().isoformat(),
        "preprocessing_parameters": {
            "S_out": S_out,
            "Nt": Nt,
            "Z_out": Z_out if Z_out is not None else Nt,
            "sampling_frequency": f,
            "max_frequency": fmax,
            "samples_per_file": samples_per_file,
        },
    }

    # Add any additional parameters
    if additional_params is not None:
        metadata["preprocessing_parameters"].update(additional_params)

    # Determine the processed subdirectory based on parameters
    processed_subdir = f"inputs3D_S{S_out}_T{Nt}_fmax{fmax}"

    # Create the full path
    metadata_path = os.path.join(processed_data_path, processed_subdir, "metadata.json")

    # Write metadata to file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata written to {metadata_path}")
