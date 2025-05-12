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


def process_trace_data(
    raw_data_path: str,
    processed_data_path: str,
    S_out: int = 32,
    Nt: int = 320,
    f: int = 50,
    fmax: float = 5,
    max_files: int = None
):  
    F_ORIG = 100  # Recording frequency of numerical simulation

    assert F_ORIG % f == 0, f'Requested sampling frequency {f}Hz that is not a divider of the recording frequency of 100Hz.'
    assert fmax <= 5, f'Requested maximum frequency {fmax}Hz that is greater than the maximum frequency of the mesh (5Hz).'

    # Create processed data dir
    processed_subdir = f'inputs3D_S{S_out}_T{Nt}_fmax{fmax}'
    os.makedirs(os.path.join(processed_data_path, processed_subdir), exist_ok=True)

    # Get file paths and check for files already processed
    fpaths_raw = np.array(sorted(
        Path(raw_data_path).joinpath('velocity').iterdir(), 
        key=lambda x: int(x.name.strip('velocity').strip('.feather').split('-')[0])
    ))[:max_files]
    fpaths_processed = np.array([
        os.path.join(processed_data_path, processed_subdir, f'shard{fnum}.h5') 
        for fnum in range(len(fpaths_raw))
    ])
    file_exists = np.array([os.path.exists(fpath_processed) for fpath_processed in fpaths_processed])
    print(f'There are {len(fpaths_raw)} files to process. {np.sum(file_exists)} files already processed. '
        f'Processing the remaining {np.sum(~file_exists)} files.')
    fpaths_raw, fpaths_processed = fpaths_raw[~file_exists], fpaths_processed[~file_exists]

    # Process (remaining) files
    for fpath_raw, fpath_processed in tqdm(zip(fpaths_raw, fpaths_processed), desc='Processing files', total=len(fpaths_raw)):
        try:
            process_trace_file(
                fpath_raw=fpath_raw,
                fpath_processed=fpath_processed,
                interpolate=True,
                f_orig=F_ORIG,
                S_out=S_out,
                Nt=Nt,
                f=f,
                fmax=fmax
            )
        except Exception as e:
            print(f'Error processing file {fpath_raw}: {e}')
            continue

def process_trace_file(
    fpath_raw: str, 
    fpath_processed: str,
    interpolate: bool = True,
    f_orig: int = 100,
    S_out: int = 32,
    Nt: int = 320,
    f: int = 50,
    fmax: float = 5
) -> None:

    if interpolate:
        x, z, t, points = calc_interpolation_points(S_out, Nt, f)

    # Load data
    trace_df = pd.read_feather(fpath_raw)

    # Filter and subsample
    trace_df = filter_and_subsample(trace_df, f_orig, f, fmax, Nt)

    # Interpolate and write to file
    with h5py.File(fpath_processed, 'w') as file:
        file.attrs['sampleIDs'] = trace_df.run.unique().astype(int)
        for comp in ['E', 'N', 'Z']:
            comp_trace = trace_df.loc[(trace_df.field == f'Veloc {comp}')].iloc[:, 2:].values.reshape(-1, 16, 16, Nt)
            file.create_group(f'u{comp}')
            for i in range(comp_trace.shape[0]):
                
                # Spatial interpolation
                if interpolate:
                    f_interp = RegularGridInterpolator(
                        points=(x, z, t), values=comp_trace[i], bounds_error=True, method='nearest'
                    )
                    u = f_interp(points).reshape(S_out, S_out, Nt)
                else:
                    u = comp_trace[i]

                # Write to file
                file[f'u{comp}'].create_dataset(f'sample{i}', data=u.astype(np.float32))

def process_material_data(
    raw_data_path: str,
    processed_data_path: str,
    S_out: int = 32,
    Z_out: int = 64,
    Nt: int = 320,
    fmax: float = 5
) -> None:
    """
    Process material data from raw data directory to processed data directory.
    """
    SAMPLES_PER_SHARD = 100

    processed_subdir = f'inputs3D_S{S_out}_T{Nt}_fmax{fmax}'

    # Get materials data fpaths and sort by index
    fpaths_raw = np.array(list(Path(raw_data_path).joinpath('materials').iterdir()))
    indices = np.array(list(map(
        lambda x: tuple(map(int, x.name.strip('materials').strip('.npy').split('-'))), 
        fpaths_raw
    )))
    order = np.argsort(indices[:, 0])
    fpaths_raw, indices = fpaths_raw[order], indices[order]

    # Get processed trace data shard ranges
    tmp = []
    for fpath in Path(processed_data_path).joinpath(processed_subdir).glob('shard*.h5'):
        with h5py.File(fpath, 'r') as f:
            tmp.extend([(str(fpath), sample_idx) for sample_idx in f.attrs['sampleIDs']])
    df = pd.DataFrame(tmp, columns=['fpath', 'sample_idx'])

    # Process each material file
    for (trace_idx_start, trace_idx_end), fpath in tqdm(zip(indices, fpaths_raw), desc='Processing material files', total=len(indices)):
        contained_sample_ids = set(range(trace_idx_start, trace_idx_end+1))
        id_intersection = set(df.sample_idx).intersection(contained_sample_ids)
        assert len(id_intersection) > 0, \
            f'No intersection between sample IDs in material file {fpath} and the ones from {processed_data_path}.'
        
        shard_ranges = []
        unique_shards = df[df.sample_idx.isin(id_intersection)]['fpath'].unique()
        
        for shard_path in unique_shards:
            # Get all sample IDs in this shard that are also in our material file
            shard_samples = set(df[df.fpath == shard_path].sample_idx.values).intersection(id_intersection)
            assert len(shard_samples) == SAMPLES_PER_SHARD, \
                f'Shard {shard_path} contains {len(shard_samples)} samples instead of {SAMPLES_PER_SHARD}.'
            
            shard_ranges.append((shard_path, np.array(sorted(shard_samples))))

        process_material_file(
            fpath_raw=fpath,
            shard_idx_ranges=shard_ranges,
            S_out=S_out,
            Z_out=Z_out
        )

def process_material_file(
    fpath_raw: str,
    shard_idx_ranges: List[Tuple[str, np.ndarray]],
    S_out: int = 32,
    Z_out: int = 64
) -> None:
    # Load and reshape data
    arr = np.load(fpath_raw)
    arr = reshape_vel_batch(arr[:, :32, :32, :32], (S_out, S_out, Z_out))

    # Write to shards
    for shard_path, shard_indices in shard_idx_ranges:
        with h5py.File(shard_path, 'a') as f:
            idx_offset = f.attrs['sampleIDs'][0]
            if f.get('material'):
                del f['material']
            f.create_group(f'material')
            local_indices = shard_indices - idx_offset
            for loc_idx in local_indices:
                f['material'].create_dataset(f'sample{loc_idx}', data=arr[loc_idx])

def calc_interpolation_points(S_out: int, Nt: int, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(150, 9450, 16)
    x2 = np.linspace(150, 9450, S_out)
    z = np.linspace(150, 9450, 16)
    z2 = np.linspace(150, 9450, S_out)
    t = np.linspace(0, (Nt-1)/f, Nt) # values are not important because interpolation is only on x,y

    # array of points containing the new values of the grid points
    indices = pd.MultiIndex.from_product([x2, z2, t])
    indices = indices.to_frame(name=['x','z','t'])
    points = indices.sort_values(by=['x','z','t']).values

    return x, z, t, points

def filter_and_subsample(
    df: pd.DataFrame, 
    f_orig: int, 
    f: int,
    fmax: float, 
    Nt: int
) -> pd.DataFrame:

    # Clean df
    df.sort_values(by=['run', 'y', 'x'], inplace=True)
    df.drop(['x','y','z'], axis=1, inplace=True)

    # Filter out higher frequencies using low-pass butter filter
    raw_traces = df.iloc[:, 2:]  # Trace data, shape=(nruns*3*Nx*Ny, Nt)
    traces_filtered = pd.DataFrame(
        butter_lowpass_filter(raw_traces, fmax, sample_rate=f_orig, order=4),
        index = df.index, 
        columns = df.columns[2:]
    )
    traces_filtered = pd.concat([
        df.loc[:, ['run','field']], 
        traces_filtered
    ], axis=1)

    # Subsample
    traces_short = pd.concat([
        df.loc[:, ['run','field']],
        # traces_filtered.iloc[f_orig:(Nt+f)*subsampling_fact:subsampling_fact] 
        traces_filtered.iloc[:, ::int(100/f)].iloc[:, f:Nt+f]
    ], axis=1)
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
    order: int = 4
) -> Union[np.ndarray, pd.Series]:
    """
    Apply a Butterworth lowpass filter to the input data.
    
    Parameters:
    -----------
    data: Union[np.ndarray, pd.Series]
        Input data to be filtered
    cutoff: float
        Cutoff frequency in Hz
    sample_rate: float
        Sampling rate in Hz
    order: int
        Filter order
        
    Returns:
    --------
    Union[np.ndarray, pd.Series]
        Filtered data in the same format as input
    """
    if isinstance(data, pd.Series):
        idx = data.index
        
    nyquist_freq = sample_rate/2
    normalized_cutoff = cutoff/nyquist_freq

    # Get the filter coefficients
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    y = filtfilt(b, a, data)
    
    if isinstance(data, pd.Series):
        return pd.Series(y, index=idx)
    
    return y

def read_processed_trace_data(
    fpath: str,
    comp: str = 'E',
    sample: int = 0
) -> np.ndarray:
    with h5py.File(fpath, 'r') as f:
        data = f[f'u{comp}'][f'sample{sample}'][:]
    return data

def reshape_vel_batch(vel_batch: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Vectorized trilinear interpolation of a batch of 3D velocity fields.
    
    Parameters
    ----------
    vel_batch : np.ndarray
        Array of shape (N, X, Y, Z) containing N samples.
    target_shape : tuple of int
        Desired output shape (X', Y', Z').
    
    Returns
    -------
    np.ndarray
        Array of shape (N, X', Y', Z') with interpolated fields.
    """
    # vel_batch -> torch tensor with shape (N, 1, X, Y, Z)
    t = torch.from_numpy(vel_batch).unsqueeze(1).float()
    
    # interpolate to (N, 1, X', Y', Z')
    t_up = F.interpolate(
        t,
        size=target_shape,
        mode='trilinear',
        align_corners=True,     # preserve endpoints like linspace(0,1)
        recompute_scale_factor=False
    )
    
    # back to numpy, squeeze out the channel dim
    return t_up.squeeze(1).numpy()