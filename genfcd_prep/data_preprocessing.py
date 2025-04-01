from typing import Union

import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import RegularGridInterpolator
import h5py
from tqdm import tqdm


def process_data(
    raw_data_path: str,
    processed_data_path: str,
    S_out: int = 32,
    Nt: int = 320,
    f: int = 50,
    fmax: float = 5,
    max_files: int = None
):  
    f_orig = 100

    assert f_orig % f == 0, f'Requested sampling frequency {f}Hz that is not a divider of the recording frequency of 100Hz.'
    assert fmax <= 5, f'Requested maximum frequency {fmax}Hz that is greater than the maximum frequency of the mesh (5Hz).'

    processed_subdir = f'inputs3D_S{S_out}_Z{S_out}_T{Nt}_fmax{fmax}' # folder containing ML inputs

    os.makedirs(os.path.join(processed_data_path, processed_subdir), exist_ok=True)

    fnames = list(sorted(os.listdir(os.path.join(raw_data_path, 'velocity'))))[:max_files]
    for fnum, fname in tqdm(enumerate(fnames), desc='Processing files', total=len(fnames)):
        fpath_raw = os.path.join(raw_data_path, 'velocity', fname)

        process_file(
            fpath_raw=fpath_raw,
            fpath_processed=os.path.join(processed_data_path, processed_subdir, f'shard{fnum}.h5'),
            interpolate=True,
            f_orig=f_orig,
            S_out=S_out,
            Nt=Nt,
            f=f,
            fmax=fmax
        )

def process_file(
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

def calc_interpolation_points(S_out: int, Nt: int, f: int) -> np.ndarray:
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

def read_processed_data(
    fpath: str,
    comp: str = 'E',
    sample: int = 0
) -> np.ndarray:
    with h5py.File(fpath, 'r') as f:
        data = f[f'u{comp}'][f'sample{sample}'][:]
    return data