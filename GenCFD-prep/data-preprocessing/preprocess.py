from preprocessing_utils import process_trace_data, process_material_data

if __name__ == '__main__':
    raw_data_path = r'/cluster/work/math/camlab-data/Wave_HemewS-3D/version1'
    processed_data_path = r'/cluster/work/math/camlab-data/Wave_HemewS-3D/processed/version1-2'
    S_out = 32
    Nt = Z_out = 64
    f = 10
    fmax = 5
    max_files = 9999

    print('Processing trace data...')
    process_trace_data(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        S_out=S_out,
        Nt=Nt,
        f=f,
        fmax=fmax,
        max_files=max_files
    )

    print('Processing material data...')
    process_material_data(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        S_out=S_out,
        Z_out=Z_out,
        Nt=Nt,
        fmax=fmax
    )
