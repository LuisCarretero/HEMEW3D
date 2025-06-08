import argparse


from preprocessing_utils import process_trace_data, process_material_data, write_metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process HemewS-3D data')
    parser.add_argument('--raw_data_path', type=str, 
                        default='/cluster/work/math/camlab-data/Wave_HemewS-3D/version1',
                        help='Path to raw data')
    parser.add_argument('--processed_data_path', type=str,
                        default='/cluster/work/math/camlab-data/Wave_HemewS-3D/processed/version1',
                        help='Path to store processed data')
    parser.add_argument('--S_out', type=int, default=32,
                        help='Spatial output dimension')
    parser.add_argument('--Nt', type=int, default=64,
                        help='Temporal output dimension')
    parser.add_argument('--Z_out', type=int, default=64,
                        help='Vertical spatial output dimension')
    parser.add_argument('--f', type=int, default=10,
                        help='Sampling frequency')
    parser.add_argument('--fmax', type=float, default=5,
                        help='Maximum frequency for filtering')
    parser.add_argument('--max_files', type=int, default=9999,
                        help='Maximum number of files to process')
    parser.add_argument('--allow_different_Z_T_size', action='store_true',
                        help='Allow different Z_out and Nt sizes')
    
    args = parser.parse_args()

    if args.Z_out != args.Nt:
        if not args.allow_different_Z_T_size:
            raise ValueError('Z_out and Nt must be the same if allow_different_Z_T_size is False. Note that GenCFD requires Z_out = Nt.')
        else:
            print('Warning: Z_out and Nt are different. This is allowed if allow_different_Z_T_size is True.')

    write_metadata(
        processed_data_path=args.processed_data_path,
        S_out=args.S_out,
        Nt=args.Nt,
        Z_out=args.Z_out,
        f=args.f,
        fmax=args.fmax
    )

    print('Processing trace data...')
    process_trace_data(
        raw_data_path=args.raw_data_path,
        processed_data_path=args.processed_data_path,
        S_out=args.S_out,
        Nt=args.Nt,
        f=args.f,
        fmax=args.fmax,
        max_files=args.max_files
    )

    print('Processing material data...')
    process_material_data(
        raw_data_path=args.raw_data_path,
        processed_data_path=args.processed_data_path,
        S_out=args.S_out,
        Z_out=args.Z_out,
        Nt=args.Nt,
        fmax=args.fmax
    )
