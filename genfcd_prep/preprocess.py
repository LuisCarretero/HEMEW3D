from preprocessing_utils import process_data

if __name__ == '__main__':

    process_data(
        raw_data_path=r'/cluster/work/math/camlab-data/Wave_HemewS-3D/version1',
        processed_data_path=r'/cluster/work/math/camlab-data/Wave_HemewS-3D/processed/version1',
        S_out=32,
        Nt=128,
        f=20,
        fmax=5,
        max_files=9999
    )
