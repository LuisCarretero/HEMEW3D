#!/bin/bash
#SBATCH --job-name=preprocess_HEMEW3D
#SBATCH --output=/cluster/home/lcarretero/workspace/rds/HEMEW3D/GenCFD-prep/data-preprocessing/preprocess.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00

# NB: preprocess.py bottlenecks are currently single-threaded anyways so no need to increase cpu count

source $HOME/python_envs/rds-misc/bin/activate

cd $HOME/workspace/rds/HEMEW3D/GenCFD-prep/data-preprocessing

echo ":- Starting preprocessing job..."
echo ":- Python environment: $VIRTUAL_ENV"
echo ":- Current directory: $(pwd)"
echo ":- Running script: preprocess.py"

python preprocess.py --S_out 32 --Nt 64 --Z_out 64 --f 10 --fmax 5 --max_files 99999

echo ":- Preprocessing job completed."
