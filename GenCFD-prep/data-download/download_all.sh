#!/bin/bash
#SBATCH --job-name=download_HEMEW3D
#SBATCH --output=/cluster/home/lcarretero/workspace/rds/HEMEW3D/GenCFD-prep/data-download/download_all.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00

source $HOME/python_envs/rds-misc/bin/activate

cd $HOME/workspace/rds/HEMEW3D/GenCFD-prep/data-download

echo ":- Starting download job..."
echo ":- Python environment: $VIRTUAL_ENV"
echo ":- Current directory: $(pwd)"
echo ":- Running script: download_all.py"

python download_all.py --download --versions 1 2 --metadata-dirpath ./ --rawdata-dirpath /cluster/work/math/camlab-data/Wave_HemewS-3D

echo ":- Download job completed."
