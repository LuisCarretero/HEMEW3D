#!/bin/bash
#SBATCH --job-name=preprocess_HEMEW3D
#SBATCH --output=/cluster/home/lcarretero/workspace/rds/HEMEW3D/genfcd_prep/preprocess.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# NB: preprocess.py bottlenecks are currently single-threaded anyways so no need to increase cpu count

source $HOME/python_envs/rds-misc/bin/activate

cd $HOME/workspace/rds/HEMEW3D/genfcd_prep

echo ":- Starting preprocessing job..."
echo ":- Python environment: $VIRTUAL_ENV"
echo ":- Current directory: $(pwd)"
echo ":- Running script: preprocess.py"

python preprocess.py

echo ":- Preprocessing job completed."
