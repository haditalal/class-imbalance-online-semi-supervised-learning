#!/bin/bash
#SBATCH --job-name=ossn_runs_rest
#SBATCH --output=ossn_runs_rest_%A_%a.log
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-30
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hxa568@student.bham.ac.uk

module purge
module load bluebear
module load bear-apps/2022b
module load GCCcore/12.2.0
module load Python/3.10.8-GCCcore-12.2.0

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR
source myenv/bin/activate

python runs30_eval.py ${SLURM_ARRAY_TASK_ID}