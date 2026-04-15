#!/bin/bash
#SBATCH --job-name=osnn_real_30runs_fast
#SBATCH --output=Logs/osnn_real_%A_%a.log
#SBATCH --error=Logs/osnn_real_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-29
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hxa568@student.bham.ac.uk

module purge
module load bluebear
module load bear-apps/2022b
module load GCCcore/12.2.0
module load Python/3.10.8-GCCcore-12.2.0

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR
source /rds/projects/m/minkull-spdisc-hadi/PythonCode/myenv/bin/activate

TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "Running OSNN real-world run ${TASK_ID} / 30"

python ossn_realworld_30runs_fast.py \
  --task_id ${TASK_ID} \
  --out_dir results_runs_osnn
