#!/bin/bash
#SBATCH --job-name=mlem_parallel
#SBATCH --output=logs/osem_%A_%a.out
#SBATCH --error=logs/osem_%A_%a.err
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tridevme@buffalo.edu

module load gcc/11.2.0 openmpi/4.1.1


python parallel_mlem_subset.py ${SLURM_ARRAY_TASK_ID}
