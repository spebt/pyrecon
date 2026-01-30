#!/bin/bash

#SBATCH --job-name=mlem_full_pipeline
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --time=01:00:00                 # Total time for flist + proj + mlem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G                       # 32GB for matrix cache + 4GB overhead
#SBATCH --mail-user=smehta28@buffalo.edu
#SBATCH --mail-type=FAIL,END

# --- Logging Setup ---
mkdir -p slurm_logs/out slurm_logs/err
#SBATCH --output=slurm_logs/out/recon_pipe_%j.out
#SBATCH --error=slurm_logs/err/recon_pipe_%j.err

# --- Environment Setup ---
echo "=========================================================="
echo "Start Time: $(date)"
echo "Running on: $(hostname)"
echo "Config:     configs/base_config.yml"
echo "=========================================================="

source ../venv/bin/activate

# --- STEP 1: Generate File List ---
echo "Step 1/3: Generating Dataset File List..."
python generate_flist.py --config configs/base_config.yml
if [ $? -ne 0 ]; then echo "Step 1 Failed"; exit 1; fi

# --- STEP 2: Forward Projection ---
echo "Step 2/3: Running Forward Projection (Generating Phantoms)..."
python fake_projection.py --config configs/base_config.yml
if [ $? -ne 0 ]; then echo "Step 2 Failed"; exit 1; fi

# --- STEP 3: MLEM Reconstruction ---
echo "Step 3/3: Running MLEM Reconstruction..."
python mlem_torch_nonmpi.py --config configs/base_config.yml
if [ $? -ne 0 ]; then echo "Step 3 Failed"; exit 1; fi

# --- Optional Step 4: Visualization ---
echo "Step 4: Generating Final Plots..."
python view_npz.py --config configs/base_config.yml

echo "=========================================================="
echo "End Time:   $(date)"
echo "Pipeline Finished Successfully"
echo "=========================================================="