#! /bin/bash
#SBATCH --job-name=mpi-recon
#SBATCH --output=mpi-recon.out
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --mem-per-cpu=2G
#SBATCH --tasks-per-node=16

export I_MPI_FABRICS_LIST=tcp
export I_MPI_DEBUG=0
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
module load intel-mpi
# srun -n $SLURM_NPROCS python MLEM-mpi.py
srun -n 50 python MLEM-mpi.py