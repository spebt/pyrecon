#!/bin/bash

# load modules
module load python/anaconda
module load intel-mpi
ulimit -s unlimited

# setup env variables
export PYTHONPATH=/util/academic/python/mpi4py/v2.0.0/lib/python2.7/site-packages:$PYTHONPATH
export I_MPI_FABRICS_LIST=tcp
export I_MPI_DEBUG=4
export I_MPI_PMI_LIBRARY=/usr/lib64/libmpi.so