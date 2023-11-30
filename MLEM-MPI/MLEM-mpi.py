from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
sendbuf = None
if rank == 0:
    size = comm.Get_size()
    sendbuf = np.empty([size, 100], dtype='i')
    sendbuf.T[:,:] = range(size)

recvbuf = np.empty(100, dtype='i')
comm.Scatter(sendbuf, recvbuf, root=0)
assert np.allclose(recvbuf, rank)

print("Rank: %d, Sum=%d"%(rank,np.sum(recvbuf)))
