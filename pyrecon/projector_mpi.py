import numpy
import mpi4py

def backward_projection_mlem_mpi(proj: numpy.ndarray, m: numpy.ndarray) -> numpy.ndarray:
    """
    Perform backword projection with MLEM algorithm, with MPI.
    This function will back project the projection with the system matrix to image space.

    Parameters
    ----------
    proj: numpy.ndarray
    m: numpy.ndarray

    Returns
    -------
    out: numpy.ndarray
    """
    out = numpy.matmul(numpy.linalg.pinv(m), proj)
    return out
