import numpy
try:
    from .proj_mpi import *
except ImportError:
    print("MPI is not available. Using non-parallel version.")
from .proj import *

def reconstruct_mlem(proj: numpy.ndarray, m: numpy.ndarray, niter: int) -> numpy.ndarray:
    """
    Perform reconstruction with MLEM algorithm. This is the non-parallel version.
    This function will reconstruct the image with the projection and the system matrix.

    Parameters
    ----------
    proj: numpy.ndarray
    m: numpy.ndarray
    niter: int

    Returns
    -------
    out: numpy.ndarray
    """

    out = numpy.ones(m.shape[1])
    for i in range(niter):
        out = backward_projection_mlem(out, proj, m)
    return out