import numpy

try:
    import pyrecon.projector_mpi as projector_mpi
except ImportError:
    print("MPI is not available. Using non-parallel version.")

import pyrecon.projector as projector


def reconstruct_mlem(
    m: numpy.ndarray, proj: numpy.ndarray, niter: int
) -> numpy.ndarray:
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
    sysmat = numpy.reshape(
        m, (int(numpy.prod(m.shape[0:-2])), int(m.shape[-1] * m.shape[-2]))
    )
    out = numpy.ones(sysmat.shape[1])
    proj = proj.flatten()
    for i in range(niter):
        # print(out.shape)
        out = projector.get_backward_projection_mlem(sysmat, out, proj)
    return out
