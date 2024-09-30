import numpy
from ._projector import get_backward_projection
from rich.progress import track

def reconstruct(
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
    sysmat = m
    out = numpy.ones(sysmat.shape[1])
    proj = proj.flatten()
    for i in track(range(niter), description="Reconstructing"):
        # print(out.shape)
        out = get_backward_projection(sysmat, out, proj)
    return out