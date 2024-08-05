import numpy


def get_forward_projection(
    m: numpy.ndarray,
    img: numpy.ndarray,
) -> numpy.ndarray:
    """
    Get forward projection with matrix multiplication.

    Parameters
    ----------
    img: numpy.ndarray
    m: numpy.ndarray

    Returns
    -------
    out: numpy.ndarray
    """
    # sysmat = numpy.reshape(
    #     m, (int(numpy.prod(m.shape[0:-2])), m.shape[-1] * m.shape[-2])
    # )
    out = numpy.matmul(m, img.flatten())
    return out


def get_backward_projection(
    m: numpy.ndarray,
    prev: numpy.ndarray,
    proj: numpy.ndarray,
) -> numpy.ndarray:
    """
    Get Backward projection wiht MLEM algorithm. This is the non-MPI version.

    Parameters
    ----------
    prev: numpy.ndarray
    proj: numpy.ndarray
    m: numpy.ndarray

    Returns
    -------
    out: numpy.ndarray
    """

    prevproj = get_forward_projection(m, prev)
    quotient = proj / prevproj
    msum = numpy.sum(m, axis=0)
    if numpy.any(msum == 0):
        print("Error: Division by zero")
    out = numpy.matmul(quotient, m) / msum * prev
    return out


def get_backward_projection_mpi(proj: numpy.ndarray, m: numpy.ndarray) -> numpy.ndarray:
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
    try:
        from mpi4py import MPI
    except ImportError:
        print("MPI not found. Using non-MPI version\n")
        return get_backward_projection(proj, m)
    out = numpy.matmul(numpy.linalg.pinv(m), proj)
    return out   
