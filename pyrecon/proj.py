import numpy

def forward_projection(img: numpy.ndarray, m: numpy.ndarray) -> numpy.ndarray:
    """
    Perform forword projection.
    This function will forward project the image with the system matrix.

    Parameters
    ----------
    img: numpy.ndarray
    m: numpy.ndarray

    Returns
    -------
    out: numpy.ndarray
    """

    out = numpy.matmul(m, img)
    return out


def backwardProj(lastArr, projArr, sysMat):
    forwardLast = np.matmul(sysMat, lastArr)
    quotients = projArr / forwardLast
    return np.matmul(quotients, sysMat) / np.sum(sysMat, axis=0) * lastArr


def backward_projection_mlem(
    prev: numpy.ndarray, proj: numpy.ndarray, m: numpy.ndarray
) -> numpy.ndarray:
    """
    Perform backword projection with MLEM algorithm.
    This function will back project the projection with the system matrix to image space.

    Parameters
    ----------
    prev: numpy.ndarray
    proj: numpy.ndarray
    m: numpy.ndarray

    Returns
    -------
    out: numpy.ndarray
    """

    prevproj = forward_projection(prev, m)
    quotient = proj / prevproj
    out = numpy.matmul(quotient, m) / numpy.sum(m, axis=0) * prev
    return out
