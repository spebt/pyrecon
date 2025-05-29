import numpy.typing as npt
from mlem_nonmpi import get_matrix, get_flist
import numpy as np
import time


def run_one_subiteration_osem(**kwargs) -> npt.NDArray[np.float32]:
    import multiprocessing as mp
    estimate_prev = kwargs["estimate_prev"]
    matrices = kwargs["matrices"]
    projs = kwargs["projs"]
    ids = kwargs["ids"]
    sfov = estimate_prev.shape[0]
    m = matrices[ids].reshape(-1, sfov)
    # forward-projection
    y = np.matmul(m, estimate_prev)
    # get the corresponding projection data
    p = projs[ids].reshape(-1)
    # compute the ratio
    r = p / y
    # compute the back-projection
    b = np.matmul(m.T, r)
    # print(b.shape)
    # compute the matrix sum
    m_sum = np.sum(m, axis=0)

    return estimate_prev * b / m_sum


if __name__ == "__main__":

    from rich.progress import Progress, TimeElapsedColumn

    columns = [*Progress.get_default_columns(), TimeElapsedColumn()]
    progress = Progress(*columns)

    flist = get_flist("../data/" + "dataset_flist.cvs")
    sfov = 128 * 128
    sproj = 864

    output = np.empty((0, sfov), dtype=np.float32)
    times = np.empty((0), dtype=np.float32)

    # load projection data
    pdata = np.load("../data/" + "projection_data.npy")

    # matrix = get_matrix(flist, sfov, sproj).reshape(-1, sfov)
    n_subsets = 12
    n_iterations = 10
    projs = pdata.reshape(n_subsets, -1, sproj)
    # flist = flist.reshape(n_subsets, -1)

    matrices = get_matrix(
        flist=flist, sfov=sfov, sproj=sproj, progress=progress
    ).reshape(n_subsets, -1, sproj, sfov)
   
    with progress:
        estimate = np.full((sfov), 1.0)
        task1 = progress.add_task("OSEM Reconstruction", total=10)
        task2 = progress.add_task("Subsets", total=12)
        for i in range(n_iterations):
            progress.update(task2, completed=0)
            start_time = time.time()
            for j in range(n_subsets):
                estimate = run_one_subiteration_osem(
                    estimate_prev=estimate, matrices=matrices, projs=projs, ids=j
                )

                progress.update(task2, advance=1)
            output = np.vstack((output, estimate))
            end_time = time.time()
            time_taken = end_time - start_time
            times = np.append(times, time_taken)
            progress.update(task1, advance=1)

    np.save("recon-osem.npy", {"estimates": output, "times": times})
