"""
This script is a non-mpi version of the MLEM algorithm. 
It reads the projection data and system matrix from the 
disk and performs the reconstruction. The script is used 
to compare the performance of the MPI version of the MLEM a
lgorithm.
"""

import numpy as np
import time


def get_matrix(**kwargs) -> np.ndarray:
    import h5py
    flist = kwargs["flist"]
    sfov = kwargs["sfov"]
    sproj = kwargs["sproj"]
    m = np.empty((0, sproj, sfov), dtype=np.float32)
    if "progress" in kwargs:
        progress = kwargs["progress"]
        with progress:
            task = progress.add_task("Loading system matrices", total=len(flist))
            for fname in flist:
                progress.update(task, description=fname)
                with h5py.File(fname, "r") as h5f:
                    m = np.vstack(
                        (m, np.array(h5f["system matrix"][:]).reshape(1, sproj, sfov))
                    )
                progress.update(task, advance=1)
            progress.remove_task(task)
    else:
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                m = np.vstack(
                    (m, np.array(h5f["system matrix"][:]).reshape(1, sproj, sfov))
                )
    return m
    

   



def get_flist(input_file: str) -> np.ndarray:
    with open(input_file, "r") as f:
        flist = f.readlines()
        flist = [f.strip() for f in flist]
        return flist


def run_one_iteration_mlem(
    prev_estimate,
    m,
    pdata,
):
    import numpy as np

    # forward-projection from a single file, 864 projections
    y = m @ prev_estimate
    # get the corresponding projection data
    p = pdata
    # compute the ratio
    r = p / y
    # compute the back-projection
    b = m.T @ r
    # compute the matrix sum
    m_sum = np.sum(m, axis=0)

    return prev_estimate * b / m_sum


if __name__ == "__main__":
    from rich.progress import Progress, TimeElapsedColumn

    columns = [*Progress.get_default_columns(), TimeElapsedColumn()]
    progress = Progress(*columns)

    flist = get_flist("../data/" + "dataset_flist.cvs")
    sfov = 128 * 128
    sproj = 864
    prev_estimate = np.full((sfov), 1.0)
    output = np.empty((0, sfov), dtype=np.float32)
    times = np.empty((0), dtype=np.float32)

    print(flist.shape)

    # load projection data
    pdata = np.load("../data/" + "projection_data.npy").reshape(-1)
    matrix = get_matrix(flist=flist, sfov=sfov, sproj=sproj,progress=progress).reshape(-1, sfov)

    
    with progress:
        task = progress.add_task("MLEM", total=10)
        for i in range(10):
            start_time = time.time()
            prev_estimate = run_one_iteration_mlem(prev_estimate, matrix, pdata)
            output = np.vstack((output, prev_estimate))
            end_time = time.time()
            time_taken = end_time - start_time
            times = np.append(times, time_taken)
            progress.update(task, advance=1)

    np.save("../data/recon-mlem.npy", {"estimates": output, "times": times})
