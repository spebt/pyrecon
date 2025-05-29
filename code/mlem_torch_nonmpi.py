"""
This script is a non-mpi version of the MLEM algorithm. 
It reads the projection data and system matrix from the 
disk and performs the reconstruction. The script is used 
to compare the performance of the MPI version of the MLEM a
lgorithm.
"""

import numpy as np


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
        return np.array(flist)


def run_one_iteration_mlem(
    estimate_prev,
    m,
    pdata,
):
    import torch

    # forward-projection from a single file, 864 projections
    y = torch.matmul(m, estimate_prev)
    # get the corresponding projection data
    p = pdata
    # compute the ratio
    r = p / y
    # compute the back-projection
    b = torch.matmul(m.transpose(0, 1), r)
    # compute the matrix sum
    m_sum = torch.sum(m, dim=0)

    x = estimate_prev * b / m_sum
    logp = torch.sum(p * torch.log(torch.matmul(m, x)) - torch.matmul(m, x))
    return x, logp


if __name__ == "__main__":
    from rich.progress import Progress, TimeElapsedColumn
    import torch
    import time

    columns = [*Progress.get_default_columns(), TimeElapsedColumn()]
    progress = Progress(*columns)
    data_dir = "../data/"
    flist = get_flist(data_dir + "dataset_flist.cvs")
    sfov = 128 * 128
    sproj = 864
    torch.device("cpu")

    # load projection data
    pdata = torch.tensor(np.load(data_dir + "derenzo-projs.npy"))
    projs = pdata.view(-1)
    # flist = flist.reshape(n_subsets, -1)

    matrix = torch.tensor(
        get_matrix(flist=flist, sfov=sfov, sproj=sproj, progress=progress)
    ).view((-1, sfov))
    output = []
    times = []
    log_p = []
    with progress:
        task = progress.add_task("MLEM", total=1000)
        estimate = torch.ones((sfov), dtype=torch.float32)
        prev_logp = -1000000
        while not progress.finished:
            start_time = time.time()
            estimate, logp = run_one_iteration_mlem(estimate, matrix, projs)
            diff_logp = logp - prev_logp
            prev_logp = logp
            end_time = time.time()
            time_taken = end_time - start_time
            output.append(estimate)
            times.append(time_taken)
            log_p.append(logp)
            if diff_logp < 0.001:
                progress.update(task, completed=1000)
                break
            progress.update(task, advance=1)
    output = np.array(output)
    times = np.array(times)
    np.save(
        data_dir + "recon-mlem-torch-derenzo.npy",
        {"estimates": output, "times": times, "log likelihood": log_p},
    )
