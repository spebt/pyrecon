import numpy.typing as npt
from mlem_nonmpi import get_matrix, get_flist

import numpy as np
import torch
import time
from typing import List
import h5py

def run_one_subiteration_osem(**kwargs) -> torch.Tensor:
    estimate_prev = kwargs["estimate_prev"]
    srm = kwargs["srm"]
    projs = kwargs["projs"]
    ids = kwargs["ids"]
    sfov = estimate_prev.shape[0]
    m = srm.reshape(-1, sfov)
    # forward-projection
    y = torch.matmul(m, estimate_prev)
    # get the corresponding projection data
    p = projs[ids].view(-1)
    # compute the ratio
    r = p / y
    # compute the back-projection
    b = torch.matmul(m.T, r)
    # print(b.shape)
    # compute the matrix sum
    m_sum = torch.sum(m, dim=0)

    x = estimate_prev * b / m_sum
    logp = torch.sum(p * torch.log(torch.matmul(m, x)) - torch.matmul(m, x))
    return x, logp


def load_srm_subset(
    flist: npt.NDArray,
    sfov: int,
    sproj: int,
    srm_keyword: str = "ppdfs",
) -> torch.Tensor:
    """
    Load the system response matrix from the file list.
    """
    matrices = torch.empty((0, sproj, sfov), dtype=torch.float32)
    # load system matrix
    for fname in flist:
        with h5py.File(fname, "r") as h5f:
            m = np.array(torch.tensor(h5f[srm_keyword][:]))
            m = torch.tensor(m).unsqueeze(0)
            matrices = torch.cat((matrices, m), dim=0)
    return matrices

def get_subset_flist(
    flist: List,
    n_subsets: int,
) -> npt.NDArray:
    """
    Get the subset file list.
    """
    subset_flist= []
    n_f_per_subset = len(flist) // n_subsets
    for i in range(n_subsets):
        subset_flist.append(
            flist[i * n_f_per_subset : (i + 1) * n_f_per_subset]
        )
    return subset_flist


if __name__ == "__main__":

    from rich.progress import Progress, TimeElapsedColumn
       
    import os

    columns = [*Progress.get_default_columns(), TimeElapsedColumn()]
    progress = Progress(*columns)
    data_dir = "data"

    flist = get_flist(os.path.join(data_dir, "srm_file_list.txt"))
    sfov = 512 * 512
    sproj = 726

    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.device("cpu")

    # load projection data
    projs_data = torch.load(
        os.path.join(data_dir, "derenzo-projs.tensor"), weights_only=True
    )
    # matrix = get_matrix(flist, sfov, sproj).reshape(-1, sfov)
    n_subsets = 12
    n_iterations = 10
    projs = projs_data.view(n_subsets, -1, sproj)
    # flist = flist.reshape(n_subsets, -1)

    subset_flist = get_subset_flist(flist, n_subsets)
    # print("Subset file list [0]: ", subset_flist[0])


    output = []
    log_p = []
    times = []
    with progress:
        estimate = torch.ones(sfov, dtype=torch.float32)
        task1 = progress.add_task("OSEM Reconstruction", total=150)
        task2 = progress.add_task("Subsets", total=12)
        prev_logp = -1000
        diff_logp = 1
        while not progress.finished:
            progress.update(task2, completed=0)
            start_time = time.time()
            for j in range(n_subsets):
                srm = load_srm_subset(
                    flist=subset_flist[j], sfov=sfov, sproj=sproj
                )
                estimate, logp = run_one_subiteration_osem(
                    estimate_prev=estimate, srm=srm, projs=projs, ids=j
                )
                diff_logp = torch.abs((logp - prev_logp) / prev_logp)
                prev_logp = logp
                del srm
                progress.update(task2, advance=1)
            end_time = time.time()
            time_taken = end_time - start_time
            times.append(time_taken)
            log_p.append(logp)
            output.append(estimate.numpy())
            # if diff_logp < 0.001:
            #     progress.update(task1, completed=1000)
            #     progress.update(task2, completed=12)
            #     print("Converged: ", diff_logp)
            #     break
            progress.update(task1, advance=1)
    output = np.array(output)
    times = np.array(times)

    np.save(
        data_dir + "recon-osem-torch-derenzo.npy",
        {"estimates": output, "times": times, "log likelihood": log_p},
    )
