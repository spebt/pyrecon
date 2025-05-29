import numpy as np
import time


def get_matrix(**kwargs) -> np.ndarray:
    import h5py

    flist = kwargs["flist"]
    sfov = kwargs["sfov"]
    sproj = kwargs["sproj"]
    srm_keyword = kwargs.get("srm_keyword", "system matrix")
    m = np.empty((0, sproj, sfov), dtype=np.float32)
    for fname in flist:
        with h5py.File(fname, "r") as h5f:
            m = np.vstack((m, np.array(h5f[srm_keyword][:]).reshape(1, sproj, sfov)))

    return m


def get_flist(input_file: str) -> np.ndarray:
    with open(input_file, "r") as f:
        flist = f.readlines()
        flist = [f.strip() for f in flist]
        return np.array(flist)


if __name__ == "__main__":
    from rich.progress import Progress, TimeElapsedColumn
    import torch
    import os

    columns = [*Progress.get_default_columns(), TimeElapsedColumn()]
    progress = Progress(*columns)

    torch.device("cpu")
    data_dir = "data"
    flist = get_flist(os.path.join(data_dir, "srm_file_list.txt"))
    sfov = 512 * 512
    sproj = 726
    prev_estimate = np.full((sfov), 1.0)

    # load projection data
    matrix = torch.tensor(
        get_matrix(flist=flist, sfov=sfov, sproj=sproj, srm_keyword="ppdfs"),
    ).view((-1, sproj, sfov))

    phantom_filename = "derenzo-phantom_512x512.npy"

    # load phantom data
    phantom = torch.tensor(
        np.load(os.path.join(data_dir, phantom_filename)), dtype=torch.float32
    ).view(-1)
    projs = torch.matmul(matrix, phantom)
    np.save(os.path.join(data_dir, "derenzo-projs.npy"), projs.numpy())
