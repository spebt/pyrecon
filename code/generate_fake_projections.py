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
    import h5py

    columns = [*Progress.get_default_columns(), TimeElapsedColumn()]
    progress = Progress(*columns)

    torch.device("cpu")
    data_dir = "data"
    flist = get_flist(os.path.join(data_dir, "srm_file_list.txt"))

    phantom_filename = "derenzo-phantom_512x512.npy"
    srm_keyword = "ppdfs"
    # load phantom data
    phantom = torch.tensor(
        np.load(os.path.join(data_dir, phantom_filename)), dtype=torch.float32
    ).view(-1)

    proj_data = torch.empty((0, 726), dtype=torch.float32)
    # load system matrix
    with progress:
        task = progress.add_task("Loading system matrix", total=len(flist))
        for fname in flist:
            progress.update(0, advance=1)

            with h5py.File(fname, "r") as h5f:
                m = torch.tensor(h5f[srm_keyword][:])
                projs_local = torch.matmul(m, phantom).view(1, -1)
                proj_data = torch.cat((proj_data, projs_local), dim=0)

            progress.update(task, advance=1)
    torch.save(proj_data, os.path.join(data_dir, "derenzo-projs.tensor"))
    # projs = torch.matmul(matrix, phantom)
    # np.save(os.path.join(data_dir, "derenzo-projs.npy"), projs.numpy())
