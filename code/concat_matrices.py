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
        return np.array(flist)



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

    # load projection data
    pdata = np.load("../data/" + "projection_data.npy").reshape(-1)
    matrix = get_matrix(flist=flist, sfov=sfov, sproj=sproj,progress=progress).reshape(-1, sfov)
    print(matrix.shape)
    np.savez_compressed("../data/matrix.npz", matrix=matrix)
