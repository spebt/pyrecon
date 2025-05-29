if __name__ == "__main__":
    import os
    import numpy as np
    home_path = os.path.expanduser("~")
    topdir = os.path.join(
        home_path, "Work/spebt/data/enhance-sampling-20250205"
    )
    file_idxs = np.arange(0, 24) * 25 + 13
    fnames = ["system_matrix_{:03d}.hdf5".format(i) for i in file_idxs]
    with open("../data/dataset_flist.cvs", "w") as f:
        for fname in fnames:
            f.write(topdir + "/" + fname + "\n")