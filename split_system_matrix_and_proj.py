import h5py
import numpy as np
import os

output_dir = ""                    #replace with your any output path
os.makedirs(output_dir, exist_ok=True)

with h5py.File("", "r") as f:          #replace with your path for system matrix
    sysmat = f["sysmat"]
    projection_shape = (144,)
    for i in range(6):
        partial_sysmat = sysmat[:, :, i, :, :]
        reshaped = partial_sysmat.reshape((144, -1))
        np.savez_compressed(f"{output_dir}/sysmat_subset_{i}.npz", sysmat=reshaped)

        proj_path = f""                                   #replace with your path for projection data
        projection = np.load(proj_path)["projection"].reshape(projection_shape)
        np.savez_compressed(f"{output_dir}/proj_subset_{i}.npz", projection=projection)
