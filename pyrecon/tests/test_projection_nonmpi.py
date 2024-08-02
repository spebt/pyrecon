import numpy as np
import pathlib, sys

# import matplotlib.pyplot as plt
import h5py

top_dir = str(pathlib.Path(__file__).parents[2])
sys.path.append(top_dir)
import pyrecon.projector as projector


phantom = np.load(str(top_dir) + "/data/" + "hotrod_phantom_data_180x180.npz")[
    "phantom"
]
with h5py.File(top_dir + "/data/test_sysmat.hdf5", "r") as f:
    data = f["sysmat"]
    projection = projector.get_forward_projection(data,phantom)
    projection = np.reshape(projection, data.shape[:-2])
    np.savez_compressed(
        str(top_dir) + "/data/" + "hotrod_phantom_data_180x180_projection.npz",
        projection=projection,
    )
