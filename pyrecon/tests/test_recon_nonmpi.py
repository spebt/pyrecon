import numpy as np
import pathlib, sys

# import matplotlib.pyplot as plt
import h5py

top_dir = str(pathlib.Path(__file__).parents[2])
sys.path.append(top_dir)
import pyrecon.reconstruct_mlem as reconstruct_mlem

if __name__ == "__main__":
    # Load system matrix
    with h5py.File(top_dir + "/data/" + "test_sysmat.hdf5", "r") as f:
        data = f['sysmat']
        # Load projection data
        proj = np.load(top_dir + "/data/hotrod_phantom_data_180x180_projection.npz")[
            "projection"
        ]
        # Perform reconstruction
        out = reconstruct_mlem.reconstruct_mlem(data, proj, 10)
    # Save the reconstructed image
    # np.savez(top_dir / "data" / "recon.npz", recon=out)
