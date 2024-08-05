import numpy
import pathlib, sys
import h5py

top_dir = str(pathlib.Path(__file__).parents[2])
sys.path.append(top_dir)
import pyrecon.mlem as mlem
if __name__ == "__main__":
    # Get number of iterations
   
    niter = input("Number of iterations:")
    try:
        niter = int(niter)
    except ValueError:
        print("Invalid input")
        sys.exit(1)

    # Load system matrix
    with h5py.File(top_dir + "/data/" + "test_sysmat.hdf5", "r") as f:
        data = f['sysmat']
        # Load projection data
        proj = numpy.load(top_dir + "/data/hotrod_phantom_data_180x180_projection.npz")[
            "projection"
        ]
        # Perform reconstruction
        out = mlem.reconstruct(data, proj, niter)
        numpy.savez_compressed(
            top_dir + "/data/" + "hotrod_phantom_data_180x180_reconstructed.npz",
            reconstructed=out,
        )