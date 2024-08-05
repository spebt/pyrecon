import numpy
import pathlib, sys
import h5py

top_dir = str(pathlib.Path(__file__).parents[2])
sys.path.append(top_dir)
import pyrecon.mlem as mlem

phantom = numpy.load(str(top_dir) + "/data/" + "hotrod_phantom_data_180x180.npz")[
    "phantom"
]
with h5py.File(top_dir + "/data/test_sysmat.hdf5", "r") as f:
    data = f["sysmat"]
    sysmat = numpy.reshape(
        data, (int(numpy.prod(data.shape[0:-2])), int(data.shape[-1] * data.shape[-2]))
    )
    projection = mlem.get_forward_projection(sysmat, phantom)
    projection = numpy.reshape(projection, data.shape[:-2])
    numpy.savez_compressed(
        str(top_dir) + "/data/" + "hotrod_phantom_data_180x180_projection.npz",
        projection=projection,
    )
