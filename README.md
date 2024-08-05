# SPEBT Pyrecon

Image reconstruction package for SPEBT project.

Features:

- Numpy-based MLEM reconstruction algorithm
- Independent MPI-based reconstrucion APIs and non-MPI-based APIs
- Example running scripts in `pyrecon/tests` folder

## Installation

### Install the dependencies

On you desktop/laptop without MPI, you can install by:
```sh
pip install -r requirements.txt
```


## Usage

Here's a simple example of how to use Pyrecon:

```python
import numpy
import pathlib, sys
import h5py

top_dir = str(pathlib.Path(__file__).parents[2])
sys.path.append(top_dir)
import pyrecon.reconstruct_mlem as reconstruct_mlem

if __name__ == "__main__":
    # Load system matrix
    with h5py.File(top_dir + "/data/" + "test_sysmat.hdf5", "r") as f:
        data = f['sysmat']
        # Load projection data
        proj = numpy.load(top_dir + "/data/hotrod_phantom_data_180x180_projection.npz")[
            "projection"
        ]
        # Perform reconstruction
        out = reconstruct_mlem.reconstruct_mlem(data, proj, 10)
        numpy.savez_compressed(
            top_dir + "/data/" + "hotrod_phantom_data_180x180_reconstruction.npz",
            reconstructed=out,
        )
```

## Documentation

For detailed documentation and examples, please refer to the [Pyrecon Documentation](https://spebt.github.io/pyrecon).

## License

Pyrecon is licensed under the [MIT License](https://opensource.org/licenses/MIT).
