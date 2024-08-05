# Quick Start

## On your desktop/laptop without paralell HDF5 library

### Install Dependencies

#### With `pipenv`
This is supposed to be the simpliest. Run following command:
```sh
pipenv install
pipenv shell
```
### Run the test scripts

1. Create a fake system matrix and store in hdf5 format.
```sh
python dev/generate_sysmat_hdf5.py
```
2. Create the projection of the hotrod(derenzo) phantom.
```sh
python pyrecon/tests/test_projection_nonmpi.py
```
3. Reconstruct from the newly-created projection.
```sh
python pyrecon/tests/test_recon_nonmpi.py
```

### Visulization Jupyter notebooks
There are three [Jupyter notebooks](https://docs.jupyter.org/en/latest/) in `/dev` folder.

- `read_sysmat_npz.ipynb`
  plots the system matrix

- `read_projection_npz.ipynb`
  plots the projection as a sinogram

- `read_reconstructed_npz.ipynb`
  plots the reconstructed image