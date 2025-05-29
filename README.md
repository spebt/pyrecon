# PyRecon

PyRecon is a Python-based library for image reconstruction algorithms, designed for medical imaging and computational tomography applications. It includes implementations of various reconstruction techniques such as MLEM, OSEM, and tools for generating projections and matrices.

## Features

- Matrix concatenation utilities.
- Downsampling and SRM processing.
- Fake projection generation.
- MLEM and OSEM reconstruction algorithms (non-MPI and PyTorch versions).

> [!NOTE]
> Do not use the phantom generation scripts in the `code/` directory as they are not functional. Instead, create phantom using the `digital_phantom` module, which is hosted in a separate repository.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/spebt/pyrecon.git
   ```

2. Navigate to the project directory:

   ```bash
   cd pyrecon
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## OSEM reconstruction

Look specifically at the `osem_torch_nonmpi.py`

It will not be trivial to run it. As the data preparation is not automatically done.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.