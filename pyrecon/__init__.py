"""
PyRecon
=======
Provides
    1. Forward projection
    2. Backward projection
    3. Image reconstruction

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a sphinx-generated HTML website 
`spebt pyrecon <https://spebt.github.io/pyrecon>`_.
"""

import pyrecon.projector as projector
import pyrecon.reconstruct_mlem as reconstruct_mlem
__version__ = '0.0.1'
__all__ = ['projector', 'reconstruct_mlem']