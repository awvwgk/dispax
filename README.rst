DFT-D dispersion for Jax M.D.
=============================

Implementation of the DFT-D3 dispersion model in Jax.
This module allows to calculate dispersion energies for arbitrary boundary conditions using Jax M.D.'s space transformation map.

For details on the D3 dispersion model see

- *J. Chem. Phys.*, **2010**, *132*, 154104 (`DOI <https://dx.doi.org/10.1063/1.3382344>`__)
- *J. Comput. Chem.*, **2011**, *32*, 1456 (`DOI <https://dx.doi.org/10.1002/jcc.21759>`__)

For alternative implementations also check out

`simple-dftd3 <https://dftd3.readthedocs.io>`__:
  Simple reimplementation of the DFT-D3 dispersion model in Fortran with Python bindings

`dftd4 <https://dftd4.readthedocs.io>`__:
  Generally applicable charge-dependent dispersion model in Fortran with Python bindings

`torch-dftd <https://tech.preferred.jp/en/blog/oss-pytorch-dftd3/>`__:
  PyTorch implementation of DFT-D2 and DFT-D3

`tad-dftd3 <https://tad-dftd3.readthedocs.io/>`__:
  Implementation of the DFT-D3 dispersion model in PyTorch


Installation
------------

This project is hosted on GitHub at `awvwgk/dispax <https://github.com/awvwgk/dispax>`__.
Obtain the source by cloning the repository with

.. code::

   git clone https://github.com/awvwgk/dispax
   cd dispax

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

   mamba env create -n jax -f environment.yml
   mamba activate jax

Install this project with pip in the environment

.. code::

   pip install .

Add the option ``-e`` for installing in development mode.

The following dependencies are required

- `numpy <https://numpy.org/>`__
- `jax <https://jax.readthedocs.io/>`__
- `jax-md <https://github.com/google/jax-md>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)

You can check your installation by running the test suite with

.. code::

   pytest tests/ --pyargs dispax --doctest-modules


Contributing
------------

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the `contributing guidelines <CONTRIBUTING.md>`__.


License
-------

Licensed under the Apache License, Version 2.0 (the “License”);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an *“as is” basis*,
*without warranties or conditions of any kind*, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project by you, as defined in the
Apache-2.0 license, shall be licensed as above, without any additional
terms or conditions.
