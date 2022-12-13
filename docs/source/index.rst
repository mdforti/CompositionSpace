.. compositionspace documentation master file, created by
   sphinx-quickstart on Tue Dec 13 10:36:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CompositionSpace
================

CompositionSpace is a python library for analysis of TEM data.

Installation
~~~~~~~~~~~~

**Installation using `Conda <https://anaconda.org/>`_**

It is **strongly** recommended to install and use `calphy` within a conda environment. To see how you can install conda see `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_.

Once a conda distribution is available, the following steps will help set up an environment to use `compositionspace`. First step is to clone the repository.

``
https://github.com/Alaukiksaxena/CompositionSpaceNFDI.git
``

After cloning, an environment can be created from the included file-

``
cd CompositionSpaceNFDI
conda env create -f environment.yml
``

Activate the environment,

``
conda activate compspace
``

then, install `compositionspace` using,

``
python setup.py install
``
The environment is now set up to run calphy.

Examples
~~~~~~~~

.. toctree::
   :maxdepth: 2

   examples

API reference
~~~~~~~~~~~~~

.. toctree::
   compositionspace

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`   