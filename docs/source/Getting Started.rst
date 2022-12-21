.. part-x documentation master file, created by
   sphinx-quickstart on Wed Jan  5 07:23:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting Started
====================

Installation
------------
We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of `poetry <https://python-poetry.org/docs/#installation>`_ .

After you've installed poetry, you can install partx by running the following command in the root of the project:

.. code-block:: python

   poetry install 

Basic Usage
-----------

To run Part-X, we need a black-box function and initialize the parameters of Part-X. Once the black-box function the parameters are passed to the algorith, the algorithm outputs information including the best point, the statistics and much more. During the course of the algorithm, various files are generated to log the information from the optimizer, store various intermediate representations of the algorithms and files that speed up the process of result generation.
These ideas are explored in detail.

There are two ways to run Part-X algorithms.

1. Standalone Usage

   The idea here is to find 0-Level Set of a function. To run Part-X standalone, the inputs and the ouputs are decribed on the following pages.

   .. toctree::
      :maxdepth: 2

      Standalone Usage
   
   Examples in demos/Non-LinearBenchmarks. To run these:

      1) Rosenbrock function:

      .. code-block:: python

         poetry run python demos/Non-LinearBenchmarks/non_linear_rosenbrock.py

      2) Himmelblau function:

      .. code-block:: python

         poetry run python demos/Non-LinearBenchmarks/non_linear_himmelblaus.py

      3) Goldstein-Price function:

      .. code-block:: python

         poetry run python demos/Non-LinearBenchmarks/Goldstein_1.py

2. Usage with Psy-Taliro

The Part-X is used along with Psy-TaLiRo tool to generate falsifying points as well as provide probabilistic guarantees. Examples are as follows:
   .. toctree::
      :maxdepth: 2

      Part-X with PsyTaLiRo

   Examples in demos/ArchBenchmarks. To run these:

   1) Automatic Transmission Benchmark with the AT1 Specification:

      .. code-block:: python

         poetry run python demos/ArchBenchmarks/benchmarks AT1 -f "demo_partx_AT1"


