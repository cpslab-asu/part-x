.. part-x documentation master file, created by
   sphinx-quickstart on Wed Jan  5 07:23:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Part-X!
====================

Introduction
------------
Requirements driven search-based testing (also known as falsification) has proven to be a practical and effective method for discovering erroneous behaviors in Cyber-Physical Systems. Despite the constant improvements on the performance and applicability of falsification methods, they all share a common characteristic. Namely, they are best-effort methods which do not provide any guarantees on the absence of erroneous behaviors (falsifiers) when the testing budget is exhausted. The absence of finite time guarantees is a major limitation which prevents falsification methods from being utilized in certification procedures. In this paper, we address the finite-time guarantees problem by developing a new stochastic algorithm. Our proposed algorithm not only estimates (bounds) the probability that falsifying behaviors exist, but also it identifies the regions where these falsifying behaviors may occur. We demonstrate the applicability of our approach on standard benchmark functions from the optimization literature and on the F16 benchmark problem.

Link to Paper
-------------
The working paper can be accessed `here <https://arxiv.org/abs/2110.10729>`_ .

Citation
--------
Please cite the following papers if you use the work in your research

.. code-block:: bib

   @misc{pedrielli2021partx,
      title={Part-X: A Family of Stochastic Algorithms for Search-Based Test Generation with Probabilistic Guarantees}, 
      author={Giulia Pedrielli and Tanmay Khandait and Surdeep Chotaliya and Quinn Thibeault and Hao Huang and Mauricio Castillo-Effen and Georgios Fainekos},
      year={2021},
      eprint={2110.10729},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      }

.. code-block:: bib

   @inproceedings{10.1145/3477244.3477984,
      author = {Cao, Yumeng and Thibeault, Quinn and Chandratre, Aniruddh and Fainekos, Georgios and Pedrielli, Giulia and Castillo-Effen, Mauricio},
      title = {Towards Assurance Case Evidence Generation through Search Based Testing: Work-in-Progress},
      year = {2021},
      isbn = {9781450387125},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3477244.3477984},
      doi = {10.1145/3477244.3477984},
      abstract = {Requirements-driven search-based testing (SBT), also known as falsification, has proven to be a practical and effective method for discovering erroneous behaviors in Cyber-Physical Systems. However, SBT techniques do not provide guarantees on correctness if no falsifying behavior is found within the test budget. Hence, the applicability of SBT methods for evidence generation supporting assurance cases is limited. In this work, we make progress towards developing finite-time guarantees for SBT techniques with associated confidence metrics. We demonstrate the applicability of our approach to the F16 GCAS benchmark challenge.},
      booktitle = {Proceedings of the 2021 International Conference on Embedded Software},
      pages = {41–42},
      numpages = {2},
      location = {Virtual Event},
      series = {EMSOFT '21}
      }

Usage
-----
.. toctree::
   :maxdepth: 2

   Getting Started
   Standalone Usage
   Part-X with PsyTaLiRo
   Outputs
   Demo 1 - Standalone Part-X
   Demo 2 - Part-X with Psy-TaLiRo