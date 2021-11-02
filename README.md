# Part-X: A Family of Stochastic Algorithms for Search-Based Test Generation with Probabilistic Guarantees

Requirements driven search-based testing (also known as falsification) has proven to be a practical and effective method for discovering erroneous behaviors in Cyber-Physical Systems. Despite the constant improvements on the performance and applicability of falsification methods, they all share a common characteristic. Namely, they are best-effort methods which do not provide any guarantees on the absence of erroneous behaviors (falsifiers) when the testing budget is exhausted. The absence of finite time guarantees is a major limitation which prevents falsification methods from being utilized in certification procedures. In this paper, we address the finite-time guarantees problem by developing a new stochastic algorithm. Our proposed algorithm not only estimates (bounds) the probability that falsifying behaviors exist, but also it identifies the regions where these falsifying behaviors may occur. We demonstrate the applicability of our approach on standard benchmark functions from the optimization literature and on the F16 benchmark problem. 


## Link to Paper
The working paper can be accessed [here](https://arxiv.org/abs/2110.10729#).

## Installation
We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

To install the package:
```bash
cd part_x
poetry update
```

To run the non-linear optimization benchmarks, download the following github repo - [Bookish Umbrella](https://github.com/DaitTan/bookish-umbrella)

It is important to download both the repositories in a single folder.
To run these non-linear benchmarks:
```bash
cd bookish_umbrella
poetry update
poetry run python BENCHMARK_NAME.py

# To run driver_goldstein_1 benchmark:
poetry run python driver_goldstein_1.py
```

## Citation
Please cite the following paper if you use the work in your research.
```
@misc{pedrielli2021partx,
      title={Part-X: A Family of Stochastic Algorithms for Search-Based Test Generation with Probabilistic Guarantees}, 
      author={Giulia Pedrielli and Tanmay Khandait and Surdeep Chotaliya and Quinn Thibeault and Hao Huang and Mauricio Castillo-Effen and Georgios Fainekos},
      year={2021},
      eprint={2110.10729},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
