#!/usr/bin/env python3

from argparse import ArgumentParser
from importlib import import_module
from sys import exit

ALL_BENCHMARKS = {"AT1", "AT2", "AT51", "AT52", "AT53", "AT54", "AT61", "AT62", "AT63", "AT64", "CC1", "CC2", "CC3", "CC4", "CC5", "CCx", "NN1", "NNx", "F16_alt4040", "SC1"}

def _get_benchmark(name, results_folder):
    if "AT" in name:
        mod = import_module(f"AT_benchmark.run_{name}")
    elif "CC" in name:
        mod = import_module(f"CC_benchmark.run_{name}")
    elif "NN" in name:
        mod = import_module(f"NN_benchmark.run_{name}")
    elif "F16" in name:
        mod = import_module(f"F16_benchmark.run_{name}")        
    elif "SC" in name:
        mod = import_module(f"SC_benchmark.run_{name}")        

    cls_name = f"Benchmark_{name}"
    ctor = getattr(mod, cls_name)

    return ctor(name, results_folder)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run arch benchmarks")
    parser.add_argument("benchmark_names", nargs="*", help="Name of benchmarks to run")
    parser.add_argument("-f", "--folder", default = "ARCHCOMP2022_PartX")
    parser.add_argument("-a", "--all", help="Run all benchmarks", action="store_true")
    parser.add_argument("-l", "--list", help="List all benchmarks", action="store_true")
    args = parser.parse_args()
    # print(args)
    if args.list:
        print(ALL_BENCHMARKS)
        exit(0)

    if args.all:
        benchmark_names = ALL_BENCHMARKS
    else:
        benchmark_names = set(args.benchmark_names)
        for name in benchmark_names:
            if name not in ALL_BENCHMARKS:
                raise ValueError(f"Unknown benchmark {name}")
    results_folder = args.folder
    
    benchmarks = [_get_benchmark(name, results_folder) for name in benchmark_names]

    if not benchmarks:
        raise ValueError("Must specify at least one benchmark to run")

    for benchmark in benchmarks:
        results = benchmark.run()
