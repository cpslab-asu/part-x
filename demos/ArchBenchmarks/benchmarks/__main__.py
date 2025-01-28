#!/usr/bin/env python3

from argparse import ArgumentParser
from importlib import import_module
from sys import exit
import json

ALL_COMBINATIONS = {
    "CC": {
        "1": ["CC1"],
    }
}

def _get_benchmark(model, property_name, instance, results_folder):
    
    if instance == "1":
        if "CC" in model:
            mod = import_module(f"CC_benchmark.run_{model}")
        
    cls_name = f"Benchmark_{model}"
    ctor = getattr(mod, cls_name)

    return ctor(property_name, int(instance), results_folder)


if __name__ == "__main__":

    parser = ArgumentParser(description=f"Run arch benchmarks \n {ALL_COMBINATIONS}")
    parser.add_argument("-m", "--model", default = "", help = "Model Name")
    parser.add_argument("-p", "--property", default = "", help = "Property Name")
    parser.add_argument("-i", "--instance", default = "", help = "Instance number")
    parser.add_argument("-f", "--folder", default = "/scratch/tkhandai/Arch_Comp_2024/PartX")
    
    args = parser.parse_args()
    
    if args.model in ALL_COMBINATIONS.keys():
        if args.instance in ALL_COMBINATIONS[args.model].keys():
            if args.property in ALL_COMBINATIONS[args.model][args.instance]:
                results_folder = args.folder
                
                benchmarks = _get_benchmark(args.model, args.property, args.instance, results_folder)

                results = benchmarks.run()
            else:
                raise ValueError(f"Unknown property {args.property} for model {args.model} Instance {args.instance}. \nAllowed = {ALL_COMBINATIONS[args.model][args.instance]}")
        else:
            raise ValueError(f"Unknown instance {args.instance} for model {args.model}. \nAllowed = {[x for x in ALL_COMBINATIONS[args.model].keys()]}")
    else:
        raise ValueError(f"Unknown model {args.model}. \nAllowed = {[x for x in ALL_COMBINATIONS.keys()]}")