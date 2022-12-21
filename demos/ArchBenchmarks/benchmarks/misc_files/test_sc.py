from copyreg import pickle
import logging

import numpy as np
from numpy.typing import NDArray

from staliro.options import Options, SignalOptions
from staliro.specifications import TLTK, RTAMTDense
from ..models import SCModel
from staliro.staliro import staliro, simulate_model
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle

signals = [
    SignalOptions(control_points = [(3.95, 4.01)]*18, signal_times=np.linspace(0.,35.,18)),
]

print(signals)
options = Options(runs=1, iterations=1, interval=(0, 35), signals=signals)



phi = "G[30,35] ((pressure <= 87.5) and (pressure >= 87))"
specification_rtamt = RTAMTDense(phi, {"pressure":0})


def generateRobustness(sample, inModel, options: Options, specification):
    
    result = simulate_model(inModel, options, sample)
    
    with open("SC_trace.pkl","wb") as f:
        pickle.dump(result, f)
    return specification.evaluate(result.states, result.times)

sample1 = [3.99]*14 + [4.01]*4
print(sample1)          
sample2 = loadmat("trace.mat")
# print(sample2["trace"][0][0][0])
times = sample2["trace"][0][0][0][0,:]
signals = sample2["trace"][0][0][1]
print(times.shape)
print(signals.shape)

autotrans_blackbox = SCModel()

rob1 = generateRobustness(sample1,  autotrans_blackbox, options, specification_rtamt)


print(f"Rob. Sample 1 = {rob1}")
print(f"Rob from mat trace = {specification_rtamt.evaluate(signals, times)}")
# print(str(rob1))

