import numpy as np
from staliro.options import SignalOptions
from staliro.specifications import RTAMTDense

def load_specification_dict(benchmark):

    phi_1 = "F[0,1] (pos >= 3.2)"
    phi_2 = "F[1,1.5] (G[0,0.5]((pos >= 1.75) and (pos <= 2.25)))"
    phi_3 = "G[2,3] ((pos >= 1.825) and (pos <= 2.175))"
    NNx_phi = f"(({phi_1}) and ({phi_2}) and ({phi_3}))"
    
    p1_phi = "(p1 >= 0)"
    p2_phi = "(F[0,2] (G[0,1] (not(p2 <= 0))))"
    NN_phi = f"G[1,18] ({p1_phi} -> {p2_phi})"

    spec_dict = {
        "NNx": RTAMTDense(NNx_phi, {"pos": 0}),
        "NN1": RTAMTDense(NN_phi, {"p1": 1, "p2":2})
        }

    


    if benchmark not in spec_dict.keys():
        raise ValueError(f"Inappropriate Benchmark name :{benchmark}. Expected one of {spec_dict.keys()}")
    
    if benchmark == "NNx":
        initial_conditions = [
            np.array([1.95, 2.05]),
            np.array([1.95, 2.05]),
            np.array([1.95, 2.05]),
        ]
    elif benchmark == "NN1":
        initial_conditions = [
            np.array([1.0, 3.0]),
            np.array([1.0, 3.0]),
            np.array([1.0, 3.0]),
        ]
    
    return spec_dict[benchmark], initial_conditions