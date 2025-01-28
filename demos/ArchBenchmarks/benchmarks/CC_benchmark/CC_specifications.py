import numpy as np
from staliro import SignalInput
from staliro.specifications import rtamt
from staliro.signals import piecewise_constant

def load_specification_dict(benchmark, instance):

    
    CC1_phi = "G[0, 100] (y54 <= 40)"
    CC2_phi = "G[0, 70] (F[0,30] (y54 >= 15))"
    CC3_phi = "G[0, 80] ((G[0, 20] (y21 <= 20)) or (F[0,20] (y54 >= 40)))"
    CC4_phi = "G[0,65] (F[0,30] (G[0,5] (y54 >= 8)))"
    CC5_phi = "G[0,72] (F[0,8] ((G[0,5] (y21 >= 9)) -> (G[5,20] (y54 >= 9))))"

    phi_1 = "(G[0, 50] (y21 >= 7.5))"
    phi_2 = "(G[0, 50] (y32 >= 7.5))"
    phi_3 = "(G[0, 50] (y43 >= 7.5))"
    phi_4 = "(G[0, 50] (y54 >= 7.5))"
    CCx_phi = phi_1 + " and " + phi_2 + " and " + phi_3 + " and " + phi_4


    spec_dict = {
        "CC1": rtamt.parse_dense(CC1_phi, {"y54": 3}),
        "CC2": rtamt.parse_dense(CC2_phi, {"y54": 3}),
        "CC3": rtamt.parse_dense(CC3_phi, {"y21": 0, "y54":3}),
        "CC4": rtamt.parse_dense(CC4_phi, {"y54": 3}),    
        "CC5": rtamt.parse_dense(CC5_phi, {"y21": 0, "y54":3}),    
        "CCx": rtamt.parse_dense(CCx_phi,{"y21":0, "y32":1, "y43":2, "y54":3}),
        }
    
    if instance == 1:
        signals = {
            "throttle": SignalInput(control_points = [(0., 1.)]*10),
            "brake": SignalInput(control_points = [(0., 1.)]*10),
        }
    
    if benchmark not in spec_dict.keys():
        raise ValueError(f"Inappropriate Benchmark name :{benchmark}. Expected one of {spec_dict.keys()}")
    
    return spec_dict[benchmark], signals