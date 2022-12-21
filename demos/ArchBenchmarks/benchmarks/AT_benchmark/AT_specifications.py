import numpy as np
from staliro.options import SignalOptions
from staliro.specifications import RTAMTDense

def load_specification_dict(benchmark):

    

    AT1_phi = "G[0, 20] (speed <= 120)"

    AT2_phi = "G[0, 10] (rpm <= 4750)"

    gear_1_phi = f"(gear <= 1.5 and gear >= 0.5)"
    AT51_phi = f"G[0, 30] (((not {gear_1_phi}) and (F[0.001,0.1] {gear_1_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_1_phi})))"

    gear_2_phi = f"(gear <= 2.5 and gear >= 1.5)"
    AT52_phi = f"G[0, 30] (((not {gear_2_phi}) and (F[0.001,0.1] {gear_2_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_2_phi})))"

    gear_3_phi = f"(gear <= 3.5 and gear >= 2.5)"
    AT53_phi = f"G[0, 30] (((not {gear_3_phi}) and (F[0.001,0.1] {gear_3_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_3_phi})))"

    gear_4_phi = f"(gear <= 4.5 and gear >= 3.5)"
    AT54_phi = f"G[0, 30] (((not {gear_4_phi}) and (F[0.001,0.1] {gear_4_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_4_phi})))"

    AT6a_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,4] (speed <= 35)))"
    AT6b_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,8] (speed <= 50)))"
    AT6c_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,20] (speed <= 65)))"
    AT6abc_phi = f"{AT6a_phi} and {AT6b_phi} and {AT6c_phi}"

    spec_dict = {
        "AT1": RTAMTDense(AT1_phi, {"speed": 0}),
        "AT2": RTAMTDense(AT2_phi, {"rpm": 1}),
        "AT51": RTAMTDense(AT51_phi, {"gear": 2}),    
        "AT52": RTAMTDense(AT52_phi, {"gear": 2}),    
        "AT53": RTAMTDense(AT53_phi, {"gear": 2}),    
        "AT54": RTAMTDense(AT54_phi, {"gear": 2}),    
        "AT61": RTAMTDense(AT6a_phi, {"speed": 0, "rpm":1}),
        "AT62": RTAMTDense(AT6b_phi, {"speed": 0, "rpm":1}),
        "AT63": RTAMTDense(AT6c_phi, {"speed": 0, "rpm":1}),
        "AT64": RTAMTDense(AT6abc_phi, {"speed": 0, "rpm":1}),
        }

    signals = [
        SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
        SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
    ]

    if benchmark not in spec_dict.keys():
        raise ValueError(f"Inappropriate Benchmark name :{benchmark}. Expected one of {spec_dict.keys()}")
    
    return spec_dict[benchmark], signals