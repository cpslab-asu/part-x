import scipy.io
from staliro.specifications import TLTK, RTAMTDense
import tltk_mtl as MTL
from staliro.staliro import staliro, simulate_model
import scipy.io
from taliro import tptaliro
import numpy as np


phi = "G[0,65] (F[0,30] (G[0,20] (y54 >= 8)))"
# specification = MTL.Global(float(0), float(65), MTL.Finally(float(0),float(30), MTL.Global(float(0),float(20), MTL.Predicate("y54", float(-1), float(-8)))))

spec_tltk = TLTK(phi, {"y54": 3})
spec_rtamtdense = RTAMTDense(phi, {"y54":3})
spec_mtl = MTL.Global(float(0), float(65), MTL.Finally(float(0),float(30), MTL.Global(float(0),float(20), MTL.Predicate("y54", float(-1), float(-8)))))
spec_tptaliro = "@Var_t1 [](({ Var_t1 >= 0 } /\ { Var_t1 <= 65 }) -> (@Var_t2 (<>(({ Var_t2 >= 0 } /\ { Var_t2 <= 30 }) /\ ([](@Var_t3 (({ Var_t3 >= 0 } /\ { Var_t3 <= 20 }) -> (y54))))))))"


states_python = scipy.io.loadmat("states_data.mat")['arr']
times_python = scipy.io.loadmat("time_data.mat")['arr']

states_matlab = scipy.io.loadmat("states_matlab.mat")['YT']
times_matlab = scipy.io.loadmat("times_matlab.mat")['TT']


print(f"TLTK python: {spec_tltk.evaluate(states_python, times_python[0,:])}")
print(f"TLTK matlab: {spec_tltk.evaluate(states_matlab.T, times_matlab[:,0])}")


print(f"RTAMTDense python: {spec_rtamtdense.evaluate(states_python, times_python[0,:])}")
print(f"RTAMTDense matlab: {spec_rtamtdense.evaluate(states_matlab.T, times_matlab[:,0])}")

spec_mtl.reset()
spec_mtl.eval_interval({"y54" : states_python[3,:].astype(np.float64)}, times_python[0,:].astype(np.float32))
print(f"MTL python: {spec_mtl.robustness}")

spec_mtl.reset()
spec_mtl.eval_interval({"y54" : (states_matlab.T)[3,:].astype(np.float64)}, times_matlab[:,0].astype(np.float32))
print(f"MTL matlab: {spec_mtl.robustness}")

tptal_rob_python = tptaliro.tptaliro(spec_tptaliro, 
                                    [{"name": "y54", "a": np.array(-1, dtype = np.double, ndmin = 2), "b": np.array(-8, dtype = np.double, ndmin = 2)}], 
                                    np.array(states_python[3,:], dtype = np.double, ndmin=2), 
                                    np.array(times_python[0,:], dtype = np.double, ndmin=2),)


tptal_rob_matlab = tptaliro.tptaliro(spec_tptaliro, 
                                    [{"name": "y54", "a": np.array(-1, dtype = np.double, ndmin = 2), "b": np.array(-8, dtype = np.double, ndmin = 2)}], 
                                    np.array((states_matlab.T)[3,:], dtype = np.double, ndmin=2), 
                                    np.array(times_matlab[:,0], dtype = np.double, ndmin=2),)                                    

print(f"TPTaliro python: {tptal_rob_python}")
print(f"TPTaliro matlab: {tptal_rob_matlab}")
