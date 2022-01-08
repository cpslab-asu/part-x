from partx.numerical.sampling import lhs_sampling
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF,WhiteKernel
import numpy as np
from scipy import stats
from partx.numerical.optimizer_gpr import optimizer_lbfgs_b
from partx.numerical.calculate_robustness import calculate_robustness
from partx.kriging_gpr.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from partx.kriging_gpr.interface.OK_Rpredict import OK_Rpredict

class gpRegressorModel:
    def __init__(self, *library_info):
        if len(library_info) == 2:
            if (library_info[0] == "kriging"):
                # print("Using default Kriging Model for Gaussian Processs Regression")
                self.library_name = library_info[0]
                self.kriging_parameter = library_info[1]

            elif (library_info[0] == "other"):
                # print("Using other model for Gaussian Process Regression")
                self.library_name = library_info[0]
                self.gpr_model = library_info[1]

            else:
                #TODO ADD true values vs errored values
                raise Exception("Error in defining model. check parameters and their names again.") 
            
        else: 
            raise Exception("Error in number of parameters.")
    
    def call_fit(self, X, Y):
        if self.library_name == "other":
            self.gpr_model.fit(X, Y)
        elif self.library_name == "kriging":
            self.gpr_model = OK_Rmodel_kd_nugget(X, Y, 0, 2, self.kriging_parameter)
    
    def call_predict(self, X):
        if self.library_name == "other":
            self.y_pred, self.pred_sigma = self.gpr_model.predict(X, return_std=True)

        elif self.library_name == "kriging":
            self.y_pred, self.pred_sigma = OK_Rpredict(self.gpr_model, X, 0)

        return self.y_pred, self.pred_sigma


