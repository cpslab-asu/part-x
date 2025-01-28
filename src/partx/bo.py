from typing import Callable, Tuple
from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import norm

from .gpr import GPR
from .utilities import uniform_sampling, compute_robustness, OracleCreator, Fn

class BO_Interface(ABC):
    @abstractmethod
    def __init__(self):
        """ Initialize BO Method for use in Part-X

        Args:
            bo_model: Bayesian Optimization Class developed with partxv2.byesianOptimization.BO_Interface factory.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """Sampling using User Defined BO.

        Args:
            test_function: Function of System Under Test.
            num_samples: Number of samples to generate from BO.
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        """
        raise NotImplementedError
    
class BOSampling:
    def __init__(self, bo_model: BO_Interface) -> None:
        """ Initialize BO Method for use in Part-X

        Args:
            bo_model: Bayesian Optimization Class developed with partxv2.byesianOptimization.BO_Interface factory.
        """
        self.bo_model = bo_model

    def sample(
        self,
        test_function: Fn,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: GPR,
        oracle_info,
        rng,
    ) -> tuple: 
        """Wrapper around user defined BO Model.

        Args:
            test_function: Function of System Under Test.
            num_samples: Number of samples to generate from BO.
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        Returns:
            x_complete
            y_complete
            x_new
            y_new
        """

        dim = region_support.shape[0]
        if len(x_train.shape) != 2 or x_train.shape[1] != dim:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {x_train.shape} instead.")
        if len(y_train.shape) != 1:
            raise TypeError(f"Received evaluations set input: Expected (n,) array, received {y_train.shape} instead.")
        if x_train.shape[0] != y_train.shape[0]:
            raise TypeError(f"x_train, y_train set mismatch. x_train has shape {x_train.shape} and y_train has shape {y_train.shape}")


        x_new = []
        y_new = []
        n_tries = oracle_info.n_tries_BO
        while len(x_new) < num_samples and n_tries >= 0:
            # print(len(x_new))
            # print(n_tries)
            point = self.bo_model.sample(
                x_train, y_train, region_support, gpr_model, oracle_info, rng
            )
            # print(point)
            # print(point)
            if oracle_info(point).sat:
                x_new.append(point)
                pred_sample_y = compute_robustness(np.array([point]), test_function)
                x_train = np.vstack((x_train, np.array([point])))
                y_train = np.hstack((y_train, pred_sample_y))
                n_tries = oracle_info.n_tries_BO
            else:
                n_tries -= 1
                
            if n_tries == 0:
                
                additional_samples = uniform_sampling(1,region_support, region_support.shape[0],oracle_info, rng)[0]
                x_new.append(additional_samples)
                pred_sample_y = compute_robustness(np.array([additional_samples]), test_function)
                x_train = np.vstack((x_train, np.array([additional_samples])))
                y_train = np.hstack((y_train, pred_sample_y))
                n_tries = oracle_info.n_tries_BO
                # print(additional_samples)
        
        
        
        if n_tries == 0 and len(x_new)!=num_samples:
            raise RuntimeError(f"Could not perform BO sampling, {oracle_info.n_tries_BO} trials exhausted. Please adjust the constraints or increase the budget for oracle_info.n_tries_BO") 
        x_new = np.array(x_new)
        y_new = np.array(y_new)
        # print(x_new)
        # print(x_new.shape)
        assert len(x_new.shape) == 2, f"Returned samples set: Expected (n, dim) array, returned {x_train.shape} instead."
        assert len(y_new.shape) == 1, f"Returned evaluations set input: Expected (n, ) array, returned {y_new.shape} instead."
        assert len(x_train.shape) == 2, f"Returned merged samples set input: Expected (n, dim) array, returned {x_train.shape} instead."
        assert len(y_train.shape) == 1, f"Returned merged evaluations set input: Expected (n, ) array, returned {y_train.shape} instead."

        return x_train, y_train, x_new, y_new


# local_oracle = None


class InternalBO(BO_Interface):
    def __init__(self):
        self.local_oracle = OracleCreator(None,1,1)

    def sample(
         self,
         x_train: NDArray,
         y_train: NDArray,
         region_support: NDArray,
         gpr_model:  GPR,
         oracle_info,
         rng,
      ) -> NDArray:

        """Internal BO Model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            oracle_info: Oracle defining the constraints.
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        Returns:
            x_new
         """
        
        model = GPR(gpr_model)
        constraint_model = GPR(gpr_model)
        model.fit(x_train, y_train)

        pred_sample_x = self._opt_acquisition(y_train, model, constraint_model, region_support, oracle_info, rng)


        return pred_sample_x

    def _opt_acquisition(self, y_train: NDArray, gpr_model: GPR, constraint_model, region_support: NDArray, oracle_info, rng) -> NDArray:
        """Get the sample points

        Args:
            X: sample points
            y: corresponding robustness values
            model: the GP models
            sbo: sample points to construct the robustness values
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)
            region_support: The bounds of the region within which the sampling is to be done.
                                        Region Bounds is M x N x O where;
                                            M = number of regions;
                                            N = test_function_dimension (Dimensionality of the test function);
                                            O = Lower and Upper bound. Should be of length 2;

        Returns:
            The new sample points by BO
        """

        tf_dim = region_support.shape[0]
        lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
        upper_bound_theta = np.ndarray.flatten(region_support[:, 1])

        random_samples = uniform_sampling(20000, region_support, tf_dim, self.local_oracle, rng)
        
        curr_best = np.min(y_train)
        # constraints_out = np.array([oracle_info(x).val for x in random_samples])
        # [print(oracle_info(x).val, oracle_info(x).sat) for x in random_samples]
        # if oracle_info.oracle_function is not None:
        #     constraint_model.fit(random_samples, constraints_out)

        # bnds = Bounds(lower_bound_theta, upper_bound_theta)
        fun = lambda x_: -1 * (self._acquisition(y_train, x_, gpr_model, constraint_model, oracle_info) - 1000*max(0,oracle_info(x_).val))

        
        constraints_out = np.array([max(0, oracle_info(x).val) for x in random_samples])
        min_bo_val = -1 * (self._acquisition(
            y_train, random_samples, gpr_model, constraint_model, oracle_info, "multiple"
        ) - 1000*constraints_out)

        min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
        # import matplotlib.pyplot as plt
        # plt.plot(random_samples, min_bo_val, ".")
        # plt.show()
        min_bo_val = np.min(min_bo_val)

        for _ in range(9):
            new_params = minimize(
                fun,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_bo,
            )

            if not new_params.success:
                continue

            if min_bo is None or fun(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = fun(min_bo)
        new_params = minimize(
            fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        min_bo = new_params.x
        # penalty = oracle_info(np.array(min_bo)).val

        return np.array(min_bo)

    def _surrogate(self, gpr_model: Callable, x_train: NDArray):
        """_surrogate Model function

        Args:
            model: Gaussian process model
            X: Input points

        Returns:
            Predicted values of points using gaussian process model
        """

        return gpr_model.predict(x_train)

    def _acquisition(self, y_train: NDArray, sample: NDArray, gpr_model: Callable, constraint_model, oracle_info, sample_type:str ="single") -> NDArray:
        """Acquisition Model: Expected Improvement

        Args:
            y_train: corresponding robustness values
            sample: Sample(s) whose EI is to be calculated
            gpr_model: GPR model
            sample_type: Single sample or list of model. Defaults to "single". other options is "multiple".

        Returns:
            EI of samples
        """
        curr_best = np.min(y_train)

        if oracle_info.oracle_function is not None:
            if sample_type == "multiple":
                mu, std = self._surrogate(gpr_model, sample)
                mu_con, std_con = constraint_model.predict(sample)
                ei_list = []
                for mu_iter, std_iter, samp, mu_con_iter, std_con_iter in zip(mu, std, sample, mu_con, std_con):
                    # if oracle_info(samp).sat:
                    pred_var = std_iter

                    if pred_var > 0:
                        con_term = norm.cdf(0, mu_con_iter, std_con_iter)
                        var_1 = curr_best - mu_iter
                        var_2 = var_1 / pred_var

                        ei = (var_1 * norm.cdf(var_2)) + (
                            pred_var * norm.pdf(var_2)
                        ) * 1
                    else:
                        ei = 0.0
                    # else:
                    #     ei = -1
                    # ei = ei + 1000 * min(0,oracle_info(samp).val)
                    ei_list.append(ei)
                
            elif sample_type == "single":
                # if oracle_info(sample).sat:
                mu, std = self._surrogate(gpr_model, sample.reshape(1, -1))
                mu_con, std_con = constraint_model.predict(np.array([sample]))
                pred_var = std[0]
                if pred_var > 0:
                    con_term = norm.cdf(0,mu_con[0], std_con[0])
                    var_1 = curr_best - mu[0]
                    var_2 = var_1 / pred_var

                    ei = (var_1 * norm.cdf(var_2)) + (
                        pred_var * norm.pdf(var_2)
                    )
                else:
                    ei = 0.0
                # print(sample)
                # ei = ei - 1 * max(0,oracle_info(sample).val)
                # else:
                #     ei = -1
                # return ei
        else:
            if sample_type == "multiple":
                mu, std = self._surrogate(gpr_model, sample)
                # mu_con, std_con = constraint_model.predict(sample)
                ei_list = []
                for mu_iter, std_iter, samp in zip(mu, std, sample):
                    # if oracle_info(samp).sat:
                    pred_var = std_iter

                    if pred_var > 0:
                        # con_term = norm.cdf(0, mu_con_iter, std_con_iter)
                        var_1 = curr_best - mu_iter
                        var_2 = var_1 / pred_var

                        ei = (var_1 * norm.cdf(var_2)) + (
                            pred_var * norm.pdf(var_2)
                        ) 
                    else:
                        ei = 0.0
                    # else:
                    #     ei = -999
                    ei_list.append(ei)
                
            elif sample_type == "single":
                    # if oracle_info(sample).sat:
                mu, std = self._surrogate(gpr_model, sample.reshape(1, -1))
                # mu_con, std_con = constraint_model.predict(np.array([sample]))
                pred_var = std[0]
                if pred_var > 0:
                    # con_term = norm.cdf(0,mu_con[0], std_con[0])
                    var_1 = curr_best - mu[0]
                    var_2 = var_1 / pred_var

                    ei = (var_1 * norm.cdf(var_2)) + (
                        pred_var * norm.pdf(var_2)
                    )
                else:
                    ei = 0.0
                # else:
                #     ei = -999
                # return ei

        if sample_type == "multiple":
            return_ei = np.array(ei_list)
        elif sample_type == "single":
            return_ei = ei

        return return_ei
