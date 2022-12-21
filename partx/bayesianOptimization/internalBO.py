from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from .bointerface import BO_Interface
from ..gprInterface import GPR
from ..sampling import uniform_sampling
from ..utils import compute_robustness



class InternalBO(BO_Interface):
    def __init__(self):
        pass

    def sample(
        self,
        test_function: Callable,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng,
    ) -> Tuple[NDArray]:

        """Internal BO Model

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

        x_pred = np.empty((0, region_support.shape[0]))
        y_pred = np.empty(0)

        for sample in range(num_samples):
            model = GPR(gpr_model)
            model.fit(x_train, y_train)

            pred_sample_x = self._opt_acquisition(y_train, model, region_support, rng)
            
            pred_sample_y = compute_robustness(np.array([pred_sample_x]), test_function)
            x_train = np.vstack((x_train, np.array([pred_sample_x])))
            y_train = np.hstack((y_train, pred_sample_y))

            x_pred = np.vstack((x_pred, np.array([pred_sample_x])))
            y_pred = np.hstack((y_pred, pred_sample_y))
        return x_train, y_train, x_pred, y_pred

    def _opt_acquisition(self, y_train: NDArray, gpr_model: Callable, region_support: NDArray, rng) -> NDArray:
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

        curr_best = np.min(y_train)

        # bnds = Bounds(lower_bound_theta, upper_bound_theta)
        fun = lambda x_: -1 * self._acquisition(y_train, x_, gpr_model)

        random_samples = uniform_sampling(2000, region_support, tf_dim, rng)
        min_bo_val = -1 * self._acquisition(
            y_train, random_samples, gpr_model, "multiple"
        )

        min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
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

    def _acquisition(self, y_train: NDArray, sample: NDArray, gpr_model: Callable, sample_type:str ="single") -> NDArray:
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

        if sample_type == "multiple":
            mu, std = self._surrogate(gpr_model, sample)
            ei_list = []
            for mu_iter, std_iter in zip(mu, std):
                pred_var = std_iter
                if pred_var > 0:
                    var_1 = curr_best - mu_iter
                    var_2 = var_1 / pred_var

                    ei = (var_1 * norm.cdf(var_2)) + (
                        pred_var * norm.pdf(var_2)
                    )
                else:
                    ei = 0.0

                ei_list.append(ei)
            # print(np.array(ei_list).shape)
            # print("*****")
            # return np.array(ei_list)
        elif sample_type == "single":
            # print("kfkf")
            mu, std = self._surrogate(gpr_model, sample.reshape(1, -1))
            pred_var = std[0]
            if pred_var > 0:
                var_1 = curr_best - mu[0]
                var_2 = var_1 / pred_var

                ei = (var_1 * norm.cdf(var_2)) + (
                    pred_var * norm.pdf(var_2)
                )
            else:
                ei = 0.0
            # return ei

        if sample_type == "multiple":
            return_ei = np.array(ei_list)
        elif sample_type == "single":
            return_ei = ei

        return return_ei
