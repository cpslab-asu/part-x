from typing import Callable

from ..sampling import uniform_sampling
from numpy.typing import NDArray
import numpy as np
from ..utils import compute_robustness

class BOSampling:
    def __init__(self, bo_model: Callable) -> None:
        """ Initialize BO Method for use in Part-X

        Args:
            bo_model: Bayesian Optimization Class developed with partxv2.byesianOptimization.BO_Interface factory.
        """
        self.bo_model = bo_model

    def sample(
        self,
        test_function: Callable,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
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
