from typing import Callable
from numpy.typing import NDArray

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

        
        x_complete, y_complete, x_new, y_new = self.bo_model.sample(
            test_function, num_samples, x_train, y_train, region_support, gpr_model, rng
        )

        assert len(x_new.shape) == 2, f"Returned samples set: Expected (n, dim) array, returned {x_train.shape} instead."
        assert len(y_new.shape) == 1, f"Returned evaluations set input: Expected (n, ) array, returned {y_new.shape} instead."
        assert len(x_complete.shape) == 2, f"Returned merged samples set input: Expected (n, dim) array, returned {x_complete.shape} instead."
        assert len(y_complete.shape) == 1, f"Returned merged evaluations set input: Expected (n, ) array, returned {y_complete.shape} instead."

        return x_complete, y_complete, x_new, y_new
