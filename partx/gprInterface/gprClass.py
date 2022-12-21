class GPR:
    def __init__(self, gpr_model) -> None:
        self.gpr_model = gpr_model

    def fit(self, x_train, y_train):
        """ Wrapper to fit user defined gpr model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train
        """
        if len(x_train.shape) != 2:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {x_train.shape} instead.")
        if len(y_train.shape) != 1:
            raise TypeError(f"Received evaluations set input: Expected (n,) array, received {y_train.shape} instead.")
        if x_train.shape[0] != y_train.shape[0]:
            raise TypeError(f"x_train, y_train set mismatch. x_train has shape {x_train.shape} and y_train has shape {y_train.shape}")

        self.gpr_model.fit_gpr(x_train, y_train)

    def predict(self, X):
        """Wrapper to predict from user defined gpr model

        Args:
            X: Samples for predicting

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array

        Returns:
            mean
            std
        """
        if len(X.shape) != 2:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {X.shape} instead.")

        mean, std = self.gpr_model.predict_gpr(X)

        assert len(mean.shape) == 1, f"Mean from GPR should be of shape (n, ). Received {mean.shape} instead."
        assert len(std.shape) == 1, f"std_dev from GPR should be of shape (n, ). Received {std.shape} instead."
        assert mean.shape == std.shape, f"Mean and std_dev mismatch. Mean has a shape of {mean.shape} and std_dev has a shape of {std.shape}."

        return mean, std