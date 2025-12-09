import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Model:
    def __init__(self):
        """ Initialize the model.
        """
        self.regressor = RandomForestRegressor(
            n_estimators=20,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X, y, X_adapt):
        """ Train the model.

        Args:
            X: Training data matrix of shape (num-samples, num-features), type
            np.ndarray.
            y: Training label vector of shape (num-samples), type np.ndarray.
            X_adapt: DA training data matrix of shape (num-samples-DA, num-features),
            type np.ndarray.
        """
        y = y.ravel()
        self.regressor.fit(X, y, X_adapt)   

    def predict(self, X):
        """ Predict the labels.

        Args:
          X: Test data matrix of shape (num-samples, num-features) to pass to the
          model for inference, type np.ndarray.
        """
        y = self.regressor.predict(X)
        return y
