import numpy as np
import pandas as pd


class preprocessing:
    class MinMaxScaler:
        def __init__(self):
            pass

        def fit(self, X):
            """
            Compute the range for later scaling
            Args:
                X: array-like shape
            Return:
                self: fitted object
            """
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            self.range = self.max - self.min
            return self

        def transform(self, X):
            """
            Compute the min-max normalized column of X
            Args:
                X: array-like shape
            Returns:
                X: array-like shape with each column normalized by its z-score
            """
            return (X - self.min) / self.range

        def fit_transform(self, X):
            """
            Compute the min-max normalized column of X in one step
            Args:
                X: array-like shape
            Returns:
                X: array-like shape with each column normalized by its z-score
            """
            min = np.min(X, axis=0)
            max = np.max(X, axis=0)
            range = max - min

            return (X - min) / range

        def inverse_transform(self, X):
            """
            Undo the scaling of X
            Args:
                X: array-like shape
            Returns:
                X: array-like shape
            """
            return X * self.range + self.min

    class StandardScaler:
        def __init__(self):
            pass

        def fit(self, X):
            """
            Computes the mean and std for later scaling
            Args:
                X: array-like shape
            Returns:
                self: fitted object
            """
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            return self

        def transform(self, X):
            """
            Compute the z-score normalized column of X
            Args:
                X: array-like shape
            Returns:
                X: array-like shape
            """
            return (X - self.mean) / self.std
        
        def fit_transform(self, X):
            """
            Compute the z-score normalized column of X in one step
            Args:
                X: array-like shape
            Returns:
                X: array-like shape with each column normalized by its z-score
            """
            mean_ = np.mean(X, axis=0)
            std_ = np.std(X, axis=0)

            return (X - mean_) / std_
        
        def inverse_transform(self, X):
            """
            Undo the scaling of X
            Args:
                X: array-like shape
            Returns:
                X: array-like shape
            """

            return X * self.std + self.mean
        
    