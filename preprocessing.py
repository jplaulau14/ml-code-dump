import numpy as np
import pandas as pd

class Preprocessing:
    class MinMaxScaler:
        def __init__(self, X):
            self.X = X
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            self.range = self.max - self.min

        def transform(self, X):
            return (X - self.min) / self.range

        def inverse_transform(self, X):
            return X * self.range + self.min
        
    class StandardScaler:
        def __init__(self, X):
            self.X = X
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)

        def transform(self, X):
            return (X - self.mean) / self.std

        def inverse_transform(self, X):
            return X * self.std + self.mean