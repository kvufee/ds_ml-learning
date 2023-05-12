from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression


X, y = make_regression(n_samples = 1000,
                       n_features = 1,
                       noise = 1,
                       random_state=True)

X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y)


class LinearRegressor:
    
    def __init__(self) -> None:
        self.learning_rate = 0.01
        self.m = 0
        self.b = 0
        
        
    def fit(self, X:np.ndarray, y:np.ndarray, n_epochs: int = 100) -> None:
        self.X = X
        self.y = y
        
        loss = np.array([])
        
        for epoch in range(n_epochs):
            for i in range(len(X)):
                loss = np.append(loss, (np.square(self.predict(X[i]) - y[i])))

                der_m, der_b = self.calculate_descent(X[i], y[i])

                self.m = self.m - self.learning_rate * der_m
                self.b = self.b - self.learning_rate * der_b

        
    def calculate_descent(self, X:List[float], y:float) -> Tuple:
        der_m = 2 * ((self.m * X + self.b) - y) * X
        der_b = 2 * ((self.m * X + self.b) - y)
        
        return der_m, der_b
    
        
    def predict(self, X:List[float]) -> float:
        return self.m * X + self.b
        
    
# my output
lineal_regressor = LinearRegressor()
lineal_regressor.fit(X_train, y_train)
print(lineal_regressor.predict(X_test.flatten()))

#imported output
linreg_imported = LinearRegression()
linreg_imported.fit(X_train, y_train)
print(linreg_imported.predict(X_test))