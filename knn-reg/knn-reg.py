from typing import List
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor


X, y = make_regression(n_samples=228,
                       n_features=1,
                       noise=20,
                       random_state=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)


class KNNRegressor:
    def __init__(self, k_nb: int) -> None:
        self.k_nb = k_nb
        
    def fit(self, X:List[float], y:List[float]) -> None:
        self.X = X
        self.y = y
        
    def predict(self, data:List[float]) -> List[float]:
        predict_data = np.array([])
        
        for item in data:
            k_distance = np.linalg.norm(self.X - item, axis=1)
            k_indices = np.argsort(k_distance)[:self.k_nb]

            predict_data = np.append(predict_data, np.mean(self.y[k_indices]))
            
        return predict_data


# bebra = KNNRegressor(k_nb = 5)
# bebra.fit(X_train, y_train)
# bebra.predict(X_test)

# bobik = KNeighborsRegressor(n_neighbors = 5)
# bobik.fit(X_train, y_train)
# bobik.predict(X_test)