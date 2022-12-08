from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor


class KNNRegressor:
    def __init__(self, k_nb: int) -> None:
        self.k_nb = k_nb
        
    def fit(self, X:List[float], y:List[float]) -> None:
        self.X = X
        self.y = y
        
    def predict(self, data:List[float]) -> List[float]:
        predictions_list = np.array([])
        
        for item in data:
            k_distance = np.linalg.norm(self.X - item, axis=1)
            k_indices = np.argsort(k_distance)[:self.k_nb]

            predictions_list = np.append(predictions_list, np.mean(self.y[k_indices]))
            
        return predictions_list


if __name__ == "__main__":
    X, y = make_regression(n_samples=228,
                           n_features=1,
                           noise=20,
                           random_state=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    my_classifier = KNNRegressor(k_nb = 5)
    my_classifier.fit(X_train, y_train)
    print(my_classifier.predict(X_test))

    imported_classifier = KNeighborsRegressor()
    imported_classifier.fit(X_train, y_train)
    print(imported_classifier.predict(X_test))