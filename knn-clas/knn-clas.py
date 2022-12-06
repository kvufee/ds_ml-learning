import numpy as np
from typing import List
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier


X, y = make_classification()

X_train, X_test, y_train, y_test = train_test_split(X, y)


class KNNClassifier:
    def __init__(self, k_nb: int) -> None:
        self.k_nb = k_nb

    def fit(self, X:List[float], y:List[float]) -> None:
        self.X = X
        self.y = y

    def predict(self, data:List[float]) -> List[int]:
        predict_data = []
        neighbors = np.array([])
        
        for item in data:
            k_distance = np.linalg.norm(self.X - item, axis=1)
            k_indices = np.argsort(k_distance)[:self.k_nb]
            
            neighbors = self.y[k_indices]

            values = Counter(neighbors)
            amount = dict(values)
    
            predict_data.append(sorted(amount.items(), key=lambda x: x[1], reverse=True)[0][0])
        
        return predict_data


# my class and function
bebra = KNNClassifier(k_nb = 5)
bebra.fit(X_train, y_train)
print(bebra.predict(X_test))

#imported function
bobik = KNeighborsClassifier()
bobik.fit(X_train, y_train)
print(bobik.predict(X_test))
