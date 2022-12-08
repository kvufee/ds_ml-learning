from typing import List
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, k_nb: int) -> None:
        self.k_nb = k_nb

    def fit(self, X:List[float], y:List[float]) -> None:
        self.X = X
        self.y = y

    def predict(self, data:List[float]) -> List[int]:
        predictions_list = []
        neighbors = np.array([])
        
        for item in data:
            k_distance = np.linalg.norm(self.X - item, axis=1)
            k_indices = np.argsort(k_distance)[:self.k_nb]
            
            neighbors = self.y[k_indices]

            values = Counter(neighbors)
    
            predictions_list.append(sorted(values.items(), key=lambda x: x[1], reverse=True)[0][0])
        
        return predictions_list


if __name__ == "__main__":
    X, y = make_classification()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    my_classifier = KNNClassifier(k_nb = 5)
    my_classifier.fit(X_train, y_train)
    print(my_classifier.predict(X_test))

    imported_classifier = KNeighborsClassifier()
    imported_classifier.fit(X_train, y_train)
    print(imported_classifier.predict(X_test))
