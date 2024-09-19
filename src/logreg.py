import numpy as np
from util import *

class LogisticRegression():
    def __init__(self):
        self.weights = None  

    def fit(self, X, T, epochs=10000, lr=3):
        print(f"X Shape: {np.shape(X)}")
        print(f"T Shape: {np.shape(T)}")
        print(T)

        """
            The (11698, 10) of the X Matrix tells us that it contains 11698 datapoints of 10 features
            The (11698, 2) of the T Matrix tells us the one-hot encoded probability of the data-point being 1 or 0
        """
        data_points, features = X.shape
        outputs = T.shape[1]

        self.weights = np.ndarray(features)

        np.random.seed(30)
        self.weights.fill(np.random.normal())
        self.weights = self.weights.reshape(1,-1)

        print(f"Weights array: {self.weights}")

        for i in range(epochs):
            Y = self.predict(X)
            E = self.error(T, Y)

            # gradient descent formula
            y_minus_t = Y - T
            print(f"Y-T Shape: {y_minus_t.shape}")
            print(f"X Shape: {X.shape}")
            gradient = (Y - T).T @ X
            self.weights -= lr * gradient

    def predict(self, X):
        scores = softmax(np.dot(self.weights, X.T))
        scores.reshape(1,-1)
        print(f'Scores: {scores.shape}')
        predictions = toHotEncoding(scores, 2)
        return predictions
    
    def error(self, T, Y):
        log = np.log
        return -np.sum(T*log(Y)+(1-T)*log(1-Y))