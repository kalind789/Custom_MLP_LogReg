import numpy as np
from util import *

class LogisticRegression():
    def __init__(self):
        self.weights = None  

    def fit(self, X, T, epochs=1000, lr=.003):
        """
            The (11698, 10) of the X Matrix tells us that it contains 11698 datapoints of 10 features
            The (11698, 2) of the T Matrix tells us the one-hot encoded probability of the data-point being 1 or 0
        """
        data_points, features = X.shape
        outputs = T.shape[1]

        np.random.seed(30)
        self.weights = np.random.randn(outputs, features)
        
        for i in range(epochs):
            Y = self.predict(X)
            E = self.error(T, Y)

            # gradient descent formula            
            gradient = np.dot((Y - T).T, X)

            self.weights -= lr * gradient

    def predict(self, X):
        scores = softmax(np.dot(X, self.weights.T))
        predictions = np.argmax(scores, axis=1)
        return predictions
    
    def error(self, T, Y):
        log = np.log
        return -np.sum(T * log(Y + 1e-9)) / T.shape[0]