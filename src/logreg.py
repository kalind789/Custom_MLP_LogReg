import numpy as np
from util import *


class LogisticRegression():

    def __init__(self):
        self.weights = []


    # Get the weights needed to calculate the answer
    def fit(self, X,T):
        shape = X.shape
        value = X.shape[1] 
        self.weights = np.random.randn(value, 2)

    # Using dataset X, and weights from fit to get value of Y
    def predict(self, X):
        calc = X @ self.weights
        print(calc)
        return calc
    
    def error(self, T, Y):
        return -np.sum(T * np.log(Y) + (1-T) * np.log(1-Y))
    
    def GradientDescent(self, X, T, Y, lr=0.03):
        prob = softmax(Y)
        value = prob - T
        grad = X.T @ value 
        self.weights = self.weights - lr * grad
        