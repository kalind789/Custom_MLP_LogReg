import numpy as np
import pandas as pd

# Sigmoid Function
def sigmoid(n):
    return 1/(1+np.exp(n))

# Calculates the Error value
def Error(value):
    np.sum(sigmoid(np.dot(weights, features)))

# Gradient Descent Algorithm
def gradient_descent():
    converged = False
    while not converged:
        new_weight = curr_weight + learning_rate * (np.gradient())
