import numpy as np
import pandas as pd
from util import loadDataset

loans_train = './data/loans.train'

ltrain_data, ltrain_targets = loadDataset(loans_train)
loans_train_features = pd.DataFrame(data=ltrain_data)
print(loans_train_features.head())
print(ltrain_data)
print(ltrain_targets)

# Logistic Regression
def LogisticReg(lltrain_targets, loans_train_features):
    pass


# Sigmoid Function
def sigmoid(n):
    return 1/(1+np.exp(n))

# Calculates the Error value
def Error(value):
    np.sum(sigmoid(np.dot(weights, features)))

# Gradient Descent Algorithm
def gradient_descent(learning_rate):
    converged = False
    while not converged:
        new_weight = curr_weight + learning_rate * (np.gradient())
