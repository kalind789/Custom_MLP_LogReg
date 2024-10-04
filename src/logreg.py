import numpy as np
from util import *

class LogisticRegression():
    def __init__(self):
        """
        Logistic Regression classifier with softmax output.
        Weights are initialized in the `fit` method.
        """
        self.weights = None
        self.mean = None
        self.std = None

    def fit(self, X, T, epochs=1000, lr=0.003, batch_size=32):
        """
        Train the logistic regression model using stochastic gradient descent.

        Parameters:
        - X: Input features, shape (N, D)
        - T: One-hot encoded targets, shape (N, K)
        - epochs: Number of training epochs (default: 1000)
        - lr: Learning rate (default: 0.003)
        - batch_size: Size of mini-batches for SGD (default: 32)
        """
        data_points, features = X.shape
        outputs = T.shape[1]

        # Feature scaling
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        X_scaled = (X - self.mean) / self.std

        np.random.seed(30)
        self.weights = np.random.randn(outputs, features)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(data_points)
            np.random.shuffle(indices)
            X_shuffled = X_scaled[indices]
            T_shuffled = T[indices]
            
            for start_idx in range(0, data_points, batch_size):
                end_idx = min(start_idx + batch_size, data_points)
                x_batch = X_shuffled[start_idx:end_idx]
                t_batch = T_shuffled[start_idx:end_idx]
                
                y_batch = self.predict(x_batch, scaled=True)
                gradient = np.dot((y_batch - t_batch).T, x_batch) / batch_size
                self.weights -= lr * gradient

    def predict(self, X, scaled=False):
        """
        Predict probabilities for input data X.

        Parameters:
        - X: Input features, shape (N, D)
        - scaled: Whether the input X has already been scaled (default: False)

        Returns:
        - scores: Predicted probabilities, shape (N, K)
        """
        if not scaled:
            X = (X - self.mean) / self.std
        scores = softmax(np.dot(X, self.weights.T))
        return scores

    def error(self, T, Y):
        """
        Compute cross-entropy loss.

        Parameters:
        - T: True labels in one-hot encoding, shape (N, K)
        - Y: Predicted probabilities, shape (N, K)

        Returns:
        - Cross-entropy loss (scalar)
        """
        log = np.log
        return -np.sum(T * log(Y + 1e-9)) / T.shape[0]

    def save(self, path):
        """
        Save the model parameters to a file.

        Parameters:
        - path: File path to save the model
        """
        np.savez(path, weights=self.weights, mean=self.mean, std=self.std)

    @staticmethod
    def load(path):
        """
        Load a model from a file.

        Parameters:
        - path: File path from which to load the model

        Returns:
        - model: An instance of LogisticRegression with loaded parameters
        """
        data = np.load(path)
        model = LogisticRegression()
        model.weights = data['weights']
        model.mean = data['mean']
        model.std = data['std']
        return model
