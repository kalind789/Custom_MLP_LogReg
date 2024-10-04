import numpy as np
from util import *

class MultilayerPerceptron:
    def __init__(self):
        """
        Multilayer Perceptron with one hidden layer of 10 tanh units.
        Weights and biases are initialized in the fit method.
        """
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.mean = None
        self.std = None

    def fit(self, X, T, epochs=1000, learning_rate=0.01, batch_size=32):
        """
        Train the MLP using mini-batch stochastic gradient descent.

        Parameters:
        - X: Input features, shape (N, D)
        - T: One-hot encoded targets, shape (N, K)
        - epochs: Number of epochs to train (default: 1000)
        - learning_rate: Learning rate for weight updates (default: 0.01)
        - batch_size: Size of each mini-batch (default: 32)
        """
        N, D = X.shape
        K = T.shape[1]
        
        # Feature scaling
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - self.mean) / self.std

        # Initialize weights and biases
        np.random.seed(30)
        self.weights1 = np.random.normal(0, 1, (D, 10))  # 10 hidden units
        self.bias1 = np.zeros(10)
        self.weights2 = np.random.normal(0, 1, (10, K))
        self.bias2 = np.zeros(K)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X_scaled[indices]
            T_shuffled = T[indices]

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                T_batch = T_shuffled[start_idx:end_idx]

                self.backprop(X_batch, T_batch, learning_rate)
        
            # Optionally, compute loss and print
            if epoch % 100 == 0:
                predictions = self.forward(X_scaled)
                loss = self.cross_entropy_loss(predictions, T)
                print(f'Epoch {epoch}, Loss: {loss}')

    def backprop(self, X_batch, T_batch, learning_rate):
        """
        Perform backpropagation and update weights.

        Parameters:
        - X_batch: Mini-batch input data
        - T_batch: Mini-batch target data
        - learning_rate: Learning rate for weight updates
        """
        N = X_batch.shape[0]

        # Forward pass
        hidden_input = self.linear(X_batch, self.weights1, self.bias1)
        hidden_output = self.tanh(hidden_input)
        output = self.linear(hidden_output, self.weights2, self.bias2)
        Y = self.softmax(output)

        # Backward pass
        output_error = Y - T_batch  # (N, K)

        weights2_grad = np.dot(hidden_output.T, output_error) / N
        bias2_grad = np.sum(output_error, axis=0) / N

        hidden_error = np.dot(output_error, self.weights2.T) * (1 - hidden_output ** 2)  # tanh derivative

        weights1_grad = np.dot(X_batch.T, hidden_error) / N
        bias1_grad = np.sum(hidden_error, axis=0) / N

        # Update weights and biases
        self.weights1 -= learning_rate * weights1_grad
        self.bias1 -= learning_rate * bias1_grad
        self.weights2 -= learning_rate * weights2_grad
        self.bias2 -= learning_rate * bias2_grad

    def predict(self, X):
        """
        Predict probabilities for input data X.

        Parameters:
        - X: Input features, shape (N, D)

        Returns:
        - Predicted probabilities, shape (N, K)
        """
        X_scaled = (X - self.mean) / self.std
        return self.forward(X_scaled)

    def forward(self, X):
        """
        Forward pass through the network.

        Parameters:
        - X: Input features, shape (N, D)

        Returns:
        - Output probabilities after softmax, shape (N, K)
        """
        hidden_input = self.linear(X, self.weights1, self.bias1)
        hidden_output = self.tanh(hidden_input)
        output = self.linear(hidden_output, self.weights2, self.bias2)
        return self.softmax(output)

    def linear(self, X, weights, bias):
        return np.dot(X, weights) + bias

    def tanh(self, X):
        return np.tanh(X)

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def cross_entropy_loss(self, Y, T):
        log_Y = np.log(Y + 1e-9)  # To avoid log(0)
        return -np.sum(T * log_Y) / T.shape[0]

    def save(self, path):
        """
        Save the model parameters to a file.

        Parameters:
        - path: File path to save the model
        """
        np.savez(path, weights1=self.weights1, bias1=self.bias1,
                 weights2=self.weights2, bias2=self.bias2,
                 mean=self.mean, std=self.std)

    @staticmethod
    def load(path):
        """
        Load a model from a file.

        Parameters:
        - path: File path from which to load the model

        Returns:
        - model: An instance of MultilayerPerceptron with loaded parameters
        """
        data = np.load(path)
        model = MultilayerPerceptron()
        model.weights1 = data['weights1']
        model.bias1 = data['bias1']
        model.weights2 = data['weights2']
        model.bias2 = data['bias2']
        model.mean = data['mean']
        model.std = data['std']
        return model
