import numpy as np
import sys
from ..utils import accuracy_fn, get_n_classes


class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, task_kind="regression"):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.weights = None  # Initialize weights as None
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        D = training_data.shape[1]
        if self.weights is None:
            # Properly initialize weights here
            self.weights = np.random.normal(0, 1e-1, D)

        self.weights = self.find_weights(
            training_data, training_labels, epochs=1000, lr=0.01)
        pred_regression_targets = self.predict(training_data)

        return pred_regression_targets

    def predict(self, test_data):
        """
        Make predictions using the linear regression model.

        Arguments:
            test_data (np.array): test data of shape (N,D)
            w (np.array): weight parameters of shape (D,M)
        Returns:
            pred_regression_targets (np.array): predicted labels of shape (N,M)
    """
        # Use the dot product to predict for multi-output regression
        pred_regression_targets = np.dot(test_data, self.weights)
        return pred_regression_targets

    def get_loss(self, w, X_train, y_train, X_test, y_test):
        """
            Calculates the loss on the training and test sets.

            Arguments:
                w (np.array): weight parameters
                X_train (np.array): training data of shape (N,D)
                y_train (np.array): training labels of shape (N,regression_target_size)
                X_test (np.array): test data of shape (N,D)
                y_test (np.array): test labels of shape (N,regression_target_size)
        """
        loss_train = (np.mean((y_train-X_train@w)**2))
        loss_test = (np.mean((y_test-X_test@w)**2))
        print("The training loss is {}. The test loss is {}.".format(
            loss_train, loss_test))

        return loss_test

    def find_gradient(self, X, y, w):
        """
        Computes the gradient of the empirical risk with respect to the weights w.

        Arguments:
            X (np.array): Input data of shape (N, D), where N is the number of samples and D is the number of features.
            y (np.array): Labels of shape (N, M), where M is the number of outputs.
            w (np.array): Weight parameters of shape (D, M).

        Returns:
            grad (np.array): Gradient of the empirical risk of shape (D, M).
        """
        N = X.shape[0]
        predictions = X @ w  # Matrix multiplication, result has shape (N, M)
        errors = predictions - y  # Result has shape (N, M)
        # Matrix multiplication, result has shape (D, M)
        grad = 2 / N * X.T @ errors
        return grad

    def find_weights(self, X_train, y_train, epochs, lr):
        """
            Computes the weight parameters w using gradient descent.

            Arguments:
                X_train (np.array): training data of shape (N,D)
                y_train (np.array): training labels of shape (N,M)
                epochs (int): number of epochs
                lr (float): learning rate
            Returns:
                w (np.array): weight parameters of shape (D,M)
        """
        # Initialize the weights with the correct shape
        # Number of features  # Number of regression targets
        print(y_train.shape)
        D = X_train.shape[1]
        M = y_train.shape[1]

        w = np.random.normal(0, 1e-1, (D, M))

        # Iterate a given number of epochs over the training data
        for i in range(epochs):
            grad = self.find_gradient(X_train, y_train, w)
            w -= lr * grad

        return w
