import numpy as np
import sys
from ..utils import accuracy_fn, get_n_classes, append_bias_term
import time


class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, task_kind="regression", lr=0.1, epochs=1000):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = task_kind
        self.lr = lr
        self.epochs = epochs

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Args:
            training_data (array): training data of shape (N,D)
            training_labels (array): training labels of shape (N,M)

        Returns:
            pred_regression_targets (array): predicted labels of shape (N,M)
        """

        self.weights = self.find_weights(training_data, training_labels)
        pred_regression_targets = self.predict(training_data)

        return pred_regression_targets

    def predict(self, test_data):
        """
        Make predictions using the linear regression model.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_regression_targets (np.array): predicted labels of shape (N,M)

    """
        pred_regression_targets = np.dot(test_data, self.weights)

        return pred_regression_targets

    def find_gradient(self, X, y, w):
        """
        Find the gradient of the loss function.

        Args:
            X (array): data of shape (N,D)
            y (array): targets of shape (N,M)
            w (array): weights of shape (D,M)

        Returns:
            grad_W (array): gradient of the loss function
        """
        N = X.shape[0]
        pred_error = y - np.dot(X, w)
        grad_W = (-2/N) * np.dot(X.T, pred_error) + 2 * self.lmda * w
        return grad_W

    def find_weights(self, X_train, y_train):
        """
        Find the weights of the linear regression model using gradient descent.

        Args:
            X_train (array): data of shape (N,D)
            y_train (array): targets of shape (N,M)

        Returns:
            w (array): weights of shape (D,M)
        """

        D = X_train.shape[1]
        M = y_train.shape[1]
        w = np.random.normal(0, 1e-5, (D, M))

        # Starting with with bigger learning
        for i in range(self.epochs):
            gradient = self.find_gradient(X_train, y_train, w)
            w -= self.lr * gradient

        # Decreasing the learning rate as we get closer
        newlr = (self.lr / 100)
        for i in range(1000):
            gradient = self.find_gradient(X_train, y_train, w)
            w -= newlr * gradient

        return w
