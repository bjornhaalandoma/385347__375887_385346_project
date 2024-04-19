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

    def __init__(self, lmda, task_kind="regression", lr=0.01, epochs=400):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = task_kind
        self.lr = lr
        self.epochs = epochs

    def fit(self, training_data, training_labels):

        s1 = time.time()

        self.weights = self.find_weights(training_data, training_labels)
        pred_regression_targets = self.predict(training_data)

        s2 = time.time()

        print(f"Time taken for training: {s2-s1}")

        return pred_regression_targets

    def predict(self, test_data):
        """
        Make predictions using the linear regression model.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_regression_targets (np.array): predicted labels of shape (N,M)

    """
        s3 = time.perf_counter()
        # Use the dot product to predict for multi-output regression
        pred_regression_targets = np.dot(test_data, self.weights)
        s4 = time.perf_counter()
        print(f"Time taken for prediction: {s4-s3}")
        return pred_regression_targets

    # TODO: Remove? Currently not using

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
        N = X.shape[0]
        pred_error = y - np.dot(X, w)
        grad_W = (-2/N) * np.dot(X.T, pred_error) + 2 * self.lmda * w
        return grad_W

    def find_weights(self, X_train, y_train):

        D = X_train.shape[1]
        M = y_train.shape[1]
        w = np.random.normal(0, 1e-5, (D, M))

        for i in range(self.epochs):
            gradient = self.find_gradient(X_train, y_train, w)
            # Implement gradient clipping
            w -= self.lr * gradient

        return w
