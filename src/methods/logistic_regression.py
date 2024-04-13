import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        self.gradient_descent(training_data, training_labels)
        pred_labels = self.logistic_regression_predict(
            training_data, self.weights)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = self.logistic_regression_predict(test_data, self.weights)

        return pred_labels

    def f_softmax(self, data, W):
        """
        Args:
            data (array): Input of shape (N,D)
            W (array): Weights of shape (D,C) where C is the number of classes 
        Returns:
            array of shape (N,C)
        """
        return np.exp(data @ W) / np.sum(np.exp(data @ W), axis=1)[:, np.newaxis]

    # TODO Remove? Not using currently, but can be used for visualization
    def loss_function(self, data, labels, w):
        """
        Args:
            data (array): Input of shape (N,D)
            labels (array): Labels of shape (N,)
            w (array): Weights of shape (D,C)
        Returns:
            float: Loss value 
        """
        return - np.sum(labels * np.log(self.f_softmax(data, w)))

    def gradient_loss_function(self, data, labels, W):
        """
        Args:
            data (array): Input of shape (N,D)
            labels (_type_): Labels of shape (N,)
            W (_type_): _description_
        """

        return data.T @ (self.f_softmax(data, W) - label_to_onehot(labels))

    def logistic_regression_predict(self, data, W):
        """
        Args:
            data (array): Dataset of shape (N,D). Contains N datasamples of dimension D
            W (array): Weights of regression model of shape (D,C).
        Return:
            array of shape (N,): Labels predictions of data 
        """
        return np.argmax(self.f_softmax(data, W), axis=1)

    def gradient_descent(self, data, labels):
        """_summary_

        Args:
            data (_type_): _description_
            labels (_type_): _description_
        """
        D = data.shape[1]
        C = get_n_classes(labels)
        weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            predictions = self.logistic_regression_predict(data, weights)
            if accuracy_fn(predictions, labels) >= 80:
                break
            weights -= self.lr * \
                self.gradient_loss_function(data, labels, weights)

        self.weights = weights
