import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.

    Implemented regularizer. Did not find any improvements from main, with different values of lambda) 
    """

    def __init__(self, lr, max_iters=500, lambda_reg=0.0075):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.lambda_reg = lambda_reg

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
            lambda_reg (float): Regularization strength
        Returns:
            float: Loss value 
    """
        # Calculate the current prediction
        softmax_pred = self.f_softmax(data, w)

        # Cross-entropy loss
        cross_entropy_loss = - np.sum(labels * np.log(softmax_pred))

    # L1 regularization term
        l1_penalty = self.lambda_reg * np.sum(np.abs(w))

        # Total loss is the sum of cross-entropy loss and L1 penalty
        total_loss = cross_entropy_loss + l1_penalty
        return total_loss

    def gradient_loss_function(self, data, labels, W):
        """
        Args:
            data (array): Input of shape (N,D)
            labels (array): Labels of shape (N,)
            W (array): Weights of shape (D,C)
            lambda_reg (float): Regularization strength
        Returns:
            numpy.ndarray: Gradient of loss with respect to W
        """
        # Compute the gradient of the cross-entropy part
        cross_entropy_grad = data.T @ (self.f_softmax(data,
                                       W) - label_to_onehot(labels))

        # Compute the gradient of the L1 penalty term
        # The gradient is lambda_reg times the sign of the weights
        l1_grad = self.lambda_reg * np.sign(W)

        # The total gradient is the sum of cross-entropy gradient and L1 gradient
        total_grad = cross_entropy_grad + l1_grad
        return total_grad

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
        """
        Args:
            data (array): Input of shape (N,D)
            labels (array): Labels of shape (N,)
            lambda_reg (float): Regularization strength
        """
        D = data.shape[1]
        C = get_n_classes(labels)
        weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            predictions = self.logistic_regression_predict(data, weights)
            if accuracy_fn(predictions, labels) >= 100:
                break
            # Include the regularization strength in the gradient calculation
            weights -= self.lr * \
                self.gradient_loss_function(
                    data, labels, weights)

        self.weights = weights
