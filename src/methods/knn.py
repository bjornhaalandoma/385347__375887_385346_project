import time

import numpy as np


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.trainingData = None
        self.trainingLabels = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        # We save the training data and training labels as fields in our object for later use
        self.trainingData = training_data
        self.trainingLabels = training_labels

        # If else condition for classification adn regression cases
        if self.task_kind == "classification":
            return self.auxiliary_classification(training_data)
        else:
            return self.auxiliary_regression(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        # If else condition for classification adn regression cases
        if self.task_kind == "classification":
            return self.auxiliary_classification(test_data)
        else:
            return self.auxiliary_regression(test_data)

    def euclidean_distance(self, row1, row2):
        # Calculate Euclidean distance between two objects
        sum = 0
        for i in range(row1.shape[0]):
            sum += (row1[i] - row2[i]) ** 2
        return np.sqrt(sum)

    def auxiliary_classification(self, data):
        # Create future return variable
        test_labels = np.zeros(data.shape[0], dtype=int)

        # Double loop to find k nearest neighbors
        for row1 in range(data.shape[0]):
            distances = list()
            for row2 in range(self.trainingData.shape[0]):
                dist = self.euclidean_distance(
                    data[row1], self.trainingData[row2])
                distances.append((row2, dist))
            distances.sort(key=lambda tup: tup[1])

            # Create histogram for appearances and average distances per label
            Times = [0] * 20
            AverageDist = [0] * 20
            for i in range(self.k):
                Times[int(self.trainingLabels[distances[i][0]])] += 1
                AverageDist[self.trainingLabels[distances[i][0]]
                            ] += distances[i][1]
            for i in range(self.k):
                if Times[i] == 0:
                    continue
                AverageDist[i] = AverageDist[i] / Times[i]
            maxValue = np.max(Times)
            count_max_value = 0
            for i in range(20):
                if Times[i] == maxValue:
                    count_max_value += 1

            # If there is only one label that appears most times for k neighbors it will be the selected label
            if count_max_value == 1:
                test_labels[row1] = np.argmax(Times)

            # If not we choose the label that has the lowest average distance from our sample
            else:
                sol = np.max(AverageDist)
                for i in range(20):
                    if Times[i] == np.max(Times):
                        if AverageDist[i] < sol:
                            sol = AverageDist[i]
                            index = i
                test_labels[row1] = index

        return test_labels

    def auxiliary_regression(self, data):
        # create future return variable
        test_labels = [None] * data.shape[0]

        # Double loop to find k nearest neighbors
        for row1 in range(data.shape[0]):
            distances = list()
            for row2 in range(self.trainingData.shape[0]):
                if row1 == row2:
                    distances.append(0)
                    continue
                dist = self.euclidean_distance(
                    data[row1], self.trainingData[row2])
                distances.append(dist)

            # Calculate average x and y of k nearest neighbors
            sorted_indexes = sorted(
                range(len(distances)), key=lambda i: distances[i])
            distances.sort()
            x = 0
            y = 0
            for i in range(self.k):
                x += self.trainingLabels[sorted_indexes[i]][0]
                y += self.trainingLabels[sorted_indexes[i]][1]
            x = x / self.k
            y = y / self.k
            # treat the case when k is larger than labels???
            # limited number of samples?
            test_labels[row1] = (x, y)
        return test_labels
