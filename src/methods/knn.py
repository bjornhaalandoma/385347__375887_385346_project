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
        s1 = time.time()

        self.trainingData = training_data
        self.trainingLabels = training_labels
        # put the two fields here
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

        ##
        ###
        # YOUR CODE HERE!
        ###
        ##

        if self.task_kind == "classification":
            # preparing return variable
            pred_labels = np.zeros(training_data.shape[0], dtype=int)

            # calculating distances and deciding labels
            for row1 in range(training_data.shape[0]):
                distances = list()
                for row2 in range(training_data.shape[0]):
                    if row1 == row2:
                        distances.append(0)
                        continue
                    dist = self.euclidean_distance(
                        training_data[row1], training_data[row2])
                    distances.append(dist)

                sorted_indexes = sorted(
                    range(len(distances)), key=lambda i: distances[i])
                distances.sort()

                Times = [0] * 20
                AverageDist = [0] * 20
                for i in range(self.k + 1):
                    if i == 0:
                        continue
                    label_index = sorted_indexes[i]
                    label = training_labels[label_index]
                    Times[label] += 1
                    AverageDist[label] += distances[i]

                # treat the case when k is larger than labels???

                for i in range(self.k):
                    if Times[i] == 0:
                        continue
                    AverageDist[i] = AverageDist[i] / Times[i]

                maxValue = np.max(Times)
                count_max_value = 0
                for i in range(20):
                    if Times[i] == maxValue:
                        count_max_value += 1

                if count_max_value == 1:
                    pred_labels[row1] = np.argmax(Times)

                else:
                    sol = np.max(AverageDist)
                    for i in range(20):
                        if Times[i] == np.max(Times):
                            if AverageDist[i] < sol:
                                sol = AverageDist[i]
                                index = i
                    pred_labels[row1] = index

            s2 = time.time()
            print(f"The time taken for fit: {s2-s1} ")
            return pred_labels
        else:
            # preparing return variable
            pred_labels = [None] * training_data.shape[0]

            # calculating distances and deciding labels
            for row1 in range(training_data.shape[0]):
                distances = list()
                for row2 in range(training_data.shape[0]):
                    if row1 == row2:
                        distances.append(0)
                        continue
                    dist = self.euclidean_distance(
                        training_data[row1], training_data[row2])
                    distances.append(dist)

                sorted_indexes = sorted(
                    range(len(distances)), key=lambda i: distances[i])
                distances.sort()

                x = 0
                y = 0
                for i in range(self.k + 1):
                    if i == 0:
                        continue
                    x += training_labels[sorted_indexes[i]][0]
                    y += training_labels[sorted_indexes[i]][1]
                x = x / self.k
                y = y / self.k
                # treat the case when k is larger than labels???

                pred_labels[row1] = (x, y)

            s2 = time.time()
            print(f"The time taken for fit: {s2-s1} ")
            return pred_labels

    def euclidean_distance(self, row1, row2):
        # Calculate Euclidean distance between two obejcts
        sum = 0
        for i in range(row1.shape[0]):
            sum += (row1[i] - row2[i]) ** 2
        return np.sqrt(sum)

    def predict(self, test_data):
        s3 = time.perf_counter()
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        # YOUR CODE HERE!
        ###
        ##
        if self.task_kind == "classification":
            test_labels = np.zeros(test_data.shape[0], dtype=int)

            # calculating distances and deciding labels
            for row1 in range(test_data.shape[0]):
                distances = list()
                for row2 in range(self.trainingData.shape[0]):
                    dist = self.euclidean_distance(
                        test_data[row1], self.trainingData[row2])
                    distances.append((row2, dist))
                distances.sort(key=lambda tup: tup[1])

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

                if count_max_value == 1:
                    test_labels[row1] = np.argmax(Times)

                else:
                    sol = np.max(AverageDist)
                    for i in range(20):
                        if Times[i] == np.max(Times):
                            if AverageDist[i] < sol:
                                sol = AverageDist[i]
                                index = i
                    test_labels[row1] = index
            s4 = time.perf_counter()
            print(f"The time taken for prediction: {s4 - s3} ")
            return test_labels

        else:
            # preparing return variable
            test_labels = [None] * test_data.shape[0]

            # calculating distances and deciding labels
            for row1 in range(test_data.shape[0]):
                distances = list()
                for row2 in range(self.trainingData.shape[0]):
                    if row1 == row2:
                        distances.append(0)
                        continue
                    dist = self.euclidean_distance(
                        test_data[row1], self.trainingData[row2])
                    distances.append(dist)

                sorted_indexes = sorted(
                    range(len(distances)), key=lambda i: distances[i])
                distances.sort()
                x = 0
                y = 0
                for i in range(self.k + 1):
                    if i == 0:
                        continue
                    x += self.trainingLabels[sorted_indexes[i]][0]
                    y += self.trainingLabels[sorted_indexes[i]][1]
                x = x / self.k
                y = y / self.k
                # treat the case when k is larger than labels???
                test_labels[row1] = (x, y)
            s4 = time.perf_counter()
            print(f"The time taken for prediction: {s4 - s3} ")
            return test_labels
