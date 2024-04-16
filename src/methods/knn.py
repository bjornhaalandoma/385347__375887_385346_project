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
        self.trainingData = 0
        self.trainingLabels = 0

    def fit(self, training_data, training_labels):

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

        # preparing return variable
        pred_labels = [0] * training_data.shape[0]

        # calculating distances and deciding labels
        for row1 in training_data:
            distances = list()
            for row2 in training_data:
                if row1 == row2:
                    continue
                dist = self.euclidean_distance(
                    training_data[row1], training_data[row2])
                distances.append((row2, dist))
            distances.sort(key=lambda tup: tup[1])

            Times = [0] * 20
            AverageDist = [0] * 20
            for i in range(self.k):
                Times[training_labels[distances[i][0]]] += 1
                AverageDist[training_labels[distances[i][0]]
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
                pred_labels[row1] = np.argmax(Times)

            else:
                sol = np.max(AverageDist)
                for i in range(20):
                    if Times[i] == np.max(Times):
                        if AverageDist[i] < sol:
                            sol = AverageDist[i]
                            index = i
                pred_labels[row1] = index

        return pred_labels

    def euclidean_distance(self, row1, row2):
        # Calculate Euclidean distance between two obejcts
        sum = 0
        for i in range(row1.shape[0]):
            sum += (row1[i] - row2[i]) ** 2
        return np.sqrt(sum)

    def predict(self, test_data):
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

        test_labels = [0] * test_data.shape[0]

        # calculating distances and deciding labels
        for row1 in test_data:
            distances = list()
            for row2 in self.trainingData:
                dist = self.euclidean_distance(
                    test_data[row1], self.trainingData[row2])
                distances.append((row2, dist))
            distances.sort(key=lambda tup: tup[1])

            Times = [0] * 20
            AverageDist = [0] * 20
            for i in range(self.k):
                Times[self.trainingLabels[distances[i][0]]] += 1
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

        return test_labels
