import numpy as np
from classifiers import Classifier
import math

class Validator:
    def __init__(self, filename) -> None:
        with open(filename, 'r') as file:
            lines = file.readlines()

        self.data = np.array([list(map(float, line.split())) for line in lines])

    def leave_one_out(self, classifier, feature_set):
        #applying min-max normalization to every feature
        for i in range(1, self.data.shape[1]):
            col = self.data[:, i]
        
            # Calculate min and max
            col_min = np.min(col)
            col_max = np.max(col)
        
            # Apply min-max normalization
            col_normalized = (col - col_min) / (col_max - col_min)
        
            # Replace the original column with the normalized column
            self.data[:, i] = col_normalized

        accuracy = []
        for i in range(len(self.data)):
            validator = self.data[i] 
            output = classifier.nearestNeighbor(np.concatenate((self.data[:i], self.data[i+1:])), validator, feature_set)
            accuracy.append(int(output == validator[0]))

        return np.mean(accuracy)

    def k_fold(self, classifier, feature_set, k):
        #applying min-max normalization to every feature
        for i in range(1, self.data.shape[1]):
            col = self.data[:, i]
        
            # Calculate min and max
            col_min = np.min(col)
            col_max = np.max(col)
        
            # Apply min-max normalization
            col_normalized = (col - col_min) / (col_max - col_min)
        
            # Replace the original column with the normalized column
            self.data[:, i] = col_normalized


        kfold_size = int(len(self.data) / k)
        k_folds = [self.data[j:j+kfold_size] for j in range(0, len(self.data), kfold_size)]

        for i in range(k):
            output = []
            testing_set = k_folds[i]
            testing_labels = np.array(testing_set)[:, 0]
            training_set = np.array(k_folds[:i] + k_folds[i+1:]).reshape(-1, len(testing_set[0]))

            for instance in testing_set:
                output.append(classifier.nearestNeighbor(training_set, instance, feature_set))
            
            accuracy = [output[j] == testing_labels[j] for j in range(len(output))]
            accuracy = np.mean(accuracy)
            print("Accuracy: ", accuracy)

        
# from searches import Problem

# problemObj = Problem([1,2,5])
# bestSet = problemObj.greedy_backward_search()
# print("Using features ", bestSet)
# test = Validator("data/small-test-dataset.txt")
# test_classifier = Classifier()
# test.k_fold(test_classifier, [3,5,7], 5)