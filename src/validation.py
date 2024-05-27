import numpy as np
from classifiers import Classifier
import math


class Validator:
    def __init__(self, filename, test_size=0.2) -> None:
        with open(filename, 'r') as file:
            lines = file.readlines()

        self.data = np.array([list(map(float, line.split())) for line in lines])


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
        print(len(k_folds))
        for i in range(k):
            output = []
            testing_set = k_folds[i]
            # print("Testing set ", testing_set)
            testing_labels = np.array(testing_set)[:, 0]
            training_set = k_folds[:i] + k_folds[i+1:]
            training_set = np.array(training_set).reshape(-1, 10)
            print(training_set)
            # print(training_set)
            for instance in testing_set:
                output.append(classifier.nearestNeighbor(training_set, instance, feature_set))
            
            accuracy = [output[j] == testing_labels[j] for j in range(len(output))]
            accuracy = np.mean(accuracy)
            print("Accuracy: ", accuracy)

        

        #this array will store whether the classifier got each instance correct with leavone-out
        #storing 1 for correct and 0 for incorrect
        '''knn_classifications = []
        for instance in self.data:
            num_class1 = 0
            num_class2 = 0
            for neighbor_index in range(k):
                if(output[instance][neighbor_index] == 1):
                    num_class1 += 1

                else:
                    num_class2 += 1

            if(num_class1 > num_class2) and (instance[0] == 1):
                knn_classifications.append(1)
                
            elif(num_class2 > num_class1) and (instance[0] == 2):
                knn_classifications.append(1)

            else:
                knn_classifications.append(0)

        validator_accuracy = sum(knn_classifications) / len(knn_classifications)

        return validator_accuracy

    def eval(self):
        return'''


# test = Validator("data/small-test-dataset.txt")
# test_classifier = Classifier("data/small-test-dataset.txt")

# test.k_fold(test_classifier, [2], 5)
from searches import Problem

problemObj = Problem([1,2,5])
bestSet = problemObj.greedy_backward_search()
print("Using features ", bestSet)
test = Validator("data/small-test-dataset.txt")
test_classifier = Classifier("data/small-test-dataset.txt")
test.k_fold(test_classifier, [2], 5)