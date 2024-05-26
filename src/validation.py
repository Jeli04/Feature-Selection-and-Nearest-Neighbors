import numpy as np
from searches import Problem
import math


class Validator:
    def __init__(self, filename, test_size=0.2) -> None:
        with open(filename, 'r') as file:
            lines = file.readlines()

        self.data = np.array([list(map(float, line.split())) for line in lines])
        self.train = self.data[:int(len(self.data) * 0.8)]
        self.test = self.data[int(len(self.data) * 0.2):]

    def k_fold(self, sampleProblem, k):


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


        #getting the features for whatever soecific problem
            
        #adding 1 since first column of data is class
        relevent_features = sampleProblem.bestNode.features
        for feature in relevent_features:
            feature += 1

        #storing the distances and class of each data point relative to instance of intererst
        distances = []
        classifications = []

        #tracking curr entry to make sure we dont include it with nearest neighbors
        curr_entry = 0

        
        for i in range(len(self.data)):
            cummulative_error = 0
            for j in range(len(self.data)):
                #making sure we dont count the current instance for nearest neighbors
                if(j != curr_entry):
                    for feature in relevent_features:
                        cummulative_error += ((self.data[i][feature] - self.data[curr_entry][feature]) ** 2)

            curr_entry += 1
            distances.append(math.sqrt(cummulative_error))
            classifications.append[self.data[i][0]]


        return 

    def eval(self):
        return


test = Validator("data/small-test-dataset.txt")
test.k_fold()
print(test.data)
