import numpy as np
from searches import Problem
import math


class Validator:
    def __init__(self, filename, test_size=0.2) -> None:
        with open(filename, 'r') as file:
            lines = file.readlines()

        self.data = np.array([list(map(float, line.split())) for line in lines])
        #self.train = self.data[:int(len(self.data) * 0.8)]
        #self.test = self.data[int(len(self.data) * 0.2):]

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


        #getting the features for specific input problem parameter
        #adding 1 since first column of data is class
        relevent_features = sampleProblem.bestNode.features
        '''for feature in relevent_features:
            feature += 1'''        
        

        #tracking curr entry to make sure we dont include it with nearest neighbors
        curr_entry = 0

        #these are both meant to be 2d arrays with each array represeenting an entry
        #and each value is the distance/class of one of its neighbors
        neighbor_distances = []
        neighbor_classifications = []

        for i in range(len(self.data)):
            distances = []
            classifications = []
            for j in range(len(self.data)):
                cummulative_error = 0
                #making sure we dont count the current instance for nearest neighbors
                if(j != curr_entry):
                    for feature in relevent_features:
                        cummulative_error += ((self.data[i][feature] - self.data[curr_entry][feature]) ** 2)

                    distances.append(math.sqrt(cummulative_error))
                    classifications.append(self.data[j][0])


            neighbor_distances.append(distances)            
            neighbor_classifications.append[classifications]

            #iterating the current entry so we know which one to not include in training
            curr_entry += 1


        #zipping and sorting lists so the actual knn can find nearest neighbors
        for i in range (len(neighbor_distances)):
             zipped_list = list(zip(neighbor_distances[i], neighbor_classifications[i]))
             sorted_zipped_lists = sorted(zipped_list, key=lambda x: x[0])

             new_distances, new_classifications = zip(*sorted_zipped_lists)

             new_distances = list(new_distances)
             new_classifications = list(new_classifications)

             neighbor_distances[i] = new_distances
             neighbor_classifications[i] = new_classifications

        #this array will store whether the classifier got each instance correct with leavone-out
        #storing 1 for correct and 0 for incorrect
        knn_classifications = []
        for instance in self.data:
            num_class1 = 0
            num_class2 = 0
            for neighbor_index in range(k):
                if(neighbor_classifications[instance][neighbor_index] == 1):
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
        return


test = Validator("data/small-test-dataset.txt")
test.k_fold()
print(test.data)
