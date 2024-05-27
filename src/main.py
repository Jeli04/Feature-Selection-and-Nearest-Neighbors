from searches import Problem
from classifiers import Classifier

import time

def main():
    print("Welcome to the Feature Selection Algorithm.")
    
    numFeatures = int(input("\nPlease enter total number of features: "))
    my_list = list(range(1, numFeatures+1))
    
    print("\n1 - Forward Selection")
    print("2 - Backward Elimination")
    algo = int(input("Type the number of the algorithm you want to run: "))
    print("\nBeginning search.\n")
    
    problemObj = Problem(my_list)
    
    start_time_feature = time.time()
    
    if (algo == 1):
        bestSet = list(problemObj.greedy_forward_search())
    elif (algo == 2):
        bestSet = problemObj.greedy_backward_search()
    
    end_time_feature = time.time()
    elasped_feature = end_time_feature - start_time_feature
    print("\nFeature Selection Elapsed Time:", elasped_feature, "seconds")
    
    print("Using features ", bestSet)
    start_time_classification = time.time()
    dataset= Classifier(filename="./data/small-test-dataset.txt")
    dataset.testTheData(dataset.test, bestSet)
    
    end_time_classification = time.time()
    elapsed_classification = end_time_classification - start_time_classification
    print("\nClassification Elapsed Time:", elapsed_classification, "seconds")

if __name__ == '__main__':
    main()
    