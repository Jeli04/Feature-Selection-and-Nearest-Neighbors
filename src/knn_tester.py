from searches import Problem
from classifiers import Classifier
from validation import Validator

import time

def main():
    print("Welcome to the Feature Selection Algorithm.")
    
    numFeatures = int(input("\nPlease enter total number of features: "))
    my_list = list(range(1, numFeatures+1))
    
    # initialize our objects
    validator = Validator("data/CS170_Spring_2024_Large_data__16.txt")
    problemObj = Problem(my_list)
    classifier = Classifier()

    print("\n1 - Forward Selection")
    print("2 - Backward Elimination")
    algo = int(input("Type the number of the algorithm you want to run: "))
    print("\nBeginning search.\n")
    
    accuracies = []

    for k in range(1, 7, 2):
        start_time_feature = time.time()
        print(f"\nPerforming Feature Selection on {k}-NN")

        if (algo == 1):
            bestSet = list(problemObj.greedy_forward_search(classifier, validator,k=k))
        elif (algo == 2):
            bestSet = problemObj.greedy_backward_search(classifier, validator,k=k)
        
        end_time_feature = time.time()
        elasped_feature = end_time_feature - start_time_feature
        print("\nFeature Selection Elapsed Time:", elasped_feature, "seconds")
        
        print("Using features ", bestSet)
        
        start_time_classification = time.time()

        # validator.k_fold(classifier, bestSet, k=5)
        accuracies.append(validator.leave_one_out(classifier, bestSet, k=k))  

        end_time_classification = time.time()
        elapsed_classification = end_time_classification - start_time_classification
        print(f"\nClassification Elapsed Time: {elapsed_classification} seconds on {k}-NN")
        print("------------------------------------------------------------------------------------")

    print(accuracies)

if __name__ == '__main__':
    main()
    