from searches import Problem
from classifiers import Classifier

def main():
    print("Welcome to the Feature Selection Algorithm.")
    
    numFeatures = int(input("\nPlease enter total number of features: "))
    my_list = list(range(1, numFeatures+1))
    
    print("\n1 - Forward Selection")
    print("2 - Backward Elimination")
    algo = int(input("Type the number of the algorithm you want to run: "))
    print("\nBeginning search.\n")
    
    problemObj = Problem(my_list)
    
    if (algo == 1):
        bestSet = list(problemObj.greedy_forward_search())
    elif (algo == 2):
        bestSet = problemObj.greedy_backward_search()
        
    print("Using features ", bestSet)
    dataset= Classifier(filename="./data/small-test-dataset.txt")
    dataset.testTheData(dataset.test, bestSet)

if __name__ == '__main__':
    main()
    