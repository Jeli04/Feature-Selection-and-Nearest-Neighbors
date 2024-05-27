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
        problemObj.greedy_forward_search()
    elif (algo == 2):
        problemObj.greedy_backward_search()

if __name__ == '__main__':
    #main()
    #main()
    test = Classifier(filename="../data/small-test-dataset.txt")
    print("Test Point: ", test.data[1][2])
    #third parameter = best feature subset from greedy algorithm
    print("Using features ", "(2, 3, 4)")
    test.nearestNeighbor(test.train, test.test[0], [2,3,4])