from searches import Problem

def main():
    print("Welcome to the Feature Selection Algorithm.")
    
    numFeatures = int(input("\nPlease enter total number of features: "))
    list = range(1, numFeatures)
    
    print("\n1 - Forward Selection")
    print("2 - Backward Elimination")
    algo = int(input("Type the number of the algorithm you want to run: "))
    print("\nBeginning search.")
    
    problemObj = Problem(list)
    
    if (algo == 1):
        problemObj.greedy_forward_search()
    elif (algo == 2):
        problemObj.greedy_backward_search()

if __name__ == '__main__':
    main()