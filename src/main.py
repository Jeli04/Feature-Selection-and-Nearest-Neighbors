from searches import greedy_forward_search
from searches import greedy_backward_search

def main():
    print("Welcome to the Feature Selection Algorithm.")
    
    numFeatures = int(input("\nPlease enter total number of features: "))
    list = range(1, numFeatures)
    
    print("\n1 - Forward Selection")
    print("2 - Backward Elimination")
    algo = int(input("Type the number of the algorithm you want to run: "))
    print("\nBeginning search.")
    
    if (algo == 1):
        greedy_forward_search(list)
    elif (algo == 2):
        greedy_backward_search(list) # need to implement

if __name__ == '__main__':
    main()