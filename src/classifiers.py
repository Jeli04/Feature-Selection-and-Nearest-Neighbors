import numpy as np
import math

class Classifier:
  def __init__(self, filename, test_size=0.2) -> None:
    with open(filename, 'r') as file:
        lines = file.readlines()

    self.data = np.array([list(map(float, line.split())) for line in lines])
    split_index = int(len(self.data) * 0.2)
    self.train = self.data[:-split_index]
    self.test = self.data[-split_index:]
    # self.train = self.data[:int(len(self.data) * 0.8)]
    # self.test = self.data[int(len(self.data) * test_size):]

  def euclideanDistance(self, point1, point2):    
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

  def nearestNeighbor(self, data, testInstance, featureSubset):
    minDistance = 1000
    nearestRowIndex = 0 # row index = nearest neighbor

    testPoint = testInstance[featureSubset]
    newPoint = []
    # print("Best Feature Subset: ", testPoint)
    # print("Class Instance: ", testInstance)
    
    for i in range(len(data)):
      print("\nInstance ", i, "is class ", data[i][0])
      newPoint = data[i][featureSubset]
      
      # convert each point to tuples to pass into euclideanDistance
      testPoint = tuple(testPoint)
      newPoint = tuple(newPoint)
      
      distance = self.euclideanDistance(testPoint, newPoint)

      # update distance and nearest neighbor when new minimum is found
      if distance < minDistance:
        minDistance = distance
        nearestRowIndex = i # nearest instance
    
      print("Its nearest neighbor is ", data[nearestRowIndex][featureSubset], "which is in class", data[nearestRowIndex][0])
      print("The min distance is ", minDistance)
    return data[nearestRowIndex][0] # return the class label of the nearest neighbor

# Using features (1,4,5)


# Instance 1 is class 2
# Its nearest neighbor is 5 which is in class 2
# Instance 2 is class 1
# Its nearest neighbor is 3 which is in class 1
# Instance 3 is class 1
# Its nearest neighbor is 4 which is in class 1
# Instance 4 is class 1
# Its nearest neighbor is 3 which is in class 1
# Instance 5 is class 2
# Its nearest neighbor is 3 which is in class 1

# Accuracy  = 0.8