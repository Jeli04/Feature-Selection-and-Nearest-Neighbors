import numpy as np
import math
import heapq

class Classifier:
  def __init__(self) -> None:
      pass

  def euclideanDistance(self, point1, point2):    
    if isinstance(point1, (list, tuple)) and len(point1) == 1 and isinstance(point1[0], np.ndarray):
      point1 = point1[0]
    if isinstance(point2, (list, tuple)) and len(point2) == 1 and isinstance(point2[0], np.ndarray):
      point2 = point2[0]

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

  def nearestNeighbor(self, data, testInstance, featureSubset):
    minDistance = float('inf')
    nearestRowIndex = 0 # row index = nearest neighbor

    testPoint = testInstance[featureSubset]

    for i in range(len(data)):
      # print("Data point", data[i])
      newPoint = data[i][featureSubset]

      # convert each point to tuples to pass into euclideanDistance
      # testPoint = tuple(testPoint)
      # newPoint = tuple(newPoint)

      distance = self.euclideanDistance([testPoint], [newPoint])

      # update distance and nearest neighbor when new minimum is found
      if distance < minDistance:
        minDistance = distance
        nearestRowIndex = i # nearest instance
    
    # print("Its nearest neighbor is training instance", nearestRowIndex, "which is in class", data[nearestRowIndex][0])
    # print("The min distance is ", minDistance)
    return data[nearestRowIndex][0] # return the class label of the nearest neighbor


  def kNearestNeighbor(self, k, data, testInstance, featureSubset):
    minheap = []

    testPoint = testInstance[featureSubset]

    for i in range(len(data)):
      newPoint = data[i][featureSubset]

      distance = self.euclideanDistance([testPoint], [newPoint])

      heapq.heappush(minheap, (distance, data[i][0]))

    kNNList= []
    for i in range(k):
      #print(heapq.heappop(minheap))
      #finds the k nearest elements popped from the heap
      kNNList.append(heapq.heappop(minheap)[1])
    majorityClass = max(set(kNNList), key = kNNList.count)
    return majorityClass
