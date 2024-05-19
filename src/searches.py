import heapq
import random as rand

class Node:
    def __init__(self, parent):
        self.accuracy = self.eval() # calls eval 
        self.parent = parent
        self.currentFeatures = [] # forward + backward
        self.remainingFeatures = [] # forward

    def printInfo(self):
        print("Accuracy: " + str(self.accuracy))
        print("Parent: " + str(self.parent))
        print("Current Features: " + str(self.currentFeatures))
        print("Remaining Features: " + str(self.remainingFeatures))

    def eval(self):
        return rand.uniform(0, 1) * 100

class Problem:
  def __init__(self, features) -> None:
    self.nodequeue = [] # max priority queue
    self.features = features
    self.seen = []
    self.overallMaxAccuracy = 0
  
  def greedy_backward_search(self):
    # initialize node with all the features 
    root = Node(None) 
    root.currentFeatures = self.features # all the features
    heapq.heappush(self.nodequeue, (-root.accuracy, root)) # add root to the queue (sort by max accuracy)

    # add root to seen list
    if root:
      self.seen.append(root.currentFeatures)
    else:
      ValueError("root is None!!")

    parent = root
  
    # while there are still features to be eliminated
    while parent.currentFeatures:
      pair = heapq.heappop(self.nodequeue) # pair is (-accuracy, node)
      currMaxAccuracy = -pair[0] # make it positive (double negative)
      self.overallMaxAccuracy = max(self.overallMaxAccuracy, currMaxAccuracy)
      
      for feature in parent.currentFeatures:
        newNode = Node(parent)
        newNode.currentFeatures = newNode.parent.currentFeatures # give child node currentFeatures of its parent
        newNode.currentFeatures.remove(feature) # remove one feature for child node

        # print accuracy of each feature
        
        #skip node if seen
        if newNode.currentFeatures in self.seen:
          continue

        #push each node into heap, where each node is a diff combination of features
        heapq.heappush(self.nodequeue, (-newNode.accuracy, newNode))
        self.seen.append(newNode.currentFeatures) # add to seen list


      #compare accuracies of current subset and output best accuracy
        
  
  def greedy_forward_search(self):
    sorted_features = {feature:eval(feature) for feature in self}    # feature to eval
    sorted_features = sorted(sorted_features.items(), key=lambda item: item[1])
    print("sorted:", sorted_features) # debug line
    result = [{}]
    curr_val = 0
    next_feature = sorted_features.pop()
    i = 0
    while curr_val < curr_val + next_feature[1]:
        cpy = copy.deepcopy(result[i])
        cpy[next_feature[0]] = next_feature[1]
        result.append(cpy)
        curr_val += next_feature[1]
        if len(sorted_features) > 0: next_feature = sorted_features.pop()
        else: break
        i+=1
  
    return result