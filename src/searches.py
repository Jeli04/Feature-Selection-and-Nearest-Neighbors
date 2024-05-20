import heapq
import random as rand
import copy

class Node:
    def __init__(self, parent):
        self.accuracy = 0.0 # calls eval 
        self.parent = parent
        self.currentFeatures = [] # forward + backward (Passed in features through main)

    def printInfo(self):
      print("Using feature(s) ", self.currentFeatures," accuracy is ", round(self.accuracy, 2))


class Problem:
    def __init__(self, features) -> None:
      self.nodequeue = [] #this will be a max heapq
      self.features = features
      self.seen = []
      self.overallMaxAccuracy = 0
      self.bestNode = Node(self)

    def eval(self):
        return rand.uniform(0, 1) * 100
    
    def greedy_backward_search(self):
      #initialize node with all the features 
      root = Node(None) 
      root.accuracy = self.eval()
      root.currentFeatures = self.features # all the features

      # add root to the queue (sort by max accuracy)
      heapq.heappush(self.nodequeue, (-root.accuracy, root)) 

      # add root to seen list
      if root:
        self.seen.append(root.currentFeatures)
        #print("Using all feature(s) ",root.currentFeatures," accuracy is ",round(root.accuracy, 2))
        root.printInfo()
      else:
        ValueError("root is None!!")

      
      parent = root
      currMaxAccuracy = 0.01
      bestFeatures = None
      
    
      #while there are still features to be eliminated
      while parent.currentFeatures:
        pair = heapq.heappop(self.nodequeue) # pair is (-accuracy, node)
        currMaxAccuracy = -pair[0] # make it positive (double negative)
        parent = pair[1]
        self.overallMaxAccuracy = max(self.overallMaxAccuracy, currMaxAccuracy)
        
        childMax = 0
        if(self.overallMaxAccuracy == currMaxAccuracy):
          self.bestNode = pair[1]
        
        for feature in parent.currentFeatures:
          newNode = Node(parent)
          newNode.accuracy = self.eval()
          childMax = max(childMax, newNode.accuracy)
          # give child node currentFeatures of its parent
          newNode.currentFeatures = list(newNode.parent.currentFeatures)
          newNode.currentFeatures.remove(feature) # remove one feature for child node

          # print accuracy of each feature
          newNode.printInfo()
          
          #skip node if seen
          if newNode.currentFeatures in self.seen:
            continue

          #push each node into heap, where each node is a diff combination of features
          heapq.heappush(self.nodequeue, (-newNode.accuracy, newNode))
          self.seen.append(newNode.currentFeatures) # add to seen list
          
          
          if(newNode.accuracy >= self.overallMaxAccuracy):
            self.bestNode = newNode
            self.overallMaxAccuracy = newNode.accuracy
            

        bestFeatures = self.bestNode.currentFeatures

        if childMax<parent.accuracy:
          print("Warning! accuracy has decreased!So we stop searching..")
          print("Feature set ", bestFeatures, " was best, accuracy is ", round(self.overallMaxAccuracy, 2), "%")
          break
        #compare accuracies of current subset and output best accuracy
        print("Feature set ", bestFeatures, " was best, accuracy is ", round(self.overallMaxAccuracy, 2), "%")
  
    def greedy_forward_search(self):
        sorted_features = {feature:eval(str(feature)) for feature in self.features}    # feature to eval
        sorted_features = sorted(sorted_features.items(), key=lambda item: item[1])
        print("sorted:", sorted_features) # debug line
        
        result = [{}]
        curr_val = 0 #current total evaluation for the subset
        next_feature = sorted_features.pop() #pops off highest accuracy value
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