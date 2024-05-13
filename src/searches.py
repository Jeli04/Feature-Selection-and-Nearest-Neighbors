import random as rand
import copy

def eval(feature):
    return rand.randint(0, 10)

def greedy_forward_search(features):
    sorted_features = {feature:eval(feature) for feature in features}    # feature to eval
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

values = [1,2,3,4,5]
print(greedy_forward_search(values))

# print(eval(1))