import numpy
import collections
import math


def select_feature_index(data_x, data_y, n_select):
    entropy_reduction = calc_entropy_reduction(data_x, data_y)
    select_feature_index = [item[1] for item in entropy_reduction[0:n_select]]
    return select_feature_index
    
def mean_entropy_reduction(reduction_results):
    base_result = reduction_results[0]
    mean_ids = [item[2] for item in base_result]
    mean_scores = []
    
    for result in reduction_results:
        item_ids = [item[2] for item in result]
        
        if mean_ids == item_ids:
            mean_scores.append([item[0] for item in result])
        else:
            raise ValueError

    mean_score = numpy.mean(numpy.array(mean_scores),0)

    mean_result = []
    for index in range(len(base_result)):
        mean_reduction = (mean_score[index],
                          base_result[index][1],
                          base_result[index][2],
                          base_result[index][3],
                          base_result[index][4],
                          )
        mean_result.append(mean_reduction)

    mean_result.sort(reverse = True)
    return mean_result
    
# Calcurate entorpy reduction by each feature
def calc_entropy_reduction(value_array, flag_array, item_ids=[], descs=[], units=[]):
    
    orig_entropy = entropy(flag_array)
#    print "Original entropy: %f"%orig_entropy
    result = [];
    for index in xrange(value_array.shape[1]):
        opt_entropy, threshold =  entropy_after_optimal_divide(flag_array, value_array[:,index])

        if len(item_ids) > 0 and len(descs) > 0 and len(units) > 0:
            reduction = ( orig_entropy - opt_entropy,
                          index,
                          item_ids[index],
                          descs[index],
                          units[index]
                          )
        else:
            reduction = ( orig_entropy - opt_entropy,
                          index
                          )
        result.append(reduction)
        
    result.sort(reverse = True)
    return result


def entropy(flags):
    counter =  collections.Counter(flags)

    if len(counter) > 2:
        raise ValueError("Flags should be binary values")

    if counter[0] == 0 or counter[1] == 0:
        entropy = 0.
    else:
        pi = float(counter[0]) / float(counter[0] + counter[1])
        entropy =  - (pi * math.log(pi,2) + (1. - pi) * math.log(1. - pi, 2))
    return entropy

def entropy_after_divide(flag, value, threshold):
    flag_r = flag[value <= threshold]
    flag_l = flag[value > threshold]
    p_r = float(len(flag_r)) / float(len(flag))
    p_l = float(len(flag_l)) / float(len(flag))

    return (p_r * entropy(flag_r)) + (p_l * entropy(flag_l))

def entropy_after_optimal_divide(flag, value):
    min_entropy = numpy.inf
    opt_th = 0
    for item in value:
        t_entropy = entropy_after_divide(flag, value, item)
        if t_entropy < min_entropy:
            opt_th = item
            min_entropy = t_entropy

    return min_entropy, opt_th
