import numpy
import collections
import math

# Calcurate entorpy reduction by each feature
def calc_entropy_reduction(value_array, flag_array, item_ids=[], descs=[], units=[]):
    
    orig_entropy = entropy(flag_array)
    print "Original entropy: %f"%orig_entropy
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
