import numpy as np
import collections
import math
from get_sample import point_data


def example(source_num):
    [x, y] = point_data.sample(source_num)
    entropy_reduction = calc_entropy_reduction(x, y)
    print entropy_reduction


def calc_entropy_reduction(x, y, item_ids=[], descs=[], units=[]):
    '''Calcurate entropy reduction with each feature.
    :param x: feature matrix
    :param y: label
    '''
    x[np.isnan(x)] = 0.

    orig_entropy = entropy(y)
    result = []
    for index in xrange(x.shape[1]):
        opt_entropy, thres = entropy_after_optimal_divide(y, x[:, index])

        if len(item_ids) > 0 and len(descs) > 0 and len(units) > 0:
            reduction = (orig_entropy - opt_entropy, index,
                         item_ids[index], descs[index], units[index])
        else:
            reduction = (orig_entropy - opt_entropy, index)

        result.append(reduction)
    result.sort(reverse=True)
    return result


def entropy(y):
    counter = collections.Counter(y)
    if len(counter) > 2:
        raise ValueError("Flags should be binary values")
    if counter[0] == 0 or counter[1] == 0:
        entropy = 0.
    else:
        pi = float(counter[0]) / float(counter[0] + counter[1])
        entropy = - (pi * math.log(pi, 2) + (1. - pi) * math.log(1. - pi, 2))
    return entropy


def entropy_after_optimal_divide(y, x):
    min_entropy = np.inf
    opt_th = 0
    for item in x:
        t_entropy = entropy_after_divide(y, x, item)
        if t_entropy < min_entropy:
            opt_th = item
            min_entropy = t_entropy

    return min_entropy, opt_th


def entropy_after_divide(y, x, threshold):
    flag_r = y[x <= threshold]
    flag_l = y[x > threshold]
    p_r = float(len(flag_r)) / float(len(y))
    p_l = float(len(flag_l)) / float(len(y))
    return (p_r * entropy(flag_r)) + (p_l * entropy(flag_l))


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

    mean_score = np.mean(np.array(mean_scores),0)

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
    
