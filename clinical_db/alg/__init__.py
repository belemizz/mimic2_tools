from collections import namedtuple
import numpy as np

## Classificaton Results
ClassificationResult = namedtuple('ClassificationResult' , 'P N TP FP rec prec f acc')

def sumup_classification_result(result_list):
    n_p = 0
    n_n = 0
    n_tp = 0
    n_fp = 0
    for result in result_list:
        n_p = n_p + result.P
        n_n = n_n + result.N
        n_tp = n_tp + result.TP
        n_fp = n_fp + result.FP
    recall, precision, f, acc = recall_precision(n_p, n_n, n_tp, n_fp)
    return ClassificationResult(n_p, n_n, n_tp, n_fp, recall, precision, f, acc)

def calc_classification_result(predict_y, test_y):
    predict_y = np.array(predict_y)
    test_y = np.array(test_y)
    
    n_positive = int( sum(test_y == 1))
    n_negative = int( sum(test_y == 0))
    n_true_positive = int( sum(predict_y[test_y == 1]))
    n_false_positive = int( sum(predict_y[test_y == 0]))
    recall, precision, f, acc = recall_precision(n_positive, n_negative, n_true_positive, n_false_positive)
    return ClassificationResult(n_positive, n_negative, n_true_positive, n_false_positive, recall, precision, f, acc)

def recall_precision(n_positive, n_negative, n_true_positive, n_false_positive):
    if n_positive > 0:
        recall = float(n_true_positive) / n_positive
    else:
        recall = 0.0

    if (n_true_positive + n_false_positive) > 0:
        precision = float(n_true_positive) / (n_true_positive + n_false_positive)
    else:
        precision = 0.0

    if (precision + recall > 0.0):
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0.0

    n_true_negative = n_negative - n_false_positive

    if n_positive + n_negative > 0.0:
        accuracy = float(n_true_positive + n_true_negative) / (n_positive + n_negative)
    else:
        accuracy = 0.0
        
    return recall, precision, f_measure, accuracy
