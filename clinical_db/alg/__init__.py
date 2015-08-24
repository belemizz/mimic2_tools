"""Algorithm for machine leaning."""
from collections import namedtuple
import numpy as np

# Classificaton Results
ClassificationResult = namedtuple('ClassificationResult',
                                  'P N TP FP rec prec f acc')


def sumup_classification_result(result_list):
    """Sum the classification result in the list."""
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
    final_result = ClassificationResult(n_p, n_n, n_tp, n_fp,
                                        recall, precision, f, acc)
    return final_result


def calc_classification_result(predict_y, test_y):
    """Compute predicted value and ground truth."""
    predict_y = np.array(predict_y)
    test_y = np.array(test_y)

    n_posi = int(sum(test_y == 1))
    n_nega = int(sum(test_y == 0))
    n_tp = int(sum(predict_y[test_y == 1]))
    n_fp = int(sum(predict_y[test_y == 0]))
    recall, precision, f, acc = recall_precision(n_posi, n_nega, n_tp, n_fp)
    result = ClassificationResult(n_posi, n_nega,
                                  n_tp, n_fp,
                                  recall, precision, f, acc)
    return result


def recall_precision(n_posi, n_nega, n_tp, n_fp):
    """Calcurate recall and precision."""
    if n_posi > 0:
        recall = float(n_tp) / n_posi
    else:
        recall = 0.0

    if (n_tp + n_fp) > 0:
        precision = float(n_tp) / (n_tp + n_fp)
    else:
        precision = 0.0

    if (precision + recall > 0.0):
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0.0

    n_true_negative = n_nega - n_fp

    if n_posi + n_nega > 0.0:
        accuracy = float(n_tp + n_true_negative) / (n_posi + n_nega)
    else:
        accuracy = 0.0

    return recall, precision, f_measure, accuracy

