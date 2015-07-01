import numpy
import matplotlib.pyplot as plt

from sklearn import svm, tree, linear_model, ensemble
from sklearn import cross_validation
from collections import namedtuple

import generate_sample
import control_graph

graph= control_graph.control_graph()
ClassificationResult = namedtuple('ClassificationResult' , 'P N TP FP rec prec f acc')

algorithm_list = ['svm', 'rsvm', 'psvm', 'dt', 'lr', 'rf', 'ab']

def get_algorithm(algorithm):
    if algorithm == 'svm':
        clf = svm.SVC(random_state = 0, kernel = 'linear', max_iter = 200000)
    elif algorithm == 'rsvm':
        clf = svm.SVC(random_state = 0, kernel = 'rbf', max_iter = 200000)
    elif algorithm == 'psvm':
        clf = svm.SVC(random_state = 0, kernel = 'poly', max_iter = 200000)
    elif algorithm == 'dt':
        clf = tree.DecisionTreeClassifier(random_state = 0)
    elif algorithm == 'lr':
        clf = linear_model.LogisticRegression(random_state = 0)
    elif algorithm == 'rf':
        clf = ensemble.RandomForestClassifier(random_state = 0)
    elif algorithm == 'ab':
        clf = ensemble.AdaBoostClassifier(random_state = 0)
    else:
        raise ValueError("algorithm has to be either %s"%algorithm_list)
    return clf

def plot_2d(x, y, x_label = "", y_label = "", filename = "", show_flag = True, algorithm = 'svm'):

    clf = get_algorithm(algorithm)

    if x.shape[1] is not 2:
        raise ValueError("Can't show: x dimension is not 2")

    clf.fit(x, y)
    
    # mesh
    x_range = x[:,0].max() - x[:,0].min()
    y_range = x[:,1].max() - x[:,1].min()

    margin_ratio = 0.1
    x_min, x_max = x[:,0].min() - margin_ratio * x_range , x[:,0].max() + margin_ratio * x_range
    y_min, y_max = x[:,1].min() - margin_ratio * y_range , x[:,1].max() + margin_ratio * y_range

    grid_num = 200.0
    h_x = x_range/grid_num
    h_y = y_range/grid_num

    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h_x),
                            numpy.arange(y_min, y_max, h_y))

    z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    graph.plot_classification_with_contour(x, y, xx, yy, z, x_label, y_label, filename, show_flag = show_flag)

    return clf

def recall_precision(n_positive, n_negative, n_true_positive, n_false_positive):
    recall = float(n_true_positive) / n_positive
    precision = float(n_true_positive) / (n_true_positive + n_false_positive)
    f_measure = 2 * precision * recall / (precision + recall)

    n_true_negative = n_negative - n_false_positive
    
    accuracy = float(n_true_positive + n_true_negative) / (n_positive + n_negative)
    return recall, precision, f_measure, accuracy

def fit_and_test(train_x, train_y, test_x, test_y, algorithm = 'dt'):
    clf = get_algorithm(algorithm)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)

    n_positive = sum(test_y == 1)
    n_negative = sum(test_y == 0)
    n_true_positive = sum(predict_y[test_y == 1])
    n_false_positive = sum(predict_y[test_y == 0])
    recall, precision, f, acc = recall_precision(n_positive, n_negative, n_true_positive, n_false_positive)

    return ClassificationResult(n_positive, n_negative, n_true_positive, n_false_positive, recall, precision, f, acc)

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

def cross_validate(x, y, n_cv_fold = 10, algorithm = 'dt'):
    clf = get_algorithm(algorithm)

    scores = cross_validation.cross_val_score(clf, x, y, cv = n_cv_fold)
    predicted = cross_validation.cross_val_predict(clf, x, y, cv = n_cv_fold)
    n_p = sum(y == 1)
    n_n = sum(y == 0)
    n_tp = sum(predicted[y == 1])
    n_fp = sum(predicted[y == 0])
    recall, precision, f, acc = recall_precision(n_p, n_n, n_tp, n_fp)
    return ClassificationResult(n_p, n_n, n_tp, n_fp, recall, precision, f, acc)

        
if __name__ == '__main__':
    source_num = 2
    n_dim = 200
    n_flag = 2
    [x,y]= generate_sample.get_samples_with_target(source_num, n_dim, n_flag)

    algorithm = 'ab'
    try:
        plot_2d(x,y, algorithm = algorithm)
        plt.waitforbuttonpress()
    except ValueError, detail:
        print detail
    
    cross_validation_num = 2

    print '----using library for cross validation---'
    print cross_validate(x, y, cross_validation_num, algorithm)

    print '----own library for cross validation---'
    kf = cross_validation.KFold(x.shape[0], n_folds = 4, shuffle = True, random_state = 0)
    result_list = []
    for train, test in kf:
        train_x = x[train, :]
        train_y = y[train]
        
        test_x = x[test, :]
        test_y = y[test]

        result = fit_and_test(train_x, train_y, test_x, test_y, algorithm = algorithm)
        print result
        result_list.append(result)

    print 'SUMUP'
    print sumup_classification_result(result_list)
