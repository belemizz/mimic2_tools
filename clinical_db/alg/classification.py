"""Algorithms for basic classification."""
import numpy as np

from sklearn import svm, tree, linear_model, ensemble
from sklearn import cross_validation, metrics

import get_sample
from mutil import Graph, p_info
from . import (sumup_classification_result, calc_classification_result)

from bunch import Bunch

graph = Graph()

L_algorithm = ['svm', 'rsvm', 'psvm', 'lr', 'dt', 'rf', 'ab']
Default_param = Bunch(name='svm', lr_dim=10, svm_max_iter=20000)


def example(source_num=1, n_dim=2, n_flag=2, param=Default_param):
    """Sample code for this package."""
    [x, y] = get_sample.vector(source_num, n_dim, n_flag)
    try:
        plot_2d(x, y, param=param)
    except ValueError, detail:
        print detail

    kf = cross_validation.KFold(
        x.shape[0], n_folds=4, shuffle=True, random_state=0)
    result_list = []
    auc_list = []
    for train, test in kf:
        train_x = x[train, :]
        train_y = y[train]

        test_x = x[test, :]
        test_y = y[test]

        result, auc = fit_and_test(train_x, train_y, test_x, test_y, param, True)
        result_list.append(result)
        auc_list.append(auc)

    p_info("Cross Validation Result")
    print sumup_classification_result(result_list)
    print ('mean AUC', np.mean(auc_list))


def get_algorithm(param=Default_param):
    """Get an algorithm for classification."""
    def _get_algorithm(name):
        if name == 'svm':
            clf = svm.SVC(random_state=0, kernel='linear', max_iter=param.svm_max_iter)
        elif name == 'rsvm':
            clf = svm.SVC(random_state=0, kernel='rbf', max_iter=param.svm_max_iter)
        elif name == 'psvm':
            clf = svm.SVC(random_state=0, kernel='poly', max_iter=param.svm_max_iter)
        elif name == 'dt':
            clf = tree.DecisionTreeClassifier(random_state=0)
        elif name == 'lr':
            clf = linear_model.LogisticRegression(random_state=0)
        elif name == 'rf':
            clf = ensemble.RandomForestClassifier(random_state=0)
        elif name == 'ab':
            clf = ensemble.AdaBoostClassifier(random_state=0)
        else:
            raise ValueError("algorithm has to be either %s" % L_algorithm)
        return clf

    if isinstance(param.name, list):
        clf = []
        for name in param.name:
            try:
                clf.append(_get_algorithm(name))
            except ValueError:
                pass
    else:
        clf = _get_algorithm(param.name)

    return clf


def plot_2d(x, y, x_label="", y_label="", filename="",
            show_flag=True, param=Default_param):
    """Show the classification result."""
    clf = get_algorithm(param)

    if x.shape[1] is not 2:
        raise ValueError("Can't show: x dimension is not 2")

    clf.fit(x, y)

    x_range = x[:, 0].max() - x[:, 0].min()
    y_range = x[:, 1].max() - x[:, 1].min()

    margin_ratio = 0.1
    x_min, x_max = x[:, 0].min() - margin_ratio * x_range, x[:, 0].max() +\
        margin_ratio * x_range
    y_min, y_max = x[:, 1].min() - margin_ratio * y_range, x[:, 1].max() +\
        margin_ratio * y_range

    grid_num = 200.0
    h_x = x_range / grid_num
    h_y = y_range / grid_num

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x),
                         np.arange(y_min, y_max, h_y))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    graph.plot_classification_with_contour(x, y, xx, yy, z, x_label, y_label,
                                           filename, show_flag=show_flag)
    return clf


def fit_and_test(train_x, train_y, test_x, test_y, param=Default_param, auc=False):
    """Fit and test the algorithm."""
    clf = get_algorithm(param)

    if isinstance(clf, list):
        result = []
        for c in clf:
            c.fit(train_x, train_y)
            predict_y = c.predict(test_x)
            result.append(calc_classification_result(predict_y, test_y))
    else:
        clf.fit(train_x, train_y)
        predict_y = clf.predict(test_x)
        result = calc_classification_result(predict_y, test_y)

        if auc:
            if param.name in ['dt', 'rf']:
                score_y = clf.predict_proba(test_x)[:, 1] - clf.predict_proba(test_x)[:, 0]
            else:
                score_y = clf.decision_function(test_x)
            auc = metrics.roc_auc_score(test_y, score_y)
            result = [result, auc]

    return result


def cv(sample_set, n_cv_fold=10, param=Default_param):
    """Execute cross validation with samples."""
    x = sample_set[0]
    y = sample_set[1]
    clf = get_algorithm(param)

    if isinstance(clf, list):
        result = []
        for c in clf:
            predict_y = cross_validation.cross_val_predict(c, x, y, cv=n_cv_fold)
            result.append(calc_classification_result(predict_y, y))
    else:
        predict_y = cross_validation.cross_val_predict(clf, x, y, cv=n_cv_fold)
        result = calc_classification_result(predict_y, y)
    return result
