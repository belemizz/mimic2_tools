"""Algorithms for basic classification."""
import numpy as np

from sklearn import svm, tree, linear_model, ensemble
from sklearn import cross_validation, metrics

from alg.metrics import BinaryClassResult, BinaryClassCVResult

import get_sample
from mutil import Graph, p_info
from . import (sumup_classification_result, calc_classification_result)

from bunch import Bunch

graph = Graph()

L_algorithm = ['svm', 'rsvm', 'psvm', 'lr', 'dt', 'rf', 'ab']
Default_param = Bunch(name='lr', lr_dim=10, svm_max_iter=20000)


def example(source_num=2, n_dim=784, n_flag=2, param=Default_param):
    """Sample code for this package."""
    [x, y] = get_sample.vector(source_num, n_dim, n_flag)
    try:
        plot_2d(x, y, param=param)
    except ValueError, detail:
        print detail

    p_info("Train and Test")
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(x, y)
    result = fit_and_test(train_x, train_y, test_x, test_y, param)
    print result.get_dict()

    p_info("Cross Validation")
    result_cv = cv(x, y)
    print result_cv.get_dict()
    print result_cv.mean_auc2

    p_info("Algorithm Comparison")
    l_alg = ['lr', 'dt', 'rf', 'ab']
    l_result = []
    for alg_name in l_alg:
        p_info(alg_name)
        param = Default_param
        param.name = alg_name
        l_result.append(cv(x, y, 10, param))

    print l_alg
    print [r.auc for r in l_result]


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


def fit_and_test(train_x, train_y, test_x, test_y, param=Default_param):
    """Train and test with samples."""
    clf = get_algorithm(param)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)

    if clf.__class__.__name__ in ['DecisionTreeClassifier', 'RandomForestClassifier']:
        score_y = clf.predict_proba(test_x)[:, 1] - clf.predict_proba(test_x)[:, 0]
    else:
        score_y = clf.decision_function(test_x)

    return BinaryClassResult(test_y, predict_y, score_y)


def cv(x, y, n_cv_fold=10, param=Default_param):
    '''Apply cross validataion'''
    l_result = []
    kf = cross_validation.StratifiedKFold(y, n_cv_fold, shuffle=True, random_state=0)
    for train, test in kf:
        l_result.append(fit_and_test(x[train, :], y[train], x[test, :], y[test], param))

    return BinaryClassCVResult(l_result)


def fit_and_test_orig(train_x, train_y, test_x, test_y, param=Default_param, auc=False):
    """Fit and test the algorithm."""
    clf = get_algorithm(param)

    def __fit_and_test(clf, auc):
        clf.fit(train_x, train_y)
        predict_y = clf.predict(test_x)
        result = calc_classification_result(predict_y, test_y)

        if auc:
            if clf.__class__.__name__ in ['DecisionTreeClassifier', 'RandomForestClassifier']:
                score_y = clf.predict_proba(test_x)[:, 1] - clf.predict_proba(test_x)[:, 0]
            else:
                score_y = clf.decision_function(test_x)
            auc = metrics.roc_auc_score(test_y, score_y)
            result = [result, auc]
        return result

    if isinstance(clf, list):
        result = []
        for c in clf:
            result.append(__fit_and_test(c, auc))
    else:
        result = __fit_and_test(clf, auc)

    return result


def cv_orig(sample_set, n_cv_fold=10, param=Default_param, auc=False):
    """Execute cross validation with samples."""
    x = sample_set[0]
    y = sample_set[1]

    if auc:
        kf = cross_validation.KFold(x.shape[0], n_cv_fold, shuffle=True, random_state=0)

        alg_result = [list([]) for _ in range(len(param.name))]
        alg_auc = [list([]) for _ in range(len(param.name))]
        for train, test in kf:
            result = fit_and_test_orig(x[train, :], y[train], x[test, :], y[test], param, True)
            for idx in range(len(result)):
                alg_result[idx].append(result[idx][0])
                alg_auc[idx].append(result[idx][1])

        alg_final_result = [sumup_classification_result(r) for r in alg_result]
        alg_final_auc = [np.mean(a) for a in alg_auc]
        cv_result = [alg_final_result, alg_final_auc]
        return cv_result

    else:
        clf = get_algorithm(param)

        def __cross_validation(clf, x, y):
            predict_y = cross_validation.cross_val_predict(clf, x, y, cv=n_cv_fold)
            cv_result = calc_classification_result(predict_y, y)
            return cv_result

        if isinstance(clf, list):
            result = []
            for c in clf:
                result.append(__cross_validation(c, x, y, n_cv_fold, param, auc))
        else:
            result = __cross_validation(c, x, y, n_cv_fold, param, auc)
        return result
