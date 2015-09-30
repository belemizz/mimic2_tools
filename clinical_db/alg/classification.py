"""Algorithms for basic classification."""
import numpy as np

from sklearn import svm, tree, linear_model, ensemble
from sklearn import cross_validation

from alg.metrics import BinaryClassResult, BinaryClassCVResult

from get_sample import point_data
from mutil import Graph, p_info

from bunch import Bunch
from dae import DAE

graph = Graph()

L_algorithm = ['svm', 'rsvm', 'psvm', 'lr', 'dt', 'rf', 'ab', 'pca_lr']
Default_param = Bunch(name='lr', lr_dim=10, svm_max_iter=20000, class_weight='auto')


def example(source_num=2, n_dim=784, n_flag=2, param=Default_param):
    """Sample code for this package."""
    [x, y] = point_data.sample(source_num, n_dim, n_flag)
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


def inbalancet_data(param=Default_param):
    p_info("Imbalanced Dataset")
    [x, y] = point_data.normal_dist(n_dim=2, l_amount=[900, 100], bias=[8.5, 10])
    param = Default_param
    param.class_weight = 'auto'
    result_cv = cv(x, y, param=param)
    plot_2d(x, y, param=param)
    print result_cv.get_dict()
    print result_cv.mean_auc2

    param.class_weight = None
    plot_2d(x, y, param=param)
    result_cv = cv(x, y, param=param)
    print result_cv.get_dict()
    print result_cv.mean_auc2


def get_algorithm(param=Default_param):
    """Get an algorithm for classification."""
    def _get_algorithm(name):
        if name == 'svm':
            clf = svm.SVC(random_state=0, kernel='linear', max_iter=param.svm_max_iter,
                          class_weight=param.class_weight)
        elif name == 'rsvm':
            clf = svm.SVC(random_state=0, kernel='rbf', max_iter=param.svm_max_iter,
                          class_weight=param.class_weight)
        elif name == 'psvm':
            clf = svm.SVC(random_state=0, kernel='poly', max_iter=param.svm_max_iter,
                          class_weight=param.class_weight)
        elif name == 'dt':
            clf = tree.DecisionTreeClassifier(random_state=0, class_weight=param.class_weight)
        elif name == 'lr':
            clf = linear_model.LogisticRegression(random_state=0, class_weight=param.class_weight)
        elif name == 'rf':
            clf = ensemble.RandomForestClassifier(random_state=0, class_weight=param.class_weight)
        elif name == 'ab':
            clf = ensemble.AdaBoostClassifier(random_state=0)
        elif name == 'pca_lr':
            clf = PCA_LR(param)
        elif name == 'ica_lr':
            clf = ICA_LR(param)
        elif name == 'dae_lr':
            clf = DAE_LR(param)
        else:
            raise ValueError("algorithm has to be either %s" % L_algorithm)
        return clf

    try:
        if isinstance(param.name, list):
            clf = []
            for name in param.name:
                try:
                    clf.append(_get_algorithm(name))
                except ValueError:
                    pass
        else:
            clf = _get_algorithm(param.name)
    except AttributeError:
        import ipdb
        ipdb.set_trace()
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
    train_x[np.isnan(train_x)] = 0.
    test_x[np.isnan(test_x)] = 0.

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
    i_cv = 0
    l_result = []
    kf = cross_validation.StratifiedKFold(y, n_cv_fold, shuffle=True, random_state=0)
    for train, test in kf:
        i_cv = i_cv + 1
        p_info("classification.cv: {0}/{1}".format(i_cv, n_cv_fold))
        l_result.append(fit_and_test(x[train, :], y[train], x[test, :], y[test], param))

    return BinaryClassCVResult(l_result)


class AE_LR():
    '''Class for classification with auto encoder'''
    def __init__(self, param):
        self.enc = self.get_encoder()
        self.clf = linear_model.LogisticRegression(random_state=0, class_weight=param.class_weight)

    def fit(self, train_x, train_y):
        self.enc.fit(train_x)
        enc_x = self.enc.transform(train_x)
        self.clf.fit(enc_x, train_y)

    def predict(self, test_x):
        enc_x = self.enc.transform(test_x)
        return self.clf.predict(enc_x)

    def decision_function(self, test_x):
        enc_x = self.enc.transform(test_x)
        return self.clf.decision_function(enc_x)


class PCA_LR(AE_LR):
    def get_encoder(self):
        from sklearn.decomposition import PCA
        n_components = 2
        return PCA(n_components=n_components)


class ICA_LR(AE_LR):
    def get_encoder(self):
        from sklearn.decomposition import FastICA
        n_components = 2
        return FastICA(n_components=n_components)


class DAE_LR(AE_LR):
    def get_encoder(self):
        return DAE()
