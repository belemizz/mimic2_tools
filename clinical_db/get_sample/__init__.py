"""Get sample by accesssing database or generating."""

import numpy as np
import theano
import theano.tensor as T

import random

from .mimic2 import Mimic2, PatientData
from .mimic2m import Mimic2m
from .timeseries import TimeSeries, SeriesData

__all__ = [Mimic2, PatientData, Mimic2m, TimeSeries, SeriesData]


def vector(source_num=0, n_dim=0, n_flag=2):
    """Get n-dim vector samples.

    :return: [x,y]
    x: 2-d array [sample, feature]
    y: 1-d array of labels
    """
    if source_num is 0:
        l_amount = [100] * n_flag
        bias = range(n_flag)
        [x, y] = normal_dist(n_dim, l_amount, bias, seed=1)

    elif source_num is 1:
        from sklearn import datasets
        iris = datasets.load_iris()
        x, y = chop_data(iris.data, iris.target, n_dim, n_flag)

    elif source_num is 2:
        from logistic_sgd import load_data
        datasets = load_data('mnist.pkl.gz')
        [shared_x, shared_y] = datasets[0]
        if n_dim > 0 and n_flag > 0:
            x, y = chop_data(shared_x.get_value(), shared_y.eval(),
                             n_dim, n_flag)
        else:
            x, y = shared_x.get_value(), shared_y.eval()

    else:
        raise ValueError
    return x, y


def select_tseries(sample_all, index):
    """Select elements in timeseries according to the index."""
    sel_x = sample_all[0][:, index, :]
    sel_m = sample_all[1][:, index]
    sel_y = sample_all[2][index]
    return [sel_x, sel_m, sel_y]


def chop_data(all_data, all_target, data_dim, n_flag):
    """Reduce the number of category of the flags to n_flag."""
    all_flag = np.unique(all_target)
    flags = all_flag[0: min(all_flag.shape[0], n_flag)]

    x_list = []
    y_list = []
    for flag in flags:
        x_list.append(all_data[all_target == flag])
        y_list.append(all_target[all_target == flag])

    x = np.vstack(x_list)
    y = np.hstack(y_list)

    if data_dim > 0:
        x = x[:, 0:data_dim]

    return x, y


def split_to_three_sets(x, y, valid_ratio=1. / 3, test_ratio=1. / 3, r_seed=1):
    """Split the dataset into three sets."""
    n_valid = int(valid_ratio * x.shape[0])
    n_test = int(test_ratio * x.shape[0])
    n_train = x.shape[0] - n_valid - n_test

    index_all = range(x.shape[0])
    random.seed(r_seed)
    random.shuffle(index_all)

    train_index = index_all[0:n_train]
    valid_index = index_all[n_train: n_train + n_valid]
    test_index = index_all[n_train + n_valid: n_train + n_valid + n_test]

    train_x = x[train_index, :]
    train_y = y[train_index]

    valid_x = x[valid_index, :]
    valid_y = y[valid_index]

    test_x = x[test_index, :]
    test_y = y[test_index]

    return [train_x, train_y, valid_x, valid_y, test_x, test_y]


def shared_array(set_x):
    """Convert array into theano sheard array."""
    shared_x = theano.shared(
        np.asarray(set_x, dtype=theano.config.floatX),
        borrow=True)
    return shared_x


def shared_flag(set_y):
    """Convert vector of flag into theano shared array."""
    shared_y = theano.shared(
        np.asarray(set_y, dtype=theano.config.floatX),
        borrow=True)
    return T.cast(shared_y, 'int32')


def normal_dist(n_dim=2, l_amount=[100, 100], bias=[-2, 2], seed=1):
    """Generate 2 element samples of normal distribution."""
    data = []

    random.seed(seed)
    np.random.seed(seed)

    for i, amount in enumerate(l_amount):
        for j in xrange(amount):
            vec = np.random.randn(1, n_dim) + bias[i]
            flag = i
            data.append([vec, flag])

    random.shuffle(data)

    x = np.array([item[0][0] for item in data])
    y = np.array([item[1] for item in data])

    return [x, y]


def uniform_dist(n_dim=2, n_sample=100, minimum=0.0, maximum=1.0, seed=1):
    """Generate samples by uniform distribution."""
    np.random.seed(seed)
    return np.random.uniform(minimum, maximum, (n_sample, n_dim))
