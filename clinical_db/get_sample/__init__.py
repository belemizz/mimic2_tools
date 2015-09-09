"""Get sample by accesssing database or generating."""

import numpy as np
import theano
import theano.tensor as T

import random

from .mimic2 import Mimic2, PatientData
from .mimic2m import Mimic2m
from .timeseries import TimeSeries, SeriesData

__all__ = [Mimic2, PatientData, Mimic2m, TimeSeries, SeriesData]


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


# to be deprecated
def select_tseries(sample_all, index):
    """Select elements in timeseries according to the index."""
    sel_x = sample_all[0][:, index, :]
    sel_m = sample_all[1][:, index]
    sel_y = sample_all[2][index]
    return [sel_x, sel_m, sel_y]
