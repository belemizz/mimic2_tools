import sys
sys.path.append('../../deep_tutorial/sample_codes/')
sys.path.append('../../DeepLearningTutorials/code/')


import numpy as np
import theano
import theano.tensor as T

import random
import imdb


def time_series(source_num = 0, n_dim = 2):
    """
    get timeseries data
    returns [x, y]:
    x: list of data series
        1st dim: time
        2nd dim: sample
        3rd dim: feature
    y: list of labels
    """
    if source_num is 0:
        [x, mask, y] = normal_timeseries(n_dim = n_dim, bias = [-0.3, 0.3])
    elif source_num is 1:
        [x, mask, y] = imdb_data()
    else:
        raise ValueError('source_num must be 0')
    return x, mask, y

def select_tseries(sample_all, index):
    sel_x = sample_all[0][:,index, :]
    sel_m = sample_all[1][:,index]
    sel_y = sample_all[2][index]
    return [sel_x, sel_m, sel_y]

def l_tseries_to_ar(ts_x):
    """
    Convert list of timeseries to numpy arrays of value and mask
    """
    max_length = max([len(s) for s in ts_x])

    if np.array(ts_x[0]).ndim == 1:
        # data is one dimentional
        dim = 1
    elif np.array(ts_x[0]).ndim == 2:
        dim = ts_x[0].shape[1]
    else:
        raise ValueError('Invalid data format')
    
    x = np.zeros( (max_length, len(ts_x), dim) )
    mask = np.zeros((max_length, len(ts_x)))

    for i_series, series in enumerate(ts_x):
        if dim == 1:
            x[:len(series), i_series] = [[val] for val in series]
        else:
            x[:len(series), i_series] = series
        mask[:len(series), i_series] = 1
    return x, mask

def normal_timeseries(length = 50, n_dim = 2, random_length = True, n_neg_sample = 1500, n_pos_sample = 500, bias = [-1, +1], seed = 0):

    random.seed(seed)
    np.random.seed(seed)

    def get_series(bias, flag, n_sample):
        data = []
        for num in xrange(0, n_sample):
            if random_length:
                s_len = np.random.randint(length) + length
            else:
                s_len = length
            sample = np.zeros([s_len, n_dim])
            for i in xrange(0, s_len):
                sample[i] = np.random.randn(1, n_dim) + bias
            data.append([sample, flag])
        return data

    negative_data = get_series(bias[0], 0, n_neg_sample)
    positive_data = get_series(bias[1], 1, n_pos_sample)

    data = negative_data+positive_data
    random.shuffle(data)

    x = [item[0] for item in data]
    x, mask = l_tseries_to_ar(x)
    y = np.array([item[1] for item in data])
    return [x,mask, y]

def imdb_data():
    train, valid, test = imdb.load_data(n_words = 10000,
                                        valid_portion = 0.05,
                                        maxlen = 100)

    x = train[0] + valid[0] + test[0]
    x, mask = l_tseries_to_ar(x)

    y = np.array(train[1] + valid[1] + test[1])
    return [x, mask,  y]

def get_samples_with_target(source_num = 0, data_dim = 0, n_flag=0):
    
    if source_num is 0:
        [x, y] = normal_dist(data_dim, 100, 100, [2,8], seed = 1)
        
    elif source_num is 1:
        from sklearn import datasets
        iris = datasets.load_iris()
        x, y = chop_data(iris.data, iris.target, data_dim, n_flag)

    elif source_num is 2:
        from logistic_sgd import load_data
        datasets = load_data('mnist.pkl.gz')
        [shared_x, shared_y] = datasets[0]
        x, y = chop_data(shared_x.get_value(), shared_y.eval(), data_dim, n_flag)

    else:
        raise ValueError

    return x, y


def chop_data(all_data, all_target, data_dim, n_flag):
    """ reduce the number of category of the flags to n_flag """
    all_flag = numpy.unique(all_target)
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

    return x,y

def split_to_three_sets(x, y, valid_ratio = 1./3, test_ratio = 1./3, r_seed = 1):
    n_valid = int(valid_ratio * x.shape[0])
    n_test = int(test_ratio * x.shape[0])
    n_train = x.shape[0] - n_valid - n_test

    index_all = range( x.shape[0])
    random.seed( r_seed)
    random.shuffle( index_all)

    train_index = index_all[0:n_train]
    valid_index = index_all[n_train:n_train+n_valid]
    test_index = index_all[n_train+n_valid: n_train+n_valid+n_test]

    train_x = x[train_index, :]
    train_y = y[train_index]

    valid_x = x[valid_index, :]
    valid_y = y[valid_index]

    test_x = x[test_index, :]
    test_y = y[test_index]

    return [train_x, train_y, valid_x, valid_y, test_x, test_y]

def shared_array(set_x):
    shared_x = theano.shared(
        np.asarray(set_x, dtype = theano.config.floatX),
        borrow = True)
    return shared_x

def shared_flag(set_y):
    shared_y = theano.shared(
        np.asarray(set_y, dtype=theano.config.floatX),
        borrow=True)
    return T.cast(shared_y, 'int32')

def normal_dist(n_dim = 2, n_neg_sample = 100, n_pos_sample = 100, bias = [-2, 2], seed = 1):
    
    """ Generate 2 element samples of normal distribution """
    data = []
    
    random.seed(seed)
    np.random.seed(seed)

    for i in xrange(0,n_neg_sample):
        vec = np.random.randn(1,n_dim) + bias[0]
        flag = 0
        data.append([vec,flag])
    for i in range(0,n_pos_sample):
        vec = np.random.randn(1,n_dim) + bias[1]
        flag = 1
        data.append([vec,flag])

    random.shuffle(data)

    x = np.array([item[0][0] for item in data])
    y = np.array([item[1] for item in data])

    return [x,y]


def uniform_dist(n_dim = 2, n_sample = 100, minimum = 0.0, maximum = 1.0, seed = 1):

    np.random.seed(seed)
    return np.random.uniform(minimum, maximum, (n_sample, n_dim))


def hoge():
    print 'hoge'
    
if __name__ == '__main__':
    print 'hoge'




