import sys
sys.path.append('../../DeepLearningTutorials/code/')

import numpy
import theano
import theano.tensor as T

import random


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

    x = numpy.vstack(x_list)
    y = numpy.hstack(y_list)

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
        numpy.asarray(set_x, dtype = theano.config.floatX),
        borrow = True)
    return shared_x

def shared_flag(set_y):
    shared_y = theano.shared(
        numpy.asarray(set_y, dtype=theano.config.floatX),
        borrow=True)
    return T.cast(shared_y, 'int32')

def normal_dist(n_dim = 2, n_neg_sample = 100, n_pos_sample = 100, bias = [-2, 2], seed = 1):
    
    """ Generate 2 element samples of normal distribution """
    data = []
    
    random.seed(seed)
    numpy.random.seed(seed)

    for i in xrange(0,n_neg_sample):
        vec = numpy.random.randn(1,n_dim) + bias[0]
        flag = 0
        data.append([vec,flag])
    for i in range(0,n_pos_sample):
        vec = numpy.random.randn(1,n_dim) + bias[1]
        flag = 1
        data.append([vec,flag])

    random.shuffle(data)

    x = numpy.array([item[0][0] for item in data])
    y = numpy.array([item[1] for item in data])

    return [x,y]

def uniform_dist(n_dim = 2, n_sample = 100, minimum = 0.0, maximum = 1.0, seed = 1):

    numpy.random.seed(seed)
    return numpy.random.uniform(minimum, maximum, (n_sample, n_dim))
