import numpy
import random
from sklearn import datasets

def get_samples_with_target(source_num = 0, data_dim = 2):
    
    if source_num is 0:
        [x, y] = normal_dist(2, 100, 100, [2,8], seed = 1)
    elif source_num is 1:
        iris = datasets.load_iris()
        [x, y] = [iris.data[:, 0:data_dim], iris.target]
    else:
        raise ValueError
    return [x, y]

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

    
 
