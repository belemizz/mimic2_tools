import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

import sys
sys.path.append('../../deep_tutorial/sample_codes/')

from logistic_sgd import load_data
import generate_sample

import cPickle
import os

import dA
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

def pca(set_x, n_components):
    pca = PCA(n_components = n_components)
    return pca.fit(set_x).transform(set_x)

def ica(set_x, n_components):
    ica = FastICA(n_components = n_components)
    return ica.fit(set_x).transform(set_x)

def dae(set_x, learning_rate = 0.1, n_epochs = 100, n_hidden = 10, batch_size = 10, corruption_level = 0.0):

    import copy
    parameters = copy.copy(locals())
    del parameters['copy']

    input_cache_path = "../data/dae_in.pkl"
    output_cache_path = "../data/dae_out.pkl"

    if os.path.isfile(input_cache_path) and os.path.isfile(output_cache_path):
        f = open(input_cache_path, 'r')
        ret = cPickle.load(f)
        if ret == parameters:
            print "Cache exists for this params"
            g = open(output_cache_path, 'r')
            ret_val = cPicle.load(g)
            g.close()
            return ret_val
        else:
            print "Cache exists but params don't much"
    else:
        print "There are no cache"
    
    ## Check type and convert to the shared valuable
    if type(set_x) is T.sharedvar.TensorSharedVariable:
        training_x = set_x
    elif type(set_x) is numpy.ndarray:
        training_x = theano.shared(
            numpy.asarray(set_x, dtype = theano.config.floatX),
            borrow = True)
    else:
        raise TypeError("Sample set, set_x should be Tensor shared valuable or numpy.ndarray")

    n_dim = training_x.shape.eval()[1]
    n_train_batches = training_x.get_value(borrow=True).shape[0] / batch_size
    print (n_dim, n_train_batches)

    ## model description
    index = T.lscalar()
    x = T.matrix('x')
    numpy_rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    da = dA.dA(
        numpy_rng,
        theano_rng,
        input = x,
        n_visible = n_dim,
        n_hidden = n_hidden
    )
    cost, updates = da.get_cost_updates(
        corruption_level = corruption_level,
        learning_rate = learning_rate
    )

    train_da = theano.function(
        [index],
        cost, 
        updates = updates,
        givens = {
            x:training_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    ## output_function
    func_cost = theano.function([x], cost)
    for epoch in xrange(n_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        print 'Epoch %d/%d, Cost %f'%(epoch+1,n_epochs, numpy.mean(c))

    # calc_hidden_value
    hidden_values = da.get_hidden_values(x)
    func_hidden_values = theano.function([x], hidden_values)
    ret_val = func_hidden_values(training_x.get_value())


    f = open(input_cache_path, 'w')
    cPickle.dump(parameters, f)
    f.close()
    
    g = open(output_cache_path, 'w')
    cPickle.dump(ret_val, g)
    g.close()

    
    return func_hidden_values(training_x.get_value())

if __name__ == '__main__':
    ## get sample
    sample_num = 1
    if sample_num == 0:
        x = generate_sample.uniform_dist(80)
    else:
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset)
        x, y = datasets[0]
    dae(x)
