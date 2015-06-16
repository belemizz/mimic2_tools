import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

import sys
sys.path.append('../../deep_tutorial/sample_codes/')
sys.path.append('../../DeepLearningTutorials/code/')

from logistic_sgd import load_data
import generate_sample

import dA
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize

import mutil




def pca_selected(set_train, flag_train, set_test, n_all, n_select, cache_key = 'pca_selected'):
    params = locals()
    cache = mutil.cache(cache_key)

    try:
        return cache.load(params)
    except ValueError:
        pca = PCA(n_components = n_components).fit(set_train)
        pca_train = pca.transform(set_train)
        pca_test = pca.transform(set_test)
        
        
    

    
def pca(set_train, set_test, n_components, cache_key = 'pca'):
    params = locals()
    cache = mutil.cache(cache_key)

    try:
        return cache.load(params)
    except ValueError:
        pca = PCA(n_components = n_components)
        ret_val = pca.fit(set_train).transform(set_test)
        return cache.save( params, ret_val)

def ica(set_train, set_test, n_components, cache_key = 'ica'):
    params = locals()
    cache = mutil.cache(cache_key)
    
    try:
        return cache.load( params)
    except ValueError:
        ica = FastICA(n_components = n_components)
        ret_val = ica.fit(set_train).transform(set_test)
        return cache.save( params, ret_val)

def dae(set_train, set_test, learning_rate = 0.1, n_epochs = 100, n_hidden = 20, batch_size = 10, corruption_level = 0.0, cache_key = 'dae'):

    params = locals()
    cache = mutil.cache(cache_key)
    try:
        return cache.load( params)
    except ValueError:

        norm_set_train = normalize(set_train)
        norm_set_test = normalize(set_test)

        ## Check type and convert to the shared valuable
        shared_train = convert_to_tensor_shared_variable(norm_set_train)
        shared_test = convert_to_tensor_shared_variable(norm_set_test)

        n_dim = shared_train.shape.eval()[1]
        n_train_batches = shared_train.get_value(borrow=True).shape[0] / batch_size
#        print (n_dim, n_train_batches)

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
                x:shared_train[index * batch_size: (index + 1) * batch_size]
            }
        )

        ## output_function
        func_cost = theano.function([x], cost)

        for epoch in xrange(n_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))
#            print 'Epoch %d/%d, Cost %f'%(epoch+1,n_epochs, numpy.mean(c))

        # calc_hidden_value
        hidden_values = da.get_hidden_values(x)
        func_hidden_values = theano.function([x], hidden_values)
        ret_val = func_hidden_values(shared_test.get_value())
        return cache.save(params, ret_val)

def convert_to_tensor_shared_variable(set_x):
    ## Check type and convert to the shared valuable
    if type(set_x) is T.sharedvar.TensorSharedVariable:
        shared_x = set_x
    elif type(set_x) is numpy.ndarray:
        shared_x = theano.shared(
            numpy.asarray(set_x, dtype = theano.config.floatX),
            borrow = True)
    else:
        raise TypeError("Sample set, set_x should be TensorSharedValuable or numpy.ndarray")
    return shared_x

if __name__ == '__main__':
    ## get sample
    sample_num = 0
    if sample_num == 0:
        x = generate_sample.uniform_dist(80)
    else:
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset)
        x, y = datasets[0]
    dae(x, x)
