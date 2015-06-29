import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

import sys
sys.path.append('../../DeepLearningTutorials/code/')
import dA
from logistic_sgd import load_data

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize

import mutil
import generate_sample
import alg_feature_selection

def pca(train_x, test_x, n_components, cache_key = 'pca'):
    params = locals()
    cache = mutil.cache(cache_key)

    try:
        return cache.load(params)
    except IOError:
        pca = PCA(n_components = n_components)
        ret_val = pca.fit(train_x).transform(test_x)
        return cache.save( ret_val, params)

def pca_selected(train_x, train_y, test_x, n_components, n_select, cache_key = 'pca_selected'):
    params = locals()
    cache = mutil.cache(cache_key)

    try:
        return cache.load(params)
    except IOError:
        pca = PCA(n_components = n_components).fit(train_x)
        pca_train = pca.transform(train_x)
        pca_test = pca.transform(test_x)

        entropy_reduction = alg_feature_selection.calc_entropy_reduction(pca_train, train_y)
        select_feature_index = [item[1] for item in entropy_reduction[0:n_select]]
        return pca_test[:,select_feature_index]
        

def ica(train_x, test_x, n_components, cache_key = 'ica'):
    params = locals()
    cache = mutil.cache(cache_key)
    
    try:
        return cache.load( params)
    except IOError:
        ica = FastICA(n_components = n_components)
        ret_val = ica.fit(train_x).transform(test_x)
        return cache.save( ret_val, params)

def ica_selected(train_x, train_y, test_x, n_components, n_select, cache_key = 'ica_selected'):
    params = locals()
    cache = mutil.cache(cache_key)

    try:
        return cache.load(params)
    except IOError:
        ica = FastICA(n_components = n_components).fit(train_x)
        ica_train = ica.transform(train_x)
        ica_test = ica.transform(test_x)

        entropy_reduction = alg_feature_selection.calc_entropy_reduction(ica_train, train_y)
        select_feature_index = [item[1] for item in entropy_reduction[0:n_select]]
        return ica_test[:,select_feature_index]

def dae(train_x, test_x, learning_rate = 0.1, n_epochs = 100, n_hidden = 20, batch_size = 10, corruption_level = 0.0, cache_key = 'dae'):

    params = locals()
    cache = mutil.cache(cache_key)
    try:
        return cache.load( params)
    except IOError:

        func_hidden_values = da_fit(train_x, learning_rate, n_epochs, n_hidden, batch_size, corruption_level)

        # calc_hidden_value
        norm_test_x = normalize(test_x)
        shared_test = convert_to_tensor_shared_variable(norm_test_x)
        ret_val = func_hidden_values(shared_test.get_value())
        return cache.save(ret_val, params )


def dae_selected(train_x, train_y, test_x, learning_rate = 0.005, n_epochs = 2000, n_hidden = 20, batch_size = 10, corruption_level = 0.0, n_select = 5, cache_key = 'dae_selected'):
    params = locals()
    cache = mutil.cache(cache_key)
    try:
        return cache.load( params)
    except IOError:
        func_hidden_values = da_fit(train_x, learning_rate, n_epochs, n_hidden, batch_size, corruption_level)

        norm_test_x = normalize(test_x)
        norm_train_x = normalize(train_x)
        shared_test = convert_to_tensor_shared_variable(norm_test_x)
        shared_train = convert_to_tensor_shared_variable(norm_train_x)

        da_train = func_hidden_values(shared_train.get_value())
        da_test = func_hidden_values(shared_test.get_value())

        entropy_reduction = alg_feature_selection.calc_entropy_reduction(da_train, train_y)
        select_feature_index = [item[1] for item in entropy_reduction[0:n_select]]
        return da_test[:,select_feature_index]
        

def da_fit(train_x, learning_rate, n_epochs, n_hidden, batch_size, corruption_level):
    ## Check type and convert to the shared valuable
    norm_train_x = normalize(train_x)
    shared_train = convert_to_tensor_shared_variable(norm_train_x)

    n_dim = shared_train.shape.eval()[1]
    n_train_batches = shared_train.get_value(borrow=True).shape[0] / batch_size

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
#        print 'Epoch %d/%d, Cost %f'%(epoch+1,n_epochs, numpy.mean(c))

    hidden_values = da.get_hidden_values(x)
    func_hidden_values = theano.function([x], hidden_values)
    return func_hidden_values

    
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

def get_encoded_values(train_x, train_y, test_x,
                       pca_components, pca_select,
                       ica_components, ica_select,
                       dae_hidden, dae_select, dae_corruption):
    
    encoded_values = {}
    if pca_components > 0:
        if pca_select > 0:
            encoded_values['pca_selected'] = pca_selected(train_x, train_y, test_x, pca_components, pca_select)
        else:
            encoded_values['pca'] =  pca(train_x, test_x, pca_components)

    if ica_components > 0:
        if ica_select > 0:
            encoded_values['ica_selected'] = ica_selected(train_x, train_y, test_x, ica_components, ica_select)
        else:
            encoded_values['ica'] = ica(train_x, test_x, ica_components)

    if dae_hidden > 0:
        if dae_select > 0:
            encoded_values['dae_selected'] = dae_selected(train_x, train_y, test_x, n_hidden = dae_hidden, n_select = dae_select, corruption_level = dae_corruption)
        else:
            encoded_values['dae'] = dae(train_x, test_x, n_hidden = dae_hidden)

    return encoded_values


def main(sample_num = 0):
    ## get sample
    if sample_num == 0:
        x, y = generate_sample.normal_dist(4)
    else:
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset)
        shared_x, shared_y = datasets[0]
        x = shared_x.get_value()
        y = shared_y.eval()

    encoded_values = get_encoded_values(x, y, x, 3, 0, 3, 0, 10, 0, 0.3)
    print encoded_values
    
if __name__ == '__main__':
    main()
    
