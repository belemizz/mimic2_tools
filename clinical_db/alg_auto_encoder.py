import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

import sys
sys.path.append('../../deep_tutorial/sample_codes/')
import dA
from logistic_sgd import load_data

import generate_sample

#class AutoEncoder():
#    def __init__(self):

def demo(set_x, learning_rate = 0.1, n_epochs = 100, n_hidden = 2, batch_size = 1):

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
        corruption_level = 0.3,
        learning_rate = learning_rate
    )

    train_da = theano.function(
        [index],
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
            # print 'Batch %d/%d, Cost %f'%(batch_index,n_train_batches, numpy.mean(c))
        print 'Epoch %d/%d, Cost %f'%(epoch+1,n_epochs, numpy.mean(c))

    # calc_hidden_value
    hidden_values = da.get_hidden_values(x)
    func_hidden_values = theano.function([x], hidden_values)
    return func_hidden_values(training_x.get_value())

if __name__ == '__main__':
    ## get sample
    sample_num = 0
    if sample_num == 0:
        # random
        n_dim = 80
        x = generate_sample.uniform_dist(n_dim)
    else:
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset)
        x, y = datasets[0]
#        n_dim = 28 * 28
    demo(x)
