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

def main():

    learning_rate = 0.1
    n_hidden = 2

    ## get sample
    used_sample = 0
    if used_sample == 0:
        n_dim = 10 * 10
        x = generate_sample.uniform_dist(n_dim) 
        training_set_x = theano.shared(
            numpy.asarray(x, dtype = theano.config.floatX),
            borrow = True)
    elif used_sample == 1:
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset)
        training_set_x, training_set_y = datasets[0]
        n_dim = 28 * 28

    batch_size = 1
    n_train_batches = training_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')

    ## model description
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
        cost,
        updates = updates,
        givens = {
            x:training_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    ## output_function
    func_cost = theano.function([x], cost)

    #    func_gparams = theano.function([x], gparams)

    n_epochs = 100
    for epoch in xrange(n_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
            # print 'Batch %d/%d, Cost %f'%(batch_index,n_train_batches, numpy.mean(c))
        print 'Epoch %d/%d, Cost %f'%(epoch+1,n_epochs, numpy.mean(c))
        

    

#        train_da(training_set_x.get_value())
#        print func_cost(training_set_x.get_value())
# Cost and Update
 
##     initial_W = numpy.asarray(
##         numpy_rng.uniform(
##             low = -4 * numpy.sqrt(6. / (n_hidden + n_dim)),
##             high = 4 * numpy.sqrt(6. / (n_hidden + n_dim)),
##             size = (n_dim, n_hidden)
##         ),
##         dtype = theano.config.floatX
##     )
##     W = theano.shared(value = initial_W, name = 'W', borrow = True)

##     b_prime = theano.shared(
##         value = numpy.zeros(
##             n_dim,
##             dtype = theano.config.floatX
##         ),
##         borrow = True
##     )

##     b = theano.shared(
##         value = numpy.zeros(
##             n_hidden,
##             dtype = theano.config.floatX
##         ),
##         name = 'b',
##         borrow = True
##     )

##     W_prime = W.T

##     params = [W, b, b_prime]

##     # hidden layer
##     y = T.nnet.sigmoid(T.dot(x, W) + b)

##     # reconstruction layer
##     z = T.nnet.sigmoid(T.dot(y, W_prime) + b_prime)

##     L = -T.sum(x * T.log(z) + (1 - x) * T.log(1-z), axis = 1)
##     cost = T.mean(L)

##     gparams = T.grad(cost, params)

##     updates = [
##         (param, param - learning_rate * gparam)
##         for param, gparam in zip(params, gparams)
##         ]


##     train_model = theano.function(
##         inputs = [x],
##         outputs = cost,
##         updates = updates
##         )

##     func_cost = theano.function([x], cost)
##     func_gparams = theano.function([x], gparams)


##     epoch = 0
##     n_epochs = 100
    


##     while epoch < n_epochs:
##         epoch = epoch + 1

##         train_model(training_set_x.get_value())
##         print func_cost(training_set_x.get_value())
##         print func_gparams(training_set_x.get_value())
## #        print W.get_value()

        
        
    
    


    


    

if __name__ == '__main__':
    main()














