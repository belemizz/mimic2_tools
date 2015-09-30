import numpy as np

from sklearn.preprocessing import normalize
from alg.theano_util import convert_to_tensor_shared_variable
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import sys
sys.path.append('../../DeepLearningTutorials/code/')
import dA


class DAE:
    def __init__(self, learning_rate=0.01, n_epochs=200, n_hidden=40,
                 batch_size=20, corruption_level=0.3):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.corruption_level = corruption_level

    def fit(self, train_x):
        norm_train_x = normalize(train_x)
        shared_train = convert_to_tensor_shared_variable(norm_train_x)

        n_dim = shared_train.shape.eval()[1]
        n_train_batches = shared_train.get_value(borrow=True).shape[0] / self.batch_size

        # model description
        index = T.lscalar()
        x = T.matrix('x')
        numpy_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.da = dA.dA(numpy_rng, theano_rng, input=x, n_visible=n_dim, n_hidden=self.n_hidden)
        cost, updates = self.da.get_cost_updates(corruption_level=self.corruption_level,
                                                 learning_rate=self.learning_rate)

        train_da = theano.function(
            [index], cost, updates=updates,
            givens={x: shared_train[index * self.batch_size: (index + 1) * self.batch_size]})

        for epoch in xrange(self.n_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))
#            print 'Epoch %d/%d, Cost %f' % (epoch + 1, n_epochs, np.mean(c))

        hidden_values = self.da.get_hidden_values(x)
        func_hidden_values = theano.function([x], hidden_values)
        self.func_hidden_values = func_hidden_values

    def transform(self, x):
        norm_x = normalize(x)
        shared_x = convert_to_tensor_shared_variable(norm_x)
        ret_val = self.func_hidden_values(shared_x.get_value())
        return ret_val
