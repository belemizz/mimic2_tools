import numpy as np
import theano
import theano.tensor as T

class BinaryClassifier():
    def __init__(self, input, n_dim):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic valurable that describes the input

        :type n_in: int
        :param n_in: number of input units
        """

        ## Parameters 
        self.W = theano.shared(
            value = np.ones(n_dim, dtype = theano.config.floatX),
            name = 'W',
            borrow = True
        )

        self.b = theano.shared(
            value = np.ones(1, dtype = theano.config.floatX),
            name = 'b',
            borrow = True,
            broadcastable = [True]
        )
        self.params = [self.W, self.b]

        # Symbol 
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
#        self.y_pred_ =

    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction"""
        return - T.mean(
            T.log(self.p_y_given_x)   *  y
            + T.log(1-self.p_y_given_x) * (1 - y)
            )
        
#    def errors(self, y):
        
