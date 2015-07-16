import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import get_sample
import mutil.graph

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
            value = numpy.ones(n_dim, dtype = theano.config.floatX),
            name = 'W',
            borrow = True
        )

        self.b = theano.shared(
            value = numpy.ones(1, dtype = theano.config.floatX),
            name = 'b',
            borrow = True,
            broadcastable = [True]
        )
        self.params = [self.W, self.b]

        # Symbol 
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction"""
        ## return - T.mean(
        ##     T.log(self.p_y_given_x)   *  y
        ##     + T.log(1-self.p_y_given_x) * (1 - y)
        ##     )
        

### show the algorithms
def show_logistic_regression(set_x, set_y, learning_rate = 0.2, n_epochs = 1000, show_span = 500, filename = "", show_flag=True, x_label="", y_label=""):
    
    train_set_x = theano.shared(
        numpy.asarray(set_x, dtype=theano.config.floatX),
                             borrow = True)
    train_set_y = T.cast( theano.shared(
                    numpy.asarray(set_y, dtype = theano.config.floatX),
                    borrow = True
                    ), 'int32')

    gr = mutil.graph.Graph()
    
    x = T.matrix('x') # design matrix
    y = T.ivectors('y') # answer

    classifier = BinaryClassifier(x,2)
    cost_function = classifier.negative_log_likelihood(y)

    g_W = T.grad(cost = cost_function, wrt = classifier.W)
    g_b = T.grad(cost = cost_function, wrt = classifier.b)

    updates = [(classifier.W, classifier.W -learning_rate * g_W),
               (classifier.b, classifier.b -learning_rate * g_b)
           ]

    train_model = theano.function(
        inputs = [x,y],
        outputs = cost_function,
        updates = updates,
        )
    func_cost = theano.function([x, y], cost_function)

    epoch = 0

    x =  train_set_x.get_value()
    max_x = max(x[:,0])
    min_x = min(x[:,0])
    
    y =  train_set_y.eval()
    positive_x = x[y==1]
    negative_x = x[y==0]

    cost_prev = numpy.inf
    improve_th = 0.001
    
    while epoch < n_epochs:
        epoch = epoch +1
        
        train_model(train_set_x.get_value(), train_set_y.eval())

        cost_value = func_cost(train_set_x.get_value(), train_set_y.eval())
        cost_improve = cost_prev - cost_value
        cost_prev = cost_value

#        print "(%d/%d):%f improve:%f leaning_rate:%f"%(epoch, n_epochs, cost_value, cost_improve, learning_rate)

        if epoch % show_span == 0:
            a_0 = T.scalar()
            a_1 = T.scalar()
            x_all = [ a_0,
                      a_1,
                      (
                          ( -classifier.W.get_value()[0] * a_0 -classifier.b.get_value() )
                          /  classifier.W.get_value()[1]
                      )[0],
                      (
                          ( -classifier.W.get_value()[0] * a_1 -classifier.b.get_value() )
                          /  classifier.W.get_value()[1]
                      )[0]
                      ]
            func_x_all = theano.function([a_0, a_1], x_all)
            linev = func_x_all(min_x, max_x)

            gr.plot_classification(positive_x, negative_x, linev, "Title", x_label = x_label, y_label = y_label, show_flag = show_flag, filename = filename)


def main():
    # generate_samples
    [x, y] = get_sample.normal_dist(2, 100, 100, [3, 8], seed = 1 )
    show_logistic_regression(x, y)
    plt.waitforbuttonpress()
            
if __name__ == '__main__':
    main()
    
