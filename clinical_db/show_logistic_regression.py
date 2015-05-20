import numpy
import theano
import theano.tensor as T

import random
import logistic_regression
import control_graph

import matplotlib.pyplot as plt

def sample_generation(n_neg_sample = 100, n_pos_sample = 100):
    data = []
    
    for i in xrange(0,n_neg_sample):
        vec = numpy.random.randn(1,2) + 2
        flag = 0
        data.append([vec,flag])
    for i in range(0,n_pos_sample):
        vec = numpy.random.randn(1,2) + 8
        flag = 1
        data.append([vec,flag])

    random.shuffle(data)

    x = numpy.array([item[0][0] for item in data])
    y = numpy.array([item[1] for item in data])

    shared_x = theano.shared(
        numpy.asarray(x, dtype=theano.config.floatX),
                             borrow = True)
    shared_y = T.cast( theano.shared(
                    numpy.asarray(y, dtype = theano.config.floatX),
                    borrow = True
                    ), 'int32')

    return [shared_x, shared_y]


    
def show_logistic_regression(train_set_x, train_set_y, learning_rate = 0.2, n_epochs = 1000, show_span = 500 ):
    gr = control_graph.control_graph()
    
    x = T.matrix('x') # design matrix
    y = T.ivectors('y') # answer

    classifier = logistic_regression.BinaryClassifier(x,2)
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


    while epoch < n_epochs:
        epoch = epoch +1
        
        train_model(train_set_x.get_value(), train_set_y.eval())
        print func_cost(train_set_x.get_value(), train_set_y.eval())

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
                
            gr.plot_classification(positive_x, negative_x, linev, "Title")
            plt.waitforbuttonpress()



def main():
    [train_set_x, train_set_y] = sample_generation()
    show_logistic_regression(train_set_x, train_set_y)
    
if __name__ == '__main__':
    main()
