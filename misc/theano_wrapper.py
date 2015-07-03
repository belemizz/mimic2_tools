import sys
sys.path.append('../../DeepLearningTutorials/code/')
sys.path.append('../clinical_db')

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import generate_sample
import mutil

from logistic_sgd import LogisticRegression
from dA import dA
from SdA import SdA
import os

from sklearn.preprocessing import normalize

def stacked_denoising_auto_encoder(x, y,
                                   finetune_lr=0.001,
                                   pretraining_epochs=15,
                                   pretrain_lr=0.001,
                                   training_epochs=1000,
                                   hidden_layers_sizes = [10],
                                   corruption_levels = [.0],
                                   batch_size=25):

#    norm_x = normalize(x)
    [ar_train_x, ar_train_y, ar_valid_x, ar_valid_y, ar_test_x, ar_test_y] = generate_sample.split_to_three_sets(x, y)

    train_x = generate_sample.shared_array(ar_train_x)
    train_y = generate_sample.shared_flag(ar_train_y)
    valid_x = generate_sample.shared_array(ar_valid_x)
    valid_y = generate_sample.shared_flag(ar_valid_y)
    test_x = generate_sample.shared_array(ar_test_x)
    test_y = generate_sample.shared_flag(ar_test_y)
    
    n_train_batches = ar_train_x.shape[0] / batch_size
    n_valid_batches = ar_valid_x.shape[0] / batch_size
    n_test_batches = ar_test_x.shape[0] / batch_size

    n_in =  train_x.get_value().shape[1]
    n_out = len( set(train_y.eval()))

    print '[INFO] building the model'
    numpy_rng = numpy.random.RandomState(89677)
    sda = SdA(
        numpy_rng = numpy_rng,
        n_ins = n_in,
        hidden_layers_sizes = hidden_layers_sizes,
        n_outs = n_out
    )

    print '[INFO] getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x = train_x,
                                                 batch_size = batch_size)

    print '[INFO] pre-training the model'
    sw = mutil.stopwatch()
    for i in xrange(sda.n_layers):
        for epoch in xrange( pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(
                    pretraining_fns[i]( index = batch_index,
                                        corruption = corruption_levels[i],
                                        lr = pretrain_lr)
                                        )
            print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),
            print numpy.mean(c)
    sw.stop()
    
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (sw.real_elapsed() / 60.))
    print '[INFO] getting the finetuning functions'
    datasets = [(train_x, train_y), (valid_x, valid_y), (train_x, train_y)]
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '[INFO] finetuning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    test_score = 0.
    
    sw.reset()
    done_looping = False
    epoch = 0
    
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    sw.stop()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (sw.real_elapsed()/ 60.))
    
def simple_lr(train_x_data, train_y_data,
              test_x_data, test_y_data,
              learning_rate = 0.13,
              n_epochs = 1000,
              batch_size = 600):

    train_x = generate_sample.shared_array(train_x_data)
    train_y = generate_sample.shared_flag(train_y_data)
    test_x = generate_sample.shared_array(test_x_data)
    test_y = generate_sample.shared_flag(test_y_data)

    n_train_batches = train_x_data.shape[0] / batch_size
    n_test_batches = test_x_data.shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    n_in = train_x.get_value().shape[1]
    n_out = len( set(train_y.eval()))
    
    classifier = LogisticRegression(input = x, n_in = n_in, n_out = n_out)

    cost = classifier.negative_log_likelihood(y)

    ## models
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: test_x[index * batch_size: (index+1) * batch_size],
            y: test_y[index * batch_size: (index+1) * batch_size]
        }
    )
    
    g_W = T.grad(cost = cost, wrt=classifier.W)
    g_b = T.grad(cost = cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
#        outputs=cost,
        outputs=classifier.errors(y),
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    for epoch in range(1, n_epochs + 1):
        print ('epoch', epoch)

        train_errors = []
        for minibatch_index in range(n_train_batches):
            train_minibatch_error = train_model(minibatch_index)
            train_errors.append(train_minibatch_error)
        train_score = 1. - numpy.mean(train_errors)

        test_errors = []
        for minibatch_index in range(n_test_batches):
            test_minibatch_avg_cost = test_model(minibatch_index)
            test_errors.append(test_minibatch_avg_cost)

        test_score = 1.0 - numpy.mean(test_errors)
        print (train_score, test_score)
        
        done_looping = False
        if done_looping: break

        
        


    


def logistic_regression(x, y,
                        learning_rate = 0.13,
                        n_epochs = 1000,
                        batch_size = 600):

    [ar_train_x, ar_train_y, ar_valid_x, ar_valid_y, ar_test_x, ar_test_y] = generate_sample.split_to_three_sets(x, y)

    train_x = generate_sample.shared_array(ar_train_x)
    train_y = generate_sample.shared_flag(ar_train_y)
    valid_x = generate_sample.shared_array(ar_valid_x)
    valid_y = generate_sample.shared_flag(ar_valid_y)
    test_x = generate_sample.shared_array(ar_test_x)
    test_y = generate_sample.shared_flag(ar_test_y)
    
    n_train_batches = ar_train_x.shape[0] / batch_size
    n_valid_batches = ar_valid_x.shape[0] / batch_size
    n_test_batches = ar_test_x.shape[0] / batch_size


    print '[INFO] building the model'
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    n_in = train_x.get_value().shape[1]
    n_out = len( set(train_y.eval()))

    classifier = LogisticRegression(input=x, n_in = n_in, n_out = n_out)

    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: test_x[index * batch_size: (index+1) * batch_size],
            y: test_y[index * batch_size: (index+1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: valid_x[index * batch_size: (index+1) * batch_size],
            y: valid_y[index * batch_size: (index+1) * batch_size]
        }
    )

    g_W = T.grad(cost = cost, wrt=classifier.W)
    g_b = T.grad(cost = cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    print '[INFO] training the model'
    patience = 5000
    patience_increase = 2

    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience /2)

    best_validation_loss = numpy.inf
    test_score = 0.

    sw = mutil.stopwatch()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %% iter:%d/%d' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.,
                        iter, patience
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    sw.stop()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / sw.real_elapsed())
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % (sw.real_elapsed()))

def denoising_auto_encoder(x,
                           learning_rate = 0.1,
                           training_epochs = 15,
                           batch_size = 25,
                           n_hidden = 8,
                           corruption_level = 0.
                           ):
    
    norm_x = normalize(x)
    train_x = generate_sample.shared_array(norm_x)
    n_train_batches = x.shape[0] / batch_size

    print '[INFO] building the model'
    n_in = train_x.get_value().shape[1]

    index = T.lscalar()
    x = T.matrix('x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng = rng,
        theano_rng = theano_rng,
        input = x,
        n_visible = n_in,
        n_hidden = n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    print '[INFO] training the model'

    sw = mutil.stopwatch()
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    sw.stop()

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (sw.real_elapsed() / 60.))
    print da.W.get_value()


if __name__ == '__main__':
    x, y = generate_sample.get_samples_with_target(1, 0)

    algorithm = 5
    if algorithm is 0:
        logistic_regression(x,y, n_epochs = 5, batch_size = 5)
        
#    elif algorithm is 1:
#        # multi-layer perceptron
#        import mlp
#        mlp.test_mlp()
#    elif algorithm is 2:
#        # convolutional neural network
#        import convolutional_mlp
#        convolutional_mlp.evaluate_lenet5()
    elif algorithm is 3:
        # denoising auto-encoder
        denoising_auto_encoder(x, training_epochs = 20)
    elif algorithm is 4:
        # stacked denoising auto-encoders
        stacked_denoising_auto_encoder(x, y)
    elif algorithm is 5:
        import rbm
        rbm.test_rbm()
        

