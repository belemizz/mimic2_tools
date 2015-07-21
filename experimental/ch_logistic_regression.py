"""Compare deep learning libraries."""

import numpy as np
import sys

sys.path.append("../clinical_db/")

from mutil import Graph, Stopwatch
graphs = Graph()
sw = Stopwatch()

import get_sample

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

import chainer
import chainer.functions as F
from chainer import optimizers


def keras_lr(train_x, train_y, test_x, test_y, batchsize, n_epoch):
    """Logistic regression by keras."""
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_y = np_utils.to_categorical(train_y, n_flag)
    test_y = np_utils.to_categorical(test_y, n_flag)

    model = Sequential()
    model.add(Dense(n_dim, n_flag))
    model.add(Activation('softmax'))

    rms = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    model.fit(train_x, train_y, batch_size=batchsize, nb_epoch=n_epoch,
              show_accuracy=True, verbose=2,
              validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y, show_accuracy=True, verbose=0)
    import ipdb
    ipdb.set_trace(frame=None)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def chainer_lr(train_x, train_y, test_x, test_y, batchsize, n_epoch):
    """Logistic Regression by chainer."""
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')

    model = chainer.FunctionSet(l1=F.Linear(n_dim, n_flag))

    def forward(x_data, y_data, train=True):
        """Forward model for chainer."""
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        y = model.l1(x)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model.collect_parameters())

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    for epoch in range(1, n_epoch + 1):
        print ('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, N, batchsize):
            batch_x = train_x[perm[i:i+batchsize]]
            batch_y = train_y[perm[i:i+batchsize]]

            optimizer.zero_grads()
            loss, acc = forward(batch_x, batch_y)
            loss.backward()
            optimizer.update()

            sum_loss += loss.data * batchsize
            sum_accuracy += acc.data * batchsize

        loss = sum_loss / N
        acc = sum_accuracy / N
        print acc

        train_acc.append(acc)
        train_loss.append(loss)

        loss, acc = forward(test_x, test_y)
        test_acc.append(acc.data)
        test_loss.append(loss.data)

        if acc.data == 1.:
            break

#    graphs.line_series([train_acc, test_acc], range(1, len(test_acc)+1),
#                       ['train', 'test'], markersize=0)
#    graphs.line_series([train_loss, test_loss], range(1, len(test_loss)+1),
#                       ['train', 'test'], markersize=0)

if __name__ == '__main__':

    theano_flag = False
    keras_flag = True
    chainer_flag = False
    [x, y] = get_sample.vector(2)
    n_dim = x.shape[1]
    n_flag = len(set(y))

    n_epoch = 20
    batchsize = 50

    N = int(x.shape[0] * 0.8)
    all_data = get_sample.split_to_three_sets(x, y, 0., 0.2)
    train_x = all_data[0]
    train_y = all_data[1]
    test_x = all_data[4]
    test_y = all_data[5]

    # Theano evaluation
    if theano_flag:
        import theano_wrapper

        sw.reset()
        theano_wrapper.simple_lr(train_x, train_y, test_x, test_y,
                                 batch_size=batchsize, n_epochs=n_epoch)
        sw.stop()
        sw.print_cpu_elapsed()

    # Keras evaluation
    if keras_flag:
        sw.reset()
        keras_lr(train_x, train_y, test_x, test_y, batchsize, n_epoch)
        sw.stop()
        sw.print_cpu_elapsed()

    # Chainer evaluation
    if chainer_flag:
        sw.reset()
        chainer_lr(train_x, train_y, test_x, test_y, batchsize, n_epoch)
        sw.stop()
        sw.print_cpu_elapsed()
