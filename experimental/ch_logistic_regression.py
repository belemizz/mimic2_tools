"""Compare deep learning libraries."""

import numpy as np

import chainer
import chainer.functions as F
from chainer import optimizers

import sys
sys.path.append("../clinical_db/")

from mutil import Graph, Stopwatch
graphs = Graph()
sw = Stopwatch()

import get_sample

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
# from keras.datasets import mnist

def keras_lr(train_x, train_y, test_x, test_y):
    """Logistic regression by keras."""
    model = Sequential()

    # mnist_data = mnist.load_data()
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')

    import ipdb
    ipdb.set_trace()

    model.add(Dense(n_dim, 64, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, 64, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, n_flag, init='uniform', activation='softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(train_x, train_y, nb_epoch=n_epoch, batch_size=batchsize)
    objective_score = model.evaluate(test_x, test_y, batch_size=batchsize)
    print objective_score


    # model.add(Dense(input_dim=train_x.shape[1], output_dim=n_flag), init="uniform")
    # model.add(Activation("tanh"))
    # model.add()
    # model.add(Activation("softmax"))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd')
    #

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
        y = F.sigmoid(model.l1(x))
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # optimizer
    optimizer = optimizers.Adam()
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

    n_dim = 700
    n_flag = 5
    [x, y] = get_sample.vector(2, n_dim, n_flag)

    n_epoch = 5
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
        keras_lr(train_x, train_y, test_x, test_y)
        sw.stop()
        sw.print_cpu_elapsed()

    # Chainer evaluation
    if chainer_flag:
        sw.reset()
        chainer_lr(train_x, train_y, test_x, test_y, batchsize, n_epoch)
        sw.stop()
        sw.print_cpu_elapsed()
