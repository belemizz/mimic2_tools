import numpy as np

import chainer
import chainer.functions as F
from chainer import optimizers

import get_sample
import mutil.graph
import mutil

import sys

graphs = mutil.graph.Graph()
sw = mutil.Stopwatch()

sys.path.append('../../DeepLearningTutorials/code/')
sys.path.append('../misc')
import theano_wrapper

if __name__ == '__main__':

    sample_dim = 700
    output_dim = 5
    [x, y] = get_sample.vector(2, sample_dim, output_dim)

    n_epoch = 5
    batchsize = 50

    N = int(x.shape[0] * 0.8)
    all_data = get_sample.split_to_three_sets(x, y, 0., 0.2)
    train_x = all_data[0]
    train_y = all_data[1]
    test_x = all_data[4]
    test_y = all_data[5]
    theano_wrapper.simple_lr(train_x, train_y, test_x, test_y,
                             batch_size=50, n_epochs=5)

    # chainer model
    model = chainer.FunctionSet(l1=F.Linear(sample_dim, output_dim))

    def forward(x_data, y_data, train = True):
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

    sw.reset()

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
            loss, acc = forward(train_x, train_y)
            loss.backward()
            optimizer.update()

            sum_loss += loss.data * batchsize
            sum_accuracy += acc.data * batchsize

        loss = sum_loss / N
        acc = sum_accuracy / N
        print acc

        train_acc.append(acc)
        train_loss.append(loss)

        # test
        loss, acc = forward(test_x, test_y)
        test_acc.append(acc.data)
        test_loss.append(loss.data)

        if acc.data == 1.:
            break

    sw.stop()
    sw.print_cpu_elapsed()
    sw.print_real_elapsed()

    graphs.line_series([train_acc, test_acc], range(1, len(test_acc)+1), ['train','test'],
                       markersize = 0)
    graphs.line_series([train_loss, test_loss], range(1, len(test_loss)+1), ['train','test'], markersize = 0)

### no minibatch ###
##     for epoch in range(1, n_epoch + 1):
##         print ('epoch', epoch)

##         #training
##         optimizer.zero_grads()
##         loss, acc = forward(train_x, train_y)
##         loss.backward()
##         optimizer.update()
##         train_acc.append(acc.data)
##         train_loss.append(loss.data)

## #        print 'loss = {}, acc = {}'.format(loss.data, acc.data)

##         #test
##         loss, acc = forward(test_x, test_y)
##         test_acc.append(acc.data)
##         test_loss.append(loss.data)
## #        print 'loss = {}, acc = {}'.format(loss.data, acc.data)

##     graphs.line_series([train_acc, test_acc], range(1, n_epoch+1), ['train','test'],
##                        markersize = 0)
##     graphs.line_series([train_loss, test_loss], range(1, n_epoch+1), ['train','test'], markersize = 0)
