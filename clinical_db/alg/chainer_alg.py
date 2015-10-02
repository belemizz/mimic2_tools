import numpy as np

import chainer
import chainer.functions as F
from chainer import optimizers


class Chainer_LR():
    '''Logistic Regression with Chainer'''

    def __init__(self, n_epoch, batchsize):
        self.n_epoch = n_epoch
        self.batchsize = batchsize

    def fit(self, train_x, train_y):
        train_x = train_x.astype('float32')
        train_y = train_y.astype('int32')
        n_dim = train_x.shape[1]
        N = train_x.shape[0]
        n_flag = len(set(train_y))
        self.model = chainer.FunctionSet(l1=F.Linear(n_dim, n_flag))

        def forward(x_data, y_data, train=True):
            """Forward model for chainer."""
            x, t = chainer.Variable(x_data), chainer.Variable(y_data)
            y = self.model.l1(x)
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

        # optimizer
        optimizer = optimizers.SGD()
        optimizer.setup(self.model)
        train_acc = []
        train_loss = []

        for epoch in range(1, self.n_epoch + 1):
            perm = np.random.permutation(N)
            sum_accuracy = 0
            sum_loss = 0
            for i in range(0, N, self.batchsize):
                batch_x = train_x[perm[i:i + self.batchsize]]
                batch_y = train_y[perm[i:i + self.batchsize]]

                optimizer.zero_grads()
                loss, acc = forward(batch_x, batch_y)
                loss.backward()
                optimizer.update()

                sum_loss += loss.data * batch_x.shape[0]
                sum_accuracy += acc.data * batch_x.shape[0]

            loss = sum_loss / N
            acc = sum_accuracy / N
            train_acc.append(acc)
            train_loss.append(loss)

    def predict(self, test_x):
        test_x = test_x.astype('float32')
        x = chainer.Variable(test_x)
        y = self.model.l1(x)
        return y.data.argmax(axis=1)
