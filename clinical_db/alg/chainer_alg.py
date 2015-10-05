import numpy as np

import chainer.functions as F
from chainer import optimizers, FunctionSet, Variable


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
        self.model = FunctionSet(l1=F.Linear(n_dim, n_flag))

        def forward(x_data, y_data, train=True):
            """Forward model for chainer."""
            x, t = Variable(x_data), Variable(y_data)
            y = self.model.l1(x)
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

        # optimizer
        optimizer = optimizers.Adam()
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
        x = Variable(test_x)
        y = self.model.l1(x)
        return y.data.argmax(axis=1)


class Chainer_DAE():
    def __init__(self, n_hidden, n_epoch, batchsize):
        self.n_hidden = n_hidden
        self.n_epoch = n_epoch
        self.batchsize = batchsize
        self.loss_param = 0.3

    def encode(self, x):
        return F.sigmoid(self.model.encode(x))

    def decode(self, y):
        return F.sigmoid(self.model.decode(y))

    def fit(self, train_x):
        train_x = train_x.astype('float32')
        n_dim = train_x.shape[1]
        N = train_x.shape[0]
        self.model = FunctionSet(encode=F.Linear(n_dim, self.n_hidden),
                                 decode=F.Linear(self.n_hidden, n_dim))
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        for epoch in range(1, self.n_epoch + 1):
            perm = np.random.permutation(N)
            sum_loss = 0.
            for i in range(0, N, self.batchsize):
                batch_x = train_x[perm[i:i + self.batchsize]]
                sum_loss = sum_loss + self.train(batch_x, self.loss_param)
            print sum_loss / N

    def train(self, x, loss_param):
        self.optimizer.zero_grads()
        loss = self.cost(x, loss_param)
        loss.backward()
        self.optimizer.update()
        return loss.data

    def cost(self, x, loss_param):
        lost_x = x * np.random.binomial(1, 1 - loss_param, len(x[0])).astype('float32')
        x_variable = Variable(lost_x)
        t_variable = Variable(x)
        y = self.encode(x_variable)
        z = self.decode(y)
        return F.mean_squared_error(z, t_variable)

    def transform(self, test_x):
        test_x = test_x.astype('float32')
        x_variable = Variable(test_x)
        y_variable = self.model.encode(x_variable)
        return y_variable.data
