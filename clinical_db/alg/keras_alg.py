from keras.models import Sequential
from keras.layers.core import Dense, Activation, AutoEncoder
from keras.layers import containers

from keras.optimizers import Adam
from keras.utils import np_utils


class Keras_LR():
    '''Logistic Regression with Chainer'''
    def __init__(self, n_epoch, batchsize):
        self.n_epoch = n_epoch
        self.batchsize = batchsize

    def fit(self, train_x, train_y):
        train_x = train_x.astype('float32')
        n_dim = train_x.shape[1]
        n_flag = len(set(train_y))
        train_y = np_utils.to_categorical(train_y, n_flag)

        self.model = Sequential()
        self.model.add(Dense(n_dim, n_flag))
        self.model.add(Activation('softmax'))
        rms = Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=rms)
        self.model.fit(train_x, train_y, batch_size=self.batchsize, nb_epoch=self.n_epoch,
                       verbose=0)

    def predict(self, test_x):
        test_x = test_x.astype('float32')
        return self.model.predict(test_x).argmax(axis=1)


class Keras_DAE():
    def __init__(self, n_epoch, batchsize):
        self.n_epoch = n_epoch
        self.batchsize = batchsize

    def fit(self, train_x):
        train_x = train_x.astype('float32')
        n_dim = train_x.shape[1]

        encoder = containers.Sequential([Dense(n_dim, 100)])
        decoder = containers.Sequential([Dense(100, n_dim)])
        self.model = Sequential()
        self.model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
        self.model.fit(train_x, train_x, batch_size=self.batchsize, nb_epoch=self.n_epoch)
#        self.model.compile(loss='categorical_crossentropy', optimizer=rms)

    def transform(self, test_x):
        return test_x
