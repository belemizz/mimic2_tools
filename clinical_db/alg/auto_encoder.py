"""Algorithms for encoding the vectors."""

import sys
sys.path.append('../../DeepLearningTutorials/code/')

from sklearn.decomposition import PCA, FastICA
import alg.feature_selection

from bunch import Bunch
from dae import DAE

L_algorithm = ['pca', 'spca', 'ica', 'sica', 'dae', 'sdae']
Default_param = Bunch(name='pca')

AE_Param = Bunch(n_components=2, n_select=1, learning_rate=0.1,
                 n_epochs=100, n_hidden=20, batch_size=10, corruption_level=0.0)


class AE():
    '''Base class for auto encoder'''
    def __init__(self, param):
        self.param = param
        self.enc = self.get_encoder(param)

    def fit(self, train_x):
        self.enc.fit(train_x)

    def transform(self, test_x):
        return self.enc.transform(test_x)

    def fit_select(self, train_x, train_y):
        self.fit(train_x)
        enc_train = self.transform(train_x)
        entropy_reduction = alg.feature_selection.calc_entropy_reduction(enc_train, train_y)
        self.select_index = [item[1] for item in entropy_reduction[0:self.param.n_select]]

    def transform_select(self, test_x):
        enc_test = self.transform(test_x)
        return enc_test[:, self.select_index]


class PCA_AE(AE):
    def get_encoder(self, param):
        return PCA(n_components=param.n_components)


class ICA_AE(AE):
    def get_encoder(self, param):
        return FastICA(n_components=param.n_components)


class DAE_AE(AE):
    def get_encoder(self, param):
        return DAE(param.learning_rate, param.n_epochs, param.n_hidden,
                   param.batch_size, param.corruption_level)
