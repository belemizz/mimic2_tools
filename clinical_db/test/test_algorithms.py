"""
Test code for algorithm codes
"""
import numpy as np
from sys import exit
from nose.tools import ok_, eq_
from nose.plugins.attrib import attr

import get_sample

from mutil import Cache

import alg.classification
import alg.timeseries
import alg.auto_encoder
import alg.feature_selection

save_result = False

if save_result:
    char = raw_input('This is the save mode. Continue[y/n]? ')
    if char is not 'y':
        exit()


class TestAutoEncoder:

    def test_real_test_mode(self):
        ok_(not save_result, 'this is save mode')

    def test_auto_encoder(self):
        x, y = get_sample.normal_dist(4)

        encoded = alg.auto_encoder.pca(x, x, 2, cache_key='')
        self.__check_data('pca', encoded)

        encoded = alg.auto_encoder.pca_selected(x, y, x, 2, 1, cache_key='')
        self.__check_data('pca_selected', encoded)

        encoded = alg.auto_encoder.ica(x, x, 4, cache_key='')
        self.__check_data('ica', encoded)

        encoded = alg.auto_encoder.ica_selected(x, y, x, 4, 2, cache_key='')
        self.__check_data('ica_selected', encoded)

        dae = alg.auto_encoder.dae(x, x, n_epochs=100, cache_key='')
        self.__check_data('dae', dae)

        dae_s = alg.auto_encoder.dae_selected(x, y, x, n_epochs=100, n_select=5, cache_key='')
        self.__check_data('dae_selected', dae_s)

        # check selection consistency
        i_index = alg.feature_selection.select_feature_index(dae, y, n_select=5)
        dae_s_c = dae[:, i_index]
        ok_((dae_s == dae_s_c).all())

    def __check_data(self, cache_key, data):
        cc = Cache(cache_key, cache_dir='../data/test/')
        if save_result:
            cc.save(data)
        else:
            correct_data = cc.load()
            if isinstance(data, np.ndarray):
                ok_((data == correct_data).all(), cache_key)
            else:
                eq_(data, correct_data, cache_key)


@attr(alg_work=True)
class TestClassification:
    def setUp(self):
        pass

    def test_example(self):

        pass

    def test_inbalance(self):
        pass


class TestTimeseries:

    def setUp(self):
        pass

    def test_example(self):
        ts = get_sample.TimeSeries()
        data = ts.sample(0, 2)
        train, test = data.split_train_test()

        result = alg.timeseries.fit_and_test(train, test)
        print result.get_dict()

        cv_result = alg.timeseries.cv(data)
        print cv_result.get_dict()

    def test_lr(self):
        # get data
        ts = get_sample.TimeSeries()
        data = ts.sample()
        train, test = data.split_train_test()

        lf = alg.timeseries.LR()
        lf.fit(train)
        eq_(lf.predict(test)[66], 0)
        lf.score(test)
