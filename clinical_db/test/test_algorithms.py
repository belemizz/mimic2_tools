"""
Test code for algorithm codes
"""
import numpy as np
from sys import exit
from nose.tools import ok_, eq_
from nose.plugins.attrib import attr
from bunch import Bunch

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

    @attr(alg_work=True)
    def test_auto_encoder_new(self):
        x, y = get_sample.point_data.normal_dist(4)
        param = alg.auto_encoder.AE_Param
        dae = alg.auto_encoder.Chainer_AE(param)
        dae.fit(x)
        encoded = dae.transform(x)
        print encoded

    def test_auto_encoder(self):
        x, y = get_sample.point_data.normal_dist(4)

        param = alg.auto_encoder.AE_Param
        pca = alg.auto_encoder.PCA_AE(param)

        pca.fit(x)
        encoded = pca.transform(x)
        self.__check_data('pca', encoded)

        pca.fit_select(x, y)
        encoded = pca.transform_select(x)
        self.__check_data('pca_selected', encoded)

        param.n_components = 4
        param.n_select = 2

        ica = alg.auto_encoder.ICA_AE(param)

        ica.fit(x)
        encoded = ica.transform(x)
        self.__check_data('ica', encoded)

        ica.fit_select(x, y)
        encoded = ica.transform_select(x)
        self.__check_data('ica_selected', encoded)

        dae = alg.auto_encoder.DAE_AE(param)
        dae.fit(x)
        encoded = dae.transform(x)
        self.__check_data('dae', encoded)

        param.n_select = 5
        dae.fit_select(x, y)
        encoded = dae.transform_select(x)
        self.__check_data('dae_selected', encoded)

    def __check_data(self, cache_key, data):
        cc = Cache(cache_key, cache_dir='../data/test/')
        if save_result:
            cc.save(data)
        else:
            correct_data = cc.load()
            if isinstance(data, np.ndarray):
                if not (data == correct_data).all():
                    import ipdb
                    ipdb.set_trace()
                ok_((data == correct_data).all(), cache_key)
            else:
                eq_(data, correct_data, cache_key)


class TestClassification:
    def setUp(self):
        pass

    def test_default(self):
        [x, y] = get_sample.point_data.sample(1, 2, 2)
        cv_result = alg.classification.cv(x, y)
        print cv_result.get_dict()

    def test_encoded(self):
        [x, y] = get_sample.point_data.sample(1, 2, 2)

        for name in ['dae_lr', 'pca_lr', 'ica_lr']:
            param = Bunch(alg.classification.Default_param.copy())
            param.name = name
            cv_result = alg.classification.cv(x, y, param=param)
            print (name, cv_result.get_dict())


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
