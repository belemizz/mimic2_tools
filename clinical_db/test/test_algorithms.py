"""
Test code for algorithm codes
"""
import unittest
import get_sample
import numpy as np

from sys import exit
from mutil import Cache
from nose.tools import ok_, eq_

save_result = False

if save_result:
    char = raw_input('This is the save mode. Continue[y/n]? ')
    if char is not 'y':
        exit()


class TestSequenceFunctions(unittest.TestCase):

    def test_real_test_mode(self):
        ok_(not save_result, 'this is save mode')

    def test_auto_encoder(self):
        import alg.auto_encoder
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
        import alg.feature_selection
        i_index = alg.feature_selection.select_feature_index(dae, y, n_select=5)
        dae_s_c = dae[:, i_index]
        ok_((dae_s == dae_s_c).all())

    def test_classification(self):
        import alg.classification
        x, y = get_sample.vector(0, 2)
        alg.classification.plot_2d(x, y, show_flag=False)
        result_cv = alg.classification.cv([x, y], 4)
        self.__check_data('cv', result_cv)
        result_cv = alg.classification.cv([x, y], 4, auc=True)
        self.__check_data('cv_auc', result_cv)
        result_ex = alg.classification.example(0, 2)
        self.__check_data('example', result_ex)

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

if __name__ == '__main__':
    unittest.main()
