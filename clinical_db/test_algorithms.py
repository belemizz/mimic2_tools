"""
Test code for algorithm codes
"""

import unittest
import alg_auto_encoder
import alg_classification

import generate_sample
from mutil import cache

from nose.tools import ok_, eq_

save_result = False

class TestSequenceFunctions(unittest.TestCase):

    def test_real_test_mode(self):
        ok_(not save_result, 'this is save mode')
        
    def test_auto_encoder(self):
        x, y = generate_sample.normal_dist(4)

        encoded = alg_auto_encoder.pca(x, x, 2, cache_key = '')
        self.__check_data('pca', encoded)

        encoded = alg_auto_encoder.pca_selected(x, y, x, 2, 1, cache_key = '')
        self.__check_data('pca_selected', encoded)

        encoded = alg_auto_encoder.ica(x, x, 4, cache_key = '')
        self.__check_data('ica', encoded)

        encoded = alg_auto_encoder.ica_selected(x, y, x, 4, 2, cache_key = '')
        self.__check_data('ica_selected', encoded)
        
        encoded = alg_auto_encoder.dae(x, x, n_epochs = 100, cache_key = '')
        self.__check_data('dae', encoded)
        
        encoded = alg_auto_encoder.dae_selected(x, y, x, n_epochs = 100, n_select = 5, cache_key = '')
        self.__check_data('dae_selected', encoded)

    def test_classification(self):
        x, y = generate_sample.get_samples_with_target(0, 2)
        alg_classification.plot_2d(x, y, show_flag = False)
        alg_classification.cross_validate(x,y,4)

    def __check_data(self, cache_key, data):
        cc = cache(cache_key, cache_dir = '../data/test/')
        if save_result:
            cc.save(data)
        else:
            correct_data = cc.load()
            ok_((data == correct_data).all(), cache_key)

if __name__ == '__main__':
    unittest.main()


