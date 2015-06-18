"""
Test code for algorithm codes
"""

import unittest
import alg_auto_encoder
import alg_svm

import generate_sample

class TestSequenceFunctions(unittest.TestCase):

    def test_auto_encoder(self):
        x, y = generate_sample.normal_dist(4)
        alg_auto_encoder.pca(x, x, 2, cache_key = '')
        alg_auto_encoder.pca_selected(x, y, x, 2, 1, cache_key = '')
        alg_auto_encoder.ica(x, x, 4, cache_key = '')
        alg_auto_encoder.ica_selected(x, y, x, 4, 2, cache_key = '')
        alg_auto_encoder.dae(x, x, n_epochs = 100)
        alg_auto_encoder.dae_selected(x, y, x, n_epochs = 100)

    def test_classification(self):
        x, y = generate_sample.get_samples_with_target(0)
        alg_svm.demo(x, y, show_flag = False)
        

if __name__ == '__main__':
    unittest.main()
    
