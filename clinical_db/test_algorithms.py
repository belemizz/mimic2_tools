"""
Test code for algorithm codes
"""

import unittest

class TestSequenceFunctions(unittest.TestCase):

    def test_auto_encoder(self):
        import alg_auto_encoder
        alg_auto_encoder.main()

if __name__ == '__main__':
    unittest.main()
    
