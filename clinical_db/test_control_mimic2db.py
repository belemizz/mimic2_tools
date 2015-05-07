"""
Test code for scripts
"""
import unittest
from nose.tools import ok_, eq_

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        import control_mimic2db as cm
        self.mimic2 = cm.control_mimic2db()
    #    def tearDown(self):

    def test_patient_class(self):
        patient_class = self.mimic2.patient_class(1855)
        eq_(len(patient_class.admissions[0].lab), 50, "Num of Labtests")



if __name__ == '__main__':
    unittest.main()
