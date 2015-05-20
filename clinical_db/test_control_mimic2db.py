"""
Test code for scripts
"""
import unittest
from nose.tools import ok_, eq_
import datetime

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        import control_mimic2db as mimic2
        self.mimic2 = mimic2.control_mimic2db()

    #    def tearDown(self):

    def test_patient_class(self):k
        patient_class = self.mimic2.get_subject(1855)
        
        eq_(len(patient_class.admissions[0].labs), 135, "Num of Labtests")
        eq_(len(patient_class.admissions[0].notes), 125, "Num of Notes")
        eq_(len(patient_class.admissions[1].icd9[1]), 5, "Dim of ICD9")
        eq_(len(patient_class.admissions[0].icustays[0].ios), 34, "Num of ios")
        eq_(len(patient_class.admissions[1].get_newest_lab_at_time(datetime.datetime(3408,6,4))),47)
        eq_(len(patient_class.admissions[1].get_newest_lab_at_time(datetime.datetime(3408,6,1))),0)
        
if __name__ == '__main__':
    unittest.main()
