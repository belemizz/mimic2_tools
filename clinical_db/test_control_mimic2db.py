"""
Test code for scripts
"""
import unittest
from nose.tools import ok_, eq_
import datetime
from  get_sample.mimic2 import Mimic2

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.mimic2 = Mimic2()

    #    def tearDown(self):

    def test_patient_class(self):
        patient_class = self.mimic2.get_subject(1855)
        
        eq_(len(patient_class.admissions[0].labs), 135, "Num of Labtests")
        eq_(len(patient_class.admissions[0].notes), 125, "Num of Notes")
        eq_(len(patient_class.admissions[1].icd9[1]), 5, "Dim of ICD9")
        eq_(len(patient_class.admissions[0].icustays[0].ios), 34, "Num of ios")
        eq_(patient_class.admissions[0].final_labs_time,
            datetime.datetime(3408,3,31,6,50))

        eq_(len(patient_class.admissions[1].get_newest_lab_at_time(datetime.datetime(3408,6,4))),47)
        eq_(len(patient_class.admissions[1].get_newest_lab_at_time(datetime.datetime(3408,6,1))),0)

        eq_(patient_class.admissions[1].get_chart_in_span(datetime.datetime(3408,6,1), datetime.datetime(3408,6,4))[0][1], 'Sodium')
        eq_(patient_class.admissions[1].get_chart_in_span(datetime.datetime(3408,6,1), datetime.datetime(3408,6,5))[3][4], [38.0, 124.0])

                                                            
if __name__ == '__main__':
    unittest.main()
