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

    def test_patient_class(self):
        patient_class = self.mimic2.patient_class(1855)
        eq_(len(patient_class.admissions[0].labs), 50, "Num of Labtests")
        eq_(len(patient_class.admissions[1].notes), 125, "Num of Notes")
        eq_(len(patient_class.admissions[0].icd9[1]), 5, "Dim of ICD9")
        eq_(len(patient_class.admissions[1].icustays[0].medications[3]),6, "Dim of Medication")
        eq_(len(patient_class.admissions[1].icustays[0].charts[2]),6, "Dim of Charts")
        eq_(len(patient_class.admissions[1].icustays[0].ios), 34, "Num of ios")
        eq_(len(patient_class.admissions[0].get_newest_lab_at_time(datetime.datetime(3408,6,4))),47)
        eq_(len(patient_class.admissions[0].get_newest_lab_at_time(datetime.datetime(3408,6,1))),0)
        
        ##Check admission variables
        ## print "available lab__"
        ## patient_class.admissions[0].display_available_lab()
        ## print "icd9__"
        ## for item in patient_class.admissions[0].icd9:
        ##     print item

        ## print "notes__"
        ## for item in patient_class.admissions[0].notes:
        ##     print item

        ##Check icustay variables
        ## print "medication__"
        ## for item in patient_class.admissions[1].icustays[0].medications:
        ##     print item[5]

        ## print "chart__"
        ## print len(patient_class.admissions[0].icustays[0].charts)
        ## for item in patient_class.admissions[0].icustays[0].charts:
        ##     print (item[1],item[3])

        ## print "io_events__"
        ## for item in patient_class.admissions[1].icustays[0].ios:
        ##     print (item[1],item[3])

        


if __name__ == '__main__':
    unittest.main()
