"""
Test code for scripts
"""
import unittest
from nose.tools import eq_

import datetime
from get_sample import Mimic2, Mimic2m


class TestMimic2m(unittest.TestCase):
    def setUp(self):
        self.mimic2m = Mimic2m()

    def test_numerics(self):
        l_numeric = self.mimic2m.get_numerics()
        l_id = self.mimic2m.get_id_numerics()
        eq_(len(l_numeric), 5266, 'Number of numeric records')
        eq_(len(l_id), 2808, 'Number of id who have numeric records')
        eq_(l_id[0], 1, 'First id')
        eq_(l_id[len(l_id) - 1], 32805, 'First id')


class TestMimic2(unittest.TestCase):

    def setUp(self):
        self.mimic2 = Mimic2()

    def test_patient_class(self):
        patient_class = self.mimic2.get_subject(1855)

        eq_(len(patient_class.admissions[0].labs), 135, "Num of Labtests")
        eq_(len(patient_class.admissions[0].notes), 125, "Num of Notes")
        eq_(len(patient_class.admissions[1].icd9[1]), 5, "Dim of ICD9")
        eq_(len(patient_class.admissions[0].icustays[0].ios), 34, "Num of ios")
        eq_(patient_class.admissions[0].final_labs_time,
            datetime.datetime(3408, 3, 31, 6, 50))

        eq_(len(
            patient_class.admissions[1].get_newest_lab_at_time(datetime.datetime(3408, 6, 4))),
            47)
        eq_(len(
            patient_class.admissions[1].get_newest_lab_at_time(datetime.datetime(3408, 6, 1))),
            0)

        eq_(patient_class.admissions[1].get_chart_in_span(
            datetime.datetime(3408, 6, 1),
            datetime.datetime(3408, 6, 4))[0][1],
            'Sodium')

        eq_(patient_class.admissions[1].get_chart_in_span(
            datetime.datetime(3408, 6, 1),
            datetime.datetime(3408, 6, 5))[3][4],
            [38.0, 124.0])

    def test_subject_with_icd9_codes(self):
        result = self.mimic2.subject_with_icd9_codes(['428.0'], True, True)
        eq_(len(result), 391)
        result = self.mimic2.subject_with_icd9_codes(['428.0'])
        eq_(len(result), 484)
        eq_(result[10], 679)


if __name__ == '__main__':
    unittest.main()
