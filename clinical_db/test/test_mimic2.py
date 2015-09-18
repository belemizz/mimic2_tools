"""
Test code for scripts
"""
from nose.plugins.attrib import attr
from nose.tools import eq_

import datetime
from get_sample import Mimic2, Mimic2m, PatientData, TimeSeries


class TestMimic2m:
    def setUp(self):
        self.mimic2m = Mimic2m()

    def test_numerics(self):
        l_numeric = self.mimic2m.get_numerics()
        l_id = self.mimic2m.get_id_numerics()
        eq_(len(l_numeric), 5266, 'Number of numeric records')
        eq_(len(l_id), 2808, 'Number of id who have numeric records')
        eq_(l_id[0], 1, 'First id')
        eq_(l_id[len(l_id) - 1], 32805, 'First id')


class TestMimic2:

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

    def test_subject_all(self):
        result = self.mimic2.subject_all(2000)
        eq_(len(result), 1993)
        eq_(result[100], 101)

    def test_subject_with_icd9_codes(self):
        result = self.mimic2.subject_with_icd9_codes(['428.0'], True, True)
        eq_(len(result), 391)
        result = self.mimic2.subject_with_icd9_codes(['428.0'])
        eq_(len(result), 484)
        eq_(result[10], 679)

    def test_subject_with_heart_failure(self):
        result = self.mimic2.subject_with_chf()
        eq_(len(result), 722)
        result = self.mimic2.subject_with_chf(max_seq=2)
        eq_(len(result), 2315)


@attr(mimic2_work=True)
class TestPatientData:
    """Test for PatientData class."""
    def setUp(self):
        mimic2 = Mimic2()
        id_list = mimic2.subject_with_chf(2000)
        self.patients = PatientData(id_list)

    def test_counter(self):
        eq_(self.patients.n_patient(), 42)
        eq_(self.patients.n_adm(), 64)

    def test_get_data(self):
        lab_list = self.patients.common_lab(2)
        eq_(lab_list[0], [50177, 50383])

        icd9_list = self.patients.common_icd9(2)
        eq_(icd9_list[0], ['428.0', '427.31'])

        med_list = self.patients.common_medication(2)
        eq_(med_list[0], [25, 43])

        span = [0, 1.]
        from_discharge = False
        self.patients.data_from_adm(lab_list[0], Mimic2.vital_charts, from_discharge)

        self.patients.point_from_adm(lab_list[0], Mimic2.vital_charts, span[1], from_discharge)
        self.patients.tseries_from_adm(lab_list[0], Mimic2.vital_charts, span, 0.1, from_discharge)
        self.patients.trend_from_adm(lab_list[0], Mimic2.vital_charts, span, from_discharge)


class TestTimeSeries:
    '''Test for TimeSeries class.'''
    def setUp(self):
        self.ts = TimeSeries()

    def test_normal(self):
        data = self.ts.normal(n_dim=3, bias=[-10, -9], length=5)
        eq_(data.series[8, 199, 2], 0.)
        eq_(data.label[199], 1)

    def test_imdb(self):
        data = self.ts.imdb_data()
        eq_(data.series[2955, 27102, 0], 4.0)
        eq_(data.label[27102], 1)
