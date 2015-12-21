"""Test code for scripts."""
from nose.plugins.attrib import attr
from bunch import Bunch

from predict_death import PredictDeath
from predict_readmission import PredictReadmission
from patient_classification import Default_db_param, Default_data_param, Default_alg_param


class TestScripts():

    def test_list_id_from_code(self):
        import list_id_form_code

    def test_analyze_icd9(self):
        import analyze_icd9

    @attr(work_script=True)
    def test_show_medical_record(self):
        import show_medical_record
        show_medical_record.visualize_data([1855], False, [0.5, 1])

    def test_predict_death(self):
        db_param = Bunch(Default_db_param.copy())
        db_param.max_id = 2000
        pd = PredictDeath(db_param=db_param)
        pd.execution()

    def test_predict_death_with_coef(self):
        db_param = Bunch(Default_db_param.copy())
        data_param = Bunch(Default_data_param.copy())
        db_param.max_id = 2000
        data_param.coef_flag = True
        data_param.coef_span = 2.
        data_param.tseries_flag = False

        pd = PredictDeath(db_param=db_param)
        pd.execution()

    def test_predict_readmission(self):
        db_param = Bunch(Default_db_param.copy())
        data_param = Bunch(Default_data_param.copy())
        alg_param = Bunch(Default_alg_param.copy())
        db_param.max_id = 2000
        alg_param.n_cv_fold = 2
        pr = PredictReadmission(db_param, data_param, alg_param)
        pr.execution()
