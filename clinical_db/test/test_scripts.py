"""Test code for scripts."""
from nose.plugins.attrib import attr


class TestScripts():

    def test_list_id_from_code(self):
        import list_id_form_code

    def test_analyze_icd9(self):
        import analyze_icd9

    def test_show_medical_record(self):
        import show_medical_record

    def test_classification(self):
        import classify_patients
        classify_patients.main(max_id=2000)

    def test_predict_death(self):
        from predict_death import PredictDeath
        pd = PredictDeath(max_id=2000)
        pd.execution()

    @attr(work=True)
    def test_predict_readmission(self):
        from predict_readmission import PredictReadmission
        pr = PredictReadmission(max_id=2000)
        pr.execution()
