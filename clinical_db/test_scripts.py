"""
Test code for scripts
"""
import unittest

class TestSequenceFunctions(unittest.TestCase):

#    def setUp(self):
#    def tearDown(self):

    def test_generate_id_list(self):
        import generate_id_list

    def test_list_id_from_code(self):
        import list_id_form_code

    def test_icu_admission_info(self):
        import icu_admission_info
        
    def test_analyze_icd9(self):
        import analyze_icd9

    def test_show_medical_record(self):
        import show_medical_record

    def test_classification(self):
        import classify_patients
        classify_patients.main(max_id = 2000)

    def test_evaluate_feature(self):
        import evaluate_feature
        ef = evaluate_feature.evaluate_fetaure(max_id = 2000, days_before_discharge = 0)
        ef.point_eval()


if __name__ == '__main__':
    unittest.main()
