"""
Test code for scripts
"""

######## test of Unittest

import unittest
import control_mimic2db as cm

class TestSequenceFunctions(unittest.TestCase):

#    def setUp(self):

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


if __name__ == '__main__':
    unittest.main()

