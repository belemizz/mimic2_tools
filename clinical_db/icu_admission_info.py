"""
Script to list the icu records matched with waveform data
"""

from get_sample import Mimic2

output_file_path = '../data/icu_admission_details_test.csv'
mimic2db = Mimic2()
mimic2db.matched_icustay_detail(output_file_path)
