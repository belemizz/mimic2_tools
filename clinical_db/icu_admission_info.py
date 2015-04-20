"""
Script to list the icu records matched with waveform data
"""

import control_mimic2db as cm

output_file_path = '../data/icu_admission_details_test.csv'
mimic2db = cm.control_mimic2db()
mimic2db.matched_icustay_detail(output_file_path)
