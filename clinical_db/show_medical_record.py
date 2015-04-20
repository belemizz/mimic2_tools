"""
Script to show summery of  medical record of a patient.
"""

import control_mimic2db as cm

patient_id = 894
# Check Notes

mimic2db = cm.control_mimic2db()
output_file_path = '../data/d_patients.csv'
mimic2db.d_patients(patient_id, output_file_path)

output_file_path = '../data/icd9.csv'
mimic2db.icd9(patient_id, output_file_path)

output_file_path = '../data/med_events.csv'
mimic2db.med_events(patient_id, output_file_path)

output_file_path = '../data/note_events.csv'
mimic2db.note_events(patient_id, output_file_path)

output_file_path = '../data/poe_order.csv'
mimic2db.poe_order(patient_id, output_file_path)

output_file_path = '../data/labevents.csv'
mimic2db.labevents(patient_id, output_file_path)

output_file_path = '../data/microbiologyevents.csv'
mimic2db.microbiologyevents(patient_id, output_file_path)
