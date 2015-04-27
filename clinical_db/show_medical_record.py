"""
Script to show summery of  medical record of a patient.
"""

import control_mimic2db as cm

patient_id = 894
mimic2db = cm.control_mimic2db()


# Check Notes
output_file_path = '../data/labevents.csv'
result = mimic2db.lab_events(patient_id, output_file_path)

creatinine_record = [item for item in result if item[12] == '2160-0']
urea_nitrogen_record = [item for item in result if item[12] == '3094-0']

creatinine_time = [item[4] for item in creatinine_record]
creatinine_value = [item[5] for item in creatinine_record]
base_time = creatinine_time[0]
creatinine_time_diff = [(item-base_time).total_seconds()/3600 for item in creatinine_time]

import matplotlib.pyplot as plt

plt.plot(creatinine_time_diff, creatinine_value, 'bs--')
plt.show()



urea_nitrogen_time = [item[4] for item in urea_nitrogen_record]
urea_nitrogen_value = [item[5] for item in urea_nitrogen_record]
urea_time_diff = [(item-base_time).total_seconds()/3600 for item in urea_nitrogen_time]
plt.plot(urea_time_diff, urea_nitrogen_value, 'bs--')
plt.show()


## output_file_path = '../data/d_patients.csv'
## mimic2db.patient(patient_id, output_file_path)

## output_file_path = '../data/icd9.csv'
## mimic2db.icd9(patient_id, output_file_path)

## output_file_path = '../data/med_events.csv'
## mimic2db.med_events(patient_id, output_file_path)

## output_file_path = '../data/note_events.csv'
## mimic2db.note_events(patient_id, output_file_path)

## output_file_path = '../data/poe_order.csv'
## mimic2db.poe_order(patient_id, output_file_path)

## output_file_path = '../data/microbiologyevents.csv'
## mimic2db.microbiology_events(patient_id, output_file_path)

