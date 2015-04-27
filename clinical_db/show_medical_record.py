"""
Script to show summery of  medical record of a patient.
"""

import control_mimic2db as cm
import matplotlib.pyplot as plt

def lab_event_extract(result, admission_id, ronic_code):
    ex_record = [item for item in result if (item[1] == admission_id and item[12] == ronic_code)]

    ex_time = [item[4] for item in ex_record]
    ex_value = [item[5] for item in ex_record]
    ex_unit = ex_record[0][8]
    return (ex_time, ex_value, ex_unit)

def time_diff_in_hour(time_seq, base_time):
    return [(item - base_time).total_seconds()/3600 for item in time_seq]

patient_id = 1855
mimic2db = cm.control_mimic2db()

# Get Data
output_file_path = '../data/admissions.csv'
admissions = mimic2db.admission(patient_id, output_file_path)

output_file_path = '../data/labevents.csv'
result = mimic2db.lab_events(patient_id, output_file_path)

output_file_path = '../data/noteevents.csv'
notes = mimic2db.note_events(patient_id)


for admission in admissions:
    base_time = admission[2]

    creatinine_time, creatinine_value, creatinine_unit = lab_event_extract(result, admission[0], '2160-0')
    creatinine_time_diff = time_diff_in_hour(creatinine_time, base_time);

    urea_time, urea_value, urea_unit = lab_event_extract(result, admission[0], '3094-0')
    urea_time_diff = time_diff_in_hour(urea_time, base_time);    

    fig, ax1 = plt.subplots()
    ax1.plot(creatinine_time_diff, creatinine_value, 'bs--')
    ax1.set_xlabel('time since the admission date (Hour)')
    ax1.set_ylabel("Creatinine in Serum or Plasma (%s)"%creatinine_unit, color = 'b')

    ax2 = ax1.twinx()
    ax2.plot(urea_time_diff, urea_value, 'rs--')
    ax2.set_ylabel("Blood Urea Nitrogen (%s)"%urea_unit, color = 'r')

    plt.title("ID:%d [%s]"%(patient_id, admission[2]))
    filename = "../data/CR_BUN_%d_%d.png"%(patient_id, admission[0])
    plt.savefig(filename)

## output_file_path = '../data/icustayevents.csv'
## admissions = mimic2db.icustay_events(patient_id, output_file_path);

## output_file_path = '../data/icustaydays.csv'
## admissions = mimic2db.icustay_days(patient_id, output_file_path);

## output_file_path = '../data/icustaydetail.csv'
## admissions = mimic2db.icustay_detail(patient_id, output_file_path);

## urea_time_diff = [(item-base_time).total_seconds()/3600 for item in urea_time]
## plt.plot(urea_time_diff, urea_value, 'bs--')
## plt.show()

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


