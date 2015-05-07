"""
Script to show summery of  medical record of a patient.
"""

import control_mimic2db
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm


def lab_event_extract(result, admission_id, ronic_code):
    ex_record = [item for item in result if (item[1] == admission_id and item[12] == ronic_code)]

    ex_time = [item[4] for item in ex_record]
    ex_value = [item[5] for item in ex_record]
    ex_unit = ex_record[0][8]
    return (ex_time, ex_value, ex_unit)

def time_diff_in_hour(time_seq, base_time):
    return [(item - base_time).total_seconds()/3600 for item in time_seq]

def icustay_detail_in_admission(admission):
    return [item for item in icustay_detail if item[8] == admission[0]]

def draw_lab_dual_graph(base_time, itemid1, itemid2,title="", filename = ""):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    result = admission.get_lab_itemid(itemid1)
    time_diff = time_diff_in_hour(result[3], base_time)
    values = result[4]

    ax1.plot(time_diff, values, 'bs--')
    ax1.set_ylabel("%s (%s)"%(result[1], result[2]), color = 'b')
    ax1.set_xlabel('time since the admission date (Hour)')

    result = admission.get_lab_itemid(itemid2)
    time_diff = time_diff_in_hour(result[3], base_time)
    values = result[4]

    ax2.plot(time_diff, values, 'rs--')
    ax2.set_ylabel("%s (%s)"%(result[1], result[2]), color = 'r')
    
    plt.title(title)
    plt.savefig(filename)    
    plt.show()


def draw_med_graph(icustay, title, filename):
    result = icustay.medications
    base = icustay.intime

    item = result[0]
    print item

    for item in result:
        time_diff = time_diff_in_hour(item[4], base)

        print item[5]
        if max(item[5]) > 100:
            value = np.array(item[5])/1000
        elif max(item[5]) > 10:
            value = np.array(item[5])/100
        elif max(item[5]) > 1:
            value = np.array(item[5])/10
        else:
            value = np.array(item[5])

        plt.plot(time_diff,value, 'o', label = item[1])
    
    plt.title(title)
    plt.legend()
    plt.show()


#subject_id = 1855
subject_id = 1916
mimic2db = control_mimic2db.control_mimic2db()
patient = mimic2db.patient_class(subject_id)

for admission in patient.admissions:
    base_time = admission.admit_dt

    lab_itemid1 = 50090
    lab_itemid2 = 50177
    filename = "../data/CR_BUN_%d_%d.png"%(subject_id, admission.hadm_id)
    title = "ID:%d [%s]"%(subject_id, admission.admit_dt)
    draw_lab_dual_graph(base_time, lab_itemid1, lab_itemid2, title, filename)

    for icustay in admission.icustays:
        icustay_base_time = icustay.intime
        filename = "../data/Med_%d.png"%icustay.icustay_id
        title = "ID:%d [%s]"%(subject_id, icustay.intime)
        draw_med_graph(icustay, title, filename)
        
        

## # Get Data
## patient = mimic2db.patient(subject_id)
## admissions = mimic2db.admission(subject_id)
## icustay_detail = mimic2db.icustay_detail(subject_id)

## icd9 = mimic2db.icd9(subject_id)
## med_events  = mimic2db.med_events(subject_id)
## lab_events  = mimic2db.lab_events(subject_id)
## note_events = mimic2db.note_events(subject_id)


## #io_events   = mimic2db.io_events(subject_id)

## for admission in admissions:


##     icd9_in_adm = [item for item in icd9 if item[1] == admission[0]]
##     icustay_detail_in_adm = icustay_detail_in_admission(admission)

##     icustay_ids_in_adm = [item[0] for item in icustay_detail_in_adm]

##     med_events_in_adm = [item for item in med_events if item[1] in icustay_ids_in_adm]
##     lab_events_in_adm = [item for item in lab_events if item[1] == admission[0]]
##     note_events_in_adm = [item for item in note_events if item[1] == admission[0]]

##     # save in a file
##     filename = "../data/Summery_%d_%d.csv"%(subject_id, admission[0])
##     import csv
##     writer = csv.writer(open(filename,'wb'))
##     elements_to_save = [patient, [admission], icd9_in_adm, icustay_detail_in_adm, lab_events_in_adm, med_events_in_adm, note_events_in_adm]
##     for item in elements_to_save:
##         writer.writerows(item)
##         writer.writerow("")

##     # graph of creatinine and urea nitrogen
##     creatinine_time, creatinine_value, creatinine_unit = lab_event_extract(lab_events, admission[0], '2160-0')
##     creatinine_time_diff = time_diff_in_hour(creatinine_time, base_time);

##     urea_time, urea_value, urea_unit = lab_event_extract(lab_events, admission[0], '3094-0')
##     urea_time_diff = time_diff_in_hour(urea_time, base_time);    

##     fig, ax1 = plt.subplots()
##     ax1.plot(creatinine_time_diff, creatinine_value, 'bs--')
##     ax1.set_xlabel('time since the admission date (Hour)')
##     ax1.set_ylabel("Creatinine in Serum or Plasma (%s)"%creatinine_unit, color = 'b')

##     ax2 = ax1.twinx()
##     ax2.plot(urea_time_diff, urea_value, 'rs--')
##     ax2.set_ylabel("Blood Urea Nitrogen (%s)"%urea_unit, color = 'r')

##     plt.title("ID:%d [%s]"%(subject_id, admission[2]))
##     filename = "../data/CR_BUN_%d_%d.png"%(subject_id, admission[0])
##     plt.savefig(filename)

## ## output_path = '../data/icustayevents.csv'
## ## admissions = mimic2db.icustay_events(subject_id, output_path);

## ## output_path = '../data/icustaydays.csv'
## ## admissions = mimic2db.icustay_days(subject_id, output_path);

## ## output_path = '../data/icustaydetail.csv'
## ## admissions = mimic2db.icustay_detail(subject_id, output_path);

## ## urea_time_diff = [(item-base_time).total_seconds()/3600 for item in urea_time]
## ## plt.plot(urea_time_diff, urea_value, 'bs--')
## ## plt.show()

## ## output_path = '../data/d_patients.csv'
## ## mimic2db.patient(subject_id, output_path)

## ## output_path = '../data/icd9.csv'
## ## mimic2db.icd9(subject_id, output_path)

## ## output_path = '../data/med_events.csv'
## ## mimic2db.med_events(subject_id, output_path)

## ## output_path = '../data/note_events.csv'
## ## mimic2db.note_events(subject_id, output_path)

## ## output_path = '../data/poe_order.csv'
## ## mimic2db.poe_order(subject_id, output_path)

## ## output_path = '../data/microbiologyevents.csv'
## ## mimic2db.microbiology_events(subject_id, output_path)
