'''
show distirbution of selected lab tests
'''

import control_mimic2db
import control_graph
import control_csv as cc

import matplotlib.pyplot as plt
import datetime

import pdb

mimic2db = control_mimic2db.control_mimic2db()
graph = control_graph.control_graph()

target_codes = ['428.0']
tmppath = '../data/tmp.csv'
ignore_order = True
# extract subjects who have each target code
for index, code in enumerate(target_codes):
    if ignore_order:
        seq_cond = "<=%d"%len(target_codes)
    else:
        seq_cond = "=%d"%(index+1)
    mimic2db.subject_with_icd9(code,seq_cond, tmppath)

    tmp_csv = cc.control_csv(tmppath);
    id_list = tmp_csv.read_first_column();

subject_ids = id_list
print subject_ids

days_before_discharge = [0, 1, 2, 3]
recover_values = [[], [], [], []]
expire_values = [[], [], [], []]

for str_id in subject_ids:
    sid = int(str_id)
    patient = mimic2db.patient_class(sid)
    if patient:
        final_adm = patient.get_final_admission()

        if len(final_adm.icd9)>0 and final_adm.icd9[0][3] == target_codes[0]:

            for index, dbd in enumerate(days_before_discharge):
                time_of_interest = final_adm.disch_dt + datetime.timedelta(1-dbd)
                lab_result =  final_adm.get_newest_lab_at_time(time_of_interest)
                cr_id = 50090
                cr_value = [item[4] for item in lab_result if item[0] == cr_id] 
                bun_id = 50177
                bun_value = [item[4] for item in lab_result if item[0] == bun_id]

                if patient.hospital_expire_flg == 'Y':
                    expire_values[index].append([cr_value, bun_value])
                else:
                    recover_values[index].append([cr_value, bun_value])

for index, dbd in enumerate(days_before_discharge):
    title = "Lab Tests %d days before discharge"%dbd
    filename = "../data/Lab_Dist_%d_days_before_disch.png"%dbd
    graph.draw_lab_distribution(expire_values[index], recover_values[index], title, filename)
