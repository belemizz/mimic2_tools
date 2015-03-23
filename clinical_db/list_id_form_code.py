tmppath = '../data/tmp.csv'

ignore_order = True
target_codes = ['428.0']
only_matched_waveform = True

def list_of(path):
    f = open(tmppath, 'rb')
    dataReader = csv.reader(f)

    id_list = []
    for row in dataReader:
        id_list.append(row[0])
    f.close()
    return id_list

import control_mimic2db as cm
import csv

mimic2db = cm.control_mimic2db()
id_lists = []

for code in target_codes:
    seq_cond = "<=%d"%len(target_codes)
    mimic2db.subject_with_icd9(code,seq_cond, tmppath)

    id_list = list_of(tmppath)
    id_lists.append(id_list)

mimic2db.subject_matched_waveforms(tmppath)
id_list = list_of(tmppath)
id_lists.append(id_list)

id_set = set(id_lists[0])
for index in range(1,len(id_lists)):
    id_set = id_set.intersection(set(id_lists[index]))
    
print id_set
