savepath = '../data/tmp.csv'

n_code_to_see = 3
sort_flag = True
target_code = '428.0' #only count codes includes this code, set '' to ignore


import control_mimic2db as cm
mimic2db = cm.control_mimic2db()
mimic2db.icd9_eq_higher_than(n_code_to_see,savepath)

import csv
f = open(savepath, 'rb')
dataReader = csv.reader(f)

prev_adm_id = 0
all_codes = []
codes = []

for row in dataReader:
    if prev_adm_id == row[1]:# admission id
        codes.append(row[3])
    else:
        if sort_flag:
            joint_code = '_'.join(sorted(codes))
        else:
            joint_code = '_'.join(codes)
        
        all_codes.append(joint_code)

        codes=[row[3]]
        prev_adm_id = row[1]

codes_and_counts = {}
for code in all_codes:
    if target_code in code:
        if codes_and_counts.has_key(code):
            codes_and_counts[code] += 1
        else:
            codes_and_counts[code] = 1

for code, count in sorted(codes_and_counts.iteritems(), key=lambda x: x[1], reverse=False):
    print code, count


#code = '428.0'
#mimic2db.icd9_incl(code, savepath)
#mimic2db.icd9_of_subject(10, savepath)

#rank = 1
#mimic2db.count_icd9_eq_higher_than(rank, savepath)

#mimic2db.count_sequence_of_icd9_eq(code, savepath)
#mimic2db.count_entry_of("d_patients", savepath)

#mimic2db.count_icd9(savepath)

#mimic2db.subject_matched_waveforms(savepath)
