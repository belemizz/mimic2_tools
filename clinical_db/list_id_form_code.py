tmppath = '../data/tmp.csv'
outpath = '../data/out.csv'
target_codes = ['410.71','414.01','428.0']

ignore_order = True

import control_mimic2db as cm
import control_csv as cc

mimic2db = cm.control_mimic2db()
id_lists = []

# extract subjects who have each target code
for index, code in enumerate(target_codes):
    if ignore_order:
        seq_cond = "<=%d"%len(target_codes)
    else:
        seq_cond = "=%d"%(index+1)
    mimic2db.subject_with_icd9(code,seq_cond, tmppath)

    tmp_csv = cc.control_csv(tmppath);
    id_list = tmp_csv.read_first_column();
    id_lists.append(id_list)

# extract subjects who have matched waveform
mimic2db.subject_matched_waveforms(tmppath)
id_list = tmp_csv.read_first_column()
id_lists.append(id_list)

# find intersection of extracted subjects
id_set = set(id_lists[0])
for index in range(1,len(id_lists)):
    id_set = id_set.intersection(set(id_lists[index]))
    
# output to outpath
output = cc.control_csv(outpath)
output.write_single_list(sorted(list(id_set)))
