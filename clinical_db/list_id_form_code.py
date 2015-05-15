"""
Script to generate subject id list who have target icd9 code.
"""


tmppath = '../data/tmp.csv'
#outpath = '../data/out.csv'
outfolder = '../data/'

#target_codes = ['410.71','414.01','428.0']
# 410.71_428.0 191
# 414.01_428.0 120
# 424.1_428.0 91
# 424.0_428.0 43
# 410.11_428.0 42

# 428.0_427.31 57
# 428.0_425.4 38
# 428.0_518.81 36

target_codes = ['428.0', '518.81']
target_codes = ['428.0']
ignore_order = False
add_icu_expire_flag = True

outpath = outfolder + '_'.join(target_codes)
if ignore_order:
    outpath = outpath + '_io'
outpath = outpath + '.csv'

import control_mimic2db as mimic2
import control_csv as cc

mimic2db = mimic2.control_mimic2db()
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
id_list = sorted(list(id_set))
output_list = [id_list]

# add expire flag
if add_icu_expire_flag:
    mimic2db.subject_with_icu_expire_flg(tmppath)
    expire_full_list = tmp_csv.read_first_column()
    expire_list = []
    for subject in id_list:
        if subject in expire_full_list:
            expire_list.append('Y')
        else:
            expire_list.append('N')
    output_list.append(expire_list)
    print "Expire Rate: %d/%d"%(expire_list.count('Y'),len(expire_list))
  
# output to outpath
output = cc.control_csv(outpath)
output.write_list(output_list)
