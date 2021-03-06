"""Count the frequency of the icd 9 codes"""

from get_sample import Mimic2
import csv
savepath = '../data/tmp.csv'

n_code_to_see = 2
ignore_order = False
target_code = '428.0'  # only count codes includes this code, set '' to ignore

# select icd code entries
mimic2db = Mimic2()
mimic2db.icd9_eq_higher_than(n_code_to_see, savepath)

# read csv and list code
f = open(savepath, 'rb')
dataReader = csv.reader(f)

prev_adm_id = 0
all_codes = []
codes = []
for row in dataReader:
    if prev_adm_id == row[1]:  # admission id
        codes.append(row[3])
    else:
        if ignore_order:
            joint_code = '_'.join(sorted(codes))
        else:
            joint_code = '_'.join(codes)

        all_codes.append(joint_code)

        codes = [row[3]]
        prev_adm_id = row[1]
f.close()

# count code sequence
codes_and_counts = {}
for code in all_codes:
    if target_code in code:
        if code in codes_and_counts:
            codes_and_counts[code] += 1
        else:
            codes_and_counts[code] = 1

for code, count in sorted(codes_and_counts.iteritems(), key=lambda x: x[1], reverse=False):
    print code, count
