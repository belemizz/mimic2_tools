"""
classify patients based on lab tests
"""
import get_sample.mimic2
from mutil import Graph

import mutil.mycsv

import time
import datetime
import random

import numpy as np
import theano
import theano.tensor as T
import alg.classification

def main( max_id = 2000, target_codes = ['428.0'], show_flag = True):

    mimic2db = get_sample.mimic2.Mimic2()
    graph = Graph()

    ## Get Subject ID ##
    id_list =  mimic2db.subject_with_icd9_codes(target_codes)
    subject_ids = [item for item in id_list if item < max_id]
    print "Number of Candidates : %d"%len(subject_ids)

    ## Get Data ##
    days_before_discharge = [0]
    recover_values = [[], [], [], []]
    expire_values = [[], [], [], []]

    start_time = time.clock()

    algo_num = 0
    time_diff = 4
    cr_id = 50090
    bun_id = 50177
    
    for str_id in subject_ids:
        sid = int(str_id)
        print sid
        patient = mimic2db.get_subject(sid)
        if patient:
            final_adm = patient.get_final_admission()

            if len(final_adm.icd9)>0 and final_adm.icd9[0][3] == target_codes[0]:

                for index, dbd in enumerate(days_before_discharge):

                    if algo_num == 0:
                        # bun_and_creatinine
                        time_of_interest = final_adm.disch_dt + datetime.timedelta(1-dbd)
                        lab_result =  final_adm.get_newest_lab_at_time(time_of_interest)
                        value1 = [item[4] for item in lab_result if item[0] == cr_id]
                        value2 = [item[4] for item in lab_result if item[0] == bun_id]
                    else:
                        # trend of BUN
                        time_of_interest1 = final_adm.disch_dt + datetime.timedelta(1-dbd)
                        time_of_interest2 = final_adm.disch_dt + datetime.timedelta(1-dbd-time_diff)
                        lab_result1 =  final_adm.get_newest_lab_at_time(time_of_interest1)
                        lab_result2 =  final_adm.get_newest_lab_at_time(time_of_interest2)
                        value1 = [item[4] for item in lab_result1 if item[0] == bun_id]
                        value2 = [item[4] for item in lab_result2 if item[0] == bun_id]
                        

                    if patient.hospital_expire_flg == 'Y':
                        expire_values[index].append([value1, value2])
                    else:
                        recover_values[index].append([value1, value2])

    end_time = time.clock()
    print "data_retrieving_time: %f sec"%(end_time - start_time)

    def transform_values(input_values):
        """ transform to numpy format """
        temp = []
        for item in input_values:
            if len(item[0])>0 and len(item[1])>0:
                temp.append([float(item[0][0]), float(item[1][0])])
        return np.array(temp)

    positive_x = transform_values(expire_values[0])
    negative_x = transform_values(recover_values[0])

    data = [[item, 1] for item in positive_x]
    data.extend([[item, 0] for item in negative_x])
    random.shuffle(data)

    x = np.array([item[0] for item in data])
    y = np.array([item[1] for item in data])

if __name__ == '__main__':
    main()
