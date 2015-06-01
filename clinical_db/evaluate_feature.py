"""
Evaluate the importance of the feature
"""


import numpy
import collections
import datetime
import math
import matplotlib.pyplot as plt

import control_mimic2db
import control_graph

#def main( max_id = 200000, target_codes = ['518.81'], n_feature = 10):
def main( max_id = 200000, target_codes = ['518.81'], n_feature = 20):

    mimic2db = control_mimic2db.control_mimic2db()
    graphs = control_graph.control_graph()

    # Get candidate ids
    id_list =  mimic2db.subject_with_icd9_codes(target_codes)
    subject_ids = [item for item in id_list if item < max_id]
    print "Number of Candidates : %d"%len(subject_ids)

    # Get subject info and lab_id info
    patients = []
    lab_ids = []
    units = {}
    descs = {}
    lab_ids_dict = {}

    for subject_id in subject_ids:
        patient = mimic2db.get_subject(subject_id)
        if patient:
            final_adm = patient.get_final_admission()
            if len(final_adm.icd9)>0 and final_adm.icd9[0][3] == target_codes[0]:
                patients.append(patient)
                lab_ids = lab_ids+[item.itemid for item in final_adm.labs]

                for item in final_adm.labs:
                    if item.itemid in lab_ids_dict:
                        lab_ids_dict[item.itemid] = lab_ids_dict[item.itemid] + 1
                    else:
                        lab_ids_dict[item.itemid] = 1
                        units[item.itemid] = item.unit
                        descs[item.itemid] = item.description
                        
    # Find most common lab tests
    counter =  collections.Counter(lab_ids_dict)
    most_common_tests = [item[0] for item in counter.most_common(n_feature)]
    print most_common_tests

    # Get the data of the most common tests
    dbd = 4
    ids = []
    values = []
    flags = []
    for patient in patients:
        final_adm = patient.get_final_admission()
        time_of_interest = final_adm.disch_dt + datetime.timedelta(1-dbd)
        lab_result =  final_adm.get_newest_lab_at_time(time_of_interest)

        value = [float('NaN')] * n_feature
        for item in lab_result:
            if item[0] in most_common_tests and is_number(item[4]):
                index = most_common_tests.index(item[0])
                value[index] = float(item[4])

        if True not in numpy.isnan(value):
            values.append(value)
            flags.append(patient.hospital_expire_flg)
            ids.append(patient.subject_id)

    value_array = numpy.array(values)
    flag_array = numpy.array(flags)
    print "Number of Patients : %d"%len(ids)
    print value_array
        
    # Calcurate entorpy reduction by each feature
    orig_entropy = entropy(flag_array)
    print "Original entropy: %f"%orig_entropy
    result = [];
    for index in xrange(value_array.shape[1]):
        opt_entropy, threshold =  entropy_after_optimal_divide(flag_array, value_array[:,index])
        result.append((orig_entropy - opt_entropy,
                       index,
                       most_common_tests[index],
                       descs[most_common_tests[index]],
                       units[most_common_tests[index]],
                ))
    result.sort(reverse = True)
    print numpy.array(result)

    # Graph
    ent_reduction = [item[0] for item in result]
    labels = [item[3] for item in result]
    graphs.bar_feature_importance(ent_reduction, labels)
    
    # Classification by 2 most important features
    important_labs = [result[0][1], result[1][1]]
    x_label = "%s [%s]"%(result[0][3],result[0][4])
    y_label = "%s [%s]"%(result[1][3],result[1][4])
    x = value_array[:, important_labs]
    y = numpy.zeros(len(flags))
    y[flag_array == 'Y'] = 1
    
    import alg_logistic_regression
    alg_logistic_regression.show_logistic_regression(x, y, 0.001, 10000, 10000, x_label, y_label)

def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
    
def entropy(flags):
    counter =  collections.Counter(flags)
    # check flags
    proper_list= ['Y', 'N']
    for item in counter.keys():
        if item not in proper_list:
            raise ValueError

    if counter['N'] == 0 or counter['Y'] == 0:
        entropy = 0.
    else:
        pi = float(counter['N']) / float(counter['N'] + counter['Y'])
        entropy =  - 0.5 * (pi * math.log(pi,2) + (1. - pi) * math.log(1. - pi, 2))
    return entropy

def entropy_after_divide(flag, value, threshold):
    flag_r = flag[value <= threshold]
    flag_l = flag[value > threshold]
    return entropy(flag_r) + entropy(flag_l)

def entropy_after_optimal_divide(flag, value):
    min_entropy = numpy.inf
    opt_th = 0
    for item in value:
        t_entropy = entropy_after_divide(flag, value, item)
        if t_entropy < min_entropy:
            opt_th = item
            min_entropy = t_entropy

    return min_entropy, opt_th


if __name__ == '__main__':
    main()
