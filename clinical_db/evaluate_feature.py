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
import alg_logistic_regression
import alg_auto_encoder

mimic2db = control_mimic2db.control_mimic2db()
graphs = control_graph.control_graph()

#def main( max_id = 200000, target_codes = ['428.0'], n_feature = 23, dbd = 0):
def main( max_id = 200000, target_codes = ['518.81'], n_feature = 20, dbd = 0):

    # Get candidate ids
    id_list =  mimic2db.subject_with_icd9_codes(target_codes)
    subject_ids = [item for item in id_list if item < max_id]
    print "Number of Candidates : %d"%len(subject_ids)
    patients, lab_ids_dict, units, descs = get_patient_and_lab_info(subject_ids, target_codes)

    # Find most common lab tests
    counter =  collections.Counter(lab_ids_dict)
    most_common_tests = [item[0] for item in counter.most_common(n_feature)]
    print most_common_tests

    # Get values of most commom tests
    value_array, flag_array, ids = get_feature_values(patients, most_common_tests, dbd, n_feature)
    print "Number of Patients : %d"%len(ids)

    # Normalize
    from sklearn.preprocessing import normalize
    n_value_array = normalize(value_array, axis = 0)

    # Get PCA features
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 5)
    pca_value = pca.fit(n_value_array).transform(n_value_array)

    from sklearn.decomposition import FastICA
    ica = FastICA(n_components = 5)
    ica_value = ica.fit(n_value_array, flag_array).transform(n_value_array)

    from sklearn.lda import LDA
    lda = LDA(n_components = 1)
    lda_value = lda.fit(n_value_array, flag_array).transform(n_value_array)

    ae_value = alg_auto_encoder.demo(n_value_array, 0.001, 1000)

    value_array = numpy.hstack([value_array, pca_value, ica_value, lda_value, ae_value])
    desc_labels = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'ICA1', 'ICA2', 'ICA3','ICA4', 'ICA5', 'LDA', 'AE1', 'AE2']

    for index, item in enumerate(desc_labels):
        most_common_tests.append(-index)
        units[-index] = 'None'
        descs[-index] = desc_labels[index]


    # Calc entropy reduction for all features
    result = calc_entropy_reduction(value_array, flag_array, most_common_tests, descs, units)
    print numpy.array(result)

    # Feature Importance Graph
    ent_reduction = [item[0] for item in result]
    labels = [item[3] for item in result]
    graphs.bar_feature_importance(ent_reduction, labels)

    # Classification with 2 most important features
    important_labs = [result[0][1], result[1][1]]
    x_label = "%s [%s]"%(result[0][3],result[0][4])
    y_label = "%s [%s]"%(result[1][3],result[1][4])

    x = value_array[:, important_labs]
    alg_logistic_regression.show_logistic_regression(x, flag_array, 0.001, 10000, 10000, x_label = x_label, y_label = y_label)

# Get the data of the tests
def get_feature_values(patients, lab_ids, days_before_discharge, n_feature):
    ids = []
    values = []
    flags = []
    for patient in patients:
        final_adm = patient.get_final_admission()
        time_of_interest = final_adm.disch_dt + datetime.timedelta(1-days_before_discharge)
        lab_result =  final_adm.get_newest_lab_at_time(time_of_interest)

        value = [float('NaN')] * n_feature
        for item in lab_result:
            if item[0] in lab_ids and is_number(item[4]):
                index = lab_ids.index(item[0])
                value[index] = float(item[4])

        if True not in numpy.isnan(value) and patient.hospital_expire_flg in ['Y', 'N']:
            values.append(value)
            flags.append(patient.hospital_expire_flg)
            ids.append(patient.subject_id)

    value_array = numpy.array(values)
    flag_array = numpy.array(flags)
    y = numpy.zeros(len(flag_array))
    y[flag_array == 'Y'] = 1

    return value_array, y, ids

# Calcurate entorpy reduction by each feature
def calc_entropy_reduction(value_array, flag_array, most_common_tests, descs, units):
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
    return result

# Get subject info and lab_id info
def get_patient_and_lab_info(subject_ids, target_codes):
    patients = []
    units = {}
    descs = {}
    lab_ids_dict = {}

    for subject_id in subject_ids:
        patient = mimic2db.get_subject(subject_id)
        if patient:
            final_adm = patient.get_final_admission()
            if len(final_adm.icd9)>0 and final_adm.icd9[0][3] == target_codes[0]:
                patients.append(patient)

                for item in final_adm.labs:
                    if item.itemid in lab_ids_dict:
                        lab_ids_dict[item.itemid] = lab_ids_dict[item.itemid] + 1
                    else:
                        lab_ids_dict[item.itemid] = 1
                        units[item.itemid] = item.unit
                        descs[item.itemid] = item.description
    return patients, lab_ids_dict, units, descs

def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def entropy(flags):
    counter =  collections.Counter(flags)

    if len(counter) > 2:
        raise ValueError("Flags should be binary values")

    if counter[0] == 0 or counter[1] == 0:
        entropy = 0.
    else:
        pi = float(counter[0]) / float(counter[0] + counter[1])
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
