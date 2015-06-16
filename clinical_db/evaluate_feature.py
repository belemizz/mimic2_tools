"""
Evaluate the importance of the feature
"""

import numpy
import collections
import datetime
import matplotlib.pyplot as plt

import control_mimic2db
import control_graph
import alg_logistic_regression
import alg_auto_encoder
import alg_feature_selection

from sklearn import cross_validation

mimic2db = control_mimic2db.control_mimic2db()
graphs = control_graph.control_graph()

def main( max_id = 200000,
          target_codes = ['428.0'],
#          target_codes = ['518.0'],
          n_feature = 20,
          days_before_discharge = 2,
          pca_components = 10,
          ica_components = 10,
          da_hidden = 10,
          da_corruption = 0.3,
          n_cv_folds = 4):

    
    file_code = "mid%d_tc%s_nf%d_dbd%d_pca%d_ica%d_da%d_%f"%(max_id, target_codes, n_feature, days_before_discharge, pca_components, ica_components, da_hidden, da_corruption)

    # Get candidate ids
    id_list =  mimic2db.subject_with_icd9_codes(target_codes)
    subject_ids = [item for item in id_list if item < max_id]
    patients, lab_ids_dict, units, descs = get_patient_and_lab_info(subject_ids, target_codes)

    # Find most common lab tests
    counter =  collections.Counter(lab_ids_dict)
    most_common_tests = [item[0] for item in counter.most_common(n_feature)]
    # Get values of most commom tests
    lab_data, vital_data, flags, ids = get_lab_feature_values(patients,
                                                          most_common_tests,
                                                          mimic2db.vital_charts,
                                                          days_before_discharge)
    print "Number of Patients : %d / %d"%( len(ids),len(subject_ids))

    # feature descriptions
    evalueate_as_single_set(most_common_tests, descs, item, da_hidden, pca_components, lab_data, flags, ica_components, units, file_code, vital_data)

    # cross validation
    kf = cross_validation.KFold(lab_data.shape[0], n_folds = n_cv_folds, shuffle = True, random_state = 0)
    for train, test in kf:

        set_train_lab = lab_data[train, :]
        set_train_vital = vital_data[train, :]
        flag_train = flags[train]

        set_test_lab = lab_data[test, :]
        set_test_vital = vital_data[test, :]
        flags_test = flags[test]

        # feature descriptions
        desc_labels = ["PCA%d"%item for item in range(1, pca_components+1)] + \
          ["ICA%d"%item for item in range(1, ica_components+1)] + \
          ["AE%d"%item for item in range(1, da_hidden+1)]

        feature_ids = range(len(desc_labels))
        feature_descs = {}
        feature_units = {}
        for index in feature_ids:
            feature_descs[index] = desc_labels[index]
            feature_units[index] = 'None'

        ## Training
        pca_value = alg_auto_encoder.pca(set_train_lab, set_test_lab, pca_components)
        ica_value = alg_auto_encoder.ica(set_train_lab, set_test_lab, ica_components)
        ae_value = alg_auto_encoder.dae(set_train_lab, set_test_lab, 0.001, 2000, n_hidden = da_hidden)
        feature_data = numpy.hstack([pca_value, ica_value, ae_value])
        
        ## Testing
        lab_descs = []
        lab_units = []
        for item_id in most_common_tests:
            lab_descs.append(descs[item_id])
            lab_units.append(units[item_id])

        lab_importance = alg_feature_selection.calc_entropy_reduction(set_test_lab, flags_test, most_common_tests, lab_descs, lab_units)
        feature_importance = alg_feature_selection.calc_entropy_reduction(feature_data, flags_test, feature_ids, feature_descs, feature_units)

        lab_feature_importance = lab_importance + feature_importance
        lab_feature_importance.sort(reverse = True)

        print  numpy.array(lab_feature_importance)



def evalueate_as_single_set(most_common_tests, descs, item, da_hidden, pca_components, lab_data, flags, ica_components, units, file_code, vital_data):
    # feature descriptions
    desc_labels = ["PCA%d"%item for item in range(1, pca_components+1)] + \
      ["ICA%d"%item for item in range(1, ica_components+1)] + \
      ["AE%d"%item for item in range(1, da_hidden+1)]

    feature_ids = range(len(desc_labels))
    feature_descs = {}
    feature_units = {}
    for index in feature_ids:
        feature_descs[index] = desc_labels[index]
        feature_units[index] = 'None'


    # Get features vectors
    pca_value = alg_auto_encoder.pca(lab_data, lab_data, pca_components)
    ica_value = alg_auto_encoder.ica(lab_data, lab_data, ica_components)
    ae_value =  alg_auto_encoder.dae(lab_data, lab_data, 0.001, 2000, n_hidden = da_hidden)
    feature_data = numpy.hstack([pca_value, ica_value, ae_value])

    lab_descs = []
    lab_units = []
    for item_id in most_common_tests:
        lab_descs.append(descs[item_id])
        lab_units.append(units[item_id])

    # Calc feature importance based on entropy reduction for all features
    lab_importance = alg_feature_selection.calc_entropy_reduction(lab_data, flags, most_common_tests, lab_descs, lab_units)
    vital_importance = alg_feature_selection.calc_entropy_reduction(vital_data, flags, mimic2db.vital_charts, mimic2db.vital_descs, mimic2db.vital_units)
    feature_importance = alg_feature_selection.calc_entropy_reduction(feature_data, flags, feature_ids, feature_descs, feature_units)

    all_importance = lab_importance + vital_importance
    all_importance.sort(reverse = True)
    lab_feature_importance = lab_importance + feature_importance
    lab_feature_importance.sort(reverse = True)

    # Feature Importance Graph
    feature_importance_graph(all_importance[0:20], graphs.dir_to_save + file_code + "all.png")
    feature_importance_graph(lab_feature_importance[0:20], graphs.dir_to_save + file_code + "lab_feature.png")

    # Classification with 2 most important features
    classify_important_feature(lab_importance, lab_data, flags, filename = graphs.dir_to_save+file_code + "_lab_imp.png")
    classify_important_feature(vital_importance, vital_data, flags, filename = graphs.dir_to_save+file_code + "_vital_imp.png")
    classify_important_feature(feature_importance, feature_data, flags, filename = graphs.dir_to_save+file_code + "_feature_imp.png")

    plt.waitforbuttonpress()
    
def classify_important_feature(lab_result, lab_data, flags, filename):
    important_labs = [lab_result[0][1], lab_result[1][1]]

    x_label = "%s [%s]"%(lab_result[0][3],lab_result[0][4])
    y_label = "%s [%s]"%(lab_result[1][3],lab_result[1][4])
    x = lab_data[:, important_labs]

#    import alg_svm
#    alg_svm.demo(x, flags, x_label, y_label)
    alg_logistic_regression.show_logistic_regression(x, flags, 0.01, 1000, 1000, x_label = x_label, y_label = y_label, filename = filename)

def feature_importance_graph(importance, filename):
    ent_reduction = [item[0] for item in importance]
    labels = [item[3] for item in importance]
    graphs.bar_feature_importance(ent_reduction, labels, filename)

# Get the data of the tests
def get_lab_feature_values(patients, lab_ids, chart_ids, days_before_discharge):
    ids = []
    lab_values = []
    chart_values = []
    flags = []
    for patient in patients:
        final_adm = patient.get_final_admission()
        time_of_interest = final_adm.disch_dt + datetime.timedelta(1-days_before_discharge)
        lab_result =  final_adm.get_newest_lab_at_time(time_of_interest)
        chart_result =  final_adm.get_newest_chart_at_time(time_of_interest)
        
        lab_value = [float('NaN')] * len(lab_ids)
        for item in lab_result:
            if item[0] in lab_ids and is_number(item[4]):
                index = lab_ids.index(item[0])
                lab_value[index] = float(item[4])

        chart_value = [float('NaN')] * len(chart_ids)
        for item in chart_result:
            if item[0] in chart_ids and is_number(item[4]):
                index = chart_ids.index(item[0])
                chart_value[index] = float(item[4])

        if True not in numpy.isnan(lab_value) and True not in numpy.isnan(chart_value) and patient.hospital_expire_flg in ['Y', 'N']:
            lab_values.append(lab_value)
            chart_values.append(chart_value)
            flags.append(patient.hospital_expire_flg)
            ids.append(patient.subject_id)

    lab_array = numpy.array(lab_values)
    chart_array = numpy.array(chart_values)
    flag_array = numpy.array(flags)

    y = numpy.zeros(len(flag_array))
    y[flag_array == 'Y'] = 1

    return lab_array, chart_array, y, ids


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


if __name__ == '__main__':
    main()
