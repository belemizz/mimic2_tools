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
import mutil

from sklearn import cross_validation


mimic2db = control_mimic2db.control_mimic2db()
graphs = control_graph.control_graph()

def main( max_id = 200000,
          target_codes = ['428.0'],
          n_feature = 20,
          days_before_discharge = 2,
          pca_components = 5,
          ica_components = 5,
          dae_hidden = 40,
          dae_corruption = 0.0,
          n_cv_folds = 4):


    sw = mutil.stopwatch()

    experiment_code = "mid%d_tc%s_nf%d_dbd%d_pca%d_ica%d_da%d_%f"%(max_id, target_codes, n_feature, days_before_discharge, pca_components, ica_components, dae_hidden, dae_corruption)
    # Get candidate ids
    print "[INFO] Getting candidate IDs and their data"
    sw.reset()
    subject_ids, lab_ids_dict, patients, units, descs = get_patient_data_form_codes(target_codes, max_id)
    sw.print_real_elapsed(True)

    # Find most common lab tests
    print "[INFO] Finding most common lab tests"
    sw.reset()
    most_common_tests, lab_descs, lab_units = find_most_common_lab_tests(n_feature, lab_ids_dict, descs, units)
    sw.print_real_elapsed(True)
    
    # Get values of most commom tests
    sw.reset()
    print "[INFO] Getting values of lab and vital"
    lab_data, vital_data, flags = get_lab_chart_values( patients,
                                                        most_common_tests,
                                                        mimic2db.vital_charts,
                                                        days_before_discharge)
    print "Used/Candidate: %d / %d"%( lab_data.shape[0], len(subject_ids))
    sw.print_real_elapsed(True)

    print "[INFO] Getting timeseries of lab and vital"
    sw.reset()
    lab_ts, vital_ts, flags_ts = get_lab_chart_timeseries( patients,
                                                        most_common_tests,
                                                        mimic2db.vital_charts,
                                                        days_before_discharge,
                                                        2)
    sw.print_real_elapsed(True)
    

    # feature descriptions
    print "[INFO] eval_as_single_set"
    eval_as_single_set(lab_data, vital_data, flags, experiment_code,
                       most_common_tests, lab_descs, lab_units,
                       pca_components, ica_components,
                       dae_hidden, dae_corruption)

    
    print "[INFO] eval cross validation"
    eval_cross_validation(lab_data, vital_data, flags, experiment_code,
                          most_common_tests, lab_descs, lab_units,
                          pca_components, ica_components,
                          dae_hidden, dae_corruption, n_cv_folds)

def get_patient_data_form_codes(target_codes, max_id):
    id_list =  mimic2db.subject_with_icd9_codes(target_codes)
    subject_ids = [item for item in id_list if item < max_id]
    patients, lab_ids_dict, units, descs = get_patient_and_lab_info(subject_ids, target_codes)
    return subject_ids, lab_ids_dict, patients, units, descs

def find_most_common_lab_tests(n_feature, lab_ids_dict, descs,units):
    counter =  collections.Counter(lab_ids_dict)
    most_common_tests = [item[0] for item in counter.most_common(n_feature)]
    lab_descs = []
    lab_units = []
    for item_id in most_common_tests:
        lab_descs.append(descs[item_id])
        lab_units.append(units[item_id])
    return most_common_tests, lab_descs, lab_units

# Get the data of the tests
def get_lab_chart_values(patients, lab_ids, chart_ids, days_before_discharge):
    ids = []
    lab_values = []
    chart_values = []
    flags = []
    for patient in patients:
        final_adm = patient.get_final_admission()
#        time_of_interest = final_adm.disch_dt + datetime.timedelta(1-days_before_discharge)

        estimated_disch_time = max([final_adm.final_labs_time,
                                    final_adm.final_chart_time,
                                    final_adm.final_ios_time,
                                    final_adm.final_medication_time
                                    ])
        
        time_of_interest = estimated_disch_time + datetime.timedelta(1-days_before_discharge)
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

    return lab_array, chart_array, y

# Get the data of the tests
def get_lab_chart_timeseries(patients, lab_ids, chart_ids, days_before_discharge, span):
    ids = []
    lab_values = []
    chart_values = []
    flags = []
    for patient in patients:
        final_adm = patient.get_final_admission()
        estimated_disch_time = max([final_adm.final_labs_time,
                                    final_adm.final_chart_time,
                                    final_adm.final_ios_time,
                                    final_adm.final_medication_time
                                    ])
        
        time_of_interest_end = estimated_disch_time \
          + datetime.timedelta(1-days_before_discharge)
        time_of_interest_begin = time_of_interest_end \
          - datetime.timedelta(span)

        lab_result = final_adm.get_lab_in_span(time_of_interest_begin,
                                               time_of_interest_end)
        
        chart_result =  final_adm.get_chart_in_span(time_of_interest_begin,
                                                    time_of_interest_end)

        
        lab_value = [float('NaN')] * len(lab_ids)
        for item in lab_result:
            if item[0] in lab_ids and is_number_list(item[4]):
                index = lab_ids.index(item[0])
                lab_value[index] = float_list(item[4])

        chart_value = [float('NaN')] * len(chart_ids)
        for item in chart_result:
            if item[0] in chart_ids and is_number_list(item[4]):
                index = chart_ids.index(item[0])
                chart_value[index] = float_list(item[4])

        valid = True
        for item in lab_value + chart_value:
            if numpy.isnan(item).any():
                valid = False
                break


        if valid:
            lab_values.append(lab_value)
            chart_values.append(chart_value)
            flags.append(patient.hospital_expire_flg)
            ids.append(patient.subject_id)
    
    lab_array = numpy.array(lab_values)
    chart_array = numpy.array(chart_values)
    flag_array = numpy.array(flags)

    y = numpy.zeros(len(flag_array))
    y[flag_array == 'Y'] = 1

    return lab_array, chart_array, y


def eval_cross_validation(lab_data, vital_data, flags, experiment_code, most_common_tests, lab_descs, lab_units, pca_components, ica_components, dae_hidden, dae_corruption, n_cv_folds):
    
    # cross validation
    kf = cross_validation.KFold(lab_data.shape[0], n_folds = n_cv_folds, shuffle = True, random_state = 0)


    results = []
    
    for train, test in kf:

        # datasets
        set_train_lab = lab_data[train, :]
        set_train_vital = vital_data[test, :]
        flag_train = flags[train]

        set_test_lab = lab_data[test, :]
        set_test_vital = vital_data[test, :]
        flags_test = flags[test]

        encoded_values = alg_auto_encoder.get_encoded_values(
            set_train_lab, flag_train, set_test_lab,
            pca_components, 1,
            ica_components, 1,
            dae_hidden, 1, dae_corruption)
        
        feature_desc = encoded_values.keys()
        feature_id = range( len(feature_desc))
        feature_unit = ['None'] * len(feature_desc)
        feature_data = numpy.hstack(encoded_values.viewvalues())
        
        lab_importance = alg_feature_selection.calc_entropy_reduction(set_test_lab, flags_test, most_common_tests, lab_descs, lab_units)
        feature_importance = alg_feature_selection.calc_entropy_reduction(feature_data, flags_test, feature_id, feature_desc, feature_unit)

        lab_feature_importance = sorted(lab_importance + feature_importance, key = lambda item:item[2])

        results.append(lab_feature_importance)

    mean_reduction = alg_feature_selection.mean_entropy_reduction(results)

    print numpy.array(mean_reduction)
    feature_importance_graph(mean_reduction[0:20], graphs.dir_to_save + experiment_code + "_cv_lab_feature.png")


def eval_as_single_set(lab_data, vital_data, flags, experiment_code,
                            most_common_tests, lab_descs, lab_units,
                            pca_components, ica_components, dae_hidden, dae_corruption):

    encoded_values = alg_auto_encoder.get_encoded_values(
        lab_data, flags, lab_data,
        pca_components, 1,
        ica_components, 1,
        dae_hidden, 1,  dae_corruption)
    
    feature_desc = encoded_values.keys()
    feature_id = range( len(feature_desc))
    feature_unit = ['None'] * len(feature_desc)
    feature_data = numpy.hstack(encoded_values.viewvalues())

    lab_importance = alg_feature_selection.calc_entropy_reduction(lab_data, flags, most_common_tests, lab_descs, lab_units)
    vital_importance = alg_feature_selection.calc_entropy_reduction(vital_data, flags, mimic2db.vital_charts, mimic2db.vital_descs, mimic2db.vital_units)
    feature_importance = alg_feature_selection.calc_entropy_reduction(feature_data, flags, feature_id, feature_desc, feature_unit)

    all_importance = lab_importance + vital_importance
    all_importance.sort(reverse = True)
    lab_feature_importance = lab_importance + feature_importance
    lab_feature_importance.sort(reverse = True)

    feature_importance_graph(all_importance[0:20], graphs.dir_to_save + experiment_code + "_all.png")
    feature_importance_graph(lab_feature_importance[0:20], graphs.dir_to_save + experiment_code + "_lab_feature.png")

    # Classification with 2 most important features
    classify_important_feature(lab_importance, lab_data, flags, filename = graphs.dir_to_save+experiment_code + "_lab_imp.png")
    classify_important_feature(vital_importance, vital_data, flags, filename = graphs.dir_to_save+experiment_code + "_vital_imp.png")
    classify_important_feature(feature_importance, feature_data, flags, filename = graphs.dir_to_save+experiment_code + "_feature_imp.png")
    
#    lab_vs_feature = lab_importance[0:1] + feature_importance[0:1]
#    classify_important_feature(lab_importance, lab_data, flags, filename = graphs.dir_to_save+experiment_code + "_vs.png")
 
    ## # feature descriptions
    ## desc_labels = ["PCA%d"%item for item in range(1, pca_components+1)] + \
    ##   ["ICA%d"%item for item in range(1, ica_components+1)] + \
    ##   ["AE%d"%item for item in range(1, dae_hidden+1)]

    ## feature_ids = range(len(desc_labels))
    ## feature_descs = {}
    ## feature_units = {}
    ## for index in feature_ids:
    ##     feature_descs[index] = desc_labels[index]
    ##     feature_units[index] = 'None'

    ## # Get features vectors
    ## pca_value = alg_auto_encoder.pca(lab_data, lab_data, pca_components)
    ## ica_value = alg_auto_encoder.ica(lab_data, lab_data, ica_components)
    ## ae_value =  alg_auto_encoder.dae(lab_data, lab_data, 0.001, 2000, n_hidden = dae_hidden, corruption_level = dae_corruption)
    ## feature_data = numpy.hstack([pca_value, ica_value, ae_value])

    # Calc feature importance based on entropy reduction for all features

    
    # Feature Importance Graph
    
def classify_important_feature(lab_result, lab_data, flags, filename):
    important_labs = [lab_result[0][1], lab_result[1][1]]

    x_label = "%s [%s]"%(lab_result[0][3],lab_result[0][4])
    y_label = "%s [%s]"%(lab_result[1][3],lab_result[1][4])
    x = lab_data[:, important_labs]

    import alg_svm
    alg_svm.demo(x, flags, x_label, y_label)
#    alg_logistic_regression.show_logistic_regression(x, flags, 0.01, 10000, 10000, x_label = x_label, y_label = y_label, filename = filename)

def feature_importance_graph(importance, filename):
    ent_reduction = [item[0] for item in importance]
    labels = [item[3] for item in importance]
    graphs.bar_feature_importance(ent_reduction, labels, filename)



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

def is_number_list(l):
    for s in l:
        if not is_number(s):
            return False
    return True

def float_list(l):
    f_list= []
    for s in l:
        f_list.append(float(s))
    return numpy.array(f_list)

if __name__ == '__main__':
    main(days_before_discharge = 0)
    main(days_before_discharge = 2)
    plt.waitforbuttonpress()
