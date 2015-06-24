"""
Evaluate the importance of the feature
"""

import numpy
import collections
import datetime
import matplotlib.pyplot as plt

import control_mimic2db
import control_graph
import alg_svm
import alg_logistic_regression
import alg_auto_encoder
import alg_feature_selection
import mutil

from sklearn import cross_validation


mimic2db = control_mimic2db.control_mimic2db()
graphs = control_graph.control_graph()

class evaluate_fetaure:
    def __init__( self,
                  max_id = 200000,
                  target_codes = ['428.0'],
                  n_lab = 20,
                  days_before_discharge = 2,
                  span = 2,
                  rp_learn_flag = True,
                  pca_components = 5,
                  ica_components = 5,
                  dae_hidden = 20,
                  dae_corruption = 0.0,
                  n_cv_folds = 4):
        self.max_id = max_id
        self.target_codes = target_codes
        self.n_lab = n_lab
        self.days_before_discharge = days_before_discharge
        self.span = span
        self.rp_learn_flag = rp_learn_flag
        self.pca_components = pca_components
        self.ica_components = ica_components
        self.dae_hidden = dae_hidden
        self.dae_corruption = dae_corruption
        self.n_cv_folds = n_cv_folds


    def compare_dbd(self, dbd_list):
        dbd_temp = self.days_before_discharge

        result = []
        for dbd in dbd_list:
            self.days_before_discharge = dbd
            result.append(self.point_eval())
            
        self.days_before_discharge = dbd_temp

        return result

    def compare_dae_hidden(self, n_list):
        dae_hidden_temp = self.dae_hidden
        result = []
        for dae_hidden in n_list:
            self.dae_hidden = dae_hidden
            result.append(self.point_eval())
        self.dae_hidden = dae_hidden_temp
        import ipdb
        ipdb.set_trace()
        return result

    def compare_dae_corruption(self, n_list):
        dae_corruption_temp = self.dae_corruption
        result = []
        for dae_corruption in n_list:
            self.dae_corruption = dae_corruption
            result.append(self.point_eval())
        self.dae_corruption = dae_corruption_temp
        import ipdb
        ipdb.set_trace()
        return result
        
    def point_eval(self):
        [most_common_tests, lab_data, lab_descs, lab_units, vital_data, flags] = self.__point_data_preperation()
        if self.n_cv_folds == 1:
            print "[INFO] eval_as_single_set"
            return self.__eval_as_single_set(
                lab_data, vital_data, flags,
                most_common_tests, lab_descs, lab_units,
                )
        elif self.n_cv_folds > 1:
            print "[INFO] eval cross validation"
            result =  self.__eval_cross_validation(
                lab_data, vital_data, flags, 
                most_common_tests, lab_descs, lab_units)

            dae_score = [item[0] for item in result if item[2] == 1][0]
            bun_score = [item[0] for item in result if item[2] == 50177][0]
            return [dae_score, bun_score]
        else:
            print 'n_cv_fold is not valid'

    def __point_data_preperation(self, cache_key = '__point_data_preperation'):

        param = self.__dict__.copy()
        del param['rp_learn_flag']
        del param['pca_components']
        del param['ica_components']
        del param['dae_hidden']
        del param['dae_corruption']
        del param['n_cv_folds']

        cache = mutil.cache(cache_key)

        try:
            return cache.load( param)
        except IOError:
            # Get candidate ids
            print "[INFO] Getting candidate IDs and their data"
            subject_ids, lab_ids_dict, patients, units, descs = self.__get_patient_data_form_codes()
            # Find most common lab tests
            print "[INFO] Finding most common lab tests"
            most_common_tests, lab_descs, lab_units = self.__find_most_common_lab_tests(lab_ids_dict, descs, units)
            # Get values of most commom tests
            print "[INFO] Getting values of lab and vital"
            lab_data, vital_data, flags = self.__get_lab_chart_values( patients,
                                                                most_common_tests,
                                                                mimic2db.vital_charts)
            ret_val =  [most_common_tests, lab_data, lab_descs, lab_units, vital_data, flags]
            return cache.save(ret_val, param)

    def __eval_as_single_set(self, lab_data, vital_data, flags,
                           most_common_tests, lab_descs, lab_units):

        ## Raw Data Evaluation
        lab_importance = alg_feature_selection.calc_entropy_reduction(lab_data, flags, most_common_tests, lab_descs, lab_units)
        vital_importance = alg_feature_selection.calc_entropy_reduction(vital_data, flags, mimic2db.vital_charts, mimic2db.vital_descs, mimic2db.vital_units)
        all_importance = lab_importance + vital_importance
        all_importance.sort(reverse = True)
        self.__feature_importance_graph(all_importance[0:20], graphs.dir_to_save + self.__param_code() + "_all.png")
    #    self.__classify_important_feature(lab_importance, lab_data, flags, filename = graphs.dir_to_save+experiment_code + "_lab_imp.png")
    #    self.__classify_important_feature(vital_importance, vital_data, flags, filename = graphs.dir_to_save+experiment_code + "_vital_imp.png")

        if self.rp_learn_flag:
            ## Encoded Feature Evalation
            encoded_values = alg_auto_encoder.get_encoded_values(
                lab_data, flags, lab_data,
                self.pca_components, 1,
                self.ica_components, 1,
                self.dae_hidden, 1,  self.dae_corruption)

            feature_desc = encoded_values.keys()
            feature_id = range( len(feature_desc))
            feature_unit = ['None'] * len(feature_desc)
            feature_data = numpy.hstack(encoded_values.viewvalues())
            feature_importance = alg_feature_selection.calc_entropy_reduction(feature_data, flags, feature_id, feature_desc, feature_unit)

            lab_feature_importance = lab_importance + feature_importance
            lab_feature_importance.sort(reverse = True)

            all_importance = lab_importance + feature_importance + vital_importance
            all_importance.sort(reverse = True)

            self.__feature_importance_graph(lab_feature_importance[0:20], graphs.dir_to_save + self.__param_code() + "_lab_feature.png")
            self.__classify_important_feature(feature_importance, feature_data, flags, filename = graphs.dir_to_save+self.__param_code() + "_feature_imp.png")

        return all_importance

        
    def __eval_cross_validation(self, lab_data, vital_data, flags, most_common_tests, lab_descs, lab_units):

        # cross validation
        kf = cross_validation.KFold(lab_data.shape[0], n_folds = self.n_cv_folds, shuffle = True, random_state = 0)

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
                self.pca_components, 1,
                self.ica_components, 1,
                self.dae_hidden, 1, self.dae_corruption)

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
        self.__feature_importance_graph(mean_reduction[0:20], graphs.dir_to_save + self.__param_code() + "_cv_lab_feature.png")
        return mean_reduction

    def __ts_eval(self):
        print 'not implemented'

    def __get_patient_data_form_codes(self):
        id_list =  mimic2db.subject_with_icd9_codes(self.target_codes)
        subject_ids = [item for item in id_list if item < self.max_id]
        patients, lab_ids_dict, units, descs = self.__get_patient_and_lab_info(subject_ids)
        return subject_ids, lab_ids_dict, patients, units, descs

    # Get subject info and lab_id info
    def __get_patient_and_lab_info(self,subject_ids):
        patients = []
        units = {}
        descs = {}
        lab_ids_dict = {}

        for subject_id in subject_ids:
            patient = mimic2db.get_subject(subject_id)
            if patient:
                final_adm = patient.get_final_admission()
                if len(final_adm.icd9)>0 and final_adm.icd9[0][3] == self.target_codes[0]:
                    patients.append(patient)

                    for item in final_adm.labs:
                        if item.itemid in lab_ids_dict:
                            lab_ids_dict[item.itemid] = lab_ids_dict[item.itemid] + 1
                        else:
                            lab_ids_dict[item.itemid] = 1
                            units[item.itemid] = item.unit
                            descs[item.itemid] = item.description
        return patients, lab_ids_dict, units, descs

    def __find_most_common_lab_tests(self, lab_ids_dict, descs,units):
        counter =  collections.Counter(lab_ids_dict)
        most_common_tests = [item[0] for item in counter.most_common(self.n_lab)]
        lab_descs = []
        lab_units = []
        for item_id in most_common_tests:
            lab_descs.append(descs[item_id])
            lab_units.append(units[item_id])
        return most_common_tests, lab_descs, lab_units
    
    # Get the data of the tests
    def __get_lab_chart_values(self, patients, lab_ids, chart_ids):
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

            time_of_interest = estimated_disch_time + datetime.timedelta(1-self.days_before_discharge)
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

    def __feature_importance_graph(self,importance, filename):
        ent_reduction = [item[0] for item in importance]
        labels = [item[3] for item in importance]
        graphs.bar_feature_importance(ent_reduction, labels, filename)

    def __classify_important_feature(self,lab_result, lab_data, flags, filename):
        important_labs = [lab_result[0][1], lab_result[1][1]]

        x_label = "%s [%s]"%(lab_result[0][3],lab_result[0][4])
        y_label = "%s [%s]"%(lab_result[1][3],lab_result[1][4])
        x = lab_data[:, important_labs]

        alg_svm.demo(x, flags, x_label, y_label)

    # Get the data of the tests
    def __get_lab_chart_timeseries(self, patients, lab_ids, chart_ids):
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
              + datetime.timedelta(1-self.days_before_discharge)
            time_of_interest_begin = time_of_interest_end \
              - datetime.timedelta(self.span)

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


    def __param_code(self):
        return "mid%d_tc%s_nf%d_dbd%d_pca%d_ica%d_da%d_%f"%(self.max_id,
                                                            self.target_codes,
                                                            self.n_lab,
                                                            self.days_before_discharge,
                                                            self.pca_components,
                                                            self.ica_components,
                                                            self.dae_hidden,
                                                            self.dae_corruption)



    

############ UTILITY ###################
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


#if __name__ == '__main__':
