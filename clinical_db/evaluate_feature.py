"""
Evaluate the importance of the feature
"""

import numpy
import collections 
import datetime
import matplotlib.pyplot as plt

import control_mimic2db
import control_graph
import alg_classification
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
                  days_before_discharge = 0,
                  span = 2,
                  rp_learn_flag = True,
                  pca_components = 5,
                  ica_components = 5,
                  dae_hidden = 40,
                  dae_corruption = 0.3,
                  dae_n_epoch = 20,
                  n_cv_folds = 4,
                  classification = False,
                  class_alg = 'dt'):
        
        # params for data retrieval
        self.max_id = max_id
        self.target_codes = target_codes
        self.n_lab = n_lab
        self.days_before_discharge = days_before_discharge
        self.span = span
        
        # params for evaluation
        self.rp_learn_flag = rp_learn_flag
        self.pca_components = pca_components
        self.ica_components = ica_components
        self.dae_hidden = dae_hidden
        self.dae_corruption = dae_corruption
        self.dae_n_epoch = dae_n_epoch
        self.n_cv_folds = n_cv_folds
        self.classification = classification
        self.class_alg = class_alg

    def __get_param_data_retrieval(self):
        return {'max_id' : self.max_id,
                'target_codes' : self.target_codes,
                'n_lab' : self.n_lab,
                'days_before_discharge' : self.days_before_discharge,
                'span' : self.span
                }

    def __param_code(self):
        return "%s"%self.__dict__.values()

    def compare_dbd(self, dbd_list):

        dbd_temp = self.days_before_discharge

        result = []
        for dbd in dbd_list:
            self.days_before_discharge = dbd
            result.append(self.point_eval())
            
        self.days_before_discharge = dbd_temp

        data = []
        for i, item_id in enumerate(mimic2db.vital_charts):
            scores = []
            for r in result:
                scores.append([item[0] for item in r if item[2] == item_id][0])
            data.append(scores)
        
        bun_score = []
        for r in result:
            bun_score.append([item[0] for item in r if item[2] == 50177][0])
        data.append(bun_score)

        label = mimic2db.vital_descs + ['BUN']

        graphs.line_series(data, dbd_list, label,
                           x_label = "Days before discharge", y_label = "Entropy Reduction",
                           filename = self.__param_code() + '_time.png' )
        return result

    def compare_dae_hidden(self, n_list):
        dae_hidden_temp = self.dae_hidden
        result = []
        for dae_hidden in n_list:
            self.dae_hidden = dae_hidden
            result.append(self.point_eval())
        self.dae_hidden = dae_hidden_temp
        return result

    def compare_dae_corruption(self, n_list):
        dae_corruption_temp = self.dae_corruption
        result = []
        for dae_corruption in n_list:
            self.dae_corruption = dae_corruption
            result.append(self.point_eval())
        self.dae_corruption = dae_corruption_temp
        return result

    def __point_data_preperation(self, cache_key = '__point_data_preperation'):

        param = self.__get_param_data_retrieval()
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

        
    def point_eval(self):
        ret_val = {'param': self.__dict__.copy()}
        
        print self.__param_code()

        """ Basic evaluation method """
        [most_common_tests, lab_data, lab_descs, lab_units, vital_data, flags] = self.__point_data_preperation()

        ## importance of each metrics
        lab_importance = alg_feature_selection.calc_entropy_reduction(lab_data, flags, most_common_tests, lab_descs, lab_units)
        vital_importance = alg_feature_selection.calc_entropy_reduction(vital_data, flags, mimic2db.vital_charts, mimic2db.vital_descs, mimic2db.vital_units)
        all_importance = lab_importance + vital_importance
        all_importance.sort(reverse = True)
#        self.__feature_importance_graph(all_importance[0:20], self.__param_code() + "_all.png")

        ## classification with combination of multiple metrics
        lab_class_result = []
        lab_priority = [item[1] for item in lab_importance]
        for n_items in range(1, self.n_lab+1):
            pri_lab = lab_data[:, lab_priority[0:n_items]]
            result = alg_classification.cross_validate(pri_lab, flags, self.n_cv_folds, self.class_alg)
            lab_class_result.append(result)

        graph_data = numpy.transpose(numpy.array([ [item.rec, item.prec, item.f] for item in lab_class_result ]))
        graphs.line_series(graph_data, range(1, self.n_lab+1) ,
                           ['recall', 'precision', 'f_measure'],
                           x_label = "Number of Metrics Used", y_label = "Recall/ Precision/ F_measure",
                           title = "%s Lab Tests"%self.class_alg, ylim = [0,1],
                           filename = self.__param_code() + '_n_metric_lab.png' )
        ret_val['lab_class'] = lab_class_result

        vital_class_result = []
        vital_priority = [item[1] for item in vital_importance]
        print vital_priority
            
        for n_items in range(1, 5):
            pri_vital = vital_data[:, vital_priority[0: n_items]]
            vital_class_result.append(alg_classification.cross_validate(pri_vital, flags, self.n_cv_folds, self.class_alg))
        
        graph_data = numpy.transpose(numpy.array([ [item.rec, item.prec, item.f] for item in vital_class_result ]))
        graphs.line_series(graph_data, range(1, 5) ,
                           ['recall', 'precision', 'f_measure'],
                           x_label = "Number of Metrics Used", y_label = "Recall/ Precision/ F_measure",
                           title = "%s Vital"%self.class_alg, ylim = [0,1],
                           filename = self.__param_code() + '_n_metric_vital.png' )
        ret_val['vital_class'] = vital_class_result

        ## representation learning
        if self.rp_learn_flag:
            
            lab_recall = []
            precisions = []
            f_measures = []

            sel_recalls = []
            sel_precisions = []
            sel_f_measures = []

            selected_feature_importance = []
            top_lab_importance = []

            eval_n_input = range(1,self.n_lab + 1)
            
            for n_input in eval_n_input:

                print "---------stage%d----------"%n_input
                self.dae_hidden = 2 * n_input
                pri_lab = lab_data[:, lab_priority[0:n_input]]

                # cross validation
                kf = cross_validation.KFold(lab_data.shape[0], n_folds = self.n_cv_folds, shuffle = True, random_state = 0)
                result_list = []
                sel_result_list = []
                importance_list = []
                lab_importance_list = []
                for train, test in kf:
                    # datasets
                    train_lab = pri_lab[train, :]
                    train_y = flags[train]
                    test_lab = pri_lab[test, :]
                    test_y = flags[test]

                    orig_train_lab = lab_data[train, :]
                    orig_test_lab = lab_data[test,:]

                    # encoding
                    enc_test_x, enc_train_x = alg_auto_encoder.dae(
                        train_lab, test_lab,
                        n_hidden = self.dae_hidden,
                        corruption_level = self.dae_corruption,
                        return_train = True,
                        n_epochs = self.dae_n_epoch
                        )

                    # classification
                    result_list.append(alg_classification.fit_and_test(enc_train_x, train_y, enc_test_x, test_y, self.class_alg))

                    # classification_after_selection
                    ratio = 0.3
                    n_sel = int(ratio * n_input + 1)
                    sel_index =  alg_feature_selection.select_feature_index(enc_train_x, train_y, n_sel)
                    sel_train_x = enc_train_x[:, sel_index]
                    sel_test_x = enc_test_x[:, sel_index]
                    sel_result_list.append(alg_classification.fit_and_test(sel_train_x, train_y, sel_test_x, test_y, self.class_alg))

                    # feature_importance
                    i_index = alg_feature_selection.select_feature_index(enc_train_x, train_y, 1)
                    feature_data = enc_test_x[:, i_index]
                    feature_importance = alg_feature_selection.calc_entropy_reduction(feature_data, test_y)                    
                    importance_list.append(feature_importance[0][0])

                    # importance of each lab_test for comparison
                    test_all_lab = lab_data[test, :]
                    lab_importance = alg_feature_selection.calc_entropy_reduction(test_all_lab, test_y)
                    sorted(lab_importance)
                    lab_importance_list.append(lab_importance[0][0])

                selected_feature_importance.append(numpy.mean(importance_list))
                top_lab_importance.append(numpy.mean(lab_importance_list))

                all_result = alg_classification.sumup_classification_result(result_list)
                precisions.append(all_result.prec)
                lab_recall.append(all_result.rec)
                f_measures.append(all_result.f)

                all_sel_result = alg_classification.sumup_classification_result(sel_result_list)
                sel_precisions.append(all_sel_result.prec)
                sel_recalls.append(all_sel_result.rec)
                sel_f_measures.append(all_sel_result.f)

            graphs.line_series(numpy.array([lab_recall, precisions, f_measures]), eval_n_input ,
                           ['recall', 'precision', 'f_measure'],
                           x_label = "Number of Metrics Used", y_label = "Recall/ Precision/ F_measure",
                           filename = self.__param_code() + '_n_metric_dae.png' )

            graphs.line_series(numpy.array([sel_recalls, sel_precisions, sel_f_measures]), eval_n_input,
                           ['recall', 'precision', 'f_measure'],
                           x_label = "Number of Metrics Used", y_label = "Recall/ Precision/ F_measure",
                           filename = self.__param_code() + '_n_metric_dae_sel0.3.png' )

            graphs.line_series(numpy.array([selected_feature_importance, top_lab_importance]), eval_n_input ,
                           ['selected', 'lab_test'],
                           x_label = "Number of Metrics Used", y_label = "Entropy Reduction",
                           filename = self.__param_code() + '_top_dae_importance.png' )

        return ret_val

            
    def point_eval_orig(self):
        [most_common_tests, lab_data, lab_descs, lab_units, vital_data, flags] = self.__point_data_preperation()
        if self.classification:
            if self.n_cv_folds > 1:
                print "[INFO] eval cross validation"
                self.__eval_with_classification(
                    lab_data, vital_data, flags,
                    most_common_tests, lab_descs, lab_units)
            else:
                raise ValueError("n_cv_fold should be 2 or larger")
        else:
            if self.n_cv_folds == 1:
                print "[INFO] eval_as_single_set"
                return self.__eval_as_single_set(
                    lab_data, vital_data, flags,
                    most_common_tests, lab_descs, lab_units,
                    )
            elif self.n_cv_folds > 1:
                print "[INFO] eval cross validation"
                return  self.__eval_cross_validation(
                    lab_data, vital_data, flags, 
                    most_common_tests, lab_descs, lab_units)
            else:
                raise ValueError("n_cv_fold should be 1 or larger")

    def __eval_with_classification(self, lab_data, vital_data, flags, most_common_tests, lab_descs, lab_units):
        # lab test only
        alg_classification.cross_validate(lab_data, flags, self.n_cv_folds, self.class_alg)
        # vital only
        alg_classification.cross_validate(vital_data, flags, self.n_cv_folds, self.class_alg)
        # Using both
        all_data= numpy.hstack([lab_data, vital_data])
        alg_classification.cross_validate(all_data, flags, self.n_cv_folds, self.class_alg)

        if self.rp_learn_flag:

            # cross validation
            kf = cross_validation.KFold(lab_data.shape[0], n_folds = self.n_cv_folds, shuffle = True, random_state = 0)
            result_list = []
            for train, test in kf:
                # datasets
                set_train_lab = lab_data[train, :]
                set_train_vital = vital_data[test, :]
                flags_train = flags[train]

                set_test_lab = lab_data[test, :]
                set_test_vital = vital_data[test, :]
                flags_test = flags[test]

                encoded_values = alg_auto_encoder.get_encoded_values(
                    set_train_lab, flags_train, set_test_lab,
                    self.pca_components, 2,
                    self.ica_components, 2,
                    self.dae_hidden, 2,  self.dae_corruption )

                result_list.append(alg_classification.fit_and_test(set_train_lab, flags_train, set_test_lab, flags_test, self.class_alg))

            all_result = alg_classification.sumup_classification_result(result_list)
            print all_result
                
    def __eval_as_single_set(self, lab_data, vital_data, flags,
                           most_common_tests, lab_descs, lab_units):

        self.__classify_important_feature(lab_importance, lab_data, flags,
                                          filename = self.__param_code() + "_lab_imp.png")
        self.__classify_important_feature(vital_importance, vital_data, flags,
                                          filename = self.__param_code() + "_vital_imp.png")

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

            self.__feature_importance_graph(lab_feature_importance[0:20], self.__param_code() + "_lab_feature.png")
            self.__classify_important_feature(feature_importance, feature_data, flags, filename = self.__param_code() + "_feature_imp.png")

        return all_importance

    def __eval_cross_validation(self, lab_data, vital_data, flags, most_common_tests, lab_descs, lab_units):

        # cross validation
        kf = cross_validation.KFold(lab_data.shape[0], n_folds = self.n_cv_folds, shuffle = True, random_state = 0)

        results = []
        for train, test in kf:

            # datasets
            train_lab = lab_data[train, :]
            set_train_vital = vital_data[test, :]
            flag_train = flags[train]

            set_test_lab = lab_data[test, :]
            set_test_vital = vital_data[test, :]
            flags_test = flags[test]

            encoded_values = alg_auto_encoder.get_encoded_values(
                train_lab, flag_train, set_test_lab,
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
        self.__feature_importance_graph(mean_reduction[0:20],  self.__param_code() + "_cv_lab_feature.png")
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
            estimated_disch_time = final_adm.get_estimated_disch_time()
            
            time_of_interest = estimated_disch_time - datetime.timedelta(self.days_before_discharge)
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

        alg_classification.plot_2d(x, flags, x_label, y_label, filename = filename)

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

if __name__ == '__main__':

    result = []
    for alg in alg_classification.algorithm_list[4:5]:
        ef = evaluate_fetaure(max_id = 200000, days_before_discharge =0, n_lab = 20, rp_learn_flag = False, class_alg = alg)
        result.append(ef.point_eval())
    


    ## recall_20 =  [item['recall'][19] for item in result]
    ## alg =  [item['param']['class_alg'] for item in result]
    ## graphs.bar_comparison(recall_20, alg, title = 'recall 20', filename = 'recall20.png')
    
    ## print "n_lab == 10"
    ## ef = evaluate_fetaure(max_id = 200000, days_before_discharge =0, n_lab = 10, dae_hidden = 20, dae_n_epoch =  20000, rp_learn_flag = True)
    ## ef.point_eval()
    
    ## print "n_lab == 20"
    ## ef = evaluate_fetaure(max_id = 200000, days_before_discharge =0, n_lab = 20, dae_hidden = 20, dae_n_epoch = 20000, rp_learn_flag = True)
    ## ef.point_eval()
    plt.waitforbuttonpress()
