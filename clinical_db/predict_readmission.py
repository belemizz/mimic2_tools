"""Predict readmission of the patients."""

from bunch import Bunch
from mutil import p_info
from get_sample import Mimic2, PatientData
from patient_classification import ControlExperiment

import alg.classification
import alg.timeseries
from alg.timeseries import SeriesData

import numpy as np
from mutil import Graph

mimic2 = Mimic2()
graph = Graph()


class PredictReadmission(ControlExperiment):
    ''''Prediction of readmission prediction.'''

    def __init__(self,
                 max_id=0,
                 target_codes='chf',
                 matched_only=False,
                 n_lab=20,
                 disch_origin=True,
                 l_poi=0.,
                 tseries_flag=True,
                 tseries_duration=1.,
                 tseries_cycle=0.1,
                 visualize_data=True,
                 class_param=alg.classification.Default_param,
                 tseries_param=alg.timeseries.Default_param,
                 n_cv_fold=10):
        '''Initializer.

        :param max_id: maximum of subject id (0 for using all ids)
        :param target_codes: keyword of a list of icd9 codes to select subjects
        :param matched_only: select only subjects with continuous record
        :param n_lab: number of lab tests to be used
        :param disch_origin: count duration from discharge point
        :param l_poi: list of point of interest
        :param tseries_flag: True for use timeseries
        :param tseries_duraction: Duration of the timeseries
        :param tseres_cycle: Cycle of the timeseries
        :param class_param: param for classification algorithm
        :param tsereis_param: param for timeseries classification algorithm
        :param n_cv_fold: number of folds in cross validation
        '''
        p_info("Initialization")
        ControlExperiment.__init__(self, max_id, target_codes, matched_only)
        self.patients = PatientData(self.id_list)

        # params for data
        self.original_data_params = (n_lab, disch_origin, l_poi,
                                     tseries_flag, tseries_duration, tseries_cycle, visualize_data)
        self.reset_data_params()

        # params for algorithm
        self.original_algo_params = (class_param, tseries_param, n_cv_fold)
        self.reset_algo_params()

    def reset_data_params(self):
        self.n_lab = self.original_data_params[0]
        self.disch_origin = self.original_data_params[1]
        self.l_poi = self.original_data_params[2]
        self.tseries_flag = self.original_data_params[3]
        self.tseries_duration = self.original_data_params[4]
        self.tseries_cycle = self.original_data_params[5]
        self.visualize_data = self.original_data_params[6]

    def reset_algo_params(self):
        self.class_param = self.original_algo_params[0]
        self.tseries_param = self.original_algo_params[1]
        self.n_cv_fold = self.original_algo_params[2]

    def execution(self):
        l_lab, l_descs, l_units = self.patients.get_common_labs(self.n_lab)

        if self.tseries_flag:
            data = self.patients.get_lab_chart_tseries_all_adm(l_lab, mimic2.vital_charts,
                                                               self.tseries_cycle,
                                                               self.tseries_duration,
                                                               self.disch_origin)

            result = self.__eval_tseries(data)
        else:
            data = self.patients.get_lab_chart_point_all_adm(l_lab, mimic2.vital_charts,
                                                             self.l_poi,
                                                             self.disch_origin)
            result = self.__eval_point(data)
        return result

    def compare_duration(self, l_duration, include_point_data=False):
        result = []
        if include_point_data:
            self.tseries_flag = False
            result.append(self.execution())

        self.tseries_flag = True
        for duration in l_duration:
            self.tseries_duration = duration
            result.append(self.execution())

        self.reset_data_params()
        return result

    def compare_cycle(self, l_cycle):
        result = []
        self.tseries_flag = True
        for cycle in l_cycle:
            self.tseries_cycle = cycle
            result.append(self.execution())

        self.reset_data_params()
        return result

    def __eval_point(self, data):
        lab_all = data[0]
        vit_all = data[1]
        readm_duration = data[4]
        death_duration = data[5]
        alive_on_disch = death_duration >= 1

        lab_data = lab_all[alive_on_disch]
        vit_data = vit_all[alive_on_disch]

        death_flag = (death_duration < 31)[alive_on_disch]
        readm_flag = (readm_duration < 31)[alive_on_disch]
        r_or_d_flag = np.logical_or(readm_flag, death_flag)

        result_lab = alg.classification.cv(lab_data, r_or_d_flag, self.n_cv_fold, self.class_param)
        result_vit = alg.classification.cv(vit_data, r_or_d_flag, self.n_cv_fold, self.class_param)

        return Bunch(lab=result_lab, vit=result_vit)

    def __eval_tseries(self, data):
        readm_duration = data[4]
        death_duration = data[5]
        alive_on_disch = death_duration >= 1

        death_flag = (death_duration < 31)
        readm_flag = (readm_duration < 31)
        r_or_d_flag = np.logical_or(readm_flag, death_flag)

        lab_data = SeriesData(data[0][0], data[0][1], r_or_d_flag)
        vit_data = SeriesData(data[1][0], data[1][1], r_or_d_flag)

        lab_select = lab_data.slice_by_sample(alive_on_disch)
        vit_select = vit_data.slice_by_sample(alive_on_disch)

        if self.visualize_data:
            graph.draw_series_data_class(vit_select, 100)
            graph.draw_series_data_class(lab_select, 100)

        result_lab = alg.timeseries.cv(lab_select, self.n_cv_fold, self.tseries_param)
        result_vit = alg.timeseries.cv(vit_select, self.n_cv_fold, self.tseries_param)

        return Bunch(lab=result_lab, vit=result_vit)

if __name__ == '__main__':
    class_param = alg.classification.Default_param
    tseries_param = alg.timeseries.Default_param

    pr = PredictReadmission(max_id=0,
                            target_codes='chf',
                            matched_only=False,
                            n_lab=20,
                            disch_origin=True,
                            l_poi=0.,
                            tseries_flag=True,
                            tseries_duration=2.,
                            tseries_cycle=0.25,
                            visualize_data=True,
                            class_param=class_param,
                            tseries_param=tseries_param,
                            n_cv_fold=10)
    pr.execution()
