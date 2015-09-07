"""Predict death in next n days."""

from get_sample import Mimic2, PatientData

import alg.timeseries
import alg.classification
from get_sample.timeseries import SeriesData

from bunch import Bunch
from mutil import Graph

from patient_classification import ControlExperiment

mimic2 = Mimic2()
graph = Graph()


class PredictDeath(ControlExperiment):
    """Evaluate metrics for predicting death."""

    def __init__(self,
                 max_id=0,
                 target_codes='chf',
                 matched_only=False,
                 n_lab=20,
                 disch_origin=True,
                 l_poi=0.,
                 tseries_flag=True,
                 tseries_duration=1.,
                 tseries_cycle=0.25,
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
        :param tseries_flag: True to evaluate timeseries
        :param tseries_duration: Duration in days of timeseries
        :param tseries_cycle: Cycle of the points of timeseries
        :param class_param: param for classification algorithm
        :param tseries_param: param for timeseries classification algorithm
        :param n_cv_fold: number of folds in cross validation
        '''
        # params for data retrieval
        ControlExperiment.__init__(self, max_id, target_codes, matched_only)
        self.patients = PatientData(self.id_list)

        # params for data
        self.original_data_params = (n_lab, disch_origin, l_poi,
                                     tseries_flag, tseries_duration, tseries_cycle)
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

    def reset_algo_params(self):
        self.class_param = self.original_algo_params[0]
        self.tseries_param = self.original_algo_params[1]
        self.n_cv_fold = self.original_algo_params[2]

    def execution(self):
        """Prediction in a single condition"""
        l_lab, l_descs, l_units = self.patients.common_lab(self.n_lab)

        if self.tseries_flag:
            data = self.patients.tseries_from_adm(l_lab, mimic2.vital_charts,
                                                      self.tseries_cycle,
                                                      self.tseries_duration,
                                                      self.disch_origin)
            result = self.__eval_tseries(data)

        else:
            data = self.patients.point_from_adm(l_lab, mimic2.vital_charts,
                                                    self.l_poi, self.disch_origin)
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
        lab_data = data[0]
        vit_data = data[1]
        label = data[2]
        lab_res = alg.classification.cv(lab_data, label, self.n_cv_fold, self.class_param)
        vit_res = alg.classification.cv(vit_data, label, self.n_cv_fold, self.class_param)
        return Bunch(lab=lab_res, vit=vit_res)

    def __eval_tseries(self, tseries):
        lab_series = SeriesData(tseries[0][0], tseries[0][1], tseries[2])
        vit_series = SeriesData(tseries[1][0], tseries[1][1], tseries[2])
        return Bunch(
            lab=alg.timeseries.cv(lab_series, self.n_cv_fold, self.tseries_param),
            vit=alg.timeseries.cv(vit_series, self.n_cv_fold, self.tseries_param))

if __name__ == '__main__':
    pd = PredictDeath()
    result = pd.execution()
