"""Predict death in next n days."""

import numpy as np
from mutil import p_info
from get_sample import Mimic2, PatientData

import alg.timeseries
import alg.classification

mimic2 = Mimic2()


class PredictDeath:
    """Evaluate metrics for predicting death."""

    def __init__(self,
                 max_id=200000,
                 target_codes=['428.0'],
                 n_lab=20,
                 disch_origin=True,
                 l_poi=[0.],
                 tseries_step=10,
                 tseries_freq=0.25,
                 class_param=alg.classification.Default_param,
                 tseries_param=alg.timeseries.Default_param,
                 n_cv_fold=10):

        # params for data retrieval
        self.max_id = max_id
        self.target_codes = target_codes
        self.n_lab = n_lab

        self.disch_origin = disch_origin
        self.l_poi = l_poi
        self.tseries_step = tseries_step
        self.tseries_freq = tseries_freq

        # params for algorithm
        self.class_param = class_param
        self.tseries_param = tseries_param
        self.n_cv_fold = n_cv_fold

    def n_day_prediction(self):
        data = self.__data_preparation()
        result = self.__eval_data(data)
        self.__visualization(result)

    def __data_preparation(self):
        id_list = mimic2.subject_with_icd9_codes(self.target_codes, True, True, self.max_id)
        patients = PatientData(id_list)
        l_lab, l_descs, l_units = patients.get_common_labs(self.n_lab)
        l_data = []
        if len(self.l_poi) > 0:
            for poi in self.l_poi:
                l_data.append(patients.get_lab_chart_point(l_lab, mimic2.vital_charts,
                                                           poi, self.disch_origin))

        if self.tseries_step > 0:
            l_data.append(patients.get_lab_chart_tseries(l_lab, mimic2.vital_charts,
                                                         self.tseries_freq, self.tseries_step,
                                                         True))
        return l_data

    def __eval_data(self, l_data):
        l_result = []
        for data in l_data:
            if isinstance(data[0], np.ndarray):
                p_info("Point Evaluation")
                l_result.append(self.__eval_point(data))

            elif isinstance(data[0], list):
                p_info("Tseries Evaluation")
                l_result.append(self.__eval_tseries(data))
        return l_result

    def __eval_point(self, data):
        lab_set = [data[0], data[2]]
        vit_set = [data[1], data[2]]

        lab_result = alg.classification.cv(lab_set, self.n_cv_fold, self.class_param)
        vit_result = alg.classification.cv(vit_set, self.n_cv_fold, self.class_param)

        return (lab_result, vit_result)

    def __eval_tseries(self, tseries):
        lab_set = [tseries[0][0], tseries[0][1], tseries[2]]
        vit_set = [tseries[1][0], tseries[1][1], tseries[2]]

        lab_result = alg.timeseries.cv(lab_set, self.n_cv_fold, self.tseries_param)
        vit_result = alg.timeseries.cv(vit_set, self.n_cv_fold, self.tseries_param)

        return [lab_result, vit_result]

    def __visualization(self, result):
        print result
        p_info("Visualize")


if __name__ == '__main__':
    pd = PredictDeath()
    pd.n_day_prediction()
