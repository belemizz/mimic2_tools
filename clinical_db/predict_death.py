"""Predict death in next n days."""

import numpy as np
from mutil import p_info
from get_sample import Mimic2, PatientData

import alg.timeseries
import alg.classification

from bunch import Bunch
from mutil import Graph

mimic2 = Mimic2()
graph = Graph()


class PredictDeath:
    """Evaluate metrics for predicting death."""

    def __init__(self,
                 max_id=200000,
                 target_codes=['428.0'],
                 n_lab=20,
                 disch_origin=True,
                 l_poi=[],
                 tseries_duration=[1., 2., 3., 4.],
                 tseries_freq=0.5,
                 class_param=alg.classification.Default_param,
                 tseries_param=alg.timeseries.Default_param,
                 n_cv_fold=10):

        # param validation
        if isinstance(tseries_duration, list) and isinstance(tseries_freq, list):
            raise ValueError("Both tseries_duration and tseries_freq can't be lists")
        # params for data retrieval
        self.max_id = max_id
        self.target_codes = target_codes
        self.n_lab = n_lab

        self.disch_origin = disch_origin
        self.l_poi = l_poi
        self.tseries_duration = tseries_duration
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

        if isinstance(self.tseries_duration, list):
            for duration in self.tseries_duration:
                l_data.append(patients.get_lab_chart_tseries(l_lab, mimic2.vital_charts,
                                                             self.tseries_freq, duration,
                                                             self.disch_origin))
        elif isinstance(self.tseries_freq, list):
            for freq in self.tseries_freq:
                l_data.append(patients.get_lab_chart_tseries(l_lab, mimic2.vital_charts,
                                                             freq, self.tseries_duration,
                                                             self.disch_origin))
        else:
            l_data.append(patients.get_lab_chart_tseries(l_lab, mimic2.vital_charts,
                                                         self.tseries_freq, self.tseries_duration,
                                                         self.disch_origin))
        return l_data

    def __eval_data(self, l_data):
        l_tresult = []
        l_presult = []
        for data in l_data:
            if isinstance(data[0], np.ndarray):
                p_info("Point Evaluation")
                l_presult.append(self.__eval_point(data))

            elif isinstance(data[0], list):
                p_info("Tseries Evaluation")
                l_tresult.append(self.__eval_tseries(data))
        return Bunch(point=l_presult, tseries=l_tresult)

    def __eval_point(self, data):
        lab_set = [data[0], data[2]]
        vit_set = [data[1], data[2]]
        return Bunch(
            lab=alg.classification.cv(lab_set, self.n_cv_fold, self.class_param),
            vit=alg.classification.cv(vit_set, self.n_cv_fold, self.class_param))

    def __eval_tseries(self, tseries):
        lab_set = [tseries[0][0], tseries[0][1], tseries[2]]
        vit_set = [tseries[1][0], tseries[1][1], tseries[2]]
        return Bunch(
            lab=alg.timeseries.cv(lab_set, self.n_cv_fold, self.tseries_param),
            vit=alg.timeseries.cv(vit_set, self.n_cv_fold, self.tseries_param))

    def __visualization(self, result):
        if result.point:
            self.__draw_graph_point(result.point)
        self.__draw_graph_tseries(result.tseries)

    def __draw_graph_point(self, result):
        l_lab = [item.lab for item in result]
        l_vit = [item.vit for item in result]
        if self.disch_origin:
            x_label = 'Days from discharge'
        else:
            x_label = 'Days from admission'
        graph.series_classification(l_lab, self.l_poi, x_label, 'lab')
        graph.series_classification(l_vit, self.l_poi, x_label, 'vit')
        graph.waitforbuttunpress()

    def __draw_graph_tseries(self, result):
        l_lab = [item.lab for item in result]
        l_vit = [item.vit for item in result]
        if isinstance(self.tseries_duration, list):
            x_label = 'Duration'
            graph.series_classification(l_lab, self.tseries_duration, x_label, 'lab')
            graph.series_classification(l_vit, self.tseries_duration, x_label, 'vit')
        if isinstance(self.tseries_freq, list):
            x_label = 'Freq'
            graph.series_classification(l_lab, self.tseries_freq, x_label, 'lab')
            graph.series_classification(l_vit, self.tseries_freq, x_label, 'vit')
        graph.waitforbuttunpress()

if __name__ == '__main__':
    pd = PredictDeath()
    pd.n_day_prediction()
