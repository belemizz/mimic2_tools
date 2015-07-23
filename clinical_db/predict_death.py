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
                 l_poi=[0., 0.25, 0.5, 1., 2., 3.],
                 tseries_step=0,
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
        results = Bunch(point=l_presult, tseries=l_tresult)
        return results

    def __eval_point(self, data):
        lab_set = [data[0], data[2]]
        vit_set = [data[1], data[2]]

        result = Bunch(
            lab=alg.classification.cv(lab_set, self.n_cv_fold, self.class_param),
            vit=alg.classification.cv(vit_set, self.n_cv_fold, self.class_param))

        return result

    def __eval_tseries(self, tseries):
        lab_set = [tseries[0][0], tseries[0][1], tseries[2]]
        vit_set = [tseries[1][0], tseries[1][1], tseries[2]]

        result = Bunch(
            lab=alg.timeseries.cv(lab_set, self.n_cv_fold, self.tseries_param),
            vit=alg.timeseries.cv(vit_set, self.n_cv_fold, self.tseries_param))

        return result

    def __visualization(self, result):
        p_info("Point")
#        print result.point
        self.__draw_graph_point(result.point)

        p_info("Series")
        print result.tseries
        self.__draw_graph_tseries(result.tseries)

    def __draw_graph_point(self, result):
        l_lab = [item.lab for item in result]
        l_vit = [item.vit for item in result]
        comparison_graph(l_lab, self.l_poi, 'lab')
        comparison_graph(l_vit, self.l_poi, 'vital')
        import matplotlib.pyplot as plt
        plt.waitforbuttonpress()


def comparison_graph(l_classification_result, label, title):
    l_rec = [item.rec for item in l_classification_result]
    l_prec = [item.prec for item in l_classification_result]
    l_f = [item.f for item in l_classification_result]
    graph.line_series([l_rec, l_prec, l_f], label, ['recall', 'precision', 'f_measure'])

if __name__ == '__main__':
    pd = PredictDeath()
    pd.n_day_prediction()
