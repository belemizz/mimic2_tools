"""Predict death in next n days."""
from bunch import Bunch

from get_sample import Mimic2, PatientData
from get_sample.timeseries import SeriesData

import alg.timeseries
import alg.classification

from mutil import Graph

from patient_classification import (ControlExperiment, Default_db_param,
                                    Default_data_param, Default_alg_param)

mimic2 = Mimic2()
graph = Graph()


class PredictDeath(ControlExperiment):
    """Evaluate metrics for predicting death."""

    def __init__(self,
                 db_param=Default_db_param,
                 data_param=Default_data_param,
                 alg_param=Default_alg_param
                 ):
        ControlExperiment.set_db_param(self, db_param)
        ControlExperiment.set_data_param(self, data_param)
        ControlExperiment.set_alg_param(self, alg_param)
        self.patients = PatientData(self.id_list)

    def execution(self):
        """Prediction in a single condition"""
        l_lab, l_descs, l_units = self.patients.common_lab(self.n_lab)

        if self.tseries_flag and self.coef_flag:
            raise NotImplementedError

        elif self.tseries_flag and not self.coef_flag:
            data = self.patients.tseries_from_adm(l_lab, mimic2.vital_charts,
                                                  self.tseries_cycle, self.tseries_duration,
                                                  self.disch_origin)
            result = self.__eval_tseries(data)

        elif not self.tseries_flag and self.coef_flag:
            data = self.patients.trend_from_adm(l_lab, mimic2.vital_charts,
                                                poi=self.l_poi, span=self.coef_span,
                                                from_discharge=self.disch_origin)
            result = self.__eval_point(data)

        else:
            data = self.patients.point_from_adm(l_lab, mimic2.vital_charts,
                                                self.l_poi, self.disch_origin)
            result = self.__eval_point(data)

        return result

    def compare_class_alg(self, l_param):
        result = []
        for param in l_param:
            self.class_param = param
            result.append(self.execution())
        ControlExperiment.set_alg_param(self)
        return result

    def compare_coef(self, l_span, include_raw_data):
        result = []
        if include_raw_data:
            self.coef_flag = False
            result.append(self.execution())

        self.coef_flag = True
        for span in l_span:
            self.coef_span = span
            result.append(self.execution())

        ControlExperiment.set_data_param(self)
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

        ControlExperiment.set_data_param(self)
        return result

    def compare_cycle(self, l_cycle):
        result = []
        self.tseries_flag = True
        for cycle in l_cycle:
            self.tseries_cycle = cycle
            result.append(self.execution())

        ControlExperiment.set_data_param(self)
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
    print result.lab.get_dict()
