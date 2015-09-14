"""Predict readmission of the patients."""
from bunch import Bunch
import numpy as np

from get_sample import Mimic2, PatientData
from patient_classification import (ControlExperiment, Default_db_param,
                                    Default_data_param, Default_alg_param)

import alg.classification
import alg.timeseries
from get_sample.timeseries import SeriesData

from mutil import Graph

mimic2 = Mimic2()
graph = Graph()


class PredictReadmission(ControlExperiment):
    ''''Prediction of readmission prediction.'''

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
    pr = PredictReadmission()
    result = pr.execution()
    print (result.lab.f, result.vit.f)
