"""Predict readmission of the patients."""
from bunch import Bunch
import numpy as np

from get_sample import Mimic2, PatientData
from patient_classification import (ControlExperiment, Default_db_param,
                                    Default_data_param, Default_alg_param)

import alg.classification
import alg.timeseries
import alg.feature_selection
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
            data = self.patients.tseries_from_adm(l_lab, mimic2.vital_charts, self.span,
                                                  self.tseries_cycle, self.disch_origin)
            result = self.__eval_tseries(data)

        elif not self.tseries_flag and self.coef_flag:
            data = self.patients.trend_from_adm(l_lab, mimic2.vital_charts, self.span,
                                                from_discharge=self.disch_origin,
                                                accept_none=True)
            result = self.__eval_point(data)

        else:
            data = self.patients.point_from_adm(l_lab, mimic2.vital_charts,
                                                self.span[1], self.disch_origin,
                                                accept_none=True)
            result = self.__eval_point(data)

        return result

    def ent_reduction_point(self):
        l_lab, l_descs, l_units = self.patients.common_lab(self.n_lab)
        data = self.patients.point_from_adm(l_lab, mimic2.vital_charts,
                                            self.span[1], self.disch_origin,
                                            accept_none=False)
        lab_data, vit_data, r_or_d_flag = self.__point_data_flag(data)

        x = np.hstack((lab_data, vit_data))
        y = r_or_d_flag
        l_id = l_lab + mimic2.vital_charts
        l_descs = l_descs + mimic2.vital_descs
        l_units = l_units + mimic2.vital_units
        result = alg.feature_selection.calc_entropy_reduction(x, y, l_id, l_descs, l_units)
        return result

    def __eval_point(self, data):
        lab_data, vit_data, r_or_d_flag = self.__point_data_flag(data)
        result_lab = alg.classification.cv(lab_data, r_or_d_flag, self.n_cv_fold, self.class_param)
        result_vit = alg.classification.cv(vit_data, r_or_d_flag, self.n_cv_fold, self.class_param)
        return Bunch(lab=result_lab, vit=result_vit)

    def __point_data_flag(self, data):
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
        return lab_data, vit_data, r_or_d_flag

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
