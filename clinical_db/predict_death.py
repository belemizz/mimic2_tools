"""Predict death in next n days."""
from bunch import Bunch
import numpy as np

from get_sample import Mimic2, PatientData
from get_sample.timeseries import SeriesData

import alg.timeseries
import alg.classification
import alg.feature_selection

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
        x = np.hstack((data[0], data[1]))
        y = data[2]
        l_id = l_lab + mimic2.vital_charts
        l_descs = l_descs + mimic2.vital_descs
        l_units = l_units + mimic2.vital_units
        result = alg.feature_selection.calc_entropy_reduction(x, y, l_id, l_descs, l_units)
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
    db_param = Default_db_param
    data_param = Default_data_param
    data_param.tseries_flag = False
    pd = PredictDeath(db_param, data_param)
    result = pd.execution()
    print result.lab.get_dict()
