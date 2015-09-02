"""Predict readmission of the patients."""

from mutil import p_info
from get_sample import Mimic2, PatientData
from patient_classification import ControlExperiment

import alg.classification
import alg.timeseries
from alg.timeseries import SeriesData

import numpy as np

mimic2 = Mimic2()


class PredictReadmission(ControlExperiment):
    ''''Prediction of readmission prediction.'''

    def __init__(self, max_id, target_codes, matched_only,
                 n_lab, disch_origin, l_poi,
                 tseries_duration, tseries_cycle,
                 class_param, tseries_param, n_cv_fold):
        '''Initializer.

        :param max_id: maximum of subject id (0 for using all ids)
        :param target_codes: keyword of a list of icd9 codes to select subjects
        :param matched_only: select only subjects with continuous record
        :param n_lab: number of lab tests to be used
        :param disch_origin: count duration from discharge point
        :param l_poi: list of point of interest
        :param class_param: param for classification algorithm
        :param n_cv_fold: number of folds in cross validation
        '''
        p_info("Initialization")
        ControlExperiment.__init__(self, max_id, target_codes, matched_only)

        # params for data
        self.n_lab = n_lab
        self.disch_origin = disch_origin
        self.l_poi = l_poi
        self.tseries_duration = tseries_duration
        self.tseries_cycle = tseries_cycle

        # params for algorithm
        self.class_param = class_param
        self.tseries_param = tseries_param
        self.n_cv_fold = n_cv_fold

    def prediction(self):
        data = self.__prepare_data()
        result = self.__eval_data(data)
        self.__visualize(result)

    def __prepare_data(self):
        p_info("Data preparation")
        patients = PatientData(self.id_list)
        l_lab, l_descs, l_units = patients.get_common_labs(self.n_lab)

        l_pdata = []
        if self.l_poi is not None:
            if isinstance(self.l_poi, list):
                for poi in self.l_poi:
                    l_pdata.append(patients.get_lab_chart_point_all_adm(l_lab, mimic2.vital_charts,
                                                                        poi, self.disch_origin))
            else:
                l_pdata.append(patients.get_lab_chart_point_all_adm(l_lab, mimic2.vital_charts,
                                                                    self.l_poi, self.disch_origin))

        l_tseries = []
        if self.tseries_duration is not None:
            l_tseries.append(patients.get_lab_chart_tseries_all_adm(l_lab, mimic2.vital_charts,
                                                                    self.tseries_cycle,
                                                                    self.tseries_duration,
                                                                    self.disch_origin))
        return l_pdata, l_tseries

    def __eval_data(self, l_data):
        p_info("Data evaluation")
        l_presult = []
        for data in l_data[0]:
            l_presult.append(self.__eval_point(data))

        l_tresult = []
        for data in l_data[1]:
            l_tresult.append(self.__eval_tseries(data))

        return l_presult, l_tresult

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

        return result_lab, result_vit

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

        result_lab = alg.timeseries.cv(lab_select, self.n_cv_fold, self.tseries_param)
        result_vit = alg.timeseries.cv(vit_select, self.n_cv_fold, self.tseries_param)

        return result_lab, result_vit

    def __visualize(self, result):
        p_result = result[0]
        t_result = result[1]

        p_info("Point Result: Lab")
        print p_result[0][0].get_dict()
        p_info("Point Result: Vit")
        print p_result[0][1].get_dict()

        p_info("TS Result: Lab")
        print t_result[0][0].get_dict()
        p_info("TS Result: Vit")
        print t_result[0][1].get_dict()


if __name__ == '__main__':
    class_param = alg.classification.Default_param
    tseries_param = alg.timeseries.Default_param

    pr = PredictReadmission(max_id=0,
                            target_codes='chf',
                            matched_only=False,
                            n_lab=20,
                            disch_origin=True,
                            l_poi=0.,
                            tseries_duration=1.,
                            tseries_cycle=0.1,
                            class_param=class_param,
                            tseries_param=tseries_param,
                            n_cv_fold=10)
    pr.prediction()
