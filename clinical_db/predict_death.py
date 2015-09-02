"""Predict death in next n days."""

from mutil import p_info
from get_sample import Mimic2, PatientData

import alg.timeseries
import alg.classification
from alg.timeseries import SeriesData

from bunch import Bunch
from mutil import Graph

from patient_classification import ControlExperiment

mimic2 = Mimic2()
graph = Graph()


class PredictDeath(ControlExperiment):
    """Evaluate metrics for predicting death."""

    def __init__(self,
                 max_id=200000,
                 target_codes=['428.0'],
                 matched_only=False,
                 n_lab=20,
                 disch_origin=True,
                 l_poi=0.,  # None to disable point eval
                 tseries_duration=1.,  # None to disable tseries eval
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
        :param tseries_duration: Duration in days of timeseries (None to disable tseries eval)
        :param tseries_cycle: Cycle of the points of timeseries
        :param class_param: param for classification algorithm
        :param tseries_param: param for timeseries classification algorithm
        :param n_cv_fold: number of folds in cross validation
        '''
        # params for data retrieval
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

        # Comparison setting
        self.__point_comparison_info()
        self.__tseries_comparison_info()

    def n_day_prediction(self):
        data = self.__prepare_data()
        result = self.__eval_data(data)
        self.__visualize(result)

    def __point_comparison_info(self):
        candidate = ([self.l_poi, 'point of interest', 'line'],
                     [self.class_param.name, 'algorithm', 'bar'])
        self.point_comp_info = self.__check_and_get_info(candidate)

    def __tseries_comparison_info(self):
        candidate = ([self.tseries_duration, 'Duration', 'line'],
                     [self.tseries_cycle, 'Freqency', 'line'])
        self.tseries_comp_info = self.__check_and_get_info(candidate)

    def __check_and_get_info(self, candidate):
        counter = 0
        info = None
        for item in candidate:
            if isinstance(item[0], list):
                info = item
                counter += 1
            if counter > 1:
                raise TypeError("Only one of the folloing can be a list %s"
                                % [item[1] for item in candidate])
        return info

    def __prepare_data(self):
        p_info("Data preparation")
        patients = PatientData(self.id_list)
        l_lab, l_descs, l_units = patients.get_common_labs(self.n_lab)

        l_pdata = []
        if self.l_poi is not None:
            if isinstance(self.l_poi, list):
                for poi in self.l_poi:
                    l_pdata.append(patients.get_lab_chart_point_final_adm(l_lab,
                                                                          mimic2.vital_charts,
                                                                          poi, self.disch_origin))
            else:
                l_pdata.append(patients.get_lab_chart_point_final_adm(l_lab, mimic2.vital_charts,
                                                                      self.l_poi,
                                                                      self.disch_origin))
        l_tseries = []
        if self.tseries_duration is not None:
            if isinstance(self.tseries_duration, list):
                for duration in self.tseries_duration:
                    l_tseries.append(patients.get_lab_chart_tseries_final_adm(
                        l_lab, mimic2.vital_charts,
                        self.tseries_cycle, duration, self.disch_origin))
            elif isinstance(self.tseries_cycle, list):
                for freq in self.tseries_cycle:
                    l_tseries.append(patients.get_lab_chart_tseries_final_adm(
                        l_lab, mimic2.vital_charts, freq,
                        self.tseries_duration, self.disch_origin))
            else:
                l_tseries.append(patients.get_lab_chart_tseries_final_adm(
                    l_lab, mimic2.vital_charts,
                    self.tseries_cycle, self.tseries_duration, self.disch_origin))
        return Bunch(point=l_pdata, tseries=l_tseries)

    def __eval_data(self, l_data):
        l_presult = []
        for data in l_data.point:
            p_info("Point Evaluation")
            l_presult.append(self.__eval_point(data))

        l_tresult = []
        for data in l_data.tseries:
            p_info("Tseries Evaluation")
            l_tresult.append(self.__eval_tseries(data))

        return Bunch(point=l_presult, tseries=l_tresult)

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

    def __visualize(self, result):
        import ipdb
        ipdb.set_trace()
        print ('lab_point', result.point[0].lab.get_dict())
        print ('vit_point', result.point[0].vit.get_dict())
        print ('lab_ts', result.tseries[0].lab.get_dict())
        print ('vit_ts', result.tseries[0].vit.get_dict())
        if self.point_comp_info:
            self.__draw_graph_point(result.point)
        if self.tseries_comp_info:
            self.__draw_graph_tseries(result.tseries)

    def __draw_graph_point(self, result):
        l_lab = [item.lab for item in result]
        l_vit = [item.vit for item in result]
        l_lab_auc = [item.lab_auc for item in result]
        print l_lab_auc

        l_lab = self.__remove_list_duplication(l_lab)
        l_vit = self.__remove_list_duplication(l_vit)
        if self.point_comp_info[2] == 'bar':
            graph.bar_classification(l_lab_auc, self.point_comp_info[0], 'lab_auc')
            graph.bar_classification(l_lab, self.point_comp_info[0], 'lab')
            graph.bar_classification(l_vit, self.point_comp_info[0], 'vit')
        else:
            x_label = self.point_comp_info[1]
            graph.series_classification(l_lab, self.point_comp_info[0], x_label, 'lab')
            graph.series_classification(l_vit, self.point_comp_info[0], x_label, 'vital')

        graph.waitforbuttunpress()

    def __remove_list_duplication(self, l_in):
        if len(l_in) == 1 and isinstance(l_in[0], list):
            l_out = l_in[0]
        else:
            l_out = l_in
        return l_out

    def __draw_graph_tseries(self, result):
        l_lab = [item.lab for item in result]
        l_vit = [item.vit for item in result]
        x_label = self.tseries_comp_info[1]
        graph.series_classification(l_lab, self.tseries_comp_info[0], x_label, 'lab')
        graph.series_classification(l_vit, self.tseries_comp_info[0], x_label, 'vit')
        graph.waitforbuttunpress()

if __name__ == '__main__':
    class_param = alg.classification.Default_param

    pd = PredictDeath(max_id=0,
                      target_codes='chf',
                      n_lab=20,
                      disch_origin=False,
                      l_poi=0.,
                      tseries_duration=1.,
                      tseries_cycle=0.1,
                      class_param=class_param,
                      tseries_param=alg.timeseries.Default_param,
                      n_cv_fold=10)

    pd.n_day_prediction()
