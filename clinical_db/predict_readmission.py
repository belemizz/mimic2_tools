"""Predict readmission of the patients."""

from mutil import p_info
from get_sample import Mimic2, PatientData
from patient_classification import ControlExperiment

import alg.classification

mimic2 = Mimic2()


class PredictReadmission(ControlExperiment):
    ''''Prediction of readmission prediction.'''

    def __init__(self, max_id, target_codes, matched_only,
                 n_lab, disch_origin, l_poi,
                 class_param, n_cv_fold):
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

        # params for algorithm
        self.class_param = class_param
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
                    l_pdata.append(patients.get_lab_chart_point_final_adm(l_lab, mimic2.vital_charts,
                                                                        poi, self.disch_origin))
                    import ipdb
                    ipdb.set_trace()
            else:
                l_pdata.append(patients.get_lab_chart_point_all_adm(l_lab, mimic2.vital_charts,
                                                                    self.l_poi, self.disch_origin))
                
        return l_pdata

    def __eval_data(self, l_data):
        p_info("Data evaluation")
        l_presult = []
        for data in l_data:
            l_presult.append(self.__eval_point(data))
        return l_presult

    def __eval_point(self, data):
        print data

    def __visualize(self, result):
        p_info("Visualization")
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    class_param = alg.classification.Default_param

    pr = PredictReadmission(max_id=2000,
                            target_codes='chf',
                            matched_only='false',
                            n_lab=10,
                            disch_origin=True,
                            l_poi=[0.],
                            class_param=class_param,
                            n_cv_fold=4)
    pr.prediction()
