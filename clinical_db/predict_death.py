"""Predict death in next n days."""

import numpy as np
from mutil import p_info, is_number
from get_sample import Mimic2
from collections import Counter
from datetime import timedelta

import alg.timeseries

mimic2 = Mimic2()


class PredictDeath:
    """Evaluate metrics for predicting death."""

    def __init__(self,
                 max_id=200000,
                 target_codes=['428.0'],
                 n_lab=20,
                 tseries_freq=0.25,
                 tseries_step=10,
                 rp_learn_flag=True,
                 algo_name='lr',
                 algo_params={},
                 n_cv_fold=30):

        # params for data retrieval
        self.max_id = max_id
        self.target_codes = target_codes
        self.n_lab = n_lab

        # params for timeseries
        self.tseries_freq = tseries_freq
        self.tseries_step = tseries_step

        # params for algorithm
        self.algo_name = algo_name
        self.algo_params = algo_params
        self.n_cv_fold = n_cv_fold
        self.predict_days = [1, 2]

    def n_day_prediction(self):
        tseries = self.__data_preparation()
        result = self.__evaluation(tseries)
        self.__visualization(result)

    def __data_preparation(self):
        id_list = mimic2.subject_with_icd9_codes(self.target_codes, True, True, self.max_id)
        patients = PatientData(id_list)
        l_common_lab, l_descs, l_units = patients.get_common_labs(self.n_lab)
        tseries = patients.get_lab_chart_tseries(l_common_lab, mimic2.vital_charts,
                                                 self.tseries_freq, self.tseries_step, True)
        return tseries

    def __evaluation(self, tseries):
        lab_set = [tseries[0][0], tseries[0][1], tseries[2]]
        vit_set = [tseries[1][0], tseries[1][1], tseries[2]]

        lab_result = alg.timeseries.cv(lab_set, self.n_cv_fold, self.algo_name)
        vit_result = alg.timeseries.cv(vit_set, self.n_cv_fold, self.algo_name)

        p_info("Lab result")
        print lab_result
        p_info("Vital result")
        print vit_result
        return [lab_result, vit_result]

    def __visualization(self, result):
        p_info("Visualize")


class PatientData:
    def __init__(self, id_list):
        self.id_list = id_list
        self.__get_patient_data()

    def __get_patient_data(self):
        self.l_patient = []
        for id in self.id_list:
            patient = mimic2.get_subject(id)
            if patient:
                self.l_patient.append(patient)
            else:
                p_info("ID %d is not available" % id)

    def get_common_labs(self, n_select):
        lab_ids_dict = {}
        units = {}
        descs = {}
        for patient in self.l_patient:
            final_adm = patient.get_final_admission()
            for item in final_adm.labs:
                if item.itemid in lab_ids_dict:
                    lab_ids_dict[item.itemid] = lab_ids_dict[item.itemid] + 1
                else:
                    lab_ids_dict[item.itemid] = 1
                    units[item.itemid] = item.unit
                    descs[item.itemid] = item.description

        counter = Counter(lab_ids_dict)
        most_common_tests = [item[0] for item in counter.most_common(n_select)]
        lab_descs = []
        lab_units = []
        for item_id in most_common_tests:
            lab_descs.append(descs[item_id])
            lab_units.append(units[item_id])
        return most_common_tests, lab_descs, lab_units

    def get_lab_chart_tseries(self, l_lab_id, l_chart_id, freq, n_steps, from_discharge=True):
        l_results = []
        for idx in range(n_steps):
            days = idx * freq
            l_results.append(self.get_lab_chart_point(l_lab_id, l_chart_id, days, from_discharge))

        ids = l_results[0][3]
        flags = l_results[0][2]

        lab_x = np.zeros([n_steps, len(ids), len(l_lab_id)])
        lab_m = np.zeros([n_steps, len(ids)])
        vit_x = np.zeros([n_steps, len(ids), len(l_chart_id)])
        vit_m = np.zeros([n_steps, len(ids)])

        for i_steps in range(n_steps):
            s_lab = l_results[i_steps][0]
            s_vit = l_results[i_steps][1]
            s_id = l_results[i_steps][3]

            for i_id, id in enumerate(ids):
                try:
                    lab_x[i_steps][i_id] = s_lab[s_id.index(id)]
                    lab_m[i_steps][i_id] = 1.
                except ValueError:
                    pass

                try:
                    vit_x[i_steps][i_id] = s_vit[s_id.index(id)]
                    vit_m[i_steps][i_id] = 1.
                except ValueError:
                    pass

        lab_tseries = [lab_x, lab_m]
        vit_tseries = [vit_x, vit_m]

        return lab_tseries, vit_tseries, flags, ids

    def get_lab_chart_point(self, l_lab_id, l_chart_id, days=0., from_discharge=True):
        """Get data of lab test and chart on a datapoint."""
        ids = []
        lab_values = []
        chart_values = []
        flags = []

        for patient in self.l_patient:
            final_adm = patient.get_final_admission()
            if from_discharge:
                time_of_interest = (final_adm.get_estimated_disch_time()
                                    - timedelta(days))
            else:
                time_of_interest = (final_adm.get_estimated_admit_time()
                                    + timedelta(days))
            lab_result = final_adm.get_newest_lab_at_time(time_of_interest)
            chart_result = final_adm.get_newest_chart_at_time(time_of_interest)

            lab_value = [float('NaN')] * len(l_lab_id)
            for item in lab_result:
                if item[0] in l_lab_id and is_number(item[4]):
                    index = l_lab_id.index(item[0])
                    lab_value[index] = float(item[4])

            chart_value = [float('NaN')] * len(l_chart_id)
            for item in chart_result:
                if item[0] in l_chart_id and is_number(item[4]):
                    index = l_chart_id.index(item[0])
                    chart_value[index] = float(item[4])

            if (True not in np.isnan(lab_value)
                    and True not in np.isnan(chart_value)
                    and patient.hospital_expire_flg in ['Y', 'N']):
                lab_values.append(lab_value)
                chart_values.append(chart_value)
                flags.append(patient.hospital_expire_flg)
                ids.append(patient.subject_id)

        lab_array = np.array(lab_values)
        chart_array = np.array(chart_values)
        flag_array = np.array(flags)

        y = np.zeros(len(flag_array), dtype='int')
        y[flag_array == 'Y'] = 1

        return lab_array, chart_array, y, ids

if __name__ == '__main__':
    pd = PredictDeath()
    pd.n_day_prediction()
