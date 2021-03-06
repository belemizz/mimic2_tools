import psycopg2
import getpass

import numpy as np

import mutil.mycsv
from mutil import Cache, p_info, is_number, include_any_number
from collections import Counter
from datetime import timedelta
from datetime import datetime
from sklearn.linear_model import LinearRegression

import os
cont_dir = '../data/matdata/'
files = os.listdir(cont_dir)

from scipy.io import loadmat


def get_rec_time(filename):
    splitted = filename.split('-')
    year = int(splitted[1])
    month = int(splitted[2])
    day = int(splitted[3])
    hour = int(splitted[4])
    minute = int(splitted[5].rstrip('n'))
    return datetime(year, month, day, hour, minute, 0)


class PatientData:
    def __init__(self, id_list):
        """Initializer of this class.
        :param id_list: list of subject id included in the instance
        """
        self.id_list = id_list
        self.l_patient = self.__get_patient_data()

    def __reproduce_param(self):
        """Parameters that are needed for reproduce this class."""
        return {'id_list': self.id_list}

    def __get_patient_data(self, cache_key='__get_patient_data'):
        param = self.__reproduce_param()
        cache = Cache(cache_key)
        try:
            return cache.load(param)
        except IOError:
            l_patient = []
            for id in self.id_list:
                mimic2 = Mimic2()
                patient = mimic2.get_subject(id)
                if patient and len(patient.admissions) > 0:
                    l_patient.append(patient)
                else:
                    p_info("ID %d is not available" % id)
            return cache.save(l_patient, param)

    def n_patient(self):
        return len(self.l_patient)

    def n_adm(self):
        n_adm = 0
        for patient in self.l_patient:
            n_adm += len(patient.admissions)
        return n_adm

    def get_patient(self, subject_id):
        p_list = [p for p in self.l_patient if p.subject_id == subject_id]
        if len(p_list) > 1:
            raise ValueError('There is duplecation in subject id')
        elif len(p_list) is 1:
            return p_list[0]
        else:
            return None

    def get_admission(self, subject_id, hadm_id):
        patient = self.get_patient(subject_id)
        return patient.get_admission(hadm_id)

    def common_lab(self, n_select):
        ids = {}
        descs = {}
        units = {}
        for patient in self.l_patient:
            final_adm = patient.get_final_admission()
            for item in final_adm.labs:
                if item.itemid in ids:
                    ids[item.itemid] = ids[item.itemid] + 1
                else:
                    ids[item.itemid] = 1
                    descs[item.itemid] = item.description
                    units[item.itemid] = item.unit
        common_tests, [l_descs, l_units], _ = self.__get_common_item(ids, n_select, [descs, units])
        return common_tests, l_descs, l_units

    def common_icd9(self, n_select):
        codes = {}
        descs = {}
        for patient in self.l_patient:
            final_adm = patient.get_final_admission()
            for item in final_adm.icd9:
                if item[3] in codes:
                    codes[item[3]] += 1
                else:
                    codes[item[3]] = 1
                    descs[item[3]] = item[4]
        common_icd9, l_desc, _ = self.__get_common_item(codes, n_select, [descs])

        return common_icd9, l_desc[0]

    def common_medication(self, n_select):
        ids = {}
        descs = {}
        units = {}
        for patient in self.l_patient:
            final_adm = patient.get_final_admission()
            fa_id, fa_descs, fa_units = final_adm.get_medication_info()
            descs.update(fa_descs)
            units.update(fa_units)
            for itemid in fa_id:
                if itemid in ids:
                    ids[itemid] += 1
                else:
                    ids[itemid] = 1
        common_meds, [l_desc, l_unit], _ = self.__get_common_item(ids, n_select, [descs, units])
        return common_meds, l_desc, l_unit

    def comat_icd9(self, l_icd9, cache_key='get_comat_icd9'):
        param = locals().copy()
        del param['self']
        param.update(self.__reproduce_param())

        cache = Cache(cache_key)
        try:
            return cache.load(param)
        except IOError:
            comat = self.__comat_helper(l_icd9, l_icd9, self.__icd9_get_func, self.__icd9_get_func)
            return cache.save(comat, param)

    def comat_med(self, l_med):
        return self.__comat_helper(l_med, l_med, self.__med_get_func, self.__med_get_func)

    def comat_icd9_med(self, l_icd9, l_med_id):
        return self.__comat_helper(l_icd9, l_med_id, self.__icd9_get_func, self.__med_get_func)

    def comat_icd9_lab(self, l_icd9, l_med_id):
        return self.__comat_helper(l_icd9, l_med_id, self.__icd9_get_func, self.__lab_get_func)

    def __icd9_get_func(self, admission):
        icd9, _ = admission.get_icd9_info()
        return icd9

    def __med_get_func(self, admission):
        med_id, _, _, = admission.get_medication_info()
        return med_id

    def __lab_get_func(self, admission):
        lab_id, _, _, = admission.get_lab_info()
        return lab_id

    def __comat_helper(self, l_1, l_2, get_func1, get_func2):
        comat = np.zeros((len(l_1), len(l_2))).astype('int')
        hist_1 = np.zeros(len(l_1)).astype('int')
        hist_2 = np.zeros(len(l_2)).astype('int')

        for patient in self.l_patient:
            final_adm = patient.get_final_admission()
            l_item1 = get_func1(final_adm)
            l_item2 = get_func2(final_adm)

            for item1 in l_item1:
                idx_1 = self.__find_index(item1, l_1)
                if idx_1 >= 0:
                    hist_1[idx_1] += 1

            for item2 in l_item2:
                idx_2 = self.__find_index(item2, l_2)
                if idx_2 >= 0:
                    hist_2[idx_2] += 1

            for item1 in l_item1:
                for item2 in l_item2:
                    idx_1 = self.__find_index(item1, l_1)
                    idx_2 = self.__find_index(item2, l_2)
                    if idx_1 >= 0 and idx_2 >= 0:
                        comat[l_1.index(item1)][l_2.index(item2)] += 1

        return comat, hist_1, hist_2

    def __find_index(self, item, l_item):
        try:
            return l_item.index(item)
        except ValueError:
            return None

    def __get_common_item(self, item_dict, n_select, l_acc_dics=[]):
        counter = Counter(item_dict)
        most_common_item = [item[0] for item in counter.most_common(n_select)]
        frequency = [item[1] for item in counter.most_common(n_select)]
        l_common_descs = []
        for acc_dict in l_acc_dics:
            l_acc = []
            for item_id in most_common_item:
                l_acc.append(acc_dict[item_id])
            l_common_descs.append(l_acc)
        return most_common_item, l_common_descs, frequency

    def __death_duration(self, patient, adm_idx):
        if patient.dod:
            admission = patient.admissions[adm_idx]
            dd = (patient.dod - admission.disch_dt).days
        else:
            dd = np.inf
        return dd

    def __readmission_duration(self, patient, adm_idx):
        if adm_idx < len(patient.admissions) - 1:
            admission = patient.admissions[adm_idx]
            rd = (patient.admissions[adm_idx + 1].admit_dt - admission.disch_dt).days
        else:
            rd = np.inf
        return rd

    def __expire_flag(self, patient, adm_idx):
        if adm_idx == len(patient.admissions) - 1 and patient.hospital_expire_flg == 'Y':
            ef = 1
        else:
            ef = 0
        return ef

    def data_from_adm(self, l_lab_id, l_chart_id, from_discharge=True):
        '''Get data from each admission
        :param l_lab_id: Lab ID list of interest
        :param l_chart_id: Chart ID list of interest
        :param from_discharge: True->Set zero point on discharge. False-> set on admission
        '''
        l_subject_id, l_hadm_id = [], []
        l_lab_data, l_chart_data = [], []
        l_lab_ts, l_chart_ts = [], []
        l_expire_flag = []

        def data_and_ts(data, l_id):
            l_data = [[]] * len(l_id)
            l_ts = [[]] * len(l_id)
            for item in data:
                if item[0] in l_id:
                    idx_item = l_id.index(item[0])
                    l_ts[idx_item] = l_ts[idx_item] + item[3]
                    l_data[idx_item] = l_data[idx_item] + item[4]
            return (l_data, l_ts)

        def append_adm_data(patient, idx, admission):
            lab_result = admission.get_lab_in_span(None, None, from_discharge)
            lab_data, lab_ts = data_and_ts(lab_result, l_lab_id)
            chart_result = admission.get_chart_in_span(None, None, from_discharge)
            chart_data, chart_ts = data_and_ts(chart_result, l_chart_id)

            l_lab_data.append(lab_data)
            l_lab_ts.append(lab_ts)
            l_chart_data.append(chart_data)
            l_chart_ts.append(chart_ts)

            ef = self.__expire_flag(patient, idx)
            l_expire_flag.append(ef)

            l_subject_id.append(patient.subject_id)
            l_hadm_id.append(admission.hadm_id)

        for patient in self.l_patient:
            for idx, admission in enumerate(patient.admissions):
                append_adm_data(patient, idx, admission)

        expire_flag = np.array(l_expire_flag).astype('int')
        return l_lab_ts, l_lab_data, l_chart_ts, l_chart_data, expire_flag, l_subject_id, l_hadm_id

    def trend_from_adm(self, l_lab_id, l_chart_id, span,
                       from_discharge=True, accept_none=True, final_adm_only=False):
        '''Get data from each admission

        :param l_lab_id: Lab ID list of interest
        :param l_chart_id: Chart ID list of interest

        :param from_discharge: True->Set zero point on discharge. False-> set on admission
        :param accept_none: True to include data with None , False to exclude data contains None.
        :param final_adm_only: Extract data only from final admissions of the patients
        '''
        l_subject_id, l_hadm_id = [], []
        l_lab_data, l_chart_data = [], []
        l_readm_duration, l_death_duration, l_expire_flag = [], [], []

        def validation(lab_value, chart_value):
            if accept_none:
                return True
            elif (True not in np.isnan(lab_value)
                    and True not in np.isnan(chart_value)):
                return True
            else:
                return False

        def append_adm_data(patient, idx, admission):
            lab_value, chart_value = self.__trend_from_adm(admission,
                                                           l_lab_id, l_chart_id,
                                                           span, from_discharge)
            rd = self.__readmission_duration(patient, idx)
            dd = self.__death_duration(patient, idx)
            ef = self.__expire_flag(patient, idx)
            if validation(lab_value, chart_value):
                l_lab_data.append(lab_value)
                l_chart_data.append(chart_value)
                l_readm_duration.append(rd)
                l_death_duration.append(dd)
                l_expire_flag.append(ef)
                l_subject_id.append(patient.subject_id)
                l_hadm_id.append(admission.hadm_id)

        for patient in self.l_patient:
            if final_adm_only:
                adm_idx = len(patient.admissions) - 1
                admission = patient.admissions[adm_idx]
                append_adm_data(patient, adm_idx, admission)
            else:
                for idx, admission in enumerate(patient.admissions):
                    append_adm_data(patient, idx, admission)

        a_lab = np.array(l_lab_data)
        a_chart = np.array(l_chart_data)
        readm_duration = np.array(l_readm_duration)
        death_duration = np.array(l_death_duration)
        expire_flag = np.array(l_expire_flag).astype('int')
        return a_lab, a_chart, expire_flag, l_subject_id, readm_duration, death_duration, l_hadm_id

    def point_from_adm(self, l_lab_id, l_chart_id, poi=0.,
                       from_discharge=True, accept_none=True,
                       final_adm_only=False):
        """Extract Point Data from MIMIC2 database.

        :param l_lab_id: list of id of lab tests to extract
        :param l_chart_id: list of id of chart tests to extract
        :param poi: point of interest (negative if from_discharge)
        :param from_discharge: set 0 point on discharge point, otherwise 0 point is on admission
        :param accept_none: True to include data with None , False to exclude data contains None.
        :param final_adm_only: Use only final admissions of each patient
        """
        l_subject_id, l_hadm_id = [], []
        l_lab_data, l_chart_data = [], []
        l_readm_duration, l_death_duration, l_expire_flag = [], [], []

        def validation(lab_value, chart_value):
            if accept_none:
                return True
            elif (True not in np.isnan(lab_value)
                  and True not in np.isnan(chart_value)):
                return True
            else:
                return False

        def append_adm_data(patient, idx, admission):
            lab_value, chart_value = self.__point_from_adm(
                admission, l_lab_id, l_chart_id, poi, from_discharge)
            rd = self.__readmission_duration(patient, idx)
            dd = self.__death_duration(patient, idx)
            ef = self.__expire_flag(patient, idx)
            if validation(lab_value, chart_value):
                l_lab_data.append(lab_value)
                l_chart_data.append(chart_value)
                l_readm_duration.append(rd)
                l_death_duration.append(dd)
                l_expire_flag.append(ef)
                l_subject_id.append(patient.subject_id)
                l_hadm_id.append(admission.hadm_id)

        for patient in self.l_patient:
            if final_adm_only:
                adm_idx = len(patient.admissions) - 1
                admission = patient.admissions[adm_idx]
                append_adm_data(patient, adm_idx, admission)
            else:
                for idx, admission in enumerate(patient.admissions):
                    append_adm_data(patient, idx, admission)

        a_lab = np.array(l_lab_data)
        a_chart = np.array(l_chart_data)

        readm_duration = np.array(l_readm_duration)
        death_duration = np.array(l_death_duration)
        expire_flag = np.array(l_expire_flag).astype('int')

        # Statistics
        death_on_disch = death_duration < 1
        alive_on_disch = death_duration >= 1
        death_within_30 = np.logical_and(alive_on_disch, death_duration < 31)
        readm_within_30 = np.logical_and(alive_on_disch, readm_duration < 31)
        death_after_30 = np.logical_and(alive_on_disch, death_duration >= 31)
        readm_after_30 = np.logical_and(alive_on_disch, readm_duration >= 31)

        p_info("#Subject: {} (Total: {})".format(len(set(l_subject_id)), self.n_patient()))
        p_info("#Admission: {} (Total: {})".format(len(l_hadm_id), self.n_adm()))
        p_info("__discharge status__")
        p_info("Death:{}".format(sum(death_on_disch)))
        p_info("Alive:{}".format(sum(alive_on_disch)))
        p_info("__within_30_days__")
        p_info("Death:{}".format(sum(death_within_30)))
        p_info("Readm:{}".format(sum(readm_within_30)))

        p_info("R/D=T/T:{}".format(sum(np.logical_and(readm_within_30, death_within_30))))
        p_info("R/D=T/F:{}".format(sum(np.logical_and(readm_within_30, death_after_30))))
        p_info("R/D=F/T:{}".format(sum(np.logical_and(readm_after_30, death_within_30))))
        p_info("R/D=F/F:{}".format(sum(np.logical_and(readm_after_30, death_after_30))))

        return a_lab, a_chart, expire_flag, l_subject_id, readm_duration, death_duration, l_hadm_id

    def tseries_from_adm(self, l_lab_id, l_chart_id, span, cycle,
                         from_discharge=True, final_adm_only=False):
        """Extract Point Data from MIMIC2 database.

        :param l_lab_id: list of id of lab tests to extract
        :param l_chart_id: list of id of chart tests to extract
        :param span: span of the time series
        :param cycle: cycle of the data
        :param from_discharge: set 0 point on discharge point, otherwise 0 point is on admission
        :param final_adm_only: Use only final admissions of each patient
        """
        l_subject_id, l_hadm_id = [], []
        l_lab_data, l_chart_data = [], []
        l_readm_duration, l_death_duration, l_expire_flag = [], [], []
        n_steps = int((span[1] - span[0]) / cycle)

        def validation(lab_value, chart_value):
            return True  # TODO:add validataion condition

        def append_adm_data(patient, adm_idx, admission):
            a_lab_value = np.zeros((n_steps, len(l_lab_id)))
            a_vit_value = np.zeros((n_steps, len(l_chart_id)))
            for idx in range(n_steps):
                days = span[0] + idx * cycle
                lab_value, chart_value = self.__point_from_adm(
                    admission, l_lab_id, l_chart_id, days, from_discharge)
                a_lab_value[idx, :] = lab_value
                a_vit_value[idx, :] = chart_value

            rd = self.__readmission_duration(patient, adm_idx)
            dd = self.__death_duration(patient, adm_idx)
            ef = self.__expire_flag(patient, adm_idx)

            if validation(lab_value, chart_value):
                l_lab_data.append(a_lab_value)
                l_chart_data.append(a_vit_value)
                l_readm_duration.append(rd)
                l_death_duration.append(dd)
                l_expire_flag.append(ef)
                l_subject_id.append(patient.subject_id)
                l_hadm_id.append(admission.hadm_id)

        for patient in self.l_patient:
            if final_adm_only:
                adm_idx = len(patient.admissions) - 1
                admission = patient.admissions[adm_idx]
                append_adm_data(patient, adm_idx, admission)
            else:
                for adm_idx, admission in enumerate(patient.admissions):
                    append_adm_data(patient, adm_idx, admission)

        lab_x = np.zeros([n_steps, len(l_hadm_id), len(l_lab_id)])
        lab_m = np.zeros([n_steps, len(l_hadm_id)])
        vit_x = np.zeros([n_steps, len(l_hadm_id), len(l_chart_id)])
        vit_m = np.zeros([n_steps, len(l_hadm_id)])

        for idx, adm_id in enumerate(l_hadm_id):
            for isteps in range(n_steps):
                lab_x[isteps][idx] = l_lab_data[idx][isteps]
                lab_m[isteps][idx] = 1.
                vit_x[isteps][idx] = l_chart_data[idx][isteps]
                vit_m[isteps][idx] = 1.

        lab_tseries = [lab_x, lab_m]
        vit_tseries = [vit_x, vit_m]
        readm_duration = np.array(l_readm_duration)
        death_duration = np.array(l_death_duration)
        expire_flag = np.array(l_expire_flag).astype('int')

        return (lab_tseries, vit_tseries, expire_flag, l_subject_id,
                readm_duration, death_duration, l_hadm_id)

    def __point_from_adm(self, admission, l_lab_id, l_chart_id, poi=0., from_discharge=True):
        '''get a datapoint of lab and chart in a admission'''
        lab_result = admission.get_lab_at_point(poi, from_discharge)
        chart_result = admission.get_chart_at_point(poi, from_discharge)

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

        return lab_value, chart_value

    def __trend_from_adm(self, admission, l_lab_id, l_chart_id, span, from_discharge):
        '''get trend data on a datapoint of lab and chart in a admission'''
        lab_result = admission.get_lab_in_span(span[0], span[1], from_discharge)
        chart_result = admission.get_chart_in_span(span[0], span[1], from_discharge)

        def get_linear_coef(data, l_id):
            n_coef = 2
            value = [float('NaN')] * len(l_id) * n_coef
            for item in data:
                if item[0] in l_id and include_any_number(item[4]):
                    idx_item = l_id.index(item[0]) * n_coef
                    X = []
                    Y = []
                    for idx_step in range(len(item[4])):
                        if is_number(item[4][idx_step]):
                            X.append(item[3][idx_step])
                            Y.append(float(item[4][idx_step]))
                    reg = LinearRegression()
                    reg.fit(np.array([X]).transpose(), np.array(Y))

                    value[idx_item] = reg.predict(0.)[0]
                    value[idx_item + 1] = reg.predict(1.)[0] - reg.predict(0.)[0]
            return value

        lab_value = get_linear_coef(lab_result, l_lab_id)
        chart_value = get_linear_coef(chart_result, l_chart_id)
        return lab_value, chart_value


class Mimic2:
    """ MIMIC2 Controller """
    vital_charts = [211, 618, 646, 455]
    vital_descs = ['Heart Rate', 'Respiratory Rate', 'SpO2', 'NBP']
    vital_units = ['BPM', 'BPM', '%', 'mmHg']

    def __init__(self):
        self.conn = psycopg2.connect("dbname=MIMIC2 user=%s" % getpass.getuser())
        self.cur = self.conn.cursor()

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def get_subject(self, subject_id):
        """Get subject data from the database."""
        cache_key = "s%d" % subject_id
        cache = mutil.Cache(cache_key)
        try:
            return cache.load()
        except IOError:
            patient = self.patient(subject_id)
            print patient
            if len(patient) > 0:
                subject_ins = subject(subject_id, patient[0][1], patient[0][2],
                                      patient[0][3], patient[0][4])
                subject_ins.set_admissions(self.get_admission(subject_id))
            return cache.save(subject_ins)

    def get_admission(self, subject_id):
        """Get admission data from the database."""
        admissions = self.admission(subject_id)

        admission_list = []
        for item in admissions:
            admission_ins = admission(item[0], item[2], item[3])

            icd9 = self.icd9_in_admission(admission_ins.hadm_id)
            admission_ins.set_icd9(icd9)

            note_events = self.note_events_in_admission(admission_ins.hadm_id)
            admission_ins.set_notes(note_events)

            icustay_list = self.get_icustay(admission_ins.hadm_id)
            admission_ins.set_icustays(icustay_list)

            labs = self.get_labs(admission_ins.hadm_id)
            admission_ins.set_labs(labs)

            mat_list = [f for f in files if 's{0:05}'.format(subject_id) in f]
            cont_filelist = []

            for filename in mat_list:
                rec_time = get_rec_time(filename)
                if admission_ins.admit_dt < rec_time < admission_ins.disch_dt + timedelta(1):
                    cont_filelist.append(filename)

            admission_ins.l_cont = cont_filelist
            admission_list.append(admission_ins)

        return admission_list

    def get_icustay(self, hadm_id):
        """Get icustay data from the database."""
        icustays = self.icustay_detail_in_admission(hadm_id)
        icustay_list = []
        for item in icustays:
            icustay_ins = icustay(item[0], item[21], item[22])

            medications = self.get_medications(icustay_ins.icustay_id)
            icustay_ins.set_medications(medications)

            charts = self.get_charts(icustay_ins.icustay_id)
            icustay_ins.set_charts(charts)

            ios = self.get_ios(icustay_ins.icustay_id)
            icustay_ins.set_ios(ios)

            icustay_list.append(icustay_ins)

        return icustay_list

    def get_labs(self, hadm_id):
        events = self.lab_events_in_admission(hadm_id)
        itemid_list = set([item[3] for item in events])
        trends = []
        for itemid in itemid_list:
            record = [item for item in events if item[3] == itemid]

            description = record[0][13]
            unit = record[0][8]
            timestamp = [item[4] for item in record]
            values = [item[5] for item in record]
            trend = series(itemid, description, unit, timestamp, values)
            trends.append(trend)
        return trends

    def get_medications(self, icustay_id):
        events = self.med_events_in_icustay(icustay_id)
        itemid_list = set([item[2] for item in events])
        trends = []
        for itemid in itemid_list:
            record = [item for item in events if item[2] == itemid and item[9] is not None]

            if len(record) > 0:
                description = record[0][16]
                doseuom = record[0][10]
                realtime = [item[5] for item in record]
                dose = [item[9] for item in record]
                trend = series(itemid, description, doseuom, realtime, dose)
                trends.append(trend)
        return trends

    def get_charts(self, icustay_id):
        events = self.chart_events_in_icustay(icustay_id)
        itemid_list = set([item[2] for item in events])
        trends = []
        for itemid in itemid_list:
            record = [item for item in events if item[2] == itemid and item[9] is not None]

            if len(record) > 0:
                description = record[0][16]
                uom = record[0][10]
                realtime = [item[5] for item in record]
                value = [item[9] for item in record]
                trend = series(itemid, description, uom, realtime, value)
                trends.append(trend)
        return trends

    def get_ios(self, icustay_id):
        events = self.io_events_in_icustay(icustay_id)
        itemid_list = set([item[2] for item in events])
        trends = []
        for itemid in itemid_list:
            record = [item for item in events if item[2] == itemid and item[9] is not None]
            if len(record) > 0:
                description = record[0][16]
                uom = record[0][10]
                realtime = [item[6] for item in record]
                value = [item[9] for item in record]
                trend = series(itemid, description, uom, realtime, value)
                trends.append(trend)
        return trends

    def subject_all(self, max_id):
        '''Return the list of all subjects.

        :param max_id: maximum subject id (0 for using all ids)
        :return:  list of the subject id
        '''
        subjects = self.all_patient()
        set_id = set([item[0] for item in subjects])
        return self.__limit_and_sort_set_id(set_id, max_id)

    def subject_with_chf(self, max_id=0, max_seq=1):
        '''Return the list of subject IDs who have at least one admission with heart failure icd9 code.

        :param max_id: maximum subject id (0 for using all ids)
        :param max_seq: conditions of sequence
        :return:  list of the subject id
        '''
        chf_related_codes = ['402,01', '402.11', '402.91',
                            '404.01', '404.03', '404.11', '404.13', '404.91', '404.93',
                            '428.%']
        seq_cond = "<=%d" % max_seq
        subjects = self.__subject_with_icd9_or(chf_related_codes, seq_cond)
        set_id = set([item[0] for item in subjects])
        return self.__limit_and_sort_set_id(set_id, max_id)

    def __subject_with_icd9_or(self, l_code, seq_cond):
        where_cond = ''
        for code in l_code:
            if where_cond is not '':
                where_cond += ' OR '
            where_cond += "code LIKE '%s'" % code
        select_seq = "SELECT subject_id,hadm_id " +\
          "FROM mimic2v26.icd9 " +\
                     "WHERE (%s) AND sequence%s " % (where_cond, seq_cond) +\
                     "GROUP BY subject_id,hadm_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq)

    def subject_with_icd9_codes(self, target_codes, ignore_order=True, final_adm=False, max_id=0):
        '''Return the list of subject IDs who have at leaset one admission with ICD9 codes given.

        :param target_codes: ICD9 codes labeled in the admissions of the subject
        :param ignore_order: True if the order of the code is not cared
        :param final_adm: True to check only the final admission of the subject
        :param max_id: maximum subject id (0 for using all ids)
        :return:  list of the subject id
        '''
        id_lists = []
        adm_lists = []
        for index, code in enumerate(target_codes):
            if ignore_order:
                seq_cond = "<=%d" % len(target_codes)
            else:
                seq_cond = "=%d" % (index + 1)
            subjects = self.__subject_with_icd9(code, seq_cond)
            id_lists.append([item[0] for item in subjects])
            adm_lists.append([item[1] for item in subjects])

        id_set = set(id_lists[0])
        adm_set = set(adm_lists[0])
        for index in range(1, len(id_lists)):
            id_set = id_set.intersection(set(id_lists[index]))
            adm_set = adm_set.intersection(set(adm_lists[index]))

        sel_id_set = id_set.copy()
        if final_adm:
            for id in id_set:
                adm = self.admission(id)
                final_adm_id = adm[len(adm) - 1][0]
                if final_adm_id not in adm_set:
                    sel_id_set.remove(id)
        return self.__limit_and_sort_set_id(sel_id_set, max_id)

    def __limit_and_sort_set_id(self, set_id, max_id):
        if max_id > 0:
            lim_id = [id for id in set_id if id < max_id]
        else:
            lim_id = list(set_id)
        return sorted(lim_id)

    def __subject_with_icd9(self, code, seq_cond):
        select_seq = "SELECT subject_id,hadm_id " +\
                     "FROM mimic2v26.icd9 " +\
                     "WHERE code='%s' AND sequence%s" % (code, seq_cond) +\
                     "GROUP BY subject_id,hadm_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq)

    # Basic queries to get items for a patient
    def all_patient(self):
        select_seq = "SELECT subject_id " +\
                     "FROM mimic2v26.D_PATIENTS "
        return self.__select_and_save(select_seq)

    def patient(self, patient_id):
        select_seq = "SELECT * " +\
                     "FROM mimic2v26.D_PATIENTS " +\
                     "WHERE subject_id =%d" % (patient_id)
        return self.__select_and_save(select_seq)

    def admission(self, patient_id):
        select_seq = "SELECT * FROM mimic2v26.ADMISSIONS " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY disch_dt"
        return self.__select_and_save(select_seq)

    def icustay_detail(self, patient_id):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DETAIL " +\
                     "WHERE subject_id =%d " % (patient_id)
        return self.__select_and_save(select_seq)

    def icustay_events(self, patient_id):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAYEVENTS " +\
                     "WHERE subject_id =%d " % (patient_id)
        return self.__select_and_save(select_seq)

    def icustay_days(self, patient_id):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DAYS " +\
                     "WHERE subject_id =%d " % (patient_id)
        return self.__select_and_save(select_seq)

    def icd9(self, patient_id):
        select_seq = "SELECT I.* FROM mimic2v26.ICD9 I " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY hadm_id, sequence"
        return self.__select_and_save(select_seq)

    def med_events(self, patient_id):
        select_seq = "SELECT M.*, T.LABEL " +\
                     "FROM mimic2v26.MEDEVENTS M, mimic2v26.D_MEDITEMS T " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "AND M.ITEMID = T.ITEMID ORDER BY ITEMID, REALTIME"
        return self.__select_and_save(select_seq)

    def lab_items(self, item_id):
        select_seq = "SELECT * " +\
                     "FROM mimic2v26.D_LABITEMS " +\
                     "WHERE itemid =%d " % (item_id)
        return self.__select_and_save(select_seq)

    def note_events(self, patient_id):
        select_seq = "SELECT N.* FROM mimic2v26.NOTEEVENTS N " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY CHARTTIME"
        return self.__select_and_save(select_seq)

    def poe_order(self, patient_id):
        select_seq = "SELECT P.* FROM mimic2v26.POE_ORDER P " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY START_DT"
        return self.__select_and_save(select_seq)

    def lab_events(self, patient_id):
        select_seq = "SELECT L.*, T.TEST_NAME, T.FLUID, T.CATEGORY, T.LOINC_CODE, T.LOINC_DESCRIPTION " +\
                     "FROM mimic2v26.LABEVENTS L, mimic2v26.D_LABITEMS T " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "AND L.ITEMID = T.ITEMID " +\
                     "ORDER BY ITEMID, CHARTTIME"
        return self.__select_and_save(select_seq)

    def io_events(self, patient_id):
        select_seq = "SELECT I.*, T.LABEL, T.CATEGORY " +\
                     "FROM mimic2v26.IOEVENTS I, mimic2v26.D_IOITEMS T " +\
                     "WHERE subject_id =%d AND I.ITEMID = T.ITEMID " % patient_id +\
                     "ORDER BY REALTIME"
        return self.__select_and_save(select_seq)

    def microbiology_events(self, patient_id):
        select_seq = "SELECT M.*, " +\
                     "C.TYPE AS STYPE, C.LABEL AS SLABEL, C.DESCRIPTION AS SDESC, " +\
                     "D.TYPE AS OTYPE, D.LABEL AS OLABEL, D.DESCRIPTION AS ODESC, " +\
                     "E.TYPE AS ATYPE, E.LABEL AS ALABEL, E.DESCRIPTION AS ADESC " +\
                     "FROM mimic2v26.MICROBIOLOGYEVENTS M, mimic2v26.D_CODEDITEMS C, mimic2v26.D_CODEDITEMS D, mimic2v26.D_CODEDITEMS E " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "AND M.SPEC_ITEMID = C.ITEMID AND M.ORG_ITEMID = D.ITEMID AND M.AB_ITEMID = E.ITEMID " +\
                     "ORDER BY CHARTTIME"
        return self.__select_and_save(select_seq)

    # RETURN ELEMENTS WHICH BELONG TO ADMISSION
    def icustay_detail_in_admission(self, hadm_id):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DETAIL " +\
                     "WHERE hadm_id =%d " % (hadm_id)
        return self.__select_and_save(select_seq)

    def icd9_in_admission(self, hadm_id):
        select_seq = "SELECT I.* FROM mimic2v26.ICD9 I " +\
                     "WHERE hadm_id =%d " % (hadm_id) +\
                     "ORDER BY sequence"
        return self.__select_and_save(select_seq)

    def note_events_in_admission(self, hadm_id):
        select_seq = "SELECT N.* FROM mimic2v26.NOTEEVENTS N " +\
                     "WHERE hadm_id =%d " % (hadm_id) +\
                     "ORDER BY CHARTTIME"
        return self.__select_and_save(select_seq)

    def lab_events_in_admission(self, hadm_id):
        select_seq = "SELECT L.*, T.TEST_NAME, T.FLUID, T.CATEGORY, T.LOINC_CODE, T.LOINC_DESCRIPTION " +\
                     "FROM mimic2v26.LABEVENTS L, mimic2v26.D_LABITEMS T " +\
                     "WHERE hadm_id =%d " % (hadm_id) +\
                     "AND L.ITEMID = T.ITEMID " +\
                     "ORDER BY ITEMID, CHARTTIME"
        return self.__select_and_save(select_seq)

    # RETURN ELEMENTS WHICH BELONG TO ICUSTAY
    def med_events_in_icustay(self, icustay_id):
        select_seq = "SELECT M.*, T.LABEL " +\
                     "FROM mimic2v26.MEDEVENTS M, mimic2v26.D_MEDITEMS T " +\
                     "WHERE icustay_id =%d " % (icustay_id) +\
                     "AND M.ITEMID = T.ITEMID ORDER BY ITEMID, REALTIME"
        return self.__select_and_save(select_seq)

    def chart_events_in_icustay(self, icustay_id):
        select_seq = "SELECT C.*, T.LABEL, T.CATEGORY, T.DESCRIPTION " +\
                     "FROM mimic2v26.CHARTEVENTS C, mimic2v26.D_CHARTITEMS T " +\
                     "WHERE icustay_id =%d AND C.ITEMID = T.ITEMID " % icustay_id +\
                     "ORDER BY ITEMID, REALTIME"
        return self.__select_and_save(select_seq)

    def io_events_in_icustay(self, icustay_id):
        select_seq = "SELECT I.*, T.LABEL, T.CATEGORY " +\
                     "FROM mimic2v26.IOEVENTS I, mimic2v26.D_IOITEMS T " +\
                     "WHERE icustay_id =%d AND I.ITEMID = T.ITEMID  " % icustay_id +\
                     "ORDER BY ITEMID, REALTIME"
        return self.__select_and_save(select_seq)

    #  Advanced Queries
    def matched_icustay_detail(self, savepath=""):
        select_seq = "SELECT * FROM mimic2v26.icustay_detail " +\
                     "WHERE matched_waveforms_num>0"
        return self.__select_and_save(select_seq, savepath)

    def icd9_eq_higher_than(self, rank, savepath=""):
        select_seq = "SELECT * FROM mimic2v26.icd9 " +\
                     "WHERE sequence<=%d" % rank +\
                     "ORDER BY subject_id, hadm_id, sequence"
        return self.__select_and_save(select_seq, savepath)

    def subject_matched_waveforms(self, savepath=""):
        select_seq = "SELECT subject_id " +\
                     "FROM mimic2v26.icustay_detail " +\
                     "WHERE matched_waveforms_num>0 " +\
                     "GROUP BY subject_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq, savepath)

    def subject_with_icu_expire_flg(self, savepath=""):
        select_seq = "SELECT subject_id " +\
                     "FROM mimic2v26.icustay_detail " +\
                     "WHERE icustay_expire_flg='Y' " +\
                     "GROUP BY subject_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq, savepath)

    def __select_and_save(self, select_seq, filepath="", print_query=False):

        if print_query:
            print "exec:"
            print select_seq

        self.cur.execute(select_seq)
        result = self.cur.fetchall()

        if len(filepath) > 0:
            import csv
            writer = csv.writer(open(filepath, 'wb'))
            writer.writerows(result)

        return result


class subject:
    """
    Subject Class
    """
    def __init__(self, subject_id, sex, dob, dod, hospital_expire_flg):
        self.subject_id = subject_id
        self.sex = sex
        self.dob = dob
        self.dod = dod
        self.hospital_expire_flg = hospital_expire_flg

    def set_admissions(self, admission_list):
        self.admissions = admission_list

    def get_final_admission(self):
        return self.admissions[len(self.admissions) - 1]

    def get_admission(self, hadm_id):
        adm_list = [a for a in self.admissions if a.hadm_id == hadm_id]
        if len(adm_list) > 1:
            raise ValueError('There is duplecation in subject id')
        elif len(adm_list) is 1:
            return adm_list[0]
        else:
            return None


class admission:
    """
    Admission Class
    """
    def __init__(self, hadm_id, admit_dt, disch_dt):
        self.hadm_id = hadm_id
        self.admit_dt = admit_dt
        self.disch_dt = disch_dt

    def set_icd9(self, icd9):
        self.icd9 = icd9

    def set_notes(self, note_events):
        self.notes = note_events

    def set_icustays(self, icustay_list):
        self.icustays = icustay_list

        if len(icustay_list) > 0:
            self.final_ios_time = max([stay.final_ios_time for stay in icustay_list])
            self.final_chart_time = max([stay.final_chart_time for stay in icustay_list])
            self.final_medication_time = max([stay.final_medication_time for stay in icustay_list])
        else:
            self.final_ios_time = datetime.min
            self.final_chart_time = datetime.min
            self.final_medication_time = datetime.min

        if len(icustay_list) > 0:
            self.first_ios_time = min([stay.first_ios_time for stay in icustay_list])
            self.first_chart_time = min([stay.first_chart_time for stay in icustay_list])
            self.first_medication_time = min([stay.first_medication_time for stay in icustay_list])
        else:
            self.first_ios_time = datetime.min
            self.first_chart_time = datetime.min
            self.first_medication_time = datetime.min

    def set_labs(self, lab_event_trends):
        self.labs = lab_event_trends
        self.final_labs_time = final_timestamp(self.labs)
        self.first_labs_time = first_timestamp(self.labs)

    # simple getter
    def get_lab_info(self):
        ids = []
        descs = {}
        units = {}
        for lab in self.labs:
            ids.append(lab.itemid)
            descs[lab.itemid] = lab.description
            units[lab.itemid] = lab.unit
        return ids, descs, units

    def get_chart_info(self):
        ids = []
        descs = {}
        units = {}
        for icustay in self.icustays:
            for chart in icustay.charts:
                if chart.itemid is not ids:
                    ids.append(chart.itemid)
                    descs[chart.itemid] = chart.description
                    units[chart.itemid] = chart.unit
        return ids, descs, units

    def get_medication_info(self):
        ids = []
        descs = {}
        units = {}
        for icustay in self.icustays:
            for medication in icustay.medications:
                if medication.itemid is not ids:
                    ids.append(medication.itemid)
                    descs[medication.itemid] = medication.description
                    units[medication.itemid] = medication.unit
        return ids, descs, units

    def get_icd9_info(self):
        codes = [item[3] for item in self.icd9]
        descs = [item[4] for item in self.icd9]
        return codes, descs

    def get_lab_itemid(self, itemid):
        result = [item for item in self.labs if item.itemid == itemid]
        if len(result) > 1:
            raise Exception("There is more than one record")
        return result[0]

    def get_lab_at_point(self, poi, from_discharge=True):
        '''Get lab data at the selected point
        :param poi: point of interest (set negative value if from_discharge)
        :param from_discharge: count time from discharge point (False: admission point)
        '''
        all_data = self.get_lab_in_span(None, poi, from_discharge)
        return self.__point_from_span(all_data)

    def get_chart_at_point(self, poi, from_discharge=True):
        '''Get chart data at the selected point
        :param poi: point of interest (set negative value if from_discharge)
        :param from_discharge: count time from discharge point (False: admission point)
        '''
        all_data = self.get_chart_in_span(None, poi, from_discharge)
        return self.__point_from_span(all_data)

    def __point_from_span(self, all_data):
        for data in all_data:
            data[3] = data[3][-1]
            data[4] = data[4][-1]
        return all_data

    def get_lab_in_span(self, begin_pt, end_pt, from_discharge=True):
        '''Get lab data in this admission.

        :param begin_pt: begin point of the span (unit: days, None for minimum)
        :param end_pt: end point of the span (unit: days, None for maximum)
        :param from_discharge: count time from discharge point (False: admission point)
        '''
        zero_point = self.__zero_point(from_discharge)
        toi_begin, toi_end = self.__toi_span(begin_pt, end_pt, zero_point)

        result = []
        for item in self.labs:
            first_timestamp = item.timestamps[0]
            final_timestamp = item.timestamps[len(item.timestamps) - 1]

            if first_timestamp <= toi_end and final_timestamp >= toi_begin:

                ts_interest = []
                val_interest = []

                for (ts, val) in zip(item.timestamps, item.values):
                    if toi_begin <= ts <= toi_end:
                        pt = (ts - zero_point).total_seconds() / 86400
                        ts_interest.append(pt)
                        val_interest.append(val)
                result.append([item.itemid, item.description,
                               item.unit, ts_interest, val_interest])
        return result

    def get_chart_in_span(self, begin_pt, end_pt, from_discharge=True):
        '''Get chart data in this admission.

        :param begin_pt: begin point of the span (unit: days, None for minimum)
        :param end_pt: end point of the span (unit: days, None for maximum)
        :param from_discharge: count time from discharge point (False: admission point)
        '''
        zero_point = self.__zero_point(from_discharge)
        toi_begin, toi_end = self.__toi_span(begin_pt, end_pt, zero_point)

        valid_stays = []
        for stay in self.icustays:
            if toi_begin <= stay.outtime and toi_end >= stay.intime:
                valid_stays.append(stay)
        result = []
        if len(valid_stays) > 0:
            for stay in valid_stays:
                for item in stay.charts:
                    first_timestamp = item.timestamps[0]
                    final_timestamp = item.timestamps[len(item.timestamps) - 1]

                    if first_timestamp <= toi_end and final_timestamp >= toi_begin:
                        ts_interest = []
                        val_interest = []

                        for (ts, val) in zip(item.timestamps, item.values):
                            if toi_begin <= ts <= toi_end:
                                pt = (ts - zero_point).total_seconds() / 86400
                                ts_interest.append(pt)
                                val_interest.append(val)
                        result.append([item.itemid, item.description,
                                       item.unit, ts_interest, val_interest])
        return result

    def get_continuous_data(self, preprocess=True):
        ts_all = None
        data_all = None
        for idx, cont_filename in enumerate(self.l_cont):
            cont_data = loadmat(cont_dir + cont_filename)
            rec_time = get_rec_time(cont_filename)
            time_dif = rec_time - self.get_estimated_admit_time()
            ts = [(cont_data['time'][0, i].flatten() + time_dif.seconds) / 3600. / 24.
                  for i in range(cont_data['time'].shape[1])]
            data = [cont_data['data'][0, i].flatten()
                    for i in range(cont_data['data'].shape[1])]
            if idx == 0:
                ts_all = ts
                data_all = data
            else:
                for idx in range(min(len(ts_all), len(ts))):
                    ts_all[idx] = np.append(ts[idx], ts_all[idx])
                    data_all[idx] = np.append(data[idx], data_all[idx])

        if preprocess:
            return self.__prepro_cont(ts_all, data_all)
        else:
            return (ts_all, data_all)

    def __prepro_cont(self, ts, data):
        ts_pre = []
        data_pre = []

        def validate_data(data_array):
                return np.logical_and(~np.isnan(data_array), ~(data_array == 0.))
        for idx in range(len(ts)):
            valid_idx = validate_data(data[idx])
            ts_pre.append(ts[idx][valid_idx])
            data_pre.append(data[idx][valid_idx])
        return ts_pre, data_pre

    def __zero_point(self, from_discharge):
        if from_discharge:
            return self.get_estimated_disch_time()
        else:
            return self.get_estimated_admit_time()

    def __toi_span(self, begin_pt, end_pt, zero_pt):
        if begin_pt is None:
            toi_begin = datetime.min
        else:
            try:
                toi_begin = zero_pt + timedelta(begin_pt)
            except OverflowError:
                toi_begin = zero_pt

        if end_pt is None:
            toi_end = datetime.max
        else:
            try:
                toi_end = zero_pt + timedelta(end_pt)
            except OverflowError:
                toi_end = zero_pt

        return (toi_begin, toi_end)

    def get_estimated_disch_time(self):
        return max([self.final_labs_time,
                    self.final_ios_time,
                    self.final_medication_time,
                    self.final_chart_time])

    def get_estimated_admit_time(self):
        return min([self.first_labs_time,
                    self.first_ios_time,
                    self.first_medication_time,
                    self.first_chart_time])


class icustay:
    """Icustay class."""
    def __init__(self, icustay_id, intime, outtime):
        self.icustay_id = icustay_id
        self.intime = intime
        self.outtime = outtime

    def set_medications(self, medications):
        self.medications = medications
        self.final_medication_time = final_timestamp(medications)
        self.first_medication_time = first_timestamp(medications)

    def set_charts(self, charts):
        self.charts = charts
        self.final_chart_time = final_timestamp(charts)
        self.first_chart_time = first_timestamp(charts)

    def set_ios(self, ios):
        self.ios = ios
        self.final_ios_time = final_timestamp(ios)
        self.first_ios_time = first_timestamp(ios)


class series:
    """
    Series Class
    """
    def __init__(self, itemid, description, unit, timestamps, values):

        self.itemid = itemid
        self.description = description
        self.unit = unit
        self.timestamps = timestamps
        self.values = values


def final_timestamp(list_of_series):
    if len(list_of_series) > 0:
        final_ts = [max(series.timestamps) for series in list_of_series]
        return max(final_ts)
    else:
        return datetime.min


def first_timestamp(list_of_series):
    if len(list_of_series) > 0:
        first_ts = [min(series.timestamps) for series in list_of_series]
        return min(first_ts)
    else:
        return datetime.max
