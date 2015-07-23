import psycopg2
import getpass

import numpy as np

import mutil.mycsv
import datetime
from mutil import Cache, p_info, is_number
from collections import Counter
from datetime import timedelta


class PatientData:
    def __init__(self, id_list):
        self.id_list = id_list
        self.l_patient = self.__get_patient_data()

    def __get_patient_data(self, cache_key='__get_patient_data'):
        param = self.__dict__
        cache = Cache(cache_key)
        try:
            return cache.load(param)
        except IOError:
            l_patient = []
            for id in self.id_list:
                mimic2 = Mimic2()
                patient = mimic2.get_subject(id)
                if patient:
                    l_patient.append(patient)
                else:
                    p_info("ID %d is not available" % id)
            return cache.save(l_patient, param)

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


class Mimic2:
    """ MIMIC2 Controller """
    def __init__(self):
        self.conn = psycopg2.connect("dbname=MIMIC2 user=%s" % getpass.getuser())
        self.cur = self.conn.cursor()
        self.vital_charts = [211, 618, 646, 455]
        self.vital_descs = ['Heart Rate', 'Respiratory Rate', 'SpO2', 'NBP']
        self.vital_units = ['BPM', 'BPM', '%', 'mmHg']

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def get_subject(self, subject_id):

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
            admission_list.append(admission_ins)

        return admission_list

    def get_icustay(self, hadm_id):
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

    def subject_with_icd9_codes(self, target_codes, ignore_order=True, final_adm=False, max_id=0):
        ''' Search subject ID the target ICD9 codes
        :return:  list of ids
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

        if max_id > 0:
            lim_id = [id for id in sel_id_set if id < max_id]
        else:
            lim_id = list(sel_id_set)

        return sorted(lim_id)

    def __subject_with_icd9(self, code, seq_cond):
        select_seq = "SELECT subject_id,hadm_id " +\
                     "FROM mimic2v26.icd9 " +\
                     "WHERE code='%s' AND sequence%s" % (code, seq_cond) +\
                     "GROUP BY subject_id,hadm_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq)

    def subject_with_numeric(self):
        ''' Search subject ID who have one or more numeric record
        :return:  list of ids
        '''

    # Basic queries to get items for a patient
    def patient(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_patient.csv" % patient_id

        select_seq = "SELECT * " +\
                     "FROM mimic2v26.D_PATIENTS " +\
                     "WHERE subject_id =%d" % (patient_id)
        return self.__select_and_save(select_seq, savepath)

    def admission(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_admission.csv" % patient_id

        select_seq = "SELECT * FROM mimic2v26.ADMISSIONS " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY disch_dt"
        return self.__select_and_save(select_seq, savepath)

    def icustay_detail(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_icustay_detail.csv" % patient_id

        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DETAIL " +\
                     "WHERE subject_id =%d " % (patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icustay_events(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_icustay_events.csv" % patient_id

        select_seq = "SELECT * FROM mimic2v26.ICUSTAYEVENTS " +\
                     "WHERE subject_id =%d " % (patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icustay_days(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_icustay_days.csv" % patient_id

        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DAYS " +\
                     "WHERE subject_id =%d " % (patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icd9(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_icd9.csv" % patient_id

        select_seq = "SELECT I.* FROM mimic2v26.ICD9 I " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY hadm_id, sequence"
        return self.__select_and_save(select_seq, savepath)

    def med_events(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_med_events.csv" % patient_id

        select_seq = "SELECT M.*, T.LABEL " +\
                     "FROM mimic2v26.MEDEVENTS M, mimic2v26.D_MEDITEMS T " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "AND M.ITEMID = T.ITEMID ORDER BY ITEMID, REALTIME"
        return self.__select_and_save(select_seq, savepath)

    def lab_items(self, item_id):
        select_seq = "SELECT * " +\
                     "FROM mimic2v26.D_LABITEMS " +\
                     "WHERE itemid =%d " % (item_id)
        return self.__select_and_save(select_seq)

    def note_events(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_note_events.csv" % patient_id

        select_seq = "SELECT N.* FROM mimic2v26.NOTEEVENTS N " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY CHARTTIME"
        return self.__select_and_save(select_seq, savepath)

    def poe_order(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_poe_order.csv" % patient_id

        select_seq = "SELECT P.* FROM mimic2v26.POE_ORDER P " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "ORDER BY START_DT"

        return self.__select_and_save(select_seq, savepath)

    def lab_events(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_lab_events.csv" % patient_id

        select_seq = "SELECT L.*, T.TEST_NAME, T.FLUID, T.CATEGORY, T.LOINC_CODE, T.LOINC_DESCRIPTION " +\
                     "FROM mimic2v26.LABEVENTS L, mimic2v26.D_LABITEMS T " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "AND L.ITEMID = T.ITEMID " +\
                     "ORDER BY ITEMID, CHARTTIME"
        return self.__select_and_save(select_seq, savepath)

    def io_events(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_io_events.csv" % patient_id

        select_seq = "SELECT I.*, T.LABEL, T.CATEGORY " +\
                     "FROM mimic2v26.IOEVENTS I, mimic2v26.D_IOITEMS T " +\
                     "WHERE subject_id =%d AND I.ITEMID = T.ITEMID " % patient_id +\
                     "ORDER BY REALTIME"
        return self.__select_and_save(select_seq, savepath)

    def microbiology_events(self, patient_id, savepath=""):
        if len(savepath) == 0:
            savepath = "../data/%d_microbiology_events.csv" % patient_id

        select_seq = "SELECT M.*, " +\
                     "C.TYPE AS STYPE, C.LABEL AS SLABEL, C.DESCRIPTION AS SDESC, " +\
                     "D.TYPE AS OTYPE, D.LABEL AS OLABEL, D.DESCRIPTION AS ODESC, " +\
                     "E.TYPE AS ATYPE, E.LABEL AS ALABEL, E.DESCRIPTION AS ADESC " +\
                     "FROM mimic2v26.MICROBIOLOGYEVENTS M, mimic2v26.D_CODEDITEMS C, mimic2v26.D_CODEDITEMS D, mimic2v26.D_CODEDITEMS E " +\
                     "WHERE subject_id =%d " % (patient_id) +\
                     "AND M.SPEC_ITEMID = C.ITEMID AND M.ORG_ITEMID = D.ITEMID AND M.AB_ITEMID = E.ITEMID " +\
                     "ORDER BY CHARTTIME"

        return self.__select_and_save(select_seq, savepath)

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
            self.final_ios_time = datetime.datetime.min
            self.final_chart_time = datetime.datetime.min
            self.final_medication_time = datetime.datetime.min

    def set_labs(self, lab_event_trends):
        self.labs = lab_event_trends
        self.final_labs_time = final_timestamp(self.labs)

    def get_lab_itemid(self, itemid):
        result = [item for item in self.labs if item.itemid == itemid]
        if len(result) > 1:
            raise Exception("There is more than one record")
        return result[0]

    def get_newest_lab_at_time(self, time_of_interest):
        result = []
        for item in self.labs:
            if item.timestamps[0] < time_of_interest:

                over = False
                for i, t in enumerate(item.timestamps):
                    if t > time_of_interest:
                        over = True
                        break

                if over:
                    timestamp = item.timestamps[i - 1]
                    value = item.values[i - 1]
                else:
                    timestamp = item.timestamps[i]
                    value = item.values[i]

                result.append([item.itemid, item.description, item.unit, timestamp, value])
        return result

    def get_newest_chart_at_time(self, time_of_interest):
        valid_stays = [stay for stay in self.icustays if stay.intime < time_of_interest]

        result = []
        if len(valid_stays) > 0:
            stay = valid_stays[len(valid_stays) - 1]
            for item in stay.charts:
                if item.timestamps[0] < time_of_interest:
                    over = False
                    for i, t in enumerate(item.timestamps):
                        if t > time_of_interest:
                            over = True
                            break
                    if over:
                        timestamp = item.timestamps[i - 1]
                        value = item.values[i - 1]
                    else:
                        timestamp = item.timestamps[i]
                        value = item.values[i]
                    result.append([item.itemid, item.description, item.unit, timestamp, value])
        return result

    def get_lab_in_span(self, toi_begin, toi_end):
        result = []
        for item in self.labs:
            first_timestamp = item.timestamps[0]
            final_timestamp = item.timestamps[len(item.timestamps) - 1]

            if first_timestamp <= toi_end and final_timestamp >= toi_begin:

                ts_interest = []
                val_interest = []

                for (ts, val) in zip(item.timestamps, item.values):
                    if toi_begin <= ts <= toi_end:
                        ts_interest.append(ts)
                        val_interest.append(val)
                result.append([item.itemid, item.description,
                               item.unit, ts_interest, val_interest])
        return result

    def get_chart_in_span(self, toi_begin, toi_end):

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
                                ts_interest.append(ts)
                                val_interest.append(val)
                        result.append([item.itemid, item.description,
                                       item.unit, ts_interest, val_interest])

        return result

    def get_estimated_disch_time(self):
        return max([self.final_labs_time,
                    self.final_ios_time,
                    self.final_medication_time,
                    self.final_chart_time])

    def get_estimated_admit_time(self):
        return min([self.final_labs_time,
                    self.final_ios_time,
                    self.final_medication_time,
                    self.final_chart_time])


class icustay:
    """
    icustay class
    """
    def __init__(self, icustay_id, intime, outtime):
        self.icustay_id = icustay_id
        self.intime = intime
        self.outtime = outtime

    def set_medications(self, medications):
        self.medications = medications
        self.final_medication_time = final_timestamp(medications)

    def set_charts(self, charts):
        self.charts = charts
        self.final_chart_time = final_timestamp(charts)

    def set_ios(self, ios):
        self.ios = ios
        self.final_ios_time = final_timestamp(ios)


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
        return datetime.datetime.min
