"""
classes for handling mimic2 database subject data
"""

import datetime

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

    def set_admissions(self,admission_list):
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
            final_ios_icustays = [stay.final_ios_time for stay in icustay_list]
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
                    timestamp = item.timestamps[i-1]
                    value = item.values[i-1]
                else:
                    timestamp = item.timestamps[i]
                    value = item.values[i]
                    
                result.append([item.itemid, item.description, item.unit, timestamp, value])
        return result

    def get_newest_chart_at_time(self, time_of_interest ):

        valid_stays= [stay for stay in self.icustays if stay.intime < time_of_interest]

        result = []
        if len(valid_stays) > 0:
            stay = valid_stays[len(valid_stays) -1 ]
            for item in stay.charts:
                if item.timestamps[0] < time_of_interest:
                    over = False
                    for i, t in enumerate(item.timestamps):
                        if t > time_of_interest:
                            over = True
                            break
                    if over:
                        timestamp = item.timestamps[i-1]
                        value = item.values[i-1]
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
                result.append([item.itemid, item.description, item.unit, ts_interest, val_interest])
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
                        result.append([item.itemid, item.description, item.unit, ts_interest, val_interest])
                    
        return result

    def get_estimated_disch_time(self):
        return max( [self.final_labs_time,
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
