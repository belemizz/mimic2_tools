"""
classes for handling mimic2 database subject data
"""

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

    def set_labs(self, lab_event_trends):
        self.labs = lab_event_trends
        
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
        
        all_labs = []
        for item in self.icustays:
            all_labs.extend(item.charts)

        result = []
        for item in all_labs:
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

    ## def display_available_lab(self):
    ##     available_labs = [(item[0],len(item[4]),item[1]) for item in self.labs]
    ##     for item in available_labs:
    ##         print item



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
        
    def set_charts(self, charts):
        self.charts = charts

    def set_ios(self, ios):
        self.ios = ios


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
