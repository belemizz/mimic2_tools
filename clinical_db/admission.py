"""
 Admission Class
"""
import pdb

class admission:
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
        result = [item for item in self.labs if item[0] == itemid]
        if len(result) > 1:
            raise Exception("There is more than one record")
        return result[0]

    def get_newest_lab_at_time(self, time_of_interest):
        result = []
        for item in self.labs:
            if item[3][0] < time_of_interest:
                
                over = False
                for i, t in enumerate(item[3]):
                    if t > time_of_interest:
                        over = True
                        break
                
                if over:
                    timestamp = item[3][i-1]
                    value = item[4][i-1]
                else:
                    timestamp = item[3][i]
                    value = item[4][i]
                    
                result.append((item[0], item[1], item[2], timestamp, value))

        return result

    def display_available_lab(self):
        available_labs = [(item[0],len(item[4]),item[1]) for item in self.labs]
        for item in available_labs:
            print item
