"""
 Admission Class
"""
class admission:
    def __init__(self, hadm_id, admit_dt, disch_dt):
        self.hadm_id = hadm_id
        self.admit_dt = admit_dt
        self.disch_dt = disch_dt

    def set_icustays(self, icustay_list):
        self.icustays = icustay_list

    def set_lab(self, lab_event_trends):
        self.lab = lab_event_trends
        
    def get_lab_itemid(self, itemid):
        result = [item for item in self.lab if item[0] == itemid]
        if len(result) > 1:
            raise Exception("There is more than one record")
        return result[0]
