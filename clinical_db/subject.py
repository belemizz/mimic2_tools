"""
 Subject Class
"""
class subject:
    def __init__(self, subject_id, sex, dob, dod, hospital_expire_flg):
        self.subject_id = subject_id
        self.sex = sex
        self.dob = dob
        self.dod = dod
        self.holpital_expire_flg = hospital_expire_flg

    def set_admissions(self,admission_list):
        self.admissions = admission_list
        
