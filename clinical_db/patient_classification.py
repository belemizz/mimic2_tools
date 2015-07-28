"""Classification of the patinets."""

from mutil import p_info
from get_sample import PatientData, Mimic2m, Mimic2

mimic2m = Mimic2m()
mimic2 = Mimic2()


class ControlExperiment:
    """Base class of all the experiments class"""
    def __init__(self,
                 max_id,
                 target_codes,
                 matched_only
                 ):
        self.max_id = max_id
        self.target_codes = target_codes
        self.matched_only = matched_only
        self.id_list = self.__get_id_list()

    def __get_id_list(self):

        if self.matched_only:
            id_list = mimic2m.get_id_numerics(self.max_id)
        else:
            id_list = mimic2.get_all_subject(self.max_id)

        if self.target_codes:
            id_code = mimic2.subject_with_icd9_codes(self.target_codes, True, True, self.max_id)
            id_list = list(set(id_code).intersection(set(id_list)))
        return sorted(id_list)


class ControlClassification(ControlExperiment):
    """Control classification of the patients."""

    def __init__(self,
                 max_id=10000,
                 target_codes=None,
                 matched_only=True,
                 n_lab=None,
                 n_med=None,
                 n_icd9=None
                 ):
        ControlExperiment.__init__(self, max_id, target_codes, matched_only)
        self.n_lab = n_lab
        self.n_med = n_med
        self.n_icd9 = n_icd9

    def classify_patient(self):
        data = self.__data_preparation()
        result = self.__eval_data(data)
        self.__visualize(result)

    def __data_preparation(self):
        p_info('Prepare Data')
        patients = PatientData(self.id_list)
        l_icd9, l_icd9_desc = patients.get_common_icd9(self.n_icd9)
        l_lab, l_lab_descs, l_lab_units = patients.get_common_labs(self.n_lab)
        l_med, l_med_descs, l_lab_units = patients.get_common_medication(self.n_med)
        coo1 = patients.get_comat_icd9_med(l_icd9, l_med)
        coo2 = patients.get_comat_icd9_lab(l_icd9, l_lab)
        coo3 = patients.get_comat_icd9(l_icd9)
        coo4 = patients.get_comat_med(l_med)

        import ipdb
        ipdb.set_trace()

    def __eval_data(self, data):
        p_info('Evaluate Data')
        pass

    def __visualize(self, result):
        p_info('Visualize Result')
        pass

if __name__ == '__main__':
    cc = ControlClassification()
    cc.classify_patient()
