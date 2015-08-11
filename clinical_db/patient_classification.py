"""Classification of the patinets."""

from mutil import p_info, Cache
from get_sample import PatientData, Mimic2m, Mimic2
import alg.clustering
import numpy as np


mimic2m = Mimic2m()
mimic2 = Mimic2()


class ControlExperiment:
    """Base class of all the experiments class"""
    def __init__(self,
                 max_id,
                 target_codes,
                 matched_only
                 ):
        param = locals().copy()
        del param['self']
        self.reproduction_param = param

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
                 max_id=2000,
                 target_codes=None,
                 matched_only=True,
                 n_lab=None,
                 n_med=None,
                 n_icd9=None
                 ):
        param = locals().copy()
        del param['self']
        self.reproduction_param = param

        ControlExperiment.__init__(self, max_id, target_codes, matched_only)
        self.n_lab = n_lab
        self.n_med = n_med
        self.n_icd9 = n_icd9

    def classify_patient(self):
        data = self.__data_preparation()
        result = self.__eval_data(data)
        self.__visualize(result)

    def __data_preparation(self, cache_key='__data_preparation'):
        p_info('Prepare Data')
        cache = Cache(cache_key)
        try:
            raise IOError
            return cache.load(self.reproduction_param)
        except IOError:
            patients = PatientData(self.id_list)
            n_patients = patients.get_n_patient()

            l_icd9, l_icd9_desc = patients.get_common_icd9(self.n_icd9)
            comat_icd9, hist_icd9, _ = patients.get_comat_icd9(l_icd9)
            retval = [n_patients, l_icd9, comat_icd9, hist_icd9]
            return retval
            return cache.save(retval, self.reproduction_param)

    def __eval_data(self, data):
        p_info('Eval Data')
        [n_patients, l_icd9, comat_icd9, hist_icd9] = data
        l_group = alg.clustering.group_comat(comat_icd9, hist_icd9, n_patients, 3.0, 0.1)
        return [l_group, l_icd9]

    def __visualize(self, result):
        p_info('Visualize Result')
        [l_group, l_icd9] = result

        for g in set(l_group):
            print "Group %d" % g
            for idx in np.where(l_group == g)[0]:
                print l_icd9[idx]
            print '_____'

if __name__ == '__main__':
    cc = ControlClassification()
    cc.classify_patient()
