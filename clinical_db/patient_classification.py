"""Classification of the patinets."""
from bunch import Bunch
import numpy as np

from mutil import p_info, Cache
from get_sample import PatientData, Mimic2m, Mimic2
import alg.clustering
import alg.classification
import alg.timeseries

mimic2m = Mimic2m()
mimic2 = Mimic2()

Default_db_param = Bunch(max_id=0, target_codes='chf', matched_only=False)
"""Param for database preparation.

:param max_id: maximum of subject id (0 for using all ids)
:param target_codes: keyword of a list of icd9 codes to select subjects
:param matched_only: select only subjects with continuous record
"""

Default_data_param = Bunch(n_lab=20, disch_origin=True, span=[-1., 0.],
                           coef_flag=False, tseries_flag=True, tseries_cycle=0.25)
"""Param for database preparation.
:param tseries_flag: True for use timeseries
:param tseres_cycle: Cycle of the timeseries
"""

Default_alg_param = Bunch(visualize_data=False,
                          class_param=alg.classification.Default_param,
                          tseries_param=alg.timeseries.Default_param,
                          n_cv_fold=10)
"""Param for algorithm
:param class_param: param for classification algorithm
:param tsereis_param: param for timeseries classification algorithm
:param n_cv_fold: number of folds in cross validation
"""


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

    def set_db_param(self, db_param):
        self.db_param = Bunch(db_param.copy())
        self.max_id = db_param.max_id
        self.target_codes = db_param.target_codes
        self.matched_only = db_param.matched_only
        self.id_list = self.__get_id_list()

    def __get_id_list(self):
        if self.target_codes == 'all':
            id_list = mimic2.subject_all(self.max_id)
        elif self.target_codes == 'chf':
            id_list = mimic2.subject_with_chf(self.max_id)
        elif self.target_codes:
            id_list = mimic2.subject_with_icd9_codes(self.target_codes, True, True, self.max_id)

        if self.matched_only:
            id_matched = mimic2m.get_id_numerics(self.max_id)
            id_list = list(set(id_list).intersection(set(id_matched)))

        return sorted(id_list)

    def set_data_param(self, data_param=None):
        """Set and Reset data_param

        :param data_param:  new parameter (None to reset param)
        """
        if data_param is not None:
            self.data_param = Bunch(data_param.copy())
        else:
            data_param = Bunch(self.data_param.copy())

        self.n_lab = data_param.n_lab
        self.disch_origin = data_param.disch_origin
        self.span = data_param.span
        self.coef_flag = data_param.coef_flag
        self.tseries_flag = data_param.tseries_flag
        self.tseries_cycle = data_param.tseries_cycle

    def set_alg_param(self, alg_param=None):
        """Set and Reset alg_param

        :param alg_param:  new parameter (None to reset param)
        """

        if alg_param is not None:
            self.alg_param = Bunch(alg_param.copy())
        else:
            alg_param = Bunch(self.alg_param.copy())

        self.visualize_data = alg_param.visualize_data
        self.class_param = alg_param.class_param
        self.tseries_param = alg_param.tseries_param
        self.n_cv_fold = alg_param.n_cv_fold

    def compare_class_alg(self, l_param):
        result = []
        for param in l_param:
            self.class_param = param
            result.append(self.execution())
        self.set_alg_param()
        return result

    def compare_span(self, l_span, include_point=False):
        result = []
        if include_point:
            self.tseries_flag = False
            result.append(self.execution())

        self.tseries_flag = True
        for span in l_span:
            self.span = span
            result.append(self.execution())

        self.set_data_param()
        return result

    def compare_cycle(self, l_cycle, include_point=False):
        result = []
        if include_point:
            self.tseries_flag = False
            result.append(self.execution())
        self.tseries_flag = True
        for cycle in l_cycle:
            self.tseries_cycle = cycle
            result.append(self.execution())

        self.set_data_param()
        return result

    def compare_coef(self, l_span, include_point=False, include_ts=False):
        result = []
        if include_point:
            self.coef_flag = False
            self.tseries_flag = False
            result.append(self.execution())

        if include_ts:
            self.coef_flag = False
            self.tseries_flag = True
            result.append(self.execution())

        self.coef_flag = True
        self.tseries_flag = False
        for span in l_span:
            self.span = span
            result.append(self.execution())

        self.set_data_param()
        return result


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
            n_patients = patients.n_patient()

            l_icd9, l_icd9_desc = patients.common_icd9(self.n_icd9)
            comat_icd9, hist_icd9, _ = patients.comat_icd9(l_icd9)
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
