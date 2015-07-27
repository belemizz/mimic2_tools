"""Classification of the patinets."""

from mutil import p_info
from get_sample import Mimic2, PatientData

mimic2 = Mimic2()


class ControlClassification:
    """Control classification of the patients."""

    def __init__(self,
                 max_id=200000):
        self.max_id = max_id

    def classify_patient(self):
        data = self.__data_preparation()
        result = self.__eval_data(data)
        self.__visualize(result)

    def __data_preparation(self):
        p_info('Prepare Data')
#        id_list = mimic2.subject_with_icd9_codes(max_id = self.max_id)
#        patients = PatientData(id_list)
        pass

    def __eval_data(self, data):
        p_info('Evaluate Data')
        pass

    def __visualize(self, result):
        p_info('Visualize Result')
        pass
