""" Learn and visualize feature"""

from get_sample import PatientData, Mimic2
from mutil import Graph

from patient_classification import (ControlExperiment, Default_db_param,
                                    Default_data_param, Default_alg_param)
import ipdb

mimic2db = Mimic2()
graph = Graph()


class LearnVisualizeFeature(ControlExperiment):

    def __init__(self,
                 db_param=Default_db_param,
                 data_param=Default_data_param,
                 alg_param=Default_alg_param
                 ):
        ControlExperiment.set_db_param(self, db_param)
        ControlExperiment.set_data_param(self, data_param)
        ControlExperiment.set_alg_param(self, alg_param)
        self.patients = PatientData(self.id_list)

    def execution(self):

        # Display some of chanks
        l_lab_id, l_lab_desc, _ = self.patients.common_lab(2)
        l_chart_id = mimic2db.vital_charts

        from_discharge = False
        [lab_b_ts, lab_b_data, ch_b_ts, ch_b_data, b_flag, subj_b_id, hadm_b_id] \
            = self.patients.data_from_adm(l_lab_id, l_chart_id,
                                          from_discharge=from_discharge)

        for idx, id in enumerate(hadm_b_id):
            title = "ID:{}  Hadm: {}".format(subj_b_id[idx], id)
            filename = "Hadm{}".format(id)
            admission = self.patients.get_admission(subj_b_id[idx], id)

            xlim = [0., 1.]
            if len(admission.l_cont) > 0:
                ts, data = admission.get_continuous_data()
                graph.line_scatter(ts, data, xlim=xlim, title=title,
                                   filename=filename + 'cont')
                l_ts, l_data = time_windows(ts, data, length=0.1, overlap=0.5)



        ipdb.set_trace()

        print "Hello"

if __name__ == '__main__':
    db_param = Default_db_param
    db_param.matched_only = True
    db_param.max_id = 200

    data_param = Default_data_param
    pd = LearnVisualizeFeature(db_param, data_param)
    result = pd.execution()
