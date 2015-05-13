'''
show distirbution of selected lab tests
'''

import control_mimic2db
import control_graph

import matplotlib.pyplot as plt

subject_ids = [1855]

mimic2db = control_mimic2db.control_mimic2db()
graph = control_graph.control_graph()

sid = subject_ids[0]

patient = mimic2db.patient_class(sid)

print (patient.subject_id, patient.holpital_expire_flg)

admission = patient.admissions[0]
print (patient.dod, admission.disch_dt)
