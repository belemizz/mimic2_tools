"""
Script to show summery of  medical record of a patient.
"""

import control_mimic2db
import control_graph
import matplotlib.pyplot as plt

#subject_id = 1924
subject_id = 1855

mimic2db = control_mimic2db.control_mimic2db()
graph = control_graph.control_graph()

patient = mimic2db.patient_class(subject_id)

for admission in patient.admissions:
    base_time = admission.admit_dt

    # CR and BUN
#    lab_itemid1 = 50090
#    lab_itemid2 = 50177
#    filename = "../data/CR_BUN_%d_%d.png"%(subject_id, admission.hadm_id)
#    title = "ID:%d [%s]"%(subject_id, admission.admit_dt)
#    graph.draw_lab_adm_itemid(admission, (lab_itemid1, lab_itemid2), title, filename)

    # All lab tests
    filename = "../data/Lab_%d_%d.png"%(subject_id, admission.hadm_id)
    title = "ID:%d [%s] (Lab)"%(subject_id, admission.admit_dt)
    graph.draw_lab_adm(admission, title, filename)

    # TODO: Save Profile,ICD9 and Notes to a text file
    filename = "../data/Notes_%d_%d.png"%(subject_id, admission.hadm_id)
    
    for icustay in admission.icustays:
        icustay_base_time = icustay.intime

        #Medication
        filename = "../data/Med_%d.png"%icustay.icustay_id
        title = "ID:%d [%s] (Medication)"%(subject_id, admission.admit_dt)
        graph.draw_med_icu(icustay, admission.admit_dt, title, filename)

        #Charts
        filename = "../data/Chart_%d.png"%icustay.icustay_id
        title = "ID:%d [%s] (Chart)"%(subject_id, admission.admit_dt)
        graph.draw_chart_icu(icustay, admission.admit_dt, title, filename)

        # TODO: Show fluid IO
        filename = "../data/Fluid_%d.png"%icustay.icustay_id
        title = "ID:%d [%s] (Fluid)"%(subject_id, admission.admit_dt)
        #graph.draw_fluid_icu(icustay, admission.admit_dt, title, filename)

plt.waitforbuttonpress()
