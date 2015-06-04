"""
Script to show summery of  medical record of a subject.
"""

import control_mimic2db
import control_graph
import control_csv

import matplotlib.pyplot as plt
subject_id = 1924
#subject_id = 1855

mimic2db = control_mimic2db.control_mimic2db()
graph = control_graph.control_graph()

subject = mimic2db.get_subject(subject_id)

for admission in subject.admissions:
    base_time = admission.admit_dt

    # CR and BUN
    lab_itemid1 = 50090
    lab_itemid2 = 50177
    filename = "../data/CR_BUN_%d_%d.png"%(subject_id, admission.hadm_id)
    title = "ID:%d [%s]"%(subject_id, admission.admit_dt)
#    graph.draw_lab_adm_itemid(admission, (lab_itemid1, lab_itemid2), title, filename)

    # All lab tests
    filename = "../data/Lab_%d_%d.png"%(subject_id, admission.hadm_id)
    title = "ID:%d [%s] (Lab)"%(subject_id, admission.admit_dt)
#    graph.draw_lab_adm(admission, title, filename)

    # Save Profile,ICD9 and Notes to a text file
    filename = "../data/Prof_Notes_%d_%d.txt"%(subject_id, admission.hadm_id)
    outfile = control_csv.control_csv(filename)
    outfile.write_single_list(["ICD9"])
    outfile.append_list(admission.icd9)
    outfile.append_single_list(["Notes"])
    outfile.append_list(admission.notes)
    
    for icustay in admission.icustays:
        icustay_base_time = icustay.intime

        #Medication
        filename = "../data/Med_%d.png"%icustay.icustay_id
        title = "ID:%d [%s] (Medication)"%(subject_id, admission.admit_dt)
#        graph.draw_med_icu(icustay, admission.admit_dt, title, filename)

        #Charts
        filename = "../data/Chart_%d.png"%icustay.icustay_id
        title = "ID:%d [%s] (Chart)"%(subject_id, admission.admit_dt)
        print "---------"
        for item in icustay.charts:
            if item.description in ['Heart Rate', 'Respiratory Rate', 'SpO2', 'NBP', 'NBP Mean']:
                print (item.itemid, item.description, item.unit, len(item.values))

#        graph.draw_chart_icu(icustay, admission.admit_dt, title, filename)
        graph.draw_selected_chart_icu(icustay, mimic2db.vital_charts, admission.admit_dt, title, filename)

        #Fluid IO
        filename = "../data/Fluid_%d.png"%icustay.icustay_id
        title = "ID:%d [%s] (Fluid IO)"%(subject_id, admission.admit_dt)
#        graph.draw_io_icu(icustay, admission.admit_dt, title, filename)


plt.waitforbuttonpress()
