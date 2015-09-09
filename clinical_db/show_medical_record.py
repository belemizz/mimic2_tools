"""
Script to show summery of  medical record of a subject.
"""
from get_sample import Mimic2, PatientData
from mutil import Graph, Csv

mimic2db = Mimic2()
graph = Graph()


def visualize_data(l_subject_id):
    patients = PatientData(l_subject_id)
    l_lab_id = [50017]
    l_chart_id = mimic2db.vital_charts

    data = patients.trend_from_adm(l_lab_id, l_chart_id, days=0., span=10.,
                                   from_discharge=False)
    import ipdb
    ipdb.set_trace()


def show_records(subject_id):
    subject = mimic2db.get_subject(subject_id)
    for admission in subject.admissions:
        # CR and BUN
        lab_itemid1 = 50090
        lab_itemid2 = 50177
        filename = "CR_BUN_%d_%d.png" % (subject_id, admission.hadm_id)
        title = "ID:%d [%s]" % (subject_id, admission.admit_dt)
        graph.draw_lab_adm_itemid(admission, (lab_itemid1, lab_itemid2), title, filename)

        # All lab tests
        filename = "Lab_%d_%d.png" % (subject_id, admission.hadm_id)
        title = "ID:%d [%s] (Lab)" % (subject_id, admission.admit_dt)
        graph.draw_lab_adm(admission, title, filename)

        # Save Profile,ICD9 and Notes to a text file
        filename = "../data/Prof_Notes_%d_%d.txt" % (subject_id, admission.hadm_id)
        outfile = Csv(filename)
        outfile.write_single_list(["ICD9"])
        outfile.append_list(admission.icd9)
        outfile.append_single_list(["Notes"])
        outfile.append_list(admission.notes)

        for icustay in admission.icustays:
            # Medication
            filename = "Med_%d.png" % icustay.icustay_id
            title = "ID:%d [%s] (Medication)" % (subject_id, admission.admit_dt)
            graph.draw_med_icu(icustay, admission.admit_dt, title, filename)

            # Charts
            filename = "Chart_%d.png" % icustay.icustay_id
            title = "ID:%d [%s] (Chart)" % (subject_id, admission.admit_dt)
            print "---------"
            for item in icustay.charts:
                if item.description in ['Heart Rate', 'Respiratory Rate', 'SpO2',
                                        'NBP', 'NBP Mean']:
                    print (item.itemid, item.description, item.unit, len(item.values))

            graph.draw_chart_icu(icustay, admission.admit_dt, title, filename)
            graph.draw_selected_chart_icu(icustay, mimic2db.vital_charts,
                                          admission.admit_dt, title, filename)

            # Fluid IO
            filename = "Fluid_%d.png" % icustay.icustay_id
            title = "ID:%d [%s] (Fluid IO)" % (subject_id, admission.admit_dt)
            graph.draw_io_icu(icustay, admission.admit_dt, title, filename)

if __name__ == '__main__':
    subject_id = 1855
    visualize_data([subject_id])
#    show_records(subject_id)

    graph.waitforbuttonpress()

