'''Statistics of mimic2 database'''

from get_sample import Mimic2
from mutil import Graph
import numpy as np
from matplotlib.pyplot import waitforbuttonpress

mimic2 = Mimic2()
graph = Graph()


def readmission_statisic():
    l_id = mimic2.subject_with_chf(max_seq=1)
    total_admission = 0

    alive_on_disch = 0
    death_on_disch = 0

    readm_and_death_within_th = 0
    readm_and_no_death_within_th = 0
    no_readm_death_within_th = 0
    no_readm_no_death_within_th = 0

    duration_th = 30

    readm_after_th = 0
    no_readm_after_th = 0

    l_adm_duration = []
    l_readm_duration = []
    l_death_duration = []
    l_n_admission = []

    for id in l_id:
        subject = mimic2.patient(id)

        death_dt = subject[0][3]
        admissions = mimic2.admission(id)
        l_n_admission.append(len(admissions))
        total_admission += len(admissions)

        for idx, adm in enumerate(admissions):
            admit_dt = admissions[idx][2]
            disch_dt = admissions[idx][3]

            adm_duratrion = (disch_dt - admit_dt).days
            l_adm_duration.append(adm_duratrion)

            if death_dt is not None:
                death_duration = (death_dt - disch_dt).days
            else:
                death_duration = np.inf
            l_death_duration.append(death_duration)

            if idx < len(admissions) - 1:
                next_adm_dt = admissions[idx + 1][2]
                readm_duration = (next_adm_dt - disch_dt).days
            else:
                readm_duration = np.inf
            l_readm_duration.append(readm_duration)

            # counter
            if death_duration < 1:
                death_on_disch += 1
            else:
                alive_on_disch += 1
                if death_duration <= duration_th and readm_duration <= duration_th:
                    readm_and_death_within_th += 1
                elif death_duration > duration_th and readm_duration <= duration_th:
                    readm_and_no_death_within_th += 1
                elif death_duration <= duration_th and readm_duration > duration_th:
                    no_readm_death_within_th += 1
                else:
                    no_readm_no_death_within_th += 1
                    if readm_duration is np.inf:
                        no_readm_after_th += 1
                    else:
                        readm_after_th += 1

    n_subject = len(l_n_admission)

    l_death_or_readm_duration = []
    for idx in range(len(l_readm_duration)):
        l_death_or_readm_duration.append(min(l_death_duration[idx], l_readm_duration[idx]))

    print "Total subject: %d" % n_subject
    print "Total admission: %d" % total_admission
    print "Mean Admission Length: %f" % np.mean(l_adm_duration)
    print "Median Admission Length: %f" % np.median(l_adm_duration)
    print "Death discharge: %d" % death_on_disch
    print "Alive discharge: %d" % alive_on_disch

    print "__Within %d days__" % duration_th
    print "Readm / Death: %d" % readm_and_death_within_th
    print "Readm / no Death: %d" % readm_and_no_death_within_th
    print "no Readm / Death: %d" % no_readm_death_within_th
    print "no Readm / no Death: %d" % no_readm_no_death_within_th

    print "__After %d days__" % duration_th
    print "Readm: %d" % readm_after_th
    print "No Readm: %d" % no_readm_after_th

    print "Histogram of #admissions per subject"
    hist, bins = np.histogram(l_adm_duration, bins=range(0, 32))
    graph.bar_histogram(hist, bins, "Number of Patients", "Admission Duration", True)

    print "Histogram of #admissions per subject"
    hist, bins = np.histogram(l_n_admission, bins=range(1, max(l_n_admission) + 1))
    graph.bar_histogram(hist, bins, "Number of Patients", "Recorded admissions per patient", True)

    print "Histogram of readmission duration"
    hist, bins = np.histogram(l_readm_duration, bins=range(1, 602, 30))
    graph.bar_histogram(hist, bins, "Number of readmissions",
                        "Duration between discharge and readmission", False)
    hist, bins = np.histogram(l_readm_duration, bins=range(1, 32, 1))
    graph.bar_histogram(hist, bins, "Number of readmissions",
                        "Duration between discharge and readmission", True)

    print "Histogram of death duration"
    hist, bins = np.histogram(l_death_duration, bins=range(1, 601, 30))
    graph.bar_histogram(hist, bins, "Number of deaths",
                        "Duration between discharge and death", False)
    hist, bins = np.histogram(l_death_duration, bins=range(1, 32, 1))
    graph.bar_histogram(hist, bins, "Number of readmissions",
                        "Duration between discharge and death", True)

    print "Histogram of death or readdm duration"
    hist, bins = np.histogram(l_death_or_readm_duration, bins=range(1, 602, 30))
    graph.bar_histogram(hist, bins, "Number of deaths",
                        "Duration between discharge and death or readmission", False,
                        filename="DorR_600")
    hist, bins = np.histogram(l_death_or_readm_duration, bins=range(1, 32, 1))
    graph.bar_histogram(hist, bins, "Number of readmissions",
                        "Duration between discharge and death or readmission", True,
                        filename="DorR_30")
    
if __name__ == '__main__':
    readmission_statisic()
    waitforbuttonpress()
