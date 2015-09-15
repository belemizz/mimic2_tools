"""
Script to show summery of  medical record of a subject.
"""
from get_sample import Mimic2, PatientData
from mutil import Graph, Csv, intersection
from get_sample import SeriesData

import numpy as np

mimic2db = Mimic2()
graph = Graph()


def visualize_data(subj_b_id):
    patients = PatientData(subj_b_id)
    l_lab_id = patients.common_lab(2)[0]
    l_chart_id = mimic2db.vital_charts

    from_discharge = True
    poi = 0.
    duration = 3.
    cycle = 0.1

    if from_discharge:
        x_label = 'Days from Discharge'
        span = [-poi - duration, -poi]
    else:
        x_label = 'Days from Admission'
        span = [poi, poi + duration]

    # Get All Data as a baseline
    [lab_b_ts, lab_b_data, ch_b_ts, ch_b_data, b_flag, subj_b_id, hadm_b_id] \
        = patients.data_from_adm(l_lab_id, l_chart_id, from_discharge=from_discharge)

    def single_sample(poi, from_discharge):
        a_lab, a_chart, _, _, _, _, hadm_p_id \
            = patients.point_from_adm(l_lab_id, l_chart_id, span[1], from_discharge)
        lab_point_ts = [span] * len(l_lab_id)
        chart_point_ts = [span] * len(l_chart_id)
        lab_p_data = np.dstack((a_lab, a_lab))
        chart_p_data = np.dstack((a_chart, a_chart))
        return lab_point_ts, lab_p_data, chart_point_ts, chart_p_data, hadm_p_id
    [lab_p_ts, lab_p_data, ch_p_ts, ch_p_data, hadm_p_id] \
        = single_sample(poi + duration, from_discharge)

    def sampled_series(poi, cycle, duration, from_discharge):
        data = patients.tseries_from_adm(l_lab_id, l_chart_id, span, cycle,
                                         from_discharge=from_discharge)
        lab_sample = SeriesData(data[0][0], data[0][1], data[2])
        chart_sample = SeriesData(data[1][0], data[1][1], data[2])
        l_hadm_sample = data[6]

        x = [span[0] + cycle * i for i in range(int(duration / cycle))]
        lab_sample_ts = [x] * len(l_lab_id)
        chart_sample_ts = [x] * len(l_chart_id)
        return lab_sample_ts, lab_sample, chart_sample_ts, chart_sample, l_hadm_sample

    [lab_s_ts, lab_s_data, ch_s_ts, ch_s_data, hadm_s_id] \
        = sampled_series(poi, cycle, duration, from_discharge)

    def coef(poi, duration, from_discharge):
        a_lab, a_chart, expire_flag, l_subject_id, readm_duration, death_duration, l_hadm_id_coef \
            = patients.trend_from_adm(l_lab_id, l_chart_id, span, from_discharge)

        lab_ts_coef = [span] * len(l_lab_id)
        chart_ts_coef = [span] * len(l_chart_id)

        def __coef(array):
            coef = np.zeros((array.shape[0], array.shape[1] / 2, 2))
            coef[:, :, 0] = array[:, range(0, array.shape[1], 2)] \
                + span[0] * array[:, range(1, array.shape[1], 2)]

            coef[:, :, 1] = array[:, range(0, array.shape[1], 2)] \
                + span[1] * array[:, range(1, array.shape[1], 2)]
            return coef

        lab_coef = __coef(a_lab)
        chart_coef = __coef(a_chart)
        return lab_ts_coef, lab_coef, chart_ts_coef, chart_coef, l_hadm_id_coef

    lab_c_ts, lab_c_data, ch_c_ts, ch_c_data, hadm_c_id \
        = coef(poi, duration, from_discharge)

    valid_hadm = intersection((hadm_c_id, hadm_b_id, hadm_p_id, hadm_s_id))
    for id in valid_hadm:
        i_b = hadm_b_id.index(id)
        i_c = hadm_c_id.index(id)
        i_p = hadm_p_id.index(id)
        i_s = hadm_s_id.index(id)

        def __sample_graph(i_b, i_c, b_ts, c_ts, b_data, c_data):
            ts = b_ts[i_b] + c_ts
            data = b_data[i_b] + c_data.slice_by_sample(i_c).series.transpose().tolist()
            graph.line_scatter(ts, data, hl_span=span)

        def __coef_graph(i_b, i_c, b_ts, c_ts, b_data, c_data):
            ts = b_ts[i_b] + c_ts
            data = b_data[i_b] + c_data[i_c].tolist()
            graph.line_scatter(ts, data, hl_span=span)

        __coef_graph(i_b, i_p, lab_b_ts, lab_p_ts, lab_b_data, lab_p_data)
        __coef_graph(i_b, i_p, ch_b_ts, ch_p_ts, ch_b_data, ch_p_data)

        __sample_graph(i_b, i_s, lab_b_ts, lab_s_ts, lab_b_data, lab_s_data)
        __sample_graph(i_b, i_s, ch_b_ts, ch_s_ts, ch_b_data, ch_s_data)

        __coef_graph(i_b, i_c, lab_b_ts, lab_c_ts, lab_b_data, lab_c_data)
        __coef_graph(i_b, i_c, ch_b_ts, ch_c_ts, ch_b_data, ch_c_data)
        graph.waitforbuttonpress()
        graph.close_all()


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
    id_list = mimic2db.subject_with_chf(0)
    idx = 6
    visualize_data(id_list[idx: idx + 3])
#    show_records(subject_id)
