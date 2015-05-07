import psycopg2
import subject
import admission
import icustay

class control_mimic2db:
    def __init__(self):
        self.conn = psycopg2.connect("dbname=MIMIC2 user=kimimizobe")
        self.cur = self.conn.cursor()
    
    def __del__(self):
        self.cur.close()
        self.conn.close()

    def patient_class(self, subject_id):
        patient = self.patient(subject_id)
        subject_ins = subject.subject(subject_id, patient[0][1], patient[0][2], patient[0][3], patient[0][4])
        subject_ins.set_admissions(self.admission_class(subject_id))
        return subject_ins

    def admission_class(self, subject_id):
        admissions = self.admission(subject_id)

        admission_list = []
        for item in admissions:
            admission_ins = admission.admission(item[0], item[2], item[3])

            #icustay info
            icustay_list = self.icustay_class(admission_ins.hadm_id)
            admission_ins.set_icustays(icustay_list)

            #labtest
            lab_events = self.lab_events_in_admission(admission_ins.hadm_id)
            lab_event_trends = self.lab_event_trends(lab_events)
            admission_ins.set_labs(lab_event_trends)

            #notes
            note_events = self.note_events_in_admission(admission_ins.hadm_id)
            admission_ins.set_notes(note_events)

            #append to admission_ins
            admission_list.append(admission_ins)

        return admission_list

    def icustay_class(self, hadm_id):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DETAIL "+\
                     "WHERE hadm_id =%d "%(hadm_id)
        icustays = self.__select_and_save(select_seq)

        icustay_list = []
        for item in icustays:
            icustay_ins = icustay.icustay(item[0],item[21],item[22])

            # medication
            med_events = self.med_events_in_icustay(icustay_ins.icustay_id)
            med_trends = self.med_event_trends(med_events)
            icustay_ins.set_medications(med_trends)
            
            icustay_list.append(icustay_ins)
            
        return icustay_list

    def med_event_trends(self, med_events):
        itemid_list = set([item[2] for item in med_events])
        trends = []
        for itemid in itemid_list:
            record = [item for item in med_events if item[2] == itemid and item[9]!=None]

            descripition = record[0][16]
            doseuom = record[0][10]
            charttime = [item[3] for item in record]
            realtime = [item[5] for item in record]
            dose = [item[9] for item in record]

            trends.append([itemid, descripition, doseuom, charttime, realtime, dose])
        return trends

    def lab_events_in_admission(self, hadm_id):
        select_seq = "SELECT L.*, T.TEST_NAME, T.FLUID, T.CATEGORY, T.LOINC_CODE, T.LOINC_DESCRIPTION "+\
                     "FROM mimic2v26.LABEVENTS L, mimic2v26.D_LABITEMS T "+\
                     "WHERE hadm_id =%d "%(hadm_id)+\
                     "AND L.ITEMID = T.ITEMID "+\
                     "ORDER BY ITEMID, CHARTTIME"
        return self.__select_and_save(select_seq)

    def lab_event_trends(self,lab_events_list):
        itemid_list = set([item[3] for item in lab_events_list])
        trends = []
        for itemid in itemid_list:
            record = [item for item in lab_events_list if item[3] == itemid]

            description = record[0][13]
            unit = record[0][8]
            timestamp = [item[4] for item in record]
            values = [item[5] for item in record]
            
            trends.append([itemid, description, unit, timestamp, values])
        return trends

    def patient(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_patient.csv"%patient_id

        select_seq = "SELECT D.* FROM mimic2v26.D_PATIENTS D "+\
                     "WHERE subject_id =%d"%(patient_id);
        return self.__select_and_save(select_seq, savepath)

    
    def admission(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_admission.csv"%patient_id

        select_seq = "SELECT * FROM mimic2v26.ADMISSIONS "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icd9(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_icd9.csv"%patient_id

        select_seq = "SELECT I.* FROM mimic2v26.ICD9 I "+\
                     "WHERE subject_id =%d "%(patient_id) +\
                     "ORDER BY hadm_id, sequence"
        return self.__select_and_save(select_seq, savepath)


    def icustay_events(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_icustay_events.csv"%patient_id

        select_seq = "SELECT * FROM mimic2v26.ICUSTAYEVENTS "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icustay_days(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_icustay_days.csv"%patient_id

        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DAYS "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icustay_detail(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_icustay_detail.csv"%patient_id

        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DETAIL "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def med_events(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_med_events.csv"%patient_id

        select_seq = "SELECT M.*, T.LABEL "+\
                     "FROM mimic2v26.MEDEVENTS M, mimic2v26.D_MEDITEMS T "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "AND M.ITEMID = T.ITEMID ORDER BY ITEMID, REALTIME";
        return self.__select_and_save(select_seq, savepath)

    def med_events_in_icustay(self, icustay_id):
        select_seq = "SELECT M.*, T.LABEL "+\
                     "FROM mimic2v26.MEDEVENTS M, mimic2v26.D_MEDITEMS T "+\
                     "WHERE icustay_id =%d "%(icustay_id)+\
                     "AND M.ITEMID = T.ITEMID ORDER BY ITEMID, REALTIME";
        return self.__select_and_save(select_seq)
        
    def note_events(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_note_events.csv"%patient_id

        select_seq = "SELECT N.* FROM mimic2v26.NOTEEVENTS N "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "ORDER BY CHARTTIME";
        return self.__select_and_save(select_seq, savepath)

    def note_events_in_admission(self, hadm_id):
        select_seq = "SELECT N.* FROM mimic2v26.NOTEEVENTS N "+\
                     "WHERE hadm_id =%d "%(hadm_id)+\
                     "ORDER BY CHARTTIME";
        return self.__select_and_save(select_seq)

    def poe_order(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_poe_order.csv"%patient_id

        select_seq = "SELECT P.* FROM mimic2v26.POE_ORDER P "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "ORDER BY START_DT";
        return self.__select_and_save(select_seq, savepath)


    def lab_events(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_lab_events.csv"%patient_id

        select_seq = "SELECT L.*, T.TEST_NAME, T.FLUID, T.CATEGORY, T.LOINC_CODE, T.LOINC_DESCRIPTION "+\
                     "FROM mimic2v26.LABEVENTS L, mimic2v26.D_LABITEMS T "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "AND L.ITEMID = T.ITEMID "+\
                     "ORDER BY ITEMID, CHARTTIME"
        return self.__select_and_save(select_seq, savepath)

    def io_events(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_io_events.csv"%patient_id

        select_seq = "SELECT I.*, T.LABEL, T.CATEGORY "+\
                     "FROM mimic2v26.IOEVENTS I, mimic2v26.D_IOITEMS T "+\
                     "WHERE subject_id =%d AND I.ITEMID = T.ITEMID "%patient_id+\
                     "ORDER BY REALTIME"
        return self.__select_and_save(select_seq, savepath)

    def microbiology_events(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_microbiology_events.csv"%patient_id

        select_seq = "SELECT M.*, "+\
          "C.TYPE AS STYPE, C.LABEL AS SLABEL, C.DESCRIPTION AS SDESC, "+\
          "D.TYPE AS OTYPE, D.LABEL AS OLABEL, D.DESCRIPTION AS ODESC, "+\
          "E.TYPE AS ATYPE, E.LABEL AS ALABEL, E.DESCRIPTION AS ADESC "+\
          "FROM mimic2v26.MICROBIOLOGYEVENTS M, mimic2v26.D_CODEDITEMS C, mimic2v26.D_CODEDITEMS D, mimic2v26.D_CODEDITEMS E "+\
          "WHERE subject_id =%d "%(patient_id)+\
          "AND M.SPEC_ITEMID = C.ITEMID AND M.ORG_ITEMID = D.ITEMID AND M.AB_ITEMID = E.ITEMID "+\
          "ORDER BY CHARTTIME";

        return self.__select_and_save(select_seq, savepath)

    def matched_icustay_detail(self,savepath = ""):
        select_seq = "SELECT * FROM mimic2v26.icustay_detail "+\
                     "WHERE matched_waveforms_num>0"
        return self.__select_and_save(select_seq, savepath)

    def subject_matched_waveforms(self,savepath = ""):
        select_seq = "SELECT subject_id "+\
                     "FROM mimic2v26.icustay_detail "+\
                     "WHERE matched_waveforms_num>0 "+\
                     "GROUP BY subject_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq, savepath)

    def subject_with_icu_expire_flg(self, savepath = ""):
        select_seq = "SELECT subject_id "+\
                     "FROM mimic2v26.icustay_detail "+\
                     "WHERE icustay_expire_flg='Y' "+\
                     "GROUP BY subject_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq, savepath)

    def subject_with_icd9(self, code, seq_cond,savepath = ""):
        select_seq = "SELECT subject_id "+\
                     "FROM mimic2v26.icd9 "+\
                     "WHERE code='%s' AND sequence%s"%(code,seq_cond) +\
                     "GROUP BY subject_id " +\
                     "ORDER BY subject_id "
        return self.__select_and_save(select_seq, savepath)

    def icd9_incl(self,code,savepath = ""):
        select_seq = "SELECT * FROM mimic2v26.icd9 "+\
                     "WHERE code='%s' "%code +\
                     "ORDER BY subject_id, hadm_id"
        return self.__select_and_save(select_seq, savepath)

    def icd9_eq_higher_than(self, rank, savepath = ""):
        select_seq = "SELECT * FROM mimic2v26.icd9 "+\
                     "WHERE sequence<=%d"%rank +\
                     "ORDER BY subject_id, hadm_id, sequence"
        return self.__select_and_save(select_seq, savepath)

    def count_entry_of(self,table,savepath = ""):
        select_seq = "SELECT count(*)"+\
                     "FROM mimic2v26.%s "%table
        return self.__select_and_save(select_seq, savepath)

    def count_icd9(self, savepath = ""):
        select_seq = "SELECT code, count(code) "+\
                     "FROM mimic2v26.icd9 "+\
                     "GROUP BY code " +\
                     "ORDER BY count(code) DESC "
        return self.__select_and_save(select_seq, savepath)

    def count_icd9_eq_higher_than(self, rank, savepath = ""):
        select_seq = "SELECT code, count(code) "+\
                     "FROM mimic2v26.icd9 "+\
                     "WHERE sequence<=%d"%rank +\
                     "GROUP BY code " +\
                     "ORDER BY count(code) DESC "
        return self.__select_and_save(select_seq, savepath)

    def count_sequence_of_icd9_eq(self, code, savepath = ""):
        select_seq = "SELECT sequence, count(sequence) "+\
                     "FROM mimic2v26.icd9 "+\
                     "WHERE code='%s' "%code +\
                     "GROUP BY sequence " +\
                     "ORDER BY sequence ASC "
        return self.__select_and_save(select_seq, savepath)

    def __select_and_save(self, select_seq, filepath=""):
        print "exec:"
        print select_seq

        self.cur.execute(select_seq)
        result = self.cur.fetchall()

        if len(filepath)>0:
            import csv
            writer = csv.writer(open(filepath, 'wb'))
            writer.writerows(result)

        return result
    
