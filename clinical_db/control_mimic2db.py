import psycopg2

class control_mimic2db:
    def __init__(self):
        self.conn = psycopg2.connect("dbname=MIMIC2 user=kimimizobe")
        self.cur = self.conn.cursor()
    
    def __del__(self):
        self.cur.close()
        self.conn.close()

    def patient(self, patient_id, savepath = ""):
        select_seq = "SELECT D.* FROM mimic2v26.D_PATIENTS D "+\
                     "WHERE subject_id =%d"%(patient_id);
        return self.__select_and_save(select_seq, savepath)

    def icd9(self, patient_id, savepath = ""):
        select_seq = "SELECT I.* FROM mimic2v26.ICD9 I "+\
                     "WHERE subject_id =%d "%(patient_id) +\
                     "ORDER BY hadm_id, sequence"
        return self.__select_and_save(select_seq, savepath)

    def admission(self, patient_id, savepath = ""):
        select_seq = "SELECT * FROM mimic2v26.ADMISSIONS "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icustay_events(self, patient_id, savepath = ""):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAYEVENTS "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icustay_days(self, patient_id, savepath = ""):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DAYS "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def icustay_detail(self, patient_id, savepath = ""):
        select_seq = "SELECT * FROM mimic2v26.ICUSTAY_DETAIL "+\
                     "WHERE subject_id =%d "%(patient_id)
        return self.__select_and_save(select_seq, savepath)

    def med_events(self, patient_id, savepath = ""):
        select_seq = "SELECT M.*, T.LABEL "+\
                     "FROM mimic2v26.MEDEVENTS M, mimic2v26.D_MEDITEMS T "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "AND M.ITEMID = T.ITEMID ORDER BY REALTIME";
        return self.__select_and_save(select_seq, savepath)

    def note_events(self, patient_id, savepath = ""):
        if len(savepath) == 0:
            savepath = "../data/%d_noteevents.csv"%patient_id
        select_seq = "SELECT N.* FROM mimic2v26.NOTEEVENTS N "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "ORDER BY CHARTTIME";
        return self.__select_and_save(select_seq, savepath)

    def poe_order(self, patient_id, savepath = ""):
        select_seq = "SELECT P.* FROM mimic2v26.POE_ORDER P "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "ORDER BY START_DT";
        return self.__select_and_save(select_seq, savepath)


    def lab_events(self, patient_id, savepath = ""):
        select_seq = "SELECT L.*, T.TEST_NAME, T.FLUID, T.CATEGORY, T.LOINC_CODE, T.LOINC_DESCRIPTION "+\
                     "FROM mimic2v26.LABEVENTS L, mimic2v26.D_LABITEMS T "+\
                     "WHERE subject_id =%d "%(patient_id)+\
                     "AND L.ITEMID = T.ITEMID "+\
                     "ORDER BY CHARTTIME"
        return self.__select_and_save(select_seq, savepath)

    def microbiology_events(self, patient_id, savepath = ""):
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

    def __select_and_save(self, select_seq, filepath):
        print "exec:"
        print select_seq

        self.cur.execute(select_seq)
        result = self.cur.fetchall()

        import csv
        writer = csv.writer(open(filepath, 'wb'))
        writer.writerows(result)

        return result
