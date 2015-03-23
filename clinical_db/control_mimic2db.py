import psycopg2

class control_mimic2db:
    def __init__(self):
        self.conn = psycopg2.connect("dbname=MIMIC2 user=mimic2")
        self.cur = self.conn.cursor()
    
    def __del__(self):
        self.cur.close()
        self.conn.close()

    def matched_icustay_detail(self,savepath):
        select_seq = "SELECT * FROM mimic2v26.icustay_detail "+\
                     "WHERE matched_waveforms_num>0"
        self.__select_and_save(select_seq, savepath)

    def subject_matched_waveforms(self,savepath):
        select_seq = "SELECT subject_id "+\
                     "FROM mimic2v26.icustay_detail "+\
                     "WHERE matched_waveforms_num>0 "+\
                     "GROUP BY subject_id " +\
                     "ORDER BY subject_id "
        self.__select_and_save(select_seq, savepath)

    def icd9_incl(self,code,savepath):
        select_seq = "SELECT * FROM mimic2v26.icd9 "+\
                     "WHERE code='%s' "%code +\
                     "ORDER BY subject_id, hadm_id"
        self.__select_and_save(select_seq, savepath)

    def icd9_of_subject(self,subject_id,savepath):
        select_seq = "SELECT * FROM mimic2v26.icd9 "+\
                     "WHERE subject_id=%d "%subject_id +\
                     "ORDER BY hadm_id, sequence"
        self.__select_and_save(select_seq, savepath)

    def icd9_eq_higher_than(self, rank, savepath):
        select_seq = "SELECT * FROM mimic2v26.icd9 "+\
                     "WHERE sequence<=%d"%rank +\
                     "ORDER BY subject_id, hadm_id, sequence"
        self.__select_and_save(select_seq, savepath)


    def count_entry_of(self,table,savepath):
        select_seq = "SELECT count(*)"+\
                     "FROM mimic2v26.%s "%table
        self.__select_and_save(select_seq, savepath)

    def count_icd9(self, savepath):
        select_seq = "SELECT code, count(code) "+\
                     "FROM mimic2v26.icd9 "+\
                     "GROUP BY code " +\
                     "ORDER BY count(code) DESC "
        self.__select_and_save(select_seq, savepath)

    def count_icd9_eq_higher_than(self, rank, savepath):
        select_seq = "SELECT code, count(code) "+\
                     "FROM mimic2v26.icd9 "+\
                     "WHERE sequence<=%d"%rank +\
                     "GROUP BY code " +\
                     "ORDER BY count(code) DESC "
        self.__select_and_save(select_seq, savepath)

    def count_sequence_of_icd9_eq(self, code, savepath):
        select_seq = "SELECT sequence, count(sequence) "+\
                     "FROM mimic2v26.icd9 "+\
                     "WHERE code='%s' "%code +\
                     "GROUP BY sequence " +\
                     "ORDER BY sequence ASC "
        self.__select_and_save(select_seq, savepath)

    def __select_and_save(self, select_seq, filepath):
        print "exec:"
        print select_seq
        
        sql_seq = "CREATE TABLE mimic2v26.test AS (%s)"%select_seq
        self.cur.execute(sql_seq)
        
        with open(filepath, 'w') as f:
            self.cur.copy_to(f, "mimic2v26.test", sep=",")
        f.closed

        self.cur.execute("DROP TABLE mimic2v26.test")
