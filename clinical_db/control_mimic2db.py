import psycopg2

class control_mimic2db:
    def __init__(self):
        self.conn = psycopg2.connect("dbname=MIMIC2 user=mimic2")
        self.cur = self.conn.cursor()

    def execute(self):
        print 'hello'
        self.cur.execute("CREATE TABLE mimic2v26.test AS (SELECT * FROM mimic2v26.icustay_detail WHERE matched_waveforms_num>0)")
        with open('../data/icu_admission_details_test.csv', 'w') as f:
            self.cur.copy_to(f, 'mimic2v26.test', sep=",")
        f.closed
        print 'hello'
    
    def __del__(self):
        self.cur.close()
        self.conn.close()
