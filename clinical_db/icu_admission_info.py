# import library
import psycopg2

# get ids included in matched records
import get_matched_records
id_list = get_matched_records.numerics_id()

# connect to database
conn = psycopg2.connect("dbname=MIMIC2 user=mimic2")
cur = conn.cursor()


current_id = id_list[4]

# generate table
#cur.execute("CREATE TABLE mimic2v26.test AS (SELECT * FROM mimic2v26.icustay_detail WHERE subject_id=%s)", (current_id,))
cur.execute("CREATE TABLE mimic2v26.test AS (SELECT * FROM mimic2v26.icustay_detail WHERE matched_waveforms_num>0)")

# save the table to csvfile
with open('../data/icu_admission_details.csv', 'w') as f:
    cur.copy_to(f, 'mimic2v26.test', sep=",")
f.closed

# delete the tablep
cur.execute("DROP TABLE mimic2v26.test")

# close
cur.close()
conn.close()
