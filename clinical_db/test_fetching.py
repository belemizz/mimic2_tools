# import libraly
import psycopg2

# connect to database
conn = psycopg2.connect("dbname=MIMIC2 user=mimic2")
cur = conn.cursor()

# generate table
cur.execute("SELECT * FROM mimic2v26.d_patients")

# save to csvfile
with open('out.csv', 'w') as f:
    cur.copy_to(f, 'mimic2v26.d_patients', sep=",")

f.closed

# close
cur.close()
conn.close()
