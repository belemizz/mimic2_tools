
## generate_id_list
import get_matched_records

# save the list of matched ids
id_list = get_matched_records.numerics_id()

f = open('../data/id_list.dat', 'w')
for id in id_list:
    f.write(str(id) + '\n')
f.close()

# save the list of nurerics
numerics_list = get_matched_records.numerics()

f = open('../data/numerics_list.dat', 'w')
for id in numerics_list:
    f.write(str(id) + '\n')
f.close()
