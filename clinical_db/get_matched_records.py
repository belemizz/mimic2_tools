import urllib2

# access to the index
response = urllib2.urlopen('http://physionet.org/physiobank/database/mimic2wdb/matched/RECORDS-numerics')

# convert idlist to array
output = response.read()
li = output.split('\n')

while li.count("") > 0:
    li.remove("")

def pick_id(str):
    return str[0:6]

id_list = list(set(map(pick_id, li)))
id_list.sort()

print(id_list)
print(len(id_list))
