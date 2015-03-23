import urllib2

def numerics():
    # set url
    url = 'http://physionet.org/physiobank/database/mimic2wdb/matched/RECORDS-numerics'
    # access to the index
    response = urllib2.urlopen(url)
    # convert idlist to array
    output = response.read()
    li = output.split('\n')
    while li.count("") > 0:
        li.remove("")
    return li
    
def numerics_id():
    li = numerics()
    id_list = list(set(map(pick_id, li)))
    id_list.sort()
    return id_list

def pick_id(str):
    return int(str[1:6])
