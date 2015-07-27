from urllib2 import urlopen

class Mimic2m:
    ''' Mimic 2 waveform databased matched subset '''
    def __init__(self):
        self.url = 'http://physionet.org/physiobank/database/mimic2wdb/matched/RECORDS-numerics'

    def get_numerics(self):
        ''' get the list of the paths of numeric records '''
        response = urlopen(self.url)
        raw_list = response.read()

        numerics = raw_list.split('\n')
        while numerics.count("") > 0:
            numerics.remove("")
        return numerics

    def get_id_numerics(self, max_id=None):
        ''' get the ids who have numeric records '''
        numerics = self.get_numerics()
        id_list = list(set(map(self.__pick_id_in_numerics, numerics)))

        if max_id:
            id_list = [item for item in id_list if item < max_id]

        id_list.sort()
        return id_list

    def __pick_id_in_numerics(self, numeric):
        return int(numeric[1:6])
