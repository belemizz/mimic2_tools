import csv

class control_csv:
    def __init__(self, path):
        self.filepath = path

    def read_first_column(self):
        result = []
        reader = self.__open_reader()
        for row in reader:
            result.append(row[0])
        self.__close()
        return result

    def read_single_list(self):
        reader = self.__open_reader()
        list_object = reader.next()
        self.__close()
        return list_object

    def write_single_list(self, list_object):
        writer = self.__open_writer()
        writer.writerow(list_object)
        self.__close()
        print list_object

    def write_list(self, list_object):
        writer = self.__open_writer()
        writer.writerows(list_object)
        self.__close()
        print list_object

    def __open_reader(self):
        self.__f = open(self.filepath, 'r')
        return csv.reader(self.__f)

    def __open_writer(self):
        self.__f = open(self.filepath, 'w')
        return csv.writer(self.__f)

    def __close(self):
        self.__f.close()
