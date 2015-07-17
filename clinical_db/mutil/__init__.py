import cPickle
import os
import time

from .mycsv import Csv
from .graph import Graph


class Cache:
    """ Cache class """

    def __init__(self, cache_key, cache_dir = '../data/cache/'):
        """ This class does nothing when cache_key is '' """
        
        self.dir = cache_dir
        self.key = cache_key
        self.param_path = self.dir + self.key + '_param.pkl'
        self.data_path = self.dir + self.key + '_data.pkl'

    def save(self, current_data, current_param = {} ):
        if self.key is not '':
            
            if len(current_param) is not 0:
                f = open(self.param_path, 'w')
                cPickle.dump(current_param, f)
                f.close()
            g = open(self.data_path, 'w')
            cPickle.dump(current_data, g)
            g.close()
            
        return current_data

    def load(self, current_param = {}):
        if self.key is not '':

            if len(current_param) is 0:
                if os.path.isfile(self.data_path):
                    g = open(self.data_path, 'r')
                    cache_data = cPickle.load(g)
                    g.close()
                    print "[INFO] Cache is used: %s"%self.data_path
                    return cache_data
            else:
                if os.path.isfile(self.param_path):
                    f = open(self.param_path, 'r')
                    cache_param = cPickle.load(f)
                    f.close()
                    if self.__is_param_eq(cache_param, current_param) and os.path.isfile(self.data_path):
                        g = open(self.data_path,'r')
                        cache_data = cPickle.load(g)
                        g.close()
                        print "[INFO] Cache is used: %s"%self.data_path
                        return cache_data
        raise IOError("Cache was not loaded")
        
    def __is_param_eq(self, param1, param2):
        if set(param1.keys()) != set(param2.keys()):
            return False
        for key in param1.keys():
            try:
                if param1[key] != param2[key]:
                    return False
            except ValueError:
                if (param1[key] == param2[key]).all() == False:
                    return False
        return True

class Stopwatch:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.base_cpu = time.clock()
        self.base_real = time.time()

    def stop(self):
        self.end_cpu = time.clock()
        self.end_real = time.time()

    def cpu_elapsed(self):
        return self.end_cpu - self.base_cpu

    def real_elapsed(self):
        return self.end_real - self.base_real

    def print_cpu_elapsed(self, stop = False):
        if stop: self.stop()
        print "CPU Elapsed:%0.4f[sec]"%self.cpu_elapsed()

    def print_real_elapsed(self, stop = False):
        if stop: self.stop()
        print "Real Elapsed:%0.4f[sec]"%self.real_elapsed()

def p_info(word):
    print '[INFO]' + word

