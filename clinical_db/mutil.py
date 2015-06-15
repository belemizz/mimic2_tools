import cPickle
import os

class cache:

    def __init__(self, cache_key, cache_dir = '../data/cache/'):
        self.dir = cache_dir
        self.key = cache_key
        self.param_path = self.dir + self.key + '_param.pkl'
        self.data_path = self.dir + self.key + '_data.pkl'

    def save(self, current_param, current_data):
        f = open(self.param_path, 'w')
        cPickle.dump(current_param, f)
        f.close()
        g = open(self.data_path, 'w')
        cPickle.dump(current_data, g)
        g.close()
        return current_data

    def load(self, current_param):
        if os.path.isfile(self.param_path) and os.path.isfile(self.data_path):
            f = open(self.param_path, 'r')
            cache_param = cPickle.load(f)
            f.close()
            
            if self.__is_param_eq(cache_param, current_param):
                g = open(self.data_path,'r')
                cache_data = cPickle.load(g)
                g.close()
                print "[INFO] Cache are used: %s"%self.data_path
                return cache_data

        raise ValueError

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
        
def sample_func(a,b,c = 5):
    params = locals()
    cache_ = cache('sample_func')

    try:
        return cache_.load( params)
    except ValueError:
        ret_val = a + b + c
        return cache_.save(params, ret_val)
    
if __name__ == '__main__':
    print sample_func(1,2)
