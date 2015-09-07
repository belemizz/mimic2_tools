import sys
sys.path.append('../../DeepLearningTutorials/code/')

import numpy as np
import random
from sklearn import cross_validation

import imdb


class SeriesData():
    '''Series Data Handler'''
    def __init__(self, series, mask, label):
        self.series = series
        self.mask = mask
        self.label = label

    def slice_by_time(self, index):
        return SeriesData(self.series[index, :, :],
                          self.mask[index, :],
                          self.label)

    def slice_by_sample(self, index):
        return SeriesData(self.series[:, index, :],
                          self.mask[:, index],
                          self.label[index])

    def slice_by_feature(self, index):
        return SeriesData(self.series[:, :, index],
                          self.mask,
                          self.label)

    def n_step(self):
        return self.series.shape[0]

    def n_sample(self):
        return self.series.shape[1]

    def n_feature(self):
        return self.series.shape[2]

    def split_train_test(self):
        cv_iter = cross_validation.StratifiedKFold(self.label)
        train_idx, test_idx = list(cv_iter)[0]
        train = self.slice_by_sample(train_idx)
        test = self.slice_by_sample(test_idx)
        return (train, test)


class TimeSeries:

    def sample(self, source_num=0, n_dim=2):
        if source_num is 0:
            data = self.normal(n_dim=n_dim, bias=[-10, -9], length=5)
        elif source_num is 1:
            data = self.imdb_data()
        else:
            raise ValueError
        return data

    def normal(self, length=50, n_dim=2, random_length=True, seed=0,
               n_negative=150, n_positive=50, bias=[-1, +1]):
        random.seed(seed)
        np.random.seed(seed)

        def get_series(bias, flag, n_sample):
            '''get timeserieses with same flag'''
            data = []
            for num in xrange(0, n_sample):
                if random_length:
                    s_len = np.random.randint(length) + length
                else:
                    s_len = length
                sample = np.zeros([s_len, n_dim])
                for i in xrange(0, s_len):
                    sample[i] = np.random.randn(1, n_dim) + bias
                data.append([sample, flag])
            return data

        negative_data = get_series(bias[0], 0, n_negative)
        positive_data = get_series(bias[1], 1, n_positive)
        data = negative_data + positive_data

        random.shuffle(data)

        x = [item[0] for item in data]
        x, mask = self.__l_tseries_to_ar(x)
        y = np.array([item[1] for item in data])

        return SeriesData(x, mask, y)

    def imdb_data(self):
        """Load IMDB data."""
        train, valid, test = imdb.load_data(n_words=10000,
                                            valid_portion=0.05,
                                            maxlen=100)

        x = train[0] + valid[0] + test[0]
        x, mask = self.__l_tseries_to_ar(x)

        y = np.array(train[1] + valid[1] + test[1])
        return SeriesData(x, mask, y)

    def __l_tseries_to_ar(self, ts_x):
        """Convert list of timeseries to numpy arrays of value and mask."""
        max_length = max([len(s) for s in ts_x])

        if np.array(ts_x[0]).ndim == 1:
            # data is one dimentional
            dim = 1
        elif np.array(ts_x[0]).ndim == 2:
            dim = ts_x[0].shape[1]
        else:
            raise ValueError('Invalid data format')

        x = np.zeros((max_length, len(ts_x), dim))
        mask = np.zeros((max_length, len(ts_x)))

        for i_series, series in enumerate(ts_x):
            if dim == 1:
                x[:len(series), i_series] = [[val] for val in series]
            else:
                x[:len(series), i_series] = series
            mask[:len(series), i_series] = 1
        return x, mask
