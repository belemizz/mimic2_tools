"""Compare deep learning libraries."""

import sys
sys.path.append("../clinical_db/")

from mutil import Graph, Stopwatch
graphs = Graph()
sw = Stopwatch()

from get_sample import point_data, split_to_three_sets

from sklearn import linear_model

from alg.metrics import BinaryClassResult
from alg.keras_alg import Keras_LR
from alg.chainer_alg import Chainer_LR

# from mutil import Graph
# import matplotlib.pyplot as plt

if __name__ == '__main__':

    keras_flag = True
    chainer_flag = True
    [x, y] = point_data.sample(0, 4, 2)
    n_dim = x.shape[1]
    n_flag = len(set(y))
    n_epoch = 1000
    batchsize = 10

    N = int(x.shape[0] * 0.8)
    all_data = split_to_three_sets(x, y, 0., 0.2)

    train_x = all_data[0]
    train_y = all_data[1]
    test_x = all_data[4]
    test_y = all_data[5]

    print 'Sklearn'
    clf = linear_model.LogisticRegression(random_state=0)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = BinaryClassResult(test_y, predict_y)
    print result.get_dict()

    # Chainer evaluation
    if chainer_flag:
        print 'Chainer'
        sw.reset()
        ch = Chainer_LR(n_epoch, batchsize)
        ch.fit(train_x, train_y)
        predict_y = ch.predict(test_x)
        ch_result = BinaryClassResult(test_y, predict_y)
        print ch_result.get_dict()
        sw.stop()
        sw.print_cpu_elapsed()

    # Keras evaluation
    if keras_flag:
        print 'Keras'
        sw.reset()
        ke = Keras_LR(n_epoch, batchsize)
        ke.fit(train_x, train_y)
        predict_y = ke.predict(test_x)
        ke_result = BinaryClassResult(test_y, predict_y)
        print ke_result.get_dict()
        sw.stop()
        sw.print_cpu_elapsed()
