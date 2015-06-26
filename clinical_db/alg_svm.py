import numpy
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import cross_validation

import generate_sample
import control_graph

graph= control_graph.control_graph()

def demo(x, y, x_label = "", y_label = "", filename = "", show_flag = True):

    if x.shape[1] is not 2:
        raise ValueError("Can't show: x dimension is not 2")

    clf = svm.LinearSVC(max_iter = 200000, random_state = 0)
    clf.fit(x, y)
    
    # mesh
    x_range = x[:,0].max() - x[:,0].min()
    y_range = x[:,1].max() - x[:,1].min()

    margin_ratio = 0.1
    x_min, x_max = x[:,0].min() - margin_ratio * x_range , x[:,0].max() + margin_ratio * x_range
    y_min, y_max = x[:,1].min() - margin_ratio * y_range , x[:,1].max() + margin_ratio * y_range

    grid_num = 200.0
    h_x = x_range/grid_num
    h_y = y_range/grid_num

    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h_x),
                            numpy.arange(y_min, y_max, h_y))

    z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    graph.plot_classification_with_contour(x, y, xx, yy, z, x_label, y_label, filename, show_flag = show_flag)

    return clf

def cross_validate(x, y, n_cv_fold = 10):
    #    clf = svm.SVC(kernel = 'linear', iter_)
    clf = svm.LinearSVC(max_iter = 200000, random_state = 0)
    scores = cross_validation.cross_val_score(clf, x, y, cv = n_cv_fold)
    predicted = cross_validation.cross_val_predict(clf, x, y, cv = n_cv_fold)

    recall = float(sum(predicted[y == 1])) / sum(y)
    precision = float(sum(predicted[y == 1])) / sum(predicted)
    f_measure = 2 * precision * recall / (precision + recall)

    print "----------cross_varidation_result_____________"
    for idx, score in enumerate(scores):
        print "E%d: %f"%(idx, score)
    print("Accuracy:  %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print "Recall:    %0.4f"%recall
    print "Precision: %0.4f"%precision
    print "F-measure: %0.4f"%f_measure
    
    print "______________________________________________"

        
if __name__ == '__main__':
    source_num = 2
    n_dim = 500
    n_flag = 2

    [x,y]= generate_sample.get_samples_with_target(source_num, n_dim, n_flag)

    cross_validation_num = 2
    cross_validate(x, y, cross_validation_num)
    
    try:
        demo(x,y)
        plt.waitforbuttonpress()
    except ValueError, detail:
        print detail
