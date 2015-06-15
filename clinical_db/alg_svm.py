import numpy
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import datasets
from sklearn import cross_validation

import generate_sample
import control_graph

graph= control_graph.control_graph()

def demo(x, y, x_label = "", y_label = "", filename = ""):

    if x.shape[1] is not 2:
        raise ValueError("Feature Dimension is not 2")
        
    clf = svm.SVC(kernel = 'linear')
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

    graph.plot_classification_with_contour(x, y, xx, yy, z, x_label, y_label, filename)
    plt.waitforbuttonpress()


def get_sample(source_num = 0):
    if source_num is 0:
        [x,y] = generate_sample.normal_dist(2,100,100,[2,8], seed = 1)
    elif source_num is 1:
        iris = datasets.load_iris()
        [x,y] = [iris.data, iris.target]
    elif source_num is 2:
        iris = datasets.load_iris()
        [x,y] = [iris.data[:,0:2], iris.target]
    else:
        raise ValueError
    return [x,y]

def cross_validate(x, y, cross_validation_num = 5):
    clf = svm.SVC(kernel = 'linear')
    scores = cross_validation.cross_val_score(clf, x, y, cv = cross_validation_num)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
if __name__ == '__main__':
    source_num = 0
    [x,y]= get_sample(source_num)

    cross_validation_num = 5
    cross_validate(x, y, cross_validation_num)

    try:
        demo(x,y)
    except ValueError:
        print "Error"
            
