
import numpy
import generate_sample
import matplotlib.pyplot as plt
from sklearn import svm

import control_graph
graph= control_graph.control_graph()

def demo(x, y, x_label = "", y_label = "", filename = ""):

    clf = svm.SVC(kernel = 'linear')
    clf.fit(x, y)

    clf10 = svm.SVC(kernel = 'linear')
    clf10.fit(10.0 * x, y)

    # mesh
    x_range = x[:,0].max() - x[:,0].min()
    y_range = x[:,1].max() - x[:,1].min()

    margin_ratio = 0.1
    x_min, x_max = x[:,0].min() - margin_ratio * x_range , x[:,0].max() + margin_ratio * x_range
    y_min, y_max = x[:,1].min() - margin_ratio * y_range , x[:,1].max() + margin_ratio * y_range

    grid_num = 20.0
    h_x = x_range/grid_num
    h_y = y_range/grid_num

    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h_x),
                            numpy.arange(y_min, y_max, h_y))

    z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    graph.plot_classification_with_contour(x, y, xx, yy, z, x_label, y_label, filename)


def main():
    [x,y] = generate_sample.normal_dist(2,100,100,[2,8], seed = 1)
    demo(x, y)
    plt.waitforbuttonpress()

if __name__ == '__main__':
    main()

## print(__doc__)

## import numpy as np
## import matplotlib.pyplot as plt
## from sklearn import svm, datasets

## # import some data to play with
## iris = datasets.load_iris()
## X = iris.data[:, :2]  # we only take the first two features. We could
##                       # avoid this ugly slicing by using a two-dim dataset
## y = iris.target

## h = .02  # step size in the mesh

## # we create an instance of SVM and fit out data. We do not scale our
## # data since we want to plot the support vectors
## C = 1.0  # SVM regularization parameter
## svc = svm.SVC(kernel='linear', C=C).fit(X, y)
## rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
## poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
## lin_svc = svm.LinearSVC(C=C).fit(X, y)

## # create a mesh to plot in
## x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
## y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
## xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                      np.arange(y_min, y_max, h))

## # title for the plots
## titles = ['SVC with linear kernel',
##           'LinearSVC (linear kernel)',
##           'SVC with RBF kernel',
##           'SVC with polynomial (degree 3) kernel']


## for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
##     # Plot the decision boundary. For that, we will assign a color to each
##     # point in the mesh [x_min, m_max]x[y_min, y_max].
##     plt.subplot(2, 2, i + 1)
##     plt.subplots_adjust(wspace=0.4, hspace=0.4)

##     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

##     # Put the result into a color plot
##     Z = Z.reshape(xx.shape)
##     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

##     # Plot also the training points
##     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
##     plt.xlabel('Sepal length')
##     plt.ylabel('Sepal width')
##     plt.xlim(xx.min(), xx.max())
##     plt.ylim(yy.min(), yy.max())
##     plt.xticks(())
##     plt.yticks(())
##     plt.title(titles[i])

## plt.show()
