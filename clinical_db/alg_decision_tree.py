import numpy
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

from sklearn.externals.six import StringIO
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)
    
## import pydot
## dot_data = StringIO()
## tree.export_graphviz(clf, out_file = dot_data)
## print dot_data.getvalue()

## graph = pydot.graph_from_edges(dot_data.getvalue())
## graph.write_pdf("iris.pdf")

