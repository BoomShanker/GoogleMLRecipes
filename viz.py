# 1. Import Dataset.
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0,50,100]


# Feature names
#print (iris.feature_names)
# Label (flower) names
# print (iris.target_names)
# Feature data
# print (iris.data[0])
# Label data
# print (iris.target[0])
# Iterate over all entries to print out entire dataset
#for i in range(len(iris.target)):
#	print ("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))


# 2. Train a classifier.

# First we need to split up the data, between examples used to 
# test the classifier's accuracy and the training data

# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Create a decision tree classifier and train it on the training data
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# 3. Predict label for new flower.
print (test_target)
print (clf.predict(test_data))

# 4. Vizualize the tree

from sklearn.externals.six import StringIO  
# Error importing pydot?
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True,
                         impurity=False) 
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("iris.pdf") 

