from sklearn import tree
# a simplified representation of features:
# first index in list is weight in grams
# second index in list is 0 for Smooth, 1 for Bumpy textures
features = [[140,1], [130,1], [150,0], [170,0]]
# 0 for Apple, 1 for Orange
labels = [0, 0, 1, 1]
# Create a decision tree classifier and train it on the training data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
# Make a prediciton of a Bumpy object weighing 160 grams
print (clf.predict([[160,0]]))