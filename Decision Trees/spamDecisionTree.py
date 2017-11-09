import sys
import csv
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.tree import export_graphviz
from subprocess import call
import graphviz
'''
methods to parse CSV file
'''

#returns contents of CSV file as 2d list
def getContents(path):
	data = []
	labels = []
	with open(path, 'rt') as csvFile:
		reader = csv.reader(csvFile)
		for row in list(reader)[1:]:
			data.append(list(map(lambda x: float(x.strip()), row[1:-1])))
			labels.append(1 if row[-1] == "spam" else 0)
	return data,labels


def main():
	if len(sys.argv) < 2:
		return

	(data, labels) = getContents(sys.argv[1])
	data = np.array(data)
	labels = np.array(labels)
	print(data,labels)

	xTrain, xTest, labelsTrain, labelsTest = train_test_split(data, labels, test_size=0.5)


	classifier = tree.DecisionTreeClassifier(max_depth = 5)
	classifier = classifier.fit(xTrain, labelsTrain)

	print("Training score:", classifier.score(xTrain, labelsTrain))
	print("Testing score:", classifier.score(xTest, labelsTest))

	export_graphviz(classifier,out_file='tree.dot')
	#call(['dot','-Tpng','tree.dot','-o','tree.png'])

if __name__ == '__main__':
	main()
