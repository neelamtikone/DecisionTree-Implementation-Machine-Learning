import pandas as pd
import numpy as np
import csv
import math
from collections import defaultdict
from collections import Counter
import operator
import sys

''' This nodes of the decision tree belong to the class TreeNode '''

class TreeNode(object):
	def __init__(self, attribute, parentValue, nodeNo = 0, depth = 0):
		self.nodeId = nodeNo
		self.nodeName = attribute
		self.parentNode = parentValue
		self.childrenDict = {}
		self.depth = depth
		self.valueDict = {}
		self.entropy = None
		self.infoGain = None	
		self.indices = []
		self.nodeData = {}	
		self.classData = []
		self.childName = {}
		self.classValue = None
		self.parent = None


class DecisionTree(object):


	''' This is the constructor for the decision tree. It initialises the path for the train and test data '''
	def __init__(self, maxDepth, trainPath, testPath= None, className = None):
		self.dictDF = self.readFile(trainPath)
		self.classVar = self.dictDF[className].values()
		del self.dictDF['class']
		self.maxDepth = maxDepth		
		self.nodeQueue = []
		self.nodeStack = []
		self.trainDF = pd.DataFrame.from_dict(self.dictDF)
		self.testDict = self.readFile(testPath)	
		self.testData = pd.DataFrame.from_dict(self.testDict)
		self.rootNode = None
		self.className = className
	
	''' get the data with specifoed indices'''
	def getData(self, dataDict, indices):
		newDict = {}
		for attribute in dataDict.keys():
			data = dataDict[attribute].values()
			data = [data[position] for position in indices]
			newDict[attribute] = dict(enumerate(data))
		return (newDict)	

	'''this method is used to create the root Node, which then calls the function to create the rest of the nodes of the tree and the Prediction function  '''

	def createTree(self, depth = 0):
		if (depth == 0):
			(root, ent) = self.attributeSelection(self.dictDF , self.classVar)
			rootNode = TreeNode(root , None, 0, 1)
			rootNode.entropy = ent
			classes = set(self.dictDF[root].values())
			rootNode.nodeData = self.dictDF
			rootNode.classData = self.classVar
			rootNode.classValue = 0
			self.nodeQueue.append(rootNode)
			self.nodeStack.append(rootNode)
			self.rootNode = rootNode
			self.rootNode.depth = 0
		self.createNodes(rootNode)
		print "Prediction on Test Dataset:\n\n"
		self.PredictOnTest(self.testData,self.rootNode,self.testData['class'])
		print "\nPrediction on Train Dataset:"
		self.PredictOnTest(self.trainDF,self.rootNode,self.classVar)
	
	''' This function is used to create the rest of the nodes of the tree until the depth specified by the user '''

	def createNodes(self, rootNode):
                if (not self.nodeStack):
                        return rootNode
                classes = set(self.dictDF[rootNode.nodeName].values())
                for value in classes:
                                indices = [i for i, x in enumerate(rootNode.nodeData[rootNode.nodeName].values()) if x == value]
                                data = self.getData(rootNode.nodeData, indices)
                                classData = [rootNode.classData[position] for position in indices]
                                rootNode.valueDict[value] = indices
                                if (rootNode.depth < self.maxDepth and classData):
					childAttribute, ent = self.attributeSelection(data, classData, rootNode.nodeName)
					gain = rootNode.entropy - ent
                                        node = TreeNode(childAttribute, rootNode.nodeName, 0, 1)
                                        node.indices = indices
                                        node.nodeData = data
                                        rootNode.childrenDict[value] = node
                                        node.depth = rootNode.depth + 1
                                        node.classData = classData
                                        node.parentNode = rootNode
					node.childName[value] = childAttribute
                                        node.infoGain = gain
                                        node.entropy = ent
					if not classData:
                                        	node.classValue = rootNode.classValue
                                	else:
                                        	node.classValue = self.mostCommonItem(node.classData)
						self.nodeStack.append(node)
                return(self.createNodes(self.nodeStack.pop()))
	

	'''This function is used for predition and calls another function to print the confusion matrix for the prediction  '''

	def PredictOnTest(self,data, rootNode, classList):
		data['pred'] = -1
		for i in xrange(0,data.shape[0]):
			pv = self.Predict(rootNode, i, data)
			data['pred'][i] = pv
		self.confusionMatrix(classList,data['pred'])

	'''This function predicts using recursion and goes to the depth until it finds the nodes do not have any further children  '''

        def Predict(self,rootNode, row, data):
                if(not rootNode.childrenDict):
                        return rootNode.classValue
                childDict = rootNode.childrenDict
                value = data[rootNode.nodeName][row]
		if value in rootNode.childrenDict.keys():
	                attribute = rootNode.childrenDict[value]
        	        return(self.Predict(attribute, row, data))
		else:
			return rootNode.classValue

	''' This function is used to print the confusion matrix  '''
	def confusionMatrix(self, t, obs):
		true_pos = 0
		true_neg = 0
		false_neg = 0
		false_pos = 0
		for i in xrange(0,len(obs)):
			if (obs[i] ==1):
				if(t[i] ==1):true_pos +=1
				else:false_pos +=1
			else:
				if(t[i] == 0):true_neg +=1
				else:false_neg +=1
		Accuracy = ((true_pos + true_neg)*1.0)/(len(obs))
		error = 1 - Accuracy
		print "Predicted\t|  0\t1"
                print "-----------------------------"
                print "Actual\t\t|"
                print "0\t\t|" , true_neg,"\t", false_pos
                print "1\t\t|" , false_neg,"\t", true_pos
		print "Accuracy: ",Accuracy
		print "Error: ", error
		print "misclassifications:" , false_pos+false_neg


	'''This function is used to print the most common element. It is used to get the class value for each node. It returns the value which has maximum occurances  '''
	def mostCommonItem(self, numList):
		d = defaultdict(int)
		for i in numList:
			d[i] += 1
		result = max(d.iteritems(), key=lambda x: x[1])[0]
		return(result)

	''' This function is used ot read the file and return the dictionary '''
	def readFile(self, pathCSV):
		df = pd.read_csv(pathCSV)
		d = df.to_dict()
		return (d)


	''' This function is used to calculate the entropy '''
	def calcEntropy(self, a):
		Calcsum = float(sum(a))
		entropy = 0
		for x in range(len(a)):
			prob = (a[x]*1.0)/Calcsum
			logCalc = math.log(prob,2)
			entropy+= -(prob * logCalc)
		return(entropy)

	''' This function is used to calculate the two attribute entropy  '''
	def twoAttrEntropy(self, att, classV):
		classEntropy = 0
		mergeList = zip(att, classV)
		dictDef = defaultdict(list)
		for attribute, classvar in mergeList:
			dictDef[attribute].append(classvar)
		count = Counter(dictDef)
		attrKeys = dictDef.keys()
		for x in attrKeys:
			dictX= Counter(dictDef[x])
			classValues = dictX.keys()
			classCount = dictX.values()
			totalX = sum(classCount)
			probX = totalX*1.0/len(att)
			classEntropy += probX * self.calcEntropy(dictX.values())
		return (classEntropy)

	''' This function is used to calculate the information gain at each node '''
	def infoGain(self, attribute, classVar, parent):
		gain = 0
		ent1 = self.calcEntropy(Counter(parent.classData).values())
		ent2 = self.twoAttrEntropy(attribute, classVar)
		gain = self.calcEntropy(Counter(parent.classData).values()) - self.twoAttrEntropy(attribute, classVar)
		return(gain)

	''' This function is used for attribute selection at each point using the entropy '''
	def attributeSelection(self, dataDict, classVar, parentNode = None):
		attributes = dataDict.keys()
		entropyDict = {}
		if (parentNode != None) : attributes.remove(parentNode)
		for x in attributes:
			attData = dataDict[x].values()
			entropyValue = self.twoAttrEntropy(attData, classVar)
			entropyDict[x] = entropyValue
		minAttr = min(entropyDict.iteritems(), key=operator.itemgetter(1))[0]
		entropy = min(entropyDict.iteritems(), key=operator.itemgetter(1))[1]
		return (minAttr, entropy)		

def main():
	sys.setrecursionlimit(50000)
	tree = DecisionTree(maxDepth = 2, trainPath = "train.csv",testPath="test.csv" ,className = "class")
	tree.createTree()


if __name__ == '__main__':
	main()

