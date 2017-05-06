from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
from scipy import stats
#Make sure to include
import sys
from random import random
from math import exp
import math

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        #self.label_num = int(label)
        self.label_str = str(label)
        pass

    def getValue(self):
        return self.label_str
        
    def __str__(self):
        return self.label_str
        pass

# the feature vectors will be stored in dictionaries so that they can be sparse structures
class FeatureVector:
    def __init__(self):
        self.feature_vec = [0]
        
    def add(self, index, value):
        if index > len(self.feature_vec) - 1:
            for i in xrange(index - len(self.feature_vec) + 1):
                self.feature_vec.append(0)
            self.feature_vec[index] = value
        else:
            self.feature_vec[index] = value
        
    def get(self, index):
        val = self.feature_vec[index]
        return val

    def getVector(self):
        return self.feature_vec
        

class Instance:
    def __init__(self, feature_vector, label):
        self.feature_vector = feature_vector
        self.label = label

    def getFeatures(self):
        return self.feature_vector.getVector()

    def getLabel(self):
        return self.label

    
# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

"""
TODO: you must implement additional data structures for
the three algorithms specified in the hw4 PDF

for example, if you want to define a data structure for the
DecisionTree algorithm, you could write

class DecisionTree(Predictor):
	# class code

Remember that if you subclass the Predictor base class, you must
include methods called train() and predict() in your subclasses
"""

class DecisionTree(Predictor):
    def __init__(self, ratio, prune):
        self.dataset = np.array([[]])
        self.features  = np.array([[]])
        self.labels = []
        self.labelsDict = {}
        self.labelsRevDict = {}
        self.prune = prune
        self.ratio = ratio
        self.tree = None
        self.labels = []

    def train(self, instances):

        #Make features and labels into arrays
        labelIndex = 0
        for instance in instances:
            feat_list = instance.getFeatures()
            feat_len = len(feat_list)

            label = instance.getLabel().getValue()
            
            if label not in self.labelsDict.keys():
                self.labelsDict[label] = labelIndex
                self.labelsRevDict[labelIndex] = label
                labelIndex += 1
            self.labels.append(self.labelsDict[label])
            
            if feat_len > self.features.shape[1]:
                b = np.zeros((self.features.shape[0], feat_len - self.features.shape[1]))
                self.features = np.hstack((self.features, b))
            elif feat_len < self.features.shape[1]:
                feat_list.append([0]*(self.features.shape[1] - feat_len))
            self.features = np.vstack((self.features, feat_list))
        self.features = np.delete(self.features, 0, 0)
        self.dataset = np.hstack((self.features, np.array(self.labels).reshape((len(self.labels), 1))))

        self.tree = self.selectSplit(self.dataset)
        self.split(self.tree)
        if self.prune == "True":
            self.pruneTree(self.tree)
    
    def informationGain(self, groups, values, dataset):
        entropyGroups = []
        groupLengths = []
        entropyTotal = 0.0
        for group in groups:
            entropy = 0.0
            size = float(len(group))
            groupLengths.append(size)
            for val in values:
                if size == 0:
                    continue
                p = [row[-1] for row in group].count(val)/float(size)
                if p == 0:
                    continue
                entropy += -(p*np.log2(p))
            entropyGroups.append(entropy)
        for val in values:
            size = float(len(dataset))
            if size == 0:
                continue
            p = [row[-1] for row in dataset].count(val)/float(size)
            if p == 0:
                continue
            entropyTotal += -(p*np.log2(p))
        entropyAfter = (groupLengths[0]/len(dataset))*entropyGroups[0] + (groupLengths[1]/len(dataset))*entropyGroups[1]
        gain = entropyTotal - entropyAfter
        return -gain

    def informationGainRatio(self, groups, values, dataset):
        entropyGroups = []
        groupLengths = []
        entropyTotal = 0.0
        split = 0.0
        for group in groups:
            entropy = 0.0
            size = float(len(group))
            groupLengths.append(size)
            for val in values:
                if size == 0:
                    continue
                p = [row[-1] for row in group].count(val)/float(size)
                if p == 0:
                    continue
                entropy += -(p*np.log2(p))
            entropyGroups.append(entropy)
        for val in values:
            size = float(len(dataset))
            if size == 0:
                continue
            p = [row[-1] for row in dataset].count(val)/float(size)
            if p == 0:
                continue
            entropyTotal += -(p*np.log2(p))
        
        if groupLengths[0] == 0 or groupLengths[1] == 0:
            return 0
        
        split1 = groupLengths[0]/len(dataset) * np.log2(groupLengths[0]/len(dataset))
        split2 = groupLengths[1]/len(dataset) * np.log2(groupLengths[1]/len(dataset))  
        split = split1 + split2
        entropyAfter = (groupLengths[0]/len(dataset))*entropyGroups[0] + (groupLengths[1]/len(dataset))*entropyGroups[1]
        gain = entropyTotal - entropyAfter
        return gain/split

    def entropy(self, groups, values, dataset):
        entropy = 0.0
        for val in values:
            for group in groups:
                size = len(group)
                if size == 0:
                    continue
                p = [row[-1] for row in group].count(val)/float(size)
                if p == 0:
                    continue
                entropy += -(p*np.log2(p))
        return entropy


    def testSplit(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def selectSplit(self, dataset):
        values = list(set(row[-1] for row in dataset))
        bestIndex = sys.maxint
        bestValue = sys.maxint
        bestScore = sys.maxint
        bestGroups = None
        for index in xrange(len(dataset[0]) - 1):
            for row in dataset:
                groups  = self.testSplit(index, row[index], dataset)
                if self.ratio == "True":
                    gain = self.informationGainRatio(groups, values, dataset)
                else:
                    gain = self.informationGain(groups, values, dataset)
                if gain < bestScore:
                    bestIndex = index
                    bestValue = row[index]
                    bestScore = gain
                    bestGroups = groups
        return{'index':bestIndex, 'value':bestValue, 'groups':bestGroups, 'data':dataset}
        
    def split(self, node):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.terminal(left + right)
            return
        #left node
        if len(left) <= 1:
            node['left'] = self.terminal(left)
        else:
            node['left'] = self.selectSplit(left)
            self.split(node['left'])
        #right node
        if len(right) <= 1:
            node['right'] = self.terminal(right)
        else:
            node['right'] = self.selectSplit(right)
            self.split(node['right'])

    def calculateExpectedValues(self, groups, labels, node):
        labels = list(set(labels))
        labelsExpected = {}
        for label in labels:
            p = 0.0
            for group in groups:
                p += [row[-1] for row in group].count(label)  
            labelsExpected[label] = float(p)/float(len(node))
        return labelsExpected

    def calculateDeviation(self, groups, labels, labelsExpected):
        dev = 0.0
        for group in groups:
            for label in labels:
                expected = len(group)*labelsExpected[label]
                actual = [row[-1] for row in group].count(label)
                dev += pow((actual - expected),2)/expected
        return dev
        
    def toPrune(self, node, groups, labels):
        prune = True
        labelsExp = self.calculateExpectedValues(groups, labels, node)
        dev = self.calculateDeviation(groups, labels, labelsExp)
        p = 1 - scipy.stats.chi2.cdf(dev, 1)
        if p < 0.05:
            prune = False
        return prune

    def pruneTree(self, node):

        right = node['right']
        left = node['left']
        data = node['data']
        
        if isinstance(left['left'], dict) and isinstance(left['right'], dict):
            return self.pruneTree(left)
        else:
            labels = [row[-1] for row in data]
            if self.toPrune(data, [left['data'], right['data']], labels):
                node = self.terminal(node)
        if isinstance(right['left'], dict) and isinstance(right['right'], dict):
            return self.pruneTree(right)
        else:
            labels = [row[-1] for row in data]
            if self.toPrune(data, [left['data'], right['data']], labels):
                node = self.terminal(node)

    def terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
        
        
    def predict(self, instance):
        data = instance.getFeatures()
        node = self.tree
        prediction = self.predictSplit(node, data)
        return self.labelsRevDict[int(prediction)]

    def predictSplit(self, node, data):
        if data[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predictSplit(node['left'], data)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predictSplit(node['right'], data)
            else:
                return node['right']

class NeuralNetwork(Predictor):
    def __init__(self, weights):
        self.dataset = np.array([[]])
        self.features  = np.array([[]])
        self.labels = []
        self.labelsDict = {}
        self.labelsRevDict = {}
        self.weights = weights
        self.network = None

    def train(self, instances):

        #Make features and labels into arrays
        labelIndex = 0
        for instance in instances:
            feat_list = instance.getFeatures()
            feat_len = len(feat_list)

            label = instance.getLabel().getValue()
            
            if label not in self.labelsDict.keys():
                self.labelsDict[label] = labelIndex
                self.labelsRevDict[labelIndex] = label
                labelIndex += 1
            self.labels.append(self.labelsDict[label])
            
            if feat_len > self.features.shape[1]:
                b = np.zeros((self.features.shape[0], feat_len - self.features.shape[1]))
                self.features = np.hstack((self.features, b))
            elif feat_len < self.features.shape[1]:
                feat_list.append([0]*(self.features.shape[1] - feat_len))
            self.features = np.vstack((self.features, feat_list))
        self.features = np.delete(self.features, 0, 0)
        self.dataset = np.hstack((self.features, np.array(self.labels).reshape((len(self.labels), 1))))

        n_inputs = len(self.dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in self.dataset]))

        self.network = initializeNetwork(n_inputs, n_inputs, n_outputs, self.weights)
        trainNetwork(self.network, self.dataset, 0.1, 100, n_outputs)

    def predict(self, instance):
        data = instance.getFeatures()
        outputs = forwardPropagate(self.network, data)
        prediction = outputs.index(max(outputs))
        return self.labelsRevDict[int(prediction)]

def initializeNetwork(n_inputs, n_hidden, n_outputs, weights):
	network = list()
        if weights == "shallow":
            hidden_layer = [{'weights':[((random()*(2.0/np.sqrt(n_inputs)))-(1.0/np.sqrt(n_inputs))) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	    network.append(hidden_layer)
	    output_layer = [{'weights':[((random()*(2.0/np.sqrt(n_hidden)))-(1.0/np.sqrt(n_hidden))) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	    network.append(output_layer)
        elif weights == "deep":
            hidden_layer = [{'weights':[((random()*((2.0*np.sqrt(6.0))/np.sqrt(n_hidden + n_inputs)))-(np.sqrt(6.0)/np.sqrt(n_hidden + n_inputs))) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	    network.append(hidden_layer)
	    output_layer = [{'weights':[((random()*((2.0*np.sqrt(6.0))/np.sqrt(n_hidden + n_outputs)))-(np.sqrt(6.0)/np.sqrt(n_hidden + n_outputs))) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	    network.append(output_layer)
        else:
            hidden_layer = [{'weights':[((random()*0.02)-0.01) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	    network.append(hidden_layer)
	    output_layer = [{'weights':[((random()*0.02)-0.01) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	    network.append(output_layer)
	return network

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

def forwardPropagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transferDerivative(output):
	return output * (1.0 - output)

def backwardPropagateError(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transferDerivative(neuron['output'])

def updateWeights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def trainNetwork(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forwardPropagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[int(row[-1])] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backwardPropagateError(network, expected)
			updateWeights(network, row, l_rate)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

class NaiveBayes(Predictor):
    def __init__(self):
        self.bayes = []
        self.features  = np.array([[]])
        self.labels = []
        self.labelsDict = {}
        self.labelsRevDict = {}
        
    def train(self, instances):
        #should output trainer

       
        #Make features and labels into arrays
        labelIndex = 0
        for instance in instances:
            feat_list = instance.getFeatures()
            feat_len = len(feat_list)

            label = instance.getLabel().getValue()
            
            if label not in self.labelsDict.keys():
                self.labelsDict[label] = labelIndex
                self.labelsRevDict[labelIndex] = label
                labelIndex += 1
            self.labels.append(self.labelsDict[label])
            
            if feat_len > self.features.shape[1]:
                b = np.zeros((self.features.shape[0], feat_len - self.features.shape[1]))
                self.features = np.hstack((self.features, b))
            elif feat_len < self.features.shape[1]:
                feat_list.append([0]*(self.features.shape[1] - feat_len))
            self.features = np.vstack((self.features, feat_list))
        self.features = np.delete(self.features, 0, 0)
        self.dataset = np.hstack((self.features, np.array(self.labels).reshape((len(self.labels), 1))))


        separated = separate(self.dataset)
        results = {}
        for value, classInstances in separated.iteritems():
            results[value] = summarize(classInstances)
        self.bayes = results
        return results
        
        
    def predict(self, instance):
        #predicted output of of a single instance
        
        features = instance.getFeatures()
                
        probs = getprobs(self.bayes, features)
        bestLabel, bestProb = None, -1
        for value, prob in probs.iteritems():
            if bestLabel is None or prob > bestProb:
                bestProb = prob
                bestLabel = value
        return self.labelsRevDict[int(bestLabel)]    
    
def separate(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
    
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def getprob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def getprobs(summaries, vector):
    probs = {}
    for value, classSummaries in summaries.iteritems():
        probs[value] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = vector[i]
            probs[value] *= getprob(x, mean, stdev)     
    return probs
