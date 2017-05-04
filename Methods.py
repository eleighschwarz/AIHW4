from abc import ABCMeta, abstractmethod
import numpy as np
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
    def __init__(self):
        self.tree = {best:{}}
        self.features  = np.array([[]])
        self.labels = []
#        self.tree = [][]

    def train(self, instances):


        #Make features and labels into arrays
        for instance in instances:
            feat_list = instance.getFeatures()
            feat_len = len(feat_list)
            
            self.labels.append(instance.getLabel())

            if feat_len > self.features.shape[1]:
                b = np.zeros((self.features.shape[0], feat_len - self.features.shape[1]))
                self.features = np.hstack((self.features, b))
            elif feat_len < self.features.shape[1]:
                feat_list.append([0]*(self.features.shape[1] - feat_len))
            self.features = np.vstack((self.features, feat_list))
        self.features = np.delete(self.features, 0, 0)

        #Check for instances
        if len(instances) == 0:
            return 0
        elif len(set(self.labels)) <= 1:
            return labels[0]
        elif self.features.shape[0] == 0:
            return 0
                

        print self.features
        print self.features.shape
        print len(self.labels)


    def createDecisionTree(self):
        return null        
        
    def predict(self, instance):
        
        #predicted output of of a single instance
        return null

        
class NaiveBayes(Predictor):
    def __init__(self):
        self.bayes = []
        self.features  = np.array([[]])
        self.labels = []
        
    def train(self, instances):
        #should output trainer

        #Make features and labels into arrays
        for instance in instances:
            feat_list = instance.getFeatures()
            feat_len = len(feat_list)
            
            self.labels.append(instance.getLabel())

            if feat_len > self.features.shape[1]:
                b = np.zeros((self.features.shape[0], feat_len - self.features.shape[1]))
                self.features = np.hstack((self.features, b))
            elif feat_len < self.features.shape[1]:
                feat_list.append([0]*(self.features.shape[1] - feat_len))
            self.features = np.vstack((self.features, feat_list))
        self.features = np.delete(self.features, 0, 0)


        separated = separateByClass(self.labels, self.features)
        results = {}
        for values, instances in separated.iteritems():
            results[values] = summarize(instances)
            self.bayes = results
        return results
        
        
    def predict(self, instance):
        #predicted output of of a single instance
        
        features = instance.getFeatures()
                
        probabilities = getProbabilities(self.bayes, features)
        bestLabel, bestProb = None, -1
        for value, probability in probabilities.iteritems():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = value
        return bestLabel
    
def separateByClass(labels, features):
    separated = {}
    for i in range(len(features)):
        vector = features[i]
        if (labels[i] not in separated):
            separated[labels[i]] = []
        separated[labels[i]].append(vector)
    return separated
    
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(data):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in data]
    return summaries

def getProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def getProbabilities(summaries, features):
    probabilities = {}
    for value, summary in summaries.iteritems():
        probabilities[value] = 1
        print range(len(summary))
        for i in range(len(summary)):
            mean, stdev = summary[i]
            for j in range(len(features)):
                x = features[j]
                probabilities[value] *= getProbability(x, mean, stdev)     
    return probabilities


        
class NeuralNetwork(Predictor):
    def __init__(self):
        #put needed data structures
        w = 0

    def train(self, instances):
        #should output trainer
        return null

    def predict(self, instance):
        #predicted output of of a single instance
        return null
