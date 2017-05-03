from abc import ABCMeta, abstractmethod

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
        print self.label_str
        pass

# the feature vectors will be stored in dictionaries so that they can be sparse structures
class FeatureVector:
    def __init__(self):
        self.feature_vec = {}
        pass
        
    def add(self, index, value):
        self.feature_vec[index] = value
        pass
        
    def get(self, index):
        val = self.feature_vec[index]
        return val
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

    
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


# Need to make this so it returns data set separated by label
def separate_by_label(dataset):
    separated = {}
    for i in range(len(dataset)):
        
        
    
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


class DecisionTree(Predictor):
    def __init__(self):
        #put needed data structures
        w = 0
        
    def train(self, instances):
        #should output trainer
        return null

    def predict(self, instance):
        #predicted output of of a single instance
        return null
        

class NaiveBayes(Predictor):
    def __init__(self):
        #put needed data structures
        w = 0 
    
    def train(self, instances):
        #should output trainer
        results = {}
        separated = separate_by_label(instances)
        for classValue, instance in separated.iteritems():
            summaries[classValue] = summarize(instance)
            return summaries

    def predict(self, instance):
        #predicted output of of a single instance
        return null


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