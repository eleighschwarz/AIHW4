import os
import argparse
import sys
import pickle
import numpy as np
from Methods import ClassificationLabel, FeatureVector, Instance, Predictor, DecisionTree, NaiveBayes, NeuralNetwork

def load_data(filename):
	instances = []
	with open(filename) as reader:
		for line in reader:
			if len(line.strip()) == 0:
				continue
			
			# Divide the line into features and label.
			split_line = line.split(",")
			label_string = split_line[0]

			label = ClassificationLabel(label_string)
			feature_vector = FeatureVector()
			
			index = 0
			for item in split_line[1:]:  
				value = float(item)

				feature_vector.add(index, value)
				index += 1

			instance = Instance(feature_vector, label)
			instances.append(instance)

	return instances

def get_args():
	parser = argparse.ArgumentParser(description="This allows you to specify the arguments you want for classification.")

	parser.add_argument("--data", type=str, required=True, help="The data files you want to use for training or testing.")
	parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Mode: train or test.")
	parser.add_argument("--model-file", type=str, required=True, help="Filename specifying where to save or load model.")
	parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")
	parser.add_argument("--ratio", type=str, help="Use information gain ratio.", default="False")
	parser.add_argument("--prune", type=str, help="Prune decision tree.", default="False")
	parser.add_argument("--weights", type=str, help="Type of neural network weights", default="normal")


	args = parser.parse_args()
	check_args(args)

	return args

def predict(predictor, instances):
        y_true = []
        y_pred = []
	for instance in instances:
		label = predictor.predict(instance)
		print(str(label))
                y_true.append(str(instance.getLabel()))
                y_pred.append(str(label))
        return y_true, y_pred

def check_args(args):
	if args.mode.lower() == "train":
		if args.algorithm is None:
			raise Exception("--algorithm must be specified in mode \"train\"")
	else:
		if not os.path.exists(args.model_file + "_" + args.algorithm + ".model"):
			raise Exception("model file specified by --model-file does not exist.")

def train(instances, algorithm, ratio, prune, weights):
	"""
	This is where you tell classify.py what algorithm to use for training
	The actual code for training should be in the Predictor subclasses
	For example, if you have a subclass DecisionTree in Methods.py
	You could say
	if algorithm == "decision_tree":
		predictor = DecisionTree()
	"""
        if algorithm.lower() == "naive_bayes":
                predictor = NaiveBayes()
                predictor.train(instances)
        elif algorithm.lower() == "decision_tree":
                predictor = DecisionTree(ratio, prune)
                predictor.train(instances)
        elif algorithm.lower() == "neural_network":
                predictor = NeuralNetwork(weights)
                predictor.train(instances)

        return predictor

def getStats(y_true, y_pred, filename):

        try:
                with open((str(filename)+"_stats.txt"), 'w') as writer:

        
                        labels = list(set(y_true).union(set(y_pred)))

                        accuracy = 0.0
                        precision = 0.0
                        recall = 0.0
                        for i in xrange(len(y_true)):
                                if y_true[i] == y_pred[i]:
                                        accuracy += 1
                        if len(y_true) > 0:
                                accuracy = accuracy/len(y_true)
                        writer.write("Accuracy " + str(accuracy))
                        writer.write('\n')

                        for label in labels:
                                pred_of_l = 0.0
                                true_of_l = 0.0
                                corr_of_l = 0.0
                                for i in xrange(len(y_true)):
                                        if y_true[i] == label:
                                                true_of_l += 1
                                        if y_pred[i] == label:
                                                pred_of_l += 1
                                        if y_pred[i] == label and y_true[i] == label:
                                                corr_of_l += 1
                                if pred_of_l and true_of_l > 0:
                                        precision = corr_of_l/pred_of_l
                                        recall = corr_of_l/true_of_l
                                writer.write("Precision of '" + str(label) + "' " + str(precision))
                                writer.write('\n')
                                writer.write("Recall of '" + str(label) + "' "  + str(recall))
                                writer.write('\n')

        except IOError:
                raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
	args = get_args()
	if args.mode.lower() == "train":
		# Load training data.
		instances = load_data(args.data)

		# Train
		predictor = train(instances, args.algorithm, args.ratio, args.prune, args.weights)
		try:
			with open((args.model_file + "_" + args.algorithm + ".model"), 'wb') as writer:
				pickle.dump(predictor, writer)
		except IOError:
			raise Exception("Exception while writing to the model file.")
		except pickle.PickleError:
			raise Exception("Exception while dumping pickle.")

	elif args.mode.lower() == "test":
		# Load the test data.
		instances = load_data(args.data)

		predictor = None
		# Load model
		try:
			with open((args.model_file + "_" + args.algorithm + ".model"), 'rb') as reader:
				predictor = pickle.load(reader)
		except IOError:
			raise Exception("Exception while reading the model file.")
		except pickle.PickleError:
			raise Exception("Exception while loading pickle.")

		y_true, y_pred = predict(predictor, instances)
                getStats(y_true, y_pred, (args.model_file + "_" + args.algorithm))

                
                
	else:
		raise Exception("Unrecognized mode.")

if __name__ == "__main__":
	main()
