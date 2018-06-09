from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
import naive_bayes
import numpy as np
import pandas as pd
import csv
import random
import math


manual_labeled_link = '../../data/manual_labeled.npy'
unlabeled_csv_link = '../../data/unlabel.csv'
unlabeled_npy_link = '../../data/unlabel.npy'
positive_labeled_link = '../../data/black_label.csv'
combined_PU_link = '../../data/combined_PU_data.npy'


def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[0] not in separated):
			separated[vector[0]] = []
		separated[vector[0]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[0]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities


def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions


def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][0] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def mark_unlabel(unlabeled_csv_link, manual_labeled_link):
	unlabeled_data = pd.read_csv(unlabeled_csv_link, header=None)
	print(unlabeled_data)
	np.save(unlabeled_npy_link, unlabeled_data)
	labeled_data = unlabeled_data
	labeled_data.iloc[:,0] = 0
	np.save(manual_labeled_link,labeled_data)



def combine_labeled_data(manual_labeled_link, positive_labeled_link, combined_PU_link):
    positive_label_data = pd.read_csv(positive_labeled_link, header=None)
    positive_label_data = positive_label_data.values
    manual_labeled_data = np.load(manual_labeled_link)
    combined_PU_data = np.concatenate((positive_label_data,manual_labeled_data),axis = 0)
    np.save(combined_PU_link, combined_PU_data)

def main():
	testTrainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = ["?!#%&"]))
	mark_unlabel(unlabeled_csv_link, manual_labeled_link)
	combine_labeled_data(manual_labeled_link, positive_labeled_link, combined_PU_link)
	training_data = np.load(combined_PU_link)
	unlabeled_data = np.load(unlabeled_npy_link)
	print(unlabeled_data)
	labels = training_data[:,0]
	features = training_data[:,1:]

	training_data, test_data = splitDataset(training_data,0.6)
	summaries = summarizeByClass(training_data)
	print(summaries)


	prediction = getPredictions(summaries,test_data)
	print(prediction)
	accuracy = getAccuracy(test_data, prediction)
	print('Accuracy: {}'.format(accuracy))


	# print('prediction {}'.format(prediction))
	# for data in training_data:
	# 	prediction = getPredictions(summaries,data)
	# 	accuracy = getAccuracy(data, prediction)
	# 	print('Accuracy: {}'.format(accuracy))

    # for feature in features:
    #     print(feature)
    #
    # for label in labels:
    #     print(label)

if __name__ == "__main__":
    main()
