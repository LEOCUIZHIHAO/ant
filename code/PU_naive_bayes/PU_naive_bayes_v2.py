import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

manual_labeled_link = '../../data/manual_labeled.npy'
unlabeled_csv_link = '../../data/unlabel.csv'
unlabeled_npy_link = '../../data/unlabel.npy'
unlabeled_csv_time_link = '../../data/unlabel_with_time.csv'
labeled_csv_time_link = '../../data/labeled_with_time.csv'
positive_labeled_link = '../../data/black_label.csv'
black_label_npy_link = '../../data/black_label.npy'
combined_PU_link = '../../data/combined_PU_data.npy'
validation_data_link = '../../data/validation_data.npy'

# mark the labeled -1 data into labeled 0 data
def mark_unlabel(unlabeled_csv_link, manual_labeled_link):
	unlabeled_data = pd.read_csv(unlabeled_csv_link, header=None)
	np.save(unlabeled_npy_link, unlabeled_data)
	labeled_data = unlabeled_data
	labeled_data.iloc[:,0] = 0
	np.save(manual_labeled_link,labeled_data)

def combine_labeled_data(manual_labeled_link, positive_labeled_link, combined_PU_link):
    positive_label_data = pd.read_csv(positive_labeled_link, header=None)
    np.save(black_label_npy_link, positive_label_data)
    positive_label_data = positive_label_data.values
    manual_labeled_data = np.load(manual_labeled_link)
    combined_PU_data = np.concatenate((positive_label_data,manual_labeled_data),axis = 0)
    np.save(combined_PU_link, combined_PU_data)

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def main():
    # Mark the unlabeled data sets as labeled
    mark_unlabel(unlabeled_csv_link, manual_labeled_link)
    # Combine both Positive training data and manual labeled training data together
    # Input: NPY file address of manual labeled data, positive labeled data, and combined data
    combine_labeled_data(manual_labeled_link, positive_labeled_link, combined_PU_link)
    # Load unlabed csv with time
    unlabeled_data_time = pd.read_csv(unlabeled_csv_time_link, header = None)
    # unlabeled_data_time = unlabeled_data_time.values
    # Load combined training data
    training_data = np.load(combined_PU_link)
    labels = training_data[:,0]
    features = training_data[:,1:]
    # Load unlabed data and extract
    unlabeled_data = np.load(unlabeled_npy_link)
    unlabeled_data_feature = unlabeled_data[:,1:]
    #Load validation dataset
    validation_data = np.load(validation_data_link)
    validation_data_feature = validation_data[:,1:]
    validation_data_label = validation_data[:,0]
    # Initiate Gaussian naive bayes object
    clf = GaussianNB()
    # Train the naive bayes classifier
    clf.fit(features,labels)

    # Cross Validation
    # print("Initiate Cross Validation process")
    # miss_label = 0
    # validation_data_size = validation_data_feature[:,0].size
    # for i in range(validation_data_size):
    #     validation_data_feature_1 = validation_data_feature[i].reshape(1,-1)
    #     prediction = clf.predict(validation_data_feature_1)
    #     if prediction != validation_data_label[i]:
    #         miss_label += 1
    #     print("Unlabeded sample: {}:".format(i) + " labeled prediction: {}".format(prediction))
    # print("Number of miss labeled: {}".format(miss_label) + " out of samples: {} ".format(validation_data_size) + ". Accuracy: {0:.2f}%".format(((validation_data_size-miss_label)/validation_data_size)*100))

    # Label process Initiate
    print("Initiate Lableling process")
    count = 0
    for i in range(unlabeled_data_feature[:,0].size):
        unlabeled_data_feature_1 = unlabeled_data_feature[i].reshape(1,-1)
        prediction = clf.predict(unlabeled_data_feature_1)
        if prediction == 1:
            unlabeled_data_time.iloc[i,1] = 1
            count += 1
        else:
            unlabeled_data_time.iloc[i,1] = 0
        print("Unlabeded sample: {}:".format(i) + " labeled prediction: {}".format(prediction) + "  labeled {}".format(unlabeled_data_time.iloc[i,1]))
    print("Number of samples labeled faud: {}".format(count) + " out of samples: {}".format(unlabeled_data_feature[:,0].size))
    print(unlabeled_data_time)
    print(unlabeled_data_time.iloc[:,1])
    labeled_data_time = unlabeled_data_time.values
    np.savetxt(labeled_csv_time_link, labeled_data_time, delimiter=",", fmt="%s")
if __name__ == "__main__":
    main()
