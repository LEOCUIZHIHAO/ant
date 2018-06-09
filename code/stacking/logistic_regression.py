from sklearn.linear_model import LogisticRegression
import numpy as np
import math

def stack_split(features,labels,number_of_model):
    # Define number of sizes per model
    fold_size = int(labels.size/number_of_model)
    print(labels.size)
    # Iterate number of models to get different fold, feature and label data
    fold_split = {}
    feature_split = {}
    label_split = {}

    for i in range(number_of_model):
        # define starting and end rows of the fold data
        start_row = fold_size * i
        end_row = fold_size * (i+1)

        if i == number_of_model - 1:

            print("\nfold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            fold_split["fold_{}".format(i+1)] = features[start_row:,:]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:], axis = 0)

        else:

            print("\nfold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            # Store extrated fold data from feature
            fold_split["fold_{}".format(i+1)] = features[start_row:end_row,:]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:(start_row + fold_size)], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:(start_row + fold_size)], axis = 0)

    print("\nEnd of split, acess via ['fold_number'], ['feature_number'] and ['label_number']")
    return fold_split, feature_split, label_split

def stack_logistic(features,labels):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    scores = []
    for i in range(len(fold_split)):
        print("\nProcessing logistic model number:{}".format(i+1))
        logistic = LogisticRegression()
        logistic.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        stack_score = logistic.predict_proba(fold_split["fold_{}".format(i+1)])
        scores = scores + stack_score[:,0].tolist()
        print(scores)
    return scores


def main():
    data = np.load("../../data/combined_PU_data.npy")
    feature = data[:,1:]
    label = data[:,0]
    scores = stack_logistic(feature,label)
    print(scores)
    # print(len(fold))
    # print(len(feature))
    # print(len(label))
    #
    # for i in range(len(feature)):
    #     print(fold["fold_{}".format(i+1)])
    #     print()

if __name__ == "__main__":
    main()
