from sklearn.linear_model import LogisticRegression
import numpy as np
import math

def stack_split(features,labels,number_of_model):
    # Define number of sizes per model
    fold_size = int(labels.size/number_of_model)
    # Iterate number of models to get different fold, feature and label data
    fold_split = {}
    feature_split = {}
    label_split = {}
    for i in range(number_of_model):
        # define starting and end rows of the fold data
        start_row = fold_size * i
        end_row = start_row + fold_size
        print(start_row)
        print(end_row)
        # Store extrated fold data from feature
        fold_split["fold_{}".format(i+1)] = features[start_row:end_row,:]
        # Delete the extrated data from feature and label data
        feature_split["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:(start_row + fold_size)], axis = 0)
        label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:(start_row + fold_size)], axis = 0)
    return fold_split, feature_split, label_split

def stack_logistic(feature,labels):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    for i in range(len(fold_split):
        logistic = LogisticRegression()
        logistic.fit(feature_split["feature_{}".format(i)])



def main():
    fold, feature, label = stack_split(features,labels,5)
    # print(len(fold))
    # print(len(feature))
    # print(len(label))
    #
    # for i in range(len(feature)):
    #     print(fold["fold_{}".format(i+1)])
    #     print()

if __name__ == "__main__":
    main()
