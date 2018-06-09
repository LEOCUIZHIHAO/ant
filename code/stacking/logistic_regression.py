from sklearn.linear_model import LogisticRegression
import numpy as np
import math

def stack_split(features,labels,number_of_model):
    # Define number of sizes per model
    test_size = int(labels.size/number_of_model)
    # Iterate number of models to get different test, feature and label data
    test = {}
    feature = {}
    label = {}
    for i in range(number_of_model):
        # define starting and end rows of the test data
        start_row = test_size * i
        end_row = start_row + test_size
        print(start_row)
        print(end_row)
        # Store extrated test data from feature
        test["test_{}".format(i+1)] = features[start_row:end_row,:]
        # Delete the extrated data from feature and label data
        feature["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:(start_row + test_size)], axis = 0)
        label["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:(start_row + test_size)], axis = 0)
    return test, feature, label

# def stack_logistic(feature,label):
#     features = {}
#     labels = {}
#     tests = {}
#     test_size = int(label.size()/number_of_model)
#     for i in range(number_of_model):
#         start_row = test_size * i
#         test["test{}".format(i+1)] = feature[start_row:(start_row + test_size),:]
#         feature["feature{}".format(i+1)] = np.delete(features, np.s_[start_row:(start_row + test_size)], 0)
#         label["label{}".format(i+1)] = np.delete(features, np.s_[start_row:(start_row + test_size)], 0)
#
#     logistic = LogisticRegression()


def main():
    data = np.load("../../data/manual_labeled.npy")
    labels = data[:,0]
    features = data[:,1:]
    test, feature, label = stack_split(features,labels,5)
    # print(len(test))
    # print(len(feature))
    # print(len(label))
    #
    # for i in range(len(feature)):
    #     print(test["test_{}".format(i+1)])
    #     print()

if __name__ == "__main__":
    main()
