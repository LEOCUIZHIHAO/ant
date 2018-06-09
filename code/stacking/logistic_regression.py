from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import pandas as pd
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
            print("fold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            fold_split["fold_{}".format(i+1)] = features[start_row:,:]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:], axis = 0)
        else:
            print("fold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            # Store extrated fold data from feature
            fold_split["fold_{}".format(i+1)] = features[start_row:end_row,:]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:end_row], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:end_row], axis = 0)

    print("\nEnd of split, acess via ['fold_number'], ['feature_number'] and ['label_number']")
    return fold_split, feature_split, label_split

def stack_logistic(features,labels,test_feature):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("Initiate stack logistic regression")
    for i in range(len(fold_split)):
        print("\nProcessing logistic model number:{}".format(i+1))
        logistic = LogisticRegression()
        logistic.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = logistic.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = logistic.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(logistic, "../../save_restore/LR_layer_1_model_{}".format(i+1))
        print("LR model numer:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

def save_layer_score(avg_test_layer1_preds, final_fold_preds, stack_train_path, stack_test_path, score_path):

    stack_test_layer1_preds = np.stack(avg_test_layer1_preds, 1)
    #averaging stacked data
    avged_test_layer1_preds = []

    for row in stack_test_layer1_preds:
        avg = np.mean(row)
        avged_test_layer1_preds.append(avg)

    print("\nAveraging test score done ......")

    layer_1_train = pd.read_csv(stack_train_path)
    #Dataframe data
    layer_1_train = pd.DataFrame(layer_1_train)
    #Feed result in score column
    layer_1_train_score = layer_1_train.assign(score = final_fold_preds)
    #Save to .csv
    layer_1_train_score.to_csv(score_path + "layer_RL_train.csv", index = None, float_format = "%.9f")
    print("\nLayer_1 xgb <train> score saved to %s \n" % (score_path+ "layer_RL_train.csv"))

    layer_1_test = pd.read_csv(stack_test_path)
    layer_1_test = pd.DataFrame(layer_1_test)
    layer_1_test_score = layer_1_test.assign(score = avged_test_layer1_preds)
    layer_1_test_score.to_csv(score_path + "layer_RL_test.csv", index = None, float_format = "%.9f")
    print("\nLayer_1 xgb <test> score saved to %s \n" % (score_path+ "layer_RL_test.csv"))

def main():
    score_path = "../../data/"
    stack_test_path = score_path + "answer_sheet.csv"
    stack_train_path = score_path + "stack_sheet.csv"
    train = np.load("../../data/train.npy")
    print("train data loaded")
    test = np.load("../../data/test_a.npy")
    print("test data loaded")
    train_feature = train[:,1:]
    train_label = train[:,0]
    scores, test_score = stack_logistic(train_feature,train_label,test)
    save_layer_score(test_score, scores, stack_train_path, stack_test_path, score_path)
    print("Program complete")
    # print(len(fold))
    # print(len(feature))
    # print(len(label))
    #
    # for i in range(len(feature)):
    #     print(fold["fold_{}".format(i+1)])
    #     print()

if __name__ == "__main__":
    main()
