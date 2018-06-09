import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


param = {
        "objective" : "binary:logistic",
        "max_depth" : 3,
        "subsample" : 0.9,
        "colsample_bytree" : 0.9,
        "min_child_weight" : 1,
        "gamma" : 0,
        "eta" : 0.09, #learning_rate
        "eval_metric" : ['error'], #early stop only effects on error
        "silent" : 0
        }

num_round = 4
early_stopping_rounds = 10
offset = 10000

path1 = os.path.abspath(".")
#***********************data_path***********************************************#
#train_data path
train_path = path1 + "/train.npy"
#test_data path
test_path = path1 + "/test_a.npy"

#***********************model & score path ***********************************************#
#model save path#
model_path = path1 + "/save_restore/"
#where to save the figure & answer & hParams
score_path = path1 + "/score/"

stack_test_path = score_path + "/answer_sheet.csv"
stack_train_path = path1 + "/stack_sheet.csv"
fmap = path1 + "/fmap/xgb.fmap"

final_preds = []
avg_test_layer1_preds = []

def stack_split(features,labels,number_of_model):
    # Define number of sizes per model
    fold_size = int(labels.size/number_of_model)
    print("fold_size", fold_size)
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


    return fold_split, feature_split, label_split


def save_layer_score(avg_test_layer1_preds, final_fold_preds, stack_train_path, stack_test_path):

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
    layer_1_train_score.to_csv(score_path + "layer_1_train.csv", index = None, float_format = "%.9f")
    print("\nLayer_1 xgb <train> score saved to %s \n" % (score_path+ "layer_1_train.csv"))

    layer_1_test = pd.read_csv(stack_test_path)
    layer_1_test = pd.DataFrame(layer_1_test)
    layer_1_test_score = layer_1_test.assign(score = avged_test_layer1_preds)
    layer_1_test_score.to_csv(score_path + "layer_1_test.csv", index = None, float_format = "%.9f")
    print("\nLayer_1 xgb <test> score saved to %s \n" % (score_path+ "layer_1_test.csv"))


def main():

    train_data = np.load(train_path)
    # TODO: fold_val
    test = np.load(test_path)

    features = train_data[:,1:]

    label = train_data[:,0]

    fold_split, feature_split, label_split = stack_split(features, label, 5)

    final_fold_preds = []

    for fold in range(1, len(fold_split)+1):

        print("\nStarting fold_{}".format(fold))
        print(len(feature_split["feature_{}".format(fold)]))
        #fold train data
        dtrain = xgb.DMatrix(feature_split["feature_{}".format(fold)], label = label_split["label_{}".format(fold)])
        #fold testing data
        dfold_val = xgb.DMatrix(fold_split["fold_{}".format(fold)])
        #test data
        dtest = xgb.DMatrix(test)

        #train model
        bst = xgb.train(param, dtrain, num_round)

        #save model
        bst.save_model(model_path + "xbg_{}.model".format(fold))
        print("\nsaved model <xbg_{}.model>".format(fold))

        print("\n" + "*" * 20 + "Starting preds fold_{}".format(fold) + "*" * 20)
        fold_val_preds = bst.predict(dfold_val)

        fold_val_preds = fold_val_preds.tolist()

        print("\n" + "*" * 20 + "Starting preds test_data by using current model" + "*" * 20)
        test_layer1_preds = bst.predict(dtest)

        final_fold_preds += fold_val_preds

        print(len(final_fold_preds))

        #append predicted test data in one list
        avg_test_layer1_preds.append(test_layer1_preds)

    #save layer 1 score
    save_layer_score(avg_test_layer1_preds, final_fold_preds, stack_train_path, stack_test_path)

    """
    # TODO:  Outside for loop
    #stack test data
    stack_test_layer1_preds = np.stack(avg_test_layer1_preds, 1)
    #flash
    avg_test_layer1_preds = []
    #averaging stacked data
    for row in stack_test_layer1_preds:
        avg = np.mean(row)
        avg_test_layer1_preds.append(avg)

    print("averaging done")

    layer_1_train = pd.read_csv(stack_path)
    #Dataframe data
    layer_1_train = pd.DataFrame(layer_1_csv)
    #Feed result in score column
    layer_1_train_score = layer_1_csv.assign(score = final_fold_preds)
    #Save to .csv
    layer_1_train_score.to_csv(score_path + "layer_1_train.csv", index = None, float_format = "%.9f")
    print("Layer_1 xgb <train> score saved to %s \n" % (score_path+ "layer_1_train.csv"))

    layer_1_test = pd.read_csv(as_path)
    layer_1_test = pd.DataFrame(layer_1_test)
    layer_1_test_score = layer_1_test.assign(score = avg_test_layer1_preds)
    layer_1_test_score.to_csv(score_path + "layer_1_test.csv", index = None, float_format = "%.9f")
    print("Layer_1 xgb <test> score saved to %s \n" % (score_path+ "layer_1_test.csv"))
    """

if __name__ == '__main__':
    main()
