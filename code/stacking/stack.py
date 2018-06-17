import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer, StandardScaler, Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.decomposition import PCA

param = {
        "objective" : "binary:logistic",
        "max_depth" : 4,
        "subsample" : 0.8,
        "colsample_bytree" : 0.9,
        "min_child_weight" : 1,
        "gamma" : 0.1,
        "eta" : 0.07, #learning_rate
        "eval_metric" : ['error'], #early stop only effects on error
        "silent" : 0
        }

num_round = 480
#early_stopping_rounds = 10

path1 = os.path.abspath(".")

#***********************model & score path *************************************#
#model save path#
model_path = path1 + "/save_restore/"
#where to save the figure & answer & hParams
score_path = path1 + "/score/"

#***********************data_path**********************************************#
data_path = path1 + "/data/"
#train_data path
#train_path = "/home/lecui/kaggle/data/train.npy"
train_path = data_path + "train.npy"
#test_data path
test_path = data_path + "test_a.npy"

#***********************Layer_1***********************************************#
stack_test_path = score_path + "stack_test_sheet.csv"
stack_train_path = score_path + "stack_train_sheet.csv"

layer1_train_path = data_path + "layer1_train.csv"
layer1_test_path = data_path + "layer1_test.csv"

layer1_train_formatted = data_path + "layer1_train_formatted.csv"
layer1_test_formatted = data_path + "layer1_test_formatted.csv"

layer1_2_layer2_train = data_path + "layer1_2_layer2_train.npy"
layer1_2_layer2_test = data_path + "layer1_2_layer2_test.npy"

final_test_path = score_path + "4_models_no_l2_all_data_no_xgb_no_std.csv"

fmap = path1 + "/fmap/xgb.fmap"


final_preds = []
avg_test_layer1_preds = []

# ####################Feature Processing####################
# Including 1. Imputation,
#           2. Standardization,
#           3. Normalizaiton
# ##########################################################
def feature_processing(features,test_feature):
    #print("Start imputating data")
    #Imputation
    #imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    #imp.fit(features)
    #features = imp.transform(features)
    #test_feature = imp.transform(test_feature)
    #Standardization
    print("Start preprocessing Standardization")
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    test_feature = scaler.transform(test_feature)
    #Normalizaiton L2,L1 norm
    #print("Start preprocessing L2 norm")
    #normalizer = Normalizer(norm='l2')
    #normalizer.fit(features)
    #features = normalizer.transform(features)
    #test_feature = normalizer.transform(test_feature)
    #PCA

    return features, test_feature

# ####################Feature Engineer######################
def select_features_from_xgb(features,labels,test_feature):

    print("\nStart selecting importance features")
    xgb = XGBClassifier(n_estimators=2, max_depth=4, learning_rate = 0.07, subsample = 0.8, colsample_bytree = 0.9)
    xgb = xgb.fit(features, labels)
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]

    model = SelectFromModel(xgb, prefit=True)
    features_new = model.transform(features)
    test_feature_new = model.transform(test_feature)
    with open(data_path + "importance_features.txt" , "w") as log:
        for f in range(features_new.shape[1]):
            log.write(str(f + 1) + "." +  " feature " +  str(indices[f]) + "  " + str(importances[indices[f]]) + "\n")
            #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("Features selection done saved new data in data path")

    return features_new, test_feature_new

# ####################CV Slicing############################
def stack_split(features, labels, number_of_model):
    # Define number of sizes per model
    fold_size = int(labels.size/number_of_model)

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

# ####################First Layer Start#####################
def save_final_layer_score(score):

	final_score = pd.read_csv(stack_test_path)
	f_score = final_score.assign(score = score)
	f_score.to_csv(final_test_path, index = None, float_format = "%.9f")
	print("\nFinal score saved to {}".format(final_test_path))

def stack_xgb_layer1(features, label, test):

    final_fold_preds = []

    fold_split, feature_split, label_split = stack_split(features, label, 5)

    for fold in range(1, len(fold_split)+1):

        print("\nStarting fold_{}".format(fold))

        #fold train data
        dtrain = xgb.DMatrix(feature_split["feature_{}".format(fold)], label = label_split["label_{}".format(fold)])
        #fold testing data
        dfold_val = xgb.DMatrix(fold_split["fold_{}".format(fold)])
        #test data
        dtest = xgb.DMatrix(test)
        #train model
        bst = xgb.train(param, dtrain, num_round)

        #save model
        bst.save_model(model_path + "XGB_layer_1_model_{}.model".format(fold))
        print("\nsaved model <XGB_layer_1_model_{}.model>".format(fold))

        print("\n" + "*" * 20 + "Starting preds fold_{}".format(fold) + "*" * 20)
        fold_val_preds = bst.predict(dfold_val)
        fold_val_preds = fold_val_preds.tolist()
        print("\n" + "*" * 20 + "Starting preds test_data by using current model" + "*" * 20)
        test_layer1_preds = bst.predict(dtest)

        final_fold_preds += fold_val_preds

        #append predicted test data in one list
        avg_test_layer1_preds.append(test_layer1_preds)


    return final_fold_preds,  avg_test_layer1_preds

def stack_logistic_layer1(features,labels,test_feature):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("Initiate stack logistic regression")
    for i in range(len(fold_split)):
        print("\nProcessing logistic model number:{}".format(i+1))
        logistic = LogisticRegression(class_weight = "balanced")
        logistic.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = logistic.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = logistic.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(logistic, model_path + "LR_layer_1_model_{}.pkl".format(i+1))
        print("LR model number:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

def stack_random_forest_layer1(features,labels,test_feature):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("\nInitiate stack random forest")
    for i in range(len(fold_split)):
        print("\nProcessing random forest model number:{}".format(i+1))
        random_forest = RandomForestClassifier(n_estimators = 450, max_depth = 4, criterion='entropy')
        random_forest.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = random_forest.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = random_forest.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(random_forest, model_path + "RF_layer_1_model_{}.pkl".format(i+1))
        print("RF model nubmer:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

def stack_mlp_layer1(features,labels,test_feature):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("\nInitiate stack MLP")
    for i in range(len(fold_split)):
        print("\nProcessing MLP model number:{}".format(i+1))
        mlp = MLPClassifier(hidden_layer_sizes=(256,128,128), activation = "logistic", batch_size = 20000)
        mlp.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = mlp.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = mlp.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(mlp, model_path + "MLP_layer_1_model_{}.pkl".format(i+1))
        print("MLP model nubmer:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

def stack_extra_trees_layer1(features,labels,test_feature):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("\nInitiate stack extra_trees")
    for i in range(len(fold_split)):
        print("\nProcessing random forest model number:{}".format(i+1))
        extra_trees = ExtraTreesClassifier(n_estimators = 450, max_depth = 4, criterion='entropy')
        extra_trees.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = extra_trees.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = extra_trees.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(extra_trees, model_path + "ET_layer_1_model_{}.pkl".format(i+1))
        print("ET model nubmer:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

def stack_knn_layer1(features,labels,test_feature):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("\nInitiate stack KNN")
    for i in range(len(fold_split)):
        print("\nProcessing KNN model number:{}".format(i+1))
        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=30, n_jobs=-1)
        knn.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = knn.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = knn.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(knn, model_path + "ET_layer_1_model_{}.pkl".format(i+1))
        print("KNN model nubmer:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

def stack_svm_layer1(features,labels,test_feature):
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("\nInitiate stack SVM")
    for i in range(len(fold_split)):
        print("\nProcessing SVM model number:{}".format(i+1))
        svm = SVC(n_estimators = 450, max_depth = 4, criterion='entropy')
        svm.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = svm.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = svm.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(svm, model_path + "SVM_layer_1_model_{}.pkl".format(i+1))
        print("SVM model nubmer:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

# ####################Output of First Layer##################
def save_layer_score(final_fold_preds, avg_test_layer1_preds, stack_train_path, stack_test_path, model, init_model=True):

	model_list = ['model_1', 'model_2', 'model_3', 'model_4','model_5']
	if model not in model_list:
		raise TypeError("{} not in model list".format(model))

	#Averaging stacked
	stack_test_layer1_preds = np.stack(avg_test_layer1_preds, 1)
	#averaging stacked data
	avged_test_preds = []

	for row in stack_test_layer1_preds:
	    avg = np.mean(row)
	    avged_test_preds.append(avg)

	print("\nAveraging test score done ......")

	if init_model:
		layer_train = pd.read_csv(stack_train_path)
	else:
		layer_train = pd.read_csv(layer1_train_path)
	layer_train = pd.DataFrame(layer_train)
	if model == "model_1":
		layer_train_score = layer_train.assign(model_1 = final_fold_preds)
	elif model == "model_2":
		layer_train_score = layer_train.assign(model_2 = final_fold_preds)
	elif model == "model_3":
		layer_train_score = layer_train.assign(model_3 = final_fold_preds)
	elif model == "model_4":
		layer_train_score = layer_train.assign(model_4 = final_fold_preds)
	elif model == "model_5":
		layer_train_score = layer_train.assign(model_5 = final_fold_preds)

	layer_train_score.to_csv(layer1_train_path, index = None, float_format = "%.9f")
	print("\nLayer_1 xgb <train> score saved to %s \n" % (layer1_train_path))
	######################test_layer#########################
	if init_model:
		layer_test = pd.read_csv(stack_test_path)
	else:
		layer_test = pd.read_csv(layer1_test_path)
	layer_test = pd.DataFrame(layer_test)
	if model == "model_1":
		layer_test_score = layer_test.assign(model_1 = avged_test_preds)
	elif model == "model_2":
		layer_test_score = layer_test.assign(model_2 = avged_test_preds)
	elif model == "model_3":
		layer_test_score = layer_test.assign(model_3 = avged_test_preds)
	elif model == "model_4":
		layer_test_score = layer_test.assign(model_4 = avged_test_preds)
	elif model == "model_5":
		layer_test_score = layer_test.assign(model_5 = avged_test_preds)

	layer_test_score.to_csv(layer1_test_path, index = None, float_format = "%.9f")
	print("\nLayer_1 xgb <test> score saved to %s \n" % (layer1_test_path))

# ####################Layer 2 Input##########################
def layer2_formation(layer1_data_path, save_path, npy_save_path):
	print("\nStart formating layer1 data")
	df = pd.read_csv(layer1_data_path)
	df = df.drop(['id'], axis = 1)
    ## TODO:  dont need to save
	df.to_csv(save_path, index = None, header = False)
	_csv = np.loadtxt(save_path, delimiter = ",")
	np.save(npy_save_path, _csv)
	print("\nFormation done ....")

# ####################Second Layer Start#####################
def stack_xgb_layer2(train_path, label, test_path):

	features = np.load(train_path)
	test = np.load(test_path)
	#label = original label
	dtrain = xgb.DMatrix(features, label=label)
	dtest = xgb.DMatrix(test)

	bst = xgb.train(param, dtrain, num_round)
	bst.save_model(model_path + "XGB_layer_2.model")
	print("\nSaved model <XGB_layer_2.model>")
	layer2_preds = bst.predict(dtest)

	return layer2_preds

def stack_extra_trees_layer2(features,labels,test_feature):
    features = np.load(features)
    test = np.load(test_feature)
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("\nInitiate stack extra_trees")
    for i in range(len(fold_split)):
        print("\nProcessing random forest model number:{}".format(i+1))
        extra_trees = ExtraTreesClassifier(n_estimators = 450, max_depth = 4, criterion='entropy')
        extra_trees.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = extra_trees.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = extra_trees.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(extra_trees, model_path + "ET_layer_2_model_{}.pkl".format(i+1))
        print("ET model nubmer:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score

def stack_logistic_layer2(features,labels,test_feature):
    features = np.load(features)
    test = np.load(test_feature)
    fold_split, feature_split, label_split = stack_split(features,labels,5)
    fold_score = []
    test_score = []
    print("Initiate stack logistic regression")
    for i in range(len(fold_split)):
        print("\nProcessing logistic model number:{}".format(i+1))
        logistic = LogisticRegression(class_weight = "balanced")
        logistic.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
        print("Training complete")
        stack_score = logistic.predict_proba(fold_split["fold_{}".format(i+1)])
        print("fold score predicted")
        test_prediction = logistic.predict_proba(test_feature)
        print("test score predicted")
        test_score.append(test_prediction[:,1].tolist())
        fold_score += stack_score[:,1].tolist()
        joblib.dump(logistic, model_path + "LR_layer_2_model_{}.pkl".format(i+1))
        print("LR model number:{}".format(i+1) + " complete")
        # print(scores)
    return fold_score, test_score


def main():

    train_data = np.load(train_path)

    test = np.load(test_path)

    features = train_data[:,1:]

    label = train_data[:,0]

    features, test = feature_processing(features, test)

    #features, test = select_features_from_xgb(features, label, test)

    final_fold_preds,  avg_test_layer1_preds = stack_xgb_layer1(features, label, test)

    save_layer_score(final_fold_preds, avg_test_layer1_preds, stack_train_path, stack_test_path, model = "model_1", init_model = True)

    final_fold_preds,  avg_test_layer1_preds = stack_logistic_layer1(features, label, test)

    save_layer_score(final_fold_preds, avg_test_layer1_preds, stack_train_path, stack_test_path, model = "model_2", init_model = False)

    final_fold_preds,  avg_test_layer1_preds = stack_random_forest_layer1(features, label, test)

    save_layer_score(final_fold_preds, avg_test_layer1_preds, stack_train_path, stack_test_path, model = "model_3", init_model = False)

    final_fold_preds,  avg_test_layer1_preds = stack_mlp_layer1(features, label, test)

    save_layer_score(final_fold_preds, avg_test_layer1_preds, stack_train_path, stack_test_path, model = "model_4", init_model = False)

    #final_fold_preds,  avg_test_layer1_preds = stack_knn_layer1(features, label, test)

    #save_layer_score(final_fold_preds, avg_test_layer1_preds, stack_train_path, stack_test_path, model = "model_5", init_model = False)

    layer2_formation(layer1_train_path, layer1_train_formatted, layer1_2_layer2_train)

    layer2_formation(layer1_test_path, layer1_test_formatted, layer1_2_layer2_test)

    final_preds = stack_xgb_layer2(layer1_2_layer2_train, label, layer1_2_layer2_test)

    save_final_layer_score(final_preds)

    """
    final_fold_preds,  avg_test_layer2_preds = stack_xgb_layer2(layer1_2_layer2_train, label, layer1_2_layer2_test)

    save_layer_score(final_fold_preds, avg_test_layer2_preds, stack_train_path, stack_test_path, model = "model_1", init_model = True)

    final_fold_preds,  avg_test_layer2_preds = stack_logistic_layer2(layer1_2_layer2_train, label, layer1_2_layer2_test)

    save_layer_score(final_fold_preds, avg_test_layer2_preds, stack_train_path, stack_test_path, model = "model_2", init_model = False)

    final_fold_preds,  avg_test_layer2_preds = stack_extra_trees_layer2(layer1_2_layer2_train, label, layer1_2_layer2_test)

    save_layer_score(final_fold_preds, avg_test_layer2_preds, stack_train_path, stack_test_path, model = "model_3", init_model = False)

    layer2_formation(layer1_train_path, layer1_train_formatted, layer1_2_layer2_train)

    layer2_formation(layer1_test_path, layer1_test_formatted, layer1_2_layer2_test)

    final_preds = stack_xgb_layer3(layer2_2_layer3_train, label, layer2_2_layer3_test)

    save_final_layer_score(final_preds)

    """




if __name__ == '__main__':
    main()
