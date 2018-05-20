"""
@Authors Leo.cui
7/5/2018
Xgboost

"""


import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

#for i in range (65, 100, 4):
#*******************************hParams ***************************************#
#eta = round(i*0.01, 3)
num_round = 5
early_stopping_rounds = 10
max_depth = 10
subsample = 0.8
colsample_bytree = 0.9
min_child_weight = 1
gamma = 0
eta = 0.1 #learning_rate
offset = 150000 #validation
eval_metric = ['auc', 'logloss', 'error'] #early stop only effects on error
print("lr : ", eta)

#***********************data_path***********************************************#
#train_data path
train_path = "/home/leo/ant/model/data/train_xgb_1.npy"
#test_data path
test_path = "/home/leo/ant/model/data/test_a_xgb_1.npy"


#***********************model & score path ***********************************************#
#model save path#
model_path = "/home/leo/ant/model/"
#where to save the figure & answer & hParams
score_path = "/home/leo/ant/score/"

#***********************Tool-box***********************************************#
#pandas read answer_sheet
#the path of answer_sheet.csv
as_path = "/home/leo/ant/tool/answer_sheet.csv"
#xgb.fmap path
fmap = "/home/leo/ant/tool/xgb.fmap"

param = {
        'num_round' : num_round,
        'early_stopping_rounds' : early_stopping_rounds,
        'validation' : offset,
        'max_depth': max_depth,
        'subsample' : subsample,
        'colsample_bytree': colsample_bytree,
        'objective':'binary:logistic',
        'min_child_weight': min_child_weight,
        'gamma' : gamma,
        'eta':eta,
        'eval_metric':eval_metric,
        'silent':1
        #'eval_metric':'error'
        #'eval_metric': 'auc'
        #'eval_metric': 'logloss'
        }

def save_hParams (param, score_path, eta):

    f = csv.writer(open(score_path + "hParams_{}.csv".format(eta), "w"))
    for key, val in param.items():
        f.writerow([key, val])

    return print("hParams saved in : ", score_path)

def load_data(train_path, test_path, offset):
    #Load train & test set
    train_data = np.load(train_path)
    test = np.load(test_path)

    #Define training set
    train = train_data[:-offset,1:]
    label = train_data[:-offset,0]

    #Define validation set
    validation = train_data[-offset:,1:]
    validation_label = train_data[-offset:,0]

    return train, test, label, validation, validation_label

def create_DMatrix(param, train, test, label, validation, validation_label):

    dtrain = xgb.DMatrix(train, label=label, missing = -999.0) #missing data
    dval = xgb.DMatrix(validation, label=validation_label, missing = -999.0)
    dtest = xgb.DMatrix(test, missing = -999.0)

    watchlist = [(dval,'val'), (dtrain,'train')]

    evals_result = {}

    print("start training...")

    bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result, early_stopping_rounds = early_stopping_rounds )

    return bst, dtest, evals_result
#computitional cost too much
#eval = xgb.cv(param, dtrain, num_round, nfold=6, metrics={'error'}, seed=0,  callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])

def predict_and_save_model(bst, dtest, model_path, fmap, eta):
    #save model
    bst.save_model(model_path + "xbg_{}.model".format(eta))
    #save xgboost structure
    #bst.dump_model(model_path + "xbg_{}.txt".format(eta))

    preds = bst.predict(dtest, ntree_limit= bst.best_iteration)
    print("the minimal loss found in : %i booster " %(bst.best_iteration))

    features = pd.Series(bst.get_fscore(fmap = fmap)).sort_values(ascending=False)
    #save features and prepare feed into NN
    features.to_csv(model_path + "xgb_features_{}.csv".format(eta))

    print("saving model & features ......")

    return preds

"""
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

"""

#xgb.plot_importance(bst)
#plt.show()

#*******************************Save AUC/Error/Logloss***************************************#
def save_figure(eta, evals_result):

    results = evals_result
    epochs = len(evals_result['val']['error'])
    x_axis = range(0, epochs)

    plt.figure()
    plt.plot(x_axis, results['train']['logloss'], label='Train')
    plt.plot(x_axis, results['val']['logloss'], label='Validation')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.savefig(score_path + "LogLoss_{}.png".format(eta))
    #plt.show()

    plt.figure()
    plt.plot(x_axis, results['train']['error'], label='Train')
    plt.plot(x_axis, results['val']['error'], label='Validation')
    plt.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.savefig(score_path + "Error{}.png".format(eta))
    #plt.show()

    plt.figure()
    plt.plot(x_axis, results['train']['auc'], label='Train')
    plt.plot(x_axis, results['val']['auc'], label='Validation')
    plt.legend()
    plt.ylabel('AUC')
    plt.title('XGBoost AUC')
    plt.savefig(score_path + "AUC{}.png".format(eta))

#*******************************Save sore*************************************************#
def save_score(as_path, score_path, preds, eta):
    answer_sheet = pd.read_csv(as_path)
    #Dataframe data
    answer_sheet = pd.DataFrame(answer_sheet)
    #Feed result in score column
    answer = answer_sheet.assign(score = preds)
    #Save to .csv
    answer.to_csv(score_path + "score_xgb_{}.csv".format(eta), index = None, float_format = "%.9f")

    return print("Score saved to %s" % (score_path+ "score_xgb_{}.csv".format(eta)))


def main():

    save_hParams(param, score_path, eta)
    train, test, label, validation, validation_label = load_data(train_path, test_path, offset)
    bst, dtest, evals_result = create_DMatrix(param, train, test, label, validation, validation_label)
    preds = predict_and_save_model(bst, dtest, model_path, fmap, eta)
    save_figure(eta, evals_result)
    save_score(as_path, score_path, preds, eta)

if __name__ == '__main__':
    main()
