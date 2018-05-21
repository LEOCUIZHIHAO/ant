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

param = {
        
        "objective" : "binary:logistic",
        "max_depth" : 10,
        "subsample" : 0.8,
        "colsample_bytree" : 0.9,
        "min_child_weight" : 1,
        "gamma" : 0,
        "eta" : 0.1, #learning_rate
        "eval_metric" : ['auc', 'logloss', 'error'], #early stop only effects on error
        "silent" : 0
        #'eval_metric':'error'
        #'eval_metric': 'auc'
        #'eval_metric': 'logloss'

        }

num_round = 5
early_stopping_rounds = 20
offset = 150000 #validation
suffix = "21_5_2018" #signle training save file name's suffix

#*******************************************************************************#
#********************************Loop param*************************************#
#**********************if dont use loop set to False****************************#
#*******************************************************************************#


loop_function = True #if False shut down loop_function
loop_param = "max_depth" #change the loop parameter here
loop_start = 3 #start loop digit
loop_end = 10  #end loop digit
loop_step = 1  #loop stop


#***********************data_path***********************************************#
#train_data path
train_path = "/home/leo/ant_leo/data/train_xgb_1.npy"
#test_data path
test_path = "/home/leo/ant_leo/data/test_a_xgb_1.npy"


#***********************model & score path ***********************************************#
#model save path#
model_path = "/home/leo/ant_leo/model/save_restore/"
#where to save the figure & answer & hParams
score_path = "/home/leo/ant_leo/score/"

#***********************Tool-box***********************************************#
#pandas read answer_sheet
#the path of answer_sheet.csv
as_path = "/home/leo/ant_leo/tool/answer_sheet.csv"
#xgb.fmap path
fmap = "/home/leo/ant_leo/tool/xgb.fmap"


def suffix():

    dict= {"min_child_weight" : "mcw",  "subsample" : "sb", "colsample_bytree" : "cs", "max_depth" : "md", "eta" : "lr", "gamma" : "gamma"}
    save_file_prefix = dict[loop_param]  # md = max_depth; sb = subsample; cs = colsample_bytree; mcw = min_child_weight

    return save_file_prefix

def parameters(loop_param_value):

    param.update({loop_param : loop_param_value})

    return param


def save_name_and_loop_param(loop_param_value):

    _save_file_prefix = suffix()
    _param_name = str(_save_file_prefix) + "_" + str(loop_param_value)
    _param = parameters(loop_param_value)

    return _param, _param_name

def save_hParams (param, param_name):

    f = csv.writer(open(score_path + "hParams_{}.csv".format(param_name), "w"))
    for key, val in param.items():
        f.writerow([key, val])

    return print("hParams saved in : %s \n"  %(score_path))


def load_data():
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

    print("start training...\n")

    bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result, early_stopping_rounds =  early_stopping_rounds)

    return bst, dtest, evals_result
#computitional cost too much
#eval = xgb.cv(param, dtrain, num_round, nfold=6, metrics={'error'}, seed=0,  callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])

def predict_and_save_model(bst, dtest, param_name):
    #save model
    bst.save_model(model_path + "xbg_{}.model".format(param_name))
    #save xgboost structure
    #bst.dump_model(model_path + "xbg_{}.txt".format(param_name))

    preds = bst.predict(dtest, ntree_limit= bst.best_iteration)
    print("\nthe minimal loss found in : %i booster \n" %(bst.best_iteration))

    features = pd.Series(bst.get_fscore(fmap = fmap)).sort_values(ascending=False)
    #save features and prepare feed into NN
    features.to_csv(score_path + "xgb_features_{}.csv".format(param_name))

    print("saving model & features ......\n")

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
def save_figure(param_name, evals_result):

    results = evals_result
    epochs = len(evals_result['val']['error'])
    x_axis = range(0, epochs)

    plt.figure()
    plt.plot(x_axis, results['train']['logloss'], label='Train')
    plt.plot(x_axis, results['val']['logloss'], label='Validation')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.savefig(score_path + "LogLoss_{}.png".format(param_name))
    #plt.show()

    plt.figure()
    plt.plot(x_axis, results['train']['error'], label='Train')
    plt.plot(x_axis, results['val']['error'], label='Validation')
    plt.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.savefig(score_path + "Error_{}.png".format(param_name))
    #plt.show()

    plt.figure()
    plt.plot(x_axis, results['train']['auc'], label='Train')
    plt.plot(x_axis, results['val']['auc'], label='Validation')
    plt.legend()
    plt.ylabel('AUC')
    plt.title('XGBoost AUC')
    plt.savefig(score_path + "AUC_{}.png".format(param_name))

#*******************************Save sore*************************************************#
def save_score(preds, param_name):
    answer_sheet = pd.read_csv(as_path)
    #Dataframe data
    answer_sheet = pd.DataFrame(answer_sheet)
    #Feed result in score column
    answer = answer_sheet.assign(score = preds)
    #Save to .csv
    answer.to_csv(score_path + "score_xgb_{}.csv".format(param_name), index = None, float_format = "%.9f")

    return print("Score saved to %s \n" % (score_path+ "score_xgb_{}.csv".format(param_name)))


def loop_function_run():

    range = np.arange(loop_start, loop_end, loop_step)

    for loop_param_value in range:

        loop_param_value = round(loop_param_value, 2)
    #for subsample in range (5, 10, 1): #when using subsample plz * 0.1. etc: 0.1*subsample = (0.5 - 1)
        print("XGBoost starting with loop_function, loop param is : %s , staring from : %s , end with : %s, step is : %s , run : %s\n"
               %(loop_param, loop_start, loop_end, loop_step, loop_param_value))

        _param, _param_name = save_name_and_loop_param(loop_param_value)
        save_hParams(_param, _param_name)
        train, test, label, validation, validation_label = load_data()
        bst, dtest, evals_result = create_DMatrix(_param, train, test, label, validation, validation_label)
        preds = predict_and_save_model(bst, dtest, _param_name)
        save_figure(_param_name, evals_result)
        save_score(preds, _param_name)
        print("*******************************************Done***************************************************\n")

    return


def main():

    if loop_function == True:

        loop_function_run()

    else:

        print("XGBoost starting with lr : %s" %(param["eta"]))
        save_hParams(param, suffix)
        train, test, label, validation, validation_label = load_data()
        bst, dtest, evals_result = create_DMatrix(param, train, test, label, validation, validation_label)
        preds = predict_and_save_model(bst, dtest, suffix)
        save_figure(suffix, evals_result)
        save_score(preds, suffix)


if __name__ == '__main__':
    main()
