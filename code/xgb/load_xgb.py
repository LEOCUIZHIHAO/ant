"""
@Authors Leo.cui
23/5/2018
Xgboost

"""

import xgboost as xgb
import numpy as np
from lib import divide_2_batch as d2b
import numpy as np
from lib.one_hot_encoder import *
from leaf_node_index import *
import os

param = {

        "objective" : "binary:logistic",
        "max_depth" : 3,
        "subsample" : 1,
        "colsample_bytree" : 1,
        "min_child_weight" : 1,
        "gamma" : 0,
        "eta" : 0.1, #learning_rate
        "silent" : 0,
        #'eval_metric':'error'
        #'eval_metric': 'auc'
        #'eval_metric': 'logloss'
        #"eval_metric" : ['auc', 'logloss', 'error'], #early stop only effects on error
        }

offset = 15000
train_path = "/home/leo/ant_leo/data/train_batch/"
save_path = "/home/leo/ant_leo/data/"
model_path = "/home/leo/ant_leo/model/save_restore/xbg_md_550.model"
score_path = "/home/leo/ant_leo/score/"


def main():

    obj_dir = os.listdir(train_path)

    for file in obj_dir:

        if file.endswith(".npy") :

            print("loading file : %s" %(file))
            train_data = np.load(train_path+file)
            train = train_data[:-offset,1:]
            label = train_data[:-offset,0]
            dtrain = xgb.DMatrix(train, missing = -999.0)

            #one hot label
            label_oh = one_hot_encode(label)
            #load xgb model
            bst = xgb.Booster(param, model_file = model_path)
            print("loaded xgb model")
            #get leaf index
            _total_leaf_index, leaf = get_leaf_node_index(bst, dtrain)
            feed_nn_data = xgb_2_nn_data(_total_leaf_index, leaf, label_oh)
            np.save(score_path + "xgb_nn_{}.npy".format(file[-5]), feed_nn_data)
            print("file saved in %s" %(score_path + file[-5] + ".npy"))


if __name__ == '__main__':
    main()
