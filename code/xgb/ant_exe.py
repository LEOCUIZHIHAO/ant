from __future__ import print_function

from ant_train_model import *
import numpy as np
import os

"""
When use this code pleas make sure the unlabel data, train data, test data and answer_sheet are in the project files.

This code including 3 modes: 1.xgb_nn; 2.xgb_only; 3. mlp_only

When using the XGB - Change the main() in xgb_only mode. Same as others.

The XGB mode will save the <model_name>.model and afer PU model which with the suffix "_PU".

Two scores will save as will named with <"init_" + scorce_name> and <"PU_" + score_name>.

"""


objective = "binary:logistic"
max_depth = 3 #3
num_trees = 480 #450
etc = 0.07
subsample = 0.8
colsample_bytree = 0.9
min_child_weight = 1
gamma = 0.1


lr = 1e-4
num_classes = 2
epochs = 15
early_stop = 45
batch_size = 20000 #how many samples per batch  #20000
validation = int(batch_size*0.1)
num_steps = 1000 #1000
display_step = 20 #100
num_threads = 2 #depends on the computer
min_after_dequeue =  40000 #how many left afer one batch #40000
capacity = min_after_dequeue + 3 * batch_size

#***********************threshold************************#
threshold = 0.8 #For the PU learning > 0.8 == 1
#***********************threshold************************#

path1 = os.path.abspath(".")
## TODO: make dir
nn_model_path = os.path.abspath("save_restore/nn_model") + "/"
xgb_model_path = os.path.abspath("save_restore/xgb_model") + "/"
model_name = "ant_model"#ant_model

#data path for NN-XGB
data_path = os.path.abspath("data") + "/"
data_name = "xgb_feats" #xgb_feats

#data path for unlabeled data
unlabel_path = data_path + "unlabel/"
unlabel_name = "unlabel"

#batch path
train_batch_path = data_path + "train_batch/"
test_batch_path = data_path + "test_batch/"

train_path = data_path + "train/"
train_1 = "train" # after adding PU data will generate with name PU_trian
train_2 = "PU_train"

test_path = data_path + "test/"
test_file = "test_a.npy"

#mlp data path is for csv only without header and label
mlp_train_data_name = "train_mlp"

mlp_test_data_path = test_path + "test_a_mlp.csv"

#Score path
score_name = "score.csv"
nn_score_path = nn_model_path  + model_name + "_score.csv"

"""
data = np.load(train_path + "train.npy")
train_data = data[:,1:]
label = data[:,0]
"""

def xgb(mode):

    print("\n"+"*" *20 + "If you changed any params please make sure the old model already been saved !!!" + "*" *20 )

    if mode == "xgb_nn":
        #Xgboost
        ant_xgb = Xgb_Model(
                        objective, max_depth, num_trees,etc,
                        subsample, colsample_bytree,
                        min_child_weight,
                        gamma,
                        xgb_model_path, model_name,
                        data_path, data_name, unlabel_path, unlabel_name,
                        train_path, train_1
                        )

        ant_xgb.set_xgb_mode("xgb_nn_train")

        train = ant_xgb.file_check(ant_xgb.model_path, ant_xgb.model_name + ".model")

        if train:
            #Initial training
            data = np.load(ant_xgb.train_path + train_1 + ".npy")
            train_data = data[:,1:]
            label = data[:,0]

            ant_xgb.train_xgb(train_data, label)

            #prepare pu label
            ant_xgb.process_unlabel(threshold)
            #Training with PU label

            data = np.load(ant_xgb.train_path + train_2 + ".npy")
            train_data = data[:,1:]
            label = data[:,0]

            ant_xgb.train_xgb(train_data, label, PU_mode = True)

            ant_xgb.batch_reader(train_batch_path, ant_xgb.load_mode)

        return train

    elif mode == "xgb_only":

        ant_xgb_only = Xgb_Model(
                        objective, max_depth, num_trees,etc,
                        subsample, colsample_bytree,
                        min_child_weight,
                        gamma,
                        xgb_model_path, model_name,
                        data_path, data_name, unlabel_path, unlabel_name,
                        train_path, train_1
                        )

        ant_xgb_only.set_xgb_mode("xgb_only")

        train = ant_xgb_only.file_check(ant_xgb_only.model_path, ant_xgb_only.model_name + ".model")

        if train:
            #Initial training
            data = np.load(ant_xgb_only.train_path + train_1 + ".npy")
            train_data = data[:,1:]
            label = data[:,0]
            ant_xgb_only.train_xgb(train_data, label)

            #save prediction without PU
            preds = ant_xgb_only.load_mode(np.load(test_path + test_file), init_xgb = True)
            ant_xgb_only.save_score(xgb_model_path + "init_xgb_" +score_name, preds)

            #prepare PU label feed the unlabel into the model
            ant_xgb_only.process_unlabel(threshold)

            #Training with PU label
            data = np.load(ant_xgb_only.train_path + train_2 + ".npy")
            train_data = data[:,1:]
            label = data[:,0]
            ant_xgb_only.train_xgb(train_data, label, PU_mode = True)

        #To get the result from the xgboost
        #ant_xgb_only.set_xgb_mode("xgb_only")
        #file check
        rewrite = ant_xgb_only.file_check(ant_xgb_only.model_path, score_name)

        if rewrite:
            #save prediction with PU
            preds = ant_xgb_only.load_mode(np.load(test_path + test_file))
            ant_xgb_only.save_score(xgb_model_path + "PU_" + score_name, preds)

def mlp(train, mode):

    if mode == "xgb_nn":

        ant_nn = Neural_Network(
                                lr,num_classes, epochs,validation,num_steps,
                                early_stop, display_step, batch_size,
                                num_threads, min_after_dequeue,capacity,
                                nn_model_path, model_name,
                                data_path, data_name
                                )

        if train:

            ant_nn.generate_tfrecords()

        nn_input = ant_nn.get_nn_input(data_path)

        ant_nn.run_model(nn_input, shuffle_batch = True)

        #test model
        #change mode to test
        ant_xgb.set_xgb_mode("xgb_nn_test")

        if train:

            ant_xgb.batch_reader(test_batch_path, ant_xgb.load_mode)

        #preds = ant_nn.model_online()
        preds = ant_nn.evl_mode(nn_input)
        #save score
        ant_nn.save_score(nn_score_path, preds)

    if mode == "mlp_only":
        #train data 2 tfrecords
        ant_nn_only = Neural_Network(
                                lr,num_classes, epochs,validation,num_steps,
                                early_stop, display_step, batch_size,
                                num_threads, min_after_dequeue,capacity,
                                nn_model_path, model_name + "_mlp",
                                train_path, mlp_data_name
                                )
        # TODO:  read input
        nn_input = 297

        ant_nn_only.generate_tfrecords()

        ant_nn_only.run_model(nn_input, shuffle_batch = True)

        preds = ant_nn_only.evl_mode(nn_input)
        #save score
        ant_nn_only.save_score(nn_score_path, preds)

def main():

    train = xgb(mode = "xgb_only")

    #mlp(train, mode = "mlp_only")

if __name__ == '__main__':
    main()
