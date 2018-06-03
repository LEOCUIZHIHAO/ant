from __future__ import print_function

from ant_train_model import *
import numpy as np
import os

max_depth = 4 #4
num_trees = 480 #450
etc = 0.07
subsample = 0.8
colsample_bytree = 0.9
min_child_weight = 1
gamma = 0.1

lr = 1e-4
num_classes = 2
epochs = 2
batch_size = 20000 #how many samples per batch  #20000
validation = int(batch_size*0.1)
num_steps = 1000 #1000
display_step = 10 #100
num_threads = 2 #depends on the computer
min_after_dequeue =  40000 #how many left afer one batch #40000
capacity = min_after_dequeue + 3 * batch_size

path1 = os.path.abspath(".")
## TODO: make dir
nn_model_path = os.path.abspath("save_restore/nn_model") + "/"
xgb_model_path = os.path.abspath("save_restore/xgb_model") + "/"
model_name = "ant_model"


data_path = os.path.abspath("data") + "/"
data_name = "xgb_feats"

train_batch_path = data_path + "train_batch/"
test_batch_path = data_path + "test_batch/"

score_path = "/home/leo/ant_project/score/"
as_path = "/home/leo/ant_project/score/answer_sheet.csv"

# TODO:  package
data = np.load(data_path + "train.npy")
train_data = data[:,1:]
label = data[:,0]


# TODO:  msg write in a functions

def main():

    print("\n"+"*" *20 + "If you changed any params please make sure the old model already been saved !!!" + "*" *20 )

    #Xgboost
    ant_xgb = Xgb_Model(
                    max_depth, num_trees,etc,
                    subsample, colsample_bytree,
                    min_child_weight,
                    gamma,
                    xgb_model_path, model_name,
                    data_path, data_name
                    )


    ant_xgb.set_xgb_mode("train")

    train = ant_xgb.file_check()

    if train:

        ant_xgb.train_xgb(train_data, label)

        ant_xgb.batch_reader(train_batch_path, ant_xgb.load_mode)

    #Neural_Network
    ant_nn = Neural_Network(
                            lr,num_classes, epochs,validation,num_steps,
                            display_step, batch_size,
                            num_threads, min_after_dequeue,capacity,
                            nn_model_path, model_name,
                            data_path, data_name
                            )

    """

    1. If xgboost_feats.tfrecords exist, then feed xgboost_feats.tfrecords NN

    """
    if train:

        ant_nn.generate_tfrecords()

    nn_input = ant_nn.get_nn_input(data_path)

    ant_nn.run_model(nn_input, shuffle_batch = True)

    """
    test model

    """
    """
    ant_test = Model_Evl(
                        num_classes,
                        nn_input, ## TODO: change in the feature
                        nn_model_path,
                        data_path, data_name,
                        score_path, as_path
                        )
    """
    #change mode to test
    ant_xgb.set_xgb_mode("test")

    if train:

        ant_xgb.batch_reader(test_batch_path, ant_xgb.load_mode)

    #preds = ant_nn.model_online()
    preds = ant_nn.evl_mode(nn_input)
    #save score
    ant_nn.save_score(score_path, as_path, preds)


if __name__ == '__main__':
    main()
