"""
@Authors Leo.cui
7/5/2018
RF-NN Model

"""
from __future__ import print_function

import tensorflow as tf
#from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python.ops import data_ops
from tensor_forest import*
import pandas as pd
import numpy as np
import time

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

start_time = time.time()

all_data = pd.read_csv("/home/leo/ant/data/train_non_test.csv")
all_data = all_data[(all_data.label==0)|(all_data.label==1)] #get rid off -1 label
all_data = all_data.fillna(0)

train_dataset = all_data

#**********************train*****************************#
#training
train_data = train_dataset.drop(['date', 'id','label'], axis=1).astype(np.float32).values
#training label
train_label = train_dataset['label'].astype(np.int32).values

train = []

for i in range(0,len(train_dataset)):
    train.append([train_data[i],train_label[i]])

train_x = [i[0] for i in train]
train_y = [i[1] for i in train]

#train_x = np.array(train_x)
#train_y = np.array(train_y)
#***************************************parameters**********************************************#

num_classes = 2 # The 10 digits
num_features = 297 # Each image is 28x28 pixels
num_trees = 400
max_nodes = 1000
bagging_fraction= 0.9 # (<1) how many sample use to bagging
base_random_seed = 0 # bagged choose row!!! if = 0 then no bagged
batch_size = 10



# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes,
                                      bagging_fraction = bagging_fraction,
                                      base_random_seed = base_random_seed
                                      ).fill()
# Build the Random Forest
forest_graph = RandomForestGraphs(hparams)

constant = int(len(train_x)/batch_size)

for iter_batch in range(0, batch_size):

    batch_x = train_x[iter_batch*constant : (iter_batch+1)*constant]

    batch_y = train_y[iter_batch*constant : (iter_batch+1)*constant]

    batch_x = np.array(batch_x)

    batch_y = np.array(batch_y)
    # Get RF every node feature and value
    tree_data, tree_labels, total_trees_data = forest_graph.training_graph(batch_x, batch_y)

    # Initialize the variables (i.e. assign their default value) and forest resources
    init_vars = tf.group(tf.global_variables_initializer(),
        resources.initialize_resources(resources.shared_resources()))

    with tf.Session() as sess:
        sess.run(init_vars)
        print("\nDate feeding into RF.......\n")
        print("Batch : %i processing .......\n" %(iter_batch+1))
        c = sess.run([tree_data, tree_labels, total_trees_data], feed_dict={X:batch_x , Y: batch_y})
        total_trees_data = np.array(c[2])
        tree_labels = np.array(c[1])
        total_trees_data = total_trees_data.reshape(total_trees_data.shape[1],-1) #total_trees_data.shape[1] = total rows number
        print("RF feature extraction done.......\n")

    white = [1,0]
    black = [0,1]
    label = []
    #one-hot label
    for i in tree_labels:
        if i == 0:
            label.append(white)
        elif i ==1:
            label.append(black)
        elif i ==-1:
          print("label -1")


    RF_NN_data = []

    for i in range(len(total_trees_data)) :
        RF_NN_data.append([total_trees_data[i],label[i]])

    samples = len(RF_NN_data)
    features = len(RF_NN_data[0][0])

    print("%i samples with %i features waiting feed into NN"  % (samples,features))

    npy_path = "/home/leo/ant/super_code/save_restore/model/"

    np.save(npy_path + "RF_NN_data_{}.npy".format(iter_batch+1), RF_NN_data)

    print("saved to :%s" % (npy_path + "RF_NN_data_{}.npy".format(iter_batch+1)))

end_time = time.time()

print("process %f percent data using %f time: " % (bagging_fraction,(end_time - start_time)))

